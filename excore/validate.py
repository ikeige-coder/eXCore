"""Audit: proves a candidate GGUF is nothing but legal encodings of the source weights.

Run before anything touches the CPU benchmark. It never executes the candidate and
never trusts the build sidecar; everything is recomputed from the manifest and the
hash-locked source:

* structure: header, alignment, contiguous non-overlapping tensors, no trailing bytes,
* identity: metadata block byte-identical, same tensors in the same order and shapes,
* frozen tensors byte-identical to the source,
* execution map: each unit tensor really is in the format the manifest declared,
* integrity: no NaN/Inf block scales or float values,
* lineage replay: regenerable quantizers are re-run and compared byte for byte
  (all of them, or a random sample).
"""

from __future__ import annotations

import random
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .build import BuildError, plan_build
from .config import load_config
from .gguf import (
    CHUNK,
    GGUFError,
    GGUFFile,
    TensorInfo,
    TYPE_TRAITS,
    align_up,
    scales_finite,
)
from .manifest import Manifest, ManifestError
from .model.qwen38_cpu import ModelSpec
from .quantizers import BuildContext, QuantizerError, get_quantizer

ERROR = "error"
INFO = "info"


@dataclass(frozen=True)
class Finding:
    code: str
    message: str
    tensor: str | None = None
    severity: str = ERROR

    def __str__(self) -> str:
        where = f" [{self.tensor}]" if self.tensor else ""
        return f"{self.code}{where}: {self.message}"


@dataclass
class AuditReport:
    findings: list[Finding] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == ERROR]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, code: str, message: str, tensor: str | None = None, severity: str = ERROR) -> None:
        self.findings.append(Finding(code, message, tensor, severity))

    def codes(self) -> set[str]:
        return {f.code for f in self.errors}

    def summary(self) -> str:
        if self.ok:
            return "audit passed"
        return "audit FAILED: " + "; ".join(str(f) for f in self.errors[:10])


def _chunk_for(t: TensorInfo) -> int:
    _, bb = TYPE_TRAITS[t.ggml_type]
    return max(bb, CHUNK // bb * bb)


def _same_bytes(a: memoryview, b: memoryview) -> bool:
    if a.nbytes != b.nbytes:
        return False
    for i in range(0, a.nbytes, CHUNK):
        if a[i : i + CHUNK] != b[i : i + CHUNK]:
            return False
    return True


def _finite(view: memoryview, t: TensorInfo) -> bool:
    step = _chunk_for(t)
    for i in range(0, view.nbytes, step):
        if not scales_finite(view[i : i + step], t.ggml_type):
            return False
    return True


def _replay_matches(chunks, view: memoryview) -> bool:
    pos = 0
    for c in chunks:
        n = memoryview(c).nbytes
        if pos + n > view.nbytes or view[pos : pos + n] != memoryview(c).cast("B"):
            return False
        pos += n
    return pos == view.nbytes


def audit(
    candidate: str | Path,
    source: str | Path,
    manifest: Manifest,
    spec: ModelSpec | None = None,
    *,
    cfg: dict | None = None,
    replay: str = "regenerable",      # "regenerable" | "sample" | "none"
    sample: int = 8,
    seed: int | None = None,
    workdir: str | Path | None = None,
) -> AuditReport:
    cfg = cfg or load_config()
    spec = spec or ModelSpec.from_mapping(cfg["model"])
    if replay not in ("regenerable", "sample", "none"):
        raise ValueError("replay must be 'regenerable', 'sample' or 'none'")
    report = AuditReport()

    with GGUFFile(source) as src:
        try:
            cand = GGUFFile(candidate)
        except (GGUFError, OSError) as exc:
            report.add("E_PARSE", f"candidate is not a readable GGUF file: {exc}")
            return report
        with cand:
            try:
                plan = plan_build(manifest, spec, src)
            except (ManifestError, BuildError) as exc:
                report.add("E_MANIFEST", str(exc))
                return report
            _structure(report, cand, src)
            if report.errors:
                return report
            _tensors(report, cand, src, plan)
            if not report.errors and replay != "none":
                _replay(report, cand, src, plan, cfg, replay, sample, seed, workdir)

            by_format: dict[str, int] = {}
            for it in plan.items:
                if it.unit:
                    by_format[it.dst_type.name] = by_format.get(it.dst_type.name, 0) + it.nbytes
            report.stats.update(
                candidate_id=plan.candidate_id,
                file_bytes=cand.size,
                unit_bytes_by_format=by_format,
                execution_map={n: a.format for n, a in plan.assignments.items()},
            )
    return report


def _structure(report: AuditReport, cand: GGUFFile, src: GGUFFile) -> None:
    if cand.version != src.version:
        report.add("E_HEADER", f"GGUF version {cand.version}, source has {src.version}")
    if cand.alignment != src.alignment:
        report.add("E_ALIGN", f"alignment {cand.alignment}, source has {src.alignment}")
    if cand.metadata_raw != src.metadata_raw or cand.kv_count != src.kv_count:
        report.add("E_METADATA", "metadata block differs from the source")

    names, src_names = [t.name for t in cand.tensors], [t.name for t in src.tensors]
    if set(src_names) - set(names):
        report.add("E_TENSOR_SET", f"missing tensors: {sorted(set(src_names) - set(names))[:5]}")
    if set(names) - set(src_names):
        report.add("E_TENSOR_SET", f"unexpected tensors: {sorted(set(names) - set(src_names))[:5]}")
    if set(names) == set(src_names) and names != src_names:
        report.add("E_ORDER", "tensor order differs from the source")

    expected = 0
    for t in cand.tensors:
        if t.offset != expected:
            report.add("E_LAYOUT", f"offset {t.offset}, expected {expected}", t.name)
            break
        if not cand.in_bounds(t):
            report.add("E_LAYOUT", "tensor data lies outside the file", t.name)
            break
        expected += align_up(t.nbytes, cand.alignment)
    else:
        if cand.size != cand.data_start + expected:
            report.add("E_LAYOUT", f"file is {cand.size} bytes, expected {cand.data_start + expected}")


def _tensors(report: AuditReport, cand: GGUFFile, src: GGUFFile, plan) -> None:
    for it in plan.items:
        t = cand.by_name[it.name]
        if t.dims != it.dims:
            report.add("E_DIMS", f"shape {t.dims}, expected {it.dims}", it.name)
            continue
        if t.ggml_type != it.dst_type:
            code = "E_FROZEN" if it.frozen else "E_FORMAT"
            report.add(code, f"type {t.ggml_type.name}, expected {it.dst_type.name}", it.name)
            continue
        view = cand.tensor_view(t)
        if it.frozen:
            if not _same_bytes(view, src.tensor_view(it.src)):
                report.add("E_FROZEN", "frozen tensor bytes differ from the source", it.name)
        elif not _finite(view, t):
            report.add("E_SCALES", "NaN or Inf in block scales or values", it.name)


def _replay(report, cand, src, plan, cfg, mode, sample, seed, workdir) -> None:
    candidates = []
    skipped: dict[str, int] = {}
    for it in plan.items:
        if it.job is None:
            continue
        q = get_quantizer(it.quantizer)
        if q.regenerable:
            candidates.append(it)
        else:
            skipped[it.quantizer] = skipped.get(it.quantizer, 0) + 1
    if mode == "sample" and len(candidates) > sample:
        candidates = random.Random(seed).sample(candidates, sample)

    own = workdir is None
    work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="excore-audit-"))
    ctx = BuildContext(work, int(cfg["limits"]["threads"]), float(cfg["limits"]["build_timeout_s"]))
    try:
        for it in candidates:
            q = get_quantizer(it.quantizer)
            view = cand.tensor_view(cand.by_name[it.name])
            try:
                same = _replay_matches(q.encode(it.job, src, ctx), view)
            except (QuantizerError, GGUFError) as exc:
                report.add("E_REPLAY", f"re-encoding failed: {exc}", it.name)
                continue
            if not same:
                report.add("E_REPLAY", f"bytes differ from a fresh {q.fingerprint} encoding", it.name)
    finally:
        ctx.cleanup()
        if own:
            shutil.rmtree(work, ignore_errors=True)
    report.stats["replayed_tensors"] = len(candidates)
    for qname, n in sorted(skipped.items()):
        report.add("I_REPLAY_SKIPPED", f"{n} tensors from non-regenerable quantizer {qname!r}", severity=INFO)
