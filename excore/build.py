"""Deterministic checkpoint compiler.

``build`` turns (manifest, source GGUF) into a candidate GGUF:

* the metadata block is copied verbatim from the source,
* every tensor not owned by a searchable unit (embeddings, norms, state parameters, ...)
  is copied byte for byte,
* every unit tensor is encoded by the quantizer the manifest names, into the format
  the manifest names,
* tensors keep the source's order and alignment.

The same manifest and source therefore always yield the same bytes.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from .config import load_config
from .gguf import FLOAT_TYPES, GGMLType, GGUFFile, TensorInfo, WriteEntry, tensor_nbytes, write_gguf
from .manifest import Manifest, load_manifest
from .model.qwen38_cpu import ModelSpec, build_inventory
from .quantizers import BuildContext, TensorJob, ensure_loaded, get_quantizer

SIDECAR_SCHEMA = "excore/build@1"


class BuildError(ValueError):
    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


@dataclass(frozen=True)
class PlanItem:
    name: str
    dims: tuple[int, ...]
    src: TensorInfo
    dst_type: GGMLType
    nbytes: int
    unit: str | None = None            # None => frozen tensor
    quantizer: str | None = None
    job: TensorJob | None = None

    @property
    def frozen(self) -> bool:
        return self.unit is None


@dataclass(frozen=True)
class BuildPlan:
    candidate_id: str
    items: tuple[PlanItem, ...]
    assignments: dict

    @property
    def total_bytes(self) -> int:
        return sum(i.nbytes for i in self.items)


@dataclass(frozen=True)
class BuildResult:
    output: Path
    sidecar: Path
    candidate_id: str
    sha256: str
    size_bytes: int
    tensors: int


def manifest_quantizer_names(manifest: Manifest) -> set[str]:
    return {manifest.default_quantizer} | {r.quantizer for r in manifest.rules}


def plan_build(manifest: Manifest, spec: ModelSpec, source: GGUFFile) -> BuildPlan:
    """Resolve the manifest against the source file. Raises ManifestError or BuildError."""
    ensure_loaded(manifest_quantizer_names(manifest))
    assignments = manifest.expand(spec)
    units = build_inventory(spec)
    owner = {t.gguf_name: (u, t) for u in units for t in u.tensors}

    problems: list[str] = []
    for name, (u, t) in owner.items():
        s = source.by_name.get(name)
        if s is None:
            problems.append(f"source is missing tensor {name} (unit {u.name})")
            continue
        if s.dims != (t.in_features, t.out_features):
            problems.append(
                f"{name}: source shape {s.dims} does not match the model config "
                f"{(t.in_features, t.out_features)}"
            )
        if s.ggml_type not in FLOAT_TYPES:
            problems.append(f"{name}: source tensor is already quantized ({s.ggml_type.name})")
    if problems:
        raise BuildError(problems)

    items: list[PlanItem] = []
    for s in source.tensors:
        if s.name in owner:
            u, _ = owner[s.name]
            a = assignments[u.name]
            gt = GGMLType[a.format]
            job = TensorJob(s.name, u.name, _fmt(a.format), s)
            items.append(
                PlanItem(s.name, s.dims, s, gt, tensor_nbytes(s.dims, gt), u.name, a.quantizer, job)
            )
        else:
            items.append(PlanItem(s.name, s.dims, s, s.ggml_type, s.nbytes))
    return BuildPlan(manifest.candidate_id(spec), tuple(items), assignments)


def _fmt(name: str):
    from .precision import resolve_format

    return resolve_format(name)


def build(
    manifest: Manifest | str | Path,
    source: str | Path,
    out: str | Path,
    *,
    spec: ModelSpec | None = None,
    cfg: dict | None = None,
    llama_quantize: str | None = None,
    workdir: str | Path | None = None,
) -> BuildResult:
    cfg = cfg or load_config()
    spec = spec or ModelSpec.from_mapping(cfg["model"])
    if not isinstance(manifest, Manifest):
        manifest = load_manifest(manifest, spec, expected_track=cfg["track"])
    limits = cfg["limits"]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    own_workdir = workdir is None
    work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="excore-build-", dir=out.parent))

    with GGUFFile(source) as src:
        plan = plan_build(manifest, spec, src)
        ctx = BuildContext(
            workdir=work,
            threads=int(limits["threads"]),
            timeout=float(limits["build_timeout_s"]),
            llama_quantize=llama_quantize,
        )
        try:
            by_quantizer: dict[str, list[TensorJob]] = {}
            for item in plan.items:
                if item.job is not None:
                    by_quantizer.setdefault(item.quantizer, []).append(item.job)
            for qname in sorted(by_quantizer):
                get_quantizer(qname).prepare(src, by_quantizer[qname], ctx)

            entries = [
                WriteEntry(it.name, it.dims, it.dst_type, partial(_produce, it, src, ctx))
                for it in plan.items
            ]
            sha = write_gguf(
                tmp,
                metadata_raw=src.metadata_raw,
                kv_count=src.kv_count,
                alignment=src.alignment,
                entries=entries,
            )
            os.replace(tmp, out)
        finally:
            ctx.cleanup()
            if own_workdir:
                shutil.rmtree(work, ignore_errors=True)
            if tmp.exists():
                tmp.unlink()

        sidecar = write_sidecar(out, manifest, plan, sha, src)
    return BuildResult(out, sidecar, plan.candidate_id, sha, out.stat().st_size, len(plan.items))


def _produce(item: PlanItem, src: GGUFFile, ctx: BuildContext):
    if item.frozen:
        return src.iter_tensor_bytes(item.src)
    return get_quantizer(item.quantizer).encode(item.job, src, ctx)


def write_sidecar(out: Path, manifest: Manifest, plan: BuildPlan, sha: str, src: GGUFFile) -> Path:
    quantizers = {}
    for item in plan.items:
        if item.quantizer and item.quantizer not in quantizers:
            quantizers[item.quantizer] = get_quantizer(item.quantizer).fingerprint
    doc = {
        "schema": SIDECAR_SCHEMA,
        "track": manifest.track,
        "manifest": manifest.name,
        "candidate_id": plan.candidate_id,
        "output_sha256": sha,
        "source": {"name": src.path.name, "size": src.size},
        "quantizers": quantizers,
        "units": {n: f"{a.format}@{a.quantizer}" for n, a in sorted(plan.assignments.items())},
        "tensors": len(plan.items),
        "bytes": plan.total_bytes,
    }
    path = out.with_name(out.name + ".build.json")
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
