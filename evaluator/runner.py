"""Evaluation runner: from a validated manifest to measured results.

``LocalRunner`` builds the candidate and audits it in the sandbox, then profiles speed and
RAM, scores accuracy against the reference, and (only if the cheap gates pass) runs the
long-context needle guards. Baseline numbers (the stock recipe, V0) are measured once per
host/config and cached on disk, because every relative number depends on them.

Guard reference: the needle guards compare against the *baseline recipe*, not the BF16
original, because retrieval at 16K on a 35B BF16 model on CPU would take hours per run (and the BF16 weights do not fit in the validator's RAM).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Protocol

import yaml

from excore.eval.accuracy import AccuracyResult, Reference, score
from excore.eval.holdout import HoldoutVerdict, holdout_verdict, load_holdout
from excore.eval.performance import PerfResult, host_fingerprint, measure
from excore.frontier.gates import GuardResult, NEEDLE_RESERVE, evaluate_gates, run_needle_guard
from excore.manifest import Manifest

from .sandbox import Sandbox, SandboxPolicy

FILLER_TOKENS = 6000


class EvalFailure(RuntimeError):
    """The candidate could not be evaluated (build/audit/measurement failed)."""

    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(f"{stage}: {message}")


@dataclass(frozen=True)
class BaselinePerf:
    gen_tps: float
    prefill_tps: float
    peak_rss_gib: float


@dataclass
class EvalBundle:
    accuracy: AccuracyResult
    perf: PerfResult
    baseline_perf: BaselinePerf
    guards: list[GuardResult]
    host: dict
    baseline_public_kl: float
    candidate_path: Path | None = None
    guards_skipped: bool = False
    notes: list[str] = field(default_factory=list)


class Runner(Protocol):
    def evaluate(self, manifest_path: Path, manifest: Manifest, candidate_id: str) -> EvalBundle: ...

    def holdout(self, bundle: EvalBundle) -> HoldoutVerdict: ...

    def cleanup(self, bundle: EvalBundle) -> None: ...


class LocalRunner:
    def __init__(
        self, cfg: Mapping, *, repo_root: Path, source: Path, reference_path: Path, workdir: Path,
        config_path: Path, baseline_manifest: Path | None = None, sandbox: Sandbox | None = None,
        perf_fn: Callable[[Path], PerfResult] | None = None,
        backend_factory: Callable[[Path], object] | None = None,
        generative_factory: Callable[[Path, int], object] | None = None,
        llama_quantize: str | None = None, bench_binary: str | None = None,
        holdout_root: str | Path | None = None, needle_seed: int | None = None,
    ):
        self.cfg, self.repo_root, self.source = cfg, Path(repo_root), Path(source)
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.config_path = Path(config_path)
        self.baseline_manifest = Path(baseline_manifest or self.repo_root / "experiments/base_variants/V0_baseline.yaml")
        lim = cfg["limits"]
        self.sandbox = sandbox or Sandbox(SandboxPolicy(timeout_s=float(lim["build_timeout_s"]) * 2))
        self.reference = Reference.load(reference_path)
        self.llama_quantize, self.holdout_root = llama_quantize, holdout_root
        self.perf_fn = perf_fn or (lambda p: measure(p, cfg, bench_binary=bench_binary))
        self.backend_factory = backend_factory or self._llama_backend
        self.generative_factory = generative_factory or self._llama_generative
        self.needle_seed = needle_seed
        self._baseline_cache: dict | None = None

    # -- real llama.cpp backends (imported lazily so tests need no bindings) ----------------

    def _llama_backend(self, path: Path):
        from excore.eval.backend import LlamaCppBackend

        lim = self.cfg["limits"]
        return LlamaCppBackend(str(path), n_ctx=int(self.cfg["accuracy"]["ctx"]),
                               n_threads=int(lim["threads"]), n_batch=int(lim["batch_size"]))

    def _llama_generative(self, path: Path, n_ctx: int):
        from excore.eval.backend import LlamaCppBackend

        lim = self.cfg["limits"]
        return LlamaCppBackend(str(path), n_ctx=n_ctx, n_threads=int(lim["threads"]),
                               n_batch=int(lim["batch_size"]), logits_all=False)

    # -- sandboxed build and audit ------------------------------------------------------------

    def _excore(self, *args: str) -> list[str]:
        return [sys.executable, "-I", "-m", "excore", "--config", str(self.config_path), *args]

    def _build_and_audit(self, manifest_path: Path, out: Path) -> None:
        env = {}
        if self.llama_quantize:
            env["EXCORE_LLAMA_QUANTIZE"] = self.llama_quantize
        sbx = self.workdir / "sandbox"
        rw, ro = [self.workdir], [self.repo_root, self.source.parent, manifest_path.parent]
        res = self.sandbox.run(self._excore("build", str(manifest_path), "--source", str(self.source), "--out", str(out)),
                               cwd=sbx, env_extra=env, rw_paths=rw, ro_paths=ro)
        if not res.ok:
            raise EvalFailure("build", "timed out" if res.timed_out else res.tail())
        res = self.sandbox.run(self._excore("audit", str(manifest_path), "--candidate", str(out), "--source",
                                            str(self.source), "--replay", "sample", "--json"),
                               cwd=sbx, env_extra=env, rw_paths=rw, ro_paths=ro)
        try:
            report = json.loads(res.stdout.strip().splitlines()[-1]) if res.stdout.strip() else None
        except json.JSONDecodeError:
            report = None
        if report is None:
            raise EvalFailure("audit", "timed out" if res.timed_out else res.tail())
        if not report["ok"]:
            errors = [f"{f['code']} {f.get('tensor') or ''}: {f['message']}" for f in report["findings"]
                      if f["severity"] == "error"]
            raise EvalFailure("audit", "; ".join(errors[:5]))

    # -- baseline (the stock recipe), cached --------------------------------------------------

    def _cache_key(self) -> str:
        blob = json.dumps({
            "cfg": yaml.safe_dump(dict(self.cfg), sort_keys=True), "reference": self.reference.digest(),
            "host": host_fingerprint(), "source_size": self.source.stat().st_size,
            "baseline": self.baseline_manifest.read_text(encoding="utf-8"),
        }, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    @property
    def baseline_path(self) -> Path:
        return self.workdir / "baseline.gguf"

    def _baseline(self) -> dict:
        if self._baseline_cache is not None:
            return self._baseline_cache
        cache_file = self.workdir / "baseline-cache.json"
        key = self._cache_key()
        if cache_file.exists() and self.baseline_path.exists():
            try:
                cached = json.loads(cache_file.read_text())
                if cached.get("key") == key:
                    self._baseline_cache = cached
                    return cached
            except json.JSONDecodeError:
                pass
        self._build_and_audit(self.baseline_manifest, self.baseline_path)
        perf = self.perf_fn(self.baseline_path)
        if not perf.stable:
            raise EvalFailure("baseline", "baseline benchmark is unstable; the host is too noisy to evaluate on")
        backend = self.backend_factory(self.baseline_path)
        try:
            acc = score(backend, self.reference)
        finally:
            _close(backend)
        self._baseline_cache = {
            "key": key,
            "perf": {"gen_tps": perf.gen_tps, "prefill_tps": perf.prefill_tps, "peak_rss_gib": perf.peak_rss_gib},
            "public_kl": acc.rp_kl, "needle": {}, "holdout": {},
        }
        self._save_baseline()
        return self._baseline_cache

    def _save_baseline(self) -> None:
        tmp = self.workdir / "baseline-cache.json.tmp"
        tmp.write_text(json.dumps(self._baseline_cache, indent=2))
        os.replace(tmp, self.workdir / "baseline-cache.json")

    # -- needle guards -------------------------------------------------------------------------

    def _seed(self) -> int:
        from excore.eval.holdout import epoch_now

        return self.needle_seed if self.needle_seed is not None else epoch_now()

    def _filler(self) -> list[int]:
        return [int(t) for t in self.reference.windows[:, 1:].ravel()[:FILLER_TOKENS]]

    def _needle_score(self, model: Path, ctx: int) -> float:
        g = self.cfg["gates"]
        backend = self.generative_factory(model, ctx + NEEDLE_RESERVE + 8)
        try:
            s, _ = run_needle_guard(backend, [ctx], [float(d) for d in g["needle_depths"]], self._filler(), self._seed())
        finally:
            _close(backend)
        return s

    def _baseline_needle(self, ctx: int) -> float:
        base = self._baseline()
        key = f"{self._seed()}:{ctx}"
        if key not in base["needle"]:
            base["needle"][key] = self._needle_score(self.baseline_path, ctx)
            self._save_baseline()
        return base["needle"][key]

    # -- the public interface ---------------------------------------------------------------------

    def evaluate(self, manifest_path: Path, manifest: Manifest, candidate_id: str) -> EvalBundle:
        base = self._baseline()
        cand = self.workdir / f"cand-{candidate_id}.gguf"
        try:
            self._build_and_audit(manifest_path, cand)
            perf = self.perf_fn(cand)
            backend = self.backend_factory(cand)
            try:
                acc = score(backend, self.reference)
            finally:
                _close(backend)
        except Exception:
            cand.unlink(missing_ok=True)
            raise
        bundle = EvalBundle(acc, perf, BaselinePerf(**base["perf"]), [], host_fingerprint(), base["public_kl"], cand)
        if not evaluate_gates(perf=perf, accuracy=acc, guards=[], cfg=self.cfg).passed:
            bundle.guards_skipped = True            # it fails anyway; do not spend hours on long-context runs
            return bundle
        g = self.cfg["gates"]
        for ctx in g["needle_contexts"]:
            bundle.guards.append(GuardResult(
                f"needle{int(ctx)}", self._baseline_needle(int(ctx)), self._needle_score(cand, int(ctx)),
                float(g["needle_max_regression_pct"])))
        return bundle

    def holdout(self, bundle: EvalBundle) -> HoldoutVerdict:
        if bundle.candidate_path is None:
            raise EvalFailure("holdout", "no candidate file to score")
        ref, shard = load_holdout(self.holdout_root)
        base = self._baseline()
        if shard not in base["holdout"]:
            backend = self.backend_factory(self.baseline_path)
            try:
                base["holdout"][shard] = score(backend, ref).rp_kl
            finally:
                _close(backend)
            self._save_baseline()
        backend = self.backend_factory(bundle.candidate_path)
        try:
            cand_kl = score(backend, ref).rp_kl
        finally:
            _close(backend)
        g = self.cfg["gates"]
        return holdout_verdict(bundle.baseline_public_kl, bundle.accuracy.rp_kl, base["holdout"][shard], cand_kl,
                               min_transfer=float(g["holdout_min_transfer"]), tolerance=float(g["holdout_tolerance"]),
                               shard=shard)

    def cleanup(self, bundle: EvalBundle) -> None:
        if bundle.candidate_path is not None:
            bundle.candidate_path.unlink(missing_ok=True)


def _close(backend) -> None:
    close = getattr(backend, "close", None)
    if close:
        close()
