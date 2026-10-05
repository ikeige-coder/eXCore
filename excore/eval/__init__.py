"""Evaluation: accuracy drift on CPU (RP-KL) and CPU speed/memory profiling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .accuracy import AccuracyResult, LogitsBackend, Reference, score
from .performance import PerfResult, measure


@dataclass(frozen=True)
class EvalResult:
    accuracy: AccuracyResult
    perf: PerfResult

    def to_dict(self) -> dict:
        return {"accuracy": self.accuracy.to_dict(), "perf": self.perf.to_dict()}


def evaluate_model(
    model: str | Path,
    reference: Reference,
    backend_factory: Callable[[], LogitsBackend],
    cfg: dict,
    *,
    bench_binary: str | None = None,
) -> EvalResult:
    """Profile speed/RAM first (clean process, nothing else resident), then score accuracy."""
    perf = measure(model, cfg, bench_binary=bench_binary)
    backend = backend_factory()
    try:
        acc = score(backend, reference)
    finally:
        close = getattr(backend, "close", None)
        if close:
            close()
    return EvalResult(acc, perf)


__all__ = ["EvalResult", "evaluate_model", "AccuracyResult", "PerfResult", "Reference", "score", "measure"]
