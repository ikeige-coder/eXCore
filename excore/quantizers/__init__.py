"""Plugin contract for compression algorithms.

A quantizer turns the *source* weights of one unit tensor into the bytes of a legal
GGUF tensor type. Two flavours exist:

* ``RowQuantizer`` -- pure Python/numpy. It receives float32 rows and returns the
  encoded bytes for those rows. Row-wise independence is what lets the builder stream
  multi-GB tensors in bounded memory, and it is what lets a validator *replay* the
  encoding byte for byte (``regenerable = True``).
* A native ``Quantizer`` (see ``runtime_gguf``) that drives an external tool in
  ``prepare`` and streams the finished tensor in ``encode``.

To add a quantizer: create ``excore/quantizers/<name>.py``, subclass one of the
bases, and call ``register(MyQuantizer())`` at import time. A manifest that names the
quantizer causes the module to be imported (``ensure_loaded``). Miner-supplied
modules are untrusted code: the evaluator imports and runs them only inside its
sandbox, never in the validator's main process.
"""

from __future__ import annotations

import importlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, ClassVar, Iterable, Iterator

import numpy as np

from ..gguf import GGUFFile, TensorInfo, row_nbytes
from ..precision import Format, PrecisionError, register_quantizer

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
CHUNK_ELEMENTS = 16 << 20   # float32 elements per row chunk (~64 MiB)


class QuantizerError(RuntimeError):
    pass


@dataclass(frozen=True)
class TensorJob:
    """One source tensor to encode into ``fmt``."""

    tensor_name: str
    unit: str
    fmt: Format
    info: TensorInfo           # the tensor as it exists in the source file


@dataclass
class BuildContext:
    workdir: Path
    threads: int = 8
    timeout: float = 1800.0
    llama_quantize: str | None = None
    state: dict = field(default_factory=dict)
    _closers: list[Callable[[], None]] = field(default_factory=list)

    def on_close(self, fn: Callable[[], None]) -> None:
        self._closers.append(fn)

    def cleanup(self) -> None:
        while self._closers:
            try:
                self._closers.pop()()
            except Exception:  # cleanup must never mask the real error
                pass


class Quantizer(ABC):
    name: ClassVar[str]
    version: ClassVar[int] = 1
    formats: ClassVar[frozenset[str]]
    regenerable: ClassVar[bool] = False   # True: validators may re-encode and byte-compare

    @property
    def fingerprint(self) -> str:
        return f"{self.name}@v{self.version}"

    def prepare(self, source: GGUFFile, jobs: list[TensorJob], ctx: BuildContext) -> None:
        """Optional batch step run once before any ``encode`` call."""

    @abstractmethod
    def encode(self, job: TensorJob, source: GGUFFile, ctx: BuildContext) -> Iterator[bytes]:
        """Yield the encoded bytes of ``job`` in order. Total length must match the tensor type."""


class RowQuantizer(Quantizer):
    """Quantizer defined by a pure function over float32 rows."""

    regenerable = True

    @abstractmethod
    def quantize_rows(self, rows: np.ndarray, fmt: Format) -> bytes:
        """``rows`` is float32 (n, row_len); return exactly n * row_bytes bytes."""

    def encode(self, job: TensorJob, source: GGUFFile, ctx: BuildContext) -> Iterator[bytes]:
        info = job.info
        gt = _ggml(job.fmt)
        rb = row_nbytes(gt, info.row_len)
        step = max(1, CHUNK_ELEMENTS // info.row_len)
        for start in range(0, info.n_rows, step):
            n = min(step, info.n_rows - start)
            rows = source.read_rows(info, start, n)
            out = self.quantize_rows(rows, job.fmt)
            if len(out) != n * rb:
                raise QuantizerError(
                    f"{self.name}: {job.tensor_name} rows {start}+{n} encoded to {len(out)} "
                    f"bytes, expected {n * rb}"
                )
            yield out


def _ggml(fmt: Format):
    from ..gguf import GGMLType

    return GGMLType[fmt.name]


# -- registry ------------------------------------------------------------------

_REGISTRY: dict[str, Quantizer] = {}
BUILTIN = ("runtime", "rtn")
_MODULE_FOR = {"runtime": "runtime_gguf", "rtn": "local_rtn"}


def register(q: Quantizer) -> Quantizer:
    if not NAME_RE.match(q.name):
        raise QuantizerError(f"bad quantizer name {q.name!r}; use [a-z][a-z0-9_]{{0,31}}")
    if q.name in _REGISTRY and type(_REGISTRY[q.name]) is not type(q):
        raise QuantizerError(f"quantizer {q.name!r} is already registered by another class")
    try:
        register_quantizer(q.name, q.formats)
    except PrecisionError as exc:
        raise QuantizerError(str(exc)) from None
    _REGISTRY[q.name] = q
    return q


def get_quantizer(name: str) -> Quantizer:
    ensure_loaded([name])
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)


def ensure_loaded(names: Iterable[str]) -> None:
    """Import the module behind each quantizer name so it can register itself."""
    for name in names:
        if name in _REGISTRY:
            continue
        if not NAME_RE.match(name):
            raise QuantizerError(f"bad quantizer name {name!r}")
        module = _MODULE_FOR.get(name, name)
        try:
            importlib.import_module(f"{__name__}.{module}")
        except ModuleNotFoundError as exc:
            if exc.name == f"{__name__}.{module}":
                raise QuantizerError(f"unknown quantizer {name!r}") from None
            raise
        if name not in _REGISTRY:
            raise QuantizerError(f"module for quantizer {name!r} did not register it")


# Built-ins register on import so manifests can reference them immediately.
from . import local_rtn, runtime_gguf  # noqa: E402,F401
