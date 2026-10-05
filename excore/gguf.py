"""Minimal, dependency-free GGUF v3 reader/writer.

eXCore needs three things from GGUF files and nothing more:

* read the tensor table of a (possibly hostile) file with strict bounds checks,
* stream tensor bytes out of it without loading multi-GB files into memory,
* write a new file whose metadata block is copied verbatim from the source.

Dimension order follows GGUF/ggml: ``dims[0]`` is the *row length* (the axis block
formats pack along), so a linear layer with ``in`` inputs and ``out`` outputs has
``dims == (in, out)``.
"""

from __future__ import annotations

import hashlib
import mmap
import struct
from dataclasses import dataclass
from enum import IntEnum
from math import prod
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np

MAGIC = b"GGUF"
SUPPORTED_VERSION = 3
DEFAULT_ALIGNMENT = 32
MAX_TENSORS = 100_000
MAX_KV = 1_000_000
MAX_STRING = 1 << 24
MAX_DIMS = 4
CHUNK = 64 << 20


class GGUFError(ValueError):
    pass


class GGMLType(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    BF16 = 30


# (elements per block, bytes per block)
TYPE_TRAITS: dict[GGMLType, tuple[int, int]] = {
    GGMLType.F32: (1, 4),
    GGMLType.F16: (1, 2),
    GGMLType.BF16: (1, 2),
    GGMLType.Q4_0: (32, 18),
    GGMLType.Q4_1: (32, 20),
    GGMLType.Q5_0: (32, 22),
    GGMLType.Q5_1: (32, 24),
    GGMLType.Q8_0: (32, 34),
    GGMLType.Q2_K: (256, 84),
    GGMLType.Q3_K: (256, 110),
    GGMLType.Q4_K: (256, 144),
    GGMLType.Q5_K: (256, 176),
    GGMLType.Q6_K: (256, 210),
}

FLOAT_TYPES = frozenset({GGMLType.F32, GGMLType.F16, GGMLType.BF16})

# Byte offsets of fp16 scale fields inside one block, per type (used to reject NaN/Inf scales).
SCALE_OFFSETS: dict[GGMLType, tuple[int, ...]] = {
    GGMLType.Q4_0: (0,),
    GGMLType.Q5_0: (0,),
    GGMLType.Q8_0: (0,),
    GGMLType.Q4_1: (0, 2),
    GGMLType.Q5_1: (0, 2),
    GGMLType.Q2_K: (80, 82),
    GGMLType.Q3_K: (108,),
    GGMLType.Q4_K: (0, 2),
    GGMLType.Q5_K: (0, 2),
    GGMLType.Q6_K: (208,),
}


def align_up(n: int, a: int) -> int:
    return (n + a - 1) // a * a


def row_nbytes(t: GGMLType, row_len: int) -> int:
    elems, nbytes = TYPE_TRAITS[t]
    if row_len <= 0 or row_len % elems:
        raise GGUFError(f"row length {row_len} is not a multiple of the {t.name} block size {elems}")
    return row_len // elems * nbytes


def tensor_nbytes(dims: Sequence[int], t: GGMLType) -> int:
    if not dims or len(dims) > MAX_DIMS or any(d <= 0 for d in dims):
        raise GGUFError(f"bad tensor dims {tuple(dims)}")
    return row_nbytes(t, dims[0]) * prod(dims[1:])


def bits_per_weight(t: GGMLType) -> float:
    elems, nbytes = TYPE_TRAITS[t]
    return nbytes * 8 / elems


# -- bf16 helpers ---------------------------------------------------------------


def f32_to_bf16(a: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even float32 -> bfloat16, returned as little-endian uint16."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return rounded.astype("<u2")


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (np.ascontiguousarray(u16, dtype="<u2").astype(np.uint32) << 16).view(np.float32)


# -- reading --------------------------------------------------------------------


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dims: tuple[int, ...]
    ggml_type: GGMLType
    offset: int            # relative to the start of the data section
    nbytes: int

    @property
    def row_len(self) -> int:
        return self.dims[0]

    @property
    def n_rows(self) -> int:
        return prod(self.dims[1:])


_SCALAR_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


class _Reader:
    def __init__(self, buf):
        self.buf = buf
        self.n = len(buf)

    def unpack(self, fmt: str, pos: int):
        try:
            return struct.unpack_from(fmt, self.buf, pos)
        except struct.error:
            raise GGUFError("truncated GGUF file") from None

    def string(self, pos: int) -> tuple[bytes, int]:
        (ln,) = self.unpack("<Q", pos)
        if ln > MAX_STRING or pos + 8 + ln > self.n:
            raise GGUFError("bad string length in GGUF file")
        return bytes(self.buf[pos + 8 : pos + 8 + ln]), pos + 8 + ln

    def skip_value(self, vtype: int, pos: int, depth: int = 0) -> int:
        if vtype in _SCALAR_SIZE:
            end = pos + _SCALAR_SIZE[vtype]
            if end > self.n:
                raise GGUFError("truncated GGUF file")
            return end
        if vtype == 8:
            return self.string(pos)[1]
        if vtype == 9:
            if depth > 4:
                raise GGUFError("GGUF arrays nested too deeply")
            (etype, count) = self.unpack("<IQ", pos)
            pos += 12
            if count > (1 << 32):
                raise GGUFError("absurd GGUF array length")
            if etype in _SCALAR_SIZE:
                end = pos + count * _SCALAR_SIZE[etype]
                if end > self.n:
                    raise GGUFError("truncated GGUF file")
                return end
            for _ in range(count):
                pos = self.skip_value(etype, pos, depth + 1)
            return pos
        raise GGUFError(f"unknown GGUF value type {vtype}")


@dataclass(frozen=True)
class _Parsed:
    version: int
    tensor_count: int
    kv_count: int
    alignment: int
    metadata_end: int
    tensors: tuple[TensorInfo, ...]
    data_start: int


def _parse(buf) -> _Parsed:
    r = _Reader(buf)
    if r.n < 24 or bytes(buf[:4]) != MAGIC:
        raise GGUFError("not a GGUF file")
    version, tensor_count, kv_count = r.unpack("<IQQ", 4)
    if version != SUPPORTED_VERSION:
        raise GGUFError(f"unsupported GGUF version {version}")
    if tensor_count > MAX_TENSORS or kv_count > MAX_KV:
        raise GGUFError("GGUF header counts are implausibly large")

    pos = 24
    alignment = DEFAULT_ALIGNMENT
    for _ in range(kv_count):
        key, pos = r.string(pos)
        (vtype,) = r.unpack("<I", pos)
        pos += 4
        if key == b"general.alignment" and vtype == 4:
            (alignment,) = r.unpack("<I", pos)
        pos = r.skip_value(vtype, pos)
    if alignment <= 0 or alignment & (alignment - 1):
        raise GGUFError(f"alignment {alignment} is not a power of two")
    metadata_end = pos

    tensors: list[TensorInfo] = []
    seen: set[str] = set()
    for _ in range(tensor_count):
        raw, pos = r.string(pos)
        try:
            name = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise GGUFError("tensor name is not valid UTF-8") from None
        (n_dims,) = r.unpack("<I", pos)
        pos += 4
        if not 1 <= n_dims <= MAX_DIMS:
            raise GGUFError(f"{name}: bad dimension count {n_dims}")
        dims = r.unpack(f"<{n_dims}Q", pos)
        pos += 8 * n_dims
        (ttype, offset) = r.unpack("<IQ", pos)
        pos += 12
        try:
            gt = GGMLType(ttype)
        except ValueError:
            raise GGUFError(f"{name}: unsupported tensor type id {ttype}") from None
        if name in seen:
            raise GGUFError(f"duplicate tensor name {name!r}")
        seen.add(name)
        tensors.append(TensorInfo(name, tuple(dims), gt, offset, tensor_nbytes(dims, gt)))
    return _Parsed(version, tensor_count, kv_count, alignment, metadata_end, tuple(tensors),
                   align_up(pos, alignment))


class GGUFFile:
    """Read-only, memory-mapped view of a GGUF file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fh = open(self.path, "rb")
        try:
            try:
                self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
            except ValueError:
                raise GGUFError("empty file") from None
            try:
                p = _parse(self._mm)
            except Exception:
                self._mm.close()
                raise
        except Exception:
            self._fh.close()
            raise
        self.size = len(self._mm)
        self.version = p.version
        self.tensor_count = p.tensor_count
        self.kv_count = p.kv_count
        self.alignment = p.alignment
        self.data_start = p.data_start
        self.metadata_raw = bytes(self._mm[24 : p.metadata_end])
        self.tensors = p.tensors
        self.by_name = {t.name: t for t in p.tensors}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        try:
            self._mm.close()
        except (BufferError, ValueError):
            pass
        self._fh.close()

    def in_bounds(self, t: TensorInfo) -> bool:
        return t.offset >= 0 and self.data_start + t.offset + t.nbytes <= self.size

    def tensor_view(self, t: TensorInfo) -> memoryview:
        if not self.in_bounds(t):
            raise GGUFError(f"{t.name}: tensor data lies outside the file")
        start = self.data_start + t.offset
        return memoryview(self._mm)[start : start + t.nbytes]

    def iter_tensor_bytes(self, t: TensorInfo, chunk: int = CHUNK) -> Iterator[memoryview]:
        view = self.tensor_view(t)
        for i in range(0, t.nbytes, chunk):
            yield view[i : i + chunk]

    def read_rows(self, t: TensorInfo, start: int, count: int) -> np.ndarray:
        """Rows ``start..start+count`` of a float tensor as a float32 array (count, row_len)."""
        if t.ggml_type not in FLOAT_TYPES:
            raise GGUFError(f"{t.name}: cannot read rows of quantized type {t.ggml_type.name}")
        if start < 0 or count <= 0 or start + count > t.n_rows:
            raise GGUFError(f"{t.name}: row range {start}+{count} outside 0..{t.n_rows}")
        rb = row_nbytes(t.ggml_type, t.row_len)
        sub = self.tensor_view(t)[start * rb : (start + count) * rb]
        if t.ggml_type == GGMLType.BF16:
            return bf16_to_f32(np.frombuffer(sub, dtype="<u2").reshape(count, t.row_len))
        if t.ggml_type == GGMLType.F16:
            return np.frombuffer(sub, dtype="<f2").reshape(count, t.row_len).astype(np.float32)
        return np.frombuffer(sub, dtype="<f4").reshape(count, t.row_len).astype(np.float32)


# -- writing --------------------------------------------------------------------


@dataclass
class WriteEntry:
    name: str
    dims: tuple[int, ...]
    ggml_type: GGMLType
    produce: Callable[[], Iterable[bytes]]   # yields the tensor's bytes in order


def write_gguf(
    path: str | Path,
    *,
    metadata_raw: bytes,
    kv_count: int,
    alignment: int,
    entries: Sequence[WriteEntry],
    version: int = SUPPORTED_VERSION,
) -> str:
    """Write a GGUF file and return the sha256 of everything written."""
    if alignment <= 0 or alignment & (alignment - 1):
        raise GGUFError("alignment must be a power of two")
    layout = []
    cur = 0
    for e in entries:
        nb = tensor_nbytes(e.dims, e.ggml_type)
        layout.append((e, cur, nb))
        cur += align_up(nb, alignment)

    header = bytearray(MAGIC + struct.pack("<IQQ", version, len(entries), kv_count) + metadata_raw)
    for e, off, _ in layout:
        nm = e.name.encode("utf-8")
        header += struct.pack("<Q", len(nm)) + nm
        header += struct.pack("<I", len(e.dims)) + struct.pack(f"<{len(e.dims)}Q", *e.dims)
        header += struct.pack("<IQ", int(e.ggml_type), off)
    header += b"\0" * (align_up(len(header), alignment) - len(header))

    h = hashlib.sha256()
    with open(path, "wb") as fh:
        fh.write(header)
        h.update(header)
        for e, _, nb in layout:
            written = 0
            for chunk in e.produce():
                n = memoryview(chunk).nbytes
                fh.write(chunk)
                h.update(chunk)
                written += n
            if written != nb:
                raise GGUFError(f"{e.name}: produced {written} bytes, expected {nb}")
            pad = align_up(nb, alignment) - nb
            if pad:
                fh.write(b"\0" * pad)
                h.update(b"\0" * pad)
        fh.flush()
    return h.hexdigest()


def pack_metadata(items: Sequence[tuple[str, str, object]]) -> tuple[bytes, int]:
    """Encode simple KV pairs (string, u32, f32, bool, strings) -> (raw bytes, count). For tools and tests."""
    out = bytearray()

    def s(x: str) -> bytes:
        b = x.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    for key, kind, value in items:
        out += s(key)
        if kind == "string":
            out += struct.pack("<I", 8) + s(value)  # type: ignore[arg-type]
        elif kind == "u32":
            out += struct.pack("<II", 4, value)  # type: ignore[arg-type]
        elif kind == "f32":
            out += struct.pack("<If", 6, value)  # type: ignore[arg-type]
        elif kind == "bool":
            out += struct.pack("<I?", 7, value)  # type: ignore[arg-type]
        elif kind == "strings":
            vals = list(value)  # type: ignore[arg-type]
            out += struct.pack("<IIQ", 9, 8, len(vals)) + b"".join(s(v) for v in vals)
        else:
            raise GGUFError(f"unsupported metadata kind {kind!r}")
    return bytes(out), len(items)


# -- integrity helpers ------------------------------------------------------------


def scales_finite(data: memoryview, t: GGMLType) -> bool:
    """False if any block scale (or any float value, for float types) is NaN/Inf."""
    if t in FLOAT_TYPES:
        if t == GGMLType.F32:
            return bool(np.isfinite(np.frombuffer(data, dtype="<f4")).all())
        u = np.frombuffer(data, dtype="<u2")
        if t == GGMLType.F16:
            return bool(np.isfinite(u.view("<f2")).all())
        return not bool(((u & 0x7F80) == 0x7F80).any())  # bf16 exponent all ones
    offsets = SCALE_OFFSETS.get(t)
    if offsets is None:
        return True
    _, bb = TYPE_TRAITS[t]
    blocks = np.frombuffer(data, dtype=np.uint8).reshape(-1, bb)
    for off in offsets:
        col = np.ascontiguousarray(blocks[:, off : off + 2]).view("<f2")
        if not np.isfinite(col).all():
            return False
    return True
