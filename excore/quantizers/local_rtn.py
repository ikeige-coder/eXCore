"""Round-To-Nearest (RTN) quantizer for the legacy block formats, in pure numpy.

Implements Q4_0, Q4_1, Q5_0, Q5_1 and Q8_0 following ggml's reference
``quantize_row_*_ref`` routines. The output is tested byte for byte against the
reference encoders in the ``gguf`` Python package (llama.cpp's own gguf-py); it has not
been compared with a compiled llama.cpp, where exact rounding ties can differ with
compiler float contraction. Every operation here is elementwise IEEE float32, so the
result is identical on every host and a validator can replay it byte for byte
(``regenerable = True``).
"""

from __future__ import annotations

import numpy as np

from ..precision import FORMATS, Format
from . import QuantizerError, RowQuantizer, register

F32 = np.float32
QK = 32


def _blocks(rows: np.ndarray) -> np.ndarray:
    n, k = rows.shape
    return rows.reshape(n, k // QK, QK)


def _f16_bytes(x: np.ndarray) -> np.ndarray:
    """(n, nb, 1) float32 -> (n, nb, 2) uint8 holding little-endian fp16."""
    return np.ascontiguousarray(x.astype(np.float16)).view(np.uint8)


def _inv(d: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d != 0, F32(1) / d, F32(0)).astype(np.float32)


def _round_away(t: np.ndarray) -> np.ndarray:
    """C roundf: nearest, ties away from zero (numpy's rint is ties-to-even)."""
    r = np.rint(t)
    tie = np.abs(t - np.trunc(t)) == F32(0.5)
    return np.where(tie, np.trunc(t) + np.sign(t), r)


def _signed_absmax(x: np.ndarray) -> np.ndarray:
    idx = np.abs(x).argmax(axis=-1)
    return np.take_along_axis(x, idx[..., None], axis=-1)


def _pack_nibbles(q: np.ndarray) -> np.ndarray:
    return (q[..., : QK // 2] & 0x0F) | ((q[..., QK // 2 :] & 0x0F) << 4)


def _pack_high_bits(q: np.ndarray) -> np.ndarray:
    bits = ((q >> 4) & 1).astype(np.uint32) << np.arange(QK, dtype=np.uint32)
    qh = np.bitwise_or.reduce(bits, axis=-1)
    return np.ascontiguousarray(qh[..., None].astype("<u4")).view(np.uint8)


def q8_0(rows: np.ndarray) -> bytes:
    x = _blocks(rows)
    amax = np.abs(x).max(axis=-1, keepdims=True)
    d = amax / F32(127)
    q = _round_away(x * _inv(d)).astype(np.int8).view(np.uint8)
    return np.concatenate([_f16_bytes(d), q], axis=-1).tobytes()


def q4_0(rows: np.ndarray) -> bytes:
    x = _blocks(rows)
    d = _signed_absmax(x) / F32(-8)
    q = np.minimum(np.floor(x * _inv(d) + F32(8.5)), 15).astype(np.uint8)
    return np.concatenate([_f16_bytes(d), _pack_nibbles(q)], axis=-1).tobytes()


def q4_1(rows: np.ndarray) -> bytes:
    x = _blocks(rows)
    mn = x.min(axis=-1, keepdims=True)
    d = (x.max(axis=-1, keepdims=True) - mn) / F32(15)
    q = np.minimum(np.floor((x - mn) * _inv(d) + F32(0.5)), 15).astype(np.uint8)
    return np.concatenate([_f16_bytes(d), _f16_bytes(mn), _pack_nibbles(q)], axis=-1).tobytes()


def q5_0(rows: np.ndarray) -> bytes:
    x = _blocks(rows)
    d = _signed_absmax(x) / F32(-16)
    q = np.minimum(np.floor(x * _inv(d) + F32(16.5)), 31).astype(np.uint8)
    return np.concatenate([_f16_bytes(d), _pack_high_bits(q), _pack_nibbles(q)], axis=-1).tobytes()


def q5_1(rows: np.ndarray) -> bytes:
    x = _blocks(rows)
    mn = x.min(axis=-1, keepdims=True)
    d = (x.max(axis=-1, keepdims=True) - mn) / F32(31)
    q = np.minimum(np.floor((x - mn) * _inv(d) + F32(0.5)), 31).astype(np.uint8)
    return np.concatenate(
        [_f16_bytes(d), _f16_bytes(mn), _pack_high_bits(q), _pack_nibbles(q)], axis=-1
    ).tobytes()


_ENCODERS = {"Q8_0": q8_0, "Q4_0": q4_0, "Q4_1": q4_1, "Q5_0": q5_0, "Q5_1": q5_1}


class LocalRTN(RowQuantizer):
    name = "rtn"
    version = 1
    formats = frozenset(_ENCODERS)

    def quantize_rows(self, rows: np.ndarray, fmt: Format) -> bytes:
        enc = _ENCODERS.get(fmt.name)
        if enc is None:
            raise QuantizerError(f"rtn cannot produce {fmt.name}")
        if rows.dtype != np.float32 or rows.ndim != 2 or rows.shape[1] % QK:
            raise QuantizerError("rtn expects float32 rows whose length is a multiple of 32")
        if not np.isfinite(rows).all():
            raise QuantizerError("source weights contain NaN or Inf")
        return enc(np.ascontiguousarray(rows))


assert set(_ENCODERS) <= set(FORMATS)
register(LocalRTN())
