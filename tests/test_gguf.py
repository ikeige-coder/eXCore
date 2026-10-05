import struct

import numpy as np
import pytest

from excore.gguf import (
    GGMLType,
    GGUFError,
    GGUFFile,
    TYPE_TRAITS,
    WriteEntry,
    bf16_to_f32,
    bits_per_weight,
    f32_to_bf16,
    pack_metadata,
    scales_finite,
    tensor_nbytes,
    write_gguf,
)
from excore.precision import FORMATS


def test_format_table_matches_gguf_traits():
    for name, fmt in FORMATS.items():
        t = GGMLType[name]
        assert bits_per_weight(t) == fmt.bits_per_weight, name
        assert TYPE_TRAITS[t][0] == fmt.block_size, name


def test_bf16_roundtrip_is_close():
    x = np.random.default_rng(0).normal(size=1000).astype(np.float32)
    y = bf16_to_f32(f32_to_bf16(x))
    assert np.allclose(x, y, rtol=1e-2, atol=1e-3)
    assert bf16_to_f32(f32_to_bf16(np.array([1.0], dtype=np.float32)))[0] == 1.0


def test_write_read_roundtrip(tmp_path):
    meta, kv = pack_metadata([("general.architecture", "string", "x"), ("a.b", "u32", 7)])
    a = np.arange(64, dtype="<f4").tobytes()
    b = f32_to_bf16(np.ones(256, dtype=np.float32)).tobytes()
    entries = [
        WriteEntry("a", (8, 8), GGMLType.F32, lambda: [a]),
        WriteEntry("b", (256,), GGMLType.BF16, lambda: [b[:100], b[100:]]),
    ]
    p = tmp_path / "t.gguf"
    sha = write_gguf(p, metadata_raw=meta, kv_count=kv, alignment=32, entries=entries)
    assert len(sha) == 64
    with GGUFFile(p) as g:
        assert [t.name for t in g.tensors] == ["a", "b"]
        assert g.metadata_raw == meta and g.kv_count == 2
        assert bytes(g.tensor_view(g.by_name["a"])) == a
        assert np.array_equal(g.read_rows(g.by_name["a"], 1, 2)[0], np.arange(8, 16, dtype=np.float32))
        assert g.data_start % 32 == 0 and g.size % 32 == 0


def test_wrong_produced_length_rejected(tmp_path):
    e = WriteEntry("a", (8,), GGMLType.F32, lambda: [b"\0" * 31])
    with pytest.raises(GGUFError, match="produced"):
        write_gguf(tmp_path / "x.gguf", metadata_raw=b"", kv_count=0, alignment=32, entries=[e])


def test_deterministic_bytes(tmp_path):
    e = lambda: [WriteEntry("a", (8,), GGMLType.F32, lambda: [b"\x01" * 32])]
    h1 = write_gguf(tmp_path / "1.gguf", metadata_raw=b"", kv_count=0, alignment=32, entries=e())
    h2 = write_gguf(tmp_path / "2.gguf", metadata_raw=b"", kv_count=0, alignment=32, entries=e())
    assert h1 == h2


@pytest.mark.parametrize(
    "blob, msg",
    [
        (b"", "empty"),
        (b"NOPE" + b"\0" * 40, "not a GGUF"),
        (b"GGUF" + struct.pack("<IQQ", 2, 0, 0), "unsupported GGUF version"),
        (b"GGUF" + struct.pack("<IQQ", 3, 10**9, 0), "implausibly large"),
        (b"GGUF" + struct.pack("<IQQ", 3, 1, 0), "truncated"),
    ],
)
def test_hostile_files_rejected(tmp_path, blob, msg):
    p = tmp_path / "bad.gguf"
    p.write_bytes(blob)
    with pytest.raises(GGUFError, match=msg):
        GGUFFile(p)


def test_truncated_data_detected(tmp_path):
    e = WriteEntry("a", (64,), GGMLType.F32, lambda: [b"\0" * 256])
    p = tmp_path / "t.gguf"
    write_gguf(p, metadata_raw=b"", kv_count=0, alignment=32, entries=[e])
    p.write_bytes(p.read_bytes()[:-100])
    with GGUFFile(p) as g:
        assert not g.in_bounds(g.tensors[0])
        with pytest.raises(GGUFError):
            g.tensor_view(g.tensors[0])


def test_block_misaligned_row_rejected():
    with pytest.raises(GGUFError):
        tensor_nbytes((100,), GGMLType.Q4_K)


def test_scales_finite():
    good = np.ones(64, dtype="<f2").tobytes()
    bad = np.array([np.nan] * 32, dtype="<f2").tobytes()
    block = lambda d: d + b"\0" * (34 - 2)
    assert scales_finite(memoryview(block(good[:2])), GGMLType.Q8_0)
    assert not scales_finite(memoryview(block(bad[:2])), GGMLType.Q8_0)
    assert not scales_finite(memoryview(np.array([np.inf], dtype="<f4").tobytes()), GGMLType.F32)
