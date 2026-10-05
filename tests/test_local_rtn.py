import numpy as np
import pytest

from excore.precision import resolve_format
from excore.quantizers import QuantizerError, get_quantizer
from excore.quantizers.local_rtn import LocalRTN

NAMES = ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0"]
rtn = LocalRTN()


def rows(seed=0, n=6, k=256, scale=0.1):
    return np.random.default_rng(seed).normal(0, scale, (n, k)).astype(np.float32)


@pytest.mark.parametrize("name", NAMES)
def test_output_size(name):
    from excore.gguf import GGMLType, row_nbytes

    fmt = resolve_format(name)
    out = rtn.quantize_rows(rows(), fmt)
    assert len(out) == 6 * row_nbytes(GGMLType[name], 256)


@pytest.mark.parametrize("name", NAMES)
def test_matches_reference_implementation(name):
    gguf_quants = pytest.importorskip("gguf.quants")
    import gguf

    x = rows(seed=3, n=16, k=512)
    ref = gguf_quants.quantize(x, getattr(gguf.GGMLQuantizationType, name))
    assert rtn.quantize_rows(x, resolve_format(name)) == ref.tobytes()


@pytest.mark.parametrize("name", NAMES)
def test_roundtrip_error_is_small(name):
    gguf_quants = pytest.importorskip("gguf.quants")
    import gguf

    x = rows(seed=5, n=8, k=256)
    q = np.frombuffer(rtn.quantize_rows(x, resolve_format(name)), dtype=np.uint8)
    t = getattr(gguf.GGMLQuantizationType, name)
    bs, ts = gguf.GGML_QUANT_SIZES[t]
    deq = gguf_quants.dequantize(q.reshape(8, 256 // bs * ts), t)
    assert np.abs(deq - x).max() < 0.1 * np.abs(x).max()


@pytest.mark.parametrize("name", NAMES)
def test_deterministic_and_chunk_invariant(name):
    fmt = resolve_format(name)
    x = rows(seed=9, n=8)
    whole = rtn.quantize_rows(x, fmt)
    parts = b"".join(rtn.quantize_rows(x[i : i + 3], fmt) for i in range(0, 8, 3))
    assert whole == parts == rtn.quantize_rows(x.copy(), fmt)


@pytest.mark.parametrize("name", NAMES)
def test_all_zero_block(name):
    out = rtn.quantize_rows(np.zeros((1, 32), dtype=np.float32), resolve_format(name))
    assert len(out) > 0


def test_rejects_bad_input():
    fmt = resolve_format("Q8_0")
    bad = rows()
    bad[0, 0] = np.nan
    with pytest.raises(QuantizerError, match="NaN"):
        rtn.quantize_rows(bad, fmt)
    with pytest.raises(QuantizerError):
        rtn.quantize_rows(rows(k=48), fmt)
    with pytest.raises(QuantizerError, match="cannot produce"):
        rtn.quantize_rows(rows(), resolve_format("Q4_K"))


def test_registry():
    assert get_quantizer("rtn").regenerable
    assert not get_quantizer("runtime").regenerable
    with pytest.raises(QuantizerError, match="unknown quantizer"):
        get_quantizer("does_not_exist")
    with pytest.raises(QuantizerError, match="bad quantizer name"):
        get_quantizer("../evil")
