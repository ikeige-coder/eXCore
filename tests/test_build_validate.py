import shutil

import numpy as np
import pytest

from excore.build import BuildError, build, plan_build
from excore.gguf import GGMLType, GGUFFile, WriteEntry, write_gguf
from excore.manifest import parse_manifest
from excore.validate import audit

from conftest import TINY
from excore.model.qwen38_cpu import ModelSpec

HEAD = "schema: excore/manifest@1\ntrack: CPU-35B\nname: t\n"
RTN = HEAD + "default: Q4_0\ndefault_quantizer: rtn\nrules:\n  - {match: 'lm_head', format: Q8_0}\n"


def manifest(text, spec):
    return parse_manifest(text, spec)


@pytest.fixture(scope="module")
def built(tmp_path_factory, tiny_source, tiny_spec, tiny_cfg):
    out = tmp_path_factory.mktemp("out") / "cand.gguf"
    m = manifest(RTN, tiny_spec)
    res = build(m, tiny_source, out, spec=tiny_spec, cfg=tiny_cfg)
    return res, m


def copy_of(built, tmp_path, name="c.gguf"):
    p = tmp_path / name
    shutil.copy(built[0].output, p)
    return p


def patch(path, tensor, offset, data):
    with GGUFFile(path) as g:
        pos = g.data_start + g.by_name[tensor].offset + offset
    with open(path, "r+b") as fh:
        fh.seek(pos)
        fh.write(data)





SRC = []


@pytest.fixture(autouse=True)
def _src(tiny_source):
    SRC[:] = [tiny_source]


# -- build -------------------------------------------------------------------------


def test_build_types_and_frozen_copy(built, tiny_source, tiny_spec):
    res, m = built
    with GGUFFile(tiny_source) as src, GGUFFile(res.output) as cand:
        assert [t.name for t in cand.tensors] == [t.name for t in src.tensors]
        for t in cand.tensors:
            s = src.by_name[t.name]
            if s.ggml_type == GGMLType.F32 or t.name == "token_embd.weight":
                assert bytes(cand.tensor_view(t)) == bytes(src.tensor_view(s))
        assert cand.by_name["output.weight"].ggml_type == GGMLType.Q8_0
        assert cand.by_name["blk.0.ffn_up.weight"].ggml_type == GGMLType.Q4_0
        assert cand.metadata_raw == src.metadata_raw
    assert res.candidate_id == m.candidate_id(tiny_spec)
    assert res.sidecar.exists() and "rtn@v1" in res.sidecar.read_text()


def test_build_is_deterministic(built, tiny_source, tiny_spec, tiny_cfg, tmp_path):
    again = build(built[1], tiny_source, tmp_path / "again.gguf", spec=tiny_spec, cfg=tiny_cfg)
    assert again.sha256 == built[0].sha256
    assert (tmp_path / "again.gguf").read_bytes() == built[0].output.read_bytes()


def test_build_leaves_no_scratch_files(built):
    leftovers = [p.name for p in built[0].output.parent.iterdir()]
    assert not any(n.startswith("excore-build-") or n.endswith(".part") for n in leftovers)


def test_build_smaller_than_source(built, tiny_source):
    assert built[0].size_bytes < tiny_source.stat().st_size


def test_plan_matches_source(tiny_source, tiny_spec):
    with GGUFFile(tiny_source) as src:
        plan = plan_build(manifest(RTN, tiny_spec), tiny_spec, src)
    assert len(plan.items) == len(src.tensors)


def test_unit_tensor_counts(tiny_source, tiny_spec):
    with GGUFFile(tiny_source) as src:
        plan = plan_build(manifest(RTN, tiny_spec), tiny_spec, src)
    # 2 gdn layers x3 + 2 attn layers x4 + 4 mlp x3 tensors + lm_head
    assert sum(1 for i in plan.items if not i.frozen) == 6 + 8 + 12 + 1


def test_shape_mismatch_and_missing_tensors(tiny_source, tiny_cfg):
    wider = ModelSpec.from_mapping({**TINY, "intermediate_size": 768})
    with pytest.raises(BuildError, match="does not match the model config"):
        build(manifest(RTN, wider), tiny_source, tiny_source.parent / "x.gguf", spec=wider, cfg=tiny_cfg)
    deeper = ModelSpec.from_mapping({**TINY, "num_layers": 6})
    with pytest.raises(BuildError, match="missing tensor"):
        build(manifest(RTN, deeper), tiny_source, tiny_source.parent / "x.gguf", spec=deeper, cfg=tiny_cfg)


def test_failed_build_cleans_up(tiny_source, tiny_spec, tiny_cfg, tmp_path):
    m = manifest(HEAD + "default: Q4_K\n", tiny_spec)  # runtime quantizer, no binary available
    import os
    os.environ.pop("EXCORE_LLAMA_QUANTIZE", None)
    old = os.environ.get("PATH")
    os.environ["PATH"] = str(tmp_path)
    try:
        with pytest.raises(Exception, match="llama-quantize not found"):
            build(m, tiny_source, tmp_path / "o" / "x.gguf", spec=tiny_spec, cfg=tiny_cfg)
    finally:
        os.environ["PATH"] = old
    assert not (tmp_path / "o" / "x.gguf").exists()
    assert not [p for p in (tmp_path / "o").iterdir()]


# -- audit -------------------------------------------------------------------------


def test_audit_passes_with_full_replay(built, tiny_spec, tiny_cfg):
    rep = audit(built[0].output, SRC[0], built[1], tiny_spec, cfg=tiny_cfg)
    assert rep.ok, rep.summary()
    assert rep.stats["replayed_tensors"] == 27
    assert rep.stats["execution_map"]["lm_head"] == "Q8_0"
    assert rep.stats["unit_bytes_by_format"]["Q4_0"] > 0


def test_audit_sample_and_none(built, tiny_spec, tiny_cfg):
    rep = audit(built[0].output, SRC[0], built[1], tiny_spec, cfg=tiny_cfg, replay="sample", sample=3, seed=1)
    assert rep.ok and rep.stats["replayed_tensors"] == 3
    rep = audit(built[0].output, SRC[0], built[1], tiny_spec, cfg=tiny_cfg, replay="none")
    assert rep.ok and "replayed_tensors" not in rep.stats


def test_detects_frozen_tamper(built, tiny_spec, tiny_cfg, tmp_path):
    c = copy_of(built, tmp_path)
    patch(c, "token_embd.weight", 100, b"\xff\xff")
    assert "E_FROZEN" in audit(c, SRC[0], built[1], tiny_spec, cfg=tiny_cfg).codes()
    c = copy_of(built, tmp_path, "d.gguf")
    patch(c, "blk.1.attn_norm.weight", 4, b"\x00\x00\x80\x3f")
    assert "E_FROZEN" in audit(c, SRC[0], built[1], tiny_spec, cfg=tiny_cfg).codes()


def test_detects_quantized_data_tamper_by_replay(built, tiny_spec, tiny_cfg, tmp_path):
    c = copy_of(built, tmp_path)
    patch(c, "blk.0.ffn_up.weight", 10, b"\x5a")
    rep = audit(c, SRC[0], built[1], tiny_spec, cfg=tiny_cfg)
    assert rep.codes() == {"E_REPLAY"}
    assert audit(c, SRC[0], built[1], tiny_spec, cfg=tiny_cfg, replay="none").ok  # replay is what catches it


def test_detects_nan_scale(built, tiny_spec, tiny_cfg, tmp_path):
    c = copy_of(built, tmp_path)
    patch(c, "blk.0.ffn_gate.weight", 0, np.array([np.nan], dtype="<f2").tobytes())
    assert "E_SCALES" in audit(c, SRC[0], built[1], tiny_spec, cfg=tiny_cfg).codes()


def test_detects_metadata_tamper(built, tiny_spec, tiny_cfg, tmp_path):
    c = copy_of(built, tmp_path)
    data = c.read_bytes()
    assert b"tiny" in data
    c.write_bytes(data.replace(b"tiny", b"tin1", 1))
    assert "E_METADATA" in audit(c, SRC[0], built[1], tiny_spec, cfg=tiny_cfg).codes()


def test_detects_format_mismatch(built, tiny_spec, tiny_cfg):
    other = manifest(HEAD + "default: Q4_0\ndefault_quantizer: rtn\nrules:\n  - {match: 'L*.mlp', format: Q8_0}\n"
                     "  - {match: 'lm_head', format: Q8_0}\n", tiny_spec)
    rep = audit(built[0].output, SRC[0], other, tiny_spec, cfg=tiny_cfg)
    assert "E_FORMAT" in rep.codes()


def _rewrite(built, tmp_path, mutate):
    out = tmp_path / "w.gguf"
    with GGUFFile(built[0].output) as g:
        entries = [WriteEntry(t.name, t.dims, t.ggml_type, (lambda t=t: g.iter_tensor_bytes(t))) for t in g.tensors]
        entries = mutate(entries)
        write_gguf(out, metadata_raw=g.metadata_raw, kv_count=g.kv_count, alignment=g.alignment, entries=entries)
    return out


def test_detects_missing_extra_and_reordered_tensors(built, tiny_spec, tiny_cfg, tmp_path):
    c = _rewrite(built, tmp_path, lambda e: e[:-1])
    assert "E_TENSOR_SET" in audit(c, SRC[0], built[1], tiny_spec, cfg=tiny_cfg).codes()
    extra = WriteEntry("smuggled.weight", (32,), GGMLType.F32, lambda: [b"\0" * 128])
    c = _rewrite(built, tmp_path, lambda e: e + [extra])
    assert "E_TENSOR_SET" in audit(c, SRC[0], built[1], tiny_spec, cfg=tiny_cfg).codes()
    c = _rewrite(built, tmp_path, lambda e: [e[1], e[0]] + e[2:])
    assert "E_ORDER" in audit(c, SRC[0], built[1], tiny_spec, cfg=tiny_cfg).codes()


def test_detects_wrong_shape(built, tiny_spec, tiny_cfg, tmp_path):
    def mutate(entries):
        out = []
        for e in entries:
            if e.name == "blk.0.ffn_up.weight":   # 256x512 -> 512x256, same byte count
                e = WriteEntry(e.name, (e.dims[1], e.dims[0]), e.ggml_type, e.produce)
            out.append(e)
        return out

    c = _rewrite(built, tmp_path, mutate)
    assert "E_DIMS" in audit(c, SRC[0], built[1], tiny_spec, cfg=tiny_cfg).codes()


def test_detects_truncation_and_garbage(built, tiny_spec, tiny_cfg, tmp_path):
    c = copy_of(built, tmp_path)
    c.write_bytes(c.read_bytes()[:-200])
    assert "E_LAYOUT" in audit(c, SRC[0], built[1], tiny_spec, cfg=tiny_cfg).codes()
    g = tmp_path / "g.gguf"
    g.write_bytes(b"not a gguf file at all" * 10)
    assert audit(g, SRC[0], built[1], tiny_spec, cfg=tiny_cfg).codes() == {"E_PARSE"}
    assert audit(tmp_path / "missing.gguf", SRC[0], built[1], tiny_spec, cfg=tiny_cfg).codes() == {"E_PARSE"}
