import stat
import sys
from pathlib import Path

import pytest

from excore.build import build
from excore.gguf import GGMLType, GGUFFile
from excore.manifest import parse_manifest
from excore.precision import resolve_format
from excore.quantizers import QuantizerError, TensorJob
from excore.quantizers.local_rtn import LocalRTN
from excore.quantizers.runtime_gguf import build_command, find_binary
from excore.validate import audit

ROOT = Path(__file__).resolve().parent.parent
HEAD = "schema: excore/manifest@1\ntrack: CPU-35B\nname: t\n"
MIXED = HEAD + """default: Q4_0
default_quantizer: runtime
rules:
  - {match: 'L*.mlp', format: Q8_0}
  - {match: 'L*.attn.*', format: Q4_0, quantizer: rtn}
  - {match: 'lm_head', format: Q8_0}
"""


@pytest.fixture()
def fake(tmp_path):
    tpl = (Path(__file__).parent / "fake_llama_quantize.py.tpl").read_text()
    p = tmp_path / "llama-quantize"
    p.write_text(tpl.format(python=sys.executable, root=str(ROOT)))
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return str(p)


def test_mixed_runtime_and_rtn_build_audits_clean(fake, tiny_source, tiny_spec, tiny_cfg, tmp_path):
    m = parse_manifest(MIXED, tiny_spec)
    res = build(m, tiny_source, tmp_path / "c.gguf", spec=tiny_spec, cfg=tiny_cfg, llama_quantize=fake)
    with GGUFFile(res.output) as c, GGUFFile(tiny_source) as s:
        assert c.by_name["blk.0.ffn_up.weight"].ggml_type == GGMLType.Q8_0
        assert c.by_name["blk.1.attn_q.weight"].ggml_type == GGMLType.Q4_0
        # runtime-produced bytes are streamed unchanged into the candidate
        want = LocalRTN().quantize_rows(s.read_rows(s.by_name["blk.0.ffn_up.weight"], 0, 512),
                                        resolve_format("Q8_0"))
        assert bytes(c.tensor_view(c.by_name["blk.0.ffn_up.weight"])) == want
    rep = audit(res.output, tiny_source, m, tiny_spec, cfg=tiny_cfg)
    assert rep.ok, rep.summary()
    assert any(f.code == "I_REPLAY_SKIPPED" for f in rep.findings)
    assert rep.stats["replayed_tensors"] == 8   # only the rtn attention tensors are replayed


def test_command_is_explicit_and_anchored(tiny_source, tiny_spec):
    with GGUFFile(tiny_source) as g:
        info = g.by_name["blk.0.ffn_up.weight"]
    job = TensorJob("blk.0.ffn_up.weight", "L0.mlp", resolve_format("Q4_K"), info)
    cmd = build_command("/bin/llama-quantize", "in.gguf", "out.gguf", [job], 8)
    assert cmd[:2] == ["/bin/llama-quantize", "--pure"]
    assert cmd[2:4] == ["--tensor-type", r"^blk\.0\.ffn_up\.weight$=q4_k"]
    assert cmd[-4:] == ["in.gguf", "out.gguf", "Q4_0", "8"]


def test_silent_fallback_is_caught(fake, tiny_source, tiny_spec, tiny_cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "ignore")
    m = parse_manifest(MIXED, tiny_spec)
    with pytest.raises(QuantizerError, match="requested Q8_0"):
        build(m, tiny_source, tmp_path / "c.gguf", spec=tiny_spec, cfg=tiny_cfg, llama_quantize=fake)
    assert not (tmp_path / "c.gguf").exists()


def test_tool_failure_is_reported(fake, tiny_source, tiny_spec, tiny_cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "fail")
    m = parse_manifest(MIXED, tiny_spec)
    with pytest.raises(QuantizerError, match="boom"):
        build(m, tiny_source, tmp_path / "c.gguf", spec=tiny_spec, cfg=tiny_cfg, llama_quantize=fake)


def test_binary_lookup(fake, monkeypatch, tmp_path):
    monkeypatch.delenv("EXCORE_LLAMA_QUANTIZE", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "nothing"))
    with pytest.raises(QuantizerError, match="setup_bench.sh"):
        find_binary(None)
    assert str(find_binary(fake)) == fake
    monkeypatch.setenv("EXCORE_LLAMA_QUANTIZE", fake)
    assert str(find_binary(None)) == fake
