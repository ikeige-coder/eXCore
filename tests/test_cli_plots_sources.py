import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

from synthetic import SyntheticBackend, sample_texts

from excore.cli import main, reference_build
from excore.eval.accuracy import Reference, lock_corpus
from excore.frontier.core_frontier import Entry, Point, Space, Spectrum
from excore.frontier.plots import PANELS, render_panel, write_plots
from excore.manifest import parse_manifest
from excore.screen import screen
from excore.sources import (
    SourcesError, is_pinned, load_lock, pin, require_pinned, source_gguf, unpinned_reasons, verify_sources,
)

ROOT = Path(__file__).resolve().parent.parent
RTN = ("schema: excore/manifest@1\ntrack: CPU-35B\nname: {n}\ndefault: Q4_0\ndefault_quantizer: rtn\n"
       "rules:\n  - {{match: 'lm_head', format: Q8_0}}\n")


@pytest.fixture()
def tiny_config(tiny_cfg, tmp_path):
    p = tmp_path / "tiny.yaml"
    p.write_text(yaml.safe_dump(dict(tiny_cfg)))
    return str(p)


# -- screening ---------------------------------------------------------------------------


def test_screen_estimates_real_recipes(spec, cfg):
    from excore.manifest import load_manifest

    v0 = screen(load_manifest(ROOT / "experiments/base_variants/V0_baseline.yaml", spec), spec, cfg)
    assert v0.units == 358 and v0.formats == {"Q4_K": 357, "Q6_K": 1}
    assert 20 < v0.file_gib < 24 and v0.within_ceiling and 4.4 < v0.avg_bits_per_weight < 4.8
    assert v0.est_ram_gib == pytest.approx(v0.file_gib - v0.embedding_gib + 1.5)          # embeddings are not resident
    assert v0.est_ram_gib < cfg['limits']['max_peak_rss_gib'] - 3                          # the baseline leaves headroom
    fat = screen(parse_manifest("schema: excore/manifest@1\ntrack: CPU-35B\nname: fat\ndefault: Q8_0\n", spec), spec, cfg)
    assert fat.file_gib > v0.file_gib * 1.5


def test_screen_flags_oversize(spec, cfg):
    big = screen(parse_manifest("schema: excore/manifest@1\ntrack: CPU-35B\nname: big\ndefault: BF16\n", spec), spec, cfg)
    assert not big.within_ceiling


# -- cli ----------------------------------------------------------------------------------


def test_check_command_exit_codes(tiny_config, tmp_path, capsys):
    ok = tmp_path / "ok.yaml"
    ok.write_text(RTN.format(n="ok"))
    assert main(["--config", tiny_config, "check", str(ok)]) == 0
    assert "the manifest is legal" in capsys.readouterr().out
    bad = tmp_path / "bad.yaml"
    bad.write_text(RTN.format(n="bad") + "  - {match: 'L*.zzz', format: Q4_0}\n")
    assert main(["--config", tiny_config, "check", str(bad)]) == 2
    assert "matches no units" in capsys.readouterr().err
    unknown = tmp_path / "u.yaml"
    unknown.write_text(RTN.format(n="u").replace("rtn", "nonexistent"))
    assert main(["--config", tiny_config, "check", str(unknown)]) == 2
    assert "unknown quantizer" in capsys.readouterr().err


def test_check_flags_a_recipe_over_the_ram_ceiling(capsys):
    assert main(["check", str(ROOT / "experiments/base_variants/V0_baseline.yaml")]) == 0
    big = ROOT / "tests" / "_big_tmp.yaml"
    big.write_text("schema: excore/manifest@1\ntrack: CPU-35B\nname: big\ndefault: BF16\n")
    try:
        assert main(["check", str(big)]) == 3
    finally:
        big.unlink()
    assert "RAM ceiling" in capsys.readouterr().err


def test_build_then_audit_via_cli(tiny_config, tiny_source, tmp_path, capsys):
    m = tmp_path / "m.yaml"
    m.write_text(RTN.format(n="m"))
    out = tmp_path / "cand.gguf"
    assert main(["--config", tiny_config, "build", str(m), "--source", str(tiny_source), "--out", str(out)]) == 0
    assert out.exists() and "candidate" in capsys.readouterr().out
    assert main(["--config", tiny_config, "audit", str(m), "--candidate", str(out), "--source", str(tiny_source)]) == 0
    assert main(["--config", tiny_config, "audit", str(m), "--candidate", str(out), "--source", str(tiny_source),
                 "--json", "--replay", "none"]) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["ok"] is True
    data = bytearray(out.read_bytes())
    data[-100] ^= 0xFF
    out.write_bytes(bytes(data))
    assert main(["--config", tiny_config, "audit", str(m), "--candidate", str(out), "--source", str(tiny_source)]) == 1
    assert "E_REPLAY" in capsys.readouterr().out


def test_reference_command(tiny_cfg, tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name, text in sample_texts():
        (corpus / name).write_text(text)
    lock_corpus(corpus)
    cfg = {**tiny_cfg, "accuracy": {**tiny_cfg["accuracy"], "ctx": 64, "top_k": 16, "max_tokens": 3000}}
    out = reference_build(cfg, "unused.gguf", str(corpus), str(tmp_path / "ref.npz"), "model-sha",
                          backend_factory=lambda: SyntheticBackend())
    ref = Reference.load(out)
    assert ref.meta["model_id"] == "model-sha" and ref.meta["ctx"] == 64 and ref.ids.shape[1] == 16
    (corpus / "extra.txt").write_text("sneaky")
    with pytest.raises(Exception, match="does not match its lock"):
        reference_build(cfg, "unused.gguf", str(corpus), str(tmp_path / "r2.npz"), "m",
                        backend_factory=lambda: SyntheticBackend())


# -- sources ---------------------------------------------------------------------------------


@pytest.fixture()
def lock_and_model(tmp_path):
    lock = tmp_path / "sources.lock.json"
    shutil.copy(ROOT / "configs/sources.lock.json", lock)
    models = tmp_path / "models"
    models.mkdir()
    (models / "base-bf16.gguf").write_bytes(b"pretend weights" * 1000)
    return lock, models


def test_default_lock_is_unpinned():
    lock = load_lock(ROOT / "configs/sources.lock.json", track="CPU-35B")
    assert not is_pinned(lock)
    assert any("no pinned files" in r for r in unpinned_reasons(lock))
    with pytest.raises(SourcesError, match="not verified"):
        require_pinned(ROOT / "configs/sources.lock.json", ROOT)


def test_pin_verify_and_tamper(lock_and_model):
    lock, models = lock_and_model
    with pytest.raises(SourcesError, match="40-character"):
        pin(lock, models, repo="org/m", revision="r1", files=["base-bf16.gguf"], llama_cpp_commit="abc")
    pin(lock, models, repo="org/m", revision="r1", files=["base-bf16.gguf"], llama_cpp_commit="a" * 40)
    cache = lock.parent / "verified.json"
    assert is_pinned(load_lock(lock))
    assert require_pinned(lock, models, track="CPU-35B", cache=cache)
    assert cache.exists() and source_gguf(load_lock(lock), models).name == "base-bf16.gguf"
    (models / "base-bf16.gguf").write_bytes(b"pretend weights" * 999 + b"tampered 123456")   # same size
    problems = verify_sources(load_lock(lock), models)
    assert problems and "sha256 does not match" in problems[0]
    (models / "base-bf16.gguf").unlink()
    assert "not found" in verify_sources(load_lock(lock), models)[0]
    with pytest.raises(SourcesError, match="track"):
        load_lock(lock, track="GPU-01")


def test_sources_cli(lock_and_model, tiny_config):
    lock, models = lock_and_model
    assert main(["--config", tiny_config, "sources", "verify", "--lock", str(lock), "--root", str(models)]) == 1
    assert main(["--config", tiny_config, "sources", "pin", "--lock", str(lock), "--root", str(models),
                 "--file", "base-bf16.gguf", "--llama-cpp-commit", "b" * 40]) == 0
    assert main(["--config", tiny_config, "sources", "verify", "--lock", str(lock), "--root", str(models)]) == 0


# -- plots -----------------------------------------------------------------------------------


def P(cid, kl, gen, pre, ram, name=None):
    return Point(cid, name or cid, kl, gen, pre, ram)


@pytest.fixture()
def spectrum(cfg):
    sp = Spectrum("CPU-35B", Space.from_config(cfg))
    for e in (Entry(P("base", 0.02, 1.0, 1.0, 20.0), 0.0, None),
              Entry(P("small", 0.04, 1.0, 1.0, 12.0), 0.1, "core:XL"),
              Entry(P("best", 0.01, 1.3, 1.3, 14.0, name="a<b&c"), 0.3, "core:Ultra")):
        sp.add(e)
    return sp


def test_plots_are_valid_svg_with_every_point(spectrum, tmp_path):
    paths = write_plots(spectrum, tmp_path / "plots")
    assert [p.name for p in paths] == [f"{p.slug}.svg" for p in PANELS]
    for p in paths:
        root = ET.fromstring(p.read_text())
        circles = [c for c in root.iter("{http://www.w3.org/2000/svg}circle")]
        assert len(circles) == 3
        assert sorted(c.get("class") for c in circles) == ["front", "old", "old"] or \
               sorted(c.get("class") for c in circles).count("front") == len(spectrum.active())
    assert "a&lt;b&amp;c" in paths[0].read_text()


def test_empty_spectrum_renders(cfg):
    sp = Spectrum("CPU-35B", Space.from_config(cfg))
    root = ET.fromstring(render_panel(sp, PANELS[0]))
    assert "no entries yet" in "".join(root.itertext())


def test_plots_cli(spectrum, tmp_path, tiny_config):
    sp_file = tmp_path / "spectrum.json"
    spectrum.save(sp_file)
    assert main(["plots", "--spectrum", str(sp_file), "--out-dir", str(tmp_path / "out")]) == 0
    assert len(list((tmp_path / "out").glob("*.svg"))) == 3
