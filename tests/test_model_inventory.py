from collections import Counter

from excore.model.qwen38_cpu import ATTN, GDN, LM_HEAD, MLP, build_inventory


def test_unit_counts(spec):
    units = build_inventory(spec)
    kinds = Counter(u.kind for u in units)
    assert len(units) == 358
    assert kinds == {GDN: 189, ATTN: 84, MLP: 84, LM_HEAD: 1}


def test_layer_layout(spec):
    assert len(spec.gdn_layers) == 63
    assert len(spec.attention_layers) == 21
    assert spec.attention_layers[:3] == (3, 7, 11)
    assert spec.layer_kind(83) == ATTN and spec.layer_kind(0) == GDN


def test_names_unique_and_stable(spec):
    names = [u.name for u in build_inventory(spec)]
    assert len(set(names)) == len(names)
    assert names[0] == "L0.gdn.qkv" and names[-1] == "lm_head"
    assert "L3.attn.q" in names and "L12.mlp" in names


def test_total_params_near_35b(spec):
    total = sum(u.params for u in build_inventory(spec)) + spec.vocab_size * spec.hidden_size   # + embeddings
    assert 34e9 < total < 36e9


def test_all_rows_block_aligned_for_k_quants(spec):
    for u in build_inventory(spec):
        for t in u.tensors:
            assert t.in_features % 256 == 0, t.gguf_name


def test_the_archived_27b_track_is_still_just_a_config():
    from pathlib import Path

    from excore.config import load_config
    from excore.model.qwen38_cpu import ModelSpec

    cfg = load_config(Path(__file__).resolve().parent.parent / "configs/tracks/cpu-01-27b.yaml")
    old = ModelSpec.from_mapping(cfg["model"])
    units = build_inventory(old)
    assert cfg["track"] == "CPU-27B" and len(units) == 273
    assert Counter(u.kind for u in units) == {GDN: 144, ATTN: 64, MLP: 64, LM_HEAD: 1}


def test_v2_track_limits(cfg, spec):
    assert cfg["track"] == "CPU-35B" and spec.num_layers == 84
    assert cfg["limits"]["max_peak_rss_gib"] <= 24.0 and cfg["limits"]["threads"] == 6
    assert cfg["frontier"]["bounds"]["ram_gib"]["worst"] == cfg["limits"]["max_peak_rss_gib"]
    assert cfg["limits"]["min_free_ram_gib"] >= cfg["limits"]["max_peak_rss_gib"]
