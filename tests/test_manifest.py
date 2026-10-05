from pathlib import Path

import pytest

from excore.manifest import ManifestError, load_manifest, parse_layers, parse_manifest

ROOT = Path(__file__).resolve().parent.parent

HEAD = "schema: excore/manifest@1\ntrack: CPU-35B\nname: t\n"


def parse(body, spec):
    return parse_manifest(HEAD + body, spec)


def test_seed_variants_load_and_expand(spec):
    for f in sorted((ROOT / "experiments" / "base_variants").glob("*.yaml")):
        m = load_manifest(f, spec)
        a = m.expand(spec)
        assert len(a) == 358, f.name


def test_v0_is_uniform_except_head(spec):
    m = load_manifest(ROOT / "experiments/base_variants/V0_baseline.yaml", spec)
    a = m.expand(spec)
    assert {v.format for n, v in a.items() if n != "lm_head"} == {"Q4_K"}
    assert a["lm_head"].format == "Q6_K"


def test_later_rules_override(spec):
    m = parse(
        "default: Q4_K\nrules:\n  - {match: 'L*.mlp', format: Q5_K}\n  - {match: 'L1.mlp', format: Q8_0}\n",
        spec,
    )
    a = m.expand(spec)
    assert a["L0.mlp"].format == "Q5_K" and a["L1.mlp"].format == "Q8_0"
    assert a["L1.mlp"].rule == 1 and a["L0.gdn.qkv"].rule == -1


def test_layer_filter(spec):
    m = parse("default: Q4_K\nrules:\n  - {match: 'L*.mlp', layers: '0-2', format: Q8_0}\n", spec)
    a = m.expand(spec)
    assert [a[f"L{i}.mlp"].format for i in range(4)] == ["Q8_0", "Q8_0", "Q8_0", "Q4_K"]


def test_candidate_id_ignores_name_and_rule_layout(spec):
    a = parse("default: Q4_K\n", spec)
    b = parse_manifest(
        "schema: excore/manifest@1\ntrack: CPU-35B\nname: other\ndefault: Q5_K\n"
        "rules:\n  - {match: '*', format: Q4_K}\n",
        spec,
    )
    assert a.candidate_id(spec) == b.candidate_id(spec)
    c = parse("default: Q5_K\n", spec)
    assert a.candidate_id(spec) != c.candidate_id(spec)


@pytest.mark.parametrize(
    "body, fragment",
    [
        ("default: Q4_K\nbogus: 1\n", "unknown top-level key"),
        ("default: Q9\n", "unknown format"),
        ("rules: []\n", "'default' format is required"),
        ("default: Q4_K\nrules:\n  - {match: 'L*.zzz', format: Q4_K}\n", "matches no units"),
        ("default: Q4_K\nrules:\n  - {match: 'lm_head', layers: 3, format: Q4_K}\n", "matches no units"),
        ("default: Q4_K\nrules:\n  - {match: 'lm_head', format: Q2_K}\n", "not a legal format"),
        ("default: Q4_K\nrules:\n  - {match: 'L*.mlp', format: Q4_K, quantizer: rtn}\n", "cannot produce"),
        ("default: Q4_K\nrules:\n  - {match: 'L*.mlp', layers: '0-99', format: Q4_K}\n", "outside"),
        ("default: Q4_K\nrules:\n  - {match: 'L*.mlp', format: Q4_K, extra: 1}\n", "unknown key"),
        ("default: Q2_K\n", "not a legal format"),
    ],
)
def test_rejections(spec, body, fragment):
    with pytest.raises(ManifestError) as exc:
        parse(body, spec).expand(spec)
    assert fragment in str(exc.value)


def test_wrong_schema_and_track(spec):
    with pytest.raises(ManifestError) as exc:
        parse_manifest("schema: x\ntrack: GPU-01\nname: t\ndefault: Q4_K\n", spec)
    assert "schema" in str(exc.value) and "track" in str(exc.value)


def test_duplicate_keys_and_aliases_rejected(spec):
    with pytest.raises(ManifestError, match="duplicate key"):
        parse_manifest(HEAD + "default: Q4_K\ndefault: Q8_0\n", spec)
    with pytest.raises(ManifestError, match="aliases"):
        parse_manifest(HEAD + "default: &a Q4_K\nrules:\n  - {match: 'lm_head', format: *a}\n", spec)


def test_oversize_rejected(spec):
    with pytest.raises(ManifestError, match="exceeds"):
        parse_manifest(HEAD + "default: Q4_K\n# " + "x" * 70000 + "\n", spec)


def test_all_problems_reported_together(spec):
    with pytest.raises(ManifestError) as exc:
        parse(
            "default: Q4_K\nrules:\n  - {match: 'L*.mlp', format: Q9}\n  - {match: 'L*.zzz', format: Q4_K}\n",
            spec,
        )
    assert len(exc.value.problems) >= 1


@pytest.mark.parametrize(
    "value, expected",
    [("0-3", (0, 1, 2, 3)), ("0-1,5,7-8", (0, 1, 5, 7, 8)), (4, (4,)), ([1, 2], (1, 2))],
)
def test_parse_layers(value, expected):
    assert parse_layers(value, 64) == expected


@pytest.mark.parametrize("bad", ["5-2", "a", "", True, 64, "0-64", 1.5])
def test_parse_layers_bad(bad):
    with pytest.raises(ValueError):
        parse_layers(bad, 64)


def test_default_only_checked_where_it_still_applies(spec):
    # Q2_K is illegal for lm_head, but a rule overrides lm_head, so the manifest is fine.
    m = parse("default: Q2_K\nrules:\n  - {match: 'lm_head', format: Q6_K}\n", spec)
    assert m.expand(spec)["lm_head"].format == "Q6_K"
    with pytest.raises(ManifestError, match="not a legal format"):
        parse("default: Q2_K\n", spec).expand(spec)
