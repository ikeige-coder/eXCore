import yaml
import pytest

from synthetic import Leaky, SyntheticBackend, sample_texts

from excore.eval.accuracy import build_reference, make_windows
from excore.eval.performance import PerfResult, PerfRun
from excore.manifest import parse_manifest
from evaluator.runner import EvalFailure, LocalRunner

HEAD = "schema: excore/manifest@1\ntrack: CPU-35B\nname: {n}\n"
V0 = HEAD.format(n="v0") + "default: Q4_0\ndefault_quantizer: rtn\nrules:\n  - {match: 'lm_head', format: Q8_0}\n"
SMALL = HEAD.format(n="small") + ("default: Q4_0\ndefault_quantizer: rtn\nrules:\n  - {match: 'lm_head', format: Q8_0}\n"
                                  "  - {match: 'L*.mlp', format: Q4_1}\n")


def perf_for(path, *, gen=10.0, stable=True):
    ram = 8.0 + path.stat().st_size / 2e6          # smaller file -> less RAM
    run = PerfRun(100.0, gen, ram, 1.0)
    return PerfResult(100.0, gen, ram, (run, run), stable, 0.0 if stable else 0.4, 0.0, 4096, 128, 8)


@pytest.fixture()
def env(tmp_path, tiny_cfg, tiny_source, tiny_spec):
    cfgfile = tmp_path / "cfg.yaml"
    cfgfile.write_text(yaml.safe_dump(dict(tiny_cfg)))
    base = SyntheticBackend()
    ref = build_reference(base, make_windows(base, sample_texts(), 64, 5000), k=16, corpus_digest="d", model_id="m")
    ref.save(tmp_path / "ref.npz")
    (tmp_path / "v0.yaml").write_text(V0)
    shards = tmp_path / "holdout"
    for n in (0, 1):
        d = shards / f"shard-{n}"
        d.mkdir(parents=True)
        w = make_windows(base, sample_texts(seed=10 + n), 64, 3000)
        build_reference(base, w, k=8, corpus_digest=f"h{n}", model_id="m").save(d / "reference.npz")
    calls = {"perf": 0}

    def perf_fn(p):
        calls["perf"] += 1
        return perf_for(p, stable=not p.name.startswith("cand-unstable"))

    def make(**kw):
        return LocalRunner(
            tiny_cfg, repo_root=tmp_path, source=tiny_source, reference_path=tmp_path / "ref.npz",
            workdir=tmp_path / "work", config_path=cfgfile, baseline_manifest=tmp_path / "v0.yaml",
            perf_fn=kw.pop("perf_fn", perf_fn),
            backend_factory=kw.pop("backend_factory", lambda p: SyntheticBackend(noise=0.1 if "baseline" in p.name else 0.2)),
            generative_factory=kw.pop("generative_factory", lambda p, n: Leaky(10 ** 6)),
            holdout_root=shards, needle_seed=7, **kw)

    return make, calls, tmp_path, tiny_spec


def evaluate(runner, text, name, tmp_path, spec):
    path = tmp_path / f"{name}.yaml"
    path.write_text(text)
    m = parse_manifest(text, spec)
    return runner.evaluate(path, m, m.candidate_id(spec))


def test_full_evaluation_builds_audits_measures_and_guards(env):
    make, calls, tmp, spec = env
    r = make()
    b = evaluate(r, SMALL, "small", tmp, spec)
    assert b.candidate_path.exists() and b.candidate_path.name.startswith("cand-")
    assert b.accuracy.rp_kl > b.baseline_public_kl > 0           # noise 0.2 vs 0.1
    assert b.baseline_perf.gen_tps == 10.0 and b.perf.peak_rss_gib < 20
    assert [g.name for g in b.guards] == ["needle8192", "needle16384"]
    assert all(g.reference_score == 1.0 and g.candidate_score == 1.0 and g.passed for g in b.guards)
    assert not b.guards_skipped and b.host["machine"]
    r.cleanup(b)
    assert not b.candidate_path.exists() and r.baseline_path.exists()


def test_baseline_is_cached_across_runners(env):
    make, calls, tmp, spec = env
    evaluate(make(), SMALL, "small", tmp, spec)
    first = calls["perf"]
    assert first == 2                                           # baseline + candidate
    evaluate(make(), SMALL.replace("name: small", "name: small2").replace("Q4_1", "Q5_0"), "small2", tmp, spec)
    assert calls["perf"] == first + 1                           # only the new candidate was measured


def test_cache_is_invalidated_when_the_baseline_recipe_changes(env):
    make, calls, tmp, spec = env
    evaluate(make(), SMALL, "small", tmp, spec)
    (tmp / "v0.yaml").write_text(V0.replace("Q4_0", "Q5_0", 1))
    evaluate(make(), SMALL, "small", tmp, spec)
    assert calls["perf"] == 4


def test_unstable_candidate_skips_the_expensive_guards(env):
    make, calls, tmp, spec = env
    r = make(perf_fn=lambda p: perf_for(p, stable="baseline" in p.name))      # only the candidate is unstable
    b = evaluate(r, SMALL, "small", tmp, spec)
    assert b.guards_skipped and b.guards == [] and not b.perf.stable


def test_long_context_regression_shows_up_in_the_guards(env):
    make, calls, tmp, spec = env
    r = make(generative_factory=lambda p, n: Leaky(10 ** 6 if "baseline" in p.name else 9000))
    b = evaluate(r, SMALL, "small", tmp, spec)
    by = {g.name: g for g in b.guards}
    assert by["needle8192"].passed and not by["needle16384"].passed


def test_build_failure_is_reported_and_leaves_nothing_behind(env):
    make, calls, tmp, spec = env
    r = make()
    evaluate(r, SMALL, "warm", tmp, spec)                        # warm the baseline cache
    broken = HEAD.format(n="broken") + "default: Q4_K\nrules:\n  - {match: 'lm_head', format: Q8_0}\n"
    with pytest.raises(EvalFailure, match="build") as e:
        evaluate(r, broken, "broken", tmp, spec)                 # runtime quantizer, no llama-quantize
    assert "llama-quantize not found" in str(e.value)
    cid = parse_manifest(broken, spec).candidate_id(spec)
    assert not (tmp / "work" / f"cand-{cid}.gguf").exists()


def test_holdout_verdict_uses_the_rotating_shard(env):
    make, calls, tmp, spec = env
    r = make()
    b = evaluate(r, SMALL, "small", tmp, spec)
    v = r.holdout(b)
    assert v.shard in {"shard-0", "shard-1"} and v.public_gain < 0       # candidate is noisier than the baseline
    assert v.passed == (v.holdout_gain >= v.required)
