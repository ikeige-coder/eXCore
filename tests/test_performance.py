import stat
import sys
from pathlib import Path

import pytest

from synthetic import SyntheticBackend, sample_texts

from excore.eval import evaluate_model
from excore.eval.accuracy import build_reference, make_windows
from excore.eval.performance import (
    PerfError,
    bench_command,
    check_free_ram,
    find_bench,
    host_fingerprint,
    measure,
    parse_bench_json,
)


@pytest.fixture()
def bench(tmp_path):
    tpl = (Path(__file__).parent / "fake_llama_bench.py.tpl").read_text()
    p = tmp_path / "llama-bench"
    p.write_text(tpl.format(python=sys.executable))
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return str(p)


@pytest.fixture()
def pcfg(tiny_cfg):
    c = dict(tiny_cfg)
    c["limits"] = {**tiny_cfg["limits"], "min_free_ram_gib": 0.0, "eval_timeout_s": 60}
    return c


def test_parse_bench_json():
    ok = '[{"n_prompt": 4096, "n_gen": 0, "avg_ts": 55.5}, {"n_prompt": 0, "n_gen": 128, "avg_ts": 7.25}]'
    assert parse_bench_json("noise\n" + ok + "\n") == (55.5, 7.25)
    for bad in ("", "[1, 2", '[{"n_prompt": 1, "n_gen": 0, "avg_ts": 1.0}]'):
        with pytest.raises(PerfError):
            parse_bench_json(bad)


def test_command_is_cpu_only_and_pinned():
    cmd = bench_command("/b/llama-bench", "m.gguf", threads=8, prompt_tokens=4096, gen_tokens=128,
                        batch=512, ubatch=256)
    assert cmd[cmd.index("-ngl") + 1] == "0" and cmd[cmd.index("-t") + 1] == "8"
    assert cmd[cmd.index("-p") + 1] == "4096" and cmd[-2:] == ["-o", "json"]


def test_measure_reports_conservative_numbers(bench, pcfg, monkeypatch):
    monkeypatch.setenv("FAKE_ALLOC_MB", "300")
    r = measure("m.gguf", pcfg, bench_binary=bench)
    assert r.prefill_tps == 100 and r.gen_tps == 10 and r.stable and len(r.runs) == 2
    assert 0.28 < r.peak_rss_gib < 1.0
    assert r.within_ram_limit(28.0) and not r.within_ram_limit(0.1)


def test_unstable_runs_are_flagged(bench, pcfg, monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_COUNTER", str(tmp_path / "count"))
    r = measure("m.gguf", pcfg, bench_binary=bench)
    assert not r.stable and r.spread_prefill > 0.3
    assert r.prefill_tps == 100          # the slower run is what gets reported


def test_failures_and_preflight(bench, pcfg, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL", "1")
    with pytest.raises(PerfError, match="exploded"):
        measure("m.gguf", pcfg, bench_binary=bench)
    monkeypatch.delenv("FAKE_FAIL")
    low = dict(pcfg)
    low["limits"] = {**pcfg["limits"], "min_free_ram_gib": 10 ** 6}
    with pytest.raises(PerfError, match="RAM available"):
        measure("m.gguf", low, bench_binary=bench)
    with pytest.raises(PerfError):
        check_free_ram(10 ** 6)


def test_bench_lookup(bench, monkeypatch, tmp_path):
    monkeypatch.delenv("EXCORE_LLAMA_BENCH", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "none"))
    with pytest.raises(PerfError, match="setup_bench.sh"):
        find_bench(None)
    assert str(find_bench(bench)) == bench


def test_host_fingerprint():
    info = host_fingerprint()
    assert info["machine"] and info["system"]


def test_evaluate_model_end_to_end(bench, pcfg):
    b = SyntheticBackend()
    ref = build_reference(b, make_windows(b, sample_texts(), 64, 5000), k=16, corpus_digest="d", model_id="m")
    res = evaluate_model("m.gguf", ref, lambda: SyntheticBackend(noise=0.3), pcfg, bench_binary=bench)
    assert res.accuracy.rp_kl > 0 and res.perf.gen_tps == 10
    assert set(res.to_dict()) == {"accuracy", "perf"}
