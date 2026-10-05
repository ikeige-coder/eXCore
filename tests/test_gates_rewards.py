import re

import numpy as np
import pytest

from synthetic import Leaky, SyntheticBackend

from excore.eval.accuracy import AccuracyResult
from excore.eval.performance import PerfResult, PerfRun
from excore.frontier.gates import (
    GuardResult, evaluate_gates, needle_cases, needle_prompt, run_needle_guard,
)
from excore.frontier.rewards import RewardConfigError, assign_tier, decide_reward, load_tiers


def acc(kl=0.02, top1=0.9, se=0.001):
    return AccuracyResult(kl, se, top1, 100, np.zeros(100, np.float32), np.full(10, kl), "r")


def perf(gen=10.0, pre=100.0, ram=16.0, stable=True):
    run = PerfRun(pre, gen, ram, 1.0)
    return PerfResult(pre, gen, ram, (run, run), stable, 0.0 if stable else 0.4, 0.0, 4096, 128, 8)


def test_clean_candidate_passes(cfg):
    r = evaluate_gates(perf=perf(), accuracy=acc(), guards=[], cfg=cfg)
    assert r.passed and r.summary() == "all gates passed"


@pytest.mark.parametrize(
    "p, a, code",
    [
        (perf(stable=False), acc(), "G_UNSTABLE"),
        (perf(ram=30.0), acc(), "G_RAM"),
        (perf(), acc(kl=0.9), "G_DRIFT"),
        (perf(), acc(top1=0.2), "G_TOP1"),
        (perf(), acc(kl=float("nan")), "G_ACC_INVALID"),
    ],
)
def test_each_gate_fires(cfg, p, a, code):
    assert code in evaluate_gates(perf=p, accuracy=a, guards=[], cfg=cfg).codes()


def test_all_failures_are_reported_together(cfg):
    r = evaluate_gates(perf=perf(ram=30, stable=False), accuracy=acc(kl=0.9), guards=[], cfg=cfg)
    assert r.codes() == {"G_UNSTABLE", "G_RAM", "G_DRIFT"}


def test_guard_regression_rules():
    assert GuardResult("t", 1.0, 1.0, 0).passed
    assert not GuardResult("t", 1.0, 0.9, 0).passed
    assert GuardResult("t", 0.8, 0.6, 25).passed and not GuardResult("t", 0.8, 0.59, 25).passed
    assert GuardResult("t", 0.0, 0.0, 0).passed


def test_failed_guard_blocks(cfg):
    g = GuardResult("needle16384", 1.0, 0.67, 0)
    assert "G_GUARD_needle16384" in evaluate_gates(perf=perf(), accuracy=acc(), guards=[g], cfg=cfg).codes()


# -- needle guard ---------------------------------------------------------------------


def filler():
    return list(b"The quick brown fox jumps over the lazy dog while the committee reviews quarterly figures. ")


def test_needle_cases_are_deterministic_and_rotate():
    a = needle_cases([8192, 16384], [0.1, 0.5, 0.9], seed=1)
    assert a == needle_cases([8192, 16384], [0.1, 0.5, 0.9], seed=1) and len(a) == 6
    assert [c.answer for c in a] != [c.answer for c in needle_cases([8192, 16384], [0.1, 0.5, 0.9], seed=2)]


def test_prompt_has_exact_length_and_needle_at_depth():
    b = Leaky(10 ** 6)
    for ctx in (8192, 16384):
        for depth in (0.1, 0.5, 0.9):
            case = needle_cases([ctx], [depth], 3)[0]
            p = needle_prompt(case, b, filler())
            assert len(p) == ctx - 24
            text = bytes(p[1:]).decode("latin-1")
            pos = text.index(case.answer)
            assert abs(pos / len(text) - depth) < 0.05
    with pytest.raises(ValueError, match="too short"):
        needle_prompt(needle_cases([100], [0.5], 3)[0], b, filler())


def test_long_context_failure_is_detected_and_blocks():
    ref_score, _ = run_needle_guard(Leaky(10 ** 6), [8192, 16384], [0.1, 0.5, 0.9], filler(), seed=5)
    short_score, outcomes = run_needle_guard(Leaky(9000), [8192, 16384], [0.1, 0.5, 0.9], filler(), seed=5)
    assert ref_score == 1.0 and 0.5 <= short_score < 1.0
    assert all(ok for case, ok in outcomes if case.ctx == 8192)
    assert not all(ok for case, ok in outcomes if case.ctx == 16384)
    assert not GuardResult("needle", ref_score, short_score, 0).passed


# -- rewards ----------------------------------------------------------------------------


def test_tiers_load_sorted_and_assign(cfg):
    tiers = load_tiers(cfg)
    assert [t.name for t in tiers] == ["core:Ultra", "core:XL", "core:L", "core:M", "core:S"]
    assert assign_tier(0.9, tiers).name == "core:Ultra"
    assert assign_tier(0.25, tiers).name == "core:Ultra"          # boundary is inclusive
    assert assign_tier(0.2499, tiers).name == "core:XL"
    assert assign_tier(0.005, tiers).name == "core:S"
    assert assign_tier(0.00499, tiers) is None


def test_decide_reward(cfg):
    tiers = load_tiers(cfg)
    d = decide_reward(0.05, tiers)
    assert (d.tier, d.multiplier) == ("core:L", 1.5)
    n = decide_reward(0.0001, tiers)
    assert n.tier is None and n.multiplier == 0.0 and "below" in n.reason


@pytest.mark.parametrize(
    "tiers, msg",
    [
        ({"Ultra": {"min_gain": 0.1, "multiplier": 2}}, "must start with"),
        ({"core:A": {"min_gain": 0.1}}, "numeric"),
        ({"core:A": {"min_gain": 0.1, "multiplier": 1}, "core:B": {"min_gain": 0.1, "multiplier": 2}}, "share a threshold"),
        ({"core:A": {"min_gain": 0.2, "multiplier": 1}, "core:B": {"min_gain": 0.1, "multiplier": 2}}, "must pay more"),
        ({"core:A": {"min_gain": 0, "multiplier": 1}}, "positive"),
        ({}, "no tiers"),
    ],
)
def test_bad_tier_configs(tiers, msg):
    with pytest.raises(RewardConfigError, match=msg):
        load_tiers({"tiers": tiers})
