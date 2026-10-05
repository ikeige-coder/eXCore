import numpy as np
import pytest

from excore.eval.accuracy import AccuracyResult
from excore.eval.performance import PerfResult, PerfRun
from excore.frontier import Entry, GuardResult, Point, Space, Spectrum, judge


def acc(kl=0.02, top1=0.9, se=0.0005):
    return AccuracyResult(kl, se, top1, 100, np.zeros(100, np.float32), np.full(10, kl), "r")


def perf(gen=10.0, pre=100.0, ram=16.0, stable=True):
    run = PerfRun(pre, gen, ram, 1.0)
    return PerfResult(pre, gen, ram, (run, run), stable, 0.0 if stable else 0.4, 0.0, 4096, 128, 8)


BASE = perf(gen=10.0, pre=100.0, ram=20.0)


@pytest.fixture()
def spectrum(cfg):
    sp = Spectrum("CPU-35B", Space.from_config(cfg))
    sp.add(Entry(Point("base", "v0", 0.02, 1.0, 1.0, 20.0), 0.0, None))
    return sp


def run(spectrum, cfg, cid="cand", a=None, p=None, guards=(), **kw):
    return judge(candidate_id=cid, name=cid, accuracy=a or acc(), perf=p or perf(), baseline_perf=BASE,
                 guards=list(guards), spectrum=spectrum, cfg=cfg, **kw)


def test_good_candidate_is_accepted_with_a_tier(spectrum, cfg):
    v = run(spectrum, cfg, p=perf(gen=12, pre=120, ram=14))
    assert v.accepted and v.stage == "accepted" and v.tier in {"core:Ultra", "core:XL", "core:L", "core:M", "core:S"}
    e = v.entry(ref="PR#1")
    assert e.point.candidate_id == "cand" and e.gain == v.reward.gain and e.ref == "PR#1"
    assert v.point.gen_rel == pytest.approx(1.2)
    assert v.point.rp_kl == 0.02 and v.point.rp_kl_stderr == 0.0005


def test_gate_failure_stops_before_the_frontier(spectrum, cfg):
    v = run(spectrum, cfg, p=perf(ram=30.0))
    assert not v.accepted and v.stage == "gates" and v.assessment is None and "G_RAM" in v.reasons[0]


def test_failed_guard_stops_it(spectrum, cfg):
    v = run(spectrum, cfg, p=perf(ram=14), guards=[GuardResult("needle16384", 1.0, 0.67, 0)])
    assert not v.accepted and v.stage == "gates"


def test_dominated_candidate_is_rejected(spectrum, cfg):
    v = run(spectrum, cfg, a=acc(kl=0.05), p=perf(gen=9, pre=90, ram=22))
    assert not v.accepted and v.stage == "frontier" and "dominated by base" in v.reasons[0]


def test_noise_level_gain_earns_nothing(spectrum, cfg):
    # a sliver of extra accuracy with no measurement noise: on the frontier, but below the lowest tier
    v = run(spectrum, cfg, a=acc(kl=0.0199, se=0.0), p=perf(ram=19.99))
    assert not v.accepted and v.stage == "reward" and v.tier is None and v.assessment.core_optimal
    # the same "gain" with realistic accuracy noise is simply dominated by the incumbent
    v = run(spectrum, cfg, a=acc(kl=0.0199, se=0.0005), p=perf(ram=19.99))
    assert not v.accepted and v.stage == "frontier" and "dominated by base" in v.reasons[0]


def test_noise_margin_penalises_an_uncertain_accuracy(spectrum, cfg):
    sure = run(spectrum, cfg, a=acc(kl=0.02, se=0.0), p=perf(ram=14))
    noisy = run(spectrum, cfg, a=acc(kl=0.02, se=0.01), p=perf(ram=14))
    assert noisy.reward.gain < sure.reward.gain


def test_duplicate_and_empty_spectrum(spectrum, cfg):
    v = run(spectrum, cfg, cid="base", p=perf(ram=14))
    assert not v.accepted and "already in the spectrum" in v.reasons[0]
    empty = Spectrum("CPU-35B", spectrum.space)
    assert not run(empty, cfg).accepted
    boot = run(empty, cfg, bootstrap=True)
    assert boot.accepted and boot.tier is None and boot.entry().gain == 0.0
    boot2 = run(empty, cfg, p=perf(ram=30), bootstrap=True)
    assert not boot2.accepted                                       # gates still apply when bootstrapping


def test_entry_requires_acceptance(spectrum, cfg):
    with pytest.raises(ValueError):
        run(spectrum, cfg, p=perf(ram=30)).entry()


def test_accepting_changes_the_next_assessment(spectrum, cfg):
    first = run(spectrum, cfg, cid="one", p=perf(gen=12, pre=120, ram=14))
    spectrum.add(first.entry())
    again = run(spectrum, cfg, cid="two", p=perf(gen=12, pre=120, ram=14))      # same performance, new id
    assert not again.accepted and "dominated by one" in again.reasons[0]
