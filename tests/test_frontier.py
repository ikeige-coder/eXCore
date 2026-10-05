import itertools
import math

import numpy as np
import pytest

from excore.frontier.core_frontier import (
    Entry, FrontierError, Point, Space, Spectrum, dominates, hypervolume, pareto_indices,
)


def brute_hv(points):
    """Inclusion-exclusion over boxes [0, p]: the textbook definition, for cross-checking."""
    total = 0.0
    for r in range(1, len(points) + 1):
        for subset in itertools.combinations(points, r):
            vol = math.prod(min(p[d] for p in subset) for d in range(len(points[0])))
            total += (-1) ** (r + 1) * vol
    return total


@pytest.mark.parametrize("d", [1, 2, 3, 4])
def test_hypervolume_matches_inclusion_exclusion(d):
    rng = np.random.default_rng(d)
    for _ in range(25):
        pts = [tuple(rng.uniform(0.05, 1.0, d)) for _ in range(rng.integers(1, 7))]
        assert hypervolume(pts) == pytest.approx(brute_hv(pts), rel=1e-9)


def test_hypervolume_basics():
    assert hypervolume([]) == 0.0
    assert hypervolume([(0.5, 0.5, 0.5, 0.5)]) == pytest.approx(0.0625)
    assert hypervolume([(1, 1, 1, 1)]) == 1.0
    assert hypervolume([(0.5, 0.0, 0.5, 0.5)]) == 0.0          # a zero axis dominates nothing
    a, b = (0.9, 0.2, 0.5, 0.5), (0.9, 0.2, 0.4, 0.4)           # b is dominated by a
    assert hypervolume([a, b]) == pytest.approx(hypervolume([a]))


def test_hypervolume_handles_a_full_size_frontier_quickly():
    rng = np.random.default_rng(0)
    pts = [tuple(rng.uniform(0.1, 1, 4)) for _ in range(120)]
    v = hypervolume(pts)
    assert 0 < v <= 1


def test_dominance_and_pareto():
    assert dominates((1, 1, 1, 1), (0.5, 1, 1, 1))
    assert not dominates((1, 1, 1, 1), (1, 1, 1, 1))
    assert not dominates((1, 0, 1, 1), (0, 1, 1, 1))
    pts = [(1, 0, 0, 0), (0, 1, 0, 0), (0.5, 0.5, 0, 0), (0.4, 0.4, 0, 0), (1, 0, 0, 0)]
    assert pareto_indices(pts) == [0, 1, 2]                     # dominated and duplicate points dropped


@pytest.fixture()
def space(cfg):
    return Space.from_config(cfg)


def P(cid, kl, gen, pre, ram, name=None):
    return Point(cid, name or cid, kl, gen, pre, ram)


def test_scores_are_monotone_and_clipped(space):
    s = space.scores(P("a", 0.01, 1.0, 1.0, 16.0))
    assert all(0 < x < 1 for x in s)
    better = space.scores(P("b", 0.005, 1.5, 1.5, 12.0))
    assert all(b > a for a, b in zip(s, better))
    top = space.scores(P("c", 1e-9, 99, 99, 0.1))
    assert top == (1.0, 1.0, 1.0, 1.0)
    bottom = space.scores(P("d", 99, 0.001, 0.001, 99))
    assert bottom == (0.0, 0.0, 0.0, 0.0)
    assert space.scores(P("e", 0.5, 1.0, 1.0, 16.0))[0] == 0.0  # at the worst bound


def test_bad_bounds_rejected(cfg):
    bad = {**cfg, "frontier": {"z": 2, "bounds": {**cfg["frontier"]["bounds"], "ram_gib": {"best": 30, "worst": 10}}}}
    with pytest.raises(FrontierError):
        Space.from_config(bad)


@pytest.fixture()
def spectrum(space):
    sp = Spectrum("CPU-35B", space)
    sp.add(Entry(P("base", 0.02, 1.0, 1.0, 20.0), 0.0, None))
    return sp


def test_assess_rejects_empty_and_duplicates(space, spectrum):
    with pytest.raises(FrontierError, match="empty"):
        Spectrum("CPU-35B", space).assess(P("x", 0.01, 1, 1, 10))
    a = spectrum.assess(P("base", 0.001, 9, 9, 9))
    assert a.duplicate and not a.core_optimal


def test_dominated_candidate_is_not_core_optimal(spectrum):
    a = spectrum.assess(P("worse", 0.03, 0.9, 0.9, 22.0))
    assert not a.core_optimal and a.gain == 0 and a.dominated_by == ("base",)


def test_tradeoff_candidate_extends_the_frontier(spectrum):
    smaller = spectrum.assess(P("small", 0.04, 1.0, 1.0, 12.0))      # less accurate, much smaller
    assert smaller.core_optimal and smaller.gain > 0 and not smaller.dominated_by
    assert spectrum.assess(P("faster", 0.02, 1.4, 1.2, 20.0)).core_optimal


def test_strictly_better_candidate_pushes_base_off_the_frontier(spectrum):
    a = spectrum.assess(P("best", 0.01, 1.3, 1.3, 14.0))
    assert a.core_optimal and a.dominates == ("base",) and a.gain > 0.2


def test_bigger_improvements_earn_bigger_gains(spectrum):
    g = [spectrum.assess(P(f"c{r}", 0.02, 1.0, 1.0, r)).gain for r in (19.0, 17.0, 13.0)]
    assert 0 < g[0] < g[1] < g[2]


def test_noise_sized_improvement_earns_less_than_the_lowest_tier(spectrum, cfg):
    # 0.5% less drift and 10 MB less RAM, inside the measurement noise margins
    assert spectrum.assess(P("noise", 0.0199, 1.0, 1.0, 19.99)).gain < cfg["tiers"]["core:S"]["min_gain"]


def test_a_candidate_must_beat_the_incumbent_by_more_than_the_margin(spectrum):
    same = spectrum.assess(P("same", 0.02, 1.0, 1.0, 20.0))
    assert not same.core_optimal and same.dominated_by == ("base",)
    barely = spectrum.assess(P("barely", 0.02, 1.03, 1.03, 20.0))      # +3% speed < 5% margin
    assert not barely.core_optimal
    clear = spectrum.assess(P("clear", 0.02, 1.20, 1.20, 20.0))        # +20% speed
    assert clear.core_optimal and clear.gain > 0.1


def test_accuracy_noise_widens_the_margin(space):
    sure = space.scores(Point("a", "a", 0.02, 1, 1, 16, rp_kl_stderr=0.0), conservative=True)
    noisy = space.scores(Point("b", "b", 0.02, 1, 1, 16, rp_kl_stderr=0.01), conservative=True)
    assert noisy[0] < sure[0] and noisy[1:] == sure[1:]


def test_archive_keeps_history_but_front_moves(spectrum):
    best = P("best", 0.01, 1.3, 1.3, 14.0)
    spectrum.add(Entry(best, 0.3, "core:Ultra"))
    assert [e.point.candidate_id for e in spectrum.entries] == ["base", "best"]
    assert [e.point.candidate_id for e in spectrum.active()] == ["best"]
    with pytest.raises(FrontierError, match="already"):
        spectrum.add(Entry(best, 0.3, None))


def test_spectrum_roundtrip_and_validation(spectrum, space, tmp_path):
    spectrum.add(Entry(P("x", 0.04, 1.0, 1.0, 12.0), 0.1, "core:XL", ref="PR#7", merged_at="2026-10-01"))
    p = tmp_path / "s.json"
    spectrum.save(p)
    back = Spectrum.load(p, space, track="CPU-35B")
    assert [e.to_dict() for e in back.entries] == [e.to_dict() for e in spectrum.entries]
    assert back.volume() == pytest.approx(spectrum.volume())
    with pytest.raises(FrontierError, match="track"):
        Spectrum.load(p, space, track="GPU-01")
    p.write_text("{}")
    with pytest.raises(FrontierError, match="not an"):
        Spectrum.load(p, space, track="CPU-35B")
    with pytest.raises(FrontierError, match="unreadable"):
        Spectrum.load(tmp_path / "nope.json", space, track="CPU-35B")
