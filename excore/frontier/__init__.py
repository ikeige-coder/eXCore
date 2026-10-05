"""Frontier: gates, the 4-D Core-Optimal frontier and reward tiers, joined into one verdict."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .core_frontier import Assessment, Entry, Point, Space, Spectrum, make_point
from .gates import GateReport, GuardResult, evaluate_gates
from .rewards import RewardDecision, decide_reward, load_tiers

STAGES = ("gates", "frontier", "reward", "accepted")


@dataclass(frozen=True)
class Verdict:
    accepted: bool
    stage: str                       # where it stopped, or "accepted"
    reasons: tuple[str, ...]
    gates: GateReport
    point: Point | None = None
    assessment: Assessment | None = None
    reward: RewardDecision | None = None

    @property
    def tier(self) -> str | None:
        return self.reward.tier if self.reward else None

    def entry(self, ref: str | None = None, merged_at: str | None = None, via: str | None = None) -> Entry:
        """The spectrum entry to record if this verdict is accepted."""
        if not self.accepted or self.point is None:
            raise ValueError("only an accepted verdict has an entry")
        gain = self.reward.gain if self.reward else 0.0
        return Entry(self.point, gain, self.tier, ref, merged_at, via)


def judge(
    *, candidate_id: str, name: str, accuracy, perf, baseline_perf, guards: Sequence[GuardResult],
    spectrum: Spectrum, cfg: Mapping, host: Mapping | None = None, bootstrap: bool = False,
) -> Verdict:
    """Gates, then frontier placement, then reward tier. Never mutates the spectrum."""
    gates = evaluate_gates(perf=perf, accuracy=accuracy, guards=guards, cfg=cfg)
    if not gates.passed:
        return Verdict(False, "gates", tuple(str(f) for f in gates.failures), gates)

    point = make_point(candidate_id, name, accuracy, perf, baseline_perf, spectrum.space, host)

    if not spectrum.entries:
        if bootstrap:
            return Verdict(True, "accepted", ("bootstrap entry: seeds the frontier, no reward",), gates, point)
        return Verdict(False, "frontier", ("the spectrum is empty; seed it with the baseline (bootstrap=True)",),
                       gates, point)

    assessment = spectrum.assess(point)
    if assessment.duplicate:
        return Verdict(False, "frontier", (f"{candidate_id} is already in the spectrum",), gates, point, assessment)
    if assessment.dominated_by:
        return Verdict(False, "frontier",
                       (f"dominated by {', '.join(assessment.dominated_by)}",), gates, point, assessment)
    if not assessment.core_optimal:
        return Verdict(False, "frontier", ("adds no dominated volume (an objective is at or beyond its bound)",),
                       gates, point, assessment)

    reward = decide_reward(assessment.gain, load_tiers(cfg))
    if reward.tier is None:
        return Verdict(False, "reward", (reward.reason,), gates, point, assessment, reward)
    return Verdict(True, "accepted", (reward.reason,), gates, point, assessment, reward)


__all__ = [
    "Verdict", "judge", "Spectrum", "Space", "Point", "Entry", "Assessment", "make_point",
    "GateReport", "GuardResult", "evaluate_gates", "RewardDecision", "decide_reward", "load_tiers",
]
