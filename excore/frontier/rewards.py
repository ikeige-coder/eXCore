"""Reward tiers: maps a frontier gain to a bracket and its multiplier.

``gain`` is the relative growth of the frontier's dominated volume that a candidate
adds (see ``core_frontier``). The brackets and multipliers live in ``configs/hpc_cpu.yaml``
so they can be audited and changed in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


class RewardConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Tier:
    name: str            # e.g. "core:Ultra"
    min_gain: float
    multiplier: float


@dataclass(frozen=True)
class RewardDecision:
    tier: str | None
    multiplier: float
    gain: float
    reason: str


def load_tiers(cfg: Mapping) -> tuple[Tier, ...]:
    """Tiers from the config, highest bracket first. Brackets must be strictly ordered."""
    raw = cfg["tiers"]
    tiers = []
    for name, t in raw.items():
        if not name.startswith("core:"):
            raise RewardConfigError(f"tier name {name!r} must start with 'core:'")
        try:
            tiers.append(Tier(name, float(t["min_gain"]), float(t["multiplier"])))
        except (KeyError, TypeError, ValueError):
            raise RewardConfigError(f"tier {name!r} needs numeric min_gain and multiplier") from None
    tiers.sort(key=lambda t: -t.min_gain)
    for t in tiers:
        if t.min_gain <= 0 or t.multiplier <= 0:
            raise RewardConfigError(f"tier {t.name}: min_gain and multiplier must be positive")
    for hi, lo in zip(tiers, tiers[1:]):
        if hi.min_gain == lo.min_gain:
            raise RewardConfigError(f"tiers {hi.name} and {lo.name} share a threshold")
        if hi.multiplier <= lo.multiplier:
            raise RewardConfigError(f"tier {hi.name} must pay more than {lo.name}")
    if not tiers:
        raise RewardConfigError("no tiers configured")
    return tuple(tiers)


def assign_tier(gain: float, tiers: Sequence[Tier]) -> Tier | None:
    for t in tiers:                       # highest first
        if gain >= t.min_gain:
            return t
    return None


def decide_reward(gain: float, tiers: Sequence[Tier]) -> RewardDecision:
    t = assign_tier(gain, tiers)
    if t is None:
        floor = min(x.min_gain for x in tiers)
        return RewardDecision(None, 0.0, gain, f"gain {gain:.5f} is below the lowest bracket ({floor:g})")
    return RewardDecision(t.name, t.multiplier, gain, f"gain {gain:.5f} reaches {t.name}")
