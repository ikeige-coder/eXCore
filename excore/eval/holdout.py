"""Private, rotating holdout: guards against recipes that overfit the public corpus.

The holdout text never lives in the repository. An operator points
``EXCORE_HOLDOUT_DIR`` at a directory of shards, each ``shard-<n>/reference.npz`` built
with ``accuracy.build_reference`` from private text. The active shard rotates with an
epoch (ISO week by default), so a miner cannot tune against it.

A candidate's improvement is measured against the incumbent on the same corpus::

    gain = (incumbent RP-KL - candidate RP-KL) / incumbent RP-KL

and passes when the holdout gain is at least what the public gain promised, after the
configured transfer share and tolerance::

    holdout_gain >= min(transfer * public_gain, public_gain) - tolerance

A real fidelity gain must therefore carry over (transfer share, default 0.5), and a
recipe that merely stays near the incumbent must not get worse on unseen text.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .accuracy import AccuracyError, Reference

ENV_DIR = "EXCORE_HOLDOUT_DIR"
SHARD_RE = re.compile(r"^shard-(\d+)$")


@dataclass(frozen=True)
class HoldoutVerdict:
    passed: bool
    public_gain: float
    holdout_gain: float
    required: float
    shard: str | None = None


def gain(incumbent_kl: float, candidate_kl: float) -> float:
    return (incumbent_kl - candidate_kl) / incumbent_kl if incumbent_kl > 0 else 0.0


def holdout_verdict(
    public_incumbent: float, public_candidate: float,
    holdout_incumbent: float, holdout_candidate: float,
    *, min_transfer: float, tolerance: float, shard: str | None = None,
) -> HoldoutVerdict:
    pub = gain(public_incumbent, public_candidate)
    hold = gain(holdout_incumbent, holdout_candidate)
    required = min(min_transfer * pub, pub) - tolerance
    return HoldoutVerdict(hold >= required, pub, hold, required, shard)


def epoch_now(today: _dt.date | None = None) -> int:
    """Rotation epoch: the ISO (year, week) folded into one integer."""
    y, w, _ = (today or _dt.date.today()).isocalendar()
    return y * 100 + w


def list_shards(root: str | Path) -> list[Path]:
    root = Path(root)
    shards = [p for p in root.iterdir() if p.is_dir() and SHARD_RE.match(p.name) and (p / "reference.npz").exists()]
    return sorted(shards, key=lambda p: int(SHARD_RE.match(p.name).group(1)))


def select_shard(root: str | Path | None = None, epoch: int | None = None) -> Path:
    root = root or os.environ.get(ENV_DIR)
    if not root:
        raise AccuracyError(f"holdout directory not configured (set {ENV_DIR})")
    shards = list_shards(root)
    if not shards:
        raise AccuracyError(f"{root}: no holdout shards found")
    return shards[(epoch_now() if epoch is None else epoch) % len(shards)]


def load_holdout(root: str | Path | None = None, epoch: int | None = None) -> tuple[Reference, str]:
    shard = select_shard(root, epoch)
    return Reference.load(shard / "reference.npz"), shard.name
