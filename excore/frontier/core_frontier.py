"""Core-Optimal scoring: a 4-D Pareto frontier over accuracy, decode, prefill and RAM.

Every objective is mapped to a 0..1 *score* (1 = best) inside the bounds in the track
config, so the four axes are comparable:

* accuracy: log scale between ``rp_kl.best`` and ``rp_kl.worst`` (lower drift is better),
* decode and prefill speed: log scale of tokens/s *relative to a baseline measured on
  the same host in the same session*, so different machines stay comparable,
* RAM: linear between ``ram_gib.best`` and ``ram_gib.worst``.

Noise cannot buy a place: a *candidate* is scored conservatively (accuracy at
``rp_kl + z * stderr``, speeds shrunk by ``speed_margin``, RAM raised by ``ram_margin_gib``)
while incumbents are scored at their measured values, so a candidate has to beat an
incumbent by more than the measurement noise.

A candidate is **Core-Optimal** when it is not dominated by the current frontier and adds
dominated volume. Its gain is the relative growth of the frontier's hypervolume:
``(HV(F + c) - HV(F)) / HV(F)``. Results beyond a bound earn no extra credit, and a
candidate at or beyond the worst bound on any axis dominates nothing.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

SPECTRUM_SCHEMA = "excore/spectrum@1"
AXES = ("accuracy", "decode", "prefill", "ram")
Scores = tuple[float, float, float, float]


class FrontierError(ValueError):
    pass


# -- scoring --------------------------------------------------------------------


def _clip01(x: float) -> float:
    return 0.0 if x != x else min(1.0, max(0.0, x))


@dataclass(frozen=True)
class Space:
    kl_best: float
    kl_worst: float
    gen_lo: float
    gen_hi: float
    prefill_lo: float
    prefill_hi: float
    ram_best: float
    ram_worst: float
    z: float = 2.0
    speed_margin: float = 0.05
    ram_margin: float = 0.25

    @classmethod
    def from_config(cls, cfg: Mapping) -> "Space":
        f = cfg["frontier"]
        b = f["bounds"]
        space = cls(
            float(b["rp_kl"]["best"]), float(b["rp_kl"]["worst"]),
            float(b["gen"]["lo"]), float(b["gen"]["hi"]),
            float(b["prefill"]["lo"]), float(b["prefill"]["hi"]),
            float(b["ram_gib"]["best"]), float(b["ram_gib"]["worst"]),
            float(f.get("z", 2.0)), float(f.get("speed_margin", 0.05)), float(f.get("ram_margin_gib", 0.25)),
        )
        if not (0 < space.kl_best < space.kl_worst and 0 < space.gen_lo < space.gen_hi
                and 0 < space.prefill_lo < space.prefill_hi and 0 < space.ram_best < space.ram_worst):
            raise FrontierError("frontier bounds must be positive and ordered")
        return space

    def scores(self, p: "Point", *, conservative: bool = False) -> Scores:
        kl, gen, pre_r, ram_gib = p.rp_kl, p.gen_rel, p.prefill_rel, p.ram_gib
        if conservative:
            kl += self.z * p.rp_kl_stderr
            gen *= 1.0 - self.speed_margin
            pre_r *= 1.0 - self.speed_margin
            ram_gib += self.ram_margin
        acc = 1.0 - math.log(max(kl, 1e-300) / self.kl_best) / math.log(self.kl_worst / self.kl_best)
        dec = math.log(max(gen, 1e-300) / self.gen_lo) / math.log(self.gen_hi / self.gen_lo)
        pre = math.log(max(pre_r, 1e-300) / self.prefill_lo) / math.log(self.prefill_hi / self.prefill_lo)
        ram = (self.ram_worst - ram_gib) / (self.ram_worst - self.ram_best)
        return (_clip01(acc), _clip01(dec), _clip01(pre), _clip01(ram))


# -- points and the archive ----------------------------------------------------------


@dataclass(frozen=True)
class Point:
    candidate_id: str
    name: str
    rp_kl: float              # measured mean drift
    gen_rel: float
    prefill_rel: float
    ram_gib: float
    rp_kl_stderr: float = 0.0
    gen_tps: float = float("nan")
    prefill_tps: float = float("nan")
    top1: float = float("nan")
    host: Mapping = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["host"] = dict(self.host)
        return {k: (None if isinstance(v, float) and v != v else v) for k, v in d.items()}

    @classmethod
    def from_dict(cls, d: Mapping) -> "Point":
        nan = float("nan")
        return cls(
            candidate_id=str(d["candidate_id"]), name=str(d["name"]), rp_kl=float(d["rp_kl"]),
            gen_rel=float(d["gen_rel"]), prefill_rel=float(d["prefill_rel"]), ram_gib=float(d["ram_gib"]),
            rp_kl_stderr=_f(d.get("rp_kl_stderr"), 0.0), gen_tps=_f(d.get("gen_tps"), nan),
            prefill_tps=_f(d.get("prefill_tps"), nan), top1=_f(d.get("top1"), nan),
            host=dict(d.get("host") or {}),
        )


def _f(v, default):
    return default if v is None else float(v)


def make_point(candidate_id: str, name: str, accuracy, perf, baseline_perf, space: Space, host: Mapping | None = None) -> Point:
    """Build a Point from an AccuracyResult and PerfResults (candidate and same-host baseline)."""
    se = accuracy.stderr if accuracy.stderr == accuracy.stderr else 0.0
    return Point(
        candidate_id=candidate_id, name=name,
        rp_kl=accuracy.rp_kl, rp_kl_stderr=se,
        gen_rel=perf.gen_tps / baseline_perf.gen_tps,
        prefill_rel=perf.prefill_tps / baseline_perf.prefill_tps,
        ram_gib=perf.peak_rss_gib, gen_tps=perf.gen_tps, prefill_tps=perf.prefill_tps,
        top1=accuracy.top1_agreement, host=host or {},
    )


@dataclass(frozen=True)
class Entry:
    point: Point
    gain: float
    tier: str | None
    ref: str | None = None         # PR number / commit, for provenance
    merged_at: str | None = None
    via: str | None = None         # bot-merge | owner-merge | either with "+override" | seed

    def to_dict(self) -> dict:
        return {"point": self.point.to_dict(), "gain": self.gain, "tier": self.tier,
                "ref": self.ref, "merged_at": self.merged_at, "via": self.via}

    @classmethod
    def from_dict(cls, d: Mapping) -> "Entry":
        return cls(Point.from_dict(d["point"]), float(d["gain"]), d.get("tier"), d.get("ref"),
                   d.get("merged_at"), d.get("via"))


# -- geometry ---------------------------------------------------------------------


def dominates(a: Scores, b: Scores) -> bool:
    """a is at least as good everywhere and strictly better somewhere."""
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def pareto_indices(scores: Sequence[Scores]) -> list[int]:
    keep = []
    for i, s in enumerate(scores):
        if any(dominates(t, s) or (t == s and j < i) for j, t in enumerate(scores) if j != i):
            continue
        keep.append(i)
    return keep


def hypervolume(points: Sequence[Sequence[float]]) -> float:
    """Exact volume dominated by ``points`` above the origin (all coordinates maximised)."""
    pts = [tuple(p) for p in points if p and all(c > 0 for c in p)]
    return _hv(pts, len(pts[0])) if pts else 0.0


def _hv(pts: list[tuple[float, ...]], d: int) -> float:
    if not pts:
        return 0.0
    if d == 1:
        return max(p[0] for p in pts)
    if d == 2:
        total, top = 0.0, 0.0
        order = sorted(pts, key=lambda p: -p[0])
        for i, p in enumerate(order):
            top = max(top, p[1])
            nxt = order[i + 1][0] if i + 1 < len(order) else 0.0
            total += (p[0] - nxt) * top
        return total
    order = sorted(pts, key=lambda p: -p[d - 1])
    total = 0.0
    for i, p in enumerate(order):
        nxt = order[i + 1][d - 1] if i + 1 < len(order) else 0.0
        layer = p[d - 1] - nxt
        if layer > 0:
            total += layer * _hv([q[: d - 1] for q in order[: i + 1]], d - 1)
    return total


# -- assessment ----------------------------------------------------------------------


@dataclass(frozen=True)
class Assessment:
    core_optimal: bool
    gain: float                      # relative growth of dominated volume
    hv_before: float
    hv_after: float
    scores: Scores
    dominated_by: tuple[str, ...]    # frontier members that already beat the candidate
    dominates: tuple[str, ...]       # frontier members the candidate would push off the frontier
    duplicate: bool = False


class Spectrum:
    """The archive of every winning point; the live frontier is its non-dominated subset."""

    def __init__(self, track: str, space: Space, entries: Sequence[Entry] = ()):
        self.track, self.space = track, space
        self.entries: list[Entry] = list(entries)

    # frontier
    def _scored(self) -> list[tuple[Entry, Scores]]:
        return [(e, self.space.scores(e.point)) for e in self.entries]

    def active(self) -> list[Entry]:
        scored = self._scored()
        return [scored[i][0] for i in pareto_indices([s for _, s in scored])]

    def volume(self) -> float:
        return hypervolume([s for _, s in self._scored()])

    def assess(self, point: Point) -> Assessment:
        if not self.entries:
            raise FrontierError("the spectrum is empty; seed it with the baseline first")
        cand = self.space.scores(point, conservative=True)
        scored = self._scored()
        if any(e.point.candidate_id == point.candidate_id for e, _ in scored):
            hv = hypervolume([s for _, s in scored])
            return Assessment(False, 0.0, hv, hv, cand, (), (), duplicate=True)
        keep = pareto_indices([s for _, s in scored])
        front = [scored[i] for i in keep]
        hv_before = hypervolume([s for _, s in front])
        hv_after = hypervolume([s for _, s in front] + [cand])
        dominated_by = tuple(e.point.candidate_id for e, s in front if dominates(s, cand) or s == cand)
        pushes_off = tuple(e.point.candidate_id for e, s in front if dominates(cand, s))
        gain = (hv_after - hv_before) / hv_before if hv_before > 0 else 1.0
        gain = max(gain, 0.0)
        return Assessment(not dominated_by and gain > 0, gain, hv_before, hv_after, cand,
                          dominated_by, pushes_off)

    def add(self, entry: Entry) -> None:
        if any(e.point.candidate_id == entry.point.candidate_id for e in self.entries):
            raise FrontierError(f"{entry.point.candidate_id} is already in the spectrum")
        self.entries.append(entry)

    # persistence
    def to_dict(self) -> dict:
        return {"schema": SPECTRUM_SCHEMA, "track": self.track,
                "entries": [e.to_dict() for e in self.entries]}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | Path, space: Space, *, track: str) -> "Spectrum":
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FrontierError(f"{path}: unreadable spectrum ({exc})") from None
        if doc.get("schema") != SPECTRUM_SCHEMA:
            raise FrontierError(f"{path}: not an {SPECTRUM_SCHEMA} file")
        if doc.get("track") != track:
            raise FrontierError(f"{path}: spectrum is for track {doc.get('track')!r}, not {track!r}")
        return cls(track, space, [Entry.from_dict(e) for e in doc.get("entries", [])])
