"""Gates: a candidate must pass every one before it can be placed on the frontier.

* perf: the benchmark must be stable and inside the RAM ceiling,
* accuracy: drift and top-1 agreement within sane limits (no "destroyed" models),
* guards: task guards compare the candidate with the reference model. The long-context
  needle guards (8K / 16K) allow no drop, which catches recipes that quietly break
  long-context attention to save memory.

Everything except ``run_needle_guard`` is a pure function over measured results.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

NEEDLE_RESERVE = 24       # tokens left free for the answer
PRE_TEXT = "The following is a long document. Somewhere in it a vault passphrase is stated once.\n\n"
_ADJ = ("amber", "brisk", "cobalt", "dusty", "ember", "frosty", "gentle", "hollow", "ivory", "jolly")
_NOUN = ("falcon", "harbor", "lantern", "meadow", "orchid", "pebble", "quartz", "river", "summit", "thistle")


@dataclass(frozen=True)
class GateFailure:
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


@dataclass(frozen=True)
class GuardResult:
    name: str
    reference_score: float            # 0..1 on the unquantized reference
    candidate_score: float            # 0..1 on the candidate
    max_regression_pct: float         # allowed relative drop vs the reference

    @property
    def passed(self) -> bool:
        floor = self.reference_score * (1.0 - self.max_regression_pct / 100.0)
        return self.candidate_score >= floor - 1e-12


@dataclass
class GateReport:
    failures: list[GateFailure] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def codes(self) -> set[str]:
        return {f.code for f in self.failures}

    def summary(self) -> str:
        return "all gates passed" if self.passed else "; ".join(str(f) for f in self.failures)


def check_perf(perf, limits: Mapping) -> list[GateFailure]:
    out = []
    if not perf.stable:
        out.append(GateFailure(
            "G_UNSTABLE",
            f"benchmark runs disagree (prefill spread {perf.spread_prefill:.1%}, decode spread "
            f"{perf.spread_gen:.1%}, tolerance {float(limits['perf_tolerance']):.1%})"))
    cap = float(limits["max_peak_rss_gib"])
    if not perf.within_ram_limit(cap):
        out.append(GateFailure("G_RAM", f"peak RAM {perf.peak_rss_gib:.1f} GiB exceeds the {cap:.1f} GiB ceiling"))
    return out


def check_accuracy(acc, gates: Mapping) -> list[GateFailure]:
    out = []
    if not acc.rp_kl == acc.rp_kl or acc.rp_kl in (float("inf"), float("-inf")):
        return [GateFailure("G_ACC_INVALID", "accuracy result is not a finite number")]
    if acc.rp_kl > float(gates["max_rp_kl"]):
        out.append(GateFailure("G_DRIFT", f"RP-KL {acc.rp_kl:.4f} exceeds the limit {float(gates['max_rp_kl']):g}"))
    if acc.top1_agreement < float(gates["min_top1_agreement"]):
        out.append(GateFailure(
            "G_TOP1", f"top-1 agreement {acc.top1_agreement:.1%} is below {float(gates['min_top1_agreement']):.0%}"))
    return out


def check_guards(guards: Sequence[GuardResult]) -> list[GateFailure]:
    return [
        GateFailure(f"G_GUARD_{g.name}",
                    f"{g.name}: {g.candidate_score:.2f} vs reference {g.reference_score:.2f} "
                    f"(allowed drop {g.max_regression_pct:g}%)")
        for g in guards if not g.passed
    ]


def evaluate_gates(*, perf, accuracy, guards: Sequence[GuardResult], cfg: Mapping) -> GateReport:
    report = GateReport()
    report.failures += check_perf(perf, cfg["limits"])
    report.failures += check_accuracy(accuracy, cfg["gates"])
    report.failures += check_guards(guards)
    return report


# -- long-context needle guard ------------------------------------------------------------


class GenerativeBackend(Protocol):
    def tokenize(self, text: str) -> list[int]: ...

    def bos_token(self) -> int: ...

    def generate(self, prompt: list[int], max_new: int) -> str: ...


@dataclass(frozen=True)
class NeedleCase:
    ctx: int
    depth: float
    key: str
    answer: str


def needle_cases(contexts: Sequence[int], depths: Sequence[float], seed: int) -> list[NeedleCase]:
    """Deterministic cases for a seed; rotate the seed (e.g. by epoch) so they cannot be memorised."""
    cases = []
    for ctx in contexts:
        for depth in depths:
            rng = random.Random(f"{seed}:{ctx}:{depth}")
            key = f"{rng.choice('ABCDEFGH')}{rng.randint(10, 99)}"
            answer = f"{rng.choice(_ADJ)}-{rng.choice(_NOUN)}-{rng.randint(100, 999)}"
            cases.append(NeedleCase(int(ctx), float(depth), key, answer))
    return cases


def needle_prompt(case: NeedleCase, backend: GenerativeBackend, filler: Sequence[int]) -> list[int]:
    """BOS + intro + filler with the needle at ``depth`` + question; total = ctx - NEEDLE_RESERVE tokens."""
    if not filler:
        raise ValueError("filler tokens are required")
    pre = backend.tokenize(PRE_TEXT)
    needle = backend.tokenize(f"\nThe secret passphrase for vault {case.key} is {case.answer}.\n")
    ask = backend.tokenize(
        f"\n\nQuestion: What is the secret passphrase for vault {case.key}?\n"
        f"Answer: The secret passphrase for vault {case.key} is")
    budget = case.ctx - NEEDLE_RESERVE - 1 - len(pre) - len(needle) - len(ask)
    if budget <= 0:
        raise ValueError(f"context {case.ctx} is too short for the needle prompt")
    reps = budget // len(filler) + 1
    body = (list(filler) * reps)[:budget]
    cut = int(budget * case.depth)
    return [backend.bos_token()] + pre + body[:cut] + needle + body[cut:] + ask


def run_needle_guard(
    backend: GenerativeBackend, contexts: Sequence[int], depths: Sequence[float],
    filler: Sequence[int], seed: int,
) -> tuple[float, list[tuple[NeedleCase, bool]]]:
    """Fraction of needles retrieved, plus the per-case outcomes."""
    outcomes = []
    for case in needle_cases(contexts, depths, seed):
        text = backend.generate(needle_prompt(case, backend, filler), NEEDLE_RESERVE)
        outcomes.append((case, case.answer.lower() in text.lower()))
    return sum(ok for _, ok in outcomes) / len(outcomes), outcomes
