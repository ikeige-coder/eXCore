"""The automated GitHub listener: one PR in, one verdict out.

Security model: a miner PR may add exactly one file, ``manifests/<name>.yaml``. The bot
reads that file's *text* through the API at the PR's head SHA and parses it as data. It
never checks out, imports or executes anything from the PR. Anything else in a PR
(including quantizer plugins, workflows or code) is refused and left for human review.

Flow: policy -> parse -> duplicate check -> build + audit (sandboxed) -> measure ->
gates -> frontier -> reward -> private holdout -> merge (pinned to the evaluated SHA) ->
registry update (spectrum.json + charts) -> comment.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from excore.build import manifest_quantizer_names
from excore.config import load_config
from excore.frontier import RewardDecision, Verdict, judge, load_tiers
from excore.frontier.core_frontier import Space, Spectrum
from excore.frontier.plots import write_plots
from excore.manifest import MAX_BYTES, ManifestError, parse_manifest
from excore.model.qwen38_cpu import ModelSpec
from excore.quantizers import QuantizerError, ensure_loaded
from excore.sources import SourcesError, require_pinned, source_gguf

from .github import ChangedFile, GitHubClient, GitHubError, GitHubHTTP, PullRequest
from .runner import EvalBundle, EvalFailure, LocalRunner, Runner
from .sandbox import Sandbox, SandboxPolicy

MANIFEST_RE = re.compile(r"^manifests/([a-z0-9][a-z0-9._-]{0,63})\.yaml$")
MAX_FILES = 20
RESULTS_DIR = "results/core_spectrum"


@dataclass(frozen=True)
class Classification:
    manifest_path: str | None
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems


def classify_files(files: Sequence[ChangedFile]) -> Classification:
    """A miner PR is exactly one *new* manifests/<name>.yaml and nothing else."""
    if len(files) > MAX_FILES:
        return Classification(None, (f"too many files changed ({len(files)})",))
    manifests, problems = [], []
    for f in files:
        if MANIFEST_RE.match(f.filename) and f.status == "added":
            manifests.append(f.filename)
        elif MANIFEST_RE.match(f.filename):
            problems.append(f"`{f.filename}` is {f.status}; recipes are immutable once submitted, add a new file instead")
        else:
            problems.append(f"`{f.filename}` is outside `manifests/`; code, plugin and workflow changes need maintainer review")
    if not manifests and not problems:
        problems.append("no manifest found; add `manifests/<name>.yaml`")
    if len(manifests) > 1:
        problems.append("one manifest per PR")
    return Classification(manifests[0] if len(manifests) == 1 and not problems else None, tuple(problems))


@dataclass(frozen=True)
class Outcome:
    pr: int
    stage: str                 # skipped | policy | manifest | duplicate | evaluation | verdict | holdout | stale | merged | dry-run
    accepted: bool
    merged: bool
    message: str
    tier: str | None = None
    registry_ok: bool = True


# -- owner override ---------------------------------------------------------------------------------
#
# Labels can only be applied by people with triage/write access, so a label is an owner decision.
#   excore-approve        record this recipe even if the frontier/reward stage rejected it
#                         (it gets the lowest tier unless a tier label is also present)
#   excore-tier:<Name>    force the tier: Ultra, XL, L, M or S
# An override can change the *reward decision* only. It can never bypass the gates (stability, RAM,
# drift, long-context guards), the private holdout, or the build/audit integrity checks.

APPROVE_LABEL = "excore-approve"
TIER_LABEL = "excore-tier:"


@dataclass(frozen=True)
class Override:
    tier: str | None           # e.g. "core:XL"; None = keep the computed tier (or the lowest if none)
    label: str


def parse_override(labels: Sequence[str], cfg: Mapping) -> tuple[Override | None, list[str]]:
    """(override, ignored labels). Conflicting or unknown tier labels are ignored, never guessed."""
    known = {t.name.split(":", 1)[1].lower(): t.name for t in load_tiers(cfg)}
    ignored: list[str] = []
    tier_labels: list[tuple[str, str]] = []
    approve = None
    for label in labels:
        low = label.lower()
        if low.startswith(TIER_LABEL):
            key = low[len(TIER_LABEL):]
            if key in known:
                tier_labels.append((label, known[key]))
            else:
                ignored.append(f"`{label}` (unknown tier; use one of {', '.join(sorted(known))})")
        elif low == APPROVE_LABEL:
            approve = label
    if len(tier_labels) > 1:
        ignored += [f"`{label}` (conflicting tier labels)" for label, _ in tier_labels]
        tier_labels = []
    if tier_labels:
        return Override(tier_labels[0][1], tier_labels[0][0]), ignored
    if approve:
        return Override(None, approve), ignored
    return None, ignored


def apply_override(verdict: Verdict, override: Override | None, cfg: Mapping) -> Verdict:
    """Turn a frontier/reward-stage rejection into an acceptance, or re-tier an accepted verdict."""
    if override is None or verdict.point is None or verdict.stage in ("gates", "holdout"):
        return verdict
    a = verdict.assessment
    if a is None and not verdict.accepted:
        return verdict                                    # e.g. empty spectrum: nothing to compare against
    if a is not None and a.duplicate:
        return verdict
    if verdict.accepted and override.tier is None:
        return verdict
    tiers = {t.name: t for t in load_tiers(cfg)}
    lowest = min(tiers.values(), key=lambda t: t.min_gain).name
    tier = override.tier or verdict.tier or lowest
    gain = a.gain if a is not None else 0.0
    reward = RewardDecision(tier, tiers[tier].multiplier, gain, f"owner override ({override.label}): {tier}")
    return Verdict(True, "accepted", (reward.reason,), verdict.gates, verdict.point, a, reward)


class Git:
    """Thin wrapper over the git CLI for the registry commit."""

    def __init__(self, repo: Path, *, remote: str = "origin", branch: str = "main"):
        self.repo, self.remote, self.branch = Path(repo), remote, branch

    def run(self, *args: str) -> str:
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        res = subprocess.run(["git", "-C", str(self.repo), *args], capture_output=True, text=True, env=env)
        if res.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {(res.stderr or res.stdout).strip()}")
        return res.stdout

    def pull(self) -> None:
        self.run("fetch", self.remote, self.branch)
        self.run("merge", "--ff-only", f"{self.remote}/{self.branch}")

    def commit_and_push(self, paths: Sequence[str], message: str) -> None:
        self.run("add", "--", *paths)
        self.run("-c", "user.name=eXCore bot", "-c", "user.email=bot@excore.invalid", "commit", "-m", message)
        self.run("push", self.remote, f"HEAD:{self.branch}")


def _fmt(v: float, nd: int = 3) -> str:
    return "n/a" if v != v else f"{v:.{nd}f}"


def render_report(name: str, verdict: Verdict, bundle: EvalBundle | None, *, merged: bool = False,
                  notes: Sequence[str] = ()) -> str:
    icon = "✅" if verdict.accepted else "❌"
    lines = [f"### {icon} eXCore evaluation: `{name}`", ""]
    if bundle is not None:
        a, p, b = bundle.accuracy, bundle.perf, bundle.baseline_perf
        lines += [
            "| metric | result | vs baseline |", "|---|---|---|",
            f"| RP-KL drift | {_fmt(a.rp_kl, 4)} ± {_fmt(a.stderr, 4)} | baseline {_fmt(bundle.baseline_public_kl, 4)} |",
            f"| top-1 agreement | {a.top1_agreement:.1%} | |",
            f"| decode | {p.gen_tps:.2f} tok/s | ×{p.gen_tps / b.gen_tps:.2f} |",
            f"| prefill | {p.prefill_tps:.2f} tok/s | ×{p.prefill_tps / b.prefill_tps:.2f} |",
            f"| peak RAM | {p.peak_rss_gib:.2f} GiB | {p.peak_rss_gib - b.peak_rss_gib:+.2f} GiB |",
            "",
        ]
        for g in bundle.guards:
            lines.append(f"- guard `{g.name}`: {g.candidate_score:.2f} (reference {g.reference_score:.2f})")
        if bundle.guards_skipped:
            lines.append("- long-context guards were skipped because a cheaper gate already failed")
        if bundle.guards or bundle.guards_skipped:
            lines.append("")
    if verdict.accepted and verdict.reward and verdict.tier:
        lines.append(f"**Tier: `{verdict.tier}`** (×{verdict.reward.multiplier:g} multiplier, frontier gain {verdict.reward.gain:.2%})")
    elif verdict.accepted:
        lines.append("**Accepted as a bootstrap entry** (seeds the frontier, no reward).")
    else:
        lines.append(f"**Not {'recorded' if merged else 'merged'}** at stage `{verdict.stage}`:")
    lines += [f"- {r}" for r in verdict.reasons]
    lines += [f"- ignored label {n}" for n in notes]
    return "\n".join(lines)


class BotEngine:
    def __init__(self, *, cfg: Mapping, repo_root: Path, gh: GitHubClient | None, runner: Runner,
                 git: Git | None = None, dry_run: bool = False, branch: str = "main"):
        self.cfg, self.repo_root, self.gh, self.runner, self.git, self.dry_run = cfg, Path(repo_root), gh, runner, git, dry_run
        self.branch = branch
        self.spec = ModelSpec.from_mapping(cfg["model"])
        self.spectrum_path = self.repo_root / RESULTS_DIR / "spectrum.json"
        self.plots_dir = self.repo_root / RESULTS_DIR / "plots"

    def spectrum(self) -> Spectrum:
        return Spectrum.load(self.spectrum_path, Space.from_config(self.cfg), track=self.cfg["track"])

    def _say(self, pr: int, body: str) -> None:
        if self.dry_run:
            print(f"--- comment on #{pr} ---\n{body}\n")
        else:
            self.gh.comment(pr, body)

    def _out(self, pr: int, stage: str, message: str, *, comment: str | None = None, **kw) -> Outcome:
        if comment:
            self._say(pr, comment)
        return Outcome(pr, stage, kw.pop("accepted", False), kw.pop("merged", False), message, **kw)

    def handle_pr(self, number: int, *, expect_sha: str | None = None) -> Outcome:
        """Evaluate an open PR (and merge it if it earns a tier), or record an already-merged one."""
        pr = self.gh.get_pr(number)
        if pr.merged:
            return self._handle_merged(pr)
        if pr.state != "open":
            return self._out(number, "skipped", "PR is not open")
        if pr.draft:
            return self._out(number, "skipped", "PR is a draft")
        if expect_sha and pr.head_sha != expect_sha:
            return self._out(number, "skipped", "head moved since the event; a newer run will evaluate it")

        cls = classify_files(self.gh.list_files(number))
        if not cls.ok:
            body = "### ❌ eXCore: this PR cannot be auto-evaluated\n\n" + "\n".join(f"- {p}" for p in cls.problems)
            return self._out(number, "policy", "; ".join(cls.problems), comment=body)
        loaded = self._load(pr, cls.manifest_path, pr.head_sha, merged=False)
        if isinstance(loaded, Outcome):
            return loaded
        return self._run(pr, *loaded, merged=False)

    def _handle_merged(self, pr: PullRequest) -> Outcome:
        """A PR the owner (or anyone) merged outside the bot: evaluate the recipe and record it too."""
        n = pr.number
        if pr.base_ref != self.branch:
            return self._out(n, "skipped", f"merged into {pr.base_ref}, not {self.branch}")
        manifests = [f.filename for f in self.gh.list_files(n)
                     if MANIFEST_RE.match(f.filename) and f.status == "added"]
        if not manifests:
            return self._out(n, "skipped", "merged PR adds no new manifest")
        if len(manifests) > 1:
            return self._out(n, "policy", "merged PR adds several manifests",
                             comment="### ⚠️ eXCore: this merged PR adds several manifests. The bot records one recipe per "
                                     "PR; please resubmit them as separate PRs so each can be evaluated.")
        loaded = self._load(pr, manifests[0], pr.merge_commit_sha or pr.head_sha, merged=True)
        if isinstance(loaded, Outcome):
            return loaded
        return self._run(pr, *loaded, merged=True)

    def _load(self, pr: PullRequest, path: str, ref: str, *, merged: bool):
        """Fetch and parse the manifest as data. Returns (manifest, text, candidate_id) or an Outcome."""
        n = pr.number
        raw = self.gh.get_file(path, ref)
        if raw is None or len(raw) > MAX_BYTES:
            return self._out(n, "manifest", "manifest missing or too large",
                             comment=f"### ❌ eXCore\n\n`{path}` is missing or larger than {MAX_BYTES} bytes.")
        try:
            text = raw.decode("utf-8")
            manifest = parse_manifest(text, self.spec, expected_track=self.cfg["track"])
            ensure_loaded(manifest_quantizer_names(manifest))
            stem = MANIFEST_RE.match(path).group(1)
            if manifest.name != stem:
                raise ManifestError([f"`name: {manifest.name}` must match the file name `{stem}`"])
            cid = manifest.candidate_id(self.spec)
        except (ManifestError, QuantizerError, UnicodeDecodeError) as exc:
            problems = exc.problems if isinstance(exc, ManifestError) else [str(exc)]
            return self._out(n, "manifest", "; ".join(problems),
                             comment="### ❌ eXCore: invalid manifest\n\n" + "\n".join(f"- {p}" for p in problems))
        dup = next((e for e in self.spectrum().entries if e.point.candidate_id == cid), None)
        if dup is not None:                                # the bot already recorded it: stay quiet then
            return self._out(n, "duplicate", "identical recipe already in the registry",
                             comment=None if merged else f"### ❌ eXCore: duplicate\n\nThis recipe is identical to "
                                     f"`{dup.point.name}` (`{cid}`), which is already in the registry.")
        return manifest, text, cid

    def _run(self, pr: PullRequest, manifest, text: str, cid: str, *, merged: bool) -> Outcome:
        n, spectrum = pr.number, self.spectrum()
        with tempfile.TemporaryDirectory(prefix="excore-pr-") as tmp:
            mpath = Path(tmp) / f"{manifest.name}.yaml"
            mpath.write_text(text, encoding="utf-8")
            bundle: EvalBundle | None = None
            try:
                try:
                    bundle = self.runner.evaluate(mpath, manifest, cid)
                except EvalFailure as exc:
                    return self._out(n, "evaluation", str(exc),
                                     comment=f"### ❌ eXCore: evaluation failed\n\n```\n{exc}\n```")
                verdict = judge(candidate_id=cid, name=manifest.name, accuracy=bundle.accuracy, perf=bundle.perf,
                                baseline_perf=bundle.baseline_perf, guards=bundle.guards, spectrum=spectrum,
                                cfg=self.cfg, host=bundle.host)
                override, ignored = parse_override(pr.labels, self.cfg)
                overridden = False
                if override is not None:
                    changed = apply_override(verdict, override, self.cfg)
                    overridden, verdict = changed is not verdict, changed
                if not verdict.accepted:
                    return self._out(n, "verdict", "; ".join(verdict.reasons),
                                     comment=render_report(manifest.name, verdict, bundle, merged=merged, notes=ignored))
                hv = self.runner.holdout(bundle)
                if not hv.passed:
                    msg = (f"holdout gain {hv.holdout_gain:+.2%} is below the {hv.required:+.2%} required "
                           f"for a public gain of {hv.public_gain:+.2%} (shard {hv.shard})")
                    rejected = Verdict(False, "holdout", (msg,), verdict.gates, verdict.point, verdict.assessment, verdict.reward)
                    return self._out(n, "holdout", msg,
                                     comment=render_report(manifest.name, rejected, bundle, merged=merged, notes=ignored))
                report = render_report(manifest.name, verdict, bundle, merged=merged, notes=ignored)
                suffix = "+override" if overridden else ""
                if merged:
                    return self._record_merged(pr, verdict, report, via="owner-merge" + suffix)
                return self._finish(pr, manifest.name, verdict, report, via="bot-merge" + suffix)
            finally:
                if bundle is not None:
                    self.runner.cleanup(bundle)

    def _record_merged(self, pr: PullRequest, verdict: Verdict, report: str, *, via: str) -> Outcome:
        if self.dry_run:
            return self._out(pr.number, "dry-run", "would record", comment=report, accepted=True, tier=verdict.tier)
        registry_ok, note = True, ""
        try:
            self.record(verdict, ref=f"PR#{pr.number}", via=via)
        except Exception as exc:
            registry_ok, note = False, f"\n\n⚠️ the registry update failed: `{exc}`. A maintainer must re-run it."
        tail = "\n\nThis PR was merged outside the bot; the recipe was evaluated and recorded in the registry." if registry_ok else ""
        return self._out(pr.number, "recorded" if registry_ok else "recorded-failed",
                         "recorded in the registry" if registry_ok else "registry update failed",
                         comment=report + tail + note, accepted=True, merged=True, tier=verdict.tier, registry_ok=registry_ok)

    def _finish(self, pr: PullRequest, name: str, verdict: Verdict, report: str, *, via: str) -> Outcome:
        if self.dry_run:
            return self._out(pr.number, "dry-run", "would merge", comment=report, accepted=True, tier=verdict.tier)
        if self.gh.get_pr(pr.number).head_sha != pr.head_sha:
            return self._out(pr.number, "stale", "head changed during evaluation",
                             comment="### ⏸ eXCore: the PR changed while it was being evaluated; it will be re-evaluated.")
        try:
            self.gh.merge(pr.number, pr.head_sha, title=f"manifests: add {name} ({verdict.tier})")
        except GitHubError as exc:
            return self._out(pr.number, "stale", f"merge refused: {exc}",
                             comment=f"### ⚠️ eXCore: accepted but the merge was refused\n\n`{exc}`", accepted=True)
        registry_ok, note = True, ""
        try:
            self.record(verdict, ref=f"PR#{pr.number}", via=via)
        except Exception as exc:  # the PR is merged either way; make the failure loud
            registry_ok, note = False, f"\n\n⚠️ merged, but the registry update failed: `{exc}`. A maintainer must re-run it."
        return self._out(pr.number, "merged", "merged and recorded" if registry_ok else "merged, registry failed",
                         comment=report + note, accepted=True, merged=True, tier=verdict.tier, registry_ok=registry_ok)

    def record(self, verdict: Verdict, *, ref: str | None = None, via: str | None = None) -> None:
        """Add the entry to the spectrum, redraw the charts, and (if a Git handle exists) commit and push."""
        from datetime import datetime, timezone

        if self.git is not None:
            self.git.pull()
        spectrum = self.spectrum()
        if any(e.point.candidate_id == verdict.point.candidate_id for e in spectrum.entries):
            return
        spectrum.add(verdict.entry(ref=ref, via=via, merged_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))
        spectrum.save(self.spectrum_path)
        write_plots(spectrum, self.plots_dir)
        if self.git is not None:
            self.git.commit_and_push([RESULTS_DIR], f"registry: add {verdict.point.name} ({verdict.tier or 'bootstrap'})")

    def seed(self, manifest_path: Path) -> Outcome:
        """Operator command: evaluate the baseline recipe and seed an empty spectrum with it."""
        text = Path(manifest_path).read_text(encoding="utf-8")
        manifest = parse_manifest(text, self.spec, expected_track=self.cfg["track"])
        ensure_loaded(manifest_quantizer_names(manifest))
        cid = manifest.candidate_id(self.spec)
        bundle = self.runner.evaluate(Path(manifest_path), manifest, cid)
        try:
            verdict = judge(candidate_id=cid, name=manifest.name, accuracy=bundle.accuracy, perf=bundle.perf,
                            baseline_perf=bundle.baseline_perf, guards=bundle.guards, spectrum=self.spectrum(),
                            cfg=self.cfg, host=bundle.host, bootstrap=True)
            if verdict.accepted:
                self.record(verdict, ref="seed", via="seed")
            return Outcome(0, "seeded" if verdict.accepted else "verdict", verdict.accepted, False,
                           "; ".join(verdict.reasons))
        finally:
            self.runner.cleanup(bundle)


# -- entry point ---------------------------------------------------------------------------------


def build_engine(args) -> BotEngine:
    cfg = load_config(args.config)
    root = Path(args.repo_root).resolve()
    env = os.environ
    for var in ("EXCORE_SOURCE_GGUF", "EXCORE_REFERENCE", "EXCORE_WORKDIR"):
        if not env.get(var):
            raise SystemExit(f"{var} is not set (see evaluator/runbook.md)")
    source, workdir = Path(env["EXCORE_SOURCE_GGUF"]), Path(env["EXCORE_WORKDIR"])
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        lock = require_pinned(root / "configs/sources.lock.json", source.parent, track=cfg["track"],
                              cache=workdir / "sources-verified.json")
    except SourcesError as exc:
        raise SystemExit(f"refusing to run: {exc}")
    if source_gguf(lock, source.parent).resolve() != source.resolve():
        raise SystemExit("EXCORE_SOURCE_GGUF is not the file pinned in configs/sources.lock.json")
    sandbox = Sandbox(SandboxPolicy(timeout_s=float(cfg["limits"]["build_timeout_s"]) * 2,
                                    require_isolation=env.get("EXCORE_REQUIRE_ISOLATION") == "1"))
    runner = LocalRunner(
        cfg, repo_root=root, sandbox=sandbox, source=source, reference_path=Path(env["EXCORE_REFERENCE"]), workdir=workdir,
        config_path=Path(args.config) if args.config else root / "configs/hpc_cpu.yaml",
        llama_quantize=env.get("EXCORE_LLAMA_QUANTIZE"), bench_binary=env.get("EXCORE_LLAMA_BENCH"),
        holdout_root=env.get("EXCORE_HOLDOUT_DIR"),
    )
    gh = None
    if args.pr:
        repo, token = env.get("GITHUB_REPOSITORY"), env.get("GITHUB_TOKEN")
        if not repo or not token:
            raise SystemExit("GITHUB_REPOSITORY and GITHUB_TOKEN are required")
        gh = GitHubHTTP(repo, token)
    return BotEngine(cfg=cfg, repo_root=root, gh=gh, runner=runner, git=Git(root, branch=args.branch), dry_run=args.dry_run,
                     branch=args.branch)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m evaluator.bot_engine")
    p.add_argument("--pr", type=int, help="pull request number to evaluate, or to record if it is already merged")
    p.add_argument("--head-sha", help="the head SHA from the triggering event")
    p.add_argument("--seed", metavar="MANIFEST", help="operator: seed an empty spectrum with the baseline manifest")
    p.add_argument("--config", default=None)
    p.add_argument("--repo-root", default=".")
    p.add_argument("--branch", default="main")
    p.add_argument("--dry-run", action="store_true", help="evaluate and print, but never comment, merge or write")
    args = p.parse_args(argv)
    if bool(args.pr) == bool(args.seed):
        p.error("give exactly one of --pr or --seed")
    engine = build_engine(args)
    if args.seed:
        out = engine.seed(Path(args.seed))
        print(f"{out.stage}: {out.message}")
        return 0 if out.accepted else 1
    out = engine.handle_pr(args.pr, expect_sha=args.head_sha)
    print(f"#{out.pr} {out.stage}: {out.message}" + (f" [{out.tier}]" if out.tier else ""))
    return 0 if out.registry_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
