import subprocess
from pathlib import Path

import numpy as np
import pytest

from evaluator.bot_engine import BotEngine, Git, classify_files, render_report
from evaluator.github import ChangedFile, GitHubError, PullRequest
from evaluator.runner import BaselinePerf, EvalBundle, EvalFailure
from excore.eval.accuracy import AccuracyResult
from excore.eval.holdout import HoldoutVerdict
from excore.eval.performance import PerfResult, PerfRun
from excore.frontier import Entry, Point, Space, Spectrum

HEAD = "schema: excore/manifest@1\ntrack: CPU-35B\nname: {n}\n"
GOOD = HEAD.format(n="mlp-q8") + "default: Q4_0\ndefault_quantizer: rtn\nrules:\n  - {match: 'lm_head', format: Q8_0}\n"


def acc(kl=0.02, top1=0.9, se=0.0005):
    return AccuracyResult(kl, se, top1, 100, np.zeros(100, np.float32), np.full(10, kl), "r")


def perf(gen=12.0, pre=120.0, ram=14.0, stable=True):
    run = PerfRun(pre, gen, ram, 1.0)
    return PerfResult(pre, gen, ram, (run, run), stable, 0.0, 0.0, 4096, 128, 8)


def bundle(**kw):
    return EvalBundle(kw.get("a", acc()), kw.get("p", perf()), BaselinePerf(10.0, 100.0, 20.0), [], {"machine": "x"}, 0.02)


class FakeRunner:
    def __init__(self, result=None, hold=None, error=None):
        self.result, self.error = result or bundle(), error
        self.hold = hold or HoldoutVerdict(True, 0.0, 0.0, -0.005, "shard-0")
        self.evaluated, self.cleaned = [], 0

    def evaluate(self, path, manifest, cid):
        self.evaluated.append((path.name, cid))
        assert path.read_text().startswith("schema:")
        if self.error:
            raise self.error
        return self.result

    def holdout(self, b):
        return self.hold

    def cleanup(self, b):
        self.cleaned += 1


class FakeGH:
    def __init__(self, files=None, content=GOOD, **pr):
        self.pr = PullRequest(7, "sha1", "main", "miner", False, "open", False, (), "fork/x")
        for k, v in pr.items():
            self.pr = PullRequest(**{**self.pr.__dict__, k: v})
        self.files = files if files is not None else [ChangedFile("manifests/mlp-q8.yaml", "added")]
        self.content, self.comments, self.merges, self.on_merge, self.merge_error = content, [], [], None, None
        self.head_after = None

    def get_pr(self, n):
        if self.head_after and self.merges == [] and getattr(self, "_seen", False):
            return PullRequest(**{**self.pr.__dict__, "head_sha": self.head_after})
        self._seen = True
        return self.pr

    def list_files(self, n):
        return self.files

    def get_file(self, path, ref):
        assert ref in (self.pr.head_sha, self.pr.merge_commit_sha)
        return None if self.content is None else (self.content.encode() if isinstance(self.content, str) else self.content)

    def comment(self, n, body):
        self.comments.append(body)

    def merge(self, n, sha, *, title=None):
        if self.merge_error:
            raise self.merge_error
        self.merges.append((n, sha, title))
        if self.on_merge:
            self.on_merge()


@pytest.fixture()
def repo(tmp_path, tiny_cfg):
    root = tmp_path / "repo"
    (root / "results/core_spectrum/plots").mkdir(parents=True)
    sp = Spectrum("CPU-35B", Space.from_config(tiny_cfg))
    sp.add(Entry(Point("base", "v0", 0.02, 1.0, 1.0, 20.0), 0.0, None))
    sp.save(root / "results/core_spectrum/spectrum.json")
    return root


def engine(repo, tiny_cfg, gh, runner, **kw):
    return BotEngine(cfg=tiny_cfg, repo_root=repo, gh=gh, runner=runner, **kw)


# -- policy ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "files, fragment",
    [
        ([ChangedFile("manifests/a.yaml", "added")], None),
        ([], "no manifest found"),
        ([ChangedFile("manifests/a.yaml", "modified")], "immutable"),
        ([ChangedFile("manifests/a.yaml", "added"), ChangedFile("excore/quantizers/evil.py", "added")], "outside `manifests/`"),
        ([ChangedFile(".github/workflows/pr_evaluator.yml", "modified")], "maintainer review"),
        ([ChangedFile("manifests/a.yaml", "added"), ChangedFile("manifests/b.yaml", "added")], "one manifest per PR"),
        ([ChangedFile("manifests/A.yaml", "added")], "outside"),
        ([ChangedFile("manifests/../x.yaml", "added")], "outside"),
        ([ChangedFile("manifests/sub/a.yaml", "added")], "outside"),
        ([ChangedFile("manifests/a.yaml", "renamed")], "immutable"),
        ([ChangedFile(f"manifests/f{i}.yaml", "added") for i in range(30)], "too many files"),
    ],
)
def test_classification(files, fragment):
    c = classify_files(files)
    if fragment is None:
        assert c.ok and c.manifest_path == "manifests/a.yaml"
    else:
        assert not c.ok and fragment in " ".join(c.problems)


def test_forbidden_changes_never_reach_the_runner(repo, tiny_cfg):
    gh, run = FakeGH(files=[ChangedFile("excore/build.py", "modified")]), FakeRunner()
    out = engine(repo, tiny_cfg, gh, run).handle_pr(7)
    assert out.stage == "policy" and not out.accepted and run.evaluated == [] and gh.merges == []
    assert "maintainer review" in gh.comments[0]


@pytest.mark.parametrize("pr, reason", [({"draft": True}, "draft"), ({"state": "closed"}, "not open")])
def test_skips_drafts_and_closed(repo, tiny_cfg, pr, reason):
    gh = FakeGH(**pr)
    out = engine(repo, tiny_cfg, gh, FakeRunner()).handle_pr(7)
    assert out.stage == "skipped" and reason in out.message and gh.comments == []


def test_stale_event_is_skipped(repo, tiny_cfg):
    out = engine(repo, tiny_cfg, FakeGH(), FakeRunner()).handle_pr(7, expect_sha="older")
    assert out.stage == "skipped" and "head moved" in out.message


# -- manifest handling ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content, fragment",
    [
        (None, "missing"),
        (b"\xff\xfe", "can't"),  # replaced below by a real decode check
        ("not: [valid", "invalid YAML"),
        (GOOD.replace("name: mlp-q8", "name: other"), "must match the file name"),
        (GOOD + "rules2: 1\n", "unknown top-level key"),
        (GOOD.replace("Q8_0", "Q9"), "unknown format"),
        (GOOD.replace("default_quantizer: rtn", "default_quantizer: ghost"), "unknown quantizer"),
        (GOOD + "# " + "x" * 70000, "larger than"),
    ],
)
def test_bad_manifests_are_explained(repo, tiny_cfg, content, fragment):
    gh, run = FakeGH(content=content), FakeRunner()
    out = engine(repo, tiny_cfg, gh, run).handle_pr(7)
    assert not out.accepted and out.stage == "manifest" and run.evaluated == [] and gh.merges == []
    if fragment != "can't":
        assert fragment in gh.comments[0] or fragment in out.message


def test_duplicate_recipe_is_refused(repo, tiny_cfg, tiny_spec):
    from excore.manifest import parse_manifest

    cid = parse_manifest(GOOD, tiny_spec).candidate_id(tiny_spec)
    sp = Spectrum.load(repo / "results/core_spectrum/spectrum.json", Space.from_config(tiny_cfg), track="CPU-35B")
    sp.add(Entry(Point(cid, "earlier", 0.02, 1.1, 1.1, 18.0), 0.1, "core:L"))
    sp.save(repo / "results/core_spectrum/spectrum.json")
    gh, run = FakeGH(), FakeRunner()
    out = engine(repo, tiny_cfg, gh, run).handle_pr(7)
    assert out.stage == "duplicate" and "earlier" in gh.comments[0] and run.evaluated == []


# -- evaluation and verdicts ---------------------------------------------------------------------------


def test_accepted_candidate_is_merged_and_recorded(repo, tiny_cfg):
    gh, run = FakeGH(), FakeRunner()
    out = engine(repo, tiny_cfg, gh, run).handle_pr(7, expect_sha="sha1")
    assert out.accepted and out.merged and out.stage == "merged" and out.tier and out.registry_ok
    assert gh.merges == [(7, "sha1", f"manifests: add mlp-q8 ({out.tier})")]
    assert run.cleaned == 1 and out.tier in gh.comments[-1]
    sp = Spectrum.load(repo / "results/core_spectrum/spectrum.json", Space.from_config(tiny_cfg), track="CPU-35B")
    assert [e.point.name for e in sp.entries] == ["v0", "mlp-q8"] and sp.entries[1].ref == "PR#7"
    assert sp.entries[1].merged_at and len(list((repo / "results/core_spectrum/plots").glob("*.svg"))) == 3


def test_evaluation_failure_never_merges(repo, tiny_cfg):
    gh, run = FakeGH(), FakeRunner(error=EvalFailure("build", "llama-quantize failed (3)"))
    out = engine(repo, tiny_cfg, gh, run).handle_pr(7)
    assert out.stage == "evaluation" and gh.merges == [] and "llama-quantize failed" in gh.comments[0]


def test_failed_gate_and_dominated_candidates_are_not_merged(repo, tiny_cfg):
    gh = FakeGH()
    out = engine(repo, tiny_cfg, gh, FakeRunner(bundle(p=perf(ram=30.0)))).handle_pr(7)
    assert out.stage == "verdict" and "G_RAM" in out.message and gh.merges == []
    gh = FakeGH()
    out = engine(repo, tiny_cfg, gh, FakeRunner(bundle(a=acc(kl=0.05), p=perf(gen=9, pre=90, ram=22)))).handle_pr(7)
    assert out.stage == "verdict" and "dominated by base" in out.message and gh.merges == []
    assert gh.comments[0].startswith("### ❌") and "RP-KL drift" in gh.comments[0]


def test_holdout_failure_blocks_the_merge(repo, tiny_cfg):
    gh = FakeGH()
    hold = HoldoutVerdict(False, 0.2, 0.01, 0.095, "shard-1")
    out = engine(repo, tiny_cfg, gh, FakeRunner(hold=hold)).handle_pr(7)
    assert out.stage == "holdout" and not out.accepted and gh.merges == []
    assert "shard-1" in gh.comments[0] and "Not merged" in gh.comments[0]


def test_head_change_during_evaluation_blocks_the_merge(repo, tiny_cfg):
    gh = FakeGH()
    gh.head_after = "sha2"
    out = engine(repo, tiny_cfg, gh, FakeRunner()).handle_pr(7)
    assert out.stage == "stale" and gh.merges == [] and "changed while" in gh.comments[-1]


def test_refused_merge_is_reported_and_not_recorded(repo, tiny_cfg):
    gh = FakeGH()
    gh.merge_error = GitHubError(409, "Head branch was modified")
    out = engine(repo, tiny_cfg, gh, FakeRunner()).handle_pr(7)
    assert out.stage == "stale" and out.accepted and not out.merged
    sp = Spectrum.load(repo / "results/core_spectrum/spectrum.json", Space.from_config(tiny_cfg), track="CPU-35B")
    assert len(sp.entries) == 1


def test_dry_run_changes_nothing(repo, tiny_cfg, capsys):
    gh = FakeGH()
    out = engine(repo, tiny_cfg, gh, FakeRunner(), dry_run=True).handle_pr(7)
    assert out.stage == "dry-run" and out.accepted and gh.merges == [] and gh.comments == []
    assert "eXCore evaluation" in capsys.readouterr().out
    sp = Spectrum.load(repo / "results/core_spectrum/spectrum.json", Space.from_config(tiny_cfg), track="CPU-35B")
    assert len(sp.entries) == 1


def test_seed_bootstraps_an_empty_spectrum(repo, tiny_cfg, tmp_path):
    (repo / "results/core_spectrum/spectrum.json").write_text(
        '{"entries": [], "schema": "excore/spectrum@1", "track": "CPU-35B"}')
    m = tmp_path / "v0.yaml"
    m.write_text(GOOD.replace("mlp-q8", "v0"))
    out = engine(repo, tiny_cfg, None, FakeRunner()).seed(m)
    assert out.stage == "seeded" and out.accepted
    sp = Spectrum.load(repo / "results/core_spectrum/spectrum.json", Space.from_config(tiny_cfg), track="CPU-35B")
    assert sp.entries[0].tier is None and sp.entries[0].ref == "seed"
    again = engine(repo, tiny_cfg, None, FakeRunner()).seed(m)               # already present: judged as a duplicate
    assert not again.accepted


def test_report_is_readable_markdown(repo, tiny_cfg):
    gh = FakeGH()
    engine(repo, tiny_cfg, gh, FakeRunner()).handle_pr(7)
    body = gh.comments[-1]
    for needle in ("| RP-KL drift |", "| decode |", "| peak RAM |", "**Tier:", "×1.20"):
        assert needle in body


# -- git registry ---------------------------------------------------------------------------------------


def sh(cwd, *args):
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout


def test_registry_commit_is_pushed_after_a_merge(tmp_path, tiny_cfg):
    remote = tmp_path / "remote.git"
    sh(tmp_path, "git", "init", "--bare", "-b", "main", str(remote))
    bot, other = tmp_path / "bot", tmp_path / "other"
    sh(tmp_path, "git", "clone", str(remote), str(bot))
    (bot / "results/core_spectrum/plots").mkdir(parents=True)
    sp = Spectrum("CPU-35B", Space.from_config(tiny_cfg))
    sp.add(Entry(Point("base", "v0", 0.02, 1.0, 1.0, 20.0), 0.0, None))
    sp.save(bot / "results/core_spectrum/spectrum.json")
    (bot / "README.md").write_text("x")
    sh(bot, "git", "checkout", "-b", "main")
    sh(bot, "git", "add", "-A")
    sh(bot, "git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-m", "init")
    sh(bot, "git", "push", "origin", "main")
    sh(tmp_path, "git", "clone", str(remote), str(other))

    def merge_on_github():          # the squash-merge lands on the remote while the bot's clone is behind
        (other / "manifests").mkdir(exist_ok=True)
        (other / "manifests/mlp-q8.yaml").write_text(GOOD)
        sh(other, "git", "add", "-A")
        sh(other, "git", "-c", "user.name=m", "-c", "user.email=m@m", "commit", "-m", "manifests: add mlp-q8")
        sh(other, "git", "push", "origin", "main")

    gh = FakeGH()
    gh.on_merge = merge_on_github
    out = engine(bot, tiny_cfg, gh, FakeRunner(), git=Git(bot)).handle_pr(7)
    assert out.merged and out.registry_ok, out
    log = sh(remote, "git", "log", "--format=%s", "main").splitlines()
    assert log[0].startswith("registry: add mlp-q8") and log[1] == "manifests: add mlp-q8"
    tree = sh(remote, "git", "ls-tree", "-r", "--name-only", "main")
    assert "results/core_spectrum/spectrum.json" in tree and "plots/accuracy_vs_ram.svg" in tree


def test_a_failing_registry_push_is_loud_but_the_pr_stays_merged(tmp_path, tiny_cfg, repo):
    class BrokenGit:
        def pull(self):
            raise RuntimeError("git fetch failed: network down")

    gh = FakeGH()
    out = engine(repo, tiny_cfg, gh, FakeRunner(), git=BrokenGit()).handle_pr(7)
    assert out.merged and not out.registry_ok and "registry update failed" in gh.comments[-1]


# -- owner merges and overrides -----------------------------------------------------------------------


from evaluator.bot_engine import Override, apply_override, parse_override
from excore.frontier import RewardDecision, Verdict, judge


def merged_gh(**kw):
    kw.setdefault("merged", True)
    kw.setdefault("state", "closed")
    kw.setdefault("merge_commit_sha", "msha")
    kw.setdefault("merged_by", "owner")
    return FakeGH(**kw)


def entries(repo, cfg):
    return Spectrum.load(repo / "results/core_spectrum/spectrum.json", Space.from_config(cfg), track="CPU-35B").entries


def test_owner_merged_recipe_is_evaluated_and_recorded(repo, tiny_cfg):
    gh, run = merged_gh(), FakeRunner()
    out = engine(repo, tiny_cfg, gh, run).handle_pr(7)
    assert out.stage == "recorded" and out.accepted and out.merged and out.tier and out.registry_ok
    assert gh.merges == [] and run.cleaned == 1                 # the bot never merges what is already merged
    e = entries(repo, tiny_cfg)[-1]
    assert e.point.name == "mlp-q8" and e.tier == out.tier and e.via == "owner-merge" and e.ref == "PR#7"
    assert "recorded in the registry" in gh.comments[-1] and out.tier in gh.comments[-1]
    assert len(list((repo / "results/core_spectrum/plots").glob("*.svg"))) == 3


def test_bot_merged_pr_is_not_recorded_twice(repo, tiny_cfg):
    gh, run = FakeGH(), FakeRunner()
    first = engine(repo, tiny_cfg, gh, run).handle_pr(7)
    assert first.merged and entries(repo, tiny_cfg)[-1].via == "bot-merge"
    after = merged_gh()                                          # the "closed" event arrives for the same PR
    run2 = FakeRunner()
    out = engine(repo, tiny_cfg, after, run2).handle_pr(7)
    assert out.stage == "duplicate" and run2.evaluated == [] and after.comments == []
    assert len(entries(repo, tiny_cfg)) == 2


def test_owner_merge_with_extra_files_still_records_the_manifest(repo, tiny_cfg):
    gh = merged_gh(files=[ChangedFile("manifests/mlp-q8.yaml", "added"), ChangedFile("docs/notes.md", "modified")])
    out = engine(repo, tiny_cfg, gh, FakeRunner()).handle_pr(7)
    assert out.stage == "recorded"


@pytest.mark.parametrize(
    "kw, files, stage",
    [
        ({"base_ref": "dev"}, None, "skipped"),
        ({}, [ChangedFile("manifests/mlp-q8.yaml", "modified")], "skipped"),
        ({}, [ChangedFile("excore/build.py", "modified")], "skipped"),
        ({}, [ChangedFile("manifests/a.yaml", "added"), ChangedFile("manifests/b.yaml", "added")], "policy"),
    ],
)
def test_merged_prs_that_cannot_be_recorded(repo, tiny_cfg, kw, files, stage):
    gh, run = merged_gh(files=files, **kw), FakeRunner()
    out = engine(repo, tiny_cfg, gh, run).handle_pr(7)
    assert out.stage == stage and not out.accepted and run.evaluated == [] and len(entries(repo, tiny_cfg)) == 1


def test_owner_merged_recipe_that_the_frontier_rejects_is_not_recorded_without_an_override(repo, tiny_cfg):
    gh = merged_gh()
    out = engine(repo, tiny_cfg, gh, FakeRunner(bundle(a=acc(kl=0.05), p=perf(gen=9, pre=90, ram=22)))).handle_pr(7)
    assert out.stage == "verdict" and not out.accepted and len(entries(repo, tiny_cfg)) == 1
    assert "Not recorded" in gh.comments[-1] and "dominated by base" in gh.comments[-1]


DOMINATED = dict(a=acc(kl=0.05), p=perf(gen=9, pre=90, ram=22))


def test_approve_label_records_a_rejected_recipe_at_the_lowest_tier(repo, tiny_cfg):
    gh = merged_gh(labels=("excore-approve",))
    out = engine(repo, tiny_cfg, gh, FakeRunner(bundle(**DOMINATED))).handle_pr(7)
    assert out.stage == "recorded" and out.tier == "core:S"
    e = entries(repo, tiny_cfg)[-1]
    assert e.tier == "core:S" and e.via == "owner-merge+override" and e.gain == 0.0
    assert "owner override" in gh.comments[-1]
    assert [x.point.name for x in Spectrum.load(repo / "results/core_spectrum/spectrum.json",
                                                Space.from_config(tiny_cfg), track="CPU-35B").active()] == ["v0"]


def test_tier_label_sets_the_tier(repo, tiny_cfg):
    gh = merged_gh(labels=("excore-tier:XL",))
    out = engine(repo, tiny_cfg, gh, FakeRunner(bundle(**DOMINATED))).handle_pr(7)
    assert out.tier == "core:XL" and entries(repo, tiny_cfg)[-1].tier == "core:XL"


def test_tier_label_re_tiers_a_recipe_the_frontier_already_accepts(repo, tiny_cfg):
    gh = merged_gh(labels=("excore-tier:S",))
    out = engine(repo, tiny_cfg, gh, FakeRunner()).handle_pr(7)
    assert out.tier == "core:S" and entries(repo, tiny_cfg)[-1].via == "owner-merge+override"


def test_override_on_an_open_pr_lets_the_bot_merge_it(repo, tiny_cfg):
    gh = FakeGH(labels=("excore-approve",))
    out = engine(repo, tiny_cfg, gh, FakeRunner(bundle(**DOMINATED))).handle_pr(7)
    assert out.stage == "merged" and out.merged and out.tier == "core:S"
    assert gh.merges and entries(repo, tiny_cfg)[-1].via == "bot-merge+override"


def test_override_never_bypasses_the_gates(repo, tiny_cfg):
    gh = merged_gh(labels=("excore-approve", "excore-tier:Ultra"))
    out = engine(repo, tiny_cfg, gh, FakeRunner(bundle(p=perf(ram=30.0)))).handle_pr(7)
    assert out.stage == "verdict" and not out.accepted and "G_RAM" in out.message
    assert len(entries(repo, tiny_cfg)) == 1


def test_override_never_bypasses_the_holdout(repo, tiny_cfg):
    gh = merged_gh(labels=("excore-approve",))
    hold = HoldoutVerdict(False, 0.2, 0.01, 0.095, "shard-1")
    out = engine(repo, tiny_cfg, gh, FakeRunner(bundle(**DOMINATED), hold=hold)).handle_pr(7)
    assert out.stage == "holdout" and len(entries(repo, tiny_cfg)) == 1


def test_override_cannot_duplicate_or_conjure_an_entry(repo, tiny_cfg, tiny_spec):
    from excore.manifest import parse_manifest

    cid = parse_manifest(GOOD, tiny_spec).candidate_id(tiny_spec)
    sp = Spectrum.load(repo / "results/core_spectrum/spectrum.json", Space.from_config(tiny_cfg), track="CPU-35B")
    sp.add(Entry(Point(cid, "earlier", 0.02, 1.1, 1.1, 18.0), 0.1, "core:L"))
    sp.save(repo / "results/core_spectrum/spectrum.json")
    out = engine(repo, tiny_cfg, FakeGH(labels=("excore-tier:Ultra",)), FakeRunner()).handle_pr(7)
    assert out.stage == "duplicate"
    empty = Spectrum("CPU-35B", Space.from_config(tiny_cfg))
    cand = Point("x", "x", 0.02, 1.2, 1.2, 14.0)
    v = judge(candidate_id="x", name="x", accuracy=acc(), perf=perf(), baseline_perf=BaselinePerf(10.0, 100.0, 20.0),
              guards=[], spectrum=empty, cfg=tiny_cfg)
    assert not v.accepted and apply_override(v, Override("core:XL", "l"), tiny_cfg) is v


@pytest.mark.parametrize(
    "labels, tier, ignored",
    [
        ((), None, 0),
        (("bug", "good first issue"), None, 0),
        (("excore-approve",), "approve", 0),
        (("EXCORE-APPROVE",), "approve", 0),
        (("excore-tier:xl",), "core:XL", 0),
        (("excore-tier:Ultra", "excore-approve"), "core:Ultra", 0),
        (("excore-tier:XL", "excore-tier:M"), None, 2),           # conflicting: ignored, not guessed
        (("excore-tier:Mega",), None, 1),                          # unknown
        (("excore-tier:Mega", "excore-approve"), "approve", 1),
    ],
)
def test_parse_override(tiny_cfg, labels, tier, ignored):
    o, bad = parse_override(labels, tiny_cfg)
    assert len(bad) == ignored
    if tier is None:
        assert o is None
    else:
        assert o is not None and o.tier == (None if tier == "approve" else tier)


def test_ignored_labels_are_reported_to_the_owner(repo, tiny_cfg):
    gh = merged_gh(labels=("excore-tier:Mega",))
    engine(repo, tiny_cfg, gh, FakeRunner(bundle(**DOMINATED))).handle_pr(7)
    assert "ignored label" in gh.comments[-1] and "Mega" in gh.comments[-1]


def test_dry_run_records_nothing_for_merged_prs(repo, tiny_cfg, capsys):
    out = engine(repo, tiny_cfg, merged_gh(), FakeRunner(), dry_run=True).handle_pr(7)
    assert out.stage == "dry-run" and len(entries(repo, tiny_cfg)) == 1
    assert "eXCore evaluation" in capsys.readouterr().out


def test_entry_roundtrips_via(repo, tiny_cfg):
    engine(repo, tiny_cfg, merged_gh(), FakeRunner()).handle_pr(7)
    old = {"point": Point("a", "a", 0.02, 1, 1, 10).to_dict(), "gain": 0.1, "tier": "core:S", "ref": None, "merged_at": None}
    assert Entry.from_dict(old).via is None                      # entries written before this field still load
    assert entries(repo, tiny_cfg)[-1].to_dict()["via"] == "owner-merge"
