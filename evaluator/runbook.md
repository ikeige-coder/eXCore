# Validator runbook

How to host the validator node that evaluates miner PRs, records winners and merges them.

## What the node does

For each PR that adds `manifests/<name>.yaml`: parse it as data, build the model, audit it,
benchmark it on CPU, score accuracy, run the gates and the frontier, check the private holdout,
merge the PR (pinned to the evaluated commit), and commit the updated registry and charts.
One evaluation runs at a time (a workflow concurrency group), so benchmarks never overlap.

## Security model

- **Untrusted code never runs.** A miner PR may add one YAML file. The bot reads it as text through
  the GitHub API at the PR's head SHA. It never checks out, imports or executes PR content.
- The workflow uses `pull_request_target` so it can comment and merge on fork PRs. That is safe
  only because of the rule above. **Never add a step that checks out
  `github.event.pull_request.head.*`.**
- Build and audit run in subprocesses with a scrubbed environment (no tokens), resource limits, a
  wall-clock timeout that kills the process group, capped output, and network isolation when the
  host supports it (`bwrap` or `unshare`). This is defence in depth: also run the validator on a
  dedicated machine or VM, as an unprivileged user, with nothing valuable on it.
- Set `EXCORE_REQUIRE_ISOLATION=1` in the runner's environment to make builds refuse to run when the
  host cannot provide network isolation. Install `bubblewrap` to make it available.
- The holdout text lives outside the repository (`EXCORE_HOLDOUT_DIR`) and is never printed.
  Treat that directory as a secret.

## Requirements

**Validator machine** (runs candidates; results are only comparable within one ISA and host class):

| | |
|---|---|
| CPU | x86-64 with AVX2, or ARM with NEON; 6 or more cores |
| RAM | 32 GB. The preflight refuses to start with less than 24 GiB available, and a candidate may use at most 24 GiB resident |
| Disk | about 130 GB free: the BF16 source (~70 GB), the cached baseline (~20 GB), one candidate (~20 GB) and one `llama-quantize` output (~20 GB) |
| OS | Linux, or Windows with WSL2 (see below) |
| Power | high-performance profile, no sleep, nothing else heavy running. A throttling machine produces unstable benchmarks, and the benchmark rejects those rather than record them |

**Reference machine** (used once): 96 GiB of RAM or more. The unquantized BF16 model is ~70 GB and does not fit
on the validator. Build the reference and holdout files (and convert the Hugging Face weights to a BF16 GGUF)
once on any machine that big (a rented CPU box for a few hours is enough). The outputs are small and portable:
copy the pinned BF16 GGUF (needed for builds) and the `reference.npz` and `holdout/` files to the validator.
Candidate evaluation never needs the BF16 model in memory.

**What to expect** (estimates, not measurements):

| | |
|---|---|
| decode | memory-bandwidth bound: roughly 1 to 2 tokens/s for a ~20 GiB model on a typical dual-channel DDR4 system |
| prefill | compute bound: a few to ten tokens/s |
| one candidate, end to end | build 15 to 30 min, benchmark ~15 min, accuracy 1 to 2 h, long-context guards 1 to 2 h: **plan on 3 to 5 hours** |

So one validator clears a handful of PRs a day. That is by design: work is queued, and `max_tokens`,
`needle_contexts` and `needle_depths` in `configs/hpc_cpu.yaml` are the knobs if you need to trade
thoroughness for throughput (changing them changes the track: re-seed afterwards).

**Running on Windows (WSL2).** WSL2 caps its memory at half the host by default (16 GiB of 32), which is
too little. In `%UserProfile%\.wslconfig` set:

```
[wsl2]
memory=29GB
swap=0
```

then run `wsl --shutdown`. Keep the models, workdir and repository **inside the WSL filesystem** (for example
`~/excore`), not under `/mnt/c`: files on the Windows drive are read through a slow bridge, which wrecks
build and benchmark times.

## One-time setup

```bash
# 1. environment
python3 -m venv ~/excore-venv && . ~/excore-venv/bin/activate
git clone <your eXCore repo> && cd eXCore
pip install -e ".[eval,llama]"        # `llama` builds llama-cpp-python from source

# 2. llama.cpp tools at a pinned commit (records the commit in the lock)
scripts/setup_bench.sh --commit <40-hex sha> --lock --with-python
export EXCORE_LLAMA_QUANTIZE=$PWD/.excore/bin/llama-quantize
export EXCORE_LLAMA_BENCH=$PWD/.excore/bin/llama-bench

# 3. the unquantized source: a BF16 GGUF (convert the Hugging Face weights once with
#    llama.cpp's convert_hf_to_gguf.py --outtype bf16), then pin it by hash
scripts/download_models.sh --url <gguf url> --out /srv/excore/models --lock --repo <org/model> --revision <rev>
#    or, for a file you converted yourself:
excore sources pin --root /srv/excore/models --file model-bf16.gguf --repo <org/model> --revision <rev>
```

**Check the model config against the real model.** `configs/hpc_cpu.yaml` ships with plausible
placeholder dimensions. Build the baseline once; any mismatch is reported by name:

```bash
excore build experiments/base_variants/V0_baseline.yaml --source /srv/excore/models/model-bf16.gguf --out /tmp/v0.gguf
```

`source is missing tensor ...` or `source shape ... does not match the model config` means the
config (or the tensor-name map in `excore/model/qwen38_cpu.py`) needs correcting. Fix it before going on.

```bash
# 4. the public corpus: put openly licensed .txt files in data/corpus/, then lock it
python -c "from excore.eval.accuracy import lock_corpus; lock_corpus('data/corpus')"

# 5. reference distributions from the unquantized model (slow, once). Run this step on the
#    reference machine from the Requirements section, then copy reference.npz to the validator.
excore reference build --model /srv/excore/models/model-bf16.gguf --corpus data/corpus \
    --out /srv/excore/reference.npz --model-id $(sha256sum /srv/excore/models/model-bf16.gguf | cut -d' ' -f1)

# 6. private holdout shards (one directory of text per shard; more shards = slower rotation)
for n in 0 1 2 3; do
  mkdir -p /srv/excore/holdout/shard-$n
  excore reference build --no-lock --model /srv/excore/models/model-bf16.gguf --corpus /srv/excore/private-text-$n \
      --out /srv/excore/holdout/shard-$n/reference.npz --model-id <same model id>
done

# 7. seed the registry with the baseline (measures V0 and records it with no reward)
python -m evaluator.bot_engine --seed experiments/base_variants/V0_baseline.yaml
git add results && git commit -m "registry: seed baseline" && git push
```

## Connect GitHub

1. Register a self-hosted runner on this machine with labels `self-hosted, linux, excore-validator`.
2. Set repository **variables** (Settings > Secrets and variables > Actions > Variables):

| variable | value |
|---|---|
| `EXCORE_PYTHON` | the venv's python, e.g. `/home/validator/excore-venv/bin/python` |
| `EXCORE_SOURCE_GGUF` | `/srv/excore/models/model-bf16.gguf` |
| `EXCORE_REFERENCE` | `/srv/excore/reference.npz` |
| `EXCORE_WORKDIR` | `/srv/excore/work` |
| `EXCORE_HOLDOUT_DIR` | `/srv/excore/holdout` |
| `EXCORE_LLAMA_QUANTIZE`, `EXCORE_LLAMA_BENCH` | paths from step 2 |

3. The bot merges the PR and then pushes a registry commit to the default branch. If that branch is
   protected, allow the bot to push (a GitHub App or a token with bypass rights). If the push is
   refused, the PR is still merged and the bot's comment says the registry update failed. Fix the
   permissions, then add the entry by hand (the PR comment holds the numbers) or have the miner
   resubmit under a new name; re-running the same PR is refused as already merged.
4. Give the **owner/maintainers the triage or write role** so they can add override labels. The bot
   never reads who added a label; the permission model of the repository is the control.
5. Require the `ci` workflow on PRs that change code. `.github/CODEOWNERS` is included: it makes the
   owner (`@ikeige-coder`) the reviewer for everything except `manifests/`, which the bot handles. It takes
   effect once branch protection has "Require review from Code Owners" on (see the note about bot push
   rights above before enabling protection). Update the handle if ownership moves.

## Owner merges and overrides

Recipes you merge by hand count too. When a PR that adds a manifest is merged by anyone (the workflow
also fires on `closed` + merged), the bot fetches the manifest from the merge commit, evaluates it
exactly like any other, and records it in the registry with its tier. A PR the bot already merged is
recognised as recorded and skipped, so nothing is counted twice. Other files in an owner-merged PR
are ignored (they are your review); a merged PR must add exactly one new manifest to be recorded.

When the bot would reject a recipe on judgement (dominated by the frontier, or gain below the lowest
tier), you can still count it with a label. Labels can only be added by people with triage/write
access, so a label is an owner decision:

| label | effect |
|---|---|
| `excore-approve` | record it even though the frontier/reward stage rejected it; lowest tier unless a tier label is also set |
| `excore-tier:Ultra` / `XL` / `L` / `M` / `S` | force that tier (also re-tiers a recipe the bot accepted) |

Add the label to an open PR and the bot re-evaluates and merges it, or add it to an already-merged PR
and the bot records it. Adding a label re-runs the full evaluation (the baseline is cached; the
candidate is rebuilt and re-measured), so expect it to take as long as a normal run. Conflicting or
unknown tier labels are ignored and listed in the bot's comment, never guessed.

**An override changes the reward decision only.** It can never bypass the gates (stable benchmark,
RAM ceiling, drift limits, long-context guards), the private holdout, or the build and audit
integrity checks: those make sure the measurement itself is valid. An overridden entry is stored with
`via: "...+override"`, its real gain, and the forced tier, so the audit trail shows what happened. It
joins the archive but only lands on the live frontier if it is actually non-dominated.

Merged by hand but the bot says "not recorded"? Read its comment: it names the stage that stopped it.
If it is a gate or the holdout, fix the recipe and resubmit; if it is the frontier or reward stage,
add a label.

## Daily operation

- **A PR is evaluated automatically** when it is opened, updated or marked ready. Drafts are skipped.
- **Dry run** a PR without commenting, merging or writing: `python -m evaluator.bot_engine --pr N --dry-run`.
- **The baseline is cached** in `EXCORE_WORKDIR/baseline-cache.json` and rebuilt automatically when
  the config, reference, host, source or baseline recipe changes.
- **Needle seeds and holdout shards rotate weekly** (ISO week). The first PR of a new week pays for
  re-measuring the baseline on the new needles and shard.
- **Disk:** each candidate GGUF is deleted after its evaluation. `baseline.gguf` is kept on purpose.

## The bot refuses to run when

| symptom | cause |
|---|---|
| `sources are not verified` | lock is unpinned, or the source file's hash/size differs |
| `EXCORE_SOURCE_GGUF is not the file pinned...` | env var points at a different file |
| `baseline benchmark is unstable` | the host is too noisy; fix the machine, not the threshold |
| `llama-quantize not found` / `llama-bench not found` | set the env vars (setup step 2) |

## Changing the rules

Everything that affects results lives in `configs/hpc_cpu.yaml` and the pinned sources. Changing
a bound, tier, gate, thread count or source makes old results incomparable. Do it deliberately:
bump the track name, re-pin, re-seed, and announce it.

The tier thresholds and frontier bounds shipped here are **calibration placeholders**. After the
first real baseline run, set the RAM and speed bounds around what you actually see, and check that a
typical improvement lands in the tier you intend.

## Gittensor

Rewards flow through Gittensor only for repositories it lists. Apply for listing following the
Gittensor documentation, and keep the repository's merge policy consistent with this bot: the bot
is the only path by which miner PRs reach the default branch.

## Known limitations

- The llama.cpp adapter (`excore/eval/backend.py`) and the `llama-bench` / `llama-quantize` output
  handling are exercised in tests with stand-ins, not against the real binaries. Run the baseline
  seed (step 7) and a dry-run PR before opening the repository to miners.
- `llama-cpp-python` bundles its own llama.cpp. Keep it on the same release as the pinned commit.
- Long-context guards compare against the baseline recipe, not the BF16 original (that would take
  hours per needle on CPU, and the BF16 weights do not fit in 32 GiB).
- The 35B unit map assumes a dense hybrid (Gated DeltaNet plus attention plus MLP) layout. If the real
  model is a mixture-of-experts, the experts need their own units: tell the maintainers before pinning.
- One validator is a single point of trust. Anyone can reproduce a result: the build is
  deterministic, and the audit and the registry record everything needed to re-run it.
