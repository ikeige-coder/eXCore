# Core-Optimal scoring

eXCore ranks recipes on a four-dimensional Pareto frontier. This page explains exactly how a
submission becomes a reward tier. All numbers live in `configs/hpc_cpu.yaml`.

## The four objectives

| objective | measured by | better |
|---|---|---|
| accuracy drift | Reference-Partition KL-divergence (RP-KL) on a locked corpus | lower |
| decode speed | `llama-bench` tokens/s, 128 generated tokens | higher |
| prefill speed | `llama-bench` tokens/s on a 2048-token prompt | higher |
| peak RAM | high-water mark of the benchmark process | lower |

**RP-KL.** For each scored position the unquantized model's next-token distribution is
partitioned into its top-256 tokens plus one tail bucket. The candidate is scored by the KL
divergence of the two distributions coarsened to that partition. It is never larger than the full
KL, it needs only 256 numbers per position to store, and it is compared position by position on
identical tokens.

**Speeds are relative.** Every run also measures the baseline recipe (V0) on the same host, in
the same session. A candidate's speed is its ratio to that baseline, so different machines stay
comparable. Each measurement runs twice; the slower run is reported and the result is flagged
unstable if the runs differ by more than 5%.

## From measurements to a gain

Each objective is mapped to a 0-1 **score** (1 = best) between the bounds in the config: drift
on a log scale between 0.0005 and 0.5, speeds on a log scale between 0.5x and 3x the baseline,
RAM linearly between 12 and 24 GiB (the track's memory budget). Results beyond a bound earn no extra credit.

A candidate's **gain** is the relative growth of the frontier's dominated *hypervolume*, the
exact 4-D volume of score space that some entry beats:

    gain = (HV(frontier + candidate) - HV(frontier)) / HV(frontier)

A candidate that is dominated (some entry is at least as good on all four) adds nothing and is
rejected. One that trades one objective for another, or beats the frontier outright, adds volume.

## Noise cannot buy a place

A candidate is scored **conservatively** (drift at mean + 2 standard errors, speeds x0.95, RAM
+0.25 GiB) while incumbents keep their measured values. To win, a candidate must beat an
incumbent by more than the measurement noise. A 3% speed-up on its own is dominated and rejected.

## Reward tiers

| tier | min gain | multiplier |
|---|---|---|
| `core:Ultra` | 0.250 | 4.0 |
| `core:XL` | 0.100 | 2.5 |
| `core:L` | 0.040 | 1.5 |
| `core:M` | 0.015 | 1.0 |
| `core:S` | 0.005 | 0.5 |

Below `core:S` there is no reward and no merge. The highest bracket reached applies.

**Worked example** (frontier = one baseline: drift 0.020, RAM 19.6 GiB, speeds 1.0x):

| candidate | result |
|---|---|
| RAM 19.6 -> 18.6 GiB, drift 0.020 -> 0.022 | on the frontier, gain 0.14, `core:XL` |
| RAM 19.6 -> 17.6 GiB, drift 0.020 -> 0.024 | gain 0.32, `core:Ultra` |
| drift 0.020 -> 0.016, same speed and RAM | gain 0.04, `core:L` |
| decode x1.10 only | gain 0.055, `core:L` |
| decode x1.35, prefill x1.20, same RAM and drift | gain 0.57, `core:Ultra` |
| only +3% decode | dominated by the baseline, rejected |

Early gains are large because the frontier is a single point. They shrink as it fills, and each
new entry makes the next one harder. The bounds and tier thresholds are calibration values to be
tuned after the first real baseline run.

## Gates (checked first)

A candidate must pass all of these before it is placed:

1. **Stable benchmark** and **peak RAM** under the ceiling (24 GiB; the track assumes 32 GB of system RAM).
2. **Drift** at most 0.5 RP-KL and at least 50% top-1 agreement with the reference.
3. **Long-context guards**: needle-in-a-haystack retrieval at 8K and 16K tokens, at two depths,
   against needles that rotate weekly. The candidate may not retrieve fewer than the baseline.
4. **Private holdout** (only for would-be winners). The gain relative to the baseline must carry
   over to unseen, weekly-rotating text:

       holdout_gain >= min(0.5 * public_gain, public_gain) - 0.005

   A real fidelity gain must keep at least half its size, and a recipe that merely stays near the
   baseline must not get worse on text it could not tune against.

## Owner decisions

A maintainer can merge a recipe by hand and it is still evaluated and recorded. A maintainer can also
label a PR `excore-approve` or `excore-tier:<Ultra|XL|L|M|S>` to set the tier of a recipe the
frontier or reward stage rejected (or to re-tier an accepted one). Overrides apply to the reward
decision only: the gates, the holdout and the build/audit checks cannot be overridden. Overridden
entries are recorded with their true gain and `via: "...+override"`.

## The registry

Accepted recipes are recorded in `results/core_spectrum/spectrum.json` (the full history) and
charted in `results/core_spectrum/plots/`. Each entry records how it got in (`bot-merge`, `owner-merge`,
`+override`, or `seed`). The live frontier is the non-dominated subset; older
winners stay in the archive after they are beaten.
