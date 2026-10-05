# Miner guide

You compete by writing a **manifest**: a short YAML file that says which GGUF format each
part of the model should use. You never run a validator, quantize a 35B model, or own a GPU.

## 1. Set up

```bash
pip install -e .
excore check experiments/base_variants/V1_mlp_focused.yaml
```

`excore check` is all you need to iterate. It validates your recipe with the same parser the
validator uses and estimates the model size, so mistakes show up in seconds, not hours.

## 2. Write a manifest

```yaml
schema: excore/manifest@1
track: CPU-35B
name: attn-q8-mlp-q4            # must equal the file name: manifests/attn-q8-mlp-q4.yaml
description: Keep attention precise, squeeze the MLPs.
default: Q5_K                   # applies to every unit no rule overrides
rules:                          # applied in order; later rules override earlier ones
  - match: "L*.mlp"             # glob over unit names
    format: Q4_K
  - match: "L*.mlp"
    layers: "10-62"             # optional layer filter: "0-3,8,10-12", 7, or [1, 2]
    format: Q3_K
  - match: "L*.attn.*"
    format: Q8_0
  - match: "lm_head"
    format: Q6_K
```

The 35B model has **358 searchable units**. Names:

| unit | meaning | count |
|---|---|---|
| `L<n>.gdn.qkv`, `L<n>.gdn.z`, `L<n>.gdn.out` | Gated DeltaNet projections (layers where `(n+1) % 4 != 0`) | 189 |
| `L<n>.attn.q`, `.k`, `.v`, `.o` | full-attention projections (layers 3, 7, 11, ..., 83) | 84 |
| `L<n>.mlp` | gate, up and down projections of one MLP block, together | 84 |
| `lm_head` | output projection | 1 |

Everything else (embeddings, norms, state parameters) is frozen and byte-identical to the source.

Optional per-rule `quantizer:` picks the encoder (`runtime` = llama.cpp, the default; `rtn` =
round-to-nearest, legacy formats only). See [precision_space.md](precision_space.md).

Strictness is deliberate. The parser rejects unknown keys, duplicate keys, YAML anchors/aliases,
rules that match nothing (usually a typo), and illegal format/unit/quantizer combinations. It
reports **all** problems at once.

## 3. Submit

Open a pull request that adds **exactly one new file**, `manifests/<name>.yaml`.

- Anything else in the PR (code, workflows, plugins, edits to existing manifests) is refused and
  needs maintainer review. Recipes are immutable once submitted.
- `name` must match the file name, and the name must be new. Two recipes that compile to the same
  per-unit formats are duplicates even if the files look different.
- Drafts are ignored. Mark the PR ready when you want it evaluated.

The bot reads your file as text, builds the model on the validator, benchmarks it, and comments
with the result. If your recipe advances the frontier and passes every gate, the bot merges it and
you earn a reward tier. If not, the comment says which stage stopped it and why.

Maintainers can also merge a recipe by hand. It is then evaluated and recorded like any other, so it
earns the same tier. A maintainer may additionally approve a recipe the frontier rejected by
labelling the PR; that sets a tier but never bypasses the gates or the holdout.

## 4. What it is judged on

Four numbers, all measured on CPU: **accuracy drift** (RP-KL against the unquantized model),
**decode tokens/s**, **prefill tokens/s** and **peak RAM**. You do not need to win all four. A
recipe earns a place if no existing entry beats it on everything at once. Details and a worked
example are in [core_frontier.md](core_frontier.md).

Before it can be placed, a recipe must pass the gates: stable benchmark, under the RAM ceiling,
drift within limits, long-context retrieval at 8K and 16K no worse than the baseline, and a
private holdout check that your gain was not tuned to the public corpus.

## 5. Tips

- Start from the baseline (`experiments/base_variants/V0_baseline.yaml`) and change one thing.
- The MLP blocks hold about 70% of the parameters, so they are where memory savings are.
- Peak RAM above **24 GiB** is rejected. A uniform `Q5_K` just fits, `Q6_K` does not.
- Attention and `lm_head` are small but sensitive. Squeezing them rarely pays.
- Long-context attention is guarded. A recipe that saves RAM by wrecking 16K retrieval is rejected.
- Gains below the measurement noise earn nothing by design: a candidate must beat an incumbent by
  more than the benchmark's own margin (5% speed, 0.25 GiB RAM, two standard errors of drift).
- `excore check` prints an **estimate**. The real number comes from the validator.
