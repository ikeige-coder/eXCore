# Precision space

The formats a miner may assign, who can produce them, and what they cost. Sizes are bits per
weight including block scales (from the GGUF block layouts).

| format | bits/weight | block | notes |
|---|---|---|---|
| `Q2_K` | 2.625 | 256 | smallest; not allowed on `lm_head` |
| `Q3_K` | 3.4375 | 256 | |
| `Q4_0` | 4.5 | 32 | legacy block format |
| `Q4_1` | 5.0 | 32 | legacy, with a per-block minimum |
| `Q4_K` | 4.5 | 256 | the usual workhorse |
| `Q5_0` | 5.5 | 32 | legacy |
| `Q5_1` | 6.0 | 32 | legacy, with a per-block minimum |
| `Q5_K` | 5.5 | 256 | |
| `Q6_K` | 6.5625 | 256 | |
| `Q8_0` | 8.5 | 32 | near-lossless |
| `F16`, `BF16` | 16 | 1 | no compression |

**Aliases.** `Q4_K_M`, `Q4_K_S`, `Q5_K_M`, `Q5_K_S`, `Q3_K_S/M/L` are llama.cpp *file types*: recipes
that mix tensor types. A manifest assigns one type per unit, so these spellings are accepted
and mean their base tensor type (`Q4_K_M` -> `Q4_K`).

**Legality.**
- `gdn`, `attn` and `mlp` units accept every format above.
- `lm_head` accepts `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `F16`, `BF16`.
- Every tensor's row length must be a multiple of the format's block size. All shipped units are
  multiples of 256, so every format fits; the check exists for other tracks.

**Quantizers** (`quantizer:` in a rule, or `default_quantizer:`):

| quantizer | formats | notes |
|---|---|---|
| `runtime` (default) | all | native llama.cpp encoders via one `llama-quantize` run with explicit `--tensor-type` overrides and `--pure` |
| `rtn` | `Q4_0 Q4_1 Q5_0 Q5_1 Q8_0` | pure-numpy round-to-nearest; deterministic, so the validator can replay it byte for byte |

After a build, the audit re-reads the file and checks that every unit tensor really is in the
format the manifest declared, because llama.cpp can silently fall back to another type.

**Adding a quantizer.** Subclass `RowQuantizer` (or `Quantizer`) in `excore/quantizers/<name>.py`
and call `register(...)`. That is code, so it goes through normal maintainer review rather than
the miner PR path. Once merged, manifests can name it.
