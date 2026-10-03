# E0: Alignment With the Prior-Work Decoder

**Date:** 2026-10-03 UTC  
**Status:** code inspection only. No generation was run, and no frozen endpoint
is affected.

## Question

The closest prior work, *The Flexibility Trap* (Ni et al., ICML 2026;
arXiv 2601.15165), reports Pass@k curves and examines block size, temperature,
and random orders. Before claiming that any result here is new, we need to know
whether this study's decoder is the same decoder as the public one, a
different setting, or a refinement of it.

## What was compared

- **Public decoder:** `LeapLabTHU/JustGRPO`, `utils/generate.py`, at the pinned
  commit `1a2fddb5c6655597e63081c0af5ebb718a849f39` (2026-07-06).
- **This study's decoder:** `src/dllm_order_transmission/decoding.py`,
  `generate_with_trace`.

## Findings

| Element | Public JustGRPO | This study | Same? |
|---|---|---|---|
| Candidate sampling | Gumbel-max at temperature `T` in float64: `exp(logits) / (-log u)^T`, argmax | Same transformation in float64, with an explicit seeded generator | Yes, except for the random stream |
| Block structure | `gen_length / block_length` blocks, `steps / num_blocks` steps each | Same | Yes |
| Tokens per step | `get_num_transfer_tokens`: one per step when steps equal length | Same function | Yes |
| Eligible set | Masked positions up to the end of the current block | Masked positions within the current block (earlier blocks are fully committed) | Equivalent |
| `low_confidence` (CF) score | `softmax(logits)` in the **model dtype** (bfloat16 on GPU), gathered at the sampled candidate | `softmax(logits.float())`, i.e. **float32**, gathered at the sampled candidate | **No** (see below) |
| `random` (RND) score | `torch.rand`, uniform over eligible positions | Same, with an explicit seeded generator | Yes |
| Committed token | The selected position's own sampled candidate | Same | Yes |
| Entropy-first (EF) | Not implemented at this commit | `high_entropy`: untempered entropy with stable ties to the lower index | Added here |
| Example and eval configuration | `block_length=32`, `steps = gen_length`, `low_confidence`; `eval.py` uses `temperature=0.0` | `B=32` CF at `T=0.6` (main) | Same schedule, different temperature |

### The CF precision difference

In the public decoder, the model runs in bfloat16 (`eval.py` loads it with
`torch_dtype=torch.bfloat16`), so `F.softmax(logits)` and the gathered
confidences are bfloat16. Between 0.99 and 1.0, bfloat16 represents only four
distinct values, and 0.999 rounds to 1.0. Many highly confident positions
therefore tie, and `torch.topk`'s tie order decides which one is committed.
This study computes the softmax in float32, so such ties are rare and the most
probable candidate wins.

The two decoders are logically identical, but CF position choice can differ
when several eligible candidates are near-certain. The direction and size of
any effect on Pass@k are unknown. The repository's equivalence tests compare
traced with untraced runs of *this* decoder, not this decoder with the public
one.

## Consequences for the manuscript

1. The paper should describe its decoder as a traced reimplementation of the
   JustGRPO decoder at commit `1a2fddb` that computes CF confidences in
   float32, not as the public decoder itself. Section 3 and Appendix A now say
   so.
2. Comparisons with the prior work's CF curves involve this precision
   difference, as well as a different temperature and possibly different
   prompts and lengths. They are close settings, not identical ones.
3. RND here is block-local random remasking (`B=32`). Whether the prior work's
   "fully random" order is block-local or global, and whether its
   negative-entropy order matches EF, must be read from its own definitions.
   The paper hosts were unreachable from the environment used for this
   inspection, so the authors must check this before submission.

## Recommended check before any new GPU run

On a few hundred stored masked states, run one CF step under bfloat16 and under
float32 softmax with the same candidates, and record how often the selected
position differs. This costs one forward pass per state. If the rate is
non-negligible, any new experiment (E1, E2) must state which precision it uses
and keep it fixed across arms.
