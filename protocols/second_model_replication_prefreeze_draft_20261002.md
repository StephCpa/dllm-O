# Second-Model Replication of the Block-Size Primary: Pre-Freeze Draft

**Protocol version:** `second-model-replication-freeze-v1` (proposed)  
**Priority (2026-10-03 UTC review):** P1. The validation on unseen problems
(E1) and the CF-resample control (E2) come first; see
`unseen_problem_validation_prefreeze_draft_20261003.md` and
`cf_resample_control_prefreeze_draft_20261003.md`. A second model alone does not
resolve the novelty concern.  
**Status:** draft for author review. Not frozen and not authorized for GPU launch.
Every item marked **[author decision]** must be resolved, and every item marked
**[bind at freeze]** must be filled in by the analyzer's `emit-freeze` mode, before
any new generation is run.  
**Selection status:** post-review prospective follow-up. It was proposed after all
existing outcomes were read. It is prospective only for the new model's outcomes
and cannot revise any earlier frozen verdict.

## 1. Question

The frozen Phase 2 primary found, for `LLaDA-8B-Instruct` under matched 256 NFEs,

`CoverageAUC_K64(ao_b32) - CoverageAUC_K64(ar_b1) = -0.044 [-0.070, -0.020]`.

All current evidence comes from one checkpoint. This study asks a single question:
**does the same frozen contrast resolve below zero on a second masked diffusion
language model?** It does not re-test the RND--CF crossing, the temperature
extension, or entropy-first selection.

## 2. Model choice **[author decision]**

| Option | Model | Strength | Cost and risk |
|---|---|---|---|
| A (preferred for external validity) | Dream 7B Instruct (`Dream-org/Dream-v0-Instruct-7B`; verify the ID) | A different model family, adapted from an autoregressive model. It is the stronger test of generality. | The JustGRPO decoder is LLaDA-specific. The schedule semantics must be ported to Dream's output convention and pass an equivalence gate (Section 6) before launch. |
| B (lower risk) | LLaDA 1.5 (`GSAI-ML/LLaDA-1.5`; verify the ID) | Same architecture and tokenizer, so the existing decoder applies with minimal change. | The same base model with further post-training, so it is weaker evidence of generality. |

The revision hash of the chosen model is **[bind at freeze]**. A failed
equivalence gate for option A must not be followed by a silent switch to option B.
Any switch is recorded as a protocol amendment before outcomes are read.

## 3. Generation design

Everything matches the original Phase 2 primary except the model:

- Data: the existing frozen 100 GSM8K and 100 MATH-500 held-out queries
  (`phase1_split_manifest_v1.json`). Reusing the same queries separates a model
  effect from a query-panel effect.
- Cells: sequential `B=1` (`ar_b1`) and confidence-first `B=32` (`ao_b32`).
- Fixed settings: 256 generated positions, 256 denoising steps, one committed
  token per NFE, temperature 0.6, CFG 0, and the existing parser and grader. The
  chat template is the chosen model's own template, recorded in the freeze.
- Rollouts: 64 per query and cell, with the same policy-independent seed keys as
  the original study (`seeding.rollout_seed_key`). Within the new model this
  pairs `ar_b1` and `ao_b32` by seed; across models the shared keys carry no
  pairing value and only keep provenance simple.
- Workload: `2 tasks x 100 queries x 2 cells x 64 = 25,600 generations`.

Optional secondary arm **[author decision before freeze]**: add `ao_b8`
(+12,800 generations) to test on the second model whether the drop is
concentrated at the first widening.

Cost fallback **[author decision before freeze]**: 32 rollouts (12,800
generations) with the grid `{4,8,16,32}`. This is a different functional from the
original `K64` endpoint and must then be named as such in the paper.

## 4. Frozen primary endpoint and verdict

`P1' = CoverageAUC_K64(ao_b32) - CoverageAUC_K64(ar_b1)` on the second model,
computed per query from the unbiased Pass@k estimator and aggregated as the
equal-weight mean of the GSM8K and MATH-500 task means.

Uncertainty: a 10,000-replicate task-stratified paired-query percentile bootstrap,
seed `20261003`.

- `replicated`: the 95% interval lies entirely below zero.
- `opposite_direction`: the 95% interval lies entirely above zero.
- `not_resolved`: every other outcome, reported with its estimate and interval.
  This label does not mean equivalence or absence of an effect.

## 5. Secondary endpoints

All are reported regardless of direction and cannot change the primary verdict:

1. Per-task `P1'` estimates and intervals.
2. Per-k contrasts for `k in {1,2,4,8,16,32,64}`, including the `k=1` precision
   contrast.
3. Zero-success query counts per cell and the share of `P1'` carried by queries
   that become unsolved under `B=32` (support loss).
4. A fragility description as in `scripts/block_size_fragility_audit.py`.
5. Parser status, empty responses, decoded length, and artifact completeness.

## 6. Launch gates

1. Implement and commit the runner, analyzer, artifact schema, and tests for the
   new run kind (`second-model-replication`). The inference checkout must be clean
   at that commit.
2. **Equivalence gate (option A only).** On a smoke set of one query per task and
   a few rollouts, the ported decoder must reproduce the model's reference sampler
   under matched settings. It must also be verified that `B=1` commits strictly
   left to right and that every schedule commits exactly one token per NFE. The
   gate is validated without reading correctness.
3. Run the full test suite; dry-run all shards and confirm 25,600 unique work keys.
4. Emit the machine freeze bound to the model revision, runner commit, split
   manifest, and shard hashes.
5. Record a validation-only invocation before the first correctness read.

Parser failures and empty responses are retained as incorrect. No query or
rollout is removed after outcomes are read.

## 7. Power note

The original primary interval implies a query-level standard error of about
0.0127. If the second model has the same query-level variance, the approximate
power of the frozen two-sided test is:

| true effect | power |
|---:|---:|
| −0.044 (same as LLaDA) | 0.93 |
| −0.033 (75%) | 0.74 |
| −0.022 (50%) | 0.41 |

These are normal approximations, not a frozen planning calculation. Between-query
variance is 20–50 times the binomial rollout reference in the existing data, so
more queries would raise power far more than more rollouts. A `not_resolved`
verdict is therefore uninformative about effects much smaller than the original.

## 8. Resource estimate

At the 26–32 seconds per generation observed for LLaDA-8B, 25,600 generations need
about 185–228 GPU-hours (about 31–38 hours on six GPUs). Smoke timing on the chosen
model replaces this estimate.

## 9. Scope boundary

A `replicated` verdict supports the claim that the matched-NFE coverage cost of
widening the eligible set is not specific to one checkpoint. It does not establish
a law across all dLLMs, decoders, tasks, or temperatures.
