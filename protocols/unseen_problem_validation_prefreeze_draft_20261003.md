# E1: Validation on Unseen Problems — Pre-Freeze Draft

**Protocol version:** `unseen-problem-validation-freeze-v1` (proposed)  
**Status:** draft for author review (2026-10-03 UTC). Not frozen and not
authorized for GPU launch. Items marked **[author decision]** or **[bind at
freeze]** must be resolved first.  
**Selection status:** a prospective validation of a result that was itself
selected after outcomes were read. Every outcome in this study is new: new
problems and newly generated rollouts in every arm.

## 1. Question

The random-ranking extension found a small pooled high-budget advantage of RND
over CF at `T=0.6`, `+0.038 [+0.003, +0.074]`, on the same 200 problems whose
first 16 RND rollouts had motivated it. This study asks: **does the high-budget
RND--CF difference hold on problems the study has never used?** The answer is
the main defence against the reviewer objection that the result rests on a
post-hoc-selected panel.

## 2. Problems

- **Pool:** problems never used in any stage. The Phase 1 manifest uses 150
  GSM8K and 150 MATH-500 rows, and the 50 Phase 0 GSM8K rows are a subset of
  them. That leaves 1,169 GSM8K and 350 MATH-500 rows.
- **Draw:** a seeded random sample, stratified by task (and by MATH-500 level
  within MATH-500), emitted as a new split manifest. Its SHA-256 is bound into
  the freeze **[bind at freeze]**. No outcome of any policy on these rows may be
  inspected before the freeze.
- **Size [author decision]:** see the power table in Section 6.

## 3. Arms

Same model, decoder commit, prompts, parser, and grader as the original study.
CF confidences are computed in float32, as in all existing cells (see
`reports/prior_work_decoder_alignment_20261003.md`). Settings are 256 positions,
256 steps, `T=0.6`, CFG 0, one committed token per NFE, and 64 rollouts per
problem and arm, with the existing seed derivation.

| Arm | Policy | Required? |
|---|---|---|
| CF | `ao_b32` (confidence-first, `B=32`) | yes |
| RND | `random_b32` | yes |
| Sequential | `ar_b1` | recommended (enables H2 below) |

## 4. Frozen hypotheses and verdicts

**H1 (primary).** `HB(RND) - HB(CF)`, where `HB = mean(Pass@16, Pass@32,
Pass@64)` estimated from 64 rollouts. This is the original random-extension
functional, computed as the equal-weight mean of the task means. Uncertainty is
a 10,000-replicate task-stratified paired-query percentile bootstrap, seed
**[bind at freeze]**.

- `replicated`: the 95% interval lies entirely above zero.
- `contradicted`: the 95% interval lies entirely below zero.
- `not_resolved`: every other outcome. This does not mean the policies are
  equivalent.

Whether the new interval excludes the original point estimate (+0.038) is also
reported, as evidence about winner's-curse shrinkage.

**H2 (key secondary, Sequential arm only).** The original block-size primary,
`CoverageAUC_K64(ao_b32) - CoverageAUC_K64(ar_b1)`, with the same verdict
taxonomy (`replicated` means entirely below zero). **[Author decision]:** either
report H1 and H2 separately, or test them in a fixed sequence (H1, then H2)
for a familywise claim.

**Secondary (descriptive).** Per-task estimates; per-k RND--CF contrasts with
pointwise and seven-endpoint simultaneous bands; the E3 problem-level
decomposition (`scripts/problem_level_reanalysis.py`) on the new panel;
subset-averaged majority voting.

## 5. Launch gates

1. Emit the new split manifest from the unused pool without reading any
   outcome. Bind its hash.
2. Commit the runner, analyzer, schema, and tests for a new run kind. The
   inference checkout must be clean.
3. Dry-run all shards and confirm the planned cardinality.
4. Run a smoke on one problem per task, validated without reading correctness.
5. Emit the machine freeze, then record a validation-only invocation before the
   first correctness read.

## 6. Power and size **[author decision]**

The original high-budget interval implies a query-level standard error of
about 0.018 with 200 problems (100 per task). The standard error scales as
`1/sqrt(problems)`. Approximate two-sided power for H1:

| New problems | Generations (CF + RND / + Sequential) | Power at +0.038 | at +0.030 | at +0.020 |
|---:|---:|---:|---:|---:|
| 200 (100 / 100) | 25,600 / 38,400 | 0.56 | 0.38 | 0.20 |
| 400 (200 / 200) | 51,200 / 76,800 | 0.85 | 0.65 | 0.35 |
| 700 (350 / 350) | 89,600 / 134,400 | 0.97 | 0.88 | 0.55 |

The 700-problem row uses every unused MATH-500 problem. The observed +0.038 was
selected after a crossing was seen and is likely an overestimate. Around 80%
power needs about 570 problems at +0.03 and about 1,280 at +0.02, the latter
more than the MATH-500 pool allows.

Recommendation: at least 400 problems; 700 with CF and RND only, if the
goal is a decisive answer on H1. Between-query variance is 20–50 times the
binomial rollout reference, so adding problems raises power far more than
adding rollouts.

## 7. Resource estimate

At 26–32 seconds per generation:

| Design | Generations | GPU-hours |
|---|---:|---:|
| 400 problems, three arms | 76,800 | 555–683 |
| 700 problems, two arms | 89,600 | 647–796 |

Smoke timing replaces these estimates.

## 8. Scope boundary

A `replicated` verdict supports the high-budget RND--CF difference on new
problems from the same two task distributions, for this model, decoder, and
temperature. It does not show that RND beats sequential decoding, and it does
not explain why the difference arises; E2 and E3 address the mechanism.
