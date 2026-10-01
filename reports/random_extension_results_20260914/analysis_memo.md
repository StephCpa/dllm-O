# Random-B32 64-Rollout Extension: Analysis Memo

**Analysis date:** 2026-09-14  
**Operative runner/analyzer commit:** `91fa69d5741709a11eb38a33689f8887cfd42685`  
**Freeze SHA-256:** `5b84f29e1b57415c654bdf377dd775fa1bbe098679215a51c2e46a06e36d44d3`  
**Evidence status:** post-unblinding-selected targeted prospective follow-up

## Executive result

The frozen primary endpoint resolves in the positive direction. Across the
equal-weight GSM8K and MATH-500 panel,

`HighBudgetCoverage(random_b32) - HighBudgetCoverage(ao_b32)`

is **+0.03750**, with a 95% task-stratified query-bootstrap interval of
**[+0.00301, +0.07370]**. HighBudgetCoverage is the per-query mean of unbiased
Pass@16, Pass@32, and Pass@64 estimates computed from 64 rollouts.

Under the frozen taxonomy this is an `entirely_above_zero` result: random
position selection has a resolved high-budget coverage advantage over the
confidence-first AO policy in this two-task panel. The estimate is modest and
its lower confidence limit is close to zero. It should not be described as a
large effect or universal random-policy superiority.

## Audit closure

- All 9,600 new generation records validate against the frozen schema.
- All 9,600 traces exist and match their recorded SHA-256 values.
- All three deterministic shard markers validate and cover 3,200 work keys
  each.
- There are no missing or invalid records and no marker errors.
- Every one of the 6,400 records per task and policy used in the final
  comparison has parser status `ok`; the empty-response rate is zero.
- Validation-only was recorded at `2026-09-14T00:38:24Z` with
  `reads_random_extension_correctness=false`.
- The first formal outcome read was recorded at `2026-09-14T00:42:20Z` with
  `reads_random_extension_correctness=true`.
- An independent standard-library recomputation from the raw records exactly
  reproduced both task estimates and the pooled primary point estimate.

## Primary result by task

| Task | Random - AO HighBudgetCoverage | 95% CI | Reading |
|---|---:|---:|---|
| GSM8K | +0.03676 | [-0.00234, +0.08313] | Directionally positive |
| MATH-500 | +0.03825 | [-0.01733, +0.09607] | Directionally positive |
| Equal-weight pooled | **+0.03750** | **[+0.00301, +0.07370]** | Frozen positive verdict |

The point estimates are unusually consistent across tasks. Neither task alone
has an interval excluding zero, so the permitted claim is panel-level. The
result does not establish independent task-level replication.

## Where the AO-to-random crossover occurs

| k | Random - AO Pass@k | 95% CI | Status |
|---:|---:|---:|---|
| 1 | -0.09359 | [-0.12094, -0.06672] | Resolved AO advantage |
| 2 | -0.04134 | [-0.06965, -0.01333] | Resolved AO advantage |
| 4 | -0.00375 | [-0.03287, +0.02599] | Unresolved |
| 8 | +0.02095 | [-0.01026, +0.05184] | Unresolved |
| 16 | +0.03405 | [+0.00317, +0.06676] | Resolved random advantage |
| 32 | +0.03847 | [+0.00357, +0.07490] | Resolved random advantage |
| 64 | +0.04000 | [-0.00012, +0.08500] | Directional; unresolved |

The extension fills the asymmetric frontier that motivated it. Confidence-first
AO is better for one or two samples, while random selection is better at the
resolved k=16 and k=32 points. At k=64, finite task ceilings reduce resolution:
the point estimate remains positive but the interval narrowly includes zero.

This is a budget-dependent policy reversal, not a single policy ranking. The
low-budget grid `{2,4,8,16}` is unresolved at +0.00248
[-0.02534, +0.03070], and the broader grid `{4,8,16,32,64}` is also unresolved
at +0.02594 [-0.00382, +0.05664]. Only the frozen high-budget functional has
the declared confirmatory interpretation within this follow-up.

## Post-outcome multiplicity audit

A separate exploratory audit was designed after these outcomes were read. It
applied Bonferroni intervals and nonstudentized maximum-deviation simultaneous
bands to the seven random-minus-AO per-k contrasts using 50,000
task-stratified paired-query bootstrap replicates (`seed=20260914`). Under the
seven-endpoint maximum-deviation family, only the `k=1` pointwise contrast
remains resolved; the high-budget `k=16` and `k=32` intervals cross zero.
Bonferroni intervals retain the negative `k=1` and `k=2` contrasts but not the
positive high-budget points. The unadjusted per-k table must therefore not be
presented as familywise-controlled evidence at both ends.

For an explicitly post-outcome two-region diagnostic, the low-budget mean over
`k={1,2}` is -0.06747 with a two-endpoint simultaneous interval of
[-0.10245, -0.03248], while the high-budget mean over `k={16,32}` is +0.03626
[+0.00127, +0.07124]. The joint sign reversal remains resolved under this
reduced family. This is a robustness description, not a revision or
strengthening of the frozen primary result.

## Comparison with sequential AR

Random is worse than sequential AR at k=1, 2, 4, and 8, with all four intervals
below zero. At k=16, 32, and 64 the differences are unresolved and remain
slightly negative. The low-budget grid difference is -0.03331
[-0.05191, -0.01545]; the full grid is -0.01827
[-0.03874, +0.00128].

Therefore the extension does not show that random block-parallel decoding
beats autoregressive-style sequential decoding in coverage. Rather, it shows
that random selection can recover part of the coverage lost by confidence-first
block-parallel commitment at sufficiently large sampling budgets. Any practical
comparison with AR must additionally include latency or compute cost.

## Comparison with entropy-first EF

Random exceeds EF at k=1, 2, 4, 8, and 16, with intervals entirely above zero.
The k=32 difference is unresolved; the k=64 interval reaches zero. Both the
low-budget and full-grid summaries favor random:

- low grid: +0.05208 [+0.03473, +0.06897];
- full grid: +0.03580 [+0.01456, +0.05621].

EF nevertheless has the greatest parsed-answer diversity. At k=64, its mean
answer entropy is 1.90 on GSM8K and 2.48 on MATH-500, compared with 1.18 and
2.10 for random. Random produces fewer unique answers than EF but more than AO.
Thus maximizing commitment-time uncertainty increases answer-string diversity
without producing the best correct-answer coverage. Random selection occupies
a better precision-diversity position than EF in this experiment.

## Solved-set evidence at k=64

Against AO, random finds four GSM8K queries and ten MATH-500 queries that AO
never solves, while losing one and five queries respectively. The net solved-set
gains are therefore +3 and +5 queries, matching the task-level Pass@64 point
differences of +0.03 and +0.05.

Against AR, random has a net loss of one solved query on GSM8K and no net change
on MATH-500. Against EF, it has no net change on GSM8K and a net gain of six
queries on MATH-500.

## What the result changes

The earlier frozen aggregate AO-random comparison remains a null and must not
be rewritten. This extension instead demonstrates why that aggregate was
insensitive: the policy effect changes sign with k. A low-budget precision
advantage and a high-budget coverage advantage can cancel when averaged over a
mixed grid.

The strongest supported scientific statement is:

> Position-selection policy changes the shape of the test-time coverage
> frontier. Confidence-first block-parallel decoding improves low-budget
> precision, whereas random position selection yields higher pooled coverage
> at k=16 and k=32 and a positive frozen high-budget summary; sequential AR
> remains unmatched in high-budget coverage.

This is more specific and more useful than either "position selection does not
matter" or "more diversity is better." It identifies sampling budget as an
interaction variable that determines which scheduling policy is favorable.

## Evidence boundaries

1. The extension was selected after observing the first-16 AO-random crossing.
   It was frozen before the 9,600 new outcomes but combines the 48 new rollouts
   with the 16 previously observed rollouts. It is not an independent
   preregistered replication and is not part of the original Gate-1 chain.
2. The positive primary interval is close to zero. Per-task intervals and the
   broader full-grid summary straddle zero.
3. The panel contains one model, two mathematical reasoning tasks, one decoder,
   and one temperature. No cross-model or broad task-family generalization is
   established.
4. Pass@k is oracle coverage, not a deployable selection rule. Realized utility
   requires a verifier or reranker capable of identifying a correct candidate.
5. Parsed-answer entropy measures answer-string diversity, not semantic
   reasoning diversity.
6. The canonical per-k secondary intervals are descriptive and unadjusted. A
   separate post-outcome multiplicity audit reports seven-endpoint familywise
   intervals and a two-region simultaneous diagnostic; neither changes the
   frozen primary verdict.

## Experimental stopping decision

The frozen protocol states that the experimental program stops after this
extension regardless of direction. The primary question is now answered and
the missing random high-budget frontier is filled. No repair arm, additional
ranking policy, or new k-grid should be added in response to these outcomes.
The remaining work is artifact backup, manuscript integration, and figure/table
generation from the audited JSON.
