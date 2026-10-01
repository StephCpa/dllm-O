# Post-Unblinding Robustness Analyses

**Status:** exploratory, post-unblinding, CPU-only. These analyses do not
alter any frozen endpoint, confidence interval, or decision rule.

## 1. Phase 2 endpoint leverage

| Quantity | Estimate |
|---|---:|
| Mean log2 P2 slope | -0.009646 |
| B=1-to-parallel step component | -0.009193 |
| Within-parallel endpoint component | -0.000454 |
| Absolute share attributed to step | 95.3% |

At the aggregate means, `beta_log2 = 3*(mean(y8,y16,y32)-y1)/14 + (y32-y8)/14`. The decomposition
shows that the resolved P2 slope is overwhelmingly leverage from the discrete
B=1-to-block-parallel transition, not evidence of a graded dose response within
the parallel regime. P1 and P2 are therefore dependent views of the same jump,
not independent confirmations.

Query-level P1/P2 Pearson correlation: 0.9840;
tie-aware Spearman: 0.9427.

### Zero-information and active-subset audit

- Zero-success cells: 132/800.
- Queries with all four cells at zero: 21/200.
- Queries with both frozen P1 endpoint arms at zero: 21/200.
- Outcome-conditioned active-endpoint P1: -0.04972 [-0.07871, -0.02148].

The active-subset estimate is descriptive selection on observed outcomes and
must never replace the frozen intention-to-treat P1 estimate.

## 2. Sampling-budget-grid sensitivity

All `full64` rows below use the same 64 EF and AO rollouts. They isolate the
choice of k-grid from the first-16-versus-64-rollout realization difference.

| Metric | EF - AO, pooled 95% bootstrap CI |
|---|---:|
| `primary_full64_k4_8_16_32_64` | -0.00986 [-0.04226, +0.02253] |
| `low_budget_full64_k2_4_8_16` | -0.04960 [-0.08200, -0.01692] |
| `high_budget_full64_k16_32_64` | +0.01009 [-0.02645, +0.04746] |
| `frozen_first16_k2_4_8_16` | -0.05616 [-0.09895, -0.01404] |

The verdict is a functional of the sampling-budget distribution, not an
intrinsic scalar property of a policy. A first-16 metric additionally mixes
budget weighting with finite-rollout realization and must not be described as
a pure grid effect.

### Per-k EF - AO contrasts from the same 64 rollouts

| k | Pooled | GSM8K point | MATH-500 point |
|---:|---:|---:|---:|
| 1 | -0.19898 [-0.23266, -0.16484] | -0.23438 | -0.16359 |
| 2 | -0.12440 [-0.16027, -0.09027] | -0.12786 | -0.12095 |
| 4 | -0.06040 [-0.09537, -0.02470] | -0.05151 | -0.06928 |
| 8 | -0.01918 [-0.05479, +0.01585] | -0.01314 | -0.02523 |
| 16 | +0.00559 [-0.02933, +0.04137] | +0.01101 | +0.00016 |
| 32 | +0.01469 [-0.02155, +0.05269] | +0.02781 | +0.00156 |
| 64 | +0.01000 [-0.03500, +0.05500] | +0.03000 | -0.01000 |

## 3. AO versus random: matched first-16 decomposition

`random_b32` has only 16 rollouts. The matched analysis therefore uses
the common first 16 AO and random rollouts; no matched-64 result exists.

| k | AO - random, matched first 16 | GSM8K point | MATH-500 point |
|---:|---:|---:|---:|
| 1 | +0.09969 [+0.06906, +0.13219] | +0.09750 | +0.10187 |
| 2 | +0.04363 [+0.01046, +0.07871] | +0.02792 | +0.05933 |
| 4 | +0.00117 [-0.03528, +0.03784] | -0.01274 | +0.01509 |
| 8 | -0.02837 [-0.06891, +0.01145] | -0.03672 | -0.02002 |
| 16 | -0.04000 [-0.08500, +0.00500] | -0.04000 | -0.04000 |

Matched low-grid AO-random aggregate: -0.00589 [-0.04173, +0.02881].
The JSON also records an AO-64-versus-random-16 estimator-precision
sensitivity analysis. It is not matched-64 evidence and is excluded from
the headline table.

### Matched first-16 checks against random

| k | EF - random | AR - random |
|---:|---:|---:|
| 1 | -0.10656 [-0.12688, -0.08594] | +0.08344 [+0.06094, +0.10720] |
| 2 | -0.08696 [-0.10971, -0.06550] | +0.05896 [+0.03508, +0.08392] |
| 4 | -0.06277 [-0.08995, -0.03652] | +0.03930 [+0.01217, +0.06782] |
| 8 | -0.05350 [-0.08939, -0.01916] | +0.01952 [-0.01136, +0.05133] |
| 16 | -0.04500 [-0.09000, +0.00000] | +0.00500 [-0.03500, +0.04500] |

These comparisons use the same 16 rollout indices in each policy. They
replace asymmetric point-estimate comparisons when assessing dominance.

## 4. Length-conditioned pass@1 diagnostics

Raw EF-AO pass@1: -0.19898 [-0.23266, -0.16469].
A task-fixed-intercept linear adjustment evaluated at zero length difference: -0.20703 [-0.25069, -0.16343].

This adjustment is not causal: decoded length is post-treatment, and zero
length difference may be outside common support. Same-seed near-length subsets
provide a complementary descriptive check:

| Pair filter | EF - AO correctness | Eligible pairs (GSM8K / MATH-500) | Queries |
|---|---:|---:|---:|
| `absolute_length_difference_le_2` | -0.16142 [-0.20479, -0.11995] | 765 / 540 | 99 / 97 |
| `absolute_length_difference_le_5` | -0.17164 [-0.20925, -0.13602] | 1609 / 1207 | 100 / 100 |
| `absolute_length_difference_le_10` | -0.17971 [-0.21574, -0.14504] | 2759 / 2143 | 100 / 100 |
| `absolute_length_difference_le_20` | -0.18551 [-0.21955, -0.15146] | 4270 / 3656 | 100 / 100 |

## Interpretation boundary

1. The Phase 2 result supports a sequential-versus-block-parallel threshold
   effect. It does not establish a smooth freedom-dose law.
2. Coverage summaries must declare their k-grid or sampling-budget weighting;
   crossing frontiers cannot be represented faithfully by one universal AUC.
3. Length-conditioned analyses are exploratory mediator-conditioned
   descriptions, not repairs to the frozen Condition A endpoint.
4. AO-random, EF-random, and AR-random contrasts share queries and the
   random control; they are complementary instances, not independent replications.
5. No new GPU inference was used.
