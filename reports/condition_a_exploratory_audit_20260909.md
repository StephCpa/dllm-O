# Condition A Exploratory Analysis Audit

**Date:** 2026-09-09  
**Status:** Post-unblinding exploratory analysis; not pre-registered  
**Source analysis:** commit `e1bba72`

## Audit disposition

The three exploratory analyses support reframing decoding schedule as a control
over the precision-diversity frontier. They do not alter the pre-registered
Condition A verdict. The initial exploratory script required correction before
paper-facing use:

1. The first query of each task was counted twice when estimating mean observed
   wall time.
2. Spearman correlations used ordinal ranks rather than average ranks for ties.
3. Majority-vote correctness was inferred from the first correct sampled
   response rather than from the modal parsed answer's frozen grader labels.
4. Wall times came from instrumented runs at different times and shared-server
   loads. They are observational runtime proxies, not controlled deployment
   latency or a hardware-normalized compute budget.

The corrected analysis uses tie-aware Spearman, matched deterministic Monte
Carlo index samples, exact values at `k=1` and `k=n`, and a majority frozen-label
rule for modal parsed answers. It records two parsed-answer groups with
inconsistent frozen grader labels. Diversity remains parsed-string diversity;
semantically equivalent strings can be counted separately.

## Results that survive correction

- At a fixed nominal rollout budget, the block-size result remains: `B=1` has
  higher high-k coverage than `B=32`, while `B=32` has slightly higher pass@1.
- At `k=16`, `ao_b8` and `random_b32` have higher pass@k than `ao_b32`. This is a
  rollout-budget comparison and does not require the observational wall-time
  proxy.
- Parsed-answer diversity remains ordered approximately
  `EF > random > AR > AO` on both tasks.
- EF commits positions with substantially higher measured entropy than AO in
  the r00-r03 trace subsample, confirming that the intervention changed the
  intended ranking behavior.
- Tie-aware length-coverage correlations remain small for policy differences:
  `rho(delta length, delta CoverageAUC)=0.0973` on GSM8K and `-0.0109` on
  MATH-500.

## Permitted interpretation

The evidence supports the following exploratory interpretation:

> Decoding parallelism and position-ranking policy reshape the trade-off between
> marginal precision and multi-sample parsed-answer diversity. The tested
> entropy-first policy increases exploration but does not improve the frozen
> aggregate CoverageAUC64 endpoint over confidence-first decoding.

The absence of a simple monotone query-level length-coverage association does
not remove the triggered length guard and does not establish that length is
causally irrelevant.

## Claims not supported

- Wall-clock Pareto dominance under controlled deployment conditions.
- Semantic solution-path diversity; the current metric is parsed-answer-string
  diversity.
- Equivalence of EF and AO; the primary confidence interval still permits
  effects within its detection range.
- A causal claim that block-parallel commitment alone produces the coverage
  loss.
- A deployment recommendation to use EF with a reranker or verifier, because no
  reranking or verification experiment has been run.

## Paper-facing next step

Use rollout count `k` as the primary budget axis. Present observed wall time only
as an implementation diagnostic, or omit it until latency is measured under a
controlled benchmark. Build the main empirical figure from full pass@k curves,
parsed-answer diversity, and the pre-registered block-size and Condition A
contrasts. Label every analysis in this document as post-unblinding exploratory.
