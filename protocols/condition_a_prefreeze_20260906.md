# Condition A Entropy-First Intervention: Pre-Freeze Protocol

**Date:** 2026-09-06

**Status:** Implementation candidate. This document does not authorize inference.

## 1. Trigger

The prospectively frozen Option B gate requires Condition A because both primary quantities resolved in the predicted negative direction:

- `CoverageAUC(B=32) - CoverageAUC(B=1) = -0.04421`, 95% CI `[-0.06974, -0.01977]`;
- pooled `log2(B)` trend `-0.00965`, 95% CI `[-0.01474, -0.00489]`, with the ordinal sensitivity analysis also negative.

The gate label is `continuous freedom-cost gradient`, but the empirical shape is concentrated at `B=1 -> B=8`; the two larger-B adjacent contrasts are unresolved. Condition A therefore tests whether the resolved sequential-to-parallel coverage cost depends on *which* eligible positions are committed first.

## 2. Intervention

`ef_b32` uses the same model, prompts, block size, denoising schedule, temperature, generation length, and policy-independent rollout seeds as `ao_b32`. The only manipulated variable is the within-block position-ranking rule.

EF ranks the full predictive distribution by entropy, whereas AO ranks the sampled candidate token by its model probability. The experiment is therefore a controlled comparison of position-ranking policies, not an algebraic sign reversal of one shared scalar score. Claims must use “entropy-first intervention” or “ranking-policy intervention,” not “exact inverse of AO confidence.”

At every denoising step:

1. Compute the clean full-vocabulary categorical entropy at every currently masked position in the active block.
2. Rank eligible positions by entropy in descending order.
3. Resolve the number of positions prescribed by the unchanged transfer schedule.
4. Recompute the ranking after the model state changes at the next step.
5. Resolve exact entropy ties by the lower absolute token position, implemented by a stable descending sort.

Positions outside the active block are never eligible. Ranking all masked positions or freezing a global order at generation start is not permitted.

## 3. Conditions and budgets

| Condition | Rollouts/query | Source | Role |
|---|---:|---|---|
| `ef_b32` | 64 | new Condition A inference | intervention |
| `ao_b32` | 64 | reused Phase 1 | primary confidence-first control |
| `random_b32` | 16 | reused Phase 1 screening | secondary random-order reference |
| `ar_b1` | 64 | reused Phase 1 | contextual sequential reference only |

The new formal workload is 2 tasks × 100 held-out queries × 64 rollouts = **12,800 generations**. The fixed launch design uses eight deterministic shards of 1,600 generations each. Formal inference uses the same task/query/rollout seed key as Phase 1 so the EF–AO contrast uses common random numbers.

`random_b32` is not compared with EF at 64 rollouts because no 64-rollout random artifact exists. Adding such a comparison after seeing Condition A outcomes is forbidden. The random reference is evaluated only on the matched first 16 rollouts.

## 4. Endpoints

### Primary

For every held-out query, compute:

`CoverageAUC64(ef_b32) - CoverageAUC64(ao_b32)`,

where CoverageAUC64 is the arithmetic mean of unbiased Pass@k over `k in {4, 8, 16, 32, 64}` from 64 rollouts. Report task-specific means and the task-stratified pooled mean with the existing frozen percentile bootstrap (`seed=20260820`, 10,000 replicates, 95% interval).

Pre-registered readings:

- CI entirely above zero: entropy-first recovers coverage relative to confidence-first.
- CI straddles zero: no resolved entropy-first benefit; confidence-guided position selection is not identified as the damaging component.
- CI entirely below zero: entropy-first reduces coverage; confidence-guided selection is protective relative to the intervention.

### Secondary

Using only rollout indices 0–15, compute ScreeningCoverageAUC16 as mean unbiased Pass@k over `k in {2, 4, 8, 16}` and report:

- `ef_b32 - random_b32`;
- `ef_b32 - ao_b32`.

Also report all per-k pass rates, task-specific contrasts, solved-set membership, parser-status counts, empty-response rates, and the precision-diversity profile. No secondary result can replace or redefine the primary endpoint.

## 5. Length and output-quality guard

The decoder always commits exactly 256 generated token positions and has no early-stopping rule, so raw generation-cap exposure is identical by construction. The empirically variable quantity is decoded response length.

For each query, compare mean whitespace-separated response length for:

- EF(64) minus AO(64);
- EF(first 16) minus random(16).

The guard triggers if either paired pooled 95% bootstrap interval excludes zero. A trigger requires reporting and qualifying the coverage interpretation; it never permits outcome deletion, post-hoc length matching, or endpoint replacement. Parser failures and empty responses remain in the analysis as incorrect outcomes.

## 6. Artifact requirements

Every new generation must record:

- the Condition A freeze hash and exact clean repository commit;
- the frozen model, dataset, prompt, policy, seed-key, and rollout provenance;
- a 256-event pre-mutation trace;
- per eligible position, top-64 token ids/logits and the full log-normalizer;
- entropy, margins, selected positions/tokens, and state checksums;
- grading output, parser status, wall time, and peak memory.

Missing records, invalid schemas, trace-hash mismatches, wrong commits, wrong freeze hashes, or incomplete shard markers block analysis. Interrupted work may resume only by the identical work key and seed; completed valid records are skipped rather than overwritten.

## 7. Smoke acceptance criteria

Before formal launch, one calibration query from each task is run through `ef_b32` with full-distribution tracing. Formal launch requires all of the following:

- both result records pass the Condition A schema;
- each trace has exactly 256 events and reconstructs the generated completion;
- every selected position is currently masked and inside the active block;
- each selected position has maximal eligible entropy at that step, with exact ties resolved to lower position;
- traced/full-distribution and untraced runs are token-identical under the same seed;
- no non-finite entropy, logit, normalizer, or margin values occur;
- the freeze hash, runner commit, model revision, prompt hash, and trace hash agree.

Any failure stops launch and requires a documented pre-outcome code repair followed by a new clean commit and a newly emitted freeze.

## 8. Freeze and unblinding order

1. Complete implementation and tests.
2. Commit the runner, analyzer, schema, protocol, and tests.
3. Emit a machine-readable freeze bound to that commit and the canonical Option B freeze/report/repair hashes.
4. Run and audit smoke artifacts.
5. Run eight formal shards.
6. Run `validate-only`, which records no Condition A correctness read.
7. Only after 12,800/12,800 artifacts and 8/8 markers validate, invoke `analyze` and append the first Condition A correctness-read timestamp.

Until step 3 is complete, no Condition A GPU inference is authorized.
