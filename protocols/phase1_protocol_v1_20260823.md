# Prospectively Frozen Phase 1 Protocol

**Project:** Order Transmission Across Diffusion Decoding and Private Aggregation  
**Protocol version:** Phase 1 v1  
**Date:** 2026-08-23  
**Status:** Design frozen before Phase 1 inference; the calibration-dependent BCI formula requires a second signed freeze before held-out analysis  
**Relationship to Phase 0:** Phase 0 is calibration evidence only and is not part of the held-out test

## 1. Research Contract

The workload is repeated mathematical-reasoning generation with a masked-diffusion language model under different token-commitment schedules. The default assumption under examination is that greater token-order flexibility preserves or expands useful reasoning alternatives. Phase 0 instead showed that later commitment strongly increases local entropy closure, while AR and adaptive arbitrary-order decoding trade one-shot accuracy against higher-k coverage.

Phase 1 tests whether a label-free boundary-closure diagnostic predicts this schedule-dependent coverage difference on unseen queries. It does not test private aggregation, establish causal forks, train a model, or claim that entropy reduction is generally harmful.

The independent unit is the query. Rollouts and token positions are repeated measurements.

## 2. Frozen Inputs

### Model and external code

- Model: `GSAI-ML/LLaDA-8B-Instruct`
- Model revision: `08b83a6feb34df1a6011b80c3c00c7563e963b07`
- JustGRPO repository commit: `1a2fddb5c6655597e63081c0af5ebb718a849f39`
- No training, adapter merge, or checkpoint substitution is permitted in Phase 1.

### Datasets

- GSM8K: `openai/gsm8k`, config `main`, split `test`, revision `740312add88f781978c0658806c59bc2815b9866`.
- MATH-500: `ankner/math-500`, split `test`, revision `412eb03a7193aed402c5e4009fc958c66e61b937`.
- File hashes, row indices, prompt hashes, and split roles are fixed in `phase1_split_manifest_v1.json`.
- The 50 GSM8K calibration queries are exactly the completed Phase 0 signal queries. They are not held-out evidence.
- MATH-500 is sampled proportionally over `type × level` strata. Calibration and held-out indices are disjoint.

## 3. Frozen Decode Conditions

All conditions use 256 generated tokens, 256 denoising steps, temperature 0.6, CFG scale 0, and mask token ID 126336.

| Policy | Block size | Selection/remasking rule | Role |
|---|---:|---|---|
| `ar_b1` | 1 | low-confidence rule, degenerate to left-to-right block commitment | Primary baseline |
| `ao_b8` | 8 | confidence-based | Dose-response screening |
| `ao_b16` | 16 | confidence-based | Dose-response screening |
| `ao_b32` | 32 | confidence-based | Primary AO condition |
| `random_b32` | 32 | random eligible-position selection | Non-left-to-right control |

The same query/rollout seed key is assigned across policies. Because policies consume random draws differently after their trajectories diverge, this is a matched seed label, not a claim of stepwise common random numbers.

## 4. Two-Stage Design

### Phase 1A: calibration

- GSM8K: reuse the first 16 rollouts from Phase 0 for `ar_b1` and `ao_b32`; run 16 rollouts for `ao_b8`, `ao_b16`, and `random_b32`.
- MATH-500: run 50 calibration queries × 5 policies × 16 rollouts.
- Calibration data may be used to define the label-free candidate-boundary score, matching calipers, BCI normalization, baseline predictor, and numerical failure thresholds.
- Calibration data may not alter held-out query IDs, primary policies, primary target, rollout counts, or Gate 1 logic.

Before any held-out run, create an append-only `phase1_bci_freeze_v1.json` containing:

1. the exact candidate-boundary score and token exclusions;
2. the exact matched-control algorithm and calipers;
3. the exact query-level BCI formula;
4. baseline and augmented prediction formulas;
5. parser and malformed-output handling;
6. all coefficients and normalization constants learned on calibration data;
7. repository commit, environment lock, split-manifest hash, and timestamp.

### Phase 1B: held-out test

- Run 100 unseen GSM8K and 100 unseen MATH-500 queries.
- Every policy receives exactly 16 rollouts.
- `ar_b1` and `ao_b32` are then extended to exactly 64 total rollouts regardless of screening outcomes. This extension is not optional and may not depend on the first 16 results.
- Held-out outcomes may not change the BCI, matching rule, target metric, baselines, exclusions, or Gate 1 criteria.

## 5. Frozen Outcomes

### 5.1 Schedule-level reasoning outcomes

For each query and policy, report pass@1 and unbiased Pass@k for `k ∈ {2,4,8,16}` in the five-policy screening set. For the primary `ar_b1` and `ao_b32` conditions, additionally report unbiased Pass@32 and Pass@64.

Define the primary high-k coverage score

```text
CoverageAUC(x, policy)
  = mean(Pass@4, Pass@8, Pass@16, Pass@32, Pass@64).
```

The primary query-level target for RQ2 is

```text
ARCoverageAdvantage(x)
  = CoverageAUC(x, ar_b1) - CoverageAUC(x, ao_b32).
```

Pass@64 alone, per-rollout correctness, unique parsed answers, output length, and AR-only/AO-only solved-set membership are secondary.

### 5.2 Closure outcomes

For every adaptively committed token, compute eligible-to-commit entropy closure, margin closure, raw deferral step, and normalized within-block commitment rank. AR closure is zero by construction and is not compared inferentially with AO closure.

The Phase 1 RQ1 endpoint is the query-level mean paired difference in entropy closure between prospectively selected candidate-boundary positions and matched non-candidate controls within `ao_b32`. Margin closure is the first secondary endpoint. Candidate positions are not called causal forks in Phase 1.

### 5.3 BCI prediction outcomes

The RQ2 primary association is the task-stratified held-out Spearman association between frozen BCI and `ARCoverageAdvantage`. Confidence intervals use a query bootstrap stratified by task.

Incremental predictive value is evaluated using two calibration-fitted, frozen predictors:

- baseline: task indicator, prompt length, output length, mean eligible entropy, and mean commit confidence;
- augmented: the same predictors plus BCI.

The metric is held-out per-query absolute error. Report `MAE_baseline - MAE_augmented` with a task-stratified paired query bootstrap interval. No model selection or coefficient refitting is allowed on held-out data.

### 5.4 Dose response and random-order control

The confidence-policy block-size trend over B=1/8/16/32 and the `random_b32` contrast are secondary. Report all conditions. Do not drop a non-monotonic condition or reinterpret `random_b32` as a causal intervention.

## 6. Statistical Resolution

- Per-task held-out `n=100` gives an approximate two-sided 80%-power correlation resolution of `|rho|≈0.277` under a Fisher-z planning approximation.
- The task-stratified pooled `n=200` approximation is `|rho|≈0.197`.
- The standardized paired-effect resolution at `n=100` is approximately `d=0.280`.
- Final uncertainty is query-clustered; token counts are never treated as independent sample size.
- Report point estimates and 95% intervals for all frozen endpoints, whether favorable or not.
- Token-level mixed models and lexical categories are exploratory unless separately frozen before held-out inference.

## 7. Gate 1 and Falsification

Proceed to forced-token intervention only if both conditions hold:

1. Candidate-minus-control entropy closure is positive on both tasks, or its 95% interval excludes zero on one task while the other task does not show a resolved effect in the opposite direction.
2. On held-out queries, the pooled task-stratified BCI association has a 95% interval above zero and the augmented predictor improves MAE with a 95% interval above zero for `MAE_baseline - MAE_augmented`.

If closure exists but condition 2 fails, BCI is not treated as an explanation of coverage loss. The project may continue only as a narrower descriptive decoder study. If candidate-control closure fails, do not proceed to causal intervention or private aggregation.

The following do not rescue a failed gate:

- selecting a favorable k after seeing held-out results;
- replacing CoverageAUC with per-rollout accuracy;
- redefining candidates using held-out correctness;
- dropping MATH-500 or a policy because it contradicts GSM8K;
- treating the Phase 0 lexical-connector result as confirmatory evidence.

## 8. Missingness and Abort Rules

- All parser failures, truncations, timeouts, repeated-token degeneracy, and incomplete traces remain in the intention-to-decode denominator.
- Report a valid-output sensitivity analysis if malformed outputs exceed 1% in any task-policy condition.
- Abort a shard if trace reconstruction fails, output logging changes sampled tokens, prompt hashes disagree with the split manifest, or the model/dataset revision is not exact.
- Hardware failure permits resumption from completed atomic artifacts with identical seeds. It does not permit replacement queries or rollout-count changes.
- A model-wide OOM or runtime increase beyond the 30% contingency pauses the launch for protocol amendment; it is not silently handled by changing batch semantics or generation length.

## 9. Artifact Contract

The machine-readable schema is `phase1_artifact_schema_v1.json`. Each completed generation must bind the dataset revision, split role, prompt hash, policy configuration, seed key, parser result, correctness, trace hash, code revisions, environment fingerprint, and completion status.

Required run-level artifacts are:

- frozen split manifest and its SHA-256;
- pre-held-out BCI freeze record and its SHA-256;
- append-only generation and trace shards;
- atomic shard-completion manifests with expected/observed key sets;
- analysis configuration and bootstrap seed;
- auto-generated tables containing every frozen endpoint, including failures.

Server paths, usernames, hostnames, credentials, and private cache locations must not appear in the public artifact.

## 10. Resource Contract

The design requires 41,600 new generations. At the Phase 0 observed mean of 28.61 seconds per generation, this is approximately 330.6 GPU-hours, or 429.8 GPU-hours with a 30% MATH/runtime contingency. Estimated compressed trace storage is 3.9 GiB.

No GPU launch is authorized by this document alone. The Phase 1 runner, parser, manifest validation, resume logic, and trace-invariance smoke tests must pass before Phase 1A begins.
