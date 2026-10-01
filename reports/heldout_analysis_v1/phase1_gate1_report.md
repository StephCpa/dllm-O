# Phase 1 Held-Out Analysis and Gate 1 Report

This report consumes the final BCI freeze directly. No coefficient, candidate rule, matching caliper, or target definition was refit or altered on held-out data. Every frozen endpoint is reported, favorable or not.

## Provenance

- BCI freeze SHA-256: `8f2b6df3f98abdb0b9060b704b02fcc428090d01d2d279514c7c108e2fb6ef17`
- Analysis freeze SHA-256: `b18eb39dba335dd0d74e915e4562f57f4b8a4a678618774852c0ed2394ba83e4`
- Analyzer commit: `debc68d75d9fb892d930ae54dd50095d3d0ad914`
- Inference runner commit: `eaf4b4f9214eec6066010fad7dba5c5319d85122`
- Bootstrap: seed 20260820, 10000 replicates, task-stratified percentile 95% intervals
- Artifacts validated: 35200/35200

## Amendment Invariants

The analysis-freeze amendment chain above cannot change any of the following; each is mechanically enforced by this analyzer on every run:

- Final BCI freeze SHA-256: `8f2b6df3f98abdb0b9060b704b02fcc428090d01d2d279514c7c108e2fb6ef17` (identity-checked on load; the two superseded calibration freezes are audit-only)
- Gate 1 boolean rules (closure condition; pooled BCI association and MAE-delta conditions) exactly as frozen
- Frozen decode parameters: 256 generated tokens, 256 denoising steps, temperature 0.6, CFG scale 0, mask token 126336, and the five policies with their block sizes and selection rules
- Inference runner commit `eaf4b4f9214eec6066010fad7dba5c5319d85122` (validated on every artifact, together with the model revision, dataset revisions, and the frozen split manifest)

## Design Cardinality and Generation Arithmetic

- Held-out queries: {'gsm8k': 100, 'math500': 100} per task, 200 total
- Screening: 5 policies x 16 rollouts = 16,000 generations
- Primary extension: ar_b1 and ao_b32 x 48 added rollouts = 19,200 generations
- Held-out total: 35,200 generations
- Protocol new-generation total 41,600 = 35,200 held-out + 6,400 new calibration (1,600 reused Phase 0 generations excluded)
- Pass@32/Pass@64/CoverageAUC use all 64 rollouts for ar_b1 and ao_b32 only; ao_b8, ao_b16, and random_b32 remain 16-rollout secondary conditions with k in {2,4,8,16}

## Power Resolution (Planning Quantities)

- Per-task correlation MDE (n=100, alpha=.05, 80% power): 0.277
- Pooled correlation MDE (n=200 approximation): 0.197
- Per-task standardized paired-effect MDE (n=100): 0.280
- Fisher-z / normal planning approximations for single-sample correlations at n=100 and n=200; the frozen statistic is the task-stratified average-rank Spearman (average ranks within task, center, concatenate, Pearson). These MDEs are planning quantities, not replacements for query-clustered bootstrap intervals, and token counts never increase effective sample size. A seeded empirical resolution check of the actual recipe is emitted separately by the power-simulation mode.

## RQ1: Candidate-Minus-Control Entropy Closure

### gsm8k

- Matched queries: 100/100 (100.0%); unmatched are reported as structurally unmatched, not dropped silently
- RQ1 unresolved for this task: False
- Mean query-level entropy closure difference: 0.193482 (95% CI [0.167758, 0.219266])
- Margin closure (secondary): 0.168211 (95% CI [-0.053522, 0.393512])

### math500

- Matched queries: 100/100 (100.0%); unmatched are reported as structurally unmatched, not dropped silently
- RQ1 unresolved for this task: False
- Mean query-level entropy closure difference: 0.121649 (95% CI [0.098352, 0.146124])
- Margin closure (secondary): 0.027012 (95% CI [-0.127544, 0.186841])

## RQ2: BCI Association With AR Coverage Advantage

- Pooled task-stratified Spearman: 0.013630 (95% CI [-0.133887, 0.160555])
- gsm8k: rho = -0.077662 (95% CI [-0.291402, 0.142018]); zero-advantage fraction 34.0%
- math500: rho = 0.105883 (95% CI [-0.095538, 0.304179]); zero-advantage fraction 39.0%

## Incremental Predictive Value

- Pooled MAE baseline: 0.092417
- Pooled MAE augmented: 0.092586
- MAE_baseline - MAE_augmented: -0.000169 (95% CI [-0.000302, -0.000037])

- gsm8k: MAE baseline 0.078174, augmented 0.078534, delta -0.000361 (95% CI [-0.000536, -0.000187])
- math500: MAE baseline 0.106661, augmented 0.106638, delta 0.000023 (95% CI [-0.000172, 0.000214])

## Gate 1

- Condition 1 (closure): **PASS** — unresolved tasks: none
- Condition 2a (pooled BCI interval above zero): **FAIL** — pre-registered classification: `straddles_zero`
- Condition 2b (MAE improvement interval above zero): **FAIL** — pre-registered classification: `entirely_below_zero`
- **Gate 1 overall: FAIL**

Verdict taxonomy (pre-registered before unblinding): a CI entirely above zero passes; a CI straddling zero fails as unresolved; a CI entirely below zero fails AND is recorded as a resolved wrong-direction falsification, not a null.

Gate 1 passes only if both conditions hold. On failure the frozen falsification rules apply: no k re-selection, no per-rollout accuracy substitution, no dropping MATH-500 or a policy, and no rescue through the Phase 0 lexical result.

## Secondary Endpoints

### Five-policy screening summary (per task)

| task | policy | pass@1 | pass@8 | pass@16 | malformed |
|---|---|---:|---:|---:|---:|
| gsm8k | ar_b1 | 0.7256 | 0.9409 | 0.9600 | 0.0000 |
| gsm8k | ao_b8 | 0.7288 | 0.9167 | 0.9500 | 0.0000 |
| gsm8k | ao_b16 | 0.7306 | 0.9058 | 0.9300 | 0.0000 |
| gsm8k | ao_b32 | 0.7431 | 0.9095 | 0.9500 | 0.0000 |
| gsm8k | random_b32 | 0.6456 | 0.9463 | 0.9900 | 0.0000 |
| math500 | ar_b1 | 0.3706 | 0.5944 | 0.6600 | 0.0000 |
| math500 | ao_b8 | 0.3887 | 0.5693 | 0.6100 | 0.0000 |
| math500 | ao_b16 | 0.3937 | 0.5458 | 0.5900 | 0.0000 |
| math500 | ao_b32 | 0.3856 | 0.5300 | 0.5800 | 0.0000 |
| math500 | random_b32 | 0.2838 | 0.5500 | 0.6200 | 0.0000 |

### Primary-policy formal coverage (64 rollouts)

| task | policy | pass@32 | pass@64 | CoverageAUC |
|---|---|---:|---:|---:|
| gsm8k | ar_b1 | 0.9929 | 1.0000 | 0.9624 |
| gsm8k | ao_b32 | 0.9454 | 0.9600 | 0.9153 |
| math500 | ar_b1 | 0.6914 | 0.7500 | 0.6326 |
| math500 | ao_b32 | 0.6446 | 0.7000 | 0.5912 |

### Solved-set membership (64 rollouts, per task)

| task | AR-only | AO-only | both | neither |
|---|---:|---:|---:|---:|
| gsm8k | 4 | 0 | 96 | 0 |
| math500 | 9 | 4 | 66 | 21 |

### Confidence-guided versus random order (ao_b32 - random_b32)

Both policies share identical order freedom at block size 32; they differ only in whether the commitment order is confidence-guided or random. Descriptive secondary contrast; not causal; not part of Gate 1.

| task | mean over k | 95% CI |
|---|---:|---|
| gsm8k | -0.0154 | [-0.0607, 0.0263] |
| math500 | 0.0036 | [-0.0522, 0.0586] |
| pooled | -0.0059 | [-0.0421, 0.0290] |

- Pre-registered classification: `straddles_zero`
- Pre-registered reading: order freedom itself is implicated; confidence-guidance is not separately identified

| task | pass@2 diff | pass@4 diff | pass@8 diff | pass@16 diff |
|---|---:|---:|---:|---:|
| gsm8k | 0.0279 | -0.0127 | -0.0367 | -0.0400 |
| math500 | 0.0593 | 0.0151 | -0.0200 | -0.0400 |
| pooled | 0.0436 | 0.0012 | -0.0284 | -0.0400 |

No task-policy malformed fraction exceeded 1%; the valid-output sensitivity analysis was not triggered.

## Audit Boundary

Parser failures, truncations, and degenerate outputs remain in every Pass@k denominator. Token counts are never treated as independent sample sizes; all intervals are query bootstraps. This analyzer did not modify any frozen rule after held-out outcomes were read.
