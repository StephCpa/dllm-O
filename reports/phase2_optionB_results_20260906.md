# Phase 2 Option B: Canonical Results and Independent Audit

**Date:** 2026-09-06

**Status:** Canonical Option B analysis complete; the frozen gate requires Condition A.

**Inference status:** 19,200/19,200 new Phase 2 generations complete, with 8/8 shard markers valid.

## 1. Provenance

- Original Phase 2 freeze SHA-256: `b83edeff717e8bfbf59feea5f38dc6e266d50fbcd42a1aa440fe510daf8f5126`
- Frozen analyzer/runner commit: `b4280b5dc58196a198ed135d94eab2abf8368a77`
- Analysis-repair commit: `61b0d69513406e8c78945ba25a84589ffbab77a4`
- Analysis-repair amendment SHA-256: `5f1f32523aac816e0f21bad690a7561a8f8df01698efa4672858042d42841673`
- BCI freeze SHA-256: `8f2b6df3f98abdb0b9060b704b02fcc428090d01d2d279514c7c108e2fb6ef17`
- Validation artifact SHA-256: `97522f75e2a2ffd43ee8ec7603ff8e6fa61320462db1a0183cf817af23a524fb`
- Canonical JSON report SHA-256: `dbcf3ae1a12fc8f47d575922fd5e1b24eb2b004433b96f832da5e8e0b4e08c3f`
- Canonical Markdown report SHA-256: `d3e78500205950cac024755fe9a1091a42eb9667e7c2ef64c27f798a248dd7f0`
- Canonical repaired-analysis console SHA-256: `08e8c2edf6f29775246e099663ea3ecda042222f8fd61dc5de4cb2e04d217018`

The canonical validation covers 51,200 records: all were present and valid, with zero missing and zero invalid records. The eight Phase 2 extension markers contain exactly 19,200 new records and have matching expected/observed key hashes.

## 2. Analysis incident and repair

The first post-unblinding analysis invocation failed before producing any P1, P2, bootstrap, or gate report. The failure was a `KeyError` on a valid cell containing 0 correct rollouts out of 64. The loader used a `defaultdict`, but an all-zero cell was never materialized before conversion to a plain dictionary.

The repair explicitly initializes every expected `(task, query, B)` cell to zero before reading its 64 rollouts. It changes neither artifacts nor any endpoint, coding, bootstrap, confidence interval, or verdict rule. The repair is bound by a machine-checked amendment to the original freeze, the failed invocation timestamp, the failed console hash, the original analyzer commit, and the repair commit.

The bug was substantively relevant: 132 of 800 valid task/query/B cells have zero successes. Their counts are:

| Task | B=1 | B=8 | B=16 | B=32 |
|---|---:|---:|---:|---:|
| GSM8K | 0 | 4 | 5 | 4 |
| MATH500 | 25 | 32 | 32 | 30 |

The repaired implementation passed seven targeted tests locally, the full 90-test suite locally, and the targeted repair tests on the server.

## 3. Frozen primary results

### P1: endpoint contrast

P1 is `CoverageAUC(B=32) - CoverageAUC(B=1)`.

| Scope | Estimate | 95% CI |
|---|---:|---:|
| Pooled | -0.04421 | [-0.06974, -0.01977] |
| GSM8K | -0.04702 | [-0.08183, -0.01718] |
| MATH500 | -0.04141 | [-0.08062, -0.00455] |

The pooled interval and both task-specific intervals are entirely below zero. The frozen endpoint reading is therefore resolved in the predicted negative direction: moving from strict sequential decoding (`B=1`) to the maximum tested order freedom (`B=32`) reduces multi-sample coverage.

### P2: block-size trend

| Coding | Pooled estimate | 95% CI | Frozen classification |
|---|---:|---:|---|
| `log2(B)` | -0.00965 | [-0.01474, -0.00489] | resolved negative |
| Ordinal sensitivity | -0.00707 | [-0.01141, -0.00301] | resolved negative |

The `log2(B)` trend is also negative within each task:

| Task | Estimate | 95% CI |
|---|---:|---:|
| GSM8K | -0.01018 | [-0.01785, -0.00363] |
| MATH500 | -0.00911 | [-0.01628, -0.00252] |

Both the primary coding and its frozen ordinal sensitivity analysis resolve in the same negative direction.

## 4. Shape of the effect

The absolute pooled CoverageAUC values are:

| Block size | Pooled | GSM8K | MATH500 |
|---:|---:|---:|---:|
| 1 | 0.79746 | 0.96236 | 0.63256 |
| 8 | 0.75960 | 0.92045 | 0.59874 |
| 16 | 0.75084 | 0.91390 | 0.58778 |
| 32 | 0.75325 | 0.91534 | 0.59115 |

The adjacent pooled contrasts are:

| Contrast | Estimate | 95% CI | Reading |
|---|---:|---:|---|
| B=1 to B=8 | -0.03786 | [-0.05747, -0.02028] | resolved decrease |
| B=8 to B=16 | -0.00876 | [-0.02698, 0.00965] | unresolved |
| B=16 to B=32 | +0.00240 | [-0.01153, 0.01733] | unresolved |

Accordingly, the result supports a resolved negative global trend across the frozen block-size axis, but not a claim that every adjacent increase in B causes a smooth monotonic decrement. Most of the measured coverage cost appears at the transition from strict sequential decoding (`B=1`) to block-parallel decoding (`B>=8`); differences among the larger block sizes remain unresolved at this design's resolution.

## 5. Precision-diversity trade-off

| Block size | Pass@1 | Pass@16 |
|---:|---:|---:|
| 1 | 0.54273 | 0.80581 |
| 8 | 0.54891 | 0.77218 |
| 16 | 0.55711 | 0.75849 |
| 32 | 0.55875 | 0.75694 |

Increasing order freedom raises single-rollout accuracy slightly while reducing the probability that a 16-sample set contains at least one correct answer. Accuracy-only evaluation would therefore miss the principal cost: trajectories become less diverse even as their marginal precision improves.

## 6. Independent recomputation

An independent standard-library script read all 51,200 result JSON files directly, initialized every expected cell explicitly, implemented unbiased Pass@k using the combinatorial estimator, and recomputed CoverageAUC, P1, and both P2 codings without importing the canonical analyzer's aggregation functions.

| Endpoint | Independent | Canonical | Absolute error |
|---|---:|---:|---:|
| P1 pooled | -0.044214199288056866 | -0.04421419928805687 | 6.94e-18 |
| P2 `log2(B)` pooled | -0.009646204670693142 | -0.009646204670693142 | 0 |
| P2 ordinal pooled | -0.007069887957355705 | -0.007069887957355705 | 0 |

This independently verifies all canonical point estimates. The confidence intervals remain those produced by the frozen 10,000-replicate, task-stratified bootstrap.

## 7. Frozen gate and next action

The frozen combination table maps resolved-negative P1 plus resolved-negative P2 to **continuous freedom-cost gradient** and sets `run_condition_a=true`. Therefore Condition A is required by the pre-registered decision rule.

The phrase “continuous gradient” is the frozen branch label, not permission to claim that every adjacent block-size contrast resolves. The paper-facing interpretation must preserve the shape qualification in Section 4.

No Condition A inference may begin until its own protocol, exact entropy-first policy, endpoint hierarchy, length/truncation guard, artifact schema, analyzer, runner commit, expected cardinality, shard plan, abort rules, and freeze hash are finalized and validated without reading Condition A outcomes.
