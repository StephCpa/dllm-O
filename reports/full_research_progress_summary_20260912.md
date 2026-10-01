# Research Progress Summary: Decoding Order, Parallelism, and Test-Time Coverage in dLLMs

**Updated:** 2026-09-12
**Project:** `dllm_order_transmission`
**Current status:** All planned GPU experiments and the targeted post-unblinding CPU robustness analyses are complete. The project is ready for manuscript construction.
**Evidence policy:** Results below are explicitly separated into calibration, prospectively frozen confirmatory tests, pre-registered secondary analyses, and post-unblinding exploratory analyses.

**Scope note:** This report intentionally covers only the dLLM decoding project.
The separate DP-ICL ICLR submission, its fixed-record allocation study, and its
submission operations are not summarized here.

## 1. Executive Summary

This project studies how token-commitment schedules in a masked diffusion
language model (dLLM) affect mathematical reasoning under repeated sampling.
The original hypothesis was that a label-free local uncertainty measure,
Boundary Closure Index (BCI), could identify queries harmed by arbitrary-order
decoding. That predictive hypothesis failed on held-out data. The subsequent
pre-registered experiments nevertheless established a narrower and more useful
empirical result:

> Decoding parallelism and position-ranking policy reshape how test-time
> sampling trades marginal precision for multi-sample coverage and
> parsed-answer diversity.

The strongest confirmatory result is a decrease in multi-sample CoverageAUC
when moving from strict sequential decoding (`B=1`) to block-parallel decoding
(`B=32`): `-0.04421`, 95% CI `[-0.06974, -0.01977]`. Most of this decrease occurs
at the transition from `B=1` to `B=8`; differences among `B=8`, `B=16`, and
`B=32` are unresolved. At the same time, pooled pass@1 rises slightly from
`0.54273` to `0.55875`, exposing a precision-coverage trade-off that an
accuracy-only evaluation would miss.

A post-unblinding endpoint-leverage audit shows that 95.3% of the absolute
log2-slope decomposition is attributable to the `B=1`-to-parallel step, and
query-level P1 and P2 are highly dependent (`r=0.984`, tie-aware
`rho=0.943`). The two frozen endpoints both resolved according to their
pre-registered rules, but they are not independent confirmations. The honest
scientific result is a discrete sequential-versus-block-parallel threshold in
this design, not a graded law over parallel block sizes.

A prospectively frozen entropy-first intervention did not recover the aggregate
coverage loss relative to confidence-first decoding: pooled
`CoverageAUC64(EF-B32) - CoverageAUC64(AO-B32) = -0.00986`, 95% CI
`[-0.04200, 0.02377]`. However, the intervention substantially changed the
shape of the pass@k frontier, response length, commitment entropy, and direct
parsed-answer diversity. Position ranking therefore matters, but not as a
simple scalar improvement in the frozen aggregate endpoint.

The apparent Condition A verdict also depends on the sampling budgets being
aggregated. Using the same 64 rollouts throughout, the low-budget grid
`k in {2,4,8,16}` gives `-0.04960 [-0.08200,-0.01692]`, whereas the high-budget
grid `k in {16,32,64}` gives `+0.01009 [-0.02645,+0.04746]`. CoverageAUC is
therefore a budget-weighted functional of a crossing pass@k frontier, not an
intrinsic scalar property of a decoding policy.

The same precision-to-coverage transition appears in the matched AO-versus-
random comparison. On their common first 16 rollouts, confidence-first AO is
resolvedly better at `k=1` and `k=2`; the contrast reaches zero near `k=4` and
turns negative but remains unresolved at `k=8` and `k=16`. Together with the
diversity ordering, this supports confidence guidance as a low-budget
precision/concentration control. It does not establish a resolved high-budget
coverage advantage for random ranking.

Matched first-16 checks further show that random ranking exceeds entropy-first
EF at every measured k, with resolved differences through `k=8`, despite EF's
substantially higher parsed-answer entropy. More answer-string diversity did
not translate into more correct-answer coverage. Strict sequential AR, in
turn, exceeds random at `k<=4` and becomes indistinguishable at `k>=8`; AR is
therefore not dominated by random on the available matched range.

The defensible paper-level conclusion is not that block-parallel commitment is
the unique cause of coverage loss. It is that sequential-to-parallel freedom is
the main measured association axis, while neither tested position-ranking
policy restores the aggregate coverage endpoint, and the policies produce
meaningfully different precision-diversity profiles.

## 2. Scientific Scope

### 2.1 Model, tasks, and fixed decoding setup

- Model: `GSAI-ML/LLaDA-8B-Instruct`, frozen revision
  `08b83a6feb34df1a6011b80c3c00c7563e963b07`.
- External decoder/evaluator: `LeapLabTHU/JustGRPO`, commit
  `1a2fddb5c6655597e63081c0af5ebb718a849f39`.
- Tasks: GSM8K and MATH-500, each pinned to a frozen dataset revision.
- Generation length: 256 token positions.
- Denoising steps: 256.
- Temperature: 0.6; classifier-free guidance scale: 0.
- Unit of inference: query. Tokens and rollouts are never treated as independent
  statistical samples.

### 2.2 Decoding policies

| Policy | Block size | Position rule | Interpretation |
|---|---:|---|---|
| `ar_b1` | 1 | No within-block choice | Strict left-to-right commitment using the same dLLM |
| `ao_b8` | 8 | Confidence-first | Moderate block-parallel arbitrary-order decoding |
| `ao_b16` | 16 | Confidence-first | Larger-block arbitrary-order decoding |
| `ao_b32` | 32 | Confidence-first | Primary paper-style arbitrary-order decoding |
| `random_b32` | 32 | Random eligible positions | Matched-block structural reference |
| `ef_b32` | 32 | Entropy-first | Intervention that prioritizes high-uncertainty positions |

`EF` and `AO` compare two position-ranking policies. EF is not an algebraic
inverse of AO because EF ranks full-distribution entropy, whereas AO ranks the
sampled candidate token by model probability.

### 2.3 Metrics

- **Pass@k:** unbiased query-level probability that at least one of `k` sampled
  solutions is correct.
- **CoverageAUC64:** mean Pass@k over `k in {4, 8, 16, 32, 64}` from 64 rollouts.
- **ScreeningCoverageAUC16:** mean Pass@k over `k in {2, 4, 8, 16}` from 16
  rollouts.
- **Parsed-answer diversity:** number, entropy, and modal share of frozen
  parsed-answer strings. This is not semantic solution-path diversity.
- **BCI:** a label-free query-level diagnostic based on entropy closure at
  candidate boundary positions in AO traces.

## 3. Experimental Scale and Evidence Hierarchy

At least **76,800 formal or signal generations** were executed, excluding smoke
runs and traced/untraced replay executions:

| Stage | New generations | Main role |
|---|---:|---|
| Phase 0 signal | 3,200 | Calibration and mechanism signal only |
| Phase 1 | 41,600 | Calibration plus prospectively held-out Gate 1 study |
| Phase 2 Option B | 19,200 | Frozen block-size dose-response extension |
| Condition A | 12,800 | Frozen entropy-first intervention |

Phase 1 contains 43,200 analytic records because 1,600 GSM8K calibration
records are reused from Phase 0. Option B analyzes 51,200 total records across
four block sizes, but only 19,200 were newly generated; the remaining endpoints
were inherited from validated Phase 1 records.

The evidence hierarchy is:

1. **Phase 0:** frozen before implementation but explicitly calibration-only.
2. **Phase 1 Gate 1:** prospectively frozen held-out test.
3. **Option B:** prospectively frozen intervention with fixed endpoint and
   decision table.
4. **Condition A:** prospectively frozen intervention triggered mechanically by
   Option B.
5. **Trade-off and robustness analyses:** post-unblinding exploratory analyses;
   hypothesis-generating, interpretation, and paper-positioning evidence only.

## 4. Phase 0: Instrumentation and Initial Signal

### 4.1 Stage 0A instrumentation equivalence

Stage 0A used five GSM8K queries, four rollouts, and both `ar_b1` and `ao_b32`.
Tracing and non-tracing executions were required to produce identical tokens.
The instrumentation audit passed and established deterministic replay, complete
256-position commitment, active-block validity, finite trace values, and
trace-based reconstruction.

On this small smoke set, both policies solved all five queries by maximum-k.
The lexical-connector analysis was retained only as a replication sanity check,
not as a causal result.

### 4.2 Stage 0B frozen signal run

The signal run evaluated 50 frozen GSM8K queries, 32 rollouts, and two policies,
for 3,200 generations.

| Policy | Pass@1 | Pass@32 | Solved queries |
|---|---:|---:|---:|
| `ar_b1` | 0.7725 | 0.9800 | 49/50 |
| `ao_b32` | 0.8019 | 0.9600 | 48/50 |

The AR-minus-AO Pass@32 contrast was `+0.0200`, with paired query-bootstrap CI
`[0.0000, 0.0600]`. This was treated as directional calibration evidence, not a
confirmatory finding.

The trace analysis showed:

- Mean AO deferral: 15.5 steps.
- Deferral versus entropy closure Spearman: `0.5092` over 409,600 token events.
- Global mean commitment-time entropy: `0.1511`.
- Query-level closure did not predict AR-minus-AO success in calibration:
  Spearman `-0.1261`.

Phase 0 therefore established that boundary closure was measurable and that AR
and AO could trade pass@1 against high-k coverage. It did not validate closure
as a per-query predictor.

## 5. Phase 1: Held-Out Boundary-Closure Study

### 5.1 Design

Phase 1 expanded to five schedules and two tasks. Calibration used 50 GSM8K and
50 MATH-500 queries. The prospectively held-out set used 100 unseen queries per
task. Each screening cell used 16 rollouts; `ar_b1` and `ao_b32` were extended
to 64 rollouts for formal CoverageAUC.

BCI, candidate selection, matching calipers, baseline predictors, failure
thresholds, bootstrap seed (`20260820`), and the Gate 1 decision rules were
frozen before held-out correctness was read. Five pre-unblinding amendments
added audit instrumentation and interpretation rules without changing the
scientific endpoint or Gate 1 Boolean criteria.

### 5.2 Calibration evidence

- Candidate-minus-control entropy closure: `+0.1603` pooled
  (`+0.2077` GSM8K; `+0.1193` MATH-500).
- Matched pairs: 13,558 from 409,600 eligible tokens.
- Calibration BCI versus coverage-advantage proxy: Spearman `-0.1969`.
- Baseline selected-CV MAE: `0.08959`; augmented MAE with BCI: `0.09033`.
- Approximately 60% of calibration queries had exactly zero 16-rollout coverage
  advantage, indicating a sparse and noisy predictive target.

Calibration established that the candidate/control construction was feasible,
but it provided negative directional evidence for BCI's predictive validity.

### 5.3 Prospectively held-out Gate 1 results

All 35,200 held-out screening and primary-extension artifacts were validated
before the first held-out correctness read.

**Condition 1: candidate-control entropy closure — PASS**

| Task | Mean paired closure difference | 95% CI |
|---|---:|---:|
| GSM8K | +0.1935 | [0.168, 0.219] |
| MATH-500 | +0.1216 | [0.098, 0.146] |

Both tasks had valid matches for 100/100 held-out queries.

**Condition 2a: BCI association — FAIL / unresolved**

- Pooled task-stratified Spearman: `+0.0136`, 95% CI
  `[-0.134, +0.161]`.
- The interval straddled zero; BCI was not accepted as an explanation or
  predictor of query-level coverage loss.

**Condition 2b: incremental predictive value — FAIL / wrong direction**

- `MAE_baseline - MAE_augmented = -0.000169`, 95% CI
  `[-0.000302, -0.000037]`.
- Adding BCI made prediction statistically detectably but practically
  negligibly worse: the magnitude is about 0.18% of the pooled baseline MAE
  (`0.092417`).

The overall Gate 1 verdict was **FAIL, branch (b)**: closure exists, but BCI does
not predict the global coverage difference. The intended targeted per-query
intervention story was abandoned.

### 5.4 Pre-registered secondary evidence

- Formal 64-rollout CoverageAUC favored `ar_b1` over `ao_b32` on both tasks:
  `0.9624` versus `0.9153` on GSM8K and `0.6326` versus `0.5912` on MATH-500.
- At 16 rollouts, `random_b32` had the highest pass@16 on both tasks
  (`0.99` GSM8K; `0.62` MATH-500) but the lowest pass@1
  (`0.646`; `0.284`).
- `CoverageAUC16(ao_b32) - CoverageAUC16(random_b32) = -0.0059`, 95% CI
  approximately `[-0.042, +0.029]`. The difference was unresolved at a pooled
  half-width of about 0.036.

These results motivated a global block-size intervention rather than a
query-targeted BCI intervention.

## 6. Phase 2 Option B: Frozen Block-Size Dose Response

### 6.1 Design and execution

Option B held the confidence-first ranking policy fixed and evaluated
`B in {1, 8, 16, 32}` with 64 rollouts on all 200 held-out queries. New inference
extended `ao_b8` and `ao_b16` from 16 to 64 rollouts, requiring 19,200 new
generations. The primary endpoints were frozen as:

- P1: `CoverageAUC(B=32) - CoverageAUC(B=1)`.
- P2: query-level OLS trend against `log2(B)`, with a frozen ordinal-coding
  sensitivity analysis.

All 51,200 records used by the analysis were present and valid. A first analysis
attempt stopped on a `KeyError` because valid all-zero cells were not
materialized in a `defaultdict`. No endpoint was produced before failure. The
repair initialized every expected cell explicitly, was bound by an amendment,
and preserved every frozen endpoint and decision rule. There were 132 valid
all-zero cells, making the repair substantively necessary and auditable.

### 6.2 Primary results

**P1: endpoint contrast**

| Scope | Estimate | 95% CI |
|---|---:|---:|
| Pooled | -0.04421 | [-0.06974, -0.01977] |
| GSM8K | -0.04702 | [-0.08183, -0.01718] |
| MATH-500 | -0.04141 | [-0.08062, -0.00455] |

**P2: trend**

| Coding | Pooled estimate | 95% CI |
|---|---:|---:|
| `log2(B)` | -0.00965 | [-0.01474, -0.00489] |
| Ordinal sensitivity | -0.00707 | [-0.01141, -0.00301] |

Both primary endpoints resolved in the pre-registered negative direction under
the frozen decision table. A later exploratory leverage audit shows that they
should not be interpreted as independent confirmations.

### 6.3 Post-unblinding endpoint-leverage audit

The log2-coded P2 slope has the exact aggregate decomposition

`beta = 3(mean(y8,y16,y32)-y1)/14 + (y32-y8)/14`.

| Component | Estimate |
|---|---:|
| Mean P2 slope | -0.009646 |
| `B=1`-to-parallel step | -0.009193 |
| Within-parallel endpoint drift | -0.000454 |

Thus, 95.3% of the absolute two-component decomposition is the discrete step.
At query level, P1 and P2 correlate at Pearson `0.9840` and tie-aware Spearman
`0.9427`. P2 carries little evidence beyond P1 in this realized design even
though both passed their frozen rules. The supported result is a
sequential-versus-block-parallel threshold effect, not a smooth freedom-dose
relationship.

The 132 zero-success cells correspond to 21, not approximately 33, queries
with all four block-size cells at zero. The same 21 queries have both P1
endpoint arms at zero. In the outcome-conditioned subset with at least one
endpoint success (`n=100` GSM8K, `n=79` MATH-500), P1 is
`-0.04972 [-0.07871,-0.02148]`. This selected-subset estimate is descriptive
and cannot replace the intention-to-treat frozen P1.

### 6.4 Shape of the block-size effect

| Block size | Pooled CoverageAUC | GSM8K | MATH-500 |
|---:|---:|---:|---:|
| 1 | 0.79746 | 0.96236 | 0.63256 |
| 8 | 0.75960 | 0.92045 | 0.59874 |
| 16 | 0.75084 | 0.91390 | 0.58778 |
| 32 | 0.75325 | 0.91534 | 0.59115 |

| Adjacent contrast | Estimate | 95% CI | Reading |
|---|---:|---:|---|
| `B=1 -> B=8` | -0.03786 | [-0.05747, -0.02028] | Resolved decrease |
| `B=8 -> B=16` | -0.00876 | [-0.02698, +0.00965] | Unresolved |
| `B=16 -> B=32` | +0.00240 | [-0.01153, +0.01733] | Unresolved |

The data support a negative global trend, but not a smooth decrement at every
adjacent block-size step. Most of the measured change occurs at the transition
from strict sequential decoding to block-parallel decoding.

### 6.5 Precision-coverage trade-off

| Block size | Pooled pass@1 | Pooled pass@16 |
|---:|---:|---:|
| 1 | 0.54273 | 0.80581 |
| 8 | 0.54891 | 0.77218 |
| 16 | 0.55711 | 0.75849 |
| 32 | 0.55875 | 0.75694 |

Increasing block size slightly improves marginal one-rollout accuracy while
reducing the probability that a 16-sample set contains at least one correct
answer. This is the central evidence that decoding order changes the returns to
test-time sampling rather than simply making all quality metrics uniformly
better or worse.

The frozen Option B decision table required Condition A because both P1 and P2
resolved in the negative direction.

## 7. Condition A: Frozen Entropy-First Intervention

### 7.1 Objective and design

Condition A tested whether the measured sequential-to-parallel coverage cost
could be recovered by changing which eligible positions were committed first.
It compared:

- `ef_b32`: high full-distribution entropy first, 64 rollouts;
- `ao_b32`: confidence-first control, 64 rollouts;
- `random_b32`: matched first 16 rollouts only, as a secondary reference.

The primary endpoint, bootstrap procedure, length guard, artifact schema,
runner commit, smoke criteria, 12,800-generation workload, and eight-shard
design were frozen before inference. The smoke audit verified 256 trace events,
maximal-entropy selection, tie handling, finite fields, checksum chains, and
traced/untraced token equivalence.

Before unblinding, `validate-only` confirmed 12,800/12,800 valid artifacts, zero
missing or invalid records, eight valid markers, and
`outcome_fields_read=false`. The first Condition A correctness read was logged
at 2026-09-09 21:40:52 CST.

### 7.2 Primary endpoint

`CoverageAUC64(ef_b32) - CoverageAUC64(ao_b32)`:

| Scope | Estimate | 95% CI | Frozen classification |
|---|---:|---:|---|
| Pooled | -0.00986 | [-0.04200, +0.02377] | Straddles zero |
| GSM8K | +0.00084 | [-0.04303, +0.04747] | Straddles zero |
| MATH-500 | -0.02056 | [-0.06708, +0.02631] | Straddles zero |

The pre-registered reading is:

> No resolved entropy-first benefit; confidence-guided position choice is not
> identified as the source of the coverage cost.

This is not an equivalence result. The pooled interval still permits effects
within a roughly `+0.024/-0.042` range.

### 7.3 Secondary endpoints

- `EF - random` ScreeningCoverageAUC16: `-0.06206`, 95% CI
  `[-0.09139, -0.03297]`.
- `EF - AO` ScreeningCoverageAUC16: `-0.05616`, 95% CI
  `[-0.09807, -0.01455]`.
- GSM8K solved-set membership at 64 rollouts: both 95, EF-only 4, AO-only 1,
  neither 0.
- MATH-500: both 62, EF-only 7, AO-only 8, neither 23.
- All parser statuses were `ok`; no empty responses were recorded.

These secondary endpoints are descriptive and do not replace the primary
endpoint.

The difference between the unresolved 64-rollout primary aggregate and the
negative 16-rollout secondary aggregate motivated a post-unblinding audit of
budget-grid sensitivity; it is not evidence that either frozen endpoint was
computed incorrectly.

### 7.4 Length guard

The pre-registered decoded-response-length guard triggered:

- EF minus AO at 64 rollouts: `-9.83` whitespace tokens, 95% CI
  `[-11.78, -7.90]`.
- EF minus random at 16 rollouts: `-4.11`, 95% CI
  `[-5.43, -2.82]`.

All policies commit exactly 256 generated token positions, so this difference
concerns decoded response length rather than generation-cap exposure. It must
qualify every coverage interpretation. No outcomes were removed and no
post-hoc length matching was performed.

### 7.5 Frontier reshaping

Condition A did not raise aggregate CoverageAUC, but it materially changed the
pass@k curve:

| Task and policy | Pass@1 | Pass@16 | Pass@32 | Pass@64 |
|---|---:|---:|---:|---:|
| GSM8K AO | 0.732 | 0.926 | 0.945 | 0.960 |
| GSM8K EF | 0.498 | 0.937 | 0.973 | 0.990 |
| MATH-500 AO | 0.385 | 0.588 | 0.645 | 0.700 |
| MATH-500 EF | 0.222 | 0.588 | 0.646 | 0.690 |

EF is resolvedly worse than AO at `k in {1,2,4}`. The difference is unresolved
from `k=8` onward: point estimates cross zero between `k=8` and `k=16`, but EF
is never resolvedly better at any tested budget. The primary aggregate endpoint
averages across several `k` values and therefore hides this crossing from a
clear low-budget loss to an unresolved high-budget difference.

The crossing completes on GSM8K, where EF exceeds AO at `k=64`, but not on
MATH-500, where EF remains slightly lower. Neither B32 ranking policy reaches
strict sequential AR's pooled pass@64 of `0.875`.

The GSM8K `k=64` point advantage rests on five discordant solved sets (four
EF-only versus one AO-only; exact two-sided sign-test `p=0.375`) near task
ceiling. MATH-500 has seven EF-only and eight AO-only queries. These counts
reinforce that the high-budget difference is unresolved rather than evidence
of a likely EF benefit.

## 8. Post-Unblinding Exploratory CPU Analyses

These analyses were conducted after Condition A unblinding. They do not alter
any pre-registered verdict and must be labeled exploratory wherever reported.

### 8.1 Audit and correction

The initial exploratory implementation at commit `e1bba72` was independently
audited and corrected at commit `6ed75dc`:

- a duplicated first-query contribution in mean wall time was removed;
- Spearman correlations now use average ranks for ties;
- majority-vote correctness now uses the modal parsed answer's majority frozen
  grader label rather than the first sampled correct answer;
- deterministic matched Monte Carlo index samples are used across policies;
- MV@1 and MV@n are exact;
- two parsed-answer groups with inconsistent frozen grader labels are recorded;
- observed wall time is explicitly scoped as an uncontrolled runtime proxy.

All 72 corrected frontier rows passed mechanical checks, and MV@1 exactly
matches pass@1.

### 8.2 Rollout-budget frontier

Pooled across the two tasks:

| Policy | Pass@1 | Pass@16 | Pass@64 availability/result |
|---|---:|---:|---:|
| `ar_b1` | 0.543 | 0.806 | 0.875 |
| `ao_b8` | 0.559 | 0.780 | Not available |
| `ao_b16` | 0.562 | 0.760 | Not available |
| `ao_b32` | 0.559 | 0.757 | 0.830 |
| `random_b32` | 0.465 | 0.805 | Not available |
| `ef_b32` | 0.360 | 0.763 | 0.840 |

The unavailable entries were never generated: k=64 comparisons are limited to
`ar_b1`, `ao_b32`, and `ef_b32`. No arm was added after unblinding.

At a fixed rollout count, confidence-first schedules are strongest at very low
`k`, while AR and more exploratory schedules become competitive or superior at
higher `k`. `ao_b8` and `random_b32` both exceed `ao_b32` at `k=16` without
relying on wall-time comparisons.

The conservative parsed-answer majority-vote metric remains below pass@k and
shows limited gains at larger `k`. This suggests that the principal benefit is
best-of-k/oracle coverage rather than unassisted self-consistency. This remains
exploratory because no learned verifier or reranker was evaluated.

Observed wall times cannot support a controlled latency Pareto claim: policies
were run at different times, under different shared-server loads, and with
instrumentation overhead. Rollout count `k` is the valid nominal budget axis.

### 8.3 Direct parsed-answer diversity

Mean parsed-answer entropy at `k=16`:

| Policy | GSM8K | MATH-500 |
|---|---:|---:|
| `ef_b32` | 1.4377 | 1.8823 |
| `random_b32` | 0.9382 | 1.6286 |
| `ar_b1` | 0.5806 | 1.3398 |
| `ao_b32` | 0.3681 | 1.0238 |

At `k=64`, EF produces a mean of 20.09 unique parsed answers on GSM8K and 26.80
on MATH-500, compared with 3.39 and 11.33 for AO. The approximate diversity
ordering `EF > random > AR > AO` is consistent across both tasks and agrees with
the pass@k frontier.

This measures frozen parsed-answer strings. Semantically equivalent forms can
remain distinct, and no claim of semantic reasoning-path diversity is currently
supported.

### 8.4 EF mechanism diagnostics

On the pre-existing r00-r03 trace subsample across all 100 queries per task,
mean entropy at committed positions was:

| Task | EF | AO |
|---|---:|---:|
| GSM8K | 0.4920 | 0.1553 |
| MATH-500 | 0.7736 | 0.3196 |

This verifies quantitatively that EF changed the intended position-ranking
behavior and committed substantially more uncertain positions.

Tie-aware query-level correlations between EF-minus-AO response-length changes
and CoverageAUC changes were small:

- GSM8K: `rho = 0.0973`.
- MATH-500: `rho = -0.0109`.

Within EF, response length versus CoverageAUC Spearman was `0.3240` on GSM8K and
`0.0277` on MATH-500. These results do not show a simple monotone explanation
of the coverage contrast by response length, but they do not remove the length
guard or establish that length is causally irrelevant.

### 8.5 Direct pass@1 length diagnostics

Because the earlier correlation used CoverageAUC, a second CPU-only analysis
targeted the more relevant EF-minus-AO pass@1 gap. The raw pooled gap is
`-0.19898 [-0.23266,-0.16469]`. A query-level regression with task-specific
intercepts, evaluated at zero mean response-length difference, gives
`-0.20703 [-0.25069,-0.16343]`; the gap does not attenuate.

As a complementary descriptive check, common-seed rollout pairs were retained
only when their decoded lengths were close:

| Absolute length difference | EF - AO correctness | Eligible pairs (GSM8K / MATH-500) |
|---:|---:|---:|
| <=2 words | -0.16142 [-0.20479,-0.11995] | 765 / 540 |
| <=5 words | -0.17164 [-0.20925,-0.13602] | 1,609 / 1,207 |
| <=10 words | -0.17971 [-0.21574,-0.14504] | 2,759 / 2,143 |
| <=20 words | -0.18551 [-0.21955,-0.15146] | 4,270 / 3,656 |

These analyses make a simple output-truncation explanation implausible, but
they are not causal adjustments: decoded length is post-treatment, selection
on it changes the estimand, and the zero-difference regression may extrapolate
outside common support. The frozen length guard remains triggered.

The two adjustment strategies disagree in direction. Near-length matching
attenuates the magnitude monotonically from `-0.199` to `-0.161` at the
two-word threshold, roughly one fifth, whereas the linear zero-difference
estimate becomes slightly more negative (`-0.207`). The matched subset also
changes composition. Both are therefore reported; neither is selected as a
preferred causal correction.

### 8.6 Sampling-budget-grid sensitivity

To separate budget weighting from first-16 sampling variation, EF and AO were
recomputed on the same 64 rollouts under multiple k-grids:

| Equal-weight k-grid | EF - AO | 95% CI |
|---|---:|---:|
| `{4,8,16,32,64}` | -0.00986 | [-0.04226,+0.02253] |
| `{2,4,8,16}` using all 64 rollouts | -0.04960 | [-0.08200,-0.01692] |
| `{16,32,64}` | +0.01009 | [-0.02645,+0.04746] |
| `{2,4,8,16}` using frozen first 16 | -0.05616 | [-0.09895,-0.01404] |

The low-budget result remains negative on all 64 rollouts, so the contrast with
the primary verdict is not merely a first-16 realization artifact. The result
is a measurement principle: an aggregate over pass@k must declare the target
sampling-budget distribution, and crossing frontiers should be shown directly
rather than collapsed into one supposedly policy-intrinsic AUC.

### 8.7 Confidence-first versus random ranking

The remaining CPU-only check decomposed the frozen matched first-16
AO-minus-random comparison:

| k | AO - random | 95% CI |
|---:|---:|---:|
| 1 | +0.09969 | [+0.06906,+0.13219] |
| 2 | +0.04363 | [+0.01046,+0.07871] |
| 4 | +0.00117 | [-0.03528,+0.03784] |
| 8 | -0.02837 | [-0.06891,+0.01145] |
| 16 | -0.04000 | [-0.08500,+0.00500] |

The equal-weight `{2,4,8,16}` aggregate is the frozen Phase 1 null,
`-0.00589 [-0.04173,+0.02881]`, because positive low-k and negative high-k
components cancel. Confidence-guided ranking therefore has a resolved
low-budget precision effect that the scalar aggregate concealed. Its apparent
high-budget coverage cost is directionally consistent with the diversity
analysis but unresolved.

`random_b32` has only 16 rollouts (3,200 records total and no `r16` records), so
no matched 64-rollout comparison exists. An AO-64-versus-random-16 estimator-
precision sensitivity analysis is retained in the JSON audit but is not used as
matched-64 evidence.

Two additional matched comparisons sharpen the frontier:

| k | EF - random | AR - random |
|---:|---:|---:|
| 1 | -0.10656 [-0.12688,-0.08594] | +0.08344 [+0.06094,+0.10720] |
| 2 | -0.08696 [-0.10971,-0.06550] | +0.05896 [+0.03508,+0.08392] |
| 4 | -0.06277 [-0.08995,-0.03652] | +0.03930 [+0.01217,+0.06782] |
| 8 | -0.05350 [-0.08939,-0.01916] | +0.01952 [-0.01136,+0.05133] |
| 16 | -0.04500 [-0.09000,+0.00000] | +0.00500 [-0.03500,+0.04500] |

EF's much higher parsed-answer entropy therefore does not buy higher correct-
answer coverage over random on the matched range. Random weakly dominates EF
on the observed point estimates, with resolved superiority through `k=8`.
AR's precision advantage over random is resolved through `k=4`; their higher-k
differences are unresolved. These are two contrasts on shared queries and
shared controls, not statistically independent replications.

## 9. Integrated Scientific Interpretation

### 9.1 Findings supported by the evidence

1. Candidate boundary positions exhibit reproducible local entropy closure on
   held-out GSM8K and MATH-500.
2. BCI does not predict which held-out queries gain coverage from strict
   sequential decoding.
3. Moving from `B=1` to `B=32` reduces multi-sample CoverageAUC while slightly
   improving pass@1.
4. Most of the measured block-size effect occurs at the sequential-to-parallel
   transition (`B=1 -> B=8`); P1 and P2 are highly collinear and do not provide
   independent evidence for a smooth decrement among larger block sizes.
5. Entropy-first ranking does not recover aggregate CoverageAUC64 over
   confidence-first ranking at the experiment's resolution.
6. Position ranking nevertheless reshapes the full pass@k curve, commitment
   entropy, response length, and parsed-answer diversity.
7. Accuracy-only evaluation can obscure losses in the returns to repeated
   sampling.
8. CoverageAUC depends on the sampling-budget grid: low-budget weighting favors
   confidence-first decoding, while high-budget weighting makes EF and AO
   unresolved in this experiment.
9. EF's pass@1 deficit persists in post-hoc length-comparable analyses, although
   post-treatment conditioning cannot establish causality.
10. Matched AO-versus-random results reveal a resolved confidence-first gain at
    `k<=2` that is canceled by unresolved negative contrasts at `k>=8` in the
    original aggregate.
11. EF maximizes parsed-answer diversity but is resolvedly worse than random in
    correct-answer coverage through `k=8`; diversity alone is not useful
    exploration without a link to correctness.

### 9.2 Best current paper framing

The strongest framing is:

> **Decoding schedules reshape the precision-diversity returns to test-time
> sampling in diffusion language models.** Parallel confidence-first decoding
> can improve marginal precision while reducing high-k solution coverage;
> entropy-first decoding increases exploration without improving the frozen
> aggregate coverage endpoint.

The second clause must remain asymmetric: confidence-first's low-budget
precision gain is resolved, while random/entropy-first high-budget coverage
gains are not resolved at any tested k. Their stronger diversity is measured,
but its practical value remains oracle-dependent without a verifier.

The evidence does not support one monotone "exploration dial." Block size and
ranking policy jointly shape a frontier, but entropy-first is an inefficient
form of exploration here: it creates the most answer-string diversity while
underperforming random ranking in correct-answer coverage. The operational
message is to evaluate precision, correct-answer coverage, and diversity
separately rather than treating diversity as a proxy for coverage.

This positions block size and position ranking as test-time sampling controls,
not merely implementation details.

A second, broadly relevant measurement contribution is that policy rankings
can reverse with the sampling-budget distribution. Reporting one CoverageAUC
without its k-grid can hide a crossing frontier and produce a deployment-
irrelevant verdict.

### 9.3 Mechanistic hypothesis for the threshold

One post-unblinding hypothesis is that `B=1 -> B>=8` crosses a deterministic-
to-stochastic scheduling boundary. At `B=1`, there is no within-block position
choice, so the position schedule is fixed and rollout variation comes from
token sampling. Once multiple eligible positions are available, confidence-
or entropy-dependent ranking can make the commitment schedule itself vary with
the sampled state; random ranking varies directly with the rollout seed. This
could explain a large first jump followed by a flat parallel regime.

This account is consistent with the observed threshold and the random-policy
frontier, but it was not prospectively tested. It should appear as a discussion
hypothesis, not as an established mechanism or a new causal conclusion.

### 9.4 Claims that are not supported

- BCI is a valid per-query predictor or targeting rule.
- P1 and P2 are independent confirmations of a graded block-size law.
- Block-parallel commitment alone causally explains all coverage loss.
- EF and AO are equivalent.
- Parsed-answer diversity is semantic reasoning diversity.
- EF should be deployed with a verifier or reranker; this has not been tested.
- Observed server wall time establishes deployment latency or compute Pareto
  dominance.
- The findings generalize to all dLLMs, non-mathematical tasks, training
  regimes, temperatures, or decoding implementations.
- Random or entropy-first ranking has a confirmed high-budget coverage benefit
  over confidence-first ranking.
- Higher parsed-answer diversity implies higher correct-answer coverage.

## 10. Reproducibility and Audit Status

- Phase 0, Phase 1, Option B, and Condition A use pinned model, dataset,
  evaluator, seed, and protocol revisions.
- Phase 1 unblinding was preceded by a five-link analysis-freeze amendment
  chain; scientific invariants remained mechanically fixed.
- Condition A freeze SHA-256:
  `a1205bd8bc5b6d9d81fdfc63c842efc3ede8cc59f192f80523b61151bf8d7be5`.
- Condition A analyzer/runner commit:
  `5faf63fcb3fd59fc714a711d2e2a1571e94cf965`.
- Condition A validation: 12,800/12,800 valid, 0 missing, 0 invalid, 8/8
  markers, correctness unread during validation.
- Independent standard-library recomputation reproduced every Condition A
  point estimate and frozen bootstrap interval exactly.
- An ENOSPC interruption during shard 4 was handled under the frozen recovery
  rule. On resume, validation recognized and skipped 1,524 already-complete
  records; only the interrupted work key was regenerated with the same command
  and seed. No valid record was overwritten or duplicated, and no data were
  lost.
- All eight Condition A shard archives and the 22-item control-plane manifest
  have been verified by SHA-256 on two physical volumes.
- Full project tests passed before Condition A launch; the corrected exploratory
  outputs additionally passed schema/row-count and metric-consistency checks.

Key result commits:

- Phase 1 Gate 1 unblinding: `7022aa0`.
- Option B canonical result and repair chain: `61b0d69` plus the canonical
  report.
- Condition A completion and unblinding: `98e99f2`.
- Initial exploratory analysis: `e1bba72`.
- Corrected exploratory audit: `6ed75dc`.
- Post-unblinding robustness analysis: see
  `reports/posthoc_robustness_20260912/`.

## 11. Current State and Recommended Next Work

### Complete

- All planned Phase 0, Phase 1, Option B, and Condition A inference.
- All required validation, frozen analysis, unblinding, independent
  recomputation, and dual-volume backups.
- Post-unblinding frontier, diversity, endpoint-leverage, budget-grid, and
  length-robustness analyses, including the matched AO-versus-random per-k
  decomposition.
- Audit and correction of the exploratory analysis implementation.

### One optional GPU extension now merits an explicit decision

The completed evidence is sufficient for a defensible paper, and no repair
experiment is required. However, the post-unblinding decomposition exposed one
specific coverage gap: `random_b32`, the strongest B32 point estimate at
`k=16`, has no observations beyond 16 rollouts. Extending it from 16 to 64
rollouts would require 9,600 new generations (`200 queries x 48 rollouts`) and
would fill the only asymmetric high-budget arm in the intended frontier.

If this extension is run, it must be a separately frozen, targeted prospective
follow-up selected after unblinding. The freeze should name one primary
endpoint, preferably pooled paired `Pass@64(random_b32)-Pass@64(ao_b32)`, with
task-specific estimates, `Pass@32`, the declared high-budget grid, and
comparisons to EF/AR as secondary results. It must preserve the original
Gate-1 null and state that the follow-up was motivated by the exploratory
crossing. The result should use the existing queries, rollout indices 16--63,
seed convention, parser, and query-level bootstrap.

The recommendation is a weak preference to run this extension if resources and
schedule are comfortable, because it resolves a visible missing arm rather than
opening a new research front. Declining remains defensible: restrict random
claims to `k<=16`, show the missing `k=32/64` entries explicitly, and do not
assert high-budget random superiority.

### Immediate manuscript work

1. Freeze an evidence-to-claim matrix that labels every result as calibration,
   confirmatory, secondary, or exploratory.
2. Build a main pass@k figure showing the crossing precision-coverage frontiers
   across block sizes and ranking policies.
3. Add a direct parsed-answer-diversity panel and a compact intervention panel
   with the Condition A CI and triggered length guard.
4. Write the results in the actual chronological logic: local closure
   replicates, prediction fails, block-size effect resolves, entropy-first does
   not rescue aggregate coverage, and exploratory analysis reveals frontier
   reshaping.
5. Keep the negative BCI and Condition A results prominent; they are part of
   the project's evidential discipline, not details to hide.
6. State the major limitations explicitly: one model, two mathematical tasks,
   oracle pass@k, parsed-string rather than semantic diversity, uncontrolled
   runtime observations, and no verifier/reranker evaluation.
7. Name the budget-grid dependence result and show the crossing pass@k curves;
   never report CoverageAUC without its defining k-grid.
8. Describe Option B as a sequential-versus-parallel threshold and disclose
   the P1/P2 leverage dependence rather than counting two confirmations.
9. Present the AO-versus-random aggregate null as cancellation across k, not as
   evidence that position ranking is irrelevant.

### Optional future study, not required for the current paper

If a stronger deployment claim becomes necessary, run a separately frozen
study measuring hardware-controlled latency and verifier/reranker utility. That
study should evaluate success under a fixed end-to-end compute budget rather
than reuse the current shared-server wall times. It should not be retrofitted
into the present confirmatory chain.

## 12. Bottom-Line Assessment

The project produced substantive progress despite falsifying its original
predictive mechanism. It now has:

- a replicated local closure phenomenon;
- an honestly failed query-level predictor;
- a prospectively confirmed block-size effect on multi-sample coverage;
- a prospectively tested ranking-policy intervention with a resolved scope
  boundary;
- direct exploratory evidence that schedule choice controls a
  precision-diversity frontier;
- an endpoint audit showing the block-size result is discrete rather than a
  smooth dose response;
- and a budget-grid audit showing why crossing test-time-scaling frontiers
  cannot be summarized by one policy-intrinsic AUC.

The work is best viewed as a rigorous empirical mechanism and measurement study
of dLLM decoding and test-time sampling. Its contribution is narrower than a
universal causal theory or a new decoding algorithm, but materially stronger
than a collection of null results. The next source of value is careful
manuscript construction, not additional opportunistic experimentation.
