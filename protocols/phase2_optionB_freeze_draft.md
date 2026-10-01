# Phase 2 Option B — Block-Size Dose-Response Freeze (DRAFT)

**Status:** draft for review. Not binding until emitted by the analyzer's `emit-freeze`
mode, hashed, and recorded in the freeze-amendment chain.

**Scope:** block-size dose-response for CoverageAUC on held-out queries at the formal
64-rollout budget. Fixes endpoints, statistics, tolerances, readings, abort rules,
logging, and lane plan **before any Phase 2 inference**.

---

## 0. Binding and provenance

| Field | Value |
|---|---|
| Model | `GSAI-ML/LLaDA-8B-Instruct`, revision `08b83a6…` (unchanged from Phase 1) |
| Inference commit | `eaf4b4f…` (unchanged; checkout must remain clean) |
| BCI freeze | `8f2b6df3…` (untouched; not used by any Phase 2 primary endpoint) |
| Analyzer commit | *(to be bound at emission)* |
| Prior freeze chain | `3bbad351…` → `5314d953…` → `cd286131…` → `68fc9b3b…` → `b18eb39d…` |
| Query set | Phase 1 held-out split, 200 queries (100 GSM8K, 100 MATH-500), unchanged |
| Decode parameters | Unchanged from Phase 1 except `block_size` |

Any drift in model revision, inference commit, query manifest, or decode parameters
other than `block_size` invalidates the design (§7).

---

## 1. Design and cardinality

Conditions: `B ∈ {1, 8, 16, 32}` at fixed confidence-guided within-block selection.

| B | Policy id | Rollouts at 64 | Source |
|---|---|---|---|
| 1 | `ar_b1` | 64 | Phase 1 formal target — **already logged** |
| 8 | `ao_b8` | 64 | 16 from Phase 1 screening + **48 new** |
| 16 | `ao_b16` | 64 | 16 from Phase 1 screening + **48 new** |
| 32 | `ao_b32` | 64 | Phase 1 formal target — **already logged** |

New generations: `200 queries × 2 policies × 48 rollouts = 19,200`.
Total Phase 2 analysis set: `200 × 4 × 64 = 51,200` generations, of which 32,000
are reused from Phase 1 without regeneration.

The analyzer must emit this cardinality block and fail on drift, as in Phase 1.

### 1.1 Design note — what `B` indexes (requires explicit statement in the paper)

`B` is **not** a nuisance parameter of a fixed policy. It indexes the *scope of order
freedom*: at each step the policy chooses among the eligible positions within the
current block, so the choice set grows with `B`. At `B = 1` the choice set is a
singleton and confidence-guided selection is **degenerate** — `ar_b1` is strictly
sequential decoding, and the policy label is vacuous there.

Consequently `B = 1` is the zero-freedom anchor of the axis, not "the same policy at
small block size." The freeze, the report, and the paper must state this; otherwise a
reviewer will read the dose-response as confounding block size with the onset of
confidence-guidance. Phase 1's `ao_b32 ≈ random_b32` null already establishes that
*how* the freedom is used is not resolvable at this resolution, which is precisely why
the surviving hypothesis is about *how much* freedom exists.

---

## 2. Primary outcome definition

Per query `q` and block size `B`:

```
Pass@k_q(B)      unbiased estimator over the 64 logged rollouts, k ∈ {4,8,16,32,64}
CoverageAUC_q(B) = (1/5) · Σ_{k∈{4,8,16,32,64}} Pass@k_q(B)
```

`CoverageAUC` is the **sole** primary metric. `pass@1` and `pass@16` are secondary
(§4.2) and do not enter any trend definition. This resolves the Phase 1 ambiguity in
which `pass@1` and `pass@k` moved in opposite directions across policies.

---

## 3. Primary endpoints

Two primary endpoints. Both are paired within query; both use query-clustered
bootstrap (seed `20260820`, 10,000 replicates, percentile 95% CI, explicit failure on
degenerate replicates), matching Phase 1.

### 3.1 Endpoint P1 — aggregate endpoint contrast

```
Δ_q = CoverageAUC_q(B=32) − CoverageAUC_q(B=1)
Δ̄   = mean over the 200 held-out queries
```

Sign convention: Phase 1 observed AR above AO on both tasks
(GSM8K 0.9624 vs 0.9153; MATH-500 0.6326 vs 0.5912), so the pre-registered expected
sign of `Δ̄` is **negative**.

Resolution (from the Phase 1 read-out): pooled paired SD `0.181`
(GSM8K `0.169`, MATH-500 `0.193`), giving pooled half-width `≈ ±0.025` and per-task
`≈ ±0.035`. The expected effect (`≈ −0.045`) exceeds the pooled half-width, so P1 is
resolvable by design.

### 3.2 Endpoint P2 — trend statistic

Per-query ordinary-least-squares slope of `CoverageAUC` on `x = log₂(B)`, with
`x ∈ {0, 3, 4, 5}`, `x̄ = 3`, `Σ(x−x̄)² = 14`:

```
β_q = [ −3·CoverageAUC_q(1) + 0·CoverageAUC_q(8)
        + 1·CoverageAUC_q(16) + 2·CoverageAUC_q(32) ] / 14
β̄   = mean over the 200 held-out queries
```

Expected sign: **negative**, with magnitude on the order of `Δ̄/5 ≈ −0.009` per
log₂ unit under a linear-in-log₂ gradient.

**Leverage caveat, pre-registered:** the `log₂` spacing `{0,3,4,5}` gives `B = 1`
leverage `9/14 ≈ 0.64` of the total. A pre-registered sensitivity analysis therefore
repeats `β̄` under ordinal coding `x ∈ {0,1,2,3}` (contrast `(−3,−1,1,3)/20`), which
distributes leverage evenly. **Disagreement in resolved sign between the two codings
is reported as an unresolved trend**, not adjudicated post hoc.

### 3.3 Pre-freeze resolution requirement for P2

Before emission, run the seeded resolution simulation for `β̄` using the Phase 1
per-query `B=1` and `B=32` CoverageAUC values with a pre-specified within-query
correlation structure for the two interior levels. Record the empirical half-width.

**Demotion rule:** if the simulated half-width for `β̄` exceeds `0.009` (the slope
implied by the observed endpoint gap), P2 is **demoted to secondary at freeze time**,
P1 becomes the sole primary, and the §5 combination table collapses to its P1 rows.
This decision is made at freeze, never after data are seen.

---

## 4. Secondary endpoints

### 4.1 Adjacent-step contrasts
`B=1→8`, `8→16`, `16→32`, paired, query-bootstrapped. Pre-registered as
**insufficiently powered**: the implied per-step effect (`≈ 0.015`) sits below the
pooled half-width (`±0.025`). Reported with intervals; **never** used to claim or deny
a gradient.

### 4.2 Precision–diversity secondary
`pass@1_q(B)` and `pass@16_q(B)` across all four `B`. Pre-registered reading: if
`pass@1` rises with `B` while `pass@k` composite falls, the Phase 1 diversity-versus-
precision trade is confirmed on the block-size axis. Does not enter P1 or P2.

### 4.3 Per-task decomposition
P1 and P2 computed per task with per-task intervals (`±0.035` scale). Reported for
transparency; pooled estimates remain primary.

---

## 5. Verdict taxonomy (pre-registered readings)

The analyzer classifies each CI mechanically and prints the matching sentence.

**P1 — endpoint contrast**

| Observed | Pre-registered reading |
|---|---|
| CI entirely below zero | Order-freedom cost confirmed at the endpoints of the block-size axis. |
| CI straddles zero | No resolved endpoint effect. Contradicts the Phase 1 formal comparison; reported as an internal replication failure with both estimates shown. |
| CI entirely above zero | Reversal. Reported as-is; no mechanistic account is pre-registered. |

**P2 — trend**

| Observed | Pre-registered reading |
|---|---|
| CI entirely below zero (both codings agree) | Monotone gradient resolved: cost scales with the scope of order freedom. |
| CI straddles zero, or codings disagree | No resolved gradient at this resolution. |
| CI entirely above zero (both codings agree) | Reversal. Reported as-is. |

**Combination table — this is the decision surface**

| P1 | P2 | Reading | Condition A |
|---|---|---|---|
| resolved (−) | resolved (−) | Continuous freedom-cost gradient. | **Run A** |
| resolved (−) | unresolved | Endpoints differ without a resolved smooth gradient — consistent with a *discrete* sequential-vs-block-parallel distinction rather than a continuous freedom axis. | **Do not run A** |
| unresolved | unresolved | Flat. Paper rests on the diversity–precision account plus resolved nulls with stated detection bounds. | **Do not run A** |
| unresolved | resolved | Internally inconsistent. Both reported; no mechanistic claim; no A. | **Do not run A** |

This table supersedes the informal Condition-A rule in the branch memo, which did not
cover the third and fourth rows.

---

## 6. Condition A gating (pre-registered second decision)

Condition A (entropy-first / anti-confidence schedule) proceeds **only** on the first
row of §5. Rationale, recorded before data: Phase 1 found `ao_b32 ≈ random_b32`
(pooled `−0.0059`, CI crossing zero, half-width `±0.036`), which weakens the premise
that *which* positions are resolved first matters. A resolved gradient would restore
that premise by showing order freedom has graded structure; a flat or discrete result
would not.

If A proceeds, its own freeze (endpoints, tolerances, readings, including the
pre-registered reading for "entropy-first also indistinguishable") is emitted before
any A inference. No A inference under this freeze.

---

## 7. Abort and incident rules

**Recoverable — pause, fix, document, resume; no design impact**
- Lane loss, tenant pre-emption, watcher failure, disk exhaustion.
- Individual generation errors below the per-cell malformed threshold.

**Design-invalidating — stop, report, do not silently repair**
- Any drift in model revision, inference commit, query manifest, or decode parameters
  other than `block_size`.
- Marker validation failure not traceable to a recoverable cause.
- Any read of Phase 2 outcome fields before the analysis freeze is emitted.

**Reported, not aborted**
- Malformed rate `> 1%` in any task × policy cell triggers the frozen valid-output
  sensitivity analysis. Phase 1 baseline for comparison: 0.0% malformed under the
  frozen definition; 0.125% empty-string `parsed_answer` (20/16,000), differential
  running *against* the AR advantage. The empty-string audit is re-run on all Phase 2
  cells, with `ao_b8` and `ao_b16` reported separately.

**Never**
- Never stop at partial markers. Never analyze partial-marker data, including
  informally for "a sense of direction."
- Never drop a condition, task, or query subset to save time.
- Never re-seed or re-run a completed shard to change its content.

**Minimum-viable-run rule (carried from Phase 1):** on lane loss, preserve all 19,200
mandatory generations with identical seeds and commands, accept slower completion.

---

## 8. Logging specification

Phase 2 must log everything Phase 1 logged, plus the following per rollout, per
unmasking step. This is the sole addition and it is **not retrofittable**.

| Field | Purpose |
|---|---|
| `step_index`, `block_index` | Step accounting |
| `eligible_positions` | Choice-set size — the operational definition of order freedom at each step |
| `selected_positions`, `selected_tokens` | Realized schedule |
| `selection_score` per eligible position | Keeps BCI-family quantities recomputable |
| **`top64_token_ids`, `top64_logits`** per unmasked position | Effective competitor mass |
| **`full_logsumexp`** per unmasked position | Exact normalizer; required to convert top-`m` logits into probabilities |
| **`full_entropy`** per unmasked position | Bounds the untracked tail contribution |

The three bolded fields exist to make the downstream privacy-feasibility analysis
(`N_eff` per position, composition over sequence length) computable **offline from
Phase 2 artifacts with no further GPU time**. They are out of scope for every Phase 2
endpoint and must not be referenced by any Phase 2 primary or secondary analysis.

**Storage:** ≈ 400 B per unmasked position ⇒ ≈ 2 GB uncompressed over 19,200
generations at ~256 positions. Both backup volumes must be checked for headroom
before launch.

**Throughput gate:** measure the logging overhead in the §9 smoke run. If it exceeds
10% of baseline generation rate, either reduce `m` from 64 to 32 (recording the change
in this freeze before launch) or extend the lane plan. This decision is made at freeze,
not mid-run.

---

## 9. Lane and shard plan

Structure mirrors the Phase 1 extension exactly: `19,200 = 8 shards × 2,400`, three
lanes on GPU 0 / 5 / 7 with the proven watcher chaining, identical commands except
`--device` and `--shard-index`.

**Shard composition:** interleave `ao_b8` and `ao_b16` work items evenly across all
eight shards. Do **not** assign a policy per shard: `ao_b8` requires roughly four times
as many denoising steps per generation as `ao_b32` and materially more than `ao_b16`,
so policy-homogeneous shards would produce badly unbalanced lane durations.

**Throughput must be re-derived, not inherited.** The Phase 1 figure
(~2.1 gen/min/lane, ~390 gen/hour across 3 lanes) was measured on GSM8K at `B=32`.
Phase 2 is slower on two independent axes: smaller blocks mean more sequential
denoising steps, and MATH-500 generations are longer than GSM8K. The branch memo's
5–6 day estimate is plausible but is currently an assumption.

**Required pre-launch smoke run:** 50 generations per (task × policy) cell — 200
generations total — measuring per-cell rate and logging overhead. The ETA is derived
from those four rates, recorded with its basis in the operations report, and re-derived
once MATH lanes are producing at scale. No ETA is published without a measured basis.

---

## 10. Amendment rules

**Amendable before unblinding** (with chain entry, preserved prior freeze, and passing
test suite): instrumentation, documentation, resolution simulations not yet run, and
the two decisions this document defers to freeze time (§3.3 P2 demotion, §8 `m`
reduction).

**Never amendable:** endpoint definitions (§2, §3), the verdict taxonomy (§5), the
Condition-A gate (§6), the condition set, query manifest, rollout budget, decode
parameters, model revision, and the BCI freeze.

The analyzer's append-only invocation log records every mode invocation before its body
runs; `analyze` is the only mode setting `reads_heldout_correctness: true`. The first
Phase 2 correctness read is timestamped there.

---

## 11. Open items requiring sign-off before emission

1. **§1.1** — accept that `B` indexes scope of order freedom with `B=1` as the
   degenerate zero-freedom anchor, and that this is stated in the report and paper.
2. **§3.2 / §3.3** — accept `log₂` coding with the ordinal-coding sensitivity and the
   disagreement rule, and accept the P2 demotion threshold of `0.009`.
3. **§5** — accept the four-row combination table, in particular the "discrete rather
   than continuous" reading that blocks Condition A when P1 resolves but P2 does not.
4. **§8** — accept the logging addition and its storage/throughput gates.
5. **§9** — accept that no ETA is published before the smoke run.

## 12. Reviewer dispositions and §3.3 simulation record (2026-09-01)

*Recorded by the analyst before sign-off. The §11 items remain the collaborators'
sign-off list; this section records the evidence and the recommended dispositions.*

### §3.3 simulation — executed, result recorded

Pre-specified generative structure for the two interior levels (recorded here so the
simulation is reproducible): per query `q`, with the observed Phase 1 64-rollout
endpoints `C1_q = CoverageAUC(ar_b1)` and `C32_q = CoverageAUC(ao_b32)`,

```
C8_q  = C1_q + (3/5)·(C32_q − C1_q) + ε8,   ε ~ N(0, σ²)
C16_q = C1_q + (4/5)·(C32_q − C1_q) + ε16,  seed 20260820 (deterministic)
```

for σ ∈ {0.00, 0.05, 0.10}; the slope statistic is computed per query and its pooled
mean bootstrapped task-stratified (seed `20260820`, 10,000 replicates, percentile 95%),
using the same machinery as the Phase 1 analyzer.

| Scenario | log₂ β̄ point | log₂ β̄ half-width | ordinal β̄ half-width |
|---|---:|---:|---:|
| σ = 0.00 | −0.0088 | **0.0049** | 0.0039 |
| σ = 0.05 | −0.0090 | **0.0049** | 0.0040 |
| σ = 0.10 | −0.0091 | **0.0050** | 0.0042 |

**§3.3 decision, applied at freeze time per the rule:** the simulated log₂ half-width
(≈0.005) is **below** the demotion threshold (0.009), and the implied slope magnitude
(≈0.009) exceeds the half-width — so **P2 survives as primary**; the demotion clause
remains in the document as the pre-registered branch and is simply not taken. The
half-width is insensitive to interior noise because the endpoint spread dominates the
slope statistic's variance.

### Dispositions on the five §11 items

1. **§1.1 (B=1 as zero-freedom anchor):** accept. The framing is correct — the choice
   set at B=1 is a singleton and confidence-guidance is degenerate, so the axis is
   *scope of order freedom*, not block size at fixed policy — and Phase 1's
   `ao_b32 ≈ random_b32` null is exactly the license for it (how the freedom is used is
   unresolvable; how much freedom exists is the surviving question). The report and
   paper must state it as the draft requires.
2. **§3.2/§3.3 (log₂ coding + ordinal sensitivity + 0.009 demotion):** accept. The
   leverage arithmetic checks out (B=1 leverage 9/14 ≈ 0.64; ordinal contrast
   (−3,−1,1,3)/20). The disagreement rule (resolved-sign mismatch between codings ⇒
   unresolved trend) is the correct guard against a spacing artifact. Simulation above:
   demotion not triggered.
3. **§5 four-row combination table:** accept, in particular the
   P1-resolved/P2-unresolved row reading as a *discrete* sequential-versus-block-
   parallel distinction that blocks Condition A — it closes the case the branch memo
   left uncovered, and A (a graded-structure test) is genuinely uninformative there.
4. **§8 logging addition:** accept. Storage arithmetic checks out (≈400 B per unmasked
   position ⇒ ≈2 GB over 19,200 generations). The top-64 logits + full logsumexp +
   full entropy triple is exactly what makes the offline N_eff computation exact
   (logsumexp gives the normalizer; full entropy bounds the untracked tail). The
   throughput gate (>10% ⇒ m=32, decided pre-launch) is the right control.
5. **§9 no ETA before smoke run:** accept. Phase 1's ~2.1 gen/min/lane was
   GSM8K@B=32; `ao_b8` runs ~4× the denoising steps per generation and MATH
   generations are longer, so the 200-generation four-cell smoke is the only honest
   basis for an ETA. Interleaved shard composition is correct.

### Corrections applied to this draft

- §0 prior-freeze chain extended with v5 `b18eb39d…` (bound to analyzer commit
  `debc68d`), which the draft had omitted.

### Not yet done (post-sign-off)

- Implement the Phase 2 Option B analyzer (emit-freeze payload for this document,
  artifact validation, P1/P2 statistics, combination table rendering, invocation
  logging) with the same discipline as Phase 1; emit the freeze only after the five
  §11 items are signed and the implementation passes the full test suite. The
  scheduling rule remains in force: no Phase 2 inference before that emission.

## 13. Sign-off record (2026-09-01)

The five §11 items were signed off by the collaborator on 2026-09-01 ("签署五项，立即开工"). This draft is now the operative pre-freeze specification. Implementation proceeds: additive runner change for the new extension run kind, Phase 2 analyzer (emit-freeze / validate-only / analyze), full test suite, then freeze emission into the amendment chain. The scheduling rule held through sign-off: no Phase 2 inference ran before this record.
