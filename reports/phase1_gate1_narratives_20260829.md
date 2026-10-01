# Phase 1 Gate 1 Pre-Unblinding Narratives and BCI Construct-Validity Record

**Date:** 2026-08-29
**Status:** Written before any held-out correctness value has been read or summarized. This document supplements the frozen protocol (§7 falsification rules) and the analysis freeze; it defines no new frozen rules and changes no Gate 1 boolean.

**摘要（中文）：** 本文在揭盲前预先写好 Gate 1 三种结局的论文叙事，并把 BCI 的定义、可重算路径与校准证据如实记录。校准数据已给出方向性不利信号（校准 pooled Spearman 为 −0.1969，且加入 BCI 后 MAE 无增益），因此“closure 成立但 BCI 无预测力”被列为首要（模态）分支，而不是与其余分支等权。冻结的 verdict taxonomy 规定：CI 全在零上 = 通过；CI 跨越零 = 未决失败；CI 全在零下 = 失败且属于方向相反的确证性证伪，而不是 null。该 taxonomy 已在揭盲前写入分析冻结（原冻结 `3bbad351…` 作废保留审计，Gate 1 布尔规则不变）。

## 1. What BCI is, concretely, and how to recompute it

BCI (Boundary Closure Index) is a label-free, query-level diagnostic computed entirely from the logged `ao_b32` held-out screening traces (`r00`–`r15`). A third party can recompute it from the published traces plus the final BCI freeze (`8f2b6df3f98abdb0b9060b704b02fcc428090d01d2d279514c7c108e2fb6ef17`) alone. No correctness labels appear anywhere in the computation.

### Step 1 — selected-token rows

For each denoising-step event line in `heldout/screening/{task}/ao_b32/q{query:04d}/r{rollout:02d}.trace.jsonl.gz`:

- Zip `selected_positions` with `selected_token_ids`; map each position to its index within `eligible_positions`.
- Read `eligible_entropy[index]`, `eligible_logit_margins[index]` (pre-commit), `entropy[index]`, `logit_margins[index]` (post-commit), and `candidate_probabilities[index]`.
- Skip the row if any value is non-finite, if the selected token is the mask token (`126336`), or if confidence ≤ 0.
- Record:
  - `entropy_closure = eligible_entropy − entropy`
  - `margin_closure = logit_margins − eligible_logit_margins`
  - `normalized_position = (position − active_block_start) / max(1, active_block_end − active_block_start − 1)`
  - `normalized_commitment_rank = step_in_block / 31` (AO-B32 blocks use 32 steps, zero-indexed)
  - `prompt_length = active_block_start − block_index × 32`

### Step 2 — candidate annotation (frozen per-task rule)

With `low_margin = −ln(max(eligible_logit_margins, 0) + 1e-12)`:

```text
entropy_z     = (eligible_entropy − entropy_median) / entropy_iqr
margin_z      = (low_margin − low_margin_median) / low_margin_iqr
candidate_score = 0.5 · clip(entropy_z, ±5) + 0.5 · clip(margin_z, ±5)
candidate     iff candidate_score ≥ candidate_threshold_q75
```

Frozen task constants (full precision in the freeze JSON; rounded here for reading):

| task | entropy_median | entropy_iqr | low_margin_median | low_margin_iqr | threshold q75 |
|---|---:|---:|---:|---:|---:|
| gsm8k | 1.709262 | 2.736059 | −0.117783 | 2.549445 | 0.487005 |
| math500 | 2.667424 | 2.960707 | 0.470004 | 2.302585 | 0.393824 |

**Scope of the frozen constants:** the four normalization constants and `candidate_threshold_q75` are computed once per task over the pooled calibration eligible-token rows (all calibration queries × rollouts that feed the freeze fit) — not per query and not per rollout. At held-out inference they are applied as fixed constants; the only grouping-level operation is the per-`(task, query, rollout)` fallback below.

Fallback (frozen): if a `(task, query, rollout)` group contains no candidate, the row with the maximum `candidate_score` (ties broken by lowest position) is marked as the candidate.

### Step 3 — rollout BCI

```text
rollout BCI = Σ w_i · entropy_closure_i / Σ w_i,
              w_i = 1 + max(candidate_score_i − q75, 0)
```

over all candidate tokens in the rollout.

### Step 4 — query BCI

```text
query BCI = arithmetic mean of the 16 rollout BCIs (r00–r15)
```

### Step 5 — matched controls (RQ1 only; not part of BCI)

Within the same `(task, query, rollout, block)`, each candidate is matched to a non-candidate control with caliper 0.25 on each of: `normalized_position`, `normalized_commitment_rank`, `|Δ eligible_entropy| / entropy_iqr`, `|Δ low_margin| / low_margin_iqr`. Selection is deterministic greedy nearest-neighbor without replacement, minimizing the sum of the four normalized distances (position as tiebreak). RQ1's endpoint is the query-level mean of `candidate.entropy_closure − control.entropy_closure`; margin closure is the first secondary endpoint.

## 2. Calibration evidence: what it does and does not establish

Frozen calibration (GSM8K 50 + MATH-500 50 queries, 16 rollouts, 5 policies):

- **Closure holds on calibration.** Candidate-minus-control entropy closure +0.1603 overall (GSM8K +0.2077, MATH-500 +0.1193). Margin closure −0.0692 (secondary; reported as-is, never reframed as supportive).
- **Matching is operational.** 409,600 eligible tokens, 102,400 candidates, 13,558 matched pairs; matched-query coverage 50/50 GSM8K, 49/50 MATH-500.
- **Predictive evidence is negative, not merely weak.** Calibration pooled Spearman between BCI and the coverage-advantage proxy is **−0.1969**; baseline selected-CV MAE 0.08959 versus augmented 0.09033, i.e. no incremental gain from BCI.

**Construct-validity position:** calibration establishes feasibility and reproducibility — the candidate rule runs, controls exist, closures are measurable. It provides *negative directional evidence* about BCI's predictive association. It does not establish construct validity of BCI as a predictor of coverage loss. The held-out RQ2 is the first real test, and the modal expectation must be failure of condition 2, not success.

**Structural-baseline pre-emption (ICLR-review analogue):** BCI's *construction* is definitionally tied to the same trace fields as the calibration margin machinery (eligible entropy and logit margin); that is by design and is disclosed here. Its *association* with coverage is an empirical claim, tested on calibration with a negative result (−0.1969). The held-out test resolves the association; it does not inherit any validity from the construction.

## 3. Pre-registered verdict taxonomy (analysis-freeze amendment)

Added to the analysis freeze before unblinding (superseding server freeze `3bbad351…`, preserved for audit; Gate 1 booleans unchanged):

| Observed pooled 95% CI | Condition verdict | Paper status |
|---|---|---|
| Entirely above zero | Pass (frozen boolean unchanged) | Supporting |
| Straddles zero | Fail, **unresolved** — no evidence either way | Null, bounded by the recorded MDE |
| Entirely below zero | Fail, **resolved wrong-direction falsification** — not a null | Negative finding, reported as such |

The same taxonomy applies to the pooled BCI association CI and the pooled `MAE_baseline − MAE_augmented` CI.

## 3.5 Pre-registered readings beyond Gate 1 (freeze amendment, 08-30)

### Confidence-vs-random contrast readings

For the `ao_b32 − random_b32` pooled CI, the reading is fixed before unblinding:

- **CI entirely below zero:** the cost of arbitrary-order decoding is attributable to *confidence-guided position selection*, not to order freedom.
- **CI straddles zero:** order freedom itself is implicated; confidence-guidance is not separately identified.
- **CI entirely above zero:** confidence-guidance is protective; the flexibility-trap mechanism does not transfer to this setting.

Readings are bounded by the screening window's resolution (16 rollouts per policy; empirical half-widths from the power-simulation mode). This matters most under branch (b), where the contrast is the likely centerpiece — it must not be interpreted for the first time after it has been seen.

### Mechanistic accounts for a negative RQ2

If the pooled BCI association CI lands entirely below zero, "wrong direction" must have a pre-committed meaning:

- **Account (a) — competence, not avoidance.** Closure measures decoder competence: queries where the arbitrary-order decoder successfully resolves candidate uncertainty are queries it handles well, so AR has less to add. The negative sign is mechanically expected, and BCI is a *misnamed quantity* rather than a failed one.
- **Account (b) — difficulty confound.** Closure and coverage advantage are both driven by query difficulty in opposite directions, making the association confounded rather than informative.

**Pre-committed distinguishing evidence:** account (a) predicts query-level BCI correlates positively with `ao_b32` coverage itself (competence); account (b) predicts BCI tracks difficulty measures (prompt length, base accuracy) without a direct competence link. Both are checked against the frozen secondary tables before either is asserted. If neither distinguishes, both are reported.

### Unblinding read order

1. Record the Gate 1 verdict (condition 1 and condition 2 blocks) and commit it.
2. Read the dose-response table.
3. Read the confidence-vs-random contrast (secondary readings apply).

The Markdown report renders Gate 1 before all secondary sections, and the JSON is key-sorted so `gate1` precedes the secondary tables; the analyze invocation is timestamped in the append-only invocation log. The order is procedural discipline, mechanically supported by the frozen output layout.

### What the amendment chain cannot change

The analysis-freeze amendment chain (now four links, each pre-unblinding and each with a stated reason) cannot change any of the following; each is mechanically enforced by the analyzer on every run and stated plainly in the Gate 1 report's "Amendment Invariants" section:

- the final BCI freeze (`8f2b6df3…`, identity-checked on load; the two superseded calibration freezes are audit-only);
- the Gate 1 boolean rules (closure condition; pooled BCI association and MAE-delta conditions);
- the frozen decode parameters (256 generated tokens, 256 denoising steps, temperature 0.6, CFG 0, mask token 126336, and the five policies with their block sizes and selection rules);
- the inference runner commit (`eaf4b4f…`, validated on every artifact, together with model/dataset revisions and the split manifest).

The chain is long because it records three days of pre-unblinding pre-registration; the invariants make explicit that none of it moved anything that was already frozen.

## 4. Branch narratives

### Branch (b) — closure holds, BCI does not predict coverage (modal)

*Why modal:* calibration association −0.1969 (negative), augmented MAE shows no gain, and 60% of calibration queries had an exactly-zero 16-rollout coverage-advantage proxy (sparse, noisy target). Branch (b) is the most carefully developed narrative and the default expectation.

**What the paper is:** a narrower descriptive dLLM decoder study. Its contributions:

1. RQ1 closure result — candidate boundary positions show higher entropy closure than matched controls, on held-out GSM8K and MATH-500 (with per-task intervals and the ≥90% matched-query rule).
2. Dose–response across block sizes B=1/8/16/32 for the confidence-guided schedule, and the `ao_b32` versus `random_b32` contrast at matched block size — this contrast separates the mechanism (which order the model chooses) from the trivial explanation (order freedom itself, or near-sequential training bias).
3. Solved-set membership, per-query resolution statements using the recorded MDE, and the malformed/differential-missingness audit.

**What it is not:** no per-query predictor claim, no private-aggregation connection. **But BCI failing kills *targeted* intervention, not *global* intervention.** A per-query predictor exists to spend selectively — probe these queries, allocate budget there. Without it the project falls back to a uniform decoding policy, which is exactly what the ICML baseline's fix does anyway. This is structurally identical to the DP paper's outcome: detection replicated, severity did not, so you could flag but not rank. Here, aggregate closure and dose-response effects may hold while per-query prediction fails — the project can *change the policy but not triage*. That is a real result with a real deployment reading, and it does not stop Phase 2 (see the Phase 2 intervention draft). BCI is reported as a tested-and-failed per-query predictor; the negative association is itself the finding, and "local uncertainty closure is decoupled from global solution coverage" becomes a candidate theoretical conclusion.

**Reporting commitments (non-negotiable):** report the pooled stratified Spearman with CI and its pre-registered classification; report the MAE delta; report the zero-advantage fraction; report all five schedules; do not re-select k, do not substitute per-rollout accuracy, do not drop MATH-500, do not invoke the Phase 0 lexical result.

### Branch (a) — Gate 1 passes

Both conditions hold with CIs entirely above zero. Phase 2 forced-token intervention proceeds under the handoff's frozen prerequisites: ~50 queries, at most three candidate positions per query, forced top-1/top-2 token substitution, controls matched on initial entropy, position, and token frequency, with answer/path divergence as the outcome. Only a resolved divergence difference upgrades candidates to causal forks.

**Phase 2 logging requirement (recorded now, out of Phase 1 scope):** the DP-layer direction (N_eff / feasibility applied to token positions) requires per-position candidate distributions at every unmasking step, not just final generations. That cannot be retrofitted from completed Phase 1 runs. Do not modify Phase 1 to add it; the Phase 2 logging spec must include it from the start.

### Branch (c) — closure fails

Candidate-minus-control closure is not positive on held-out data (or a task is unresolved below 90% matched queries). Stop: no causal intervention, no private aggregation. The paper is a mechanism null with the descriptive tables; the schedule-vs-closure premise itself is the falsified claim.

## 5. Interpretation guardrails

- **Resolution:** per-task correlation MDE ≈ 0.277, pooled ≈ 0.197 (Fisher-z planning approximations at n=100/200, alpha=.05, 80% power); paired-effect MDE ≈ 0.280. These are planning quantities, not substitutes for the query-clustered bootstrap intervals; a seeded simulation of the exact stratified recipe is emitted by the analyzer's `power-simulation` mode and referenced in the Gate 1 report. A null whose CI lies inside these bounds is unresolved, not evidence of absence.
- **Pre-registered resolution prediction:** with a calibration association of −0.1969 and an empirical pooled CI half-width of ±0.122 (at ρ≈0.3), a held-out association near the calibration value would produce a CI entirely below zero and therefore be classified — under the pre-registered verdict taxonomy — as a **resolved wrong-direction falsification, not an unresolved null**. Only a held-out point estimate within roughly ±0.12 of zero would be unresolved. This prediction is recorded before unblinding.
- **Zero-advantage fraction:** a high fraction of exactly-zero query-level advantages (60% on calibration) attenuates the correlation mechanically; report and interpret it, do not use it to excuse a null.
- **Differential missingness:** if the malformed-output rate differs materially between `ar_b1` and `ao_b32` (or across policies), the coverage advantage may be inflated by missingness rather than mechanism; the frozen valid-output sensitivity is the designated remedy and runs automatically above the 1% threshold.
- **Parser semantics (verified at the frozen JustGRPO commit, pre-unblinding):** `parser_status == "ok"` means the frozen extractor ran without raising — the extractor can return an empty string, so "ok" is not by itself a guarantee of a numeric answer. Measured on all 16,000 screening records: 0 null `parsed_answer`, 20 empty-string answers (0.125%, 1–7 per task-policy cell), 15,980 non-empty. Empty-string answers are not malformed under the frozen definition; they remain in the intention-to-decode denominator and grade as incorrect, which is the conservative direction. The empty-string rate is slightly higher for `ar_b1` than `ao_b32` (math500: 7 vs 1 per 1,600), so this artifact mildly deflates the AR advantage — it works against the hypothesis, not for it, and does not confound in the hypothesis-favoring direction. **Sign, not magnitude:** at these counts, 7/1,600 versus 1/1,600 is within ordinary sampling noise, so the differential must not be over-interpreted; the paper states it as "the one measurement artifact we found works against our claim," not as a precisely measured effect. The frozen malformed definition is deliberately left unchanged — the reclassification it would buy is 0.125% in the already-conservative direction, and documenting costs less credibility than amending.
- **No-rescue rules (protocol §7):** no favorable-k selection after seeing results; no per-rollout accuracy substitution; no redefining candidates with held-out correctness; no dropping a task or policy that contradicts expectations; Phase 0 lexical results are not confirmatory evidence.
- **Unit of inference:** the query remains the independent unit. Token counts and rollout counts are never treated as independent sample sizes.

## 6. Process

Extension completion ETA is derived from measured per-lane throughput and recorded in the operations report; Gate 1 analysis, snapshot backup, and verdict delivery are scheduled in the buffer before rebuttal-period work, never under deadline pressure, and partial-marker data are never analyzed even informally. The lane-loss minimum-viable rule and the backup-redundancy checklist are stated in the operations report.

**Phase 2 and timing:** the entropy-first (anti-confidence) decoding-schedule intervention is drafted in `protocols/phase2_intervention_protocol_draft_20260830.md` — one policy plus a matched rollout budget, endpoints/tolerances/readings to be frozen before any Phase 2 inference, and a logging spec (full candidate distribution or top-m logits with normalizer at every unmasking step) that makes the DP-layer N_eff analysis computable offline. The binding scheduling rule: **no Phase 2 inference until the Gate 1 report is delivered and the branch decision is made** — the post-negative-result temptation to immediately run something that might work better is the scope-expansion failure mode this project has demonstrated repeatedly. Nominal timeline: Phase 2 inference September–October, analysis before November 11, December for writing, ICML 2027 late-January submission. Papers stay separate: paper two is the decoder-order mechanism study (Phase 1 + intervention), paper three is DP generation feasibility — merging them recreates the two-studies-spliced problem.

## 7. Post-unblinding interpretation addendum (2026-09-01, after the collaborator review)

*Status: these statements are post-unblinding interpretation, written after the frozen report was generated and committed (`7022aa0`). They do not modify any pre-registered reading; the pre-registered readings were applied mechanically by the analyzer. Where a statement goes beyond the pre-registration, it is labeled as such.*

### 7.1 Condition 2b: magnitude in interpretable units

The pre-registered taxonomy correctly classifies the MAE delta CI [−0.000302, −0.000037] as entirely below zero. The honest substantive statement, stated alongside: **adding BCI makes prediction very slightly worse — δ = −0.000169, which is 0.18% of the pooled baseline MAE (0.092417)** — and the interval excludes zero only because the estimate is extremely precise, not because anything of practical size happened. Statistically resolved, practically indistinguishable from zero. This is the mirror image of the tiny-effect point the ICLR review made about paper one: an effect can be resolved and still be nothing.

### 7.2 The contrast's resolution — what the study could have found

The pooled `ao_b32 − random_b32` CI is [−0.0421, +0.0290], half-width ≈ **±0.036** (gsm8k ±0.044, math500 ±0.055). Effects smaller than that were never detectable at 16 rollouts per policy. The straddling-zero verdict is therefore read as: no resolved difference at this resolution, not as evidence that the two schedules are equal. A null reported with its detection floor is interpretable; a bare null is not.

### 7.3 The natural spine: diversity versus precision, and divergence from ICML

Steps 2 and 3 together carry the paper now. `random_b32` gives the highest pass@16 on both tasks (gsm8k 0.99, math500 0.62) and the lowest pass@1 (0.646 / 0.284) — a diversity-versus-precision trade in its cleanest form: random scheduling produces more varied trajectories, so coverage improves while single-shot accuracy degrades. At 64 rollouts, AR beats AO on CoverageAUC on both tasks (gsm8k +0.047, math500 +0.041), so order freedom does cost something; and `ao_b32` versus `random_b32` at matched block size is indistinguishable (−0.0059, CI crossing zero), so the cost is not attributable to confidence-guidance.

**This is a divergence from the ICML result, not a replication failure.** ICML removed order flexibility and trained with GRPO; this study held training fixed and varied the schedule at inference time. The finding — at matched block size, *which* positions get chosen does not measurably matter, while *having* the freedom does — is compatible with ICML and locates the mechanism differently. That is the contribution, and the paper's spine: a mechanism study at inference time, without training.

### 7.4 Consequence for the Phase 2 branch decision (recorded for the collaborators)

The confidence-vs-random contrast came back null, which lowers the prior on the entropy-first intervention: its premise is that which positions are resolved first matters, and the data just failed to detect that at matched block size. Entropy-first remains informative (it actively inverts the ordering, a stronger intervention than random), and a null there would consolidate the order-freedom-per-se account — but the frozen prediction must be written with the lowered prior, including the pre-specified reading for "entropy-first also indistinguishable." The alternative, cheaper and arguably better-targeted, is a **properly powered block-size dose-response at fixed confidence policy**, since block size is the actual axis along which order freedom varies. Both options, with pre-specified endpoints and readings, are drafted in `protocols/phase2_intervention_protocol_draft_20260830.md` (§9). The decision is the collaborators' to make, from this report; endpoints and readings are frozen before any Phase 2 inference, and the scheduling rule stays in force until the decision is delivered.
