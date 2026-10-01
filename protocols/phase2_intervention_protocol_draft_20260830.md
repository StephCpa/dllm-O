# Phase 2 Intervention Protocol (Draft for Review — NOT FROZEN)

**Date:** 2026-08-30
**Status:** Draft. Nothing in this document authorizes inference. It becomes a frozen protocol only after Gate 1 is delivered, the branch decision is made, and the collaborator signs the freeze. The scheduling rule below is binding immediately.

## 0. Scheduling rule (binding now)

**Phase 2 inference does not start until the Gate 1 report is delivered and the branch decision is made.** After a modal-negative result the temptation is to immediately run something that might work better; that is how scope expands, and it is the one failure mode this project has demonstrated repeatedly. September–October is reserved for Phase 2 inference only if Gate 1 lands as planned (~Sep 1) and the branch decision authorizes it.

## 1. Motivation and causal claim

Phase 1 is correlational: schedules differ in coverage, and candidate positions close uncertainty, but nothing forces the mechanism. The natural next experiment is a decoding-schedule intervention: an **entropy-first (anti-confidence) unmasking schedule** at matched block size (32) and matched rollout budget — same queries, same model, same frozen decode conditions, everything identical except *which positions get unmasked first*.

**Falsifiable prediction:** if confidence-guided position selection is what costs coverage (the flexibility-trap mechanism), forcing the decoder to resolve high-uncertainty positions early should recover coverage relative to `ao_b32`, and should land at or above `random_b32`. If entropy-first performs no better than confidence-first, order freedom per se is the problem and the mechanism story is wrong.

**Updated prior (2026-09-01, post-Gate-1):** the held-out `ao_b32 − random_b32` contrast came back null (pooled −0.0059, CI [−0.042, +0.029]) at matched block size. The entropy-first premise is that *which* positions are resolved first matters; Phase 1 just failed to detect that at block size 32. The prior on the intervention working is therefore lower than when this draft was written. Entropy-first is still informative — it actively inverts the ordering, a stronger manipulation than random — and a null there consolidates the order-freedom-per-se account at higher strength than the current contrast. The frozen prediction (below) is written with this lowered prior.

This converts correlational findings into a causal claim, reuses the entire Phase 1 harness (runner, traces, analyzer, freeze machinery), and costs one policy plus one rollout budget.

**Positioning:** the ICML baseline removes flexibility and trains with GRPO. This experiment shows *at inference time, without training, which component of flexibility does the damage*. That is a mechanism contribution, not a method replication — the framing the mechanism paper leads with.

## 2. Scope and triggers

Phase 2 intervention runs only if:

1. Gate 1 is delivered and the branch decision is made (binding rule above); and
2. RQ1 closure holds (candidate-minus-control closure positive per frozen Gate 1 condition 1), and either the dose-response across block sizes or the `ao_b32` vs `random_b32` contrast is non-null (CI excluding zero).

A failed Gate 1 condition 2 (BCI does not predict per-query) does **not** block Phase 2: BCI failing kills *targeted* intervention (per-query triage), not *global* intervention. Aggregate closure and dose-response effects may hold while per-query prediction fails — the result is "change the policy but not triage," structurally identical to the DP paper's outcome (detection replicated, severity did not: flag but not rank). Phase 2's global intervention is exactly the uniform-policy fix the ICML paper applies.

## 3. Conditions (to be frozen before any Phase 2 inference)

| Condition | Block size | Unmasking rule | Role |
|---|---:|---|---|
| `ao_b32` | 32 | confidence-guided (lowest-uncertainty positions first) | Reference (reused Phase 1) |
| `random_b32` | 32 | random eligible positions | Order-freedom control (reused Phase 1) |
| `ef_b32` (entropy-first) | 32 | highest-uncertainty (eligible entropy) positions first, ties by position | **Intervention** |
| `ar_b1` | 1 | left-to-right | Sequential reference (reused Phase 1) |

Frozen decode conditions identical to Phase 1: 256 generated tokens, 256 denoising steps, temperature 0.6, CFG 0, mask token 126336, same seed derivation, same queries (held-out set), matched rollout budget. The `ef_b32` remasking rule is the confidence rule with the ordering inverted; a tie-break and any numerical tolerance (e.g., how "highest entropy" is selected among near-ties) must be specified exactly in the freeze.

### 3.1 Implementation decision: what "entropy-first" means (settled now, to be frozen)

There is an implementation ambiguity that determines whether the result is interpretable, and it is settled here rather than at inference time:

- **Entropy-first over currently *eligible* positions, ranking recomputed at every denoising step** — this is the policy that tests the mechanism.
- Alternative A — entropy-first over *all masked* positions: rejected, because positions outside the current block are not committable yet; ranking them mixes deferred-commitment structure with the eligibility structure and no longer isolates the selection mechanism.
- Alternative B — fixing the unmasking *order in advance* (a static ranking computed once at generation start): rejected, because the flexibility-trap mechanism is specifically about deferring high-uncertainty positions *until surrounding context has collapsed*; a static order removes exactly the step-wise interaction with accumulating context that the mechanism predicts. Resolving high-entropy positions early *while context is still open* requires recomputing the ranking each step.
- Both rejected alternatives are recorded here so the freeze can state "we picked this one for this reason" rather than "we picked one." The per-step recomputation also mirrors `ao_b32` (confidence ranking recomputed per step over eligible positions) and `random_b32` (random eligible selection per step), so the three conditions differ only in the ranking rule — the isolation the contrast requires.

## 4. Endpoints, tolerances, and readings (to be frozen before inference)

Pre-registered in the same style as the Phase 1 verdict taxonomy:

- **Primary:** `ef_b32` vs `ao_b32` coverage (CoverageAUC-style mean over k, matched rollout budget), per task and pooled, query-bootstrap 95% CI.
- **Pre-registered readings:**
  - `ef_b32` ≥ `random_b32` with CI excluding zero *and* `ef_b32` > `ao_b32`: confidence-guided selection costs coverage; the flexibility-trap mechanism holds in this setting.
  - `ef_b32` ≈ `ao_b32` (CI straddles zero): confidence-guidance is not the damaging component; order freedom per se is implicated; the mechanism story is wrong.
  - `ef_b32` < `ao_b32`: confidence-guidance is protective against entropy-first commitment; the trap mechanism fails in the opposite direction.
  - **`ef_b32` indistinguishable from both `ao_b32` and `random_b32` (all contrasts straddle zero):** order freedom per se is the implicated component, and which positions are resolved first does not measurably matter even under an inverted ordering — the order-freedom-per-se account is consolidated and the confidence-guidance mechanism is rejected at the design's resolution. (Added 2026-09-01 in response to the null Phase 1 contrast.)
  - CI straddling zero for all contrasts: unresolved at the recorded resolution — reported as such, with the empirical half-widths from the frozen recipe simulation.
- **Secondary:** per-k pass rates, solved-set membership, malformed/differential missingness (same frozen definitions as Phase 1).
- **Pre-registered length check (analogous to the Phase 1 empty-string audit):** entropy-first may produce systematically different generation-length distributions than confidence-first, and a length difference is a confound on any coverage metric sensitive to truncation. Before the primary contrast is interpreted, report the output-length distribution and the truncation rate per condition; if `ef_b32` differs systematically from `ao_b32` (or `random_b32`) on length or truncation, that difference is reported alongside the coverage contrast and the coverage reading is explicitly qualified — the check and its trigger rule are frozen before inference.
- Tolerance rules (resolution, missingness, abort conditions) copied from the Phase 1 protocol unless explicitly amended in the freeze.

## 5. Logging specification (gates the DP layer)

The DP-layer direction (N_eff / feasibility applied to token positions) requires per-position candidate distributions at each unmasking step — not just final generations. Phase 1 traces cannot be retrofitted; the Phase 2 logging spec must therefore include, **for every rollout and every denoising step**:

- the full candidate distribution over the vocabulary at each eligible position, **or at minimum the top-m logits with the log-normalizer** (so the full softmax is reconstructable), with m fixed in the freeze;
- the entropy and margin summary fields already present in Phase 1 traces (for continuity with the BCI/closure machinery).

With this, the DP feasibility analysis is computable offline from Phase 2 artifacts with no further GPU time: N_eff per position, composition arithmetic over sequence length, and identification of which positions dominate the privacy cost. If the mechanism story holds, those pivotal positions should be the same high-uncertainty positions the flexibility trap is about — the two threads meeting on the same object.

## 6. Paper separation (binding)

- **Paper two:** the decoder-order mechanism study (Phase 1 results + Phase 2 intervention).
- **Paper three:** DP generation feasibility (N_eff per position, composition, privacy cost).

Merging them recreates the two-studies-spliced problem this project already spent months fixing. The DP layer is not discussed in paper two beyond a forward pointer.

## 7. Timeline (nominal)

- Gate 1 delivered ~Sep 1, branch decision within the week.
- Phase 2 inference September–October; analysis lands before November 11.
- ICLR rebuttals November 11 – December 3 consume that window; December is writing time.
- ICML 2027 submission late January is comfortable under this schedule.

## 8. To be completed before freezing

- Exact `ef_b32` remasking rule with tie-breaks and tolerances, written to pass the same third-party-recomputability standard as the Phase 1 BCI spec.
- Rollout budget for Phase 2 (matched to the comparison conditions) and the shard/lane plan.
- Endpoint tolerances and abort rules, copying Phase 1 where not amended.
- Freeze SHA chain and analyzer binding, same as Phase 1.

## 9. Branch options for the collaborators' decision (2026-09-01, post-Gate-1)

The Gate 1 report is delivered; the decision below is the collaborators' to make, from that report. Endpoints and readings are frozen before any Phase 2 inference, and the scheduling rule (no Phase 2 inference before the decision is delivered) stays in force either way.

### Option A — entropy-first intervention (original)

As specified in §1–§4, with the updated prior and the added pre-specified reading for "entropy-first also indistinguishable" (§4). Rationale: the strongest possible test of the confidence-guidance mechanism; a null here consolidates order-freedom-per-se at higher strength than the Phase 1 contrast alone. Cost: one policy plus one matched rollout budget; the prior on a positive result is now lower.

### Option B — powered block-size dose-response at fixed confidence policy

Block size is the actual axis along which order freedom varies, and the Phase 1 screening window (16 rollouts per policy) showed no monotonic trend — underpowered for the question. A properly powered version tests the surviving hypothesis directly and is cheaper than Option A.

- Conditions: `ao_b8`, `ao_b16`, `ao_b32` at fixed confidence policy, plus `ar_b1` and `random_b32` references; rollout budget raised (e.g., 64 rollouts per policy, matching the formal primary budget) on the same held-out queries; same frozen decode conditions and seeds.
- Primary endpoint: CoverageAUC-style mean over k at the matched budget, per task and pooled, query-bootstrap 95% CI.
- Pre-specified readings:
  - Coverage declining with B (CI excluding zero): order freedom costs coverage in proportion to block size; the mechanism is freedom itself, and block size quantifies the dose.
  - Flat across B (CIs straddling zero at the recorded resolution): freedom per se is not measurably harmful at these block sizes — the surviving claim is the diversity-vs-precision trade, not a coverage loss; the ICML divergence sharpens into "their training-time effect does not reproduce as an inference-time block-size effect."
  - Non-monotonic: reported as-is; no post-hoc re-selection of B.
- Also pre-register the pass@1-versus-pass@16 diversity-precision trade as a secondary endpoint, since it was the cleanest signal in Phase 1.

### Decision rule

The collaborators pick A, B, both, or neither, from this report; whichever runs, the freeze procedure is the same as Phase 1 (endpoints, tolerances, readings, and the logging spec frozen before inference, analyzer bound by commit, invocation log on). Nothing in this section is frozen yet.

### 2026-09-01 branch decision (collaborator): **B, with A held conditional**

Recorded before any Phase 2 inference. B runs; A runs only under the pre-registered condition below.

#### B pre-freeze specification (to be finalized in the freeze; not yet frozen)

- **Conditions:** `ao_b8`, `ao_b16`, `ao_b32` at fixed confidence policy, plus **`ar_b1` as the B=1 endpoint of the same axis** (its 64-rollout CoverageAUC is already logged from Phase 1 — one endpoint of the dose-response costs zero new inference). Axis: **B ∈ {1, 8, 16, 32}**, same held-out queries, same frozen decode conditions and seed derivation, 64 rollouts per policy. New inference needed: `ao_b8` and `ao_b16` extension to 64 rollouts = 200 queries × 2 policies × 48 = 19,200 generations (~5–6 days on 3–4 lanes, comparable to the Phase 1 extension). `random_b32` at 64 rollouts is optional and only for the diversity-precision secondary; decide at freeze time.
- **Named primary metric (ambiguity resolved before freezing):** the monotonicity trend is defined on **CoverageAUC** — the mean over k ∈ {4, 8, 16, 32, 64} of unbiased Pass@k at 64 rollouts — the composite that resolved the AR-vs-AO gap in Phase 1 (+0.047 / +0.041). The divergence between pass@1 and pass@16 across B (the diversity-vs-precision shape already visible in Phase 1) is a **pre-registered secondary**, not the trend definition.
- **Primary endpoint:** pooled monotone trend in CoverageAUC across B ∈ {1, 8, 16, 32} (linear trend statistic over B, per task and pooled, query-bootstrap 95% CI; the exact statistic is fixed in the freeze). The **anchor contrast is B=1 vs B=32** (both endpoints already logged at 64 rollouts); adjacent-step contrasts (1→8, 8→16, 16→32) are secondary and reported with their detection bounds.
- **Resolution simulation for the B design (run pre-freeze, 2026-09-01, deterministic, from the Phase 1 64-rollout per-query data):** the pooled query-level SD of the 64-rollout CoverageAUC(AR−AO) difference is **0.181** (gsm8k 0.169, math500 0.193) — between-query heterogeneity dominates, so 64 rollouts do not shrink the paired CI as far as a clean rollout-noise scaling would predict. Pooled paired-difference half-width at n=200 is ≈ **±0.025** (per task ≈ ±0.035 at n=100). Consequences, recorded: a linear gradient spread over three steps (≈0.015/step) is **not** resolvable by adjacent-step contrasts; the **total B=1→32 contrast is resolvable** (observed gap ≈0.047 pooled vs ±0.025). The B design therefore pre-registers the **total contrast plus the trend statistic** as primary, and explicitly labels adjacent steps as underpowered for a smooth gradient. If the freeze needs more power on intermediate steps, the options are more rollouts or fewer conditions — decided now, not after data.
- **Conditional-A rule (pre-registered):** if B shows the pre-registered monotone gradient (trend statistic resolves in the pooled CI, or the total B=1→32 contrast CI excludes zero), **A becomes worth running** — a gradient means order freedom has structure, and entropy-first then tests whether that structure is about position choice after all. If B is flat (trend CI straddles zero at the recorded resolution), **A is close to pointless and is not run**; the paper is the diversity-vs-precision account plus two resolved nulls (contrast, dose-response) with stated detection bounds. This second decision is made by the pre-registered rule, not in the moment.
- **Timeline:** 19,200 new generations on 3–4 lanes ≈ 5–6 days; starting mid-September puts analysis well before November 11 (ICLR rebuttal window), December free for writing toward ICML late January, and leaves November–December room for A if B returns a gradient. The scheduling rule remains in force: nothing runs until the B freeze is finalized and the decision is delivered.
