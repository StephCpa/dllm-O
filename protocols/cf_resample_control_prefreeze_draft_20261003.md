# E2: CF-Resample Control — Pre-Freeze Draft

**Protocol version:** `cf-resample-freeze-v1` (proposed)  
**Status:** draft for author review (2026-10-03 UTC). Not frozen and not
authorized for GPU launch. Items marked **[author decision]** or **[bind at
freeze]** must be resolved first.  
**Selection status:** a new mechanistic intervention, proposed after all
existing outcomes were read. It is prospective only for the new arm's
outcomes, and it is not an independent replication of any earlier result.

## 1. Question

Confidence-first (CF) draws one candidate token per eligible position, ranks
positions by the untempered probability of their own candidate, and commits
the winning candidate. The same random draw therefore decides both *where* to
commit and *what* to commit. Even when all positions share one distribution,
this filters the committed tokens: with `P(a)=0.9`, `P(b)=0.1`, and `B`
eligible positions, CF at `T=1` commits `a` with probability `1 - 0.1^B`
instead of `0.9`.

This study asks: **how much of CF's behaviour — its temperature stability and
its high-budget coverage cost — comes from this candidate filtering, rather
than from the choice of position?**

## 2. The control arm

`low_confidence_resample` (implemented in
`src/dllm_order_transmission/decoding.py`, with tests in
`tests/test_decoding.py`) ranks and selects positions exactly as CF does. It
then discards the winning candidate and commits a fresh draw from the same
tempered distribution at the selected position. The extra cost is one
categorical draw per step and no extra NFE. At `T=0` the control is identical
to CF, and a test verifies this.

Unit tests also verify, on a toy model with identical position distributions
(`P(a)=0.9`, `B=8`, `T=1`), that CF commits `a` in 100% of 600 runs while the
control commits it in 90.2% (theory: 90%).

## 3. Design

- Same model, 200 held-out problems, prompts, parser, grader, and settings as
  the temperature extension (256 positions, 256 steps, one token per NFE,
  CFG 0). CF confidences are computed in float32.
- **New cells:** CF-resample at `T in {0.6, 1.2}`, with 32 rollouts per problem
  and the existing seed keys. That is `2 x 200 x 32 = 12,800` generations.
- **Reused controls:** CF, RND, and Sequential at `T=0.6` (first 32 rollouts)
  and at `T=1.2`, bound by a no-outcome-read inventory.
- **Gate:** before launch, regenerate a few stored CF records with the new
  commit and confirm that they are bit-identical to the stored ones. This shows
  the code change left CF untouched.

## 4. Frozen endpoints

The two endpoints form one family with a nonstudentized maximum-deviation
simultaneous 95% band (10,000 task-stratified paired-query bootstrap
replicates, seed **[bind at freeze]**):

- **E2a (temperature sensitivity, Pass@1).**
  `D = [P1(CFr, 1.2) - P1(CFr, 0.6)] - [P1(CF, 1.2) - P1(CF, 0.6)]`.
  For reference, the same contrast for RND against CF is `-0.361` (unadjusted
  `[-0.402, -0.321]`).
- **E2b (high-budget coverage at T=0.6).**
  `HB(CFr, 0.6) - HB(CF, 0.6)`, with `HB = mean(Pass@16, Pass@32)`.

### Readings, fixed before outcomes

- E2a band entirely below zero: **filtering contributes to CF's temperature
  stability.** The size of `D` relative to RND's `-0.361` is reported
  descriptively.
- E2a band within `±delta_D`: **filtering is not needed for the stability**,
  within that tolerance. The proposed `delta_D = 0.05` Pass@1 points (about 14%
  of the RND gap) is an **[author decision]**, to be justified substantively
  before the freeze.
- Otherwise: `not_resolved`.
- E2b band entirely above zero: **filtering contributes to CF's high-budget
  coverage cost.** Band entirely below zero: the control loses more coverage.
  Otherwise: `not_resolved`.

## 5. One-step diagnostic under identical states (secondary)

Whole rollouts diverge, so rollout-level differences mix the selection rule
with state evolution. As a secondary analysis, take about 1,000 masked states
reconstructed from stored CF traces at `T=0.6` and `T=1.2`. On each state, run
one forward pass and use Monte Carlo over candidate draws, which needs no
further model evaluations, to compare CF and CF-resample on:

- the expected untempered probability of the committed token;
- the probability that the committed token is the argmax token;
- the entropy of the committed-token distribution at the selected position.

The two rules share the identical state, so these differences isolate
candidate filtering. This is descriptive and cannot change the verdicts above.

## 6. Power note

From the temperature report, the standard error of a CF Pass@1 temperature
change is about 0.008, so `D` has a standard error of about 0.011. Effects
larger than about 0.035 should therefore resolve. RND's effect is ten times
that size. E2b has a standard error of about 0.02 and can resolve differences
of about 0.06 or more.

## 7. Resource estimate

About 92–114 GPU-hours for 12,800 generations, plus about 1,000 single forward
passes for the diagnostic.

## 8. Scope boundary

Even a decisive E2a result applies to this model, task panel, and pair of
temperatures. It separates candidate filtering from position choice under CF
only. Neither study changes any earlier frozen verdict.
