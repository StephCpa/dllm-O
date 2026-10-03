# Progress After the 2026-10-03 Pre-Review

## 1. Overall Status

The manuscript is now a submission-shaped ACL-format paper rather than a claim
of a universally optimal decoding policy. Its central message is deliberately
narrow:

> Under matched neural-function-evaluation budgets, decoding schedules change
> repeated-sampling returns; the observed policy ranking depends on the declared
> sampling budget and temperature, and aggregate curves alone do not identify a
> diversity mechanism.

The new pre-review confirms that the experimental workflow is unusually careful,
but also identifies the main remaining risk: the rigor of the process is
currently stronger than the scientific increment established by the current
evidence. The highest-value next steps are therefore independent validation and
mechanism separation, not additional polishing of the existing point estimates.

No new GPU experiment has been launched from the current draft protocols.

## 2. Manuscript Revisions Completed

The latest manuscript revision has already incorporated the main low-risk
recommendations from the pre-review:

- The introduction explicitly treats *The Flexibility Trap* as the nearest
  prior work and no longer presents arbitrary-order degradation as a new
  phenomenon.
- The contribution list has been reduced to three calibrated contributions:
  controlled testing of eligible-set width, boundary analysis of the
  random-ranking effect, and an estimand/mechanism analysis.
- The paper now distinguishes eligible-set width from simultaneous commitment
  and from throughput acceleration. Every condition commits one token per NFE
  under 256 matched NFEs.
- The decoder description now states that confidence-first (CF) samples a
  candidate token at every eligible position, ranks positions by the
  untempered probability of their own sampled candidate, and commits the
  winning candidate. This makes the position-selection and candidate-filtering
  coupling explicit.
- The manuscript records the implementation difference between the public
  decoder and this study: the public implementation computes confidence in the
  model's bfloat16 dtype, whereas this study uses float32 confidence scores.
- A proposition now states the necessary condition for an aggregate Pass@k
  crossing: if one policy has at least as high a per-query success probability
  on every query, its expected coverage cannot be lower at any k. Therefore a
  crossing requires heterogeneous query-level advantages. The text explicitly
  labels this as a necessary condition, not a mechanism theorem.
- The title and section headings no longer imply that the best policy switches
  globally. The current wording describes a pairwise CF--RND ranking change;
  sequential decoding retains the highest high-budget point estimates.
- Temperature results are reported as a frozen non-replication at T=0.9 and a
  direction reversal relative to T=0.6 at T=1.2, without claiming universal
  temperature invariance.
- The evidence ledger is in the main text and records selection history for the
  block-size study, entropy-first study, random extension, temperature
  extension, and post-outcome audits.
- The majority-vote paragraph explicitly states that the current reported
  values use fixed prefixes, whereas Pass@k uses a subset-based estimator. The
  distinction is not treated as a contradiction.
- The limitations now state that the random-ranking effect has not been tested
  on unseen problems and that CF position selection has not yet been separated
  from candidate filtering.

## 3. Completed Evidence

### 3.1 Block-size intervention

- The study compares `B=1, 8, 16, 32` under matched 256 NFEs and one committed
  token per NFE.
- The resolved adjacent contrast is confined to `B=1 -> 8`; `B=2` and `B=4`
  were not measured, so the onset inside that interval is not identified.
- The wider eligible set lowers multi-sample coverage in the primary contrast,
  while the Pass@1 loss is not resolved.
- The effect is concentrated in a minority of queries; the fragility audit is
  explicitly labeled outcome-selected and descriptive.

### 3.2 Random-ranking extension

- The original random arm had 16 rollouts. After an initial crossing was
  observed, 48 additional rollouts were generated under a frozen follow-up
  protocol.
- The final 64-rollout estimator combines the original 16 and the new 48, so
  it is prospective only for the new outcomes and is not an independent
  replication.
- At `T=0.6`, the pooled high-budget random-minus-CF contrast is positive and
  small. Task-specific intervals cross zero.
- Sequential decoding retains the highest high-budget point estimates; the
  evidence supports a scoped pairwise ranking change, not random-policy
  superiority.
- The 48-outcome independence audit indicates that the result is not carried
  entirely by the first 16 outcome-selected rollouts, but it does not upgrade
  the study to an independent validation.

### 3.3 Temperature extension

- A second follow-up was selected after review and frozen before evaluating new
  outcomes at `T=0.9` and `T=1.2`.
- The frozen reversal criterion is not resolved at either temperature: the
  `T=0.9` high-budget contrast is inconclusive, while the `T=1.2` high-budget
  contrast has the opposite sign from the `T=0.6` result.
- The exploratory interaction analysis shows a large differential change in
  random-minus-CF Pass@1 at high temperature, while CF's own change is small
  over the tested grid. This is reported as descriptive evidence, not as a
  proof of invariance or causality.
- Trace diagnostics are consistent with CF selecting relatively low-entropy
  positions, but trajectories differ across policies and CF also filters the
  committed candidate. The mechanism remains non-identified.

### 3.4 Entropy-first and diversity

- Entropy-first increases parsed-answer diversity but does not improve the
  frozen aggregate coverage endpoint.
- Parsed-answer entropy is explicitly distinguished from semantic reasoning
  path diversity.
- The result is used as a negative finding against treating surface answer
  diversity as a sufficient proxy for correct-answer coverage, not as a general
  theorem that diversity is harmful.

### 3.5 CPU-only analyses already available

The repository contains CPU analysis code and tests for:

- block-size fragility and leave-one-query-out influence;
- query-level wins/losses and joint success-count structure;
- rollout-split heterogeneity checks;
- random-subset majority voting;
- exact-string and mathematical-equivalence answer grouping;
- temperature interaction and trace diagnostics;
- multiplicity audits for per-k contrasts.

The problem-level reanalysis script is implemented, but it has not yet been run
on the server's raw rollout records in this work cycle.

## 4. Code and Reproducibility Status

- The latest code branch contains the `low_confidence_resample` decoder mode,
  which preserves CF's position selection but redraws the committed token from
  the same tempered distribution.
- Decoder unit tests verify that the new mode is identical to CF at zero
  temperature and behaves as expected on a controlled toy distribution.
- The latest branch passes `146` Python tests.
- `make all` completes successfully and produces a 13-page ACL-format PDF.
  After `b5a61c7`, the Conclusion ran two lines onto page 9; two redundant
  sentences in Section 7 were removed, and the main text again ends on page 8.
- `make check` on a machine outside the pinned environment is expected not to
  pass. It reports `STATUS: FAIL` only when figures or macros differ, as with
  an unpinned plotting stack. When only the TeX build differs, it reports
  `PDF NOT VERIFIED` (exit 3). In the pinned environment, the rebuilt PDF and
  refreshed manifest pass with `STATUS: PASS`. Final verification must run in
  the pinned environment.
- The wording-only manuscript tightening is commit `b5a61c7`, now pushed. It
  did not rebuild `main.pdf` or refresh `build_manifest.json`; a follow-up
  commit does both.

## 5. Main Remaining Scientific Risks

### 5.1 The random effect is not validated on unseen problems

This is the most important unresolved issue. The current positive pooled
random-minus-CF result was selected after an observed crossing and uses the
same 200-query panel. The proposed E1 validation uses a fresh problem panel
from the unused GSM8K/MATH-500 pool.

Approximate planning figures are:

| Design | Arms | New problems | Generations | Approx. GPU-hours |
|---|---|---:|---:|---:|
| Minimum exploratory validation | CF + RND + sequential | 200 | 38,400 | 277--342 |
| Recommended validation | CF + RND + sequential | 400 | 76,800 | 555--683 |
| Maximum two-arm validation | CF + RND | 700 | 89,600 | 647--796 |

These are planning estimates, not completed results. The current protocol
draft recommends at least 400 new problems if the goal is useful power near a
shrunk effect size.

### 5.2 CF mechanism remains confounded

CF's sampled candidate determines both the position score and the token that
is committed. Thus a trajectory-level comparison cannot tell whether the
coverage cost comes from position ordering, candidate filtering, or their
interaction. The proposed E2 CF-resample control addresses this directly by
keeping the CF position choice and redrawing only the committed token.

### 5.3 Prior-work and implementation alignment

The public JustGRPO implementation and this study are close but not identical
because of float32 versus bfloat16 confidence computation. The direction and
size of any resulting position-selection discrepancy have not yet been
measured. A stored-state alignment audit should precede any interpretation that
depends on exact reproduction of the nearest-neighbor implementation.

### 5.4 Majority-vote comparability

The manuscript currently reports the prefix-based majority-vote analysis and
clearly distinguishes it from subset-averaged Pass@k. The CPU reanalysis should
report random-subset majority voting and mathematical-equivalence grouping
before the final manuscript claims any practical implication.

## 6. Pending Work, Ordered by Value

### Immediate CPU work, no GPU required

1. Run `scripts/problem_level_reanalysis.py` on the server's raw records.
2. Export the random-subset majority-vote and mathematical-equivalence results
   with fixed seeds and bootstrap metadata.
3. Check every new result against the evidence ledger before changing the main
   text. Post-outcome analyses must remain labeled exploratory.

### Short GPU job before E2

- Run the float32/bfloat16 CF position-selection agreement audit on stored
  masked states, if the required trace fields are available. It needs one
  forward pass of the 8B model per state, so it is GPU work, not CPU work.
  E2's one-step diagnostic has the same requirement.

### First GPU experiment when resources are stable: E2

E2 is the highest-value mechanism experiment per GPU-hour:

- 12,800 new generations;
- CF-resample at `T=0.6` and `T=1.2`;
- approximately 92--114 GPU-hours;
- no extra neural function evaluation per rollout;
- requires a frozen runner, artifact schema, CF-preservation gate, and smoke
  audit before formal launch.

The current draft uses `delta_D=0.05` as a frozen verdict threshold for the
"filtering is not needed" reading of E2a, not as a descriptive quantity. It
remains an author decision and must be justified and frozen before outcome
generation.

### Second GPU experiment: E1

E1 is the highest-value generalization experiment:

- fresh, never-used problems;
- same model, prompt, parser, grader, and temperature as the main study;
- recommended 400-problem CF/RND/sequential design;
- frozen primary high-budget RND-minus-CF endpoint;
- no correctness read before the split manifest, code, smoke, and freeze are
  recorded.

E1 is more expensive than E2 but directly addresses the strongest reviewer
objection to the current random-ranking result.

### Lower-priority extensions

- `B=2,4` to localize the onset between sequential and blockwise any-order
  decoding;
- a second model replication;
- a length/completion audit or `L=512` condition;
- development-set temperature tuning if deployment optimization becomes a
  central claim.

These should not displace E1 or E2 unless new resources make them essentially
free.

## 7. Evidence Discipline for the Next Revision

The manuscript should preserve the following labels:

- **Primary/confirmatory:** the frozen block-size endpoint and its declared
  adjacent comparisons.
- **Scoped prospective follow-up:** the random and temperature extensions,
  which were frozen before their new outcomes but selected after earlier
  evidence or review.
- **Exploratory/post-outcome:** per-k decompositions, fragility, trace
  diagnostics, diversity plots, interaction contrasts, and CPU reanalyses
  selected after outcomes were available.
- **Not yet evidence:** all E1/E2 protocol predictions, power calculations,
  and mechanistic hypotheses before formal inference.

The paper is strongest when it reports nulls and non-replications as results,
does not convert point estimates into policy dominance, and keeps the selection
history visible. The next experimental additions should strengthen that same
discipline rather than add more post-hoc diagnostics to the existing panel.

## 8. Recommended Near-Term Sequence

1. Finish and archive the CPU-only E3 analysis.
2. Freeze and implement the E2 runner/schema; run smoke, then formal E2 on two
   GPUs if the gate passes.
3. Reassess whether E2 changes the mechanism paragraph before launching E1.
4. Freeze the fresh-problem E1 manifest and run the recommended validation when
   sufficient GPU capacity is available.
5. Regenerate the manuscript numbers from audited artifacts, rebuild in the
   pinned environment, and perform the final reference, anonymization, and
   artifact audit.

