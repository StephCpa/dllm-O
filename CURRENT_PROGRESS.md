# Current Progress

Last updated: 2026-10-03 UTC. All dates in this file are UTC.

## Completed experimental evidence

- Instrumentation and protocol checks for the decoding-study pipeline are
  recorded in the phase-1 and phase-2 reports.
- The held-out Condition A analysis found no resolved pooled advantage of
  entropy-first over confidence-first for the preregistered coverage endpoint;
  the confidence-vs-entropy position choice is therefore not identified as the
  source of the observed coverage loss.
- The matched block-size study supports a schedule-associated coverage cost,
  concentrated near the sequential-to-block-parallel transition.
- The random-64 extension filled the high-budget comparison gap. It found a
  resolved policy reversal across the budget range, while the aggregate
  full-grid contrast remains modest. This extension was selected after the
  first-16 results and is labeled post-Gate-1 exploratory evidence.
- The temperature extension shows a resolved differential change: confidence-
  first is comparatively stable across the tested temperature grid, whereas
  random and sequential decoding lose substantially more single-sample
  precision at high temperature. The extension was selected after review and
  is also labeled post-Gate-1 exploratory evidence.
- A fragility audit (2026-10-02) shows that the block-size effect is
  concentrated: removing the 9 most-hurt of 200 queries brings its interval to
  zero. This is outcome-selected and descriptive.
- CPU-only audits now cover query-level heterogeneity, leave-one-query-out
  influence, same-policy rollout-split checks, paired majority-vote contrasts,
  diversity summaries, and the temperature-extension diagnostics.

## Current manuscript state

- The active manuscript is `paper/main.tex` in ACL review format.
- The title currently emphasizes repeated-sampling returns rather than a
  universal policy-ranking law.
- The main text ends within the ACL review-style page budget in the current
  build; references and appendix material follow it.
- Figure 1 is a vector schematic drawn by `paper/figures/fig1_overview.py`.
  Panel (a) shows one committed token per NFE under every schedule; panel (b)
  is an explicitly non-empirical crossing sketch. An empirical Figure 1(b)
  candidate is included in `paper/figures/` for comparison only and is not
  wired into the manuscript because its curves overlap conceptually with the
  full empirical Figure 3.
- The manuscript build is bit-reproducible in the pinned environment
  (`paper/requirements-figures.txt`). `make check` rebuilds everything and
  compares it with `paper/build_manifest.json`.
- The 2026-10-01 revision (`paper/REVISION_NOTES_20261001.md`) checked every
  manuscript number against the reports. It reports the frozen trend
  endpoint, states the temperature result as a frozen non-replication,
  redraws Figures 2--5 at column width, and adds the temperature diagnostics
  to Appendix F.
- The current narrative avoids claiming that confidence-first is universally
  invariant to temperature, that exploratory selection improves coverage, or
  that one aggregate metric is sufficient without conditioning on budget and
  temperature.

## Immediate next steps

1. Decide whether Figure 1(b) should remain conceptual or use the optional
   empirical teaser after a final-size comparison.
2. Review the 2026-10-01 revision and respond to any external reviewer
   comments that arrived after the pre-review audit. Before submission, confirm
   that the authors completed the verification attested in the revised AI Use
   Statement (`paper/REVISION_NOTES_20261001.md`, Section 10).
3. Have the build verified on a second machine with `make check` in the
   pinned environment. Cross-environment validation is not yet complete.
4. Recheck references, anonymization, artifact links, and the target venue's
   current submission requirements before uploading.
5. Decide which follow-up experiments to run. Draft protocols, with power and
   cost, are in `protocols/*_prefreeze_draft_20261002.md`. Priority, confirmed
   by the independent verification: first the second-model replication of the
   block-size primary, then the intermediate block sizes `B=2,4`. The
   cold-sequential experiment is deferred. None is frozen or authorized for
   launch. Each first needs a finalized model and pinned revision, a decoder
   equivalence check where relevant, an emitted machine freeze, and a
   validation-only run.

## Important interpretation guardrails

- The tested setting is not a universal theory of all diffusion decoding or
  test-time scaling.
- Post-Gate-1 extensions are useful for hypothesis generation and robustness
  context, but they do not change the frozen primary verdicts.
- Candidate-token selection scores are policy-dependent quantities. In
  particular, random-selection scores must not be interpreted as calibrated
  model probabilities.
- A high Pass@k value is an oracle-style coverage statistic; it is not
  equivalent to majority-vote accuracy or deployment utility.
