# Manuscript Revision Notes (2026-10-01; updated 2026-10-03)

All dates in these notes are UTC.

This revision is a CPU-only consistency, presentation, and claim-scoping pass
over `main.tex`, its generated assets, and the build. No frozen endpoint,
interval, verdict, or protocol was changed, and no new inference was run. The
only new numbers in the manuscript are values that already existed in the
committed report JSON/Markdown and were previously unreported or hard-coded.

**Basis for the revision.** The request referred to "the latest reviewer
comments", but no comments were attached and none exist as GitHub issues or
pull requests in this repository. The revision therefore addresses the two
adversarial pre-reviews summarized in `ACL_PRE_REVIEW_GAP_AUDIT_20260928.md`,
plus an independent check of every number in the manuscript against
`reports/`. If newer reviewer comments exist, they still need a separate
response pass.

## 1. Corrections

| Issue | Change | Location |
|---|---|---|
| Figure 1(a) showed three tokens (`w1`, `w3`, `w6`) committed in a single "rank, then commit selected positions" step. That contradicts the one-token-per-NFE design that the paper rests on. The arrows (pos1, pos3, pos7) also did not match the committed tokens (w1, w3, w6), panel (b) read "Policy rankings reverses", and the whole figure was a JPEG raster. | Redrawn as a vector figure by a new script, `figures/fig1_overview.py`. It shows three successive NFEs that each commit exactly one token, marks eligible and ineligible masks explicitly, and draws panel (b) as an unmarked schematic labelled "illustrative; not data". | Figure 1 and its caption |
| Negative estimates inserted through generated macros used in running text printed with a hyphen instead of a minus sign (e.g. "It is -0.044"). | Every numeric macro is now wrapped in `\ensuremath`, and every point estimate is typeset in math mode. `\ConditionAHigh` is now signed (`+0.024`), matching the other intervals. | `scripts/generate_manuscript_assets.py`, Sections 4, 5, 9 |
| Figures 2–5 were drawn 7 in wide but placed in a single 3 in column, so their text printed at about 4 pt. | All data figures are now drawn at column width with 6–7 pt text. | `scripts/generate_manuscript_assets.py` |
| `make distclean` deleted `figures/*.pdf` and `figures/*.png`, including the hand-made Figure 1 that the build does not regenerate, so a clean rebuild failed. It also left four of the five generated `.tex` files in place. | `distclean` now removes only the generated figures and all `generated/*.tex`. A clean `make distclean && make` was verified. | `Makefile` |
| Abstract was 207 words (ACL guideline: at most 200). | Tightened to 199 words. | Abstract |

## 2. Reporting gaps closed

| Gap | Change | Source |
|---|---|---|
| The ledger lists the block-size trend as a frozen endpoint, but its estimate was never reported. | Section 4 now reports the frozen `log2(B)` slope, −0.0096 per doubling [−0.0147, −0.0049], and the exact decomposition that assigns 95% of its magnitude to the B=1-to-parallel step. | `phase2_optionB_results_20260906.md`, `posthoc_robustness.json` |
| The temperature section compared a 64-rollout functional over k∈{16,32,64} at T=0.6 with a 32-rollout functional over k∈{16,32} at T=0.9/1.2 without saying so. | Section 6 and the abstract now name each functional. Table 2 adds a descriptive T=0.6 reference row under the temperature functional (+0.027 high-budget; −0.095 at k=1). | `temperature_extension_analysis_v1.json` (`secondary_within_temperature_contrasts`) |
| The CF selected-position entropy (0.238/0.238/0.250) was reported without any comparison values. | The text now gives the eligible-position mean (≈1.26), RND (0.41→3.58), and sequential (0.28→3.12). A new Appendix F tabulates all trace entropies and the temperature diagnostics. Both tables use macros that were already generated but never used. | `temperature_extension_exploratory_v2.json`, `temperature_extension_analysis_v1.json` |
| The temperature mechanism depends on an implementation detail that the paper never stated. | Section 2 now says that temperature enters only through Gumbel-noise candidate sampling, while CF/EF rank positions with the untempered distribution (verified in `src/dllm_order_transmission/decoding.py`). | Section 2 |
| Majority-vote intervals were described only qualitatively. | The k=1 paired intervals are now given (Sequential−RND +0.075 [+0.005, +0.140]; CF−RND +0.090 [+0.030, +0.155]). | `query_level_exploratory_audits.json` |
| The query-level heterogeneity sentence was uninformative. | It now reports that the median query effect is zero and the 10th percentile is −0.26, i.e. the pooled effect is concentrated in a minority of queries. | `query_level_exploratory_audits.json` |
| The appendix multiplicity table used 50,000 bootstrap replicates while the text states 10,000, with no explanation of the small differences in the third decimal. | The caption now states the replicate count and the source of those differences. | Appendix C |

## 3. Claim scoping

- **Temperature result.** The abstract, introduction, contributions,
  Section 6, and conclusion no longer say the reversal "reverses at T=1.2".
  They now state the frozen verdict: the reversal criterion is met at neither
  temperature. They also say that the T=1.2 high-budget contrast is resolved
  in the opposite direction, and that the T=0.9 result is not evidence of
  equivalence. The contribution bullet is renamed from "temperature
  interaction test" to "frozen temperature non-replication", because no
  interaction test was frozen.
- **Sequential vs. random.** "Intervals overlap at k≥16" (an overlap of
  marginal intervals, which is not a test) is replaced by the paired
  RND−sequential statement.
- **Deployment relevance.** The Discussion now says that, among the four
  tabulated policies, sequential decoding has the highest majority-vote point
  estimate at every k≥8. It also says
  that CF's Pass@1 advantage over sequential is unresolved
  (+0.016 [−0.005, +0.037]).
- **Limitations.** Added the panel size (100 queries per task), the
  different 32-rollout functional, and the absence of prespecified
  equivalence margins.

## 4. Figures

- **Figure 2.** Panel (a) now shows the change relative to B=1 on a log2(B)
  axis. It shades the unmeasured B=2,4 region, draws dotted segments across
  that gap, and adds the frozen per-task P1 intervals (audit items 3 and 5).
- **Figure 3.** Panel (b) now shows both the pointwise and the seven-endpoint
  simultaneous bands, and shades the region of the frozen high-budget
  functional (audit item 6). Both bands come from the 50,000-replicate audit,
  so the figure matches Appendix C exactly.
- **Figure 4.** The redundant title was removed, and the policy labels were
  abbreviated to fit the column.
- **Figure 5.** The frozen high-budget window k∈{16,32} is now shaded.

## 5. Structure

- The entropy-first subsection ("Maximizing answer diversity is not
  sufficient") had been placed under the temperature section, although it
  concerns the T=0.6 policy comparison. It now closes Section 5.
- `Limitations` now follows the Conclusion, before the references. This is
  the ACL convention: it does not count toward the page limit. The main text
  still ends on page 8.
- The ledger table columns are now ragged-right, removing the forced
  hyphenation ("Prospective be-fore", "confir-matory").

## 6. Verification performed

- Every hard-coded number in the main text and appendix was checked against
  the committed reports. Values that are now drawn from the reports
  automatically: Pass@1 and Pass@16 by block size, the B=1→8 contrast, the
  P1/P2 dependence correlations, the trend endpoint, the temperature
  endpoints, the temperature interaction, and the CF trace entropies.
Status as of 2026-10-03 UTC:

- `make check` rebuilds everything from scratch without errors, undefined
  references, or overfull boxes. All figures, generated macros, and `main.pdf`
  then match `build_manifest.json` bit for bit (Section 9).
- The 146 repository tests pass (`python -m pytest`, run 2026-10-03 UTC): the
  original 123, plus 4 for the fragility audit, 10 for the build-manifest
  checker in `paper/scripts/`, 3 for the CF-resample decoder mode, and 6 for
  the problem-level reanalysis.

## 7. Open decisions for the authors

1. **AI Use Statement.** It now discloses Anthropic Claude alongside OpenAI
   Codex, because this revision was prepared with Claude. Please confirm the
   wording.
2. **Figure 1(b).** It remains conceptual, which matches the current plan in
   `CURRENT_PROGRESS.md`. The empirical candidate in
   `figures/fig1b_empirical_candidate.*` is still unused.
3. **Figure 4.** It still has no uncertainty intervals (audit item 7). A
   per-query c/n ECDF would need the per-query success-count vectors. The
   committed reports hold only their quantiles.
4. **Venue checks.** Before submission, recheck the venue's CFP, checklist,
   and reference metadata (`REFERENCE_AUDIT.md`).

## 8. Addendum (2026-10-02): fragility audit and follow-up protocols

**Fragility audit (CPU-only, post-outcome exploratory).** The new script
`scripts/block_size_fragility_audit.py` reads the per-query effects already
committed in `reports/query_level_exploratory_audits_20261001/`. It writes
`reports/block_size_fragility_audit_20261002/`.

- 134 of 200 queries change by less than 0.01.
- The ten most-hurt queries carry 71% of the summed block-size effect.
- Removing the 9 most-hurt queries (5 GSM8K, 4 MATH-500) brings the interval
  to zero, and removing 17 makes the point estimate nonnegative.

Section 4 now states this in one sentence through generated macros, and the
Limitations section notes it. The removal is outcome-selected, so it describes
concentration rather than testing the effect. Its practical message is that new
queries or a second model are the informative robustness checks. Four unit tests
cover the script (`tests/test_block_size_fragility_audit.py`).

**Draft follow-up protocols.** These are not frozen and not authorized for
launch. Each lists the decisions the authors must make before freezing, along
with launch gates, approximate power, and cost.

| Draft | Question | New generations | Approximate power |
|---|---|---:|---|
| `protocols/second_model_replication_prefreeze_draft_20261002.md` | Does the frozen B=32 vs. B=1 contrast replicate on a second dLLM? | 25,600 | 0.93 if the effect matches LLaDA; 0.41 at half size |
| `protocols/intermediate_block_size_prefreeze_draft_20261002.md` | Where inside [1, 8] does the cost appear? | 25,600 | 0.95 if the whole drop happens at B=2; 0.35 at half |
| `protocols/cold_sequential_prefreeze_draft_20261002.md` | Can a colder sequential decoder match CF's Pass@1 while keeping more coverage? | 6,400 | 0.66 at best on the current panel, so probably inconclusive |

In the existing data, between-query variance is 20–50 times the binomial
rollout reference. In all three designs, adding queries therefore raises power
far more than adding rollouts.

## 9. Release cleanup after independent verification (2026-10-03)

An independent verification of commit `08a179b` confirmed the test suite, the
clean rebuild, the 8-page main text, the corrected Figure 1, the labelling of
the fragility audit, and the unfrozen status of the protocols. It listed three
items to address before submission.

**1. Figures were not bit-reproducible across environments.** In the
verifier's environment, a rebuild changed Figures 2–5. For example, Figure 2's
PNG went from 679×404 to 637×406, and the PDF page boxes changed. Nothing
pinned the plotting stack, so a different Matplotlib build or local font
settings changed the text extents. Fixes:

- `requirements-figures.txt` pins the exact plotting packages.
- `scripts/plot_environment.py` resets Matplotlib to its built-in defaults,
  pins the bundled DejaVu fonts, and warns when installed versions differ from
  the pins. Both figure scripts use it.
- The Makefile fixes `SOURCE_DATE_EPOCH`, so pdfTeX writes the same dates and
  `/ID`. Before this change, two clean builds of `main.pdf` differed; after it
  they are identical.
- `build_manifest.json` and `scripts/build_manifest.py` back the new
  `make check` and `make manifest` targets (see `README.md`).

In this environment the committed figures were already the pinned output, so
the style reset left them byte-identical. Three checks, all run by Claude in a
single container (one TeX Live 2023 installation), support the fix:

- **Fresh environment.** A fresh checkout in a fresh virtual environment with
  only `requirements-figures.txt` installed reproduces all 15 figure and macro
  hashes and `main.pdf`.
- **Conflicting local settings.** The result is the same with a deliberately
  conflicting `matplotlibrc` (sans-serif, 20 pt, thick lines) that was
  confirmed to be loaded.
- **Negative control.** With matplotlib 3.10.9, Figure 2's PNG comes out at
  637×406, the size the verifier saw, so that environment most likely had
  matplotlib 3.10. Here the scripts warn about the version mismatch, and
  `make check` fails and names the mismatch as the cause.

**2. Outdated records.** Section 6 now gives the current test count, and this
file's title and `CURRENT_PROGRESS.md` carry the current date.

**3. AI Use Statement.** Claude was involved, so the statement should be
retained. For the authors' confirmation, here is what Claude did in this
revision:

- checked every manuscript number against the committed reports;
- drafted revised manuscript text, including the abstract, parts of the
  introduction and contributions, Sections 4–6, the Discussion, Conclusion,
  Limitations, figure captions, and Appendix F;
- wrote or rewrote the figure scripts (Figure 1 and Figures 2–5), the macro
  generator changes, the Makefile, and the build-manifest tooling;
- designed and ran the post-outcome fragility audit, including its script and
  tests, at the authors' request;
- drafted the three follow-up protocols.

The current wording ("grammar checking, linguistic polishing, ... figure
scripting ...") understates the drafted text and the fragility analysis. The
sentence "All research design, experiments and analysis were conducted by the
authors" is accurate only if the authors adopt the AI-drafted analysis and
protocols as their own after review. Suggested wording, for the authors to
accept or edit:

> During research and manuscript preparation, the authors used OpenAI Codex
> and Anthropic Claude for grammar checking and linguistic polishing, drafting
> revisions of manuscript text, standardizing reference formatting, checking
> reported values against the audited result files, writing figure and
> post-outcome audit scripts, and code debugging and implementation
> suggestions. The study design, all experiments, and all frozen analyses were
> conducted by the authors. AI-drafted text, scripts, and exploratory analyses
> were reviewed, verified, and revised by the authors, who take full
> responsibility for all content of this paper.

At this stage `main.tex` was left unchanged on this point, because this is an
attestation that only the authors can make. Section 10 records the wording that
replaced it.

**Experiment priorities.** The verification agrees with running the
second-model replication first and `B=2,4` second. It recommends deferring the
cold-sequential experiment, which needs a new non-inferiority margin and will
probably end `not_resolved`. Its protocol is now marked deferred. None of the
protocols can be executed as written. Each first needs a finalized model and
its pinned revision, a passing decoder equivalence check where relevant, an
emitted machine freeze, and a validation-only run.

## 10. Second verification round (2026-10-03 UTC)

A second independent review of commit `a0a0fe6` accepted the main body of the
build fix. Its local Matplotlib 3.10.7, against the pinned 3.11.2, supports the
environment-drift explanation. The review asked for three further changes.

**PDF checker guardrails.** The reviewer showed that under a different TeX build
the checker returned exit code 0 even when `main.pdf` had changed arbitrarily.
Claude reproduced this: with the old checker, a manifest recording another TeX
build plus an inserted sentence still gave exit 0. `scripts/build_manifest.py`
now behaves as follows:

- **Recorded build.** Under the recorded TeX build and `SOURCE_DATE_EPOCH`, any
  difference in `main.pdf` is a failure (exit 1).
- **Other build.** Under another TeX build or epoch, the checker compares the
  page count and the per-page normalized text with the manifest. The line-number
  gutters are cropped away, and whitespace, hyphenation, and ligatures are
  normalized while minus signs are kept. Any difference is a failure (exit 1). A
  match is reported as `PDF NOT VERIFIED` (exit 3), never as a pass, because the
  layout still needs manual inspection.
- **Manifest writing.** `make manifest` now refuses to write when any build
  output is missing, so `"missing"` can no longer enter the baseline. It also
  refuses when the plotting environment is unpinned, and when `main.pdf` was not
  built with the Makefile's `SOURCE_DATE_EPOCH`.

The scenarios were rerun in a scratch worktree:

| Scenario | Result |
|---|---|
| Clean rebuild | PASS, exit 0 |
| Another epoch, identical content | PDF NOT VERIFIED, exit 3 |
| Another epoch plus an inserted sentence | FAIL, exit 1 (text differs on pages 8 and 9) |
| Recorded build plus an inserted sentence | FAIL, exit 1 |
| An extra page | FAIL, exit 1 (14 pages; manifest records 13) |
| Missing figure at check | FAIL, exit 1 |
| `make manifest` with a missing figure | refused |

Ten unit tests (`tests/test_build_manifest.py`) cover every branch of the PDF
verdict and the missing-artifact refusals. Because only one TeX installation was
available, the "another TeX build" case was simulated by changing the epoch.
**Cross-environment validation is therefore not complete.** It still needs an
independent rebuild in the pinned Python environment on a second machine, and
ideally under a second TeX release.

**AI Use Statement.** On the reviewer's recommendation, the sentence "All research
design, experiments and analysis were conducted by the authors" was removed. The
statement in `main.tex` now reads:

> During research and manuscript preparation, the authors used OpenAI Codex and
> Anthropic Claude to assist with manuscript drafting and revision, reference
> formatting and checking, experimental-protocol development, code
> implementation and debugging, figure generation, and statistical analysis.
> The authors made the final research decisions, supervised computational
> execution, and reviewed and verified the AI-assisted outputs, including
> reported results and interpretations. The authors take full responsibility for
> the paper.

The second sentence is an attestation. It is accurate only if the authors have
completed the verification it describes. This must be confirmed before
submission.

**Dates.** Dates in these records are UTC, and both this file and
`CURRENT_PROGRESS.md` now say so, rather than shifting dates between time zones.

**Retracted comment.** The earlier report of a duplicate row in the appendix
configuration table was withdrawn by the reviewer. The source has one row per
item, so no change was made.

**Experiments.** The ordering is unchanged: the second-model replication first,
then `B=2,4`, with cold sequential deferred. No protocol is frozen, and none may
be launched as written.

## 11. Simulated ARR review: positioning, mechanism, and validation (2026-10-03 UTC)

The review's verdict was that the workflow is more rigorous than the
established contribution. It scored the paper Soundness 3, Excitement 2–2.5,
Overall 2.5 (Borderline Findings). Its three top priorities were: reposition
against *The Flexibility Trap*, validate the new finding on unseen problems,
and separate CF's position selection from its candidate filtering.

### What Claude could and could not verify

- **Verified:** the public JustGRPO decoder at the pinned commit
  (`reports/prior_work_decoder_alignment_20261003.md`), and the README figure in
  the JustGRPO repository, which shows arbitrary order ahead at `k=1` and behind
  at large `k` on GSM8K.
- **Not verified:** the review's specific figure and table claims about the
  prior paper (a block-size sweep over `B=1,8,32,128`, a temperature appendix
  with tuned crossings, a random-order table at Pass@1 and Pass@128). arXiv and
  its mirrors were blocked by the environment's network policy. The new
  Introduction and Related Work state these axes without figure numbers. The
  authors must check them against the current version (arXiv 2601.15165).

### Manuscript changes

- **Positioning.** The Introduction and Related Work now take the prior
  work's phenomenon as given. The claimed contribution is a frozen,
  compute-matched test of it, the boundaries of the random-ranking effect, and
  what aggregate crossings and trajectory diagnostics can identify.
- **Contributions** condensed from five to three scientific ones. The
  protocols are described as support for credibility, not as a contribution.
- **Section 2.1** writes out CF's full stochastic process (candidate sampling,
  scoring by the position's own candidate, committing that candidate). It adds
  the `1 - 0.1^B` example of candidate filtering.
- **Section 2.2** states that CoverageAUC is a discrete grid mean, and that
  Pass@64 with `n=64` is the indicator that some rollout is correct.
- **Section 3 and Appendix A** describe the decoder as a reimplementation with
  float32 CF confidences, giving the commit.
- **Section 4** is retitled and framed as the total effect of widening the
  eligible set under CF.
- **Section 5** is retitled "Sampling Budget Changes a Pairwise Policy
  Ranking", and the text states that the best tested policy does not switch.
- **Section 6** adds the direct post-hoc interaction on the shared window:
  `-0.014 [-0.057, +0.027]` at `T=0.9`, `-0.223 [-0.289, -0.161]` at `T=1.2`.
- **Section 7** is rewritten around a proposition: if one policy's per-query
  success probability is never lower, its coverage is never lower at any `k`.
  So aggregate crossings require heterogeneous per-query differences, and the
  section gives a two-query example.
- **Discussion** names the CF-resample control as the discriminating test,
  and moves the boundary-closure detail to Appendix G.
- **Figure 1** relabels "Block-parallel" as "Blockwise any-order", and its
  caption points to Section 7.
- **Figure 4** caption and text are now explicitly descriptive.
- **Limitations** add that the random-ranking effect has not been validated on
  unseen problems and that filtering has not been separated from position
  choice.

The abstract is 181 words, and the main text still ends on page 8.

### New code, protocols, and reports

- **E0:** `reports/prior_work_decoder_alignment_20261003.md`. The public decoder
  computes CF confidences in bfloat16 (only four distinct values between 0.99
  and 1), while this study uses float32. CF position choice can therefore
  differ near ties, so the two are close settings, not identical ones.
- **E1:** `protocols/unseen_problem_validation_prefreeze_draft_20261003.md`.
  The power table shows 200 new problems are underpowered (0.56 at the observed
  effect), and recommends at least 400.
- **E2:** `protocols/cf_resample_control_prefreeze_draft_20261003.md`, plus the
  additive decoder mode `low_confidence_resample` with 3 tests. The existing
  modes are unchanged, and at `T=0` the control equals CF.
- **E3:** `scripts/problem_level_reanalysis.py`, with 6 tests. It runs on the
  server's raw records and gives joint success counts, the k=64 win/loss
  decomposition, split-sample heterogeneity, subset-averaged majority voting,
  and optional `math_equal` grouping.

### Not done in this round

- No GPU experiment was run.
- The tuned-temperature comparison on a development set, the length
  (`L=512`) controls, and a second model remain P1 options.
- The artifact paragraph still promises an anonymous supplement. Packaging it
  (code commit, patches, prompts, problem IDs, environment, recompute
  entry points) is a release task for the authors.
