# Manuscript Revision Notes (2026-10-01)

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
- `make distclean && make` builds without errors, undefined references, or
  overfull boxes.
- The 123 repository tests pass (`python -m pytest`). The tests do not cover
  `paper/scripts/`.

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
