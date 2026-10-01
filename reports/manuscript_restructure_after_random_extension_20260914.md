# Manuscript Restructuring Blueprint After the Random-Policy Extension

**Prepared:** 2026-09-14

**Project:** `dllm_order_transmission`

**Purpose:** Convert the completed experimental program into a disciplined
manuscript narrative without changing any frozen verdict.

**Status:** Writing blueprint. No dLLM manuscript source currently exists in
this repository; this document therefore specifies the next manuscript rather
than editing the unrelated DP-ICL paper.

## 1. Recommended Paper Identity

The paper should be organized around one primary empirical result and one
measurement consequence:

> Moving from strictly sequential to block-parallel commitment changes the
> returns to repeated sampling in the tested diffusion language model. Within
> the block-parallel regime, position-selection policies trade low-budget
> precision against higher-budget coverage, so a scalar average over sampling
> budgets can conceal a policy crossing.

This is stronger and more accurate than the earlier BCI-centered story. It is
also narrower than saying that block-parallel decoding universally causes
coverage loss, that random selection is generally superior, or that aggregate
coverage metrics are always invalid.

### 1.1 Suggested title directions

Preferred:

> **When Decoding Policies Cross: Sampling-Budget-Dependent Coverage in
> Diffusion Language Models**

More mechanism-forward:

> **Sequential or Parallel? Decoding Schedules Reshape Test-Time Coverage in
> Diffusion Language Models**

More measurement-forward, but only if the final paper gives the metric result
enough formal and empirical space:

> **Averaging Across Sampling Budgets Can Hide Decoding-Policy Reversals**

Avoid titles claiming a universal "flexibility trap," a general law of
parallel decoding, or broad failure of test-time scaling evaluation. The
evidence is one dLLM, two mathematical-reasoning tasks, and one frozen decoding
configuration.

## 2. Claim Hierarchy

The abstract, introduction, results headings, figures, and conclusion should
use the same hierarchy.

### Tier 1: Primary confirmatory finding

Under a prospectively frozen intervention, changing from strict sequential
commitment (`B=1`) to confidence-first block-parallel commitment (`B=32`)
reduced pooled multi-sample CoverageAUC by `0.04421`, with a 95% CI of
`[0.01977, 0.06974]` in magnitude. The effect resolved in both GSM8K and
MATH-500. Most of the measured change occurred at `B=1 -> B=8`; differences
among `B=8`, `B=16`, and `B=32` were unresolved.

**Permitted interpretation:** the measured phenomenon is a discrete
sequential-to-parallel threshold in this design, not a smooth dose-response law
over block size.

### Tier 2: Frozen ranking-policy result

The prospectively frozen entropy-first intervention did not improve the
predeclared aggregate CoverageAUC64 endpoint over confidence-first decoding:
`-0.00986 [-0.04200, +0.02377]`. This is an unresolved contrast, not an
equivalence result. The preregistered length guard triggered and must accompany
the interpretation.

**Permitted interpretation:** neither tested alternative position ranking was
shown to recover the aggregate coverage lost relative to strict sequential
decoding. The result does not identify position ranking as irrelevant.

### Tier 3: Targeted prospective follow-up selected after unblinding

After the first-16 random-policy frontier suggested a crossing, a separately
frozen 9,600-generation extension completed random decoding to 64 rollouts.
Its frozen high-budget functional,
`mean_k in {16,32,64}(Pass@k_random - Pass@k_AO)`, was
`+0.03750 [+0.00301, +0.07370]` pooled across tasks. Neither task-specific
interval resolved alone.

**Permitted interpretation:** in this two-task panel, random selection had a
modest, resolved high-budget coverage advantage over confidence-first AO. This
is not an independent preregistered replication: the experiment was selected
after observing the first 16 rollouts and the final estimator combines those
16 with 48 new rollouts.

### Tier 4: Secondary and exploratory structure

The complete random-versus-AO curves show a budget-dependent reversal: AO is
resolvedly better at `k=1,2`, whereas random is nominally resolvedly better at
`k=16,32` under unadjusted pointwise intervals. In a post-outcome exploratory
audit over all seven per-k endpoints, the maximum-deviation simultaneous bands
retain only the `k=1` pointwise contrast; Bonferroni intervals retain `k=1,2`
but not either high-k point. The individual per-k claims therefore do not
survive familywise correction at both ends.

An explicitly post-outcome two-region summary is more stable: the low-budget
mean over `k={1,2}` is `-0.06747 [-0.10245,-0.03248]`, and the high-budget mean
over `k={16,32}` is `+0.03626 [+0.00127,+0.07124]` under a two-endpoint
simultaneous band. This diagnoses a jointly resolved reversal after outcome
inspection, but cannot replace or strengthen the separately frozen
high-budget primary endpoint.

Entropy-first decoding generates the greatest parsed-answer diversity but is
worse than random in correct-answer coverage over most tested budgets. This
rejects the naive proxy assumption that more answer-string diversity
necessarily yields more useful exploration. It does not show that diversity is
never useful or that semantic reasoning diversity behaves identically.

## 3. The Measurement Contribution

Define the per-budget paired policy contrast as

```text
Delta(k) = Pass@k(policy A) - Pass@k(policy B).
```

Any scalar grid summary is a weighted functional

```text
Delta_w = sum_k w_k Delta(k),  where w_k >= 0 and sum_k w_k = 1.
```

When `Delta(k)` changes sign, positive and negative regions can cancel. The
value and even the sign of `Delta_w` can then depend on the chosen budget grid
and weights. Therefore an unresolved scalar summary cannot by itself establish
that policies have similar frontiers.

### 3.1 What may be claimed

- Aggregate pass@k summaries can attenuate or conceal budget-dependent policy
  crossings.
- A coverage AUC is not an intrinsic property of a policy unless its target
  sampling-budget distribution is declared.
- When policies cross, report the curve and deployment-relevant budgets in
  addition to any scalar summary.
- The present data contain repeated examples of cancellation across shared
  policies, queries, and rollouts.

### 3.2 What must not be claimed

- Do not write that *any* average spanning a crossing is necessarily
  insensitive. Cancellation depends on magnitudes and weights.
- Do not call the EF-AO, first-16 AO-random, and 64-rollout random-AO analyses
  independent replications. They share tasks, queries, policies, and in some
  cases rollouts.
- Do not say that an aggregate null is always a measurement artifact. A scalar
  estimand may be scientifically appropriate if its budget weighting matches
  deployment; the problem is treating it as a policy-intrinsic ranking.
- Do not turn exploratory grid decompositions into revisions of frozen null
  verdicts.

## 4. Recommended Manuscript Structure

### 4.1 Introduction

Open with the practical decision, not BCI:

1. dLLMs can commit multiple token positions per step, and this schedule is
   often treated as an efficiency detail.
2. Test-time scaling is evaluated across rollout budgets, but a schedule may
   improve single-sample precision while reducing the rate at which repeated
   samples discover new correct solutions.
3. The paper asks two questions: what changes when commitment becomes parallel,
   and whether one policy ranking remains best as sampling budget grows.
4. Preview the answer: a sequential-to-parallel coverage drop, followed by a
   budget-dependent ranking-policy reversal that scalar summaries can hide.

BCI should appear as an investigated mechanistic hypothesis that failed, not as
the paper's promised solution.

### 4.2 Contributions

Use four contributions, each with an evidence qualifier:

1. **Prospective block-size intervention.** We prospectively test four
   commitment block sizes and find that the measured multi-sample coverage loss
   is concentrated at the transition from sequential to block-parallel
   decoding, rather than following a resolved smooth law within the tested
   parallel regime.
2. **Budget-dependent policy comparison.** A separately frozen targeted
   extension shows that confidence-first selection is favored at very small
   rollout budgets, while random position selection has a modest pooled
   high-budget coverage advantage. We explicitly scope this result as a
   post-unblinding-selected prospective follow-up.
3. **Evaluation diagnosis.** We show analytically and empirically that weighted
   summaries over pass@k can attenuate policy crossings, and argue that
   schedule comparisons should report their target budget distribution and
   full frontier.
4. **Negative mechanism evidence.** A prospectively frozen entropy-first
   intervention increases commitment-time uncertainty and answer-string
   diversity but does not improve the frozen aggregate coverage endpoint;
   additional held-out tests also reject BCI as a query-level targeting rule.

Do not describe the work as providing a new decoding algorithm, universal
mechanism, or deployment-optimal policy.

### 4.3 Setup and estimands

- Define dLLM commitment, block size, and the four principal policies:
  sequential AR-style `B=1`, confidence-first AO, entropy-first EF, and random
  selection.
- Define Pass@k as an unbiased query-level coverage estimator.
- Define CoverageAUC together with its k-grid. Never use "AUC" without the
  grid or an earlier explicit definition.
- Explain that Pass@k is oracle coverage, not deployable accuracy without a
  verifier or reranker.
- State the query as the unit of inference and the task-stratified paired
  bootstrap as the interval procedure.

### 4.4 Results I: the sequential-to-parallel threshold

Lead with frozen Option B. Show absolute CoverageAUC by block size and adjacent
contrasts. Immediately disclose that P1 and P2 are highly dependent
(`Pearson r=0.984`, tie-aware `Spearman rho=0.943`) and that 95.3% of the
post-hoc slope decomposition comes from the `B=1`-to-parallel step.

The correct heading is a result sentence such as:

> **Coverage drops at the sequential-to-parallel transition, not smoothly
> across tested parallel block sizes.**

### 4.5 Results II: ranking policy reshapes the budget frontier

Present AO, random, EF, and sequential AR curves over the same k-axis. Separate
three statements:

- AO versus random: low-budget AO advantage and frozen high-budget random
  advantage.
- Random versus EF: more useful correct coverage despite lower answer-string
  diversity.
- Sequential baseline: AR retains the strongest high-budget point estimates in
  the measured setting, but it does not dominate every policy at every budget;
  for example, AO has a slightly higher pooled Pass@1. No latency-controlled
  frontier was measured.

Do not write that sequential decoding is "off the frontier" or "dominates the
interior." Both formulations are unsupported, and the former contradicts the
latter.

### 4.6 Results III: why one scalar can miss the result

Introduce `Delta_w` only after readers have seen the crossing. Use the
EF-versus-AO frozen aggregate and the random-versus-AO grid summaries as case
studies. Preserve every original frozen reading, then explain post hoc why a
mixed grid can average across regions with opposite signs.

This section should be framed as an estimand lesson, not as evidence that the
original analyses were erroneous.

### 4.7 Results IV or appendix: failed local targeting mechanism

Compress BCI into a short negative-results subsection or appendix:

- Local matched entropy closure replicated on both held-out tasks.
- Held-out BCI-to-coverage Spearman was `+0.0136 [-0.134,+0.161]`.
- Adding BCI made cross-validated MAE worse by `0.000169`, only about 0.18% of
  baseline MAE.
- Therefore local closure is measurable but not a useful query-level targeting
  rule in this setting.

This result strengthens credibility and delimits mechanism; it should not
consume the narrative space of the policy frontier.

### 4.8 Discussion

Organize around decisions:

- If only one or two samples are available, confidence-first parallel decoding
  has the strongest measured low-budget performance among the B32 ranking
  policies.
- If oracle coverage at larger k is the target, random ranking recovers part of
  AO's loss in this panel, but sequential decoding retains slightly higher
  point estimates and a practical choice also requires controlled latency and
  verifier quality.
- Entropy-first is not an efficient exploration heuristic here: it increases
  answer-string diversity without improving correct-answer coverage over
  random.
- Evaluation should select k and weights from the anticipated deployment
  budget, or report the full frontier when no single deployment distribution is
  justified.

State the threshold mechanism as a hypothesis: once `B>1`, the commitment
schedule itself can vary with sampled model state. The current design is
consistent with that explanation but does not causally isolate it.

## 5. Abstract Blueprint

Suggested logic, not final prose:

1. **Problem:** block-parallel dLLM decoders change both token order and
   test-time efficiency, but are commonly evaluated with single-budget accuracy
   or scalar multi-budget summaries.
2. **Confirmatory finding:** in one frozen dLLM across GSM8K and MATH-500, a
   prospective block-size intervention finds a resolved multi-sample coverage
   decrease concentrated at the sequential-to-parallel transition, despite a
   slight increase in Pass@1.
3. **Policy interaction:** confidence-first ranking is better at very low k,
   while a targeted frozen follow-up finds a modest pooled high-budget advantage
   for random position selection.
4. **Measurement implication:** because the paired contrast changes sign with
   k, averaging across a mixed budget grid can attenuate the comparison.
5. **Negative result and scope:** entropy-first ranking increases answer-string
   diversity without improving the frozen aggregate coverage endpoint; all
   conclusions are limited to one model, two reasoning tasks, and oracle
   coverage.

Only two numerical results are necessary in the abstract:

- `B=32 - B=1 CoverageAUC = -0.044 [-0.070,-0.020]`;
- frozen random-minus-AO high-budget summary
  `+0.038 [+0.003,+0.074]`.

Do not place unadjusted per-k intervals in the abstract. The safest abstract
language is that the curves reverse direction and that the frozen high-budget
functional resolves positively. If space permits, a sentence may add that a
post-outcome, multiplicity-aware two-region diagnostic also resolves the
crossing; it must not be described as preregistered or independent evidence.

## 6. Figure and Table Plan

### Figure 1: the decision and the crossing

Left: a compact diagram contrasting sequential commitment (`B=1`) with
block-parallel commitment (`B>1`) and its position-ranking choices. Right: a
schematic Pass@k crossing illustrating why low-budget precision and
high-budget coverage are different objectives. Mark the schematic clearly; do
not insert empirical values into a conceptual panel.

### Figure 2: confirmatory block-size result

Plot pooled CoverageAUC at `B={1,8,16,32}` with query-bootstrap intervals and
small task-specific points. Add an inset or paired lower panel with Pass@1 and
Pass@16. Visually emphasize the `B=1 -> B=8` step; do not draw a fitted smooth
law that the adjacent contrasts do not support.

### Figure 3: full policy-by-budget frontier

Plot Pass@k for AR, AO-B32, random-B32, and EF-B32 on a log2 k-axis. Use direct
labels or a compact legend and confidence ribbons or interval marks at the key
contrasts. Annotate:

- AO over random at `k=1,2`;
- frozen random high-budget functional over `k={16,32,64}`;
- AR as the strongest high-budget point-estimate baseline;
- EF as the highest-diversity policy, not the highest-coverage policy.

Avoid stars on all 21 pairwise pointwise comparisons. A separate contrast
panel for random-minus-AO is more legible and makes the sign change explicit.

### Figure 4: diversity is not correct coverage

Use a scatter plot with parsed-answer entropy on the x-axis and Pass@k or
CoverageAUC on the y-axis, with policies labeled. The key visual is EF to the
right of random but below it in coverage. State that entropy is computed over
parsed answer strings.

### Table 1: evidence ledger

Include columns for study, selection timing, frozen endpoint, result, and
permitted claim. This is essential because the paper combines prospectively
frozen studies and post-unblinding-selected extensions.

### Appendix figures and tables

- Per-task Pass@k frontiers and all pointwise intervals.
- Multiplicity-aware random-versus-AO family audit.
- Solved-set overlap at k=64.
- Length guard and post-treatment sensitivity analyses.
- Commitment entropy and trace-level mechanism checks.
- BCI construction, matching diagnostics, and held-out null.
- Malformed-output and parser audits.
- Exact protocol hashes, frozen revisions, and result provenance.

## 7. Statistical Reporting Rules

1. The unit of inference is the query; do not count rollouts or token events as
   independent observations.
2. State whether each interval is pointwise, simultaneous, or Bonferroni
   adjusted.
3. The main random-versus-AO per-k family contains seven endpoints. Across the
   three baseline families in the extension there are 21 unadjusted pointwise
   contrasts; do not imply familywise control for the whole table.
4. If describing a joint reversal, define the joint event before reporting it.
   A post-outcome low/high summary may diagnose robustness but cannot inherit
   the frozen primary status.
5. Do not multiply marginal p-values or imply that shared-endpoint contrasts
   are independent.
6. Report exact zero-touching intervals as unresolved, including the
   random-versus-EF k=64 lower bound of `0.00000`.
7. Keep the Condition A length guard. Post-hoc matching and regression make a
   simple truncation-only account less plausible, but neither is a causal
   correction because decoded length is post-treatment.
8. Report effect magnitudes and uncertainty, not only labels such as PASS,
   FAIL, or resolved.

## 8. Claims to Use and Avoid

### Safe manuscript-level claims

- In the tested model and tasks, the main coverage change occurs when decoding
  moves from sequential to block-parallel commitment.
- Block-parallel position ranking changes the shape of the Pass@k frontier.
- Confidence-first AO favors low-budget precision, whereas random selection
  has a modest pooled high-budget coverage advantage in the targeted extension.
- More parsed-answer diversity is not sufficient for more correct-answer
  coverage.
- Scalar summaries over k can conceal policy crossings and must be tied to an
  explicit deployment budget distribution.
- The original local BCI hypothesis failed its held-out predictive test.

### Claims to prohibit

- "Random decoding is universally best."
- "Sequential decoding dominates all parallel policies."
- "Block-parallel structure itself causes the harm."
- "Position ranking does not matter."
- "The entropy-first and confidence-first policies are equivalent."
- "Aggregate coverage AUC is invalid."
- "Any average over a crossing must be null."
- "Three independent replications show metric failure."
- "Answer entropy measures semantic reasoning diversity."
- "Pass@k directly measures deployable utility."
- "The findings generalize to all dLLMs or test-time scaling systems."

## 9. Evidence-Status Language to Reuse Verbatim

For Option B:

> We prospectively froze the block-size intervention, endpoints, bootstrap
> procedure, and decision rules before reading the held-out outcomes.

For Condition A:

> The frozen aggregate contrast was unresolved and remains reported as such;
> subsequent frontier analyses explain its budget sensitivity but do not revise
> that verdict.

For the random extension:

> This targeted follow-up was selected after the first-16 frontier suggested a
> crossing, frozen before 9,600 new generations were evaluated, and combines
> 48 new with 16 previously observed rollouts. It is outside the original Gate-1
> confirmatory chain.

For secondary comparisons:

> Pointwise intervals are descriptive and unadjusted unless explicitly labeled
> simultaneous; the complete comparison family is reported in the appendix.

## 10. Remaining Work

### Required before drafting final claims

- Preserve and cite the completed 50,000-replicate multiplicity audit for the
  seven random-minus-AO per-k contrasts; distinguish its familywise pointwise
  bands from its post-outcome two-region summary.
- Generate every manuscript numeral and plotting table from committed JSON,
  with no manual transcription.
- Preserve the random-extension raw artifacts and control-plane files in both
  verified backup locations.

### Required for a submission-ready manuscript

- Create a dedicated manuscript directory for this dLLM study rather than
  modifying the unrelated DP-ICL LaTeX source.
- Build Figures 2-4 directly from canonical reports.
- Add a reproducibility appendix with model, dataset, parser, seed, protocol,
  and commit hashes.
- Release code and nonrestricted artifacts in an anonymized package.
- Audit title, abstract, contributions, and conclusion against the prohibited
  claim list above.

### No further GPU experiments

The stopping rule has been reached. The remaining uncertainties concern
cross-model and cross-domain generalization, latency-controlled deployment
trade-offs, and verifier-based utility. Those are limitations or future-study
questions, not repair experiments for this manuscript.
