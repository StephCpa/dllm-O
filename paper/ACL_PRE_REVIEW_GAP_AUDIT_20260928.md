# ACL/ARR Pre-Review Gap Audit

Audit basis: the two adversarial pre-reviews supplied on 2026-09-28 and the current ACL/ARR working draft.

Status update: 2026-09-30. The main text now reports the available
Sequential--RND contrast, and its values are generated from the audited random
extension JSON. The result is intentionally framed as partial identification:
low-budget sequential advantage is resolved, whereas the full high-budget
grid remains unresolved. The generator also uses the headless Matplotlib
backend so the manuscript asset build is reproducible on non-interactive
machines.

Status update: 2026-10-01. The archived result JSON was extracted to a separate
analysis workspace and passed the existing CPU audit. A new query-level audit
found substantial heterogeneity (73/200 primary-grid query effects exactly
zero), no dominant query (maximum absolute leave-one-query-out change
0.00393), a resolved majority-vote advantage for Sequential and CF over RND at
$k=1$ but no resolved advantage at $k\geq16$, and near-zero same-policy A/A
split differences. These are exploratory diagnostics; they narrow the
statistical objections but do not create new confirmatory endpoints.

Status update: 2026-10-01 UTC (manuscript revision; see
`REVISION_NOTES_20261001.md`). Figure 1 was redrawn as a vector schematic in
which every schedule commits one token per NFE, which resolves the panel (a)
objection under "What remains unresolved", item 4. Panel (b) remains
conceptual, but it now has no data-like markers and is labelled as
illustrative. Figure 2 now plots changes relative to $B=1$ on a $\log_2 B$
axis with the unmeasured $B=2,4$ region drawn explicitly (item 5). Figure 3
shows the seven-endpoint simultaneous band alongside the pointwise band
(item 6). The frozen trend endpoint, the descriptive $T=0.6$ reference under
the temperature functional, the paired majority-vote intervals at $k=1$, and
the majority-vote deployment sentence are now in the main text. Item 7
(Figure 4 uncertainty and a per-query distribution view) remains open.

## What the current draft has fixed

1. **NFE-matched intervention is explicit.** The manuscript now states that every condition commits one token per NFE, so `B` is eligible-set width rather than simultaneous multi-token commitment or a throughput claim.
2. **Temperature is no longer a single-point omission.** The draft includes the $T=0.9$ and $T=1.2$ extensions and labels them as targeted, post-review follow-ups. The remaining limitation is that all temperatures use one checkpoint and the same two-task panel.
3. **Evidence provenance is visible.** The evidence ledger records selection history, frozen questions, and evidentiary role. The random and temperature extensions are not presented as independent confirmatory replications.
4. **Multiplicity is acknowledged.** The per-$k$ audit reports simultaneous bands and the text no longer treats every pointwise positive high-budget contrast as confirmatory.
5. **Scope and limitations are present.** The draft now has a dedicated `Limitations` section and explicitly limits claims to one model, two math datasets, fixed decoding settings, and oracle coverage.
6. **Reproducibility and reference closure are strong.** The model revision, parser behavior, bootstrap unit, artifact accounting, and selection history are documented. The bibliography is internally closed and mechanically checked.

## What remains only partially solved

### 1. The title is fixed; the teaser remains conceptual

The title now centers matched-compute decoding schedules and repeated-sampling returns. The RND--CF crossing remains a targeted follow-up selected after an initial crossing was observed; its positive per-$k$ points do not survive the full seven-endpoint simultaneous band. Figure 1(b) remains conceptual, but its caption explicitly says that the values are not empirical; the actual frontier and intervals appear in Figure 3.

**Remaining decision:** keep the conceptual panel only if its role as a schematic is clear at final-size rendering; otherwise replace it with a compact empirical teaser without duplicating Figure 3.

### 2. Temperature identification is improved but not complete

The temperature extension is valuable and addresses the original single-temperature criticism. It does not establish cross-model robustness, and the temperature study itself was selected after review. The current draft appropriately calls the trace explanation mechanism-consistent rather than causal. No further temperature run should be started before the zero-GPU decomposition is complete.

### 3. Width and ranking are only partially decomposed

At $B=1$ the ranking rule is inert, while at $B=32$ confidence-first, random, and entropy-first are active. The revised main text now places the paired Sequential--RND contrast beside the primary $B=32$--$B=1$ contrast: the low-budget grid favors sequential with a resolved interval, but the full high-budget grid straddles zero. This narrows the objection, but it does not causally identify width or ranking as the sole source of the primary effect.

**Remaining fix:** use query-level artifacts, if recovered, to add per-query decomposition or an explicit compact panel. Do not call the current result causal, and do not replace the unresolved full-grid contrast with a stronger ranking claim.

### 4. Deployment relevance remains underdeveloped

The main text now gives two anchor points from the descriptive majority-vote curve and states that sequential is the strongest practical baseline at most high-budget points; the full curve remains appendix-only and has no uncertainty intervals. The manuscript correctly says that Pass@$k$ is oracle coverage, but the practical decision map therefore remains abstract. The paper also does not measure latency, memory, throughput, or a verifier/reranker.

**Recommended fix:** promote a compact majority-vote summary with explicit `post-outcome descriptive` status and query-bootstrap intervals if they can be recomputed from the stored query-level data. Rephrase the conclusion as an oracle-coverage result unless a real selector is evaluated. Treat throughput and verifier experiments as optional extensions, not prerequisites for the October ARR draft.

## What remains unresolved

1. **Formal interpretation of dispersion.** The query-level audit now reports effect heterogeneity, leave-one-query-out influence, variance-to-binomial-reference ratios, and moment-based beta-binomial rho values. These remain descriptive: no beta-binomial null is used for inference, and the manuscript should not present them as a mechanism test.
2. **A/A calibration.** A same-policy rollout-half split is now available and is near zero across the tested policies and budgets. It is a calibration diagnostic rather than an independent replication, so it does not remove the need for query-panel independence discussion.
3. **The $B=2,4$ gap.** The current text discloses that these settings were not measured. The figure should make the missing interval visually explicit rather than relying only on prose.
4. **Figure 1 panel (b).** It is still a schematic crossing. Since the actual data are now available, a real RND--CF contrast or a more defensible empirical teaser would reduce the expectation gap created by invented curves. Panel (a) still visually places multiple committed tokens in a single `State at t+1`; the caption explains the intended reading, but the drawing should show the accumulated sequence explicitly.
5. **Figure 2 resolution.** Absolute task-level curves mostly show that GSM8K is easier than MATH-500. Plotting changes relative to $B=1$ and marking $B=2,4$ as unmeasured would make the scientific effect visible.
6. **Figure 3 uncertainty hierarchy.** The policy frontier should make the simultaneous band visible and distinguish pointwise from familywise evidence. A compact Sequential--RND panel would also resolve the width/ranking issue visually.
7. **Figure 4 and missing distributional view.** The diversity figure is sparse and lacks uncertainty. A per-query $c/n$ ECDF or zero-success-mass plot would better address the generic crossing explanation than more point estimates.
8. **Analyzer and screening provenance.** The current draft now explains that the zero-success repair produced no pre-repair endpoint and that screening was a planned first-stage run, not an outcome-selected pilot. This should remain synchronized with the artifact documentation.

## Recommended execution order

### Phase 1: zero GPU, highest value

1. Compute paired Sequential--RND contrasts with query-bootstrap CIs at the declared low/high grids and the primary grid.
2. Compute per-query paired-effect distributions, nonzero counts, leave-one-query-out influence, and a beta-binomial/mean-dispersion diagnostic.
3. Recompute majority-vote query-level intervals and a same-policy A/A split control if the stored query-level outcomes support them.
4. Decide whether the query-level heterogeneity sentence belongs in the main text or appendix; do not promote the dispersion ratios to a causal explanation.
5. Regenerate the main manuscript macros and update the conclusion according to the resulting evidence, not beforehand.

### Phase 2: narrative and figures

1. Decide whether to retitle around the confirmatory matched-NFE result.
2. Replace the schematic Figure 1(b) with empirical data or clearly label it as a conceptual teaser and move the real crossing to the first results figure.
3. Redraw Figure 2 using baseline-relative effects and an explicit unmeasured $B=2,4$ gap.
4. Add the simultaneous-band distinction and Sequential--RND decomposition to the policy figure.
5. Add a compact per-query distribution figure if it can be produced from existing artifacts without new inference.

### Phase 3: optional GPU work

Do not start another GPU run until Phases 1--2 change the claim hierarchy. If one additional experiment is later justified, a second checkpoint on one task has higher external-validity value than another temperature point. A verifier or throughput study is a separate systems paper direction and should not be started merely to answer this review.

## Bottom line

The current ACL draft is substantially stronger than the earlier ICLR version and is no longer vulnerable on protocol transparency or the single-temperature omission. It is not yet insulated against the three most serious scientific objections: the title overweights a post-hoc crossing, width and ranking are not decomposed, and the deployment-facing selector result is hidden in the appendix. The best next move is a CPU-only analysis and narrative revision, not additional GPU inference.
