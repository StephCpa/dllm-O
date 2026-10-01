# Post-Review Follow-Up Protocol Draft

**Status:** DRAFT, NON-BINDING, NOT FROZEN. No outcome should be generated or read under this document. A machine-readable freeze, exact split manifest, hashes, and launch record must be created before any formal run.

## Purpose

The existing study isolates eligible-set width and position ranking at fixed generation length (256), fixed denoising steps (256), and therefore fixed 256-NFE compute. Two remaining questions could materially change the paper's interpretation:

1. Is the schedule effect distinct from ordinary sampling temperature?
2. Does the resolved change begin when order choice first becomes available, and is it caused by eligible-set width or by confidence ranking within that set?

This draft deliberately does not test true multi-token-per-NFE parallelism. That is a separate efficiency study and is not required for the paper's revised compute-matched decoding-order claim.

## Study A: Temperature-by-Schedule Interaction

### Design

- Model: the exact frozen LLaDA-8B-Instruct revision used in the main study.
- Tasks: GSM8K and MATH-500, using the existing held-out source IDs and prompts.
- Temperature: `T in {0.6, 0.9, 1.2}`.
- Schedules: sequential `B=1`, confidence-first `B=32`, and random `B=32`.
- Fixed settings: generation length 256, denoising steps 256, CFG 0, identical parser and grader, and one committed token per NFE.
- The existing `T=0.6` cells should be reused rather than regenerated if all hashes match.
- Proposed new sampling budget: 32 rollouts per query for each new temperature-schedule cell. A final freeze may reduce this only after a blinded precision simulation.

### Primary Estimands

- Equal-task-pooled Pass@k curves on `k in {1,2,4,8,16,32}`.
- Schedule contrasts within each temperature, reported per k rather than only as an aggregate.
- Dominance or crossing of the empirical `(Pass@1, Pass@16, Pass@32)` frontier across temperature and schedule.

### Interpretation Rules

- If a higher-temperature CF curve matches or dominates RND at `T=0.6`, schedule is not established as an independent practical control.
- If schedule contrasts persist after comparing the joint temperature-schedule frontier, eligible-position choice remains a distinct axis within the tested range.
- No universal claim beyond this model, two tasks, and temperature grid is permitted.

## Study B: Onset and Ranking Decoupling

### Design

- Confidence-first at `B in {2,4}` to fill the gap between `B=1` and `B=8`.
- Random ranking at `B in {8,16}` to compare against the existing random `B=32` cell.
- Same model, tasks, held-out source IDs, generation length, denoising steps, temperature 0.6, parser, grader, and seed-key construction as the existing block-size study.
- Proposed budget: 64 rollouts per query, matching the existing block-size cells.

### Primary Estimands

- Per-k contrasts against the nearest existing width, with `k in {1,2,4,8,16,32,64}`.
- A descriptive shape comparison between onset-of-choice and gradual-saturation hypotheses.
- Ranking decoupling: compare random and confidence-first trajectories over their overlapping widths without interpreting `B=1` as having an active ranking policy.

### Interpretation Rules

- Drop the previous log-block-size slope diagnostic; the uneven grid makes it difficult to interpret.
- Do not call the effect monotone unless all adjacent contrasts support that claim.
- If random remains stable across widths while CF changes, attribute the effect primarily to confidence ranking enabled by a wider eligible set, not to width alone.
- If both policies change similarly, eligible-set width has evidence independent of the ranking rule.

## Evidence Status and Launch Requirements

Both studies are motivated by post-outcome reviewer critique and must be labeled as post-review follow-ups. Before launch:

1. Freeze exact source IDs, model and dataset revisions, prompts, seed keys, cell sizes, endpoints, bootstrap implementation, multiplicity treatment, and abort rules.
2. Run validate-only and smoke modes without reading correctness outcomes.
3. Record the first outcome-reading timestamp in an append-only invocation log.
4. Preserve existing `T=0.6` artifacts and bind any reused cells by checksums.
5. Keep the inference checkout at the frozen code revision; develop analyzers in a separate worktree.

## Priority

Study A is higher priority because temperature is the strongest alternative explanation for the practical schedule claim. Study B is second: it sharpens mechanism attribution but does not determine whether the observed frontier is reproducible by temperature alone.
