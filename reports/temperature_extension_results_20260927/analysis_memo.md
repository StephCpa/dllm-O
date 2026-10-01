# Temperature-by-Schedule Extension: Analysis Memo

**Analysis date:** 2026-09-27  
**Evidence status:** post-review prospective follow-up, frozen before the new T=0.9 and T=1.2 outcomes; outside the original Gate-1 chain.  
**Frozen runner/analyzer commit:** `42a6e94e00c743d077a6081a3902f0b2a5e0d32b`  
**Freeze SHA-256:** `b83226472f49b07a638854e8a7c1c2d964115661af081aaa9539f635e2914d2c`

## Frozen result

The preregistered question was whether the random-minus-confidence-first (RND-CF) contrast has a resolved negative sign at `Pass@1` and a resolved positive sign at the mean of `Pass@16` and `Pass@32` at each new temperature. Neither temperature meets that joint criterion.

| Temperature | RND-CF at Pass@1 | Simultaneous 95% CI | RND-CF at high budget | Simultaneous 95% CI | Frozen verdict |
|---|---:|---:|---:|---:|---|
| 0.9 | -0.17141 | [-0.23125, -0.11157] | +0.01314 | [-0.04670, +0.07298] | `not_resolved_as_reversal` |
| 1.2 | -0.45656 | [-0.51640, -0.39672] | -0.19598 | [-0.25582, -0.13614] | `not_resolved_as_reversal` |

These are equal-weight GSM8K/MATH-500 estimates from 100 held-out queries per task and 32 rollouts per query and cell. Intervals are from the frozen 10,000-replicate, task-stratified paired-query bootstrap (`seed=20260914`) with a single nonstudentized maximum-absolute-deviation 95% band for the four primary endpoints. The high-budget T=0.9 point estimates differ by task: +0.03517 on GSM8K and -0.00889 on MATH-500. At T=1.2 they are -0.19672 and -0.19524, respectively. Per-task values are descriptive, not separately resolved task-level claims.

`not_resolved_as_reversal` is not an equivalence claim. At T=0.9 the high-budget interval includes both a modest random advantage and a modest disadvantage. At T=1.2 the high-budget random disadvantage is resolved under the four-endpoint family.

## Secondary frontier and diagnostics

| T | Schedule | Pass@1 | Pass@8 | Pass@16 | Pass@32 | Mean response words | Answer entropy at k=32 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.6 | confidence-first | 0.563 | 0.719 | 0.759 | 0.790 | 136.3 | 0.795 |
| 0.6 | random | 0.468 | 0.735 | 0.783 | 0.820 | 130.9 | 1.462 |
| 0.6 | sequential | 0.544 | 0.763 | 0.807 | 0.850 | 135.2 | 1.109 |
| 0.9 | confidence-first | 0.556 | 0.728 | 0.774 | 0.815 | 136.6 | 0.888 |
| 0.9 | random | 0.385 | 0.720 | 0.786 | 0.830 | 127.4 | 1.910 |
| 0.9 | sequential | 0.505 | 0.745 | 0.793 | 0.835 | 132.2 | 1.385 |
| 1.2 | confidence-first | 0.557 | 0.735 | 0.776 | 0.800 | 136.6 | 0.938 |
| 1.2 | random | 0.100 | 0.427 | 0.549 | 0.635 | 92.4 | 2.669 |
| 1.2 | sequential | 0.213 | 0.549 | 0.639 | 0.720 | 94.8 | 2.324 |

The T=0.6 arm reuses the frozen first 32 rollouts of the earlier study; it is a descriptive reference, not a new prospective outcome. At T=0.9 the pooled RND-CF per-k point difference changes sign between k=8 (-0.0076) and k=16 (+0.0113), but the frozen high-budget simultaneous interval spans zero. At T=1.2 RND-CF remains negative at every measured k through 32. Raising temperature is not a simple substitute for random position selection in this setup: confidence-first coverage stays near its T=0.6 levels, while random and sequential degrade sharply at T=1.2.

At T=1.2 the random and sequential responses are substantially shorter and show greater parsed-answer diversity, alongside lower coverage. This is a diagnostic association, not evidence that shortening or diversity causes the loss. All nine temperature-by-schedule cells have 6,400 records, zero empty responses, and zero parser-status errors. `Pass@k` is oracle correct-solution coverage; answer entropy is diversity of parsed answer strings, not semantic reasoning diversity, and neither is realized deployment utility without a verifier.

## Audit trail and claim boundary

- The no-outcome-read gate validated all 19,200 reused T=0.6 controls against the frozen inventory SHA-256 `4efe7daccfc72a09395cd54c3425cbb9db3f43f3befcc09bf971c28290eee4d7`.
- Formal validation found 38,400/38,400 valid new records, 0 missing or invalid, and 6/6 valid shard markers; `outcome_fields_read=false`.
- The first `analyze` invocation logged `2026-09-27T09:58:33.105094+00:00` with `reads_temperature_correctness=true`, before loading outcomes. The remote inference checkout remained clean at the frozen commit.
- The pooled primary point estimates were independently recomputed from the stored equal-task Pass@k curves: low = `RND(1)-CF(1)` and high = the mean of the k=16 and k=32 differences. Both reproduce the analyzer output to floating-point precision.
- The generated JSON and Markdown reports, invocation log, and machine freeze are archived alongside this memo. Their SHA-256 values are, respectively, `bb022b0309060e7a1e231946178c268b533553d5a97b7043925d700c651998c1`, `8bca5f92e9d8e80217d5edd0be0fd66d26f4c56f997706c1e8f2899183b1504c`, `319ec0445318f1944173172e994db6d73c609ac4db6709f31b9f9d700c1e5e5b`, and the freeze hash above.

The defensible manuscript statement is that the T=0.6 policy reversal is **temperature-conditional** in this model, task panel, and fixed-256-NFE decoder: it is not resolved at T=0.9 and is absent, in the opposite high-budget direction, at T=1.2. This result limits a temperature-invariant frontier claim; it does not erase the earlier T=0.6 finding. No post-outcome repair arm or rewritten primary endpoint is warranted. The next manuscript pass should show the three-temperature frontier, foreground the frozen non-replication, and state that higher sampling temperature does not automatically purchase useful coverage.

## Post-outcome critique audit (exploratory)

An external critique proposed a differential-temperature-robustness and low-entropy-selection account. We tested these suggestions **after** the frozen outcomes were read. The following calculations are hypothesis-generating and do not alter either frozen verdict above. The CPU-only script uses all 200 matched queries and the first 32 rollouts per cell; it resamples paired queries within task 10,000 times (`seed=20260927`). Its intervals are **unadjusted** percentile 95% intervals over exploratory comparisons.

| Change from T=0.6 | CF Pass@1 | RND Pass@1 | RND-vs-CF change in Pass@1 | RND-vs-CF change in high budget |
|---|---:|---:|---:|---:|
| T=0.9 | -0.0069 [-0.0186, +0.0055] | -0.0831 [-0.0984, -0.0677] | -0.0763 [-0.0958, -0.0569] | -0.0142 [-0.0572, +0.0270] |
| T=1.2 | -0.0061 [-0.0214, +0.0097] | -0.3675 [-0.4037, -0.3308] | -0.3614 [-0.4017, -0.3206] | -0.2233 [-0.2895, -0.1615] |

The differential decline is large, especially at T=1.2. Yet the CF intervals do **not** establish temperature invariance: no equivalence margin or equivalence test was specified, and its high-budget T=1.2 change is +0.0136 with an unadjusted interval [-0.0205, +0.0488]. Prefer "comparatively stable over this grid" to "temperature-invariant" or a confirmatory "resolved robustness" claim.

The complete held-out query set was also inspected at the trace level for rollouts 0 and 1 only (400 traces per temperature/schedule cell, 256 commit events per trace). Mean entropy of the **selected** position, computed from the model's untempered token distribution immediately before commitment, was:

| Schedule | T=0.6 | T=0.9 | T=1.2 |
|---|---:|---:|---:|
| CF | 0.238 | 0.238 | 0.250 |
| RND | 0.411 | 0.662 | 3.580 |
| Sequential | 0.277 | 0.378 | 3.117 |

Within CF traces, the mean entropy across all currently eligible positions is about 1.26 at every temperature, far above its selected-position entropy. For RND, selected and eligible means are nearly equal, as expected from random selection. This is strong *mechanism-consistent diagnostic evidence* that CF filters toward low-entropy commitments while the other trajectories become more uncertain at high temperature. It is not proof that this filter alone causes the accuracy pattern: token sampling, committed tokens, and subsequent model states all change together. The implementation adds temperature-controlled Gumbel noise to logits to sample candidate tokens; CF then ranks the sampled candidates by their probability under the **untempered** model softmax. RND's trace field named `candidate_probabilities` stores random selection scores, not model probabilities, so it must not be compared across policies as if it were calibrated confidence.

Two other proposed interpretations require restraint. At T=1.2, RND's Pass@32-minus-majority-vote gap is 0.635-0.258=0.377, versus 0.820-0.613=0.207 at T=0.6. This is an **oracle-versus-majority-vote selection gap**, not a measured fraction of unusable coverage or evidence that answers are mere guesses; a verifier could behave differently from either selector. Likewise, shorter responses and higher answer-string entropy at T=1.2 are compatible with trajectory degradation but do not identify its cause. The statement that sequential decoding dominates CF must be explicitly scoped to the original T=0.6 comparison: at T=1.2 the CF pooled point estimates exceed sequential at every measured k, although these per-k comparisons are secondary and not familywise tested.

For manuscript integration, retain the frozen non-replication as the lead result; add a compact schedule-by-temperature figure and the comparative-robustness/trace diagnostics as labeled post-outcome analysis. The main-text evidence ledger already separates primary, gate-triggered, targeted, and exploratory work; add the temperature extension as a **post-review-selected prospective follow-up**, and make the earlier random-64 selection history equally explicit. Do not promote the oracle-majority gap or the one-mechanism explanation to independent contributions without a separately designed test.

**Exploratory provenance:** script `scripts/temperature_extension_exploratory.py` SHA-256 `a9ee610e4503d1b772660cb6a42346398c61f2959336f3c945d772bc0a717172`; generated `temperature_extension_exploratory_v2.json` SHA-256 `8e15f307493ac13e705660a0fc1007bd22cc09255071f368d831f79e0589fde2`.
