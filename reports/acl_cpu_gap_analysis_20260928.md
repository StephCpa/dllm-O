# ACL CPU Gap Analysis

Date: 2026-09-28

This report uses the already committed post-outcome aggregate audits. It does
not read new GPU artifacts, change any frozen verdict, or upgrade exploratory
results to confirmatory evidence.

## 1. Width-versus-ranking decomposition

The existing random-extension audit contains the paired random-minus-sequential
contrast:

| Functional | Estimate | 95% CI | Reading |
|---|---:|---:|---|
| $k\in\{2,4,8,16\}$ | $-0.0333$ | $[-0.0519,-0.0154]$ | Sequential is higher at low budget |
| $k\in\{4,8,16,32,64\}$ | $-0.0183$ | $[-0.0387,+0.0013]$ | Full-grid difference unresolved |

The per-$k$ estimates are negative and resolved for random minus sequential at
$k=1,2,4,8$; they straddle zero at $k=16,32,64$. This means the existing data
support a scoped statement that sequential decoding is more reliable than random
selection at low sampling budgets, but they do not identify a resolved random
or sequential advantage over the full high-budget grid. In particular, these
results do not justify attributing the block-size primary contrast entirely to
confidence ranking or entirely to eligible-set width.

The current manuscript should therefore use language such as “the width and
ranking components are not separately identified at high budget; the available
contrast shows a low-budget sequential advantage over random selection.”

## 2. Majority-vote summary

The existing descriptive prefix table gives the following equal-task pooled
majority-vote accuracies:

| Policy | $k=1$ | $k=2$ | $k=4$ | $k=8$ | $k=16$ | $k=32$ | $k=64$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sequential ($B=1$) | .535 | .545 | .565 | .608 | .610 | .618 | .630 |
| CF ($B=32$) | .550 | .542 | .585 | .587 | .585 | .585 | .585 |
| RND ($B=32$) | .460 | .468 | .539 | .589 | .589 | .612 | .611 |
| EF ($B=32$) | .355 | .380 | .480 | .536 | .558 | .579 | .575 |

These are post-outcome descriptive prefix estimates and do not replace oracle
coverage. They nevertheless change the deployment wording: sequential is the
strongest practical baseline at most high-budget majority-vote points, whereas
CF's one-shot advantage is only a comparison among the $B=32$ policies. A
main-text sentence should state this distinction rather than leaving the table
only in the appendix.

## 3. What is still blocked by missing query-level artifacts

The committed report directory contains aggregate point estimates and intervals,
but not the query-by-policy count vectors needed to recompute:

- the per-query effect histogram and nonzero-query count;
- leave-one-query-out or jackknife influence for the block-size endpoint;
- a beta-binomial/mean-dispersion fit to the crossing;
- paired majority-vote bootstrap intervals;
- a same-policy A/A split control.

These analyses remain CPU-only once the raw query-level artifacts are available.
They should be run from the audited server/backup artifacts rather than inferred
from rounded manuscript tables.

## 4. Manuscript consequence

The CPU evidence strengthens the paper's honesty but does not eliminate the
main narrative risk. The strongest current claims are:

1. widening the eligible set lowers matched-NFE multi-sample coverage in the
   frozen block-size intervention;
2. position-policy rankings depend on the declared sampling budget, with the
   cleanest resolved low-budget contrast being sequential versus random;
3. the RND--CF high-budget crossing is a targeted, scoped follow-up whose
   direction changes with temperature and whose positive per-$k$ evidence is
   not uniformly familywise-resolved;
4. more parsed-answer diversity does not guarantee greater correct-answer
   coverage.

The title and Figure 1(b) should not be finalized until the query-level CPU
checks are complete. No additional GPU run is justified by this audit alone.
