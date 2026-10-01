"""Exploratory post-unblinding CPU analyses for the decoding-tradeoff reframing.

Three analyses, all labeled EXPLORATORY (post-unblinding, not pre-registered):

A. pass@k / majority-vote@k vs compute-budget Pareto table per policy.
B. Direct diversity measures: unique parsed answers, answer entropy, modal share.
C. EF vs AO mechanism: length-coverage relations and commit-entropy subsample.

Reads only existing Phase 1 + Condition A records/traces. Writes JSON and CSV.
"""

from __future__ import annotations

import gzip
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

# Resolve the repository root instead of embedding a machine-specific path.
ROOT = Path(__file__).resolve().parents[1]
P1 = ROOT / "outputs/dllm_order_phase1/phase1-v1/heldout"
CA = ROOT / "outputs/dllm_order_condition_a/condition-a-v1/condition-a-formal"
OUT = ROOT / "outputs/dllm_order_condition_a/condition-a-v1/control/exploratory_20260909"
MANIFEST = json.loads((ROOT / "protocols/phase1_split_manifest_v1.json").read_text())
TASKS = ("gsm8k", "math500")
QUERIES = {
    t: sorted(e["row_index"] for e in MANIFEST["splits"][t]["heldout"]) for t in TASKS
}
MC_DRAWS = 2_000
MC_SEED = 20260909
BUDGET_KS = (1, 2, 4, 8, 16, 32, 64)
SIXTEEN_POLICIES = ("ar_b1", "ao_b8", "ao_b16", "ao_b32", "random_b32", "ef_b32")
SIXTYFOUR_POLICIES = ("ar_b1", "ao_b32", "ef_b32")
ENTROPY_SUBSAMPLE_ROLLOUTS = (0, 1, 2, 3)


def load_rollouts(policy: str, task: str, row: int) -> list[dict]:
    recs = []
    if policy == "ef_b32":
        d = CA / task / policy / f"q{row:04d}"
        for r in range(64):
            recs.append(json.loads((d / f"r{r:02d}.json").read_text()))
    else:
        s = P1 / "screening" / task / policy / f"q{row:04d}"
        for r in range(16):
            recs.append(json.loads((s / f"r{r:02d}.json").read_text()))
        if policy in ("ar_b1", "ao_b32"):
            e = P1 / "primary-extension" / task / policy / f"q{row:04d}"
            for r in range(16, 64):
                recs.append(json.loads((e / f"r{r:02d}.json").read_text()))
    return recs


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    from math import comb
    return 1.0 - comb(n - c, k) / comb(n, k)


def majority_vote_accuracy(
    answers: list[str],
    correct_flags: list[bool],
    k: int,
    rng: random.Random,
) -> tuple[float, int]:
    """Estimate conservative self-consistency accuracy.

    Exact modal ties count as incorrect. A parsed answer's correctness is the
    majority frozen-grader label among records producing that answer. The
    inconsistency count is surfaced rather than silently choosing the first
    correct rollout.
    """
    labels: dict[str, list[bool]] = defaultdict(list)
    for answer, correct in zip(answers, correct_flags):
        labels[answer].append(correct)
    answer_is_correct = {
        answer: sum(flags) > len(flags) / 2 for answer, flags in labels.items()
    }
    inconsistent = sum(len(set(flags)) > 1 for flags in labels.values())

    if k == 1:
        return statistics.mean(correct_flags), inconsistent

    draws = 1 if k == len(answers) else MC_DRAWS
    wins = 0
    n = len(answers)
    for _ in range(draws):
        idx = list(range(n)) if k == n else rng.sample(range(n), k)
        counts = Counter(answers[i] for i in idx)
        top = max(counts.values())
        leaders = [a for a, c in counts.items() if c == top]
        if len(leaders) == 1 and answer_is_correct[leaders[0]]:
            wins += 1
        # ties count as incorrect (conservative)
    return wins / draws, inconsistent


def answer_entropy(answers: list[str]) -> float:
    counts = Counter(a for a in answers if a is not None)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log(c / total) for c in counts.values())


def average_ranks(values: list[float]) -> list[float]:
    """Return one-based average ranks, including exact ties."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = ((start + 1) + end) / 2
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    rank_x = average_ranks(x)
    rank_y = average_ranks(y)
    mean_x = statistics.mean(rank_x)
    mean_y = statistics.mean(rank_y)
    numerator = sum(
        (left - mean_x) * (right - mean_y)
        for left, right in zip(rank_x, rank_y)
    )
    denominator = math.sqrt(
        sum((value - mean_x) ** 2 for value in rank_x)
        * sum((value - mean_y) ** 2 for value in rank_y)
    )
    return numerator / denominator if denominator else 0.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {}  # (policy, task, row) -> recs
    for policy in SIXTEEN_POLICIES:
        for t in TASKS:
            for row in QUERIES[t]:
                data[(policy, t, row)] = load_rollouts(policy, t, row)

    # ---------- A. budget frontier ----------
    rows = []
    inconsistent_answer_groups: set[tuple[str, str, int, str]] = set()
    for policy in SIXTEEN_POLICIES:
        for t in TASKS:
            wall = []
            for row in QUERIES[t]:
                wall += [r["wall_time_seconds"] for r in data[(policy, t, row)]]
            mean_wall = statistics.mean(wall)
            max_k = len(data[(policy, t, QUERIES[t][0])])
            for k in BUDGET_KS:
                if k > max_k:
                    continue
                p_list, mv_list = [], []
                for row in QUERIES[t]:
                    recs = data[(policy, t, row)]
                    c = sum(1 for r in recs if r["correct"])
                    p_list.append(pass_at_k(len(recs), c, k))
                    answers = [r["parsed_answer"] for r in recs]
                    flags = [bool(r["correct"]) for r in recs]
                    # Policies with the same rollout count use identical sampled
                    # index sets, reducing Monte Carlo noise in policy contrasts.
                    rng = random.Random(f"{MC_SEED}|{t}|{row}|{k}|{len(recs)}")
                    mv, inconsistent = majority_vote_accuracy(answers, flags, k, rng)
                    mv_list.append(mv)
                    if inconsistent:
                        grouped = defaultdict(set)
                        for answer, correct in zip(answers, flags):
                            grouped[answer].add(correct)
                        for answer, answer_flags in grouped.items():
                            if len(answer_flags) > 1:
                                inconsistent_answer_groups.add((policy, t, row, answer))
                rows.append({
                    "policy": policy, "task": t, "k": k,
                    "observed_serial_runtime_proxy_s": round(k * mean_wall, 2),
                    "mean_observed_wall_per_generation_s": round(mean_wall, 2),
                    "mean_pass_at_k": round(statistics.mean(p_list), 5),
                    "mean_majority_vote_k": round(statistics.mean(mv_list), 5),
                })
    with (OUT / "pareto_frontier.csv").open("w") as f:
        f.write("policy,task,k,observed_serial_runtime_proxy_s,mean_observed_wall_per_generation_s,mean_pass_at_k,mean_majority_vote_k\n")
        for r in rows:
            f.write(f"{r['policy']},{r['task']},{r['k']},{r['observed_serial_runtime_proxy_s']},{r['mean_observed_wall_per_generation_s']},{r['mean_pass_at_k']},{r['mean_majority_vote_k']}\n")

    # ---------- B. diversity ----------
    div = {}
    for k in (16, 64):
        policies = SIXTEEN_POLICIES if k == 16 else SIXTYFOUR_POLICIES
        for policy in policies:
            for t in TASKS:
                uniq, ent, modal = [], [], []
                for row in QUERIES[t]:
                    recs = data[(policy, t, row)][:k]
                    answers = [r["parsed_answer"] for r in recs]
                    uniq.append(len(set(a for a in answers if a is not None)))
                    ent.append(answer_entropy(answers))
                    c = Counter(a for a in answers if a is not None)
                    modal.append(max(c.values()) / k if c else 0.0)
                div[f"{policy}|{t}|k{k}"] = {
                    "mean_unique_answers": round(statistics.mean(uniq), 4),
                    "mean_answer_entropy": round(statistics.mean(ent), 4),
                    "mean_modal_share": round(statistics.mean(modal), 4),
                }
    (OUT / "diversity.json").write_text(json.dumps(div, indent=2, sort_keys=True))

    # ---------- C. EF vs AO mechanism ----------
    mech = {"per_task": {}}
    for t in TASKS:
        d_len, d_cov = [], []
        ef_lens, ao_lens, ef_cov, ao_cov = [], [], [], []
        for row in QUERIES[t]:
            ef = data[("ef_b32", t, row)]
            ao = data[("ao_b32", t, row)]
            el = statistics.mean(len(r["response"].split()) for r in ef)
            al = statistics.mean(len(r["response"].split()) for r in ao)
            ec = sum(1 for r in ef if r["correct"])
            ac = sum(1 for r in ao if r["correct"])
            e_cov = statistics.mean(pass_at_k(64, ec, k) for k in (4, 8, 16, 32, 64))
            a_cov = statistics.mean(pass_at_k(64, ac, k) for k in (4, 8, 16, 32, 64))
            ef_lens.append(el); ao_lens.append(al); ef_cov.append(e_cov); ao_cov.append(a_cov)
            d_len.append(el - al); d_cov.append(e_cov - a_cov)
        mech["per_task"][t] = {
            "mean_ef_len": round(statistics.mean(ef_lens), 3),
            "mean_ao_len": round(statistics.mean(ao_lens), 3),
            "mean_ef_cov": round(statistics.mean(ef_cov), 5),
            "mean_ao_cov": round(statistics.mean(ao_cov), 5),
            "spearman_len_cov_ef": round(spearman(ef_lens, ef_cov), 4),
            "spearman_len_cov_ao": round(spearman(ao_lens, ao_cov), 4),
            "spearman_dlen_dcov": round(spearman(d_len, d_cov), 4),
        }

    # commit-entropy subsample: traces r00-r03, EF vs AO
    def mean_commit_entropy(policy: str, t: str, row: int) -> float:
        vals = []
        for r in ENTROPY_SUBSAMPLE_ROLLOUTS:
            if policy == "ef_b32":
                p = CA / t / policy / f"q{row:04d}" / f"r{r:02d}.trace.jsonl.gz"
            else:
                p = P1 / "screening" / t / policy / f"q{row:04d}" / f"r{r:02d}.trace.jsonl.gz"
            with gzip.open(p, "rt") as h:
                for line in h:
                    ev = json.loads(line)
                    ents = ev.get("entropy") or []
                    sel = ev.get("selected_positions") or []
                    pos_index = {q: i for i, q in enumerate(ev.get("eligible_positions") or [])}
                    for s in sel:
                        i = pos_index.get(s)
                        if i is not None and i < len(ents):
                            v = ents[i]
                            if isinstance(v, (int, float)) and math.isfinite(v):
                                vals.append(float(v))
        return statistics.mean(vals) if vals else float("nan")

    ent_out = {}
    for t in TASKS:
        ef_e, ao_e = [], []
        for row in QUERIES[t]:
            ef_e.append(mean_commit_entropy("ef_b32", t, row))
            ao_e.append(mean_commit_entropy("ao_b32", t, row))
        ent_out[t] = {
            "ef_mean_commit_entropy": round(statistics.mean(ef_e), 4),
            "ao_mean_commit_entropy": round(statistics.mean(ao_e), 4),
            "subsample": f"rollouts r00-r03, {len(QUERIES[t])} queries",
        }
    mech["commit_entropy_subsample"] = ent_out
    (OUT / "ef_mechanism.json").write_text(json.dumps(mech, indent=2, sort_keys=True))

    summary = {
        "status": "exploratory post-unblinding analyses complete",
        "outputs": [
            "pareto_frontier.csv",
            "diversity.json",
            "ef_mechanism.json",
            "summary.json",
        ],
        "mc_draws": MC_DRAWS, "mc_seed": MC_SEED,
        "majority_vote_definition": (
            "Unique parsed-answer mode; exact modal ties count as incorrect. "
            "Correctness of an answer string is the majority frozen-grader label "
            "among records with that string."
        ),
        "inconsistent_answer_groups": len(inconsistent_answer_groups),
        "runtime_scope": (
            "Observed serial wall-time proxy from instrumented runs conducted at "
            "different times and shared-server loads; not controlled deployment "
            "latency. Rollout count k is the comparable nominal inference budget."
        ),
        "diversity_scope": (
            "Diversity is computed over frozen parsed-answer strings. Semantically "
            "equivalent strings may remain distinct."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
