"""Frozen Phase 0 result aggregation and mechanism sanity analysis."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from scipy.stats import spearmanr

from .artifacts import write_json_atomic, write_text_atomic
from .metrics import aggregate_pass_at_k, estimate_pass_at_k
from .protocol import (
    MODEL_ID,
    MODEL_REVISION,
    POLICIES,
    stage_query_indices,
    stage_rollout_count,
)


K_GRID = (1, 2, 4, 8, 16, 32)
BOOTSTRAP_SEED = 20260820
BOOTSTRAP_REPLICATES = 10_000
CONNECTORS = {
    "therefore", "thus", "so", "since", "when", "given", "however",
    "let", "first", "then", "next", "finally", "now", "similarly",
    "calculate", "solving", "notice", "specifically", "follows",
    "because", "but", "or", "consider", "also", "express", "write",
}


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a quantile of an empty list")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def paired_bootstrap_ci(
    first: list[float],
    second: list[float],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    if len(first) != len(second) or not first:
        raise ValueError("paired samples must be nonempty and have equal length")
    rng = random.Random(seed)
    size = len(first)
    estimates = []
    for _ in range(replicates):
        indices = [rng.randrange(size) for _ in range(size)]
        estimates.append(
            mean(first[index] - second[index] for index in indices)
        )
    estimates.sort()
    return _quantile(estimates, 0.025), _quantile(estimates, 0.975)


def load_result_records(root: Path, stage: str, allow_incomplete: bool) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = {}
    expected_per_policy = len(stage_query_indices(stage)) * stage_rollout_count(stage)
    for policy in POLICIES:
        policy_root = root / stage / policy.name
        rows = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(policy_root.glob("q*/r*.json"))
        ]
        keys = [(row["query_index"], row["rollout_index"]) for row in rows]
        if len(keys) != len(set(keys)):
            raise RuntimeError(f"duplicate result keys under {policy_root}")
        if not allow_incomplete and len(rows) != expected_per_policy:
            raise RuntimeError(
                f"{policy.name}: expected {expected_per_policy} results, found {len(rows)}"
            )
        records[policy.name] = rows
    return records


def _success_counts(rows: list[dict[str, Any]]) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for row in rows:
        counts[int(row["query_index"])] += int(bool(row["correct"]))
    return dict(counts)


def _policy_summary(rows: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    n = stage_rollout_count(stage)
    counts = _success_counts(rows)
    ordered_queries = stage_query_indices(stage)
    observed_counts = [counts.get(query_index, 0) for query_index in ordered_queries]
    return {
        "result_count": len(rows),
        "query_count": len(set(row["query_index"] for row in rows)),
        "parser_failure_count": sum(
            not str(row.get("parser_status", "")).startswith("ok") for row in rows
        ),
        "incomplete_count": sum(row.get("completion_status") != "complete" for row in rows),
        "mean_wall_time_seconds": mean(row["wall_time_seconds"] for row in rows) if rows else None,
        "pass_at_k": {
            str(k): aggregate_pass_at_k(observed_counts, n=n, k=k)
            for k in K_GRID
            if k <= n
        },
        "solved_query_count": sum(count > 0 for count in observed_counts),
        "success_counts": {
            str(query_index): counts.get(query_index, 0)
            for query_index in ordered_queries
        },
    }


def _load_tokenizer(model_path: Path | None = None) -> Any:
    from transformers import AutoTokenizer

    if model_path is not None:
        return AutoTokenizer.from_pretrained(
            str(model_path.resolve()),
            trust_remote_code=True,
            local_files_only=True,
        )
    return AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
    )


def _normalize_token(tokenizer: Any, token_id: int) -> str:
    text = tokenizer.decode([token_id], skip_special_tokens=True).strip().lower()
    words = re.findall(r"[a-z]+", text)
    return words[0] if len(words) == 1 else ""


def _finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def _describe(values: list[float]) -> dict[str, float | int | None]:
    values = sorted(_finite(values))
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "upper_quartile": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "upper_quartile": _quantile(values, 0.75),
        "minimum": values[0],
        "maximum": values[-1],
    }


def _spearman(first: list[float], second: list[float]) -> dict[str, float | int | None]:
    paired = [
        (left, right)
        for left, right in zip(first, second, strict=True)
        if math.isfinite(left) and math.isfinite(right)
    ]
    if len(paired) < 3:
        return {"n": len(paired), "rho": None, "p_value": None}
    correlation, p_value = spearmanr(
        [left for left, _ in paired], [right for _, right in paired]
    )
    return {"n": len(paired), "rho": float(correlation), "p_value": float(p_value)}


def _closure_records(root: Path, stage: str, tokenizer: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    token_cache: dict[int, str] = {}
    for path in sorted((root / stage / "ao_b32").glob("q*/r*.trace.jsonl.gz")):
        query_index = int(path.parent.name.removeprefix("q"))
        rollout_index = int(path.name.split(".")[0].removeprefix("r"))
        events: list[dict[str, Any]] = []
        with gzip.open(path, mode="rt", encoding="utf-8") as handle:
            events.extend(json.loads(line) for line in handle)
        max_step = max((event["step_in_block"] for event in events), default=0)
        for event in events:
            position_to_index = {
                position: index
                for index, position in enumerate(event["eligible_positions"])
            }
            for position, token_id in zip(
                event["selected_positions"],
                event["selected_token_ids"],
                strict=True,
            ):
                index = position_to_index[position]
                if token_id not in token_cache:
                    token_cache[token_id] = _normalize_token(tokenizer, token_id)
                rows.append(
                    {
                        "query_index": query_index,
                        "rollout_index": rollout_index,
                        "position": position,
                        "token_id": token_id,
                        "normalized_token": token_cache[token_id],
                        "is_connector": token_cache[token_id] in CONNECTORS,
                        "deferral_steps": event["step_in_block"],
                        "normalized_commitment_rank": (
                            event["step_in_block"] / max_step if max_step else 0.0
                        ),
                        "commit_entropy": event["entropy"][index],
                        "entropy_closure": event["eligible_entropy"][index]
                        - event["entropy"][index],
                        "margin_closure": event["logit_margins"][index]
                        - event["eligible_logit_margins"][index],
                    }
                )
    return rows


def _closure_summary(
    closures: list[dict[str, Any]],
    ar_summary: dict[str, Any],
    ao_summary: dict[str, Any],
    stage: str,
) -> dict[str, Any]:
    connector = [row for row in closures if row["is_connector"]]
    other = [row for row in closures if not row["is_connector"]]
    per_query_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in closures:
        per_query_rows[row["query_index"]].append(row)

    query_indices = stage_query_indices(stage)
    n = stage_rollout_count(stage)
    closure_values = [
        mean(row["entropy_closure"] for row in per_query_rows[query_index])
        for query_index in query_indices
        if per_query_rows.get(query_index)
    ]
    coverage_contrasts = [
        (
            ar_summary["success_counts"][str(query_index)]
            - ao_summary["success_counts"][str(query_index)]
        )
        / n
        for query_index in query_indices
        if per_query_rows.get(query_index)
    ]
    correlation, p_value = (
        spearmanr(closure_values, coverage_contrasts)
        if len(closure_values) >= 3
        else (float("nan"), float("nan"))
    )
    deferral_values = [row["deferral_steps"] for row in closures]
    closure_values_all = [row["entropy_closure"] for row in closures]
    rank_bins = [
        ("early (0.00-0.24)", 0.0, 0.25),
        ("early-middle (0.25-0.49)", 0.25, 0.50),
        ("late-middle (0.50-0.74)", 0.50, 0.75),
        ("late (0.75-1.00)", 0.75, 1.0000001),
    ]
    closure_by_rank = []
    for label, lower, upper in rank_bins:
        values = [
            row["entropy_closure"]
            for row in closures
            if lower <= row["normalized_commitment_rank"] < upper
        ]
        closure_by_rank.append(
            {"bin": label, "rank_lower": lower, "rank_upper": upper, **_describe(values)}
        )
    query_summaries = []
    for query_index in query_indices:
        query_rows = per_query_rows.get(query_index, [])
        if not query_rows:
            continue
        query_summaries.append(
            {
                "query_index": query_index,
                "token_count": len(query_rows),
                "entropy_closure": _describe(
                    [row["entropy_closure"] for row in query_rows]
                ),
                "deferral_steps": _describe(
                    [row["deferral_steps"] for row in query_rows]
                ),
            }
        )
    return {
        "token_count": len(closures),
        "connector_token_count": len(connector),
        "connector_query_count": len({row["query_index"] for row in connector}),
        "median_entropy_closure_all": median(
            row["entropy_closure"] for row in closures
        ) if closures else None,
        "median_entropy_closure_connector": median(
            row["entropy_closure"] for row in connector
        ) if connector else None,
        "median_entropy_closure_other": median(
            row["entropy_closure"] for row in other
        ) if other else None,
        "median_margin_closure_connector": median(
            row["margin_closure"] for row in connector
        ) if connector else None,
        "spearman_query_closure_vs_success_contrast": float(correlation),
        "spearman_p_value_descriptive": float(p_value),
        "deferral_steps": _describe(deferral_values),
        "deferral_vs_entropy_closure": _spearman(
            deferral_values, closure_values_all
        ),
        "deferral_vs_margin_closure": _spearman(
            deferral_values, [row["margin_closure"] for row in closures]
        ),
        "closure_by_normalized_commitment_rank": closure_by_rank,
        "query_level": {
            "query_count": len(query_summaries),
            "mean_entropy_closure": _describe(
                [
                    item["entropy_closure"]["mean"]
                    for item in query_summaries
                    if item["entropy_closure"]["mean"] is not None
                ]
            ),
            "upper_quartile_entropy_closure": _describe(
                [
                    item["entropy_closure"]["upper_quartile"]
                    for item in query_summaries
                    if item["entropy_closure"]["upper_quartile"] is not None
                ]
            ),
            "per_query": query_summaries,
        },
        "global_average_commit_time_entropy": mean(
            row["commit_entropy"] for row in closures
        ) if closures else None,
    }


def analyze(
    root: Path,
    stage: str,
    allow_incomplete: bool = False,
    model_path: Path | None = None,
) -> dict[str, Any]:
    records = load_result_records(root, stage, allow_incomplete)
    summaries = {
        policy.name: _policy_summary(records[policy.name], stage)
        for policy in POLICIES
    }
    query_indices = stage_query_indices(stage)
    n = stage_rollout_count(stage)
    ar_pass = [
        estimate_pass_at_k(n, summaries["ar_b1"]["success_counts"][str(query)], n)
        for query in query_indices
    ]
    ao_pass = [
        estimate_pass_at_k(n, summaries["ao_b32"]["success_counts"][str(query)], n)
        for query in query_indices
    ]
    delta = mean(first - second for first, second in zip(ar_pass, ao_pass, strict=True))
    ci_low, ci_high = paired_bootstrap_ci(ar_pass, ao_pass)

    ar_solved = {query for query, value in zip(query_indices, ar_pass, strict=True) if value}
    ao_solved = {query for query, value in zip(query_indices, ao_pass, strict=True) if value}

    tokenizer = _load_tokenizer(model_path)
    closures = _closure_records(root, stage, tokenizer)
    return {
        "stage": stage,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "policies": summaries,
        "coverage_contrast": {
            "delta_pass_at_max_k_ar_minus_ao": delta,
            "paired_query_bootstrap_95_ci": [ci_low, ci_high],
            "ar_only_solved": len(ar_solved - ao_solved),
            "ao_only_solved": len(ao_solved - ar_solved),
            "both_solved": len(ar_solved & ao_solved),
            "neither_solved": len(set(query_indices) - (ar_solved | ao_solved)),
        },
        "closure": _closure_summary(
            closures, summaries["ar_b1"], summaries["ao_b32"], stage
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    coverage = report["coverage_contrast"]
    closure = report["closure"]
    analysis_kind = report["stage"]
    lines = [
        f"# Phase 0 Analysis: {report['stage']}",
        "",
        "## Coverage",
        "",
        "| Policy | Pass@1 | Pass@32/max-k | Solved queries |",
        "|---|---:|---:|---:|",
    ]
    for policy in POLICIES:
        summary = report["policies"][policy.name]
        max_k = max(int(k) for k in summary["pass_at_k"])
        lines.append(
            f"| {policy.name} | {summary['pass_at_k']['1']:.4f} | "
            f"{summary['pass_at_k'][str(max_k)]:.4f} | "
            f"{summary['solved_query_count']} |"
        )
    ci = coverage["paired_query_bootstrap_95_ci"]
    lines.extend(
        [
            "",
            f"AR minus AO max-k contrast: **{coverage['delta_pass_at_max_k_ar_minus_ao']:.4f}** "
            f"(paired query bootstrap 95% CI [{ci[0]:.4f}, {ci[1]:.4f}]).",
            "",
            f"Solved-set counts: AR-only {coverage['ar_only_solved']}, "
            f"AO-only {coverage['ao_only_solved']}, both {coverage['both_solved']}, "
            f"neither {coverage['neither_solved']}.",
            "",
            "## AO Closure Sanity Check",
            "",
            f"- Selected tokens: {closure['token_count']}",
            f"- Lexical connector tokens: {closure['connector_token_count']} "
            f"across {closure['connector_query_count']} queries",
            f"- Median entropy closure, all: {closure['median_entropy_closure_all']}",
            f"- Median entropy closure, connectors: {closure['median_entropy_closure_connector']}",
            f"- Median entropy closure, others: {closure['median_entropy_closure_other']}",
            f"- Query closure vs. AR-minus-AO success contrast Spearman: "
            f"{closure['spearman_query_closure_vs_success_contrast']:.4f}",
            f"- Deferral steps: mean {closure['deferral_steps']['mean']:.4f}, "
            f"median {closure['deferral_steps']['median']:.4f}, "
            f"upper quartile {closure['deferral_steps']['upper_quartile']:.4f}",
            f"- Deferral vs. entropy-closure Spearman: "
            f"{closure['deferral_vs_entropy_closure']['rho']:.4f} "
            f"(n={closure['deferral_vs_entropy_closure']['n']})",
            f"- Global mean commit-time entropy: "
            f"{closure['global_average_commit_time_entropy']:.6f}",
            "- Query-level entropy closure mean distribution: "
            f"median {closure['query_level']['mean_entropy_closure']['median']:.6f}, "
            f"upper quartile {closure['query_level']['mean_entropy_closure']['upper_quartile']:.6f}",
            "- Query-level entropy closure upper-quartile distribution: "
            f"median {closure['query_level']['upper_quartile_entropy_closure']['median']:.6f}, "
            f"upper quartile {closure['query_level']['upper_quartile_entropy_closure']['upper_quartile']:.6f}",
            "- Closure by normalized commitment rank is included in the JSON report.",
            "",
            f"This is a frozen Phase 0 {analysis_kind} analysis. Lexical connectors are a replication sanity set, not a causal fork endpoint.",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--stage", choices=["equivalence", "signal"], required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = analyze(
        args.input,
        args.stage,
        args.allow_incomplete,
        model_path=args.model_path,
    )
    write_json_atomic(args.output_json, report)
    write_text_atomic(args.output_md, render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
