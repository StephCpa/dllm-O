"""Fit and freeze calibration-only Phase 1 boundary-closure analyses."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np
from scipy.stats import spearmanr

from .artifacts import sha256_file, write_json_atomic, write_text_atomic
from .metrics import estimate_pass_at_k
from .phase1_protocol import SPLIT_MANIFEST_SHA256, load_split_manifest


FREEZE_VERSION = "phase1-bci-freeze-v1"
BOOTSTRAP_SEED = 20260825
RIDGE_FOLD_SEED = 20260825
RIDGE_ALPHAS = (0.0, 0.1, 1.0, 10.0, 100.0)
SCREENING_K_GRID = (4, 8, 16)
FORMAL_HELDOUT_K_GRID = (4, 8, 16, 32, 64)
MASK_TOKEN_ID = 126336
EPSILON = 1e-12
SUPERSEDED_FREEZES = (
    {
        "sha256": "dea7e7a6e76a33f00be402f86ec8bb8b1d62c52c17c19fe83e7fccc864bfb64e",
        "reason": "Matching did not balance eligible entropy or eligible margin, mechanically coupling candidate status to the closure endpoint.",
    },
    {
        "sha256": "ef4feb365ddc823ec0b8e6c8895935580cac556dfdfc8f7c1ab3325dc9c7ec20",
        "reason": "Balanced matching was retained, but the freeze omitted a prospective zero-match policy and minimum matched-query coverage threshold.",
    },
)


def _quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a quantile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _robust_location_scale(values: Iterable[float]) -> tuple[float, float]:
    ordered = sorted(float(value) for value in values if math.isfinite(value))
    if not ordered:
        raise ValueError("cannot normalize an empty sequence")
    location = median(ordered)
    scale = _quantile(ordered, 0.75) - _quantile(ordered, 0.25)
    return float(location), float(scale if scale > EPSILON else 1.0)


def _candidate_score(
    eligible_entropy: float,
    eligible_margin: float,
    normalization: dict[str, float],
) -> float:
    entropy_z = (eligible_entropy - normalization["entropy_median"]) / normalization[
        "entropy_iqr"
    ]
    low_margin = -math.log(max(eligible_margin, 0.0) + EPSILON)
    margin_z = (low_margin - normalization["low_margin_median"]) / normalization[
        "low_margin_iqr"
    ]
    return 0.5 * max(-5.0, min(5.0, entropy_z)) + 0.5 * max(-5.0, min(5.0, margin_z))


def _read_trace(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, mode="rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _selected_token_rows(
    events: list[dict[str, Any]], *, task: str, query: int, rollout: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        positions = event["eligible_positions"]
        position_to_index = {
            int(position): index for index, position in enumerate(positions)
        }
        block_start = int(event["active_block_start"])
        block_end = int(event["active_block_end"])
        block_width = max(1, block_end - block_start - 1)
        # AO-B32 uses 32 denoising steps per block, indexed from zero.
        rank_denominator = 31.0
        for position, token_id in zip(
            event["selected_positions"], event["selected_token_ids"], strict=True
        ):
            index = position_to_index[int(position)]
            eligible_entropy = float(event["eligible_entropy"][index])
            eligible_margin = float(event["eligible_logit_margins"][index])
            commit_entropy = float(event["entropy"][index])
            commit_margin = float(event["logit_margins"][index])
            confidence = float(event["candidate_probabilities"][index])
            if not all(
                math.isfinite(value)
                for value in (
                    eligible_entropy,
                    eligible_margin,
                    commit_entropy,
                    commit_margin,
                    confidence,
                )
            ):
                continue
            if int(token_id) == MASK_TOKEN_ID or confidence <= 0.0:
                continue
            rows.append(
                {
                    "task": task,
                    "query": query,
                    "rollout": rollout,
                    "block": int(event["block_index"]),
                    "prompt_length": block_start - int(event["block_index"]) * 32,
                    "position": int(position),
                    "normalized_position": (int(position) - block_start) / block_width,
                    "normalized_commitment_rank": int(event["step_in_block"])
                    / rank_denominator,
                    "eligible_entropy": eligible_entropy,
                    "eligible_margin": eligible_margin,
                    "commit_confidence": confidence,
                    "entropy_closure": eligible_entropy - commit_entropy,
                    "margin_closure": commit_margin - eligible_margin,
                }
            )
    return rows


def _record_digest(entries: list[tuple[str, str, str]]) -> str:
    payload = "".join(
        f"{path}\t{result_hash}\t{trace_hash}\n"
        for path, result_hash, trace_hash in sorted(entries)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_calibration(
    phase0_root: Path, phase1_root: Path, manifest: dict[str, Any]
) -> tuple[
    dict[tuple[str, int, str], list[dict[str, Any]]],
    list[dict[str, Any]],
    dict[str, str],
]:
    records: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    token_rows: list[dict[str, Any]] = []
    digest_entries: dict[str, list[tuple[str, str, str]]] = {
        "phase0_gsm8k_reuse": [],
        "phase1_calibration": [],
    }

    calibration_rows = {
        task: {
            int(entry["row_index"]): entry
            for entry in manifest["splits"][task]["calibration"]
        }
        for task in ("gsm8k", "math500")
    }
    for task in ("gsm8k", "math500"):
        for query in sorted(calibration_rows[task]):
            # Freeze construction must not inspect secondary-policy outcomes.
            for policy in ("ar_b1", "ao_b32"):
                for rollout in range(16):
                    if task == "gsm8k" and policy in {"ar_b1", "ao_b32"}:
                        result_path = (
                            phase0_root
                            / "signal"
                            / policy
                            / f"q{query:04d}"
                            / f"r{rollout:02d}.json"
                        )
                        source_group = "phase0_gsm8k_reuse"
                    else:
                        result_path = (
                            phase1_root
                            / "phase1-v1"
                            / "calibration"
                            / "screening"
                            / task
                            / policy
                            / f"q{query:04d}"
                            / f"r{rollout:02d}.json"
                        )
                        source_group = "phase1_calibration"
                    if not result_path.is_file():
                        raise FileNotFoundError(
                            f"missing calibration result: {result_path}"
                        )
                    record = _load_json(result_path)
                    if record.get("completion_status") != "complete":
                        raise RuntimeError(
                            f"incomplete calibration result: {result_path}"
                        )
                    trace_path = result_path.with_name(
                        result_path.stem + ".trace.jsonl.gz"
                    )
                    if (
                        not trace_path.is_file()
                        or sha256_file(trace_path) != record["trace_sha256"]
                    ):
                        raise RuntimeError(f"trace hash mismatch: {trace_path}")
                    if source_group == "phase0_gsm8k_reuse":
                        observed_query = int(record["query_index"])
                        observed_policy = str(record["policy"])
                    else:
                        observed_query = int(record["source"]["row_index"])
                        observed_policy = str(record["policy"]["name"])
                        if (
                            record.get("run_kind") != "screening"
                            or record.get("split_manifest_sha256")
                            != SPLIT_MANIFEST_SHA256
                            or record["source"].get("split_role") != "calibration"
                            or record["source"].get("prompt_sha256")
                            != calibration_rows[task][query]["prompt_sha256"]
                        ):
                            raise RuntimeError(
                                f"Phase 1 calibration provenance mismatch: {result_path}"
                            )
                    if (
                        observed_query != query
                        or observed_policy != policy
                        or int(record["rollout_index"]) != rollout
                    ):
                        raise RuntimeError(f"calibration key mismatch: {result_path}")
                    records[(task, query, policy)].append(record)
                    relative = f"{source_group}/{task}/{policy}/q{query:04d}/r{rollout:02d}.json"
                    digest_entries[source_group].append(
                        (
                            relative,
                            sha256_file(result_path),
                            str(record["trace_sha256"]),
                        )
                    )
                    if policy == "ao_b32":
                        token_rows.extend(
                            _selected_token_rows(
                                _read_trace(trace_path),
                                task=task,
                                query=query,
                                rollout=rollout,
                            )
                        )
    hashes = {
        group: _record_digest(entries) for group, entries in digest_entries.items()
    }
    return records, token_rows, hashes


def _fit_candidate_rule(token_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in token_rows:
        by_task[row["task"]].append(row)
    fitted: dict[str, Any] = {}
    for task, rows in sorted(by_task.items()):
        entropy_location, entropy_scale = _robust_location_scale(
            row["eligible_entropy"] for row in rows
        )
        margin_location, margin_scale = _robust_location_scale(
            -math.log(max(row["eligible_margin"], 0.0) + EPSILON) for row in rows
        )
        normalization = {
            "entropy_median": entropy_location,
            "entropy_iqr": entropy_scale,
            "low_margin_median": margin_location,
            "low_margin_iqr": margin_scale,
        }
        scores = [
            _candidate_score(
                row["eligible_entropy"], row["eligible_margin"], normalization
            )
            for row in rows
        ]
        fitted[task] = {
            **normalization,
            "candidate_threshold_q75": _quantile(scores, 0.75),
            "eligible_token_count": len(rows),
        }
    return fitted


def _annotate_candidates(
    token_rows: list[dict[str, Any]], fitted_rule: dict[str, Any]
) -> None:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in token_rows:
        params = fitted_rule[row["task"]]
        row["candidate_score"] = _candidate_score(
            row["eligible_entropy"], row["eligible_margin"], params
        )
        row["candidate"] = row["candidate_score"] >= params["candidate_threshold_q75"]
        grouped[(row["task"], row["query"], row["rollout"])].append(row)
    for rows in grouped.values():
        if not any(row["candidate"] for row in rows):
            max(rows, key=lambda row: (row["candidate_score"], -row["position"]))[
                "candidate"
            ] = True


def _match_candidates(
    token_rows: list[dict[str, Any]], fitted_rule: dict[str, Any]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in token_rows:
        grouped[(row["task"], row["query"], row["rollout"], row["block"])].append(row)
    matches: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        task = key[0]
        candidates = sorted(
            (row for row in rows if row["candidate"]),
            key=lambda row: (-row["candidate_score"], row["position"]),
        )
        controls = {row["position"]: row for row in rows if not row["candidate"]}
        for candidate in candidates:
            entropy_scale = fitted_rule[task]["entropy_iqr"]
            margin_scale = fitted_rule[task]["low_margin_iqr"]
            candidate_low_margin = -math.log(
                max(candidate["eligible_margin"], 0.0) + EPSILON
            )
            eligible = [
                control
                for control in controls.values()
                if abs(
                    candidate["normalized_position"] - control["normalized_position"]
                )
                <= 0.25
                and abs(
                    candidate["normalized_commitment_rank"]
                    - control["normalized_commitment_rank"]
                )
                <= 0.25
                and abs(candidate["eligible_entropy"] - control["eligible_entropy"])
                / entropy_scale
                <= 0.25
                and abs(
                    candidate_low_margin
                    - (-math.log(max(control["eligible_margin"], 0.0) + EPSILON))
                )
                / margin_scale
                <= 0.25
            ]
            if not eligible:
                continue
            control = min(
                eligible,
                key=lambda row: (
                    abs(candidate["normalized_position"] - row["normalized_position"])
                    + abs(
                        candidate["normalized_commitment_rank"]
                        - row["normalized_commitment_rank"]
                    )
                    + abs(candidate["eligible_entropy"] - row["eligible_entropy"])
                    / entropy_scale
                    + abs(
                        candidate_low_margin
                        - (-math.log(max(row["eligible_margin"], 0.0) + EPSILON))
                    )
                    / margin_scale,
                    row["position"],
                ),
            )
            controls.pop(control["position"])
            matches.append(
                {
                    "task": task,
                    "query": candidate["query"],
                    "rollout": candidate["rollout"],
                    "block": candidate["block"],
                    "candidate_position": candidate["position"],
                    "control_position": control["position"],
                    "eligible_entropy_difference": candidate["eligible_entropy"]
                    - control["eligible_entropy"],
                    "low_margin_difference": candidate_low_margin
                    - (-math.log(max(control["eligible_margin"], 0.0) + EPSILON)),
                    "entropy_closure_difference": candidate["entropy_closure"]
                    - control["entropy_closure"],
                    "margin_closure_difference": candidate["margin_closure"]
                    - control["margin_closure"],
                }
            )
    return matches


def _query_features(
    records: dict[tuple[str, int, str], list[dict[str, Any]]],
    token_rows: list[dict[str, Any]],
    fitted_rule: dict[str, Any],
) -> list[dict[str, Any]]:
    by_query: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    by_rollout: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in token_rows:
        by_query[(row["task"], row["query"])].append(row)
        by_rollout[(row["task"], row["query"], row["rollout"])].append(row)
    output: list[dict[str, Any]] = []
    for task, query in sorted(by_query):
        query_rows = by_query[(task, query)]
        rollout_bci = []
        for rollout in range(16):
            candidates = [
                row for row in by_rollout[(task, query, rollout)] if row["candidate"]
            ]
            if not candidates:
                raise RuntimeError(
                    f"candidate fallback failed for {task} query {query}"
                )
            threshold = fitted_rule[task]["candidate_threshold_q75"]
            weights = [
                1.0 + max(row["candidate_score"] - threshold, 0.0) for row in candidates
            ]
            rollout_bci.append(
                sum(
                    weight * row["entropy_closure"]
                    for weight, row in zip(weights, candidates, strict=True)
                )
                / sum(weights)
            )
        ar_rows = records[(task, query, "ar_b1")]
        ao_rows = records[(task, query, "ao_b32")]
        target_by_k = {}
        for k in SCREENING_K_GRID:
            ar_successes = sum(bool(row["correct"]) for row in ar_rows)
            ao_successes = sum(bool(row["correct"]) for row in ao_rows)
            target_by_k[str(k)] = estimate_pass_at_k(
                16, ar_successes, k
            ) - estimate_pass_at_k(16, ao_successes, k)
        first_trace_rows = by_rollout[(task, query, 0)]
        prompt_lengths = {int(row["prompt_length"]) for row in first_trace_rows}
        if len(prompt_lengths) != 1:
            raise RuntimeError(f"inconsistent prompt length for {task} query {query}")
        prompt_length = prompt_lengths.pop()
        response_lengths = []
        for row in ao_rows:
            response = row.get("decoded_text", row.get("response", ""))
            response_lengths.append(len(str(response).split()))
        output.append(
            {
                "task": task,
                "query": query,
                "task_math500": float(task == "math500"),
                "prompt_length": float(prompt_length),
                "output_length": float(mean(response_lengths)),
                "mean_eligible_entropy": float(
                    mean(row["eligible_entropy"] for row in query_rows)
                ),
                "mean_commit_confidence": float(
                    mean(row["commit_confidence"] for row in query_rows)
                ),
                "bci": float(mean(rollout_bci)),
                "screening_coverage_auc_advantage": float(mean(target_by_k.values())),
                "coverage_advantage_by_k": target_by_k,
            }
        )
    return output


def _stratified_folds(rows: list[dict[str, Any]], folds: int = 5) -> list[list[int]]:
    assignments: list[list[int]] = [[] for _ in range(folds)]
    rng = random.Random(RIDGE_FOLD_SEED)
    for task in sorted({row["task"] for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["task"] == task]
        rng.shuffle(indices)
        for offset, index in enumerate(indices):
            assignments[offset % folds].append(index)
    return assignments


def _standardization(
    rows: list[dict[str, Any]], features: list[str]
) -> dict[str, dict[str, float]]:
    params = {}
    for feature in features:
        values = np.asarray([row[feature] for row in rows], dtype=float)
        scale = float(values.std(ddof=0))
        params[feature] = {
            "mean": float(values.mean()),
            "scale": scale if scale > EPSILON else 1.0,
        }
    return params


def _design(
    rows: list[dict[str, Any]],
    features: list[str],
    normalization: dict[str, dict[str, float]],
) -> np.ndarray:
    columns = [
        np.ones(len(rows), dtype=float),
        np.asarray([row["task_math500"] for row in rows]),
    ]
    for feature in features:
        params = normalization[feature]
        columns.append(
            (np.asarray([row[feature] for row in rows], dtype=float) - params["mean"])
            / params["scale"]
        )
    return np.column_stack(columns)


def _ridge_coefficients(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(x.shape[1], dtype=float) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.pinv(x.T @ x + penalty) @ x.T @ y


def _fit_predictor(rows: list[dict[str, Any]], features: list[str]) -> dict[str, Any]:
    folds = _stratified_folds(rows)
    scores: dict[str, float] = {}
    for alpha in RIDGE_ALPHAS:
        errors = []
        for test_indices in folds:
            test_set = set(test_indices)
            train = [row for index, row in enumerate(rows) if index not in test_set]
            test = [row for index, row in enumerate(rows) if index in test_set]
            normalization = _standardization(train, features)
            coefficients = _ridge_coefficients(
                _design(train, features, normalization),
                np.asarray([row["screening_coverage_auc_advantage"] for row in train]),
                alpha,
            )
            predictions = _design(test, features, normalization) @ coefficients
            errors.extend(
                abs(float(prediction) - row["screening_coverage_auc_advantage"])
                for prediction, row in zip(predictions, test, strict=True)
            )
        scores[str(alpha)] = float(mean(errors))
    selected_alpha = min(RIDGE_ALPHAS, key=lambda alpha: (scores[str(alpha)], alpha))
    normalization = _standardization(rows, features)
    coefficients = _ridge_coefficients(
        _design(rows, features, normalization),
        np.asarray([row["screening_coverage_auc_advantage"] for row in rows]),
        selected_alpha,
    )
    names = ["intercept", "task_math500", *features]
    return {
        "model": "ridge_regression",
        "target": "screening_coverage_auc_advantage_mean_pass_at_4_8_16",
        "heldout_evaluation_target": (
            "formal_coverage_auc_advantage_mean_pass_at_4_8_16_32_64"
        ),
        "alpha_grid": list(RIDGE_ALPHAS),
        "selected_alpha": selected_alpha,
        "fold_count": 5,
        "fold_seed": RIDGE_FOLD_SEED,
        "cross_validated_mae_by_alpha": scores,
        "feature_order": names,
        "coefficients": {
            name: float(value) for name, value in zip(names, coefficients, strict=True)
        },
        "standardization": normalization,
    }


def _repo_commit(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def calibrate(
    *, phase0_root: Path, phase1_root: Path, split_manifest: Path, repo_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_split_manifest(split_manifest)
    records, token_rows, artifact_hashes = _load_calibration(
        phase0_root, phase1_root, manifest
    )
    fitted_rule = _fit_candidate_rule(token_rows)
    _annotate_candidates(token_rows, fitted_rule)
    matches = _match_candidates(token_rows, fitted_rule)
    query_rows = _query_features(records, token_rows, fitted_rule)
    baseline_features = [
        "prompt_length",
        "output_length",
        "mean_eligible_entropy",
        "mean_commit_confidence",
    ]
    augmented_features = [*baseline_features, "bci"]
    baseline = _fit_predictor(query_rows, baseline_features)
    augmented = _fit_predictor(query_rows, augmented_features)
    candidate_count = sum(bool(row["candidate"]) for row in token_rows)
    by_task_matches = {
        task: [row for row in matches if row["task"] == task]
        for task in ("gsm8k", "math500")
    }

    def match_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        per_query: dict[tuple[str, int], int] = defaultdict(int)
        for row in rows:
            per_query[(row["task"], row["query"])] += 1
        return {
            "matched_pair_count": len(rows),
            "matched_query_count": len(per_query),
            "matched_pairs_per_query_minimum": (
                min(per_query.values()) if per_query else 0
            ),
            "matched_pairs_per_query_median": (
                median(per_query.values()) if per_query else 0
            ),
            "entropy_closure_difference_mean": (
                mean(row["entropy_closure_difference"] for row in rows)
                if rows
                else None
            ),
            "margin_closure_difference_mean": (
                mean(row["margin_closure_difference"] for row in rows) if rows else None
            ),
            "eligible_entropy_difference_mean": (
                mean(row["eligible_entropy_difference"] for row in rows)
                if rows
                else None
            ),
            "low_margin_difference_mean": (
                mean(row["low_margin_difference"] for row in rows) if rows else None
            ),
        }

    target_values = [row["screening_coverage_auc_advantage"] for row in query_rows]
    bci_values = [row["bci"] for row in query_rows]
    bci_correlation = spearmanr(bci_values, target_values)
    report = {
        "analysis_version": "phase1-calibration-analysis-v3",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "revision_history": {
            "superseded_preheldout_freezes": list(SUPERSEDED_FREEZES),
            "heldout_inference_started_before_this_freeze": False,
        },
        "split_manifest_sha256": SPLIT_MANIFEST_SHA256,
        "calibration_artifact_hashes": artifact_hashes,
        "calibration_scope": {
            "query_count": len(query_rows),
            "rollouts_per_policy": 16,
            "policies_read": ["ar_b1", "ao_b32"],
            "secondary_policy_outcomes_read": False,
            "target_proxy": "ScreeningCoverageAUC uses Pass@4/8/16 only",
            "formal_heldout_target": "CoverageAUC uses Pass@4/8/16/32/64",
            "phase0_reuse": "GSM8K AR-B1 and AO-B32 first 16 signal rollouts are Phase 1 calibration data only",
        },
        "candidate_rule": fitted_rule,
        "candidate_diagnostics": {
            "eligible_token_count": len(token_rows),
            "candidate_count": candidate_count,
            "candidate_fraction": candidate_count / len(token_rows),
            "matched_pair_count": len(matches),
            "match_rate": len(matches) / candidate_count,
            "entropy_closure_difference_mean": (
                mean(row["entropy_closure_difference"] for row in matches)
                if matches
                else None
            ),
            "margin_closure_difference_mean": (
                mean(row["margin_closure_difference"] for row in matches)
                if matches
                else None
            ),
            "eligible_entropy_difference_mean": (
                mean(row["eligible_entropy_difference"] for row in matches)
                if matches
                else None
            ),
            "low_margin_difference_mean": (
                mean(row["low_margin_difference"] for row in matches)
                if matches
                else None
            ),
            "by_task": {
                task: match_summary(rows) for task, rows in by_task_matches.items()
            },
        },
        "baseline_predictor": baseline,
        "augmented_predictor": augmented,
        "calibration_signal_diagnostics": {
            "screening_target_zero_fraction": sum(
                value == 0.0 for value in target_values
            )
            / len(target_values),
            "bci_target_spearman": {
                "rho": float(bci_correlation.statistic),
                "p_value_descriptive": float(bci_correlation.pvalue),
            },
            "baseline_selected_cv_mae": baseline["cross_validated_mae_by_alpha"][
                str(baseline["selected_alpha"])
            ],
            "augmented_selected_cv_mae": augmented["cross_validated_mae_by_alpha"][
                str(augmented["selected_alpha"])
            ],
        },
        "query_level_calibration": query_rows,
    }
    freeze = {
        "protocol_version": FREEZE_VERSION,
        "created_at_utc": report["created_at_utc"],
        "revision_history": report["revision_history"],
        "split_manifest_sha256": SPLIT_MANIFEST_SHA256,
        "calibration_artifact_hashes": artifact_hashes,
        "candidate_boundary_rule": {
            "formula": "0.5*clip(z_task(eligible_entropy),-5,5)+0.5*clip(z_task(-ln(max(eligible_logit_margin,0)+1e-12)),-5,5)",
            "task_parameters": fitted_rule,
            "selection": "score >= task calibration q75; if a rollout has none, select its highest-scoring eligible token",
            "uses_outcomes_or_final_answers": False,
        },
        "matching_rule": {
            "scope": "same task, query, rollout, and block",
            "control": "any non-candidate position below the task q75 threshold",
            "algorithm": "deterministic greedy nearest-neighbor without replacement; candidates ordered by descending score then position",
            "distance": "L1 over normalized within-block position, normalized commitment rank, task-IQR-scaled eligible entropy, and task-IQR-scaled negative-log eligible margin",
            "calipers": {
                "normalized_position": 0.25,
                "normalized_commitment_rank": 0.25,
                "eligible_entropy_in_task_iqr": 0.25,
                "negative_log_eligible_margin_in_task_iqr": 0.25,
            },
            "rationale": "balance both terms entering entropy/margin closure to prevent mechanical candidate-control separation",
            "endpoint_missingness": {
                "query_endpoint_defined_if": "at least one matched pair exists",
                "zero_match_query": "exclude from the closure-effect estimate, report as structurally unmatched, and do not alter calipers",
                "minimum_matched_query_fraction_per_task": 0.90,
                "below_minimum_consequence": "declare the task-level RQ1 endpoint unresolved; do not relax matching after held-out inspection",
            },
        },
        "bci_formula": {
            "token_weight": "1 + max(candidate_score - task_q75, 0)",
            "rollout": "weighted mean eligible-to-commit entropy closure over candidate positions",
            "query": "arithmetic mean of the 16 AO-B32 rollout BCIs",
        },
        "baseline_predictor": baseline,
        "augmented_predictor": augmented,
        "normalization": {
            "candidate_score": "task-specific calibration median/IQR with IQR fallback 1.0 and clipping to [-5,5]",
            "predictors": "calibration mean/population-SD; task indicator and intercept unstandardized",
        },
        "parser_policy": "parser errors and incorrect/truncated outputs remain failures in all Pass@k denominators",
        "exclusion_policy": {
            "excluded_tokens": "nonfinite trace metrics, selected mask token 126336, or nonpositive/nonfinite commit confidence",
            "not_excluded": "no lexical, semantic, correctness, or final-answer exclusions",
        },
        "target_policy": {
            "calibration_proxy": {
                "k_grid": list(SCREENING_K_GRID),
                "rollout_count": 16,
                "definition": "mean unbiased Pass@k over the listed k values",
            },
            "formal_heldout": {
                "k_grid": list(FORMAL_HELDOUT_K_GRID),
                "rollout_count": 64,
                "definition": "mean unbiased Pass@k over the listed k values",
            },
            "no_refit_on_heldout": True,
        },
        "failure_thresholds": {
            "minimum_matched_query_fraction_per_task": 0.90,
            "candidate_construction": "abort if any AO-B32 rollout has no eligible finite token for the deterministic fallback",
            "predictor_features": "abort on nonfinite frozen feature or prediction; do not impute from held-out outcomes",
            "heldout_rule_changes": "none permitted after any held-out outcome is read",
        },
        "bootstrap_seed": BOOTSTRAP_SEED,
        "runner_commit": _repo_commit(repo_root),
    }
    return report, freeze


def render_markdown(report: dict[str, Any]) -> str:
    diagnostics = report["candidate_diagnostics"]
    signal = report["calibration_signal_diagnostics"]
    lines = [
        "# Phase 1 Calibration and BCI Freeze Report",
        "",
        "This report uses calibration outcomes only. Its three-k ScreeningCoverageAUC is a fitting proxy, not the five-k formal held-out endpoint.",
        "",
        "## Candidate and Matching Diagnostics",
        "",
        f"- Eligible tokens: {diagnostics['eligible_token_count']}",
        f"- Candidate tokens: {diagnostics['candidate_count']} ({diagnostics['candidate_fraction']:.3%})",
        f"- Matched pairs: {diagnostics['matched_pair_count']} (match rate {diagnostics['match_rate']:.3%})",
        f"- Mean candidate-minus-control entropy closure: {diagnostics['entropy_closure_difference_mean']}",
        f"- Mean candidate-minus-control margin closure: {diagnostics['margin_closure_difference_mean']}",
        f"- Mean matched eligible-entropy difference: {diagnostics['eligible_entropy_difference_mean']}",
        f"- Mean matched negative-log-margin difference: {diagnostics['low_margin_difference_mean']}",
        "",
        "## Calibration Signal Diagnostics",
        "",
        f"- Screening target zero fraction: {signal['screening_target_zero_fraction']:.3%}",
        f"- BCI-target Spearman (descriptive): {signal['bci_target_spearman']['rho']:.6f}",
        f"- Baseline selected-CV MAE: {signal['baseline_selected_cv_mae']:.6f}",
        f"- Augmented selected-CV MAE: {signal['augmented_selected_cv_mae']:.6f}",
        "- These diagnostics are reported regardless of direction and do not select or alter the BCI formula.",
        "",
        "## Frozen Predictors",
        "",
        f"- Baseline selected ridge alpha: {report['baseline_predictor']['selected_alpha']}",
        f"- Augmented selected ridge alpha: {report['augmented_predictor']['selected_alpha']}",
        "- Both predictors are fitted to ScreeningCoverageAUC and will be evaluated without refitting against the formal held-out CoverageAUC.",
        "",
        "## Audit Boundary",
        "",
        "GSM8K AR-B1/AO-B32 uses the first 16 previously generated Phase 0 signal rollouts, reclassified solely as Phase 1 calibration data. MATH-500 and the other GSM8K policies use Phase 1 calibration artifacts. No held-out outcome is read.",
    ]
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase0-root", type=Path, required=True)
    parser.add_argument("--phase1-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-report-json", type=Path, required=True)
    parser.add_argument("--output-report-md", type=Path, required=True)
    parser.add_argument("--output-freeze", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for path in (args.output_report_json, args.output_report_md, args.output_freeze):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite frozen output: {path}")
    report, freeze = calibrate(
        phase0_root=args.phase0_root.resolve(),
        phase1_root=args.phase1_root.resolve(),
        split_manifest=args.split_manifest.resolve(),
        repo_root=args.repo_root.resolve(),
    )
    write_json_atomic(args.output_report_json, report)
    write_text_atomic(args.output_report_md, render_markdown(report))
    write_json_atomic(args.output_freeze, freeze)
    print(
        json.dumps(
            {
                "report": str(args.output_report_json),
                "freeze": str(args.output_freeze),
                "freeze_sha256": sha256_file(args.output_freeze),
                "runner_commit": freeze["runner_commit"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
