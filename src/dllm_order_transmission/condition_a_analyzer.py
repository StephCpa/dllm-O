"""Freeze, validate, and analyze the Condition A entropy-first intervention."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from jsonschema import Draft202012Validator

from .artifacts import sha256_file, write_json_atomic, write_text_atomic
from .condition_a_protocol import (
    ARTIFACT_SCHEMA_SHA256,
    DEFAULT_NUM_SHARDS,
    EXPECTED_FORMAL_GENERATIONS,
    FORMAL_ROLLOUTS,
    FREEZE_VERSION,
    OPTION_B_FREEZE_SHA256,
    OPTION_B_REPAIR_AMENDMENT_SHA256,
    OPTION_B_REPORT_SHA256,
    POLICY_NAME,
    PROTOCOL_VERSION,
    build_work_items,
    validate_freeze,
)
from .metrics import estimate_pass_at_k
from .phase1_heldout_analyzer import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    FrozenRuleError,
    _classify_ci,
    task_stratified_bootstrap,
)
from .phase1_protocol import (
    BCI_FREEZE_SHA256,
    MODEL_ID,
    MODEL_REVISION,
    PROTOCOL_SEED,
    SPLIT_MANIFEST_SHA256,
    load_split_manifest,
)
from .seeding import derive_rollout_seed, rollout_seed_key


TASKS = ("gsm8k", "math500")
PHASE1_RUNNER_COMMIT = "eaf4b4f9214eec6066010fad7dba5c5319d85122"
PRIMARY_K_GRID = (4, 8, 16, 32, 64)
SCREENING_K_GRID = (2, 4, 8, 16)


def _repo_commit(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _condition_result_path(root: Path, task: str, row: int, rollout: int) -> Path:
    return (
        root
        / PROTOCOL_VERSION
        / "condition-a-formal"
        / task
        / POLICY_NAME
        / f"q{row:04d}"
        / f"r{rollout:02d}.json"
    )


def _smoke_result_path(root: Path, task: str, row: int) -> Path:
    return (
        root
        / PROTOCOL_VERSION
        / "condition-a-smoke"
        / task
        / POLICY_NAME
        / f"q{row:04d}"
        / "r00.json"
    )


def _condition_trace_path(root: Path, task: str, row: int, rollout: int) -> Path:
    return _condition_result_path(root, task, row, rollout).with_suffix(
        ".trace.jsonl.gz"
    )


def _phase1_result_path(
    root: Path, task: str, row: int, policy: str, rollout: int
) -> Path:
    run_kind = "screening" if rollout < 16 else "primary-extension"
    return (
        root
        / "phase1-v1"
        / "heldout"
        / run_kind
        / task
        / policy
        / f"q{row:04d}"
        / f"r{rollout:02d}.json"
    )


def _validate_control_record(
    *,
    path: Path,
    record: dict[str, Any],
    manifest: dict[str, Any],
    task: str,
    row: int,
    policy: str,
    rollout: int,
) -> None:
    expected_run_kind = "screening" if rollout < 16 else "primary-extension"
    source = manifest["sources"][task]
    prompt_hash = next(
        entry["prompt_sha256"]
        for entry in manifest["splits"][task]["heldout"]
        if int(entry["row_index"]) == row
    )
    checks = (
        record.get("run_kind") == expected_run_kind,
        record.get("policy", {}).get("name") == policy,
        int(record.get("rollout_index", -1)) == rollout,
        record.get("split_manifest_sha256") == SPLIT_MANIFEST_SHA256,
        record.get("bci_freeze_sha256") == BCI_FREEZE_SHA256,
        record.get("model_id") == MODEL_ID,
        record.get("model_revision") == MODEL_REVISION,
        record.get("external_repository_commit")
        == "1a2fddb5c6655597e63081c0af5ebb718a849f39",
        record.get("runner_repository")
        == {"commit": PHASE1_RUNNER_COMMIT, "dirty": False},
        record.get("completion_status") == "complete",
        record.get("seed_key") == _seed_key(task, row, rollout),
        record.get("source", {}).get("split_role") == "heldout",
        int(record.get("source", {}).get("row_index", -1)) == row,
        record.get("source", {}).get("prompt_sha256") == prompt_hash,
        record.get("source", {}).get("dataset_id") == source["dataset_id"],
        record.get("source", {}).get("dataset_revision") == source["revision"],
        record.get("source", {}).get("dataset_split") == source["split"],
    )
    trace_path = path.with_name(path.stem + ".trace.jsonl.gz")
    if not all(checks) or not trace_path.is_file():
        raise FrozenRuleError(f"invalid reused control artifact: {path}")
    if sha256_file(trace_path) != record.get("trace_sha256"):
        raise FrozenRuleError(f"reused control trace hash mismatch: {path}")


def _load_schema(path: Path) -> Draft202012Validator:
    if sha256_file(path) != ARTIFACT_SCHEMA_SHA256:
        raise FrozenRuleError("Condition A artifact-schema hash mismatch")
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _validate_option_b_trigger(
    *,
    option_b_report: Path,
    option_b_freeze: Path,
    repair_amendment: Path,
) -> dict[str, Any]:
    expected_hashes = {
        option_b_report: OPTION_B_REPORT_SHA256,
        option_b_freeze: OPTION_B_FREEZE_SHA256,
        repair_amendment: OPTION_B_REPAIR_AMENDMENT_SHA256,
    }
    for path, expected in expected_hashes.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise FrozenRuleError(f"Condition A provenance mismatch: {path}")
    report = json.loads(option_b_report.read_text(encoding="utf-8"))
    required = {
        "p1_class": "entirely_below_zero",
        "p2_class": "resolved_negative",
        "run_condition_a": True,
    }
    observed = {
        "p1_class": report.get("gate", {}).get("p1_class"),
        "p2_class": report.get("gate", {}).get("p2_class"),
        "run_condition_a": report.get("gate", {}).get("run_condition_a"),
    }
    if observed != required:
        raise FrozenRuleError(
            f"Option B did not trigger Condition A: {json.dumps(observed)}"
        )
    return observed


def build_freeze_payload(
    *, analyzer_commit: str, runner_commit: str, trigger: dict[str, Any]
) -> dict[str, Any]:
    return {
        "protocol_version": FREEZE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analyzer_commit": analyzer_commit,
        "runner_commit": runner_commit,
        "split_manifest_sha256": SPLIT_MANIFEST_SHA256,
        "artifact_schema_sha256": ARTIFACT_SCHEMA_SHA256,
        "bci_freeze_sha256": BCI_FREEZE_SHA256,
        "option_b_freeze_sha256": OPTION_B_FREEZE_SHA256,
        "option_b_report_sha256": OPTION_B_REPORT_SHA256,
        "option_b_repair_amendment_sha256": OPTION_B_REPAIR_AMENDMENT_SHA256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "trigger": trigger,
        "intervention": {
            "policy": POLICY_NAME,
            "block_size": 32,
            "selection": (
                "at every denoising step, rank currently masked positions in the "
                "active block by full-vocabulary entropy descending; resolve exact "
                "ties by lower absolute token position"
            ),
            "isolation_note": (
                "EF uses full-distribution entropy whereas AO ranks sampled-candidate "
                "confidence; this is a ranking-policy intervention, not the algebraic "
                "sign reversal of one shared scalar score"
            ),
            "recompute_ranking_each_step": True,
            "controls": {
                "primary": "ao_b32, reused at 64 rollouts",
                "secondary": "random_b32, reused at 16 rollouts",
            },
            "common_random_numbers": (
                "reuse Phase 1 task/query/rollout seed keys; policy is excluded "
                "from seed derivation"
            ),
        },
        "decode": {
            "steps": 256,
            "generation_length": 256,
            "block_length": 32,
            "temperature": 0.6,
            "cfg_scale": 0.0,
            "mask_id": 126336,
        },
        "cardinality": {
            "tasks": list(TASKS),
            "heldout_queries_per_task": 100,
            "rollouts_per_query": FORMAL_ROLLOUTS,
            "new_generations": EXPECTED_FORMAL_GENERATIONS,
            "num_shards": DEFAULT_NUM_SHARDS,
            "generations_per_shard": EXPECTED_FORMAL_GENERATIONS
            // DEFAULT_NUM_SHARDS,
        },
        "primary_endpoint": {
            "contrast": "CoverageAUC64(ef_b32) - CoverageAUC64(ao_b32)",
            "coverage_auc": "mean unbiased Pass@k over k in {4,8,16,32,64}",
            "aggregation": "paired per query, then task-stratified pooled mean",
            "uncertainty": {
                "method": "task-stratified query bootstrap percentile interval",
                "seed": BOOTSTRAP_SEED,
                "replicates": BOOTSTRAP_REPLICATES,
                "level": 0.95,
            },
            "classification": {
                "ci_entirely_above_zero": (
                    "entropy-first recovers coverage relative to confidence-first"
                ),
                "ci_straddles_zero": (
                    "no resolved entropy-first benefit; confidence-guided selection "
                    "is not identified as the damaging component"
                ),
                "ci_entirely_below_zero": (
                    "entropy-first reduces coverage; confidence-guidance is protective "
                    "relative to the intervention"
                ),
            },
        },
        "secondary_endpoints": {
            "matched_16_rollout_contrasts": {
                "metric": "ScreeningCoverageAUC16, mean Pass@k over k in {2,4,8,16}",
                "contrasts": [
                    "ef_b32 - random_b32",
                    "ef_b32 - ao_b32",
                ],
                "reason": (
                    "random_b32 has only 16 frozen rollouts; no unmatched 64-rollout "
                    "comparison is permitted"
                ),
            },
            "report_all": [
                "per-task primary contrasts",
                "per-k pass rates",
                "solved-set membership",
                "parser status and empty-response rates",
            ],
        },
        "length_guard": {
            "metric": "whitespace-separated token count of the decoded response",
            "comparisons": [
                "ef_b32(64) - ao_b32(64)",
                "ef_b32(first 16) - random_b32(16)",
            ],
            "trigger": "paired pooled 95% bootstrap CI excludes zero",
            "action": (
                "report the difference and qualify coverage interpretation; never "
                "exclude or length-match outcomes post hoc"
            ),
            "fixed_length_note": (
                "the decoder always commits 256 generated token positions and has no "
                "early-stopping rule, so raw generation-cap exposure is identical by design"
            ),
        },
        "artifact_rules": {
            "full_distribution_trace": True,
            "top_m": 64,
            "full_logsumexp": True,
            "missing_or_invalid_records": "block analysis",
            "parser_failures_and_empty_responses": "retain as incorrect",
            "partial_shards": "resume only with the identical work key and seed",
        },
        "scope": {
            "primary_claim": (
                "causal effect of inverting the within-block position-ranking policy "
                "at B=32 in this model and these two tasks"
            ),
            "not_permitted": [
                "universal superiority of entropy-first decoding",
                "a claim that every increment in block size has equal effect",
                "a 64-rollout comparison to random_b32 without new matched data",
            ],
        },
    }


def emit_freeze(
    *,
    output: Path,
    repo_root: Path,
    option_b_report: Path,
    option_b_freeze: Path,
    repair_amendment: Path,
) -> str:
    if output.exists():
        raise FrozenRuleError(f"refusing to overwrite Condition A freeze: {output}")
    trigger = _validate_option_b_trigger(
        option_b_report=option_b_report,
        option_b_freeze=option_b_freeze,
        repair_amendment=repair_amendment,
    )
    commit = _repo_commit(repo_root)
    payload = build_freeze_payload(
        analyzer_commit=commit, runner_commit=commit, trigger=trigger
    )
    write_json_atomic(output, payload)
    return sha256_file(output)


def _seed_key(task: str, row: int, rollout: int) -> str:
    return rollout_seed_key(
        row,
        rollout,
        global_seed=PROTOCOL_SEED,
        protocol_prefix=f"dllm-order-phase1-v1|{task}",
    )


def _validate_artifacts(
    *,
    condition_root: Path,
    manifest: dict[str, Any],
    freeze_sha: str,
    runner_commit: str,
    validator: Draft202012Validator,
) -> dict[str, Any]:
    expected = present = invalid = 0
    for item in build_work_items(manifest, run_kind="condition-a-formal"):
        expected += 1
        result_path = _condition_result_path(
            condition_root, item.task, item.row_index, item.rollout_index
        )
        trace_path = _condition_trace_path(
            condition_root, item.task, item.row_index, item.rollout_index
        )
        if not result_path.is_file() or not trace_path.is_file():
            continue
        try:
            record = json.loads(result_path.read_text(encoding="utf-8"))
            validator.validate(record)
            checks = {
                "condition_a_freeze_sha256": freeze_sha,
                "split_manifest_sha256": SPLIT_MANIFEST_SHA256,
                "run_kind": "condition-a-formal",
                "rollout_index": item.rollout_index,
                "seed_key": _seed_key(item.task, item.row_index, item.rollout_index),
                "completion_status": "complete",
            }
            if any(record.get(key) != value for key, value in checks.items()):
                raise FrozenRuleError("record provenance mismatch")
            if record["runner_repository"] != {
                "commit": runner_commit,
                "dirty": False,
            }:
                raise FrozenRuleError("runner repository mismatch")
            if sha256_file(trace_path) != record["trace_sha256"]:
                raise FrozenRuleError("trace hash mismatch")
        except Exception:
            invalid += 1
            continue
        present += 1
    return {
        "expected": expected,
        "present_and_valid": present,
        "missing": expected - present - invalid,
        "invalid": invalid,
        "outcome_fields_read": False,
    }


def _validate_markers(condition_root: Path) -> dict[str, int]:
    total = 0
    for shard in range(DEFAULT_NUM_SHARDS):
        path = (
            condition_root
            / PROTOCOL_VERSION
            / "manifests"
            / "condition-a-formal"
            / "all"
            / f"shard-{shard:03d}-of-{DEFAULT_NUM_SHARDS:03d}.json"
        )
        if not path.is_file():
            raise FrozenRuleError(f"missing Condition A marker: {path}")
        marker = json.loads(path.read_text(encoding="utf-8"))
        if marker.get("expected_keys_sha256") != marker.get("observed_keys_sha256"):
            raise FrozenRuleError(f"marker key mismatch: {path}")
        expected_count = EXPECTED_FORMAL_GENERATIONS // DEFAULT_NUM_SHARDS
        if int(marker.get("result_count", -1)) != expected_count:
            raise FrozenRuleError(f"marker count mismatch: {path}")
        total += int(marker["result_count"])
    return {"markers": DEFAULT_NUM_SHARDS, "total_result_count": total}


def audit_smoke(
    *,
    condition_root: Path,
    manifest: dict[str, Any],
    freeze_sha: str,
    runner_commit: str,
    validator: Draft202012Validator,
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    errors: list[str] = []
    for task in TASKS:
        entry = manifest["splits"][task]["calibration"][0]
        row = int(entry["row_index"])
        result_path = _smoke_result_path(condition_root, task, row)
        trace_path = result_path.with_name(result_path.stem + ".trace.jsonl.gz")
        cell_errors: list[str] = []
        if not result_path.is_file() or not trace_path.is_file():
            cell_errors.append("missing result or trace")
            records: list[dict[str, Any]] = []
        else:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            try:
                validator.validate(result)
            except Exception as error:
                cell_errors.append(f"schema: {error}")
            expected = {
                "condition_a_freeze_sha256": freeze_sha,
                "split_manifest_sha256": SPLIT_MANIFEST_SHA256,
                "run_kind": "condition-a-smoke",
                "rollout_index": 0,
                "seed_key": _seed_key(task, row, 0),
                "completion_status": "complete",
            }
            for key, value in expected.items():
                if result.get(key) != value:
                    cell_errors.append(f"mismatched {key}")
            if result.get("runner_repository") != {
                "commit": runner_commit,
                "dirty": False,
            }:
                cell_errors.append("runner repository mismatch")
            source = result.get("source", {})
            if (
                source.get("prompt_sha256") != entry["prompt_sha256"]
                or int(source.get("row_index", -1)) != row
                or source.get("split_role") != "calibration"
            ):
                cell_errors.append("source provenance mismatch")
            if sha256_file(trace_path) != result.get("trace_sha256"):
                cell_errors.append("trace hash mismatch")
            if not result.get("environment_fingerprint", {}).get(
                "trace_equivalence_verified_for_this_result"
            ):
                cell_errors.append("traced/untraced equivalence not verified")
            with gzip.open(trace_path, mode="rt", encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle]

        if len(records) != 256:
            cell_errors.append(f"expected 256 trace events, found {len(records)}")
        if [record.get("global_step") for record in records] != list(range(256)):
            cell_errors.append("global steps are not contiguous")
        committed: set[int] = set()
        for index, record in enumerate(records):
            eligible = [int(value) for value in record.get("eligible_positions", [])]
            entropy = [float(value) for value in record.get("entropy", [])]
            selected = [int(value) for value in record.get("selected_positions", [])]
            if len(eligible) != len(entropy):
                cell_errors.append(f"step {index}: entropy alignment mismatch")
                continue
            if not entropy or any(not math.isfinite(value) for value in entropy):
                cell_errors.append(f"step {index}: invalid entropy")
                continue
            aligned_numeric_fields = (
                "top1_logits",
                "top2_logits",
                "logit_margins",
                "probability_margins",
                "candidate_probabilities",
                "eligible_entropy",
                "eligible_logit_margins",
            )
            if any(
                not isinstance(record.get(field), list)
                or len(record[field]) != len(eligible)
                or any(not math.isfinite(float(value)) for value in record[field])
                for field in aligned_numeric_fields
            ):
                cell_errors.append(f"step {index}: aligned numeric fields invalid")
            count = len(selected)
            expected_selected = [
                position
                for position, _ in sorted(
                    zip(eligible, entropy, strict=True),
                    key=lambda pair: (-pair[1], pair[0]),
                )[:count]
            ]
            if selected != expected_selected:
                cell_errors.append(f"step {index}: entropy ordering mismatch")
            block_start = int(record.get("active_block_start", -1))
            block_end = int(record.get("active_block_end", -1))
            if any(
                position not in set(eligible)
                or not block_start <= position < block_end
                for position in selected
            ):
                cell_errors.append(f"step {index}: selected ineligible position")
            if committed.intersection(selected):
                cell_errors.append(f"step {index}: position committed twice")
            committed.update(selected)
            top_ids = record.get("top64_token_ids")
            top_logits = record.get("top64_logits")
            normalizer = record.get("full_logsumexp")
            if not (
                isinstance(top_ids, list)
                and isinstance(top_logits, list)
                and isinstance(normalizer, list)
                and len(top_ids) == len(eligible)
                and len(top_logits) == len(eligible)
                and len(normalizer) == len(eligible)
                and all(len(values) == 64 for values in top_ids)
                and all(len(values) == 64 for values in top_logits)
                and all(
                    math.isfinite(float(value))
                    for values in top_logits
                    for value in values
                )
                and all(math.isfinite(float(value)) for value in normalizer)
            ):
                cell_errors.append(f"step {index}: full-distribution fields invalid")
            if index + 1 < len(records):
                if record.get("state_checksum_after") != records[index + 1].get(
                    "state_checksum_before"
                ):
                    cell_errors.append(f"step {index}: checksum chain mismatch")
        if len(committed) != 256:
            cell_errors.append(f"expected 256 committed positions, found {len(committed)}")
        errors.extend(f"{task}: {error}" for error in cell_errors)
        cells.append(
            {
                "task": task,
                "row_index": row,
                "trace_events": len(records),
                "committed_positions": len(committed),
                "errors": cell_errors,
            }
        )

    marker_path = (
        condition_root
        / PROTOCOL_VERSION
        / "manifests"
        / "condition-a-smoke"
        / "all"
        / "shard-000-of-001.json"
    )
    if not marker_path.is_file():
        errors.append("missing smoke shard marker")
    else:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            marker.get("result_count") != 2
            or marker.get("expected_keys_sha256")
            != marker.get("observed_keys_sha256")
        ):
            errors.append("invalid smoke shard marker")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "condition_a_freeze_sha256": freeze_sha,
        "cells": cells,
        "errors": errors,
        "passed": not errors,
        "outcome_fields_read": False,
    }


def _coverage(successes: int, n: int, grid: tuple[int, ...]) -> float:
    return mean(estimate_pass_at_k(n, successes, k) for k in grid)


def _bootstrap(values: dict[str, list[float]], label: str) -> dict[str, float]:
    return task_stratified_bootstrap(
        label=label,
        tasks=values,
        statistic=lambda samples: mean(
            value for task_values in samples.values() for value in task_values
        ),
    )


def _classify(interval: dict[str, float]) -> str:
    return _classify_ci(interval["ci_low"], interval["ci_high"])


def analyze(
    *,
    condition_root: Path,
    phase1_root: Path,
    manifest: dict[str, Any],
    freeze_payload: dict[str, Any],
    freeze_sha: str,
    validator: Draft202012Validator,
) -> dict[str, Any]:
    validation = _validate_artifacts(
        condition_root=condition_root,
        manifest=manifest,
        freeze_sha=freeze_sha,
        runner_commit=freeze_payload["runner_commit"],
        validator=validator,
    )
    if validation["missing"] or validation["invalid"]:
        raise FrozenRuleError(f"incomplete Condition A artifacts: {validation}")
    markers = _validate_markers(condition_root)

    records: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for task in TASKS:
        rows = sorted(
            int(entry["row_index"])
            for entry in manifest["splits"][task]["heldout"]
        )
        for row in rows:
            ef = [
                json.loads(
                    _condition_result_path(condition_root, task, row, rollout).read_text(
                        encoding="utf-8"
                    )
                )
                for rollout in range(64)
            ]
            ao = [
                json.loads(
                    _phase1_result_path(
                        phase1_root, task, row, "ao_b32", rollout
                    ).read_text(encoding="utf-8")
                )
                for rollout in range(64)
            ]
            random = [
                json.loads(
                    _phase1_result_path(
                        phase1_root, task, row, "random_b32", rollout
                    ).read_text(encoding="utf-8")
                )
                for rollout in range(16)
            ]
            for policy, policy_records in (("ao_b32", ao), ("random_b32", random)):
                for rollout, record in enumerate(policy_records):
                    path = _phase1_result_path(
                        phase1_root, task, row, policy, rollout
                    )
                    _validate_control_record(
                        path=path,
                        record=record,
                        manifest=manifest,
                        task=task,
                        row=row,
                        policy=policy,
                        rollout=rollout,
                    )
            records[(task, row, "ef_b32")] = ef
            records[(task, row, "ao_b32")] = ao
            records[(task, row, "random_b32")] = random

    primary: dict[str, list[float]] = {task: [] for task in TASKS}
    secondary_random: dict[str, list[float]] = {task: [] for task in TASKS}
    secondary_ao: dict[str, list[float]] = {task: [] for task in TASKS}
    length_ao: dict[str, list[float]] = {task: [] for task in TASKS}
    length_random: dict[str, list[float]] = {task: [] for task in TASKS}
    summaries: dict[str, dict[str, dict[str, float]]] = {}
    quality: dict[str, dict[str, Any]] = {}
    solved_membership: dict[str, dict[str, dict[str, int]]] = {}

    for task in TASKS:
        rows = sorted(
            int(entry["row_index"])
            for entry in manifest["splits"][task]["heldout"]
        )
        summaries[task] = {}
        quality[task] = {}
        solved_membership[task] = {
            "ef64_vs_ao64": {
                "ef_only": 0,
                "ao_only": 0,
                "both": 0,
                "neither": 0,
            },
            "ef16_vs_random16": {
                "ef_only": 0,
                "random_only": 0,
                "both": 0,
                "neither": 0,
            },
        }
        for row in rows:
            ef = records[(task, row, "ef_b32")]
            ao = records[(task, row, "ao_b32")]
            random = records[(task, row, "random_b32")]
            counts = {
                "ef_b32": sum(bool(record["correct"]) for record in ef),
                "ao_b32": sum(bool(record["correct"]) for record in ao),
                "ef16": sum(bool(record["correct"]) for record in ef[:16]),
                "ao16": sum(bool(record["correct"]) for record in ao[:16]),
                "random16": sum(bool(record["correct"]) for record in random),
            }
            primary[task].append(
                _coverage(counts["ef_b32"], 64, PRIMARY_K_GRID)
                - _coverage(counts["ao_b32"], 64, PRIMARY_K_GRID)
            )
            secondary_random[task].append(
                _coverage(counts["ef16"], 16, SCREENING_K_GRID)
                - _coverage(counts["random16"], 16, SCREENING_K_GRID)
            )
            secondary_ao[task].append(
                _coverage(counts["ef16"], 16, SCREENING_K_GRID)
                - _coverage(counts["ao16"], 16, SCREENING_K_GRID)
            )
            length_ao[task].append(
                mean(len(str(record["response"]).split()) for record in ef)
                - mean(len(str(record["response"]).split()) for record in ao)
            )
            length_random[task].append(
                mean(len(str(record["response"]).split()) for record in ef[:16])
                - mean(len(str(record["response"]).split()) for record in random)
            )
            ef64_solved = counts["ef_b32"] > 0
            ao64_solved = counts["ao_b32"] > 0
            if ef64_solved and ao64_solved:
                solved_membership[task]["ef64_vs_ao64"]["both"] += 1
            elif ef64_solved:
                solved_membership[task]["ef64_vs_ao64"]["ef_only"] += 1
            elif ao64_solved:
                solved_membership[task]["ef64_vs_ao64"]["ao_only"] += 1
            else:
                solved_membership[task]["ef64_vs_ao64"]["neither"] += 1

            ef16_solved = counts["ef16"] > 0
            random16_solved = counts["random16"] > 0
            if ef16_solved and random16_solved:
                solved_membership[task]["ef16_vs_random16"]["both"] += 1
            elif ef16_solved:
                solved_membership[task]["ef16_vs_random16"]["ef_only"] += 1
            elif random16_solved:
                solved_membership[task]["ef16_vs_random16"]["random_only"] += 1
            else:
                solved_membership[task]["ef16_vs_random16"]["neither"] += 1

        for policy, n in (("ef_b32", 64), ("ao_b32", 64), ("random_b32", 16)):
            success_counts = [
                sum(
                    bool(record["correct"])
                    for record in records[(task, row, policy)]
                )
                for row in rows
            ]
            grid = PRIMARY_K_GRID if n == 64 else SCREENING_K_GRID
            summaries[task][policy] = {
                "rollouts": n,
                "pass_at_1": mean(count / n for count in success_counts),
                **{
                    f"pass_at_{k}": mean(
                        estimate_pass_at_k(n, count, k) for count in success_counts
                    )
                    for k in grid
                },
                "coverage_auc": mean(
                    _coverage(count, n, grid) for count in success_counts
                ),
            }
            policy_records = [
                record
                for row in rows
                for record in records[(task, row, policy)]
            ]
            quality[task][policy] = {
                "empty_response_rate": mean(
                    not str(record["response"]).strip() for record in policy_records
                ),
                "parser_status_counts": dict(
                    sorted(Counter(record["parser_status"] for record in policy_records).items())
                ),
                "mean_whitespace_tokens": mean(
                    len(str(record["response"]).split()) for record in policy_records
                ),
            }

    p = _bootstrap(primary, "condition-a|primary|ef-minus-ao")
    sr = _bootstrap(secondary_random, "condition-a|secondary|ef-minus-random16")
    sa = _bootstrap(secondary_ao, "condition-a|secondary|ef-minus-ao16")
    la = _bootstrap(length_ao, "condition-a|length|ef-minus-ao")
    lr = _bootstrap(length_random, "condition-a|length|ef-minus-random16")
    primary_class = _classify(p)
    random_class = _classify(sr)
    if primary_class == "entirely_above_zero":
        if random_class == "entirely_above_zero":
            reading = "entropy-first improves over both confidence-first and random"
        elif random_class == "entirely_below_zero":
            reading = (
                "entropy-first improves over confidence-first but remains below random"
            )
        else:
            reading = (
                "entropy-first improves over confidence-first and is unresolved versus random"
            )
    elif primary_class == "entirely_below_zero":
        reading = "entropy-first harms coverage relative to confidence-first"
    else:
        reading = (
            "no resolved entropy-first benefit; confidence-guided position choice is "
            "not identified as the source of the coverage cost"
        )

    primary_per_task = {
        task: _bootstrap({task: primary[task]}, f"condition-a|primary|{task}")
        for task in TASKS
    }
    secondary_random_per_task = {
        task: _bootstrap(
            {task: secondary_random[task]},
            f"condition-a|secondary-random|{task}",
        )
        for task in TASKS
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "condition_a_freeze_sha256": freeze_sha,
        "validation": validation,
        "markers": markers,
        "primary": {
            "contrast": "CoverageAUC64(ef_b32) - CoverageAUC64(ao_b32)",
            "pooled": p,
            "classification": primary_class,
            "per_task": primary_per_task,
        },
        "secondary": {
            "ef_minus_random_screening16": sr,
            "ef_minus_ao_screening16": sa,
            "ef_minus_random_screening16_per_task": secondary_random_per_task,
        },
        "length_guard": {
            "ef_minus_ao64": la,
            "ef_minus_random16": lr,
            "triggered": (
                _classify(la) != "straddles_zero"
                or _classify(lr) != "straddles_zero"
            ),
            "fixed_generation_positions": 256,
        },
        "summaries": summaries,
        "solved_set_membership": solved_membership,
        "quality": quality,
        "pre_registered_reading": reading,
    }


def render_markdown(report: dict[str, Any]) -> str:
    p = report["primary"]["pooled"]
    sr = report["secondary"]["ef_minus_random_screening16"]
    lines = [
        "# Condition A Results",
        "",
        f"- Freeze SHA-256: `{report['condition_a_freeze_sha256']}`",
        f"- Valid artifacts: {report['validation']['present_and_valid']}/"
        f"{report['validation']['expected']}",
        "",
        "## Primary",
        "",
        f"- EF-B32 minus AO-B32 CoverageAUC64: {p['point']:.5f} "
        f"(95% CI [{p['ci_low']:.5f}, {p['ci_high']:.5f}])",
        f"- Classification: `{report['primary']['classification']}`",
        f"- Pre-registered reading: {report['pre_registered_reading']}",
        "",
        "## Matched 16-rollout random reference",
        "",
        f"- EF-B32 minus random-B32 ScreeningCoverageAUC16: {sr['point']:.5f} "
        f"(95% CI [{sr['ci_low']:.5f}, {sr['ci_high']:.5f}])",
        "",
        "## Length guard",
        "",
        f"- Triggered: `{report['length_guard']['triggered']}`",
        "- All policies commit exactly 256 generated positions by design; the guard "
        "tests decoded-response whitespace length and does not remove outcomes.",
        "",
    ]
    return "\n".join(lines)


def _append_invocation(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=["emit-freeze", "audit-smoke", "validate-only", "analyze"],
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--artifact-schema", type=Path, required=True)
    parser.add_argument("--condition-root", type=Path)
    parser.add_argument("--phase1-root", type=Path)
    parser.add_argument("--condition-a-freeze", type=Path, required=True)
    parser.add_argument("--option-b-report", type=Path)
    parser.add_argument("--option-b-freeze", type=Path)
    parser.add_argument("--repair-amendment", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--invocation-log", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    invocation = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "analyzer_commit": _repo_commit(args.repo_root),
        "reads_condition_a_correctness": args.mode == "analyze",
    }
    if args.invocation_log is not None:
        _append_invocation(args.invocation_log, invocation)

    # These protocol inputs are checked in every mode, including emit-freeze,
    # before any output or Condition A outcome is read.
    load_split_manifest(args.split_manifest)
    _load_schema(args.artifact_schema)

    if args.mode == "emit-freeze":
        for path in (args.option_b_report, args.option_b_freeze, args.repair_amendment):
            if path is None:
                raise FrozenRuleError("emit-freeze requires all Option B provenance")
        freeze_sha = emit_freeze(
            output=args.condition_a_freeze,
            repo_root=args.repo_root,
            option_b_report=args.option_b_report,
            option_b_freeze=args.option_b_freeze,
            repair_amendment=args.repair_amendment,
        )
        print(f"condition_a_freeze_sha256={freeze_sha}")
        return 0

    if args.condition_root is None:
        raise FrozenRuleError("validation and analysis require --condition-root")
    manifest = load_split_manifest(args.split_manifest)
    validator = _load_schema(args.artifact_schema)
    commit = _repo_commit(args.repo_root)
    freeze_sha = validate_freeze(args.condition_a_freeze)
    freeze_payload = json.loads(args.condition_a_freeze.read_text(encoding="utf-8"))
    if freeze_payload["analyzer_commit"] != commit:
        raise FrozenRuleError("Condition A freeze is bound to another analyzer commit")
    if args.mode == "audit-smoke":
        smoke = audit_smoke(
            condition_root=args.condition_root,
            manifest=manifest,
            freeze_sha=freeze_sha,
            runner_commit=freeze_payload["runner_commit"],
            validator=validator,
        )
        if args.output_json is not None:
            write_json_atomic(args.output_json, smoke)
        print(json.dumps(smoke, indent=2, sort_keys=True))
        return int(not smoke["passed"])
    validation = _validate_artifacts(
        condition_root=args.condition_root,
        manifest=manifest,
        freeze_sha=freeze_sha,
        runner_commit=freeze_payload["runner_commit"],
        validator=validator,
    )
    if args.mode == "validate-only":
        if validation["missing"] == 0 and validation["invalid"] == 0:
            validation["markers"] = _validate_markers(args.condition_root)
        if args.output_json is not None:
            write_json_atomic(args.output_json, validation)
        print(json.dumps(validation, indent=2, sort_keys=True))
        return int(validation["missing"] > 0 or validation["invalid"] > 0)

    if args.phase1_root is None:
        raise FrozenRuleError("analyze requires --phase1-root")
    report = analyze(
        condition_root=args.condition_root,
        phase1_root=args.phase1_root,
        manifest=manifest,
        freeze_payload=freeze_payload,
        freeze_sha=freeze_sha,
        validator=validator,
    )
    if args.output_json is None or args.output_markdown is None:
        raise FrozenRuleError("analyze requires JSON and Markdown outputs")
    write_json_atomic(args.output_json, report)
    write_text_atomic(args.output_markdown, render_markdown(report))
    print(json.dumps(report["primary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
