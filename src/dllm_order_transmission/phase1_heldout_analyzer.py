"""Frozen Phase 1 held-out analysis: artifact validation, RQ1/RQ2 endpoints, Gate 1.

Three mutually exclusive modes:

- ``emit-freeze`` writes the pre-registration ``heldout_analysis_freeze.json``
  before any held-out outcome is read.
- ``validate-only`` checks provenance, schema, hashes, and completeness without
  touching correctness or parser outcome fields.
- ``analyze`` runs the full frozen analysis and writes the Gate 1 report.

The analyzer consumes the final BCI freeze
``8f2b6df3f98abdb0b9060b704b02fcc428090d01d2d279514c7c108e2fb6ef17`` and never
refits anything on held-out data.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable

import numpy as np
from jsonschema import Draft202012Validator
from scipy.stats import rankdata

from .artifacts import sha256_file, write_json_atomic, write_text_atomic
from .metrics import aggregate_pass_at_k, estimate_pass_at_k
from .phase1_calibration import (
    EPSILON,
    FORMAL_HELDOUT_K_GRID,
    MASK_TOKEN_ID,
    _annotate_candidates,
    _match_candidates,
    _quantile,
    _read_trace,
    _selected_token_rows,
)
from .phase1_protocol import (
    JUSTGRPO_COMMIT,
    MODEL_ID,
    MODEL_REVISION,
    PROTOCOL_VERSION,
    SPLIT_MANIFEST_SHA256,
    load_split_manifest,
)
from .phase1_planning import (
    minimum_detectable_correlation,
    minimum_detectable_paired_effect,
)


ANALYSIS_VERSION = "phase1-heldout-analysis-v1"
BOOTSTRAP_SEED = 20260820
BOOTSTRAP_REPLICATES = 10_000
EXPECTED_FREEZE_SHA256 = (
    "8f2b6df3f98abdb0b9060b704b02fcc428090d01d2d279514c7c108e2fb6ef17"
)
EXPECTED_RUNNER_COMMIT = "eaf4b4f9214eec6066010fad7dba5c5319d85122"
EXPECTED_FREEZE_VERSION = "phase1-bci-freeze-v1"
TASKS = ("gsm8k", "math500")
SCREENING_POLICIES = ("ar_b1", "ao_b8", "ao_b16", "ao_b32", "random_b32")
PRIMARY_POLICIES = ("ar_b1", "ao_b32")
SCREENING_ROLLOUT_COUNT = 16
TOTAL_ROLLOUT_COUNT = 64
# Frozen windows: BCI, closure matching, and predictor features use only the
# sixteen held-out screening rollouts r00-r15 of ao_b32, mirroring the
# calibration structure. The formal CoverageAUC target uses all 64 rollouts.
DIAGNOSTIC_ROLLOUTS = tuple(range(SCREENING_ROLLOUT_COUNT))
SCREENING_REPORT_K_GRID = (2, 4, 8, 16)
MALFORMED_SENSITIVITY_THRESHOLD = 0.01
MINIMUM_MATCHED_QUERY_FRACTION = 0.90
# Pre-unblinding amendment audit trail: the analysis freeze deployed on the
# server (bound to analyzer commit a48f020) is superseded by the freeze emitted
# from this analyzer version. Both files are preserved for audit.
SUPERSEDED_ANALYSIS_FREEZE_SHA256 = (
    "3bbad351b42a7c540f23efdfea1ab1294e6e0acc18395ab30558ab19f408fac7"
)
SUPERSEDED_ANALYZER_COMMIT = "a48f020d51edd68f4574732eb420016fc1434823"
SECOND_SUPERSEDED_ANALYSIS_FREEZE_SHA256 = (
    "5314d95336cc572f175a30aaef995169bb3d9149c86227ffb3d74090ca3be15c"
)
SECOND_SUPERSEDED_ANALYZER_COMMIT = "ea344f25d216334897e8378727b9cab2126b5d05"
THIRD_SUPERSEDED_ANALYSIS_FREEZE_SHA256 = (
    "cd286131415092ae3f41b0ad6eebf9ebbdf1775e77dcbc508e98bbe183566615"
)
THIRD_SUPERSEDED_ANALYZER_COMMIT = "8772804a00d439b76e998c8f6f11bfb0c88d891a"
FOURTH_SUPERSEDED_ANALYSIS_FREEZE_SHA256 = (
    "68fc9b3b859da4b05f44828527542934096534d6b43fe3a9204d3e420a3b37ca"
)
FOURTH_SUPERSEDED_ANALYZER_COMMIT = "ba16b5cbda119c7a6f12c26202a2b48de7ec7837"
AUDIT_VERSION = "phase1-screening-malformed-audit-v1"
RESOLUTION_SIMULATION_SEED = 20260820
RESOLUTION_SIMULATION_REPLICATES = 10_000
RESOLUTION_SIMULATION_RHO = 0.3
REQUIRED_ANALYSIS_FREEZE_SECTIONS = (
    "verdict_taxonomy",
    "power_resolution",
    "design_cardinality",
    "freeze_amendments",
    "secondary_contrast_readings",
    "negative_rq2_mechanistic_readings",
    "unblinding_read_order",
)


class FrozenRuleError(RuntimeError):
    """A frozen precondition failed; the analysis must stop, not adapt."""


# ---------------------------------------------------------------------------
# Frozen statistical machinery
# ---------------------------------------------------------------------------


def _pearson(left: list[float], right: list[float]) -> float:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if x.size < 2:
        raise FrozenRuleError("cannot correlate fewer than two observations")
    if float(x.std()) == 0.0 or float(y.std()) == 0.0:
        raise FrozenRuleError(
            "zero variance after within-task rank centering; the frozen "
            "stratified statistic is undefined"
        )
    return float(np.corrcoef(x, y)[0, 1])


def stratified_spearman(
    assignments: dict[str, list[tuple[float, float]]]
) -> float:
    """Task-stratified Spearman: average ranks, within-task centering, Pearson.

    ``assignments`` maps each task to its ``(BCI, advantage)`` pairs. Ties are
    handled by average ranks and are never broken arbitrarily.
    """
    centered_left: list[float] = []
    centered_right: list[float] = []
    for task in sorted(assignments):
        pairs = assignments[task]
        if not pairs:
            raise FrozenRuleError(f"task {task} has no observations")
        rank_left = rankdata([pair[0] for pair in pairs], method="average")
        rank_right = rankdata([pair[1] for pair in pairs], method="average")
        centered_left.extend(rank_left - rank_left.mean())
        centered_right.extend(rank_right - rank_right.mean())
    return _pearson(centered_left, centered_right)


def _bootstrap_seed(label: str) -> random.Random:
    return random.Random(f"{BOOTSTRAP_SEED}|{label}")


def task_stratified_bootstrap(
    *,
    label: str,
    tasks: dict[str, list[Any]],
    statistic: Callable[[dict[str, list[Any]]], float],
) -> dict[str, float]:
    """Percentile bootstrap resampling queries independently within each task.

    Degenerate replicates raise instead of being silently dropped.
    """
    if not tasks or any(not rows for rows in tasks.values()):
        raise FrozenRuleError(f"bootstrap {label}: empty stratum")
    rng = _bootstrap_seed(label)
    statistics: list[float] = []
    for replicate in range(BOOTSTRAP_REPLICATES):
        resample = {
            task: [rows[rng.randrange(len(rows))] for _ in rows]
            for task, rows in tasks.items()
        }
        try:
            statistics.append(statistic(resample))
        except FrozenRuleError as error:
            raise FrozenRuleError(
                f"bootstrap {label}: statistic undefined at replicate "
                f"{replicate}: {error}"
            ) from error
    ordered = sorted(statistics)
    return {
        "point": statistic(tasks),
        "ci_low": _quantile(ordered, 0.025),
        "ci_high": _quantile(ordered, 0.975),
    }


# ---------------------------------------------------------------------------
# Inputs, provenance, and artifact validation
# ---------------------------------------------------------------------------


def _repo_commit(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_freeze(path: Path, expected_sha256: str) -> dict[str, Any]:
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise FrozenRuleError(
            f"BCI freeze hash mismatch: expected {expected_sha256}, found {observed}"
        )
    freeze = json.loads(path.read_text(encoding="utf-8"))
    if freeze.get("protocol_version") != EXPECTED_FREEZE_VERSION:
        raise FrozenRuleError("unexpected BCI freeze protocol version")
    if freeze.get("runner_commit") != EXPECTED_RUNNER_COMMIT:
        raise FrozenRuleError(
            "BCI freeze is bound to a different runner commit: "
            f"{freeze.get('runner_commit')}"
        )
    if freeze.get("split_manifest_sha256") != SPLIT_MANIFEST_SHA256:
        raise FrozenRuleError("BCI freeze references a different split manifest")
    if freeze["target_policy"]["formal_heldout"]["k_grid"] != list(
        FORMAL_HELDOUT_K_GRID
    ):
        raise FrozenRuleError("formal k grid disagrees with the BCI freeze")
    return freeze


def _expected_items(manifest: dict[str, Any]) -> list[tuple[str, int, str, str, int]]:
    """(task, row_index, policy, run_kind, rollout) for the whole held-out run."""
    items: list[tuple[str, int, str, str, int]] = []
    for task in TASKS:
        rows = sorted(
            entry["row_index"]
            for entry in manifest["splits"][task]["heldout"]
        )
        for row in rows:
            for policy in SCREENING_POLICIES:
                for rollout in range(SCREENING_ROLLOUT_COUNT):
                    items.append((task, row, policy, "screening", rollout))
            for policy in PRIMARY_POLICIES:
                for rollout in range(
                    SCREENING_ROLLOUT_COUNT, TOTAL_ROLLOUT_COUNT
                ):
                    items.append((task, row, policy, "primary-extension", rollout))
    return items


def _design_cardinality(manifest: dict[str, Any]) -> dict[str, Any]:
    """Expected design counts derived from the split manifest.

    Resolves the generation arithmetic explicitly: the five-policy screening
    stage contributes 16,000 generations and the two-policy primary extension
    contributes 19,200, for 35,200 held-out generations. Pass@32/Pass@64/
    CoverageAUC use all 64 rollouts for ``ar_b1`` and ``ao_b32`` only; the
    other three schedules remain 16-rollout secondary conditions.
    """
    queries_per_task = {
        task: len(manifest["splits"][task]["heldout"]) for task in TASKS
    }
    screening_generations = sum(
        queries_per_task[task] * len(SCREENING_POLICIES) * SCREENING_ROLLOUT_COUNT
        for task in TASKS
    )
    extension_generations = sum(
        queries_per_task[task]
        * len(PRIMARY_POLICIES)
        * (TOTAL_ROLLOUT_COUNT - SCREENING_ROLLOUT_COUNT)
        for task in TASKS
    )
    return {
        "heldout_queries_per_task": queries_per_task,
        "heldout_query_total": sum(queries_per_task.values()),
        "screening_generations": screening_generations,
        "primary_extension_generations": extension_generations,
        "heldout_generations_total": screening_generations + extension_generations,
        "new_calibration_generations": 6_400,
        "phase0_reused_generations": 1_600,
        "protocol_new_generation_total": 41_600,
        "formal_pass_at_64_scope": (
            "Pass@32/Pass@64/CoverageAUC use all 64 rollouts for ar_b1 and "
            "ao_b32 only; ao_b8, ao_b16, and random_b32 remain 16-rollout "
            "secondary conditions with k in {2,4,8,16}"
        ),
    }


def _result_path(phase1_root: Path, item: tuple[str, int, str, str, int]) -> Path:
    task, row, policy, run_kind, rollout = item
    return (
        phase1_root
        / PROTOCOL_VERSION
        / "heldout"
        / run_kind
        / task
        / policy
        / f"q{row:04d}"
        / f"r{rollout:02d}.json"
    )


class HeldoutRecordValidator:
    """Schema plus frozen provenance checks that never read outcome fields."""

    def __init__(
        self,
        *,
        manifest: dict[str, Any],
        freeze_sha256: str,
        schema_path: Path,
    ) -> None:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        # Validate against the full document so $defs references resolve; a
        # generation result matches exactly the generation_result oneOf branch.
        self.validator = Draft202012Validator(schema)
        self.prompt_hashes = {
            (task, int(entry["row_index"])): entry["prompt_sha256"]
            for task in TASKS
            for entry in manifest["splits"][task]["heldout"]
        }
        self.dataset = {
            task: (
                manifest["sources"][task]["dataset_id"],
                manifest["sources"][task]["revision"],
                manifest["sources"][task]["split"],
            )
            for task in TASKS
        }
        self.freeze_sha256 = freeze_sha256

    def validate(
        self,
        record: dict[str, Any],
        *,
        item: tuple[str, int, str, str, int],
    ) -> None:
        task, row, policy, run_kind, rollout = item
        self.validator.validate(record)
        if record["completion_status"] != "complete":
            raise FrozenRuleError(f"incomplete record for {item}")
        checks = (
            record["run_kind"] == run_kind,
            record["policy"]["name"] == policy,
            int(record["rollout_index"]) == rollout,
            record["split_manifest_sha256"] == SPLIT_MANIFEST_SHA256,
            record["model_id"] == MODEL_ID,
            record["model_revision"] == MODEL_REVISION,
            record["external_repository_commit"] == JUSTGRPO_COMMIT,
            record["bci_freeze_sha256"] == self.freeze_sha256,
            record["source"]["split_role"] == "heldout",
            int(record["source"]["row_index"]) == row,
            record["source"]["prompt_sha256"]
            == self.prompt_hashes[(task, row)],
            record["source"]["dataset_id"] == self.dataset[task][0],
            record["source"]["dataset_revision"] == self.dataset[task][1],
            record["source"]["dataset_split"] == self.dataset[task][2],
            record["runner_repository"]["commit"] == EXPECTED_RUNNER_COMMIT,
            record["runner_repository"]["dirty"] is False,
        )
        if not all(checks):
            failed = [
                name
                for name, ok in zip(
                    (
                        "run_kind",
                        "policy",
                        "rollout",
                        "manifest",
                        "model_id",
                        "model_revision",
                        "justgrpo",
                        "freeze_sha",
                        "split_role",
                        "row_index",
                        "prompt_hash",
                        "dataset_id",
                        "dataset_revision",
                        "dataset_split",
                        "runner_commit",
                        "runner_dirty",
                    ),
                    checks,
                    strict=True,
                )
                if not ok
            ]
            raise FrozenRuleError(f"provenance mismatch {failed} for {item}")

    def trace_hash_matches(self, result_path: Path, record: dict[str, Any]) -> bool:
        trace_path = result_path.with_name(result_path.stem + ".trace.jsonl.gz")
        if not trace_path.is_file():
            return False
        return sha256_file(trace_path) == record["trace_sha256"]


def _validate_artifacts(
    *,
    phase1_root: Path,
    manifest: dict[str, Any],
    record_validator: HeldoutRecordValidator,
    allow_partial: bool,
) -> dict[str, Any]:
    items = _expected_items(manifest)
    missing: list[str] = []
    invalid: list[str] = []
    present = 0
    for item in items:
        path = _result_path(phase1_root, item)
        if not path.is_file():
            missing.append(str(path))
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        try:
            record_validator.validate(record, item=item)
            if not record_validator.trace_hash_matches(path, record):
                raise FrozenRuleError(f"trace hash mismatch for {item}")
        except FrozenRuleError as error:
            invalid.append(str(error))
            continue
        present += 1
    if not allow_partial and (missing or invalid):
        head = (missing + invalid)[:5]
        raise FrozenRuleError(
            f"held-out artifacts incomplete: {len(missing)} missing, "
            f"{len(invalid)} invalid; first issues: {head}"
        )
    return {
        "expected": len(items),
        "present_and_valid": present,
        "missing": len(missing),
        "invalid": len(invalid),
        "first_missing": missing[:5],
        "first_invalid": invalid[:5],
    }


def _validate_shard_manifests(
    phase1_root: Path, *, run_kind: str, num_shards: int
) -> dict[str, Any]:
    markers: list[dict[str, Any]] = []
    for shard in range(num_shards):
        path = (
            phase1_root
            / PROTOCOL_VERSION
            / "manifests"
            / "heldout"
            / run_kind
            / "all"
            / f"shard-{shard:03d}-of-{num_shards:03d}.json"
        )
        if not path.is_file():
            raise FrozenRuleError(f"missing shard completion marker: {path}")
        marker = json.loads(path.read_text(encoding="utf-8"))
        if marker.get("expected_keys_sha256") != marker.get("observed_keys_sha256"):
            raise FrozenRuleError(
                f"shard marker expected/observed key mismatch: {path}"
            )
        markers.append(marker)
    total = sum(int(marker["result_count"]) for marker in markers)
    return {
        "run_kind": run_kind,
        "num_shards": num_shards,
        "total_result_count": total,
        "marker_paths_checked": num_shards,
    }


# ---------------------------------------------------------------------------
# Pre-registered verdict taxonomy and screening missingness audit
# ---------------------------------------------------------------------------


def _classify_ci(low: float, high: float) -> str:
    if low > 0.0:
        return "entirely_above_zero"
    if high < 0.0:
        return "entirely_below_zero"
    return "straddles_zero"


def _differential_missingness(
    cells: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    """Descriptive differential-missingness comparison; no p-values."""
    tasks_out: dict[str, Any] = {}
    for task in TASKS:
        rates = {
            policy: (
                cells[(task, policy)]["malformed_fraction"]
                if cells[(task, policy)]["malformed_fraction"] is not None
                else None
            )
            for policy in SCREENING_POLICIES
        }
        pairs = []
        for left_index, left in enumerate(SCREENING_POLICIES):
            for right in SCREENING_POLICIES[left_index + 1 :]:
                if rates[left] is None or rates[right] is None:
                    difference = None
                else:
                    difference = rates[left] - rates[right]
                pairs.append(
                    {"left": left, "right": right, "difference": difference}
                )
        pairs.sort(
            key=lambda entry: abs(entry["difference"])
            if entry["difference"] is not None
            else -1.0,
            reverse=True,
        )
        primary = (
            rates["ar_b1"] - rates["ao_b32"]
            if rates["ar_b1"] is not None and rates["ao_b32"] is not None
            else None
        )
        tasks_out[task] = {
            "primary_ar_minus_ao_rate_difference": primary,
            "flagged_for_confound_risk": (
                primary is not None and abs(primary) >= 0.01
            ),
            "max_abs_pairwise_rate_difference": pairs[0],
            "all_policy_rates": rates,
            "note": (
                "Descriptive only; no p-values, no formal inference. A large "
                "ar_b1-vs-ao_b32 differential would confound the primary "
                "contrast independently of any coverage mechanism; the frozen "
                "valid-output sensitivity is the designated remedy and will "
                "run automatically in the final analysis."
            ),
        }
    return tasks_out


def audit_screening_malformed(
    *,
    phase1_root: Path,
    manifest: dict[str, Any],
    record_validator: HeldoutRecordValidator,
) -> dict[str, Any]:
    """Screening-only missingness audit; never reads correctness fields.

    Reads only ``parser_status`` and ``parsed_answer`` (the frozen malformed
    definition). Its output is invariant to ``correct`` and ``response``
    values, proven by sentinel-substitution test. This is an operational
    audit, not a Gate 1 analysis, and it may run while the extension is
    still in progress.
    """
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for task in TASKS:
        rows = sorted(
            entry["row_index"] for entry in manifest["splits"][task]["heldout"]
        )
        for policy in SCREENING_POLICIES:
            key = (task, policy)
            cell: dict[str, Any] = {
                "task": task,
                "policy": policy,
                "expected": len(rows) * SCREENING_ROLLOUT_COUNT,
                "present_and_valid": 0,
                "missing": 0,
                "invalid": 0,
                "malformed_count": 0,
                "malformed_fraction": None,
                "parser_status_counts": defaultdict(int),
            }
            for row in rows:
                for rollout in range(SCREENING_ROLLOUT_COUNT):
                    item = (task, row, policy, "screening", rollout)
                    path = _result_path(phase1_root, item)
                    if not path.is_file():
                        cell["missing"] += 1
                        continue
                    try:
                        record = json.loads(path.read_text(encoding="utf-8"))
                        record_validator.validate(record, item=item)
                    except FrozenRuleError:
                        cell["invalid"] += 1
                        continue
                    cell["present_and_valid"] += 1
                    cell["parser_status_counts"][record["parser_status"]] += 1
                    if _malformed(record):
                        cell["malformed_count"] += 1
            denominator = cell["present_and_valid"]
            cell["malformed_fraction"] = (
                cell["malformed_count"] / denominator if denominator else None
            )
            cell["parser_status_counts"] = dict(
                sorted(cell["parser_status_counts"].items())
            )
            cell["exceeds_sensitivity_threshold"] = (
                cell["malformed_fraction"] is not None
                and cell["malformed_fraction"] > MALFORMED_SENSITIVITY_THRESHOLD
            )
            cells[key] = cell
    triggered = [
        {"task": cell["task"], "policy": cell["policy"]}
        for cell in cells.values()
        if cell["exceeds_sensitivity_threshold"]
    ]
    counts = {
        "expected": sum(cell["expected"] for cell in cells.values()),
        "present_and_valid": sum(
            cell["present_and_valid"] for cell in cells.values()
        ),
        "missing": sum(cell["missing"] for cell in cells.values()),
        "invalid": sum(cell["invalid"] for cell in cells.values()),
    }
    return {
        "audit_version": AUDIT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "held-out screening artifacts only; runs before extension "
            "completion and reads no correctness values"
        ),
        "correctness_read": False,
        "fields_read": ["parser_status", "parsed_answer"],
        "outcome_invariance": (
            "output is invariant to 'correct' and 'response' fields; proven "
            "by sentinel-substitution test"
        ),
        "malformed_definition": "parser_status != 'ok' or parsed_answer is null",
        "sensitivity_threshold": MALFORMED_SENSITIVITY_THRESHOLD,
        "counts": counts,
        "by_cell": {
            f"{key[0]}|{key[1]}": cell for key, cell in sorted(cells.items())
        },
        "triggered": triggered,
        "differential_missingness": _differential_missingness(cells),
        "note": (
            "Operational missingness audit, not a Gate 1 analysis. Any cell "
            "above the frozen threshold means the frozen valid-output "
            "sensitivity will run in the final analysis."
        ),
    }


def render_malformed_audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Phase 1 Screening Malformed-Output Audit",
        "",
        f"- Audit version: `{audit['audit_version']}`",
        f"- Created (UTC): `{audit['created_at_utc']}`",
        f"- Scope: {audit['scope']}",
        f"- Correctness fields read: **{audit['correctness_read']}**; fields "
        f"read by design: {', '.join(audit['fields_read'])}",
        f"- {audit['outcome_invariance']}",
        f"- Malformed definition: `{audit['malformed_definition']}`",
        f"- Frozen sensitivity threshold: "
        f"{audit['sensitivity_threshold']:.0%} (trigger is strictly greater)",
        "",
        "## Counts",
        "",
        f"- Expected screening records: {audit['counts']['expected']:,}",
        f"- Present and valid: {audit['counts']['present_and_valid']:,}",
        f"- Missing: {audit['counts']['missing']:,}",
        f"- Invalid: {audit['counts']['invalid']:,}",
        "",
        "## Per task-policy cell",
        "",
        "| task | policy | expected | valid | missing | invalid | malformed | rate | >1%? |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in sorted(
        audit["by_cell"].values(), key=lambda c: (c["task"], c["policy"])
    ):
        task, policy = cell["task"], cell["policy"]
        rate = (
            f"{cell['malformed_fraction']:.4f}"
            if cell["malformed_fraction"] is not None
            else "n/a"
        )
        lines.append(
            f"| {task} | {policy} | {cell['expected']} | "
            f"{cell['present_and_valid']} | {cell['missing']} | "
            f"{cell['invalid']} | {cell['malformed_count']} | {rate} | "
            f"{'yes' if cell['exceeds_sensitivity_threshold'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"Triggered cells: {audit['triggered'] or 'none'}",
            "",
            "## Differential missingness (descriptive, no p-values)",
            "",
            "| task | ar_b1 - ao_b32 rate | flagged (>=1pp) | max |pairwise| gap |",
            "|---|---:|---:|---|---|",
        ]
    )
    for task in TASKS:
        entry = audit["differential_missingness"][task]
        primary = entry["primary_ar_minus_ao_rate_difference"]
        primary_text = f"{primary:.4f}" if primary is not None else "n/a"
        worst = entry["max_abs_pairwise_rate_difference"]
        worst_text = (
            f"{worst['left']} vs {worst['right']}: "
            f"{worst['difference']:.4f}"
            if worst["difference"] is not None
            else f"{worst['left']} vs {worst['right']}: n/a"
        )
        lines.append(
            f"| {task} | {primary_text} | "
            f"{'yes' if entry['flagged_for_confound_risk'] else 'no'} | {worst_text} |"
        )
    lines.extend(
        [
            "",
            "A flagged differential would confound the primary contrast "
            "independently of any coverage mechanism; the frozen valid-output "
            "sensitivity is the designated remedy.",
            "",
            f"{audit['note']}",
            "",
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Outcome extraction (analyze mode only)
# ---------------------------------------------------------------------------


def _load_records(
    phase1_root: Path,
    manifest: dict[str, Any],
    *,
    record_validator: HeldoutRecordValidator,
) -> dict[tuple[str, int, str, str], list[dict[str, Any]]]:
    records: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for task in TASKS:
        rows = sorted(
            entry["row_index"] for entry in manifest["splits"][task]["heldout"]
        )
        for row in rows:
            for policy in SCREENING_POLICIES:
                for rollout in range(SCREENING_ROLLOUT_COUNT):
                    item = (task, row, policy, "screening", rollout)
                    path = _result_path(phase1_root, item)
                    record = json.loads(path.read_text(encoding="utf-8"))
                    record_validator.validate(record, item=item)
                    records[(task, row, policy, "screening")].append(record)
            for policy in PRIMARY_POLICIES:
                for rollout in range(
                    SCREENING_ROLLOUT_COUNT, TOTAL_ROLLOUT_COUNT
                ):
                    item = (task, row, policy, "primary-extension", rollout)
                    path = _result_path(phase1_root, item)
                    record = json.loads(path.read_text(encoding="utf-8"))
                    record_validator.validate(record, item=item)
                    records[(task, row, policy, "primary-extension")].append(record)
    return records


def _diagnostic_token_rows(
    phase1_root: Path, manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    """Token rows for the frozen diagnostic window (ao_b32, r00-r15)."""
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        queries = sorted(
            entry["row_index"] for entry in manifest["splits"][task]["heldout"]
        )
        for query in queries:
            for rollout in DIAGNOSTIC_ROLLOUTS:
                trace_path = (
                    phase1_root
                    / PROTOCOL_VERSION
                    / "heldout"
                    / "screening"
                    / task
                    / "ao_b32"
                    / f"q{query:04d}"
                    / f"r{rollout:02d}.trace.jsonl.gz"
                )
                rows.extend(
                    _selected_token_rows(
                        _read_trace(trace_path),
                        task=task,
                        query=query,
                        rollout=rollout,
                    )
                )
    return rows


def _query_bci(
    token_rows: list[dict[str, Any]],
    task_parameters: dict[str, Any],
    expected_queries: set[tuple[str, int]],
) -> dict[tuple[str, int], float]:
    """Frozen BCI: weighted candidate closure per rollout, mean over 16 rollouts."""
    by_rollout: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in token_rows:
        by_rollout[(row["task"], row["query"], row["rollout"])].append(row)
    bci: dict[tuple[str, int], list[float]] = defaultdict(list)
    for task, query, rollout in sorted(by_rollout):
        if rollout not in DIAGNOSTIC_ROLLOUTS:
            continue
        candidates = [row for row in by_rollout[(task, query, rollout)] if row["candidate"]]
        if not candidates:
            raise FrozenRuleError(
                f"candidate fallback failed for {task} query {query} rollout {rollout}"
            )
        threshold = task_parameters[task]["candidate_threshold_q75"]
        weights = [
            1.0 + max(row["candidate_score"] - threshold, 0.0)
            for row in candidates
        ]
        bci[(task, query)].append(
            sum(
                weight * row["entropy_closure"]
                for weight, row in zip(weights, candidates, strict=True)
            )
            / sum(weights)
        )
    for key in sorted(expected_queries):
        if key not in bci or len(bci[key]) != len(DIAGNOSTIC_ROLLOUTS):
            raise FrozenRuleError(
                f"BCI window incomplete for {key}: "
                f"{len(bci.get(key, []))} of {len(DIAGNOSTIC_ROLLOUTS)} rollouts"
            )
    return {key: mean(values) for key, values in bci.items()}


def _formal_coverage_advantage(
    ar_records: list[dict[str, Any]], ao_records: list[dict[str, Any]]
) -> float:
    if len(ar_records) != TOTAL_ROLLOUT_COUNT or len(ao_records) != TOTAL_ROLLOUT_COUNT:
        raise FrozenRuleError("formal target requires exactly 64 rollouts per policy")
    ar_successes = sum(bool(record["correct"]) for record in ar_records)
    ao_successes = sum(bool(record["correct"]) for record in ao_records)
    return mean(
        estimate_pass_at_k(TOTAL_ROLLOUT_COUNT, ar_successes, k)
        - estimate_pass_at_k(TOTAL_ROLLOUT_COUNT, ao_successes, k)
        for k in FORMAL_HELDOUT_K_GRID
    )


def _coverage_auc(records: list[dict[str, Any]]) -> float:
    successes = sum(bool(record["correct"]) for record in records)
    return mean(
        estimate_pass_at_k(len(records), successes, k) for k in FORMAL_HELDOUT_K_GRID
    )


def _predict(
    features: dict[str, float], predictor: dict[str, Any]
) -> float:
    if not all(math.isfinite(float(features[name])) for name in features):
        raise FrozenRuleError("nonfinite frozen predictor feature")
    terms = {"intercept": 1.0, "task_math500": features["task_math500"]}
    for name, params in predictor["standardization"].items():
        terms[name] = (features[name] - params["mean"]) / params["scale"]
    if set(terms) != set(predictor["coefficients"]):
        raise FrozenRuleError("frozen predictor feature set mismatch")
    value = sum(
        predictor["coefficients"][name] * term for name, term in terms.items()
    )
    if not math.isfinite(value):
        raise FrozenRuleError("nonfinite frozen prediction")
    return value


def _malformed(record: dict[str, Any]) -> bool:
    return record["parser_status"] != "ok" or record["parsed_answer"] is None


# ---------------------------------------------------------------------------
# Endpoint computation
# ---------------------------------------------------------------------------


def _closure_endpoints(
    matches: list[dict[str, Any]],
) -> dict[str, Any]:
    """RQ1: query-level candidate-minus-control closure with frozen missingness."""
    endpoint_field = "entropy_closure_difference"
    margin_field = "margin_closure_difference"
    pairs_by_query: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for match in matches:
        pairs_by_query[(match["task"], match["query"])].append(match)

    per_task: dict[str, Any] = {}
    for task in TASKS:
        total_queries = 100
        query_rows = [
            (query, rows)
            for (task_name, query), rows in sorted(pairs_by_query.items())
            if task_name == task
        ]
        matched = [
            {
                "query": query,
                "entropy": mean(row[endpoint_field] for row in rows),
                "margin": mean(row[margin_field] for row in rows),
                "pairs": len(rows),
            }
            for query, rows in query_rows
        ]
        matched_fraction = len(matched) / total_queries
        unresolved = matched_fraction < MINIMUM_MATCHED_QUERY_FRACTION
        task_endpoints = {row["query"]: row["entropy"] for row in matched}
        margin_endpoints = {row["query"]: row["margin"] for row in matched}
        ci = task_stratified_bootstrap(
            label=f"rq1_entropy_closure|{task}",
            tasks={task: list(task_endpoints.values())},
            statistic=lambda sample: mean(
                value for values in sample.values() for value in values
            ),
        )
        margin_ci = task_stratified_bootstrap(
            label=f"rq1_margin_closure|{task}",
            tasks={task: list(margin_endpoints.values())},
            statistic=lambda sample: mean(
                value for values in sample.values() for value in values
            ),
        )
        per_task[task] = {
            "total_queries": total_queries,
            "matched_queries": len(matched),
            "structurally_unmatched_queries": total_queries - len(matched),
            "matched_fraction": matched_fraction,
            "minimum_matched_fraction": MINIMUM_MATCHED_QUERY_FRACTION,
            "rq1_unresolved": unresolved,
            "matched_pair_count": sum(row["pairs"] for row in matched),
            "entropy_closure": ci,
            "margin_closure": margin_ci,
            "per_query_entropy_closure": task_endpoints,
        }
    pooled = {
        "entropy_closure_mean_of_task_means": mean(
            per_task[task]["entropy_closure"]["point"] for task in TASKS
        ),
        "margin_closure_mean_of_task_means": mean(
            per_task[task]["margin_closure"]["point"] for task in TASKS
        ),
    }
    return {"by_task": per_task, "pooled": pooled}


def _closure_gate(per_task: dict[str, Any]) -> dict[str, Any]:
    """Gate 1 condition 1 exactly as written in the frozen protocol."""
    unresolved = [task for task in TASKS if per_task[task]["rq1_unresolved"]]
    positive_both = all(per_task[task]["entropy_closure"]["point"] > 0.0 for task in TASKS)
    any_positive_interval = any(
        per_task[task]["entropy_closure"]["ci_low"] > 0.0 for task in TASKS
    )
    any_negative_interval = any(
        per_task[task]["entropy_closure"]["ci_high"] < 0.0 for task in TASKS
    )
    passed = (
        not unresolved
        and (positive_both or (any_positive_interval and not any_negative_interval))
    )
    return {
        "passed": passed,
        "unresolved_tasks": unresolved,
        "positive_on_both_tasks": positive_both,
        "interval_excludes_zero_positive": {
            task: per_task[task]["entropy_closure"]["ci_low"] > 0.0
            for task in TASKS
        },
        "interval_excludes_zero_negative": {
            task: per_task[task]["entropy_closure"]["ci_high"] < 0.0
            for task in TASKS
        },
        "rule": (
            "pass iff no task is unresolved AND (mean > 0 on both tasks OR "
            "some 95% interval lies entirely above zero while no interval "
            "lies entirely below zero)"
        ),
    }


def _bci_association(
    per_query: dict[tuple[str, int], dict[str, float]]
) -> dict[str, Any]:
    assignments = {
        task: [
            (row["bci"], row["formal_advantage"])
            for (task_name, _), row in sorted(per_query.items())
            if task_name == task
        ]
        for task in TASKS
    }
    per_task = {
        task: task_stratified_bootstrap(
            label=f"rq2_spearman|{task}",
            tasks={task: assignments[task]},
            statistic=lambda sample: stratified_spearman(sample),
        )
        for task in TASKS
    }
    pooled = task_stratified_bootstrap(
        label="rq2_spearman|pooled",
        tasks=assignments,
        statistic=stratified_spearman,
    )
    zero_fraction = {
        task: mean(
            1.0 if pair[1] == 0.0 else 0.0 for pair in assignments[task]
        )
        for task in TASKS
    }
    return {
        "per_task": per_task,
        "pooled": pooled,
        "pooled_ci_above_zero": pooled["ci_low"] > 0.0,
        "advantage_zero_fraction": zero_fraction,
    }


def _predictive_value(
    per_query: dict[tuple[str, int], dict[str, float]],
    baseline: dict[str, Any],
    augmented: dict[str, Any],
) -> dict[str, Any]:
    def per_task_mae(
        predictor: dict[str, Any], task: str
    ) -> dict[tuple[str, int], float]:
        errors = {
            key: abs(
                _predict(row["features"], predictor) - row["formal_advantage"]
            )
            for key, row in per_query.items()
            if key[0] == task
        }
        if len(errors) != 100:
            raise FrozenRuleError(
                f"expected 100 query errors for {task}, found {len(errors)}"
            )
        return errors

    baseline_by_task = {task: per_task_mae(baseline, task) for task in TASKS}
    augmented_by_task = {task: per_task_mae(augmented, task) for task in TASKS}
    paired_delta = {
        key: baseline_by_task[key[0]][key] - augmented_by_task[key[0]][key]
        for key in per_query
    }
    tasks_input = {
        task: [
            paired_delta[key] for key in sorted(per_query) if key[0] == task
        ]
        for task in TASKS
    }
    pooled = task_stratified_bootstrap(
        label="predictive_mae_delta|pooled",
        tasks=tasks_input,
        statistic=lambda sample: mean(
            value for values in sample.values() for value in values
        ),
    )
    per_task = {}
    for task in TASKS:
        per_task[task] = task_stratified_bootstrap(
            label=f"predictive_mae_delta|{task}",
            tasks={task: tasks_input[task]},
            statistic=lambda sample: mean(
                value for values in sample.values() for value in values
            ),
        )
        per_task[task]["mae_baseline"] = mean(
            value for value in baseline_by_task[task].values()
        )
        per_task[task]["mae_augmented"] = mean(
            value for value in augmented_by_task[task].values()
        )
    return {
        "per_task": per_task,
        "pooled": pooled,
        "pooled_mae_baseline": mean(
            value for task in TASKS for value in baseline_by_task[task].values()
        ),
        "pooled_mae_augmented": mean(
            value for task in TASKS for value in augmented_by_task[task].values()
        ),
        "pooled_delta_ci_above_zero": pooled["ci_low"] > 0.0,
    }


def _policy_summaries(
    records: dict[tuple[str, int, str, str], list[dict[str, Any]]],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for task in TASKS:
        queries = sorted(
            entry["row_index"] for entry in manifest["splits"][task]["heldout"]
        )
        for policy in SCREENING_POLICIES:
            rows = [
                records[(task, query, policy, "screening")]
                for query in queries
            ]
            success_counts = [
                sum(bool(record["correct"]) for record in rollouts)
                for rollouts in rows
            ]
            malformed = [
                sum(1 for record in rollouts if _malformed(record))
                for rollouts in rows
            ]
            entry: dict[str, Any] = {
                "task": task,
                "policy": policy,
                "rollout_count": SCREENING_ROLLOUT_COUNT,
                "pass_at_1": mean(
                    mean(bool(record["correct"]) for record in rollouts)
                    for rollouts in rows
                ),
                "malformed_fraction": mean(
                    count / len(rollouts)
                    for count, rollouts in zip(malformed, rows, strict=True)
                ),
                "parser_status_counts": dict(
                    sorted(
                        {
                            status: sum(
                                1
                                for rollouts in rows
                                for record in rollouts
                                if record["parser_status"] == status
                            )
                            for status in {
                                record["parser_status"]
                                for rollouts in rows
                                for record in rollouts
                            }
                        }.items()
                    )
                ),
                "mean_unique_parsed_answers": mean(
                    len(
                        {
                            record["parsed_answer"]
                            for record in rollouts
                            if record["parsed_answer"] is not None
                        }
                    )
                    for rollouts in rows
                ),
            }
            for k in SCREENING_REPORT_K_GRID:
                entry[f"pass_at_{k}"] = aggregate_pass_at_k(
                    success_counts, SCREENING_ROLLOUT_COUNT, k
                )
            if policy in PRIMARY_POLICIES:
                full = [
                    records[(task, query, policy, "screening")]
                    + records[(task, query, policy, "primary-extension")]
                    for query in queries
                ]
                full_successes = [
                    sum(bool(record["correct"]) for record in rollouts)
                    for rollouts in full
                ]
                entry["pass_at_32"] = aggregate_pass_at_k(
                    full_successes, TOTAL_ROLLOUT_COUNT, 32
                )
                entry["pass_at_64"] = aggregate_pass_at_k(
                    full_successes, TOTAL_ROLLOUT_COUNT, 64
                )
                entry["coverage_auc"] = mean(
                    _coverage_auc(rollouts) for rollouts in full
                )
            summaries.append(entry)
    return summaries


def _dose_response(
    records: dict[tuple[str, int, str, str], list[dict[str, Any]]],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    for task in TASKS:
        queries = sorted(
            entry["row_index"] for entry in manifest["splits"][task]["heldout"]
        )
        for policy in SCREENING_POLICIES:
            success_counts = [
                sum(
                    bool(record["correct"])
                    for record in records[(task, query, policy, "screening")]
                )
                for query in queries
            ]
            row = {
                "task": task,
                "policy": policy,
                "block_length": {
                    "ar_b1": 1,
                    "ao_b8": 8,
                    "ao_b16": 16,
                    "ao_b32": 32,
                    "random_b32": 32,
                }[policy],
            }
            for k in SCREENING_REPORT_K_GRID:
                row[f"pass_at_{k}"] = aggregate_pass_at_k(
                    success_counts, SCREENING_ROLLOUT_COUNT, k
                )
            rows_out.append(row)
    return rows_out


def _solved_membership(
    records: dict[tuple[str, int, str, str], list[dict[str, Any]]],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        queries = sorted(
            entry["row_index"] for entry in manifest["splits"][task]["heldout"]
        )
        ar_only = ao_only = both = neither = 0
        for query in queries:
            ar_solved = any(
                bool(record["correct"])
                for record in records[(task, query, "ar_b1", "screening")]
                + records[(task, query, "ar_b1", "primary-extension")]
            )
            ao_solved = any(
                bool(record["correct"])
                for record in records[(task, query, "ao_b32", "screening")]
                + records[(task, query, "ao_b32", "primary-extension")]
            )
            if ar_solved and ao_solved:
                both += 1
            elif ar_solved:
                ar_only += 1
            elif ao_solved:
                ao_only += 1
            else:
                neither += 1
        rows.append(
            {
                "task": task,
                "ar_only": ar_only,
                "ao_only": ao_only,
                "both": both,
                "neither": neither,
            }
        )
    return rows


def _confidence_vs_random_contrast(
    records: dict[tuple[str, int, str, str], list[dict[str, Any]]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """``ao_b32 - random_b32`` at matched block size over 16 rollouts.

    Both policies share identical order freedom; they differ only in whether
    the commitment order is confidence-guided or random. This is a frozen
    descriptive secondary contrast, not a causal comparison, and it is not
    part of Gate 1. It consumes screening rollouts only, so it cannot be
    affected by extension outcomes.
    """

    def pass_diff(task: str, query: int, k: int) -> float:
        ao_successes = sum(
            bool(record["correct"])
            for record in records[(task, query, "ao_b32", "screening")]
        )
        random_successes = sum(
            bool(record["correct"])
            for record in records[(task, query, "random_b32", "screening")]
        )
        return estimate_pass_at_k(
            SCREENING_ROLLOUT_COUNT, ao_successes, k
        ) - estimate_pass_at_k(SCREENING_ROLLOUT_COUNT, random_successes, k)

    per_query: dict[tuple[str, int], dict[str, float]] = {}
    for task in TASKS:
        queries = sorted(
            entry["row_index"] for entry in manifest["splits"][task]["heldout"]
        )
        for query in queries:
            per_k = {
                str(k): pass_diff(task, query, k) for k in SCREENING_REPORT_K_GRID
            }
            per_query[(task, query)] = {
                **per_k,
                "mean_over_k": mean(per_k.values()),
            }

    def query_values(task: str) -> list[float]:
        return [
            per_query[(task, query)]["mean_over_k"]
            for query in sorted(
                entry["row_index"]
                for entry in manifest["splits"][task]["heldout"]
            )
        ]

    per_task = {}
    for task in TASKS:
        values = query_values(task)
        ci = task_stratified_bootstrap(
            label=f"confidence_vs_random|{task}",
            tasks={task: values},
            statistic=lambda sample: mean(
                value for bucket in sample.values() for value in bucket
            ),
        )
        queries = sorted(
            entry["row_index"] for entry in manifest["splits"][task]["heldout"]
        )
        per_task[task] = {
            "point": ci["point"],
            "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"],
            "per_k": {
                str(k): mean(per_query[(task, query)][str(k)] for query in queries)
                for k in SCREENING_REPORT_K_GRID
            },
        }
    pooled = task_stratified_bootstrap(
        label="confidence_vs_random|pooled",
        tasks={task: query_values(task) for task in TASKS},
        statistic=lambda sample: mean(
            value for bucket in sample.values() for value in bucket
        ),
    )
    pooled_per_k = {}
    for k in SCREENING_REPORT_K_GRID:
        pooled_per_k[str(k)] = mean(
            per_query[(task, query)][str(k)]
            for task in TASKS
            for query in sorted(
                entry["row_index"]
                for entry in manifest["splits"][task]["heldout"]
            )
        )
    return {
        "scope": (
            "ao_b32 - random_b32, matched block size 32, 16-rollout screening "
            "window, k in {2,4,8,16}; descriptive secondary, not causal, not "
            "part of Gate 1"
        ),
        "k_grid": list(SCREENING_REPORT_K_GRID),
        "per_task": per_task,
        "pooled": pooled,
        "pooled_ci_classification": _classify_ci(
            pooled["ci_low"], pooled["ci_high"]
        ),
        "pooled_per_k": pooled_per_k,
        "pooled_weighting": (
            "equal query weight across both tasks (equivalent to equal task "
            "weight at 100 queries per task)"
        ),
    }


def _malformed_sensitivity(
    records: dict[tuple[str, int, str, str], list[dict[str, Any]]],
    per_query: dict[tuple[str, int], dict[str, float]],
    summaries: list[dict[str, Any]],
) -> dict[str, Any] | None:
    trigger = [
        (entry["task"], entry["policy"])
        for entry in summaries
        if entry["malformed_fraction"] > MALFORMED_SENSITIVITY_THRESHOLD
    ]
    if not trigger:
        return None
    result_rows = []
    for task in TASKS:
        queries = sorted(
            {
                query
                for (task_name, query) in per_query
                if task_name == task
            }
        )
        sensitivity_values = []
        dropped_queries = 0
        for query in queries:
            counts = {}
            for policy in PRIMARY_POLICIES:
                rollouts = records[(task, query, policy, "screening")] + records[
                    (task, query, policy, "primary-extension")
                ]
                valid = [record for record in rollouts if not _malformed(record)]
                counts[policy] = (
                    len(valid),
                    sum(bool(record["correct"]) for record in valid),
                )
            usable_k = [
                k
                for k in FORMAL_HELDOUT_K_GRID
                if k <= min(counts[policy][0] for policy in PRIMARY_POLICIES)
            ]
            if len(usable_k) < len(FORMAL_HELDOUT_K_GRID):
                dropped_queries += 1
            if not usable_k:
                continue
            sensitivity_values.append(
                mean(
                    estimate_pass_at_k(counts["ar_b1"][0], counts["ar_b1"][1], k)
                    - estimate_pass_at_k(counts["ao_b32"][0], counts["ao_b32"][1], k)
                    for k in usable_k
                )
            )
        result_rows.append(
            {
                "task": task,
                "mean_sensitivity_advantage": (
                    mean(sensitivity_values) if sensitivity_values else None
                ),
                "queries_with_dropped_k": dropped_queries,
                "queries_evaluated": len(sensitivity_values),
            }
        )
    return {
        "triggered_by": trigger,
        "threshold": MALFORMED_SENSITIVITY_THRESHOLD,
        "by_task": result_rows,
        "note": (
            "valid-output sensitivity: parser-failed rollouts removed from the "
            "denominator; k values exceeding a query's valid count are dropped "
            "from that query's mean (descriptive only)"
        ),
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _flatten_rows(tables: dict[str, list[dict[str, Any]]]) -> str:
    import csv
    import io

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["table", "row_key", "field", "value"])
    for table in sorted(tables):
        for entry in tables[table]:
            key_fields = ("task", "policy", "condition")
            row_key = ":".join(
                str(entry[key]) for key in key_fields if key in entry
            )
            for field, value in entry.items():
                if field in key_fields:
                    continue
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, sort_keys=True)
                writer.writerow([table, row_key, field, value])
    return buffer.getvalue()


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    closure = report["rq1_closure"]
    association = report["rq2_bci_association"]
    predictive = report["predictive_value"]
    gate = report["gate1"]
    lines = [
        "# Phase 1 Held-Out Analysis and Gate 1 Report",
        "",
        "This report consumes the final BCI freeze directly. No coefficient, "
        "candidate rule, matching caliper, or target definition was refit or "
        "altered on held-out data. Every frozen endpoint is reported, "
        "favorable or not.",
        "",
        "## Provenance",
        "",
        f"- BCI freeze SHA-256: `{report['bci_freeze_sha256']}`",
        f"- Analysis freeze SHA-256: `{report['analysis_freeze_sha256']}`",
        f"- Analyzer commit: `{report['analyzer_commit']}`",
        f"- Inference runner commit: `{EXPECTED_RUNNER_COMMIT}`",
        f"- Bootstrap: seed {BOOTSTRAP_SEED}, {BOOTSTRAP_REPLICATES} replicates, "
        "task-stratified percentile 95% intervals",
        f"- Artifacts validated: {report['artifact_validation']['present_and_valid']}"
        f"/{report['artifact_validation']['expected']}",
        "",
        "## Amendment Invariants",
        "",
        "The analysis-freeze amendment chain above cannot change any of the "
        "following; each is mechanically enforced by this analyzer on every "
        "run:",
        "",
        f"- Final BCI freeze SHA-256: `{EXPECTED_FREEZE_SHA256}` "
        "(identity-checked on load; the two superseded calibration freezes "
        "are audit-only)",
        "- Gate 1 boolean rules (closure condition; pooled BCI association "
        "and MAE-delta conditions) exactly as frozen",
        "- Frozen decode parameters: 256 generated tokens, 256 denoising "
        "steps, temperature 0.6, CFG scale 0, mask token 126336, and the "
        "five policies with their block sizes and selection rules",
        f"- Inference runner commit `{EXPECTED_RUNNER_COMMIT}` "
        "(validated on every artifact, together with the model revision, "
        "dataset revisions, and the frozen split manifest)",
        "",
        "## Design Cardinality and Generation Arithmetic",
        "",
    ]
    cardinality = report["design_cardinality"]
    lines.extend(
        [
            f"- Held-out queries: {cardinality['heldout_queries_per_task']} per task"
            f", {cardinality['heldout_query_total']} total",
            f"- Screening: 5 policies x 16 rollouts = "
            f"{cardinality['screening_generations']:,} generations",
            f"- Primary extension: ar_b1 and ao_b32 x 48 added rollouts = "
            f"{cardinality['primary_extension_generations']:,} generations",
            f"- Held-out total: {cardinality['heldout_generations_total']:,} "
            "generations",
            f"- Protocol new-generation total 41,600 = 35,200 held-out + 6,400 "
            "new calibration (1,600 reused Phase 0 generations excluded)",
            f"- {cardinality['formal_pass_at_64_scope']}",
            "",
            "## Power Resolution (Planning Quantities)",
            "",
        ]
    )
    power = report["power_resolution"]
    lines.extend(
        [
            f"- Per-task correlation MDE (n=100, alpha=.05, 80% power): "
            f"{power['per_task_correlation_mde_n100']:.3f}",
            f"- Pooled correlation MDE (n=200 approximation): "
            f"{power['pooled_correlation_mde_n200']:.3f}",
            f"- Per-task standardized paired-effect MDE (n=100): "
            f"{power['per_task_paired_effect_mde_n100']:.3f}",
            f"- {power['approximation_note']}",
            "",
            "## RQ1: Candidate-Minus-Control Entropy Closure",
            "",
        ]
    )
    for task in TASKS:
        entry = closure["by_task"][task]
        ci = entry["entropy_closure"]
        margin = entry["margin_closure"]
        lines.extend(
            [
                f"### {task}",
                "",
                f"- Matched queries: {entry['matched_queries']}/{entry['total_queries']}"
                f" ({entry['matched_fraction']:.1%}); unmatched are reported as "
                "structurally unmatched, not dropped silently",
                f"- RQ1 unresolved for this task: {entry['rq1_unresolved']}",
                f"- Mean query-level entropy closure difference: "
                f"{ci['point']:.6f} (95% CI [{ci['ci_low']:.6f}, {ci['ci_high']:.6f}])",
                f"- Margin closure (secondary): {margin['point']:.6f} "
                f"(95% CI [{margin['ci_low']:.6f}, {margin['ci_high']:.6f}])",
                "",
            ]
        )
    lines.extend(
        [
            "## RQ2: BCI Association With AR Coverage Advantage",
            "",
            f"- Pooled task-stratified Spearman: "
            f"{association['pooled']['point']:.6f} "
            f"(95% CI [{association['pooled']['ci_low']:.6f}, "
            f"{association['pooled']['ci_high']:.6f}])",
        ]
    )
    for task in TASKS:
        entry = association["per_task"][task]
        lines.append(
            f"- {task}: rho = {entry['point']:.6f} "
            f"(95% CI [{entry['ci_low']:.6f}, {entry['ci_high']:.6f}]); "
            f"zero-advantage fraction {association['advantage_zero_fraction'][task]:.1%}"
        )
    lines.extend(
        [
            "",
            "## Incremental Predictive Value",
            "",
            f"- Pooled MAE baseline: {predictive['pooled_mae_baseline']:.6f}",
            f"- Pooled MAE augmented: {predictive['pooled_mae_augmented']:.6f}",
            f"- MAE_baseline - MAE_augmented: {predictive['pooled']['point']:.6f} "
            f"(95% CI [{predictive['pooled']['ci_low']:.6f}, "
            f"{predictive['pooled']['ci_high']:.6f}])",
            "",
        ]
    )
    for task in TASKS:
        entry = predictive["per_task"][task]
        lines.append(
            f"- {task}: MAE baseline {entry['mae_baseline']:.6f}, augmented "
            f"{entry['mae_augmented']:.6f}, delta "
            f"{entry['point']:.6f} (95% CI [{entry['ci_low']:.6f}, {entry['ci_high']:.6f}])"
        )
    lines.extend(
        [
            "",
            "## Gate 1",
            "",
            f"- Condition 1 (closure): **{'PASS' if gate['condition_1_closure']['passed'] else 'FAIL'}**"
            f" — unresolved tasks: {gate['condition_1_closure']['unresolved_tasks'] or 'none'}",
            f"- Condition 2a (pooled BCI interval above zero): "
            f"**{'PASS' if gate['condition_2_bci']['pooled_ci_above_zero'] else 'FAIL'}**"
            f" — pre-registered classification: "
            f"`{gate['condition_2_bci']['pooled_ci_classification']}`",
            f"- Condition 2b (MAE improvement interval above zero): "
            f"**{'PASS' if gate['condition_2_bci']['pooled_delta_ci_above_zero'] else 'FAIL'}**"
            f" — pre-registered classification: "
            f"`{gate['condition_2_bci']['mae_delta_ci_classification']}`",
            f"- **Gate 1 overall: {'PASS' if gate['overall_pass'] else 'FAIL'}**",
            "",
            "Verdict taxonomy (pre-registered before unblinding): a CI entirely "
            "above zero passes; a CI straddling zero fails as unresolved; a CI "
            "entirely below zero fails AND is recorded as a resolved "
            "wrong-direction falsification, not a null.",
            "",
            "Gate 1 passes only if both conditions hold. On failure the frozen "
            "falsification rules apply: no k re-selection, no per-rollout "
            "accuracy substitution, no dropping MATH-500 or a policy, and no "
            "rescue through the Phase 0 lexical result.",
            "",
            "## Secondary Endpoints",
            "",
            "### Five-policy screening summary (per task)",
            "",
            "| task | policy | pass@1 | pass@8 | pass@16 | malformed |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for entry in report["policy_summaries"]:
        lines.append(
            f"| {entry['task']} | {entry['policy']} | {entry['pass_at_1']:.4f} | "
            f"{entry['pass_at_8']:.4f} | {entry['pass_at_16']:.4f} | "
            f"{entry['malformed_fraction']:.4f} |"
        )
    lines.extend(
        [
            "",
            "### Primary-policy formal coverage (64 rollouts)",
            "",
            "| task | policy | pass@32 | pass@64 | CoverageAUC |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for entry in report["policy_summaries"]:
        if "coverage_auc" in entry:
            lines.append(
                f"| {entry['task']} | {entry['policy']} | {entry['pass_at_32']:.4f} | "
                f"{entry['pass_at_64']:.4f} | {entry['coverage_auc']:.4f} |"
            )
    lines.extend(
        [
            "",
            "### Solved-set membership (64 rollouts, per task)",
            "",
            "| task | AR-only | AO-only | both | neither |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for entry in report["solved_membership"]:
        lines.append(
            f"| {entry['task']} | {entry['ar_only']} | {entry['ao_only']} | "
            f"{entry['both']} | {entry['neither']} |"
        )
    contrast = report["confidence_vs_random_contrast"]
    lines.extend(
        [
            "",
            "### Confidence-guided versus random order (ao_b32 - random_b32)",
            "",
            "Both policies share identical order freedom at block size 32; "
            "they differ only in whether the commitment order is "
            "confidence-guided or random. Descriptive secondary contrast; not "
            "causal; not part of Gate 1.",
            "",
            "| task | mean over k | 95% CI |",
            "|---|---:|---|",
        ]
    )
    for task in TASKS:
        entry = contrast["per_task"][task]
        lines.append(
            f"| {task} | {entry['point']:.4f} | "
            f"[{entry['ci_low']:.4f}, {entry['ci_high']:.4f}] |"
        )
    pooled = contrast["pooled"]
    lines.append(
        f"| pooled | {pooled['point']:.4f} | "
        f"[{pooled['ci_low']:.4f}, {pooled['ci_high']:.4f}] |"
    )
    classification = contrast["pooled_ci_classification"]
    readings = report["secondary_contrast_readings"]
    reading = {
        "entirely_above_zero": readings["entirely_above_zero"],
        "straddles_zero": readings["straddles_zero"],
        "entirely_below_zero": readings["entirely_below_zero"],
    }[classification]
    lines.extend(
        [
            "",
            f"- Pre-registered classification: `{classification}`",
            f"- Pre-registered reading: {reading}",
        ]
    )
    lines.extend(
        [
            "",
            "| task | pass@2 diff | pass@4 diff | pass@8 diff | pass@16 diff |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for task in TASKS:
        per_k = contrast["per_task"][task]["per_k"]
        lines.append(
            f"| {task} | {per_k['2']:.4f} | {per_k['4']:.4f} | "
            f"{per_k['8']:.4f} | {per_k['16']:.4f} |"
        )
    pooled_k = contrast["pooled_per_k"]
    lines.append(
        f"| pooled | {pooled_k['2']:.4f} | {pooled_k['4']:.4f} | "
        f"{pooled_k['8']:.4f} | {pooled_k['16']:.4f} |"
    )
    sensitivity = report["malformed_sensitivity"]
    if sensitivity is None:
        lines.extend(
            [
                "",
                "No task-policy malformed fraction exceeded "
                f"{MALFORMED_SENSITIVITY_THRESHOLD:.0%}; the valid-output "
                "sensitivity analysis was not triggered.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "### Valid-Output Sensitivity",
                "",
                f"Triggered by {sensitivity['triggered_by']}.",
                "",
                "| task | mean sensitivity advantage | queries with dropped k |",
                "|---|---:|---:|",
            ]
        )
        for entry in sensitivity["by_task"]:
            lines.append(
                f"| {entry['task']} | {_fmt(entry['mean_sensitivity_advantage'])} | "
                f"{entry['queries_with_dropped_k']} |"
            )
    lines.extend(
        [
            "",
            "## Audit Boundary",
            "",
            "Parser failures, truncations, and degenerate outputs remain in "
            "every Pass@k denominator. Token counts are never treated as "
            "independent sample sizes; all intervals are query bootstraps. "
            "This analyzer did not modify any frozen rule after held-out "
            "outcomes were read.",
            "",
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis freeze
# ---------------------------------------------------------------------------


def build_analysis_freeze(
    *,
    analyzer_commit: str,
    bci_freeze_sha256: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "analysis_version": ANALYSIS_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bci_freeze_sha256": bci_freeze_sha256,
        "runner_commit": EXPECTED_RUNNER_COMMIT,
        "analyzer_commit": analyzer_commit,
        "analyzer_commit_policy": (
            "the analyzer may run only from a checkout whose HEAD equals "
            "analyzer_commit; the inference checkout stays on runner_commit"
        ),
        "split_manifest_sha256": SPLIT_MANIFEST_SHA256,
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "interval": "percentile 95%",
            "scheme": (
                "queries resampled with replacement independently within each "
                "task; per-endpoint seeds derived deterministically from the "
                "master seed and an endpoint label"
            ),
            "degenerate_replicates": (
                "a replicate whose frozen statistic is undefined aborts the "
                "analysis with an explicit error; replicates are never dropped"
            ),
        },
        "stratified_spearman": {
            "step_1": "rank BCI and ARCoverageAdvantage with average ranks within each task",
            "step_2": "center both rank vectors on their within-task means",
            "step_3": "concatenate tasks and compute the Pearson correlation",
            "step_4": "bootstrap resamples recompute ranks from scratch",
            "step_5": "zero variance after centering aborts with an explicit error",
        },
        "diagnostic_window": {
            "rollouts": "r00-r15",
            "applies_to": [
                "BCI",
                "candidate annotation",
                "matched-control closure endpoints",
                "predictor features (prompt length, output length, mean eligible entropy, mean commit confidence)",
            ],
            "policy": "ao_b32 held-out screening artifacts",
            "rationale": "mirrors the sixteen-rollout calibration structure; the BCI freeze defines the query BCI as the mean of sixteen rollout BCIs",
        },
        "formal_target": {
            "definition": "mean over k in {4,8,16,32,64} of unbiased Pass@k(64)",
            "advantage": "CoverageAUC(ar_b1) - CoverageAUC(ao_b32)",
            "requires": "screening plus primary-extension artifacts for both primary policies",
        },
        "rq1_aggregation": {
            "query": "unweighted mean of matched-pair entropy closure differences",
            "task": "unweighted mean over matched query endpoints",
            "pooled": "unweighted mean of the two task means (equal task weight)",
            "missingness": "zero-match queries are excluded and counted; below 90% matched queries per task the task endpoint is unresolved",
        },
        "mae_pooled_weighting": (
            "equal query weight across both tasks (each task contributes 100 "
            "queries, so equal query weight equals equal task weight)"
        ),
        "gate_rules": {
            "condition_1": (
                "pass iff no task unresolved AND (mean closure > 0 on both "
                "tasks OR some 95% task interval entirely above zero while no "
                "interval is entirely below zero)"
            ),
            "condition_2": (
                "pass iff pooled task-stratified Spearman 95% CI low > 0 AND "
                "pooled MAE_baseline - MAE_augmented 95% CI low > 0"
            ),
        },
        "malformed_sensitivity": {
            "threshold": MALFORMED_SENSITIVITY_THRESHOLD,
            "rule": (
                "if any task-policy malformed fraction exceeds the threshold, "
                "recompute the primary advantage on valid-parse rollouts, "
                "dropping k values above a query's valid count (descriptive)"
            ),
        },
        "verdict_taxonomy": {
            "applies_to": [
                "pooled BCI association 95% CI",
                "pooled MAE_baseline - MAE_augmented 95% CI",
            ],
            "entirely_above_zero": (
                "condition passes; the frozen gate boolean is unchanged"
            ),
            "straddles_zero": (
                "condition fails as unresolved; no evidence in either direction"
            ),
            "entirely_below_zero": (
                "condition fails AND is recorded as a resolved wrong-direction "
                "falsification, not a null; the paper must report it as such"
            ),
            "gate_booleans_unchanged": True,
        },
        "secondary_contrast_readings": {
            "contrast": (
                "ao_b32 - random_b32, matched block size 32, 16-rollout "
                "screening window"
            ),
            "entirely_below_zero": (
                "the cost of arbitrary-order decoding is attributable to "
                "confidence-guided position selection, not to order freedom"
            ),
            "straddles_zero": (
                "order freedom itself is implicated; confidence-guidance is "
                "not separately identified"
            ),
            "entirely_above_zero": (
                "confidence-guidance is protective; the flexibility-trap "
                "mechanism does not transfer to this setting"
            ),
            "resolution_note": (
                "these readings are bounded by the screening window's "
                "resolution (16 rollouts per policy; empirical CI half-widths "
                "recorded by the power-simulation mode)"
            ),
        },
        "negative_rq2_mechanistic_readings": {
            "context": (
                "pre-registered accounts for a resolved wrong-direction "
                "falsification (pooled BCI association CI entirely below zero)"
            ),
            "account_a_competence": (
                "closure measures decoder competence rather than avoidance: "
                "queries where the arbitrary-order decoder successfully "
                "resolves candidate uncertainty are queries it handles well, "
                "so AR has less to add; the negative sign is mechanically "
                "expected and BCI is a misnamed quantity rather than a "
                "failed one"
            ),
            "account_b_confound": (
                "closure and coverage advantage are both driven by query "
                "difficulty in opposite directions, so the association is "
                "confounded rather than informative"
            ),
            "distinguishing_evidence": (
                "account (a) predicts query-level BCI correlates positively "
                "with ao_b32 coverage itself (competence); account (b) "
                "predicts BCI tracks difficulty measures (prompt length, "
                "base accuracy) without a direct competence link; both are "
                "checked against the frozen secondary tables before either "
                "is asserted"
            ),
            "rule": (
                "the account is chosen by the pre-committed evidence above, "
                "not selected post hoc; if neither distinguishes, both are "
                "reported"
            ),
        },
        "unblinding_read_order": {
            "order": [
                "1. record the Gate 1 verdict (condition 1 and condition 2 blocks) and commit it",
                "2. read the dose-response table",
                "3. read the confidence-vs-random contrast (secondary readings apply)",
            ],
            "mechanics": (
                "the Markdown report renders Gate 1 before all secondary "
                "sections and the JSON is key-sorted so gate1 precedes the "
                "secondary tables; the analyze invocation is timestamped in "
                "the append-only invocation log"
            ),
        },
        "power_resolution": {
            "alpha": 0.05,
            "nominal_power": 0.80,
            "per_task_correlation_mde_n100": minimum_detectable_correlation(100),
            "pooled_correlation_mde_n200": minimum_detectable_correlation(200),
            "per_task_paired_effect_mde_n100": minimum_detectable_paired_effect(
                100
            ),
            "approximation_note": (
                "Fisher-z / normal planning approximations for single-sample "
                "correlations at n=100 and n=200; the frozen statistic is the "
                "task-stratified average-rank Spearman (average ranks within "
                "task, center, concatenate, Pearson). These MDEs are planning "
                "quantities, not replacements for query-clustered bootstrap "
                "intervals, and token counts never increase effective sample "
                "size. A seeded empirical resolution check of the actual "
                "recipe is emitted separately by the power-simulation mode."
            ),
        },
        "design_cardinality": _design_cardinality(manifest),
        "freeze_amendments": [
            {
                "superseded_analysis_freeze_sha256": (
                    SUPERSEDED_ANALYSIS_FREEZE_SHA256
                ),
                "superseded_analyzer_commit": SUPERSEDED_ANALYZER_COMMIT,
                "reason": (
                    "pre-unblinding amendment: pre-register the verdict "
                    "taxonomy for negative/unresolved CIs, bind the design "
                    "cardinality, and record the Fisher-z power/MDE caveats; "
                    "the Gate 1 boolean rules and the final BCI freeze are "
                    "unchanged"
                ),
                "recorded_at": "see created_at_utc of this freeze",
            },
            {
                "superseded_analysis_freeze_sha256": (
                    SECOND_SUPERSEDED_ANALYSIS_FREEZE_SHA256
                ),
                "superseded_analyzer_commit": SECOND_SUPERSEDED_ANALYZER_COMMIT,
                "reason": (
                    "instrumentation only: append-only invocation logging for "
                    "every analyzer mode, so the no-outcome-read claim and "
                    "the timing of the first correctness read are "
                    "machine-checkable; no frozen rule, endpoint, or Gate 1 "
                    "boolean changed"
                ),
                "recorded_at": "see created_at_utc of this freeze",
            },
            {
                "superseded_analysis_freeze_sha256": (
                    THIRD_SUPERSEDED_ANALYSIS_FREEZE_SHA256
                ),
                "superseded_analyzer_commit": THIRD_SUPERSEDED_ANALYZER_COMMIT,
                "reason": (
                    "pre-register the confidence-vs-random secondary-contrast "
                    "readings, the mechanistic accounts for a negative RQ2 "
                    "falsification, and the unblinding read order; no frozen "
                    "endpoint or Gate 1 boolean changed"
                ),
                "recorded_at": "see created_at_utc of this freeze",
            },
            {
                "superseded_analysis_freeze_sha256": (
                    FOURTH_SUPERSEDED_ANALYSIS_FREEZE_SHA256
                ),
                "superseded_analyzer_commit": FOURTH_SUPERSEDED_ANALYZER_COMMIT,
                "reason": (
                    "documentation only: render the amendment invariants "
                    "(final BCI freeze, Gate 1 boolean rules, decode "
                    "parameters, inference commit) explicitly in the Gate 1 "
                    "report; no frozen rule or endpoint changed"
                ),
                "recorded_at": "see created_at_utc of this freeze",
            },
        ],
        "tables": {
            "format": "one CSV in long format (table, row_key, field, value)",
            "tables": [
                "rq1_closure",
                "rq2_bci",
                "predictive_value",
                "policy_summary",
                "dose_response",
                "solved_membership",
                "gate1",
            ],
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
        },
    }


def _load_analysis_freeze(
    path: Path, *, analyzer_commit: str, expected_bci_freeze_sha256: str
) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FrozenRuleError(
            f"missing analysis freeze {path}; emit it before any held-out analysis"
        )
    freeze_sha = sha256_file(path)
    freeze = json.loads(path.read_text(encoding="utf-8"))
    if freeze.get("analysis_version") != ANALYSIS_VERSION:
        raise FrozenRuleError("analysis freeze version mismatch")
    if freeze.get("analyzer_commit") != analyzer_commit:
        raise FrozenRuleError(
            "analysis freeze is bound to analyzer commit "
            f"{freeze.get('analyzer_commit')}, but this checkout is at "
            f"{analyzer_commit}"
        )
    if freeze.get("bci_freeze_sha256") != expected_bci_freeze_sha256:
        raise FrozenRuleError("analysis freeze is bound to a different BCI freeze")
    missing_sections = [
        section
        for section in REQUIRED_ANALYSIS_FREEZE_SECTIONS
        if section not in freeze
    ]
    if missing_sections:
        raise FrozenRuleError(
            "analysis freeze lacks required pre-unblinding sections "
            f"{missing_sections}; re-emit it from the current analyzer commit "
            "before any held-out outcome is read"
        )
    return freeze, freeze_sha


# ---------------------------------------------------------------------------
# Resolution simulation (pre-unblinding planning documentation)
# ---------------------------------------------------------------------------


def empirical_resolution_note(
    *,
    seed: int = RESOLUTION_SIMULATION_SEED,
    replicates: int = RESOLUTION_SIMULATION_REPLICATES,
    rho: float = RESOLUTION_SIMULATION_RHO,
    n_per_task: int = 100,
) -> dict[str, Any]:
    """Empirical CI half-width of the frozen stratified recipe.

    Generates one latent bivariate-normal dataset per task and runs the exact
    frozen bootstrap (average ranks within task, center, concatenate, Pearson)
    on it. This is pre-unblinding planning documentation: it does not touch
    held-out data, is not a frozen endpoint, and cannot modify Gate 1.
    """
    rng = np.random.RandomState(seed)
    datasets: dict[str, list[tuple[float, float]]] = {}
    for task in TASKS:
        x = rng.normal(size=n_per_task)
        z = rng.normal(size=n_per_task)
        y = rho * x + math.sqrt(max(1.0 - rho * rho, 0.0)) * z
        datasets[task] = [
            (float(xi), float(yi)) for xi, yi in zip(x, y, strict=True)
        ]

    single: dict[str, Any] = {}
    for task in TASKS:
        ci = task_stratified_bootstrap(
            label=f"resolution|single|{task}",
            tasks={task: datasets[task]},
            statistic=stratified_spearman,
        )
        single[task] = {
            "point": ci["point"],
            "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"],
            "half_width": (ci["ci_high"] - ci["ci_low"]) / 2.0,
        }
    pooled_ci = task_stratified_bootstrap(
        label="resolution|pooled",
        tasks=datasets,
        statistic=stratified_spearman,
    )
    return {
        "simulation_seed": seed,
        "replicates": replicates,
        "latent_rho": rho,
        "n_per_task": n_per_task,
        "single_task": single,
        "pooled": {
            "point": pooled_ci["point"],
            "ci_low": pooled_ci["ci_low"],
            "ci_high": pooled_ci["ci_high"],
            "half_width": (pooled_ci["ci_high"] - pooled_ci["ci_low"]) / 2.0,
        },
        "fisher_z_planning_mde": {
            "per_task_n100": minimum_detectable_correlation(100),
            "pooled_n200": minimum_detectable_correlation(200),
        },
        "status": (
            "pre-unblinding planning documentation; not a frozen endpoint and "
            "cannot modify Gate 1"
        ),
        "note": (
            "Half-widths are for the frozen average-rank stratified statistic "
            "under a bivariate-normal copula. The Fisher-z MDEs are "
            "single-sample approximations, not calibrated to the stratified "
            "recipe; the empirical half-widths here measure the actual "
            "recipe's resolution at the stated latent correlation."
        ),
    }


def render_resolution_note_markdown(note: dict[str, Any]) -> str:
    lines = [
        "# Frozen Recipe Resolution Check (Pre-Unblinding Planning Note)",
        "",
        f"- Simulation seed: `{note['simulation_seed']}`",
        f"- Replicates: {note['replicates']:,}",
        f"- Latent bivariate-normal correlation: {note['latent_rho']}",
        f"- Queries per task: {note['n_per_task']}",
        f"- Status: {note['status']}",
        "",
        "## Empirical half-widths of the frozen stratified bootstrap CI",
        "",
        "| scope | point | CI low | CI high | half-width |",
        "|---|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        entry = note["single_task"][task]
        lines.append(
            f"| {task} (n=100) | {entry['point']:.4f} | {entry['ci_low']:.4f} | "
            f"{entry['ci_high']:.4f} | {entry['half_width']:.4f} |"
        )
    pooled = note["pooled"]
    lines.append(
        f"| pooled stratified recipe | {pooled['point']:.4f} | "
        f"{pooled['ci_low']:.4f} | {pooled['ci_high']:.4f} | "
        f"{pooled['half_width']:.4f} |"
    )
    lines.extend(
        [
            "",
            "## Fisher-z planning MDEs (for comparison)",
            "",
            f"- Per-task n=100: {note['fisher_z_planning_mde']['per_task_n100']:.3f}",
            f"- Pooled n=200 approximation: "
            f"{note['fisher_z_planning_mde']['pooled_n200']:.3f}",
            "",
            note["note"],
            "",
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def analyze(
    *,
    phase1_root: Path,
    split_manifest: Path,
    artifact_schema: Path,
    freeze_path: Path,
    analysis_freeze_path: Path,
    repo_root: Path,
    expected_freeze_sha256: str,
) -> dict[str, Any]:
    manifest = load_split_manifest(split_manifest)
    freeze = _load_freeze(freeze_path, expected_freeze_sha256)
    analyzer_commit = _repo_commit(repo_root)
    analysis_freeze, analysis_freeze_sha = _load_analysis_freeze(
        analysis_freeze_path,
        analyzer_commit=analyzer_commit,
        expected_bci_freeze_sha256=expected_freeze_sha256,
    )
    cardinality = _design_cardinality(manifest)
    if cardinality != analysis_freeze["design_cardinality"]:
        raise FrozenRuleError(
            "split-manifest design cardinality disagrees with the analysis "
            "freeze; the freeze must be re-emitted from this analyzer commit"
        )
    record_validator = HeldoutRecordValidator(
        manifest=manifest,
        freeze_sha256=expected_freeze_sha256,
        schema_path=artifact_schema,
    )
    validation = _validate_artifacts(
        phase1_root=phase1_root,
        manifest=manifest,
        record_validator=record_validator,
        allow_partial=False,
    )
    screening_shards = _validate_shard_manifests(
        phase1_root, run_kind="screening", num_shards=8
    )
    extension_shards = _validate_shard_manifests(
        phase1_root, run_kind="primary-extension", num_shards=8
    )
    if screening_shards["total_result_count"] != 16_000:
        raise FrozenRuleError(
            f"screening shard markers count {screening_shards['total_result_count']} != 16000"
        )
    if extension_shards["total_result_count"] != 19_200:
        raise FrozenRuleError(
            f"extension shard markers count {extension_shards['total_result_count']} != 19200"
        )

    records = _load_records(phase1_root, manifest, record_validator=record_validator)
    token_rows = _diagnostic_token_rows(phase1_root, manifest)
    task_parameters = freeze["candidate_boundary_rule"]["task_parameters"]
    _annotate_candidates(token_rows, task_parameters)
    matches = _match_candidates(token_rows, task_parameters)
    expected_queries = {
        (task, int(entry["row_index"]))
        for task in TASKS
        for entry in manifest["splits"][task]["heldout"]
    }
    bci_by_query = _query_bci(token_rows, task_parameters, expected_queries)

    per_query: dict[tuple[str, int], dict[str, float]] = {}
    for task in TASKS:
        queries = sorted(
            entry["row_index"] for entry in manifest["splits"][task]["heldout"]
        )
        for query in queries:
            ar = records[(task, query, "ar_b1", "screening")] + records[
                (task, query, "ar_b1", "primary-extension")
            ]
            ao = records[(task, query, "ao_b32", "screening")] + records[
                (task, query, "ao_b32", "primary-extension")
            ]
            diagnostic_rows = [
                row
                for row in token_rows
                if row["task"] == task
                and row["query"] == query
                and row["rollout"] in DIAGNOSTIC_ROLLOUTS
            ]
            if not diagnostic_rows:
                raise FrozenRuleError(
                    f"no diagnostic tokens for {task} query {query}"
                )
            prompt_lengths = {
                int(row["prompt_length"])
                for row in diagnostic_rows
                if row["rollout"] == 0
            }
            if len(prompt_lengths) != 1:
                raise FrozenRuleError(
                    f"inconsistent prompt length for {task} query {query}"
                )
            ao_screening = records[(task, query, "ao_b32", "screening")]
            features = {
                "task_math500": float(task == "math500"),
                "prompt_length": float(prompt_lengths.pop()),
                "output_length": float(
                    mean(
                        len(str(record["response"]).split())
                        for record in ao_screening
                    )
                ),
                "mean_eligible_entropy": float(
                    mean(row["eligible_entropy"] for row in diagnostic_rows)
                ),
                "mean_commit_confidence": float(
                    mean(row["commit_confidence"] for row in diagnostic_rows)
                ),
                "bci": float(bci_by_query[(task, query)]),
            }
            per_query[(task, query)] = {
                "features": features,
                "bci": features["bci"],
                "formal_advantage": _formal_coverage_advantage(ar, ao),
                "coverage_auc_ar": _coverage_auc(ar),
                "coverage_auc_ao": _coverage_auc(ao),
            }

    closure = _closure_endpoints(matches)
    closure_gate = _closure_gate(closure["by_task"])
    association = _bci_association(per_query)
    predictive = _predictive_value(
        per_query,
        freeze["baseline_predictor"],
        freeze["augmented_predictor"],
    )
    policy_summaries = _policy_summaries(records, manifest)
    dose_response = _dose_response(records, manifest)
    solved_membership = _solved_membership(records, manifest)
    sensitivity = _malformed_sensitivity(records, per_query, policy_summaries)
    random_contrast = _confidence_vs_random_contrast(records, manifest)

    gate_pass = (
        closure_gate["passed"]
        and association["pooled_ci_above_zero"]
        and predictive["pooled_delta_ci_above_zero"]
    )
    report: dict[str, Any] = {
        "analysis_version": ANALYSIS_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bci_freeze_sha256": expected_freeze_sha256,
        "analysis_freeze_sha256": analysis_freeze_sha,
        "analyzer_commit": analyzer_commit,
        "runner_commit": EXPECTED_RUNNER_COMMIT,
        "split_manifest_sha256": SPLIT_MANIFEST_SHA256,
        "artifact_validation": validation,
        "shard_manifests": {"screening": screening_shards, "primary_extension": extension_shards},
        "rq1_closure": closure,
        "rq2_bci_association": association,
        "predictive_value": predictive,
        "policy_summaries": policy_summaries,
        "dose_response": dose_response,
        "solved_membership": solved_membership,
        "malformed_sensitivity": sensitivity,
        "confidence_vs_random_contrast": random_contrast,
        "design_cardinality": cardinality,
        "power_resolution": analysis_freeze["power_resolution"],
        "verdict_taxonomy": analysis_freeze["verdict_taxonomy"],
        "secondary_contrast_readings": analysis_freeze[
            "secondary_contrast_readings"
        ],
        "negative_rq2_mechanistic_readings": analysis_freeze[
            "negative_rq2_mechanistic_readings"
        ],
        "unblinding_read_order": analysis_freeze["unblinding_read_order"],
        "freeze_amendments": analysis_freeze["freeze_amendments"],
        "per_query": {
            f"{task}|q{query:04d}": {
                "bci": row["bci"],
                "formal_advantage": row["formal_advantage"],
                "coverage_auc_ar": row["coverage_auc_ar"],
                "coverage_auc_ao": row["coverage_auc_ao"],
            }
            for (task, query), row in sorted(per_query.items())
        },
        "gate1": {
            "condition_1_closure": closure_gate,
            "condition_2_bci": {
                "pooled_ci_above_zero": association["pooled_ci_above_zero"],
                "pooled": association["pooled"],
                "pooled_ci_classification": _classify_ci(
                    association["pooled"]["ci_low"],
                    association["pooled"]["ci_high"],
                ),
                "mae_delta": predictive["pooled"],
                "pooled_delta_ci_above_zero": predictive[
                    "pooled_delta_ci_above_zero"
                ],
                "mae_delta_ci_classification": _classify_ci(
                    predictive["pooled"]["ci_low"],
                    predictive["pooled"]["ci_high"],
                ),
            },
            "overall_pass": gate_pass,
        },
    }
    return report


def build_tables(report: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rq1 = []
    for task in TASKS:
        entry = report["rq1_closure"]["by_task"][task]
        rq1.append(
            {
                "task": task,
                "matched_queries": entry["matched_queries"],
                "matched_fraction": round(entry["matched_fraction"], 4),
                "unresolved": entry["rq1_unresolved"],
                "pairs": entry["matched_pair_count"],
                "entropy_closure_mean": round(entry["entropy_closure"]["point"], 6),
                "entropy_closure_ci_low": round(entry["entropy_closure"]["ci_low"], 6),
                "entropy_closure_ci_high": round(entry["entropy_closure"]["ci_high"], 6),
                "margin_closure_mean": round(entry["margin_closure"]["point"], 6),
                "margin_closure_ci_low": round(entry["margin_closure"]["ci_low"], 6),
                "margin_closure_ci_high": round(entry["margin_closure"]["ci_high"], 6),
            }
        )
    rq2 = [
        {
            "task": task,
            "rho": round(report["rq2_bci_association"]["per_task"][task]["point"], 6),
            "ci_low": round(report["rq2_bci_association"]["per_task"][task]["ci_low"], 6),
            "ci_high": round(report["rq2_bci_association"]["per_task"][task]["ci_high"], 6),
        }
        for task in TASKS
    ]
    rq2.append(
        {
            "task": "pooled",
            "rho": round(report["rq2_bci_association"]["pooled"]["point"], 6),
            "ci_low": round(report["rq2_bci_association"]["pooled"]["ci_low"], 6),
            "ci_high": round(report["rq2_bci_association"]["pooled"]["ci_high"], 6),
        }
    )
    predictive = [
        {
            "task": task,
            "mae_baseline": round(report["predictive_value"]["per_task"][task]["mae_baseline"], 6),
            "mae_augmented": round(report["predictive_value"]["per_task"][task]["mae_augmented"], 6),
            "delta": round(report["predictive_value"]["per_task"][task]["point"], 6),
            "delta_ci_low": round(report["predictive_value"]["per_task"][task]["ci_low"], 6),
            "delta_ci_high": round(report["predictive_value"]["per_task"][task]["ci_high"], 6),
        }
        for task in TASKS
    ]
    predictive.append(
        {
            "task": "pooled",
            "mae_baseline": round(report["predictive_value"]["pooled_mae_baseline"], 6),
            "mae_augmented": round(report["predictive_value"]["pooled_mae_augmented"], 6),
            "delta": round(report["predictive_value"]["pooled"]["point"], 6),
            "delta_ci_low": round(report["predictive_value"]["pooled"]["ci_low"], 6),
            "delta_ci_high": round(report["predictive_value"]["pooled"]["ci_high"], 6),
        }
    )
    gate = report["gate1"]
    gate_rows = [
        {"condition": "condition_1_closure", "passed": gate["condition_1_closure"]["passed"]},
        {
            "condition": "condition_2a_pooled_spearman_ci_above_zero",
            "passed": gate["condition_2_bci"]["pooled_ci_above_zero"],
        },
        {
            "condition": "condition_2a_pooled_spearman_ci_classification",
            "passed": gate["condition_2_bci"]["pooled_ci_classification"],
        },
        {
            "condition": "condition_2b_pooled_mae_delta_ci_above_zero",
            "passed": gate["condition_2_bci"]["pooled_delta_ci_above_zero"],
        },
        {
            "condition": "condition_2b_pooled_mae_delta_ci_classification",
            "passed": gate["condition_2_bci"]["mae_delta_ci_classification"],
        },
        {"condition": "gate1_overall", "passed": gate["overall_pass"]},
    ]
    contrast = report["confidence_vs_random_contrast"]
    contrast_rows = [
        {
            "task": task,
            "point": round(contrast["per_task"][task]["point"], 6),
            "ci_low": round(contrast["per_task"][task]["ci_low"], 6),
            "ci_high": round(contrast["per_task"][task]["ci_high"], 6),
            "per_k": json.dumps(
                contrast["per_task"][task]["per_k"], sort_keys=True
            ),
        }
        for task in TASKS
    ]
    contrast_rows.append(
        {
            "task": "pooled",
            "point": round(contrast["pooled"]["point"], 6),
            "ci_low": round(contrast["pooled"]["ci_low"], 6),
            "ci_high": round(contrast["pooled"]["ci_high"], 6),
            "per_k": json.dumps(contrast["pooled_per_k"], sort_keys=True),
        }
    )
    return {
        "rq1_closure": rq1,
        "rq2_bci": rq2,
        "predictive_value": predictive,
        "policy_summary": report["policy_summaries"],
        "dose_response": report["dose_response"],
        "solved_membership": report["solved_membership"],
        "gate1": gate_rows,
        "confidence_vs_random": contrast_rows,
        "design_cardinality": [report["design_cardinality"]],
        "power_resolution": [report["power_resolution"]],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--artifact-schema", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument(
        "--expected-freeze-sha256",
        default=EXPECTED_FREEZE_SHA256,
        help="SHA-256 of the final BCI freeze (identity check)",
    )
    parser.add_argument("--phase1-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "emit-freeze",
            "validate-only",
            "analyze",
            "audit-malformed",
            "power-simulation",
        ),
        required=True,
    )
    parser.add_argument("--analysis-freeze", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output-validation", type=Path)
    parser.add_argument("--output-report-json", type=Path)
    parser.add_argument("--output-report-md", type=Path)
    parser.add_argument("--output-tables-csv", type=Path)
    parser.add_argument("--output-audit-json", type=Path)
    parser.add_argument("--output-audit-md", type=Path)
    parser.add_argument("--output-simulation-json", type=Path)
    parser.add_argument("--output-simulation-md", type=Path)
    parser.add_argument("--invocation-log", type=Path)
    return parser.parse_args(argv)


def _invocation_output_paths(args: argparse.Namespace) -> dict[str, str]:
    fields = (
        "analysis_freeze",
        "output_validation",
        "output_report_json",
        "output_report_md",
        "output_tables_csv",
        "output_audit_json",
        "output_audit_md",
        "output_simulation_json",
        "output_simulation_md",
    )
    return {
        field: str(getattr(args, field))
        for field in fields
        if getattr(args, field) is not None
    }


def _append_invocation_log(
    path: Path,
    *,
    mode: str,
    analyzer_commit: str,
    bci_freeze_sha256: str,
    reads_heldout_correctness: bool,
    output_paths: dict[str, str],
) -> None:
    """Append one JSONL record per invocation attempt; append-only by use.

    Written before the mode body runs, so an aborted invocation still leaves
    a record. Only ``analyze`` mode sets ``reads_heldout_correctness``.
    """
    record = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "analyzer_commit": analyzer_commit,
        "bci_freeze_sha256": bci_freeze_sha256,
        "reads_heldout_correctness": reads_heldout_correctness,
        "output_paths": output_paths,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_split_manifest(args.split_manifest)
    freeze = _load_freeze(args.freeze, args.expected_freeze_sha256)
    record_validator = HeldoutRecordValidator(
        manifest=manifest,
        freeze_sha256=args.expected_freeze_sha256,
        schema_path=args.artifact_schema,
    )
    if args.invocation_log is not None:
        _append_invocation_log(
            args.invocation_log,
            mode=args.mode,
            analyzer_commit=_repo_commit(args.repo_root),
            bci_freeze_sha256=args.expected_freeze_sha256,
            reads_heldout_correctness=(args.mode == "analyze"),
            output_paths=_invocation_output_paths(args),
        )

    if args.mode == "emit-freeze":
        if args.analysis_freeze.exists():
            raise RuntimeError(
                f"refusing to overwrite frozen output: {args.analysis_freeze}"
            )
        payload = build_analysis_freeze(
            analyzer_commit=_repo_commit(args.repo_root),
            bci_freeze_sha256=args.expected_freeze_sha256,
            manifest=manifest,
        )
        write_json_atomic(args.analysis_freeze, payload)
        print(
            json.dumps(
                {
                    "analysis_freeze": str(args.analysis_freeze),
                    "sha256": sha256_file(args.analysis_freeze),
                    "analyzer_commit": payload["analyzer_commit"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.mode == "audit-malformed":
        audit = audit_screening_malformed(
            phase1_root=args.phase1_root,
            manifest=manifest,
            record_validator=record_validator,
        )
        for path in (args.output_audit_json, args.output_audit_md):
            if path is not None and path.exists():
                raise RuntimeError(f"refusing to overwrite: {path}")
        if args.output_audit_json is not None:
            write_json_atomic(args.output_audit_json, audit)
        if args.output_audit_md is not None:
            write_text_atomic(
                args.output_audit_md, render_malformed_audit_markdown(audit)
            )
        print(
            json.dumps(
                {
                    "audit_version": audit["audit_version"],
                    "counts": audit["counts"],
                    "triggered": audit["triggered"],
                    "correctness_read": audit["correctness_read"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.mode == "power-simulation":
        note = empirical_resolution_note()
        for path in (args.output_simulation_json, args.output_simulation_md):
            if path is not None and path.exists():
                raise RuntimeError(f"refusing to overwrite: {path}")
        if args.output_simulation_json is not None:
            write_json_atomic(args.output_simulation_json, note)
        if args.output_simulation_md is not None:
            write_text_atomic(
                args.output_simulation_md, render_resolution_note_markdown(note)
            )
        print(
            json.dumps(
                {
                    "single_task_half_widths": {
                        task: note["single_task"][task]["half_width"]
                        for task in TASKS
                    },
                    "pooled_half_width": note["pooled"]["half_width"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.mode == "validate-only":
        validation = _validate_artifacts(
            phase1_root=args.phase1_root,
            manifest=manifest,
            record_validator=record_validator,
            allow_partial=args.allow_partial,
        )
        shard_status: dict[str, Any] = {}
        for run_kind in ("screening", "primary-extension"):
            marker_dir = (
                args.phase1_root
                / PROTOCOL_VERSION
                / "manifests"
                / "heldout"
                / run_kind
                / "all"
            )
            shard_status[run_kind] = {
                "markers_present": len(list(marker_dir.glob("shard-*.json")))
                if marker_dir.is_dir()
                else 0
            }
        summary = {
            "validated_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_validation": validation,
            "shard_marker_counts": shard_status,
            "outcome_fields_read": False,
        }
        if args.output_validation is not None:
            if args.output_validation.exists():
                raise RuntimeError(
                    f"refusing to overwrite: {args.output_validation}"
                )
            write_json_atomic(args.output_validation, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    report = analyze(
        phase1_root=args.phase1_root,
        split_manifest=args.split_manifest,
        artifact_schema=args.artifact_schema,
        freeze_path=args.freeze,
        analysis_freeze_path=args.analysis_freeze,
        repo_root=args.repo_root,
        expected_freeze_sha256=args.expected_freeze_sha256,
    )
    for path in (args.output_report_json, args.output_report_md, args.output_tables_csv):
        if path is not None and path.exists():
            raise RuntimeError(f"refusing to overwrite frozen output: {path}")
    if args.output_report_json is not None:
        write_json_atomic(args.output_report_json, report)
    if args.output_report_md is not None:
        write_text_atomic(args.output_report_md, render_markdown(report))
    if args.output_tables_csv is not None:
        write_text_atomic(args.output_tables_csv, _flatten_rows(build_tables(report)))
    print(
        json.dumps(
            {
                "gate1_overall_pass": report["gate1"]["overall_pass"],
                "rq1_closure_by_task": {
                    task: {
                        "mean": report["rq1_closure"]["by_task"][task]["entropy_closure"]["point"],
                        "ci": [
                            report["rq1_closure"]["by_task"][task]["entropy_closure"]["ci_low"],
                            report["rq1_closure"]["by_task"][task]["entropy_closure"]["ci_high"],
                        ],
                        "unresolved": report["rq1_closure"]["by_task"][task]["rq1_unresolved"],
                    }
                    for task in TASKS
                },
                "rq2_pooled_spearman": {
                    "rho": report["rq2_bci_association"]["pooled"]["point"],
                    "ci": [
                        report["rq2_bci_association"]["pooled"]["ci_low"],
                        report["rq2_bci_association"]["pooled"]["ci_high"],
                    ],
                },
                "mae_delta_pooled": {
                    "delta": report["predictive_value"]["pooled"]["point"],
                    "ci": [
                        report["predictive_value"]["pooled"]["ci_low"],
                        report["predictive_value"]["pooled"]["ci_high"],
                    ],
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
