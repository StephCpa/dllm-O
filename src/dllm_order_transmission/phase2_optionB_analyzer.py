"""Phase 2 Option B analyzer: block-size dose-response freeze and analysis.

Three mutually exclusive modes:

- ``emit-freeze`` writes the pre-registration ``phase2_optionB_freeze.json``
  before any Phase 2 outcome is read.
- ``validate-only`` checks provenance, schemas, hashes, and completeness of the
  Phase 2 extension artifacts without touching outcome fields.
- ``analyze`` computes P1 (B=32 minus B=1 CoverageAUC contrast), P2 (log2-coded
  trend statistic with ordinal sensitivity), the pre-registered combination
  table, and the Condition A gate.

Reuses Phase 1 held-out artifacts for ``ar_b1``/``ao_b32`` (64 rollouts) and
screening r00-r15 for all four policies. New Phase 2 artifacts are
``ao_b8``/``ao_b16`` extension rollouts r16-r63 under run kind
``phase2-extension``, stamped with the Phase 2 freeze SHA.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from jsonschema import Draft202012Validator

from .artifacts import sha256_file, write_json_atomic, write_text_atomic
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
    PHASE2_ARTIFACT_SCHEMA_SHA256,
    PHASE2_POLICY_NAMES,
    SPLIT_MANIFEST_SHA256,
    load_split_manifest,
)

FREEZE_VERSION = "phase2-optionb-freeze-v1"
ANALYSIS_REPAIR_VERSION = "phase2-analysis-repair-v1"
ANALYSIS_REPAIR_SCOPE = ["initialize_zero_success_cells"]
SUPERSEDED_PHASE2_FREEZE_SHA256 = (
    "6c97f7a82bd97d2b9b1d1e1da2b44729c9c980075090c944c6590119a805e40c"
)
EXPECTED_PHASE1_RUNNER_COMMIT = "eaf4b4f9214eec6066010fad7dba5c5319d85122"
PHASE1_FREEZE_CHAIN = (
    "3bbad351b42a7c540f23efdfea1ab1294e6e0acc18395ab30558ab19f408fac7",
    "5314d95336cc572f175a30aaef995169bb3d9149c86227ffb3d74090ca3be15c",
    "cd286131415092ae3f41b0ad6eebf9ebbdf1775e77dcbc508e98bbe183566615",
    "68fc9b3b859da4b05f44828527542934096534d6b43fe3a9204d3e420a3b37ca",
    "b18eb39dba335dd0d74e915e4562f57f4b8a4a678618774852c0ed2394ba83e4",
)
P2_DEMOTION_HALF_WIDTH = 0.009
P2_SIMULATION_HALF_WIDTH = 0.0050  # recorded 2026-09-01, see protocol §12

TASKS = ("gsm8k", "math500")
B_VALUES = (1, 8, 16, 32)
POLICY_BY_B = {1: "ar_b1", 8: "ao_b8", 16: "ao_b16", 32: "ao_b32"}
COVERAGE_K_GRID = (4, 8, 16, 32, 64)
SCREENING_ROLLOUTS = tuple(range(16))
EXTENSION_ROLLOUTS = tuple(range(16, 64))
PHASE2_NEW_POLICIES = ("ao_b8", "ao_b16")
REQUIRED_FREEZE_SECTIONS = (
    "primary_endpoints",
    "verdict_taxonomy",
    "condition_a_gate",
    "cardinality",
    "logging_spec",
    "amendment_chain",
)


def _repo_commit(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _expected_items(manifest: dict[str, Any]) -> list[tuple[str, int, str, str, int]]:
    """(task, row, policy, run_kind, rollout) for the full Phase 2 analysis set."""
    items: list[tuple[str, int, str, str, int]] = []
    for task in TASKS:
        rows = sorted(
            entry["row_index"] for entry in manifest["splits"][task]["heldout"]
        )
        for row in rows:
            for policy in ("ar_b1", "ao_b32"):
                for rollout in range(64):
                    run_kind = (
                        "screening" if rollout < 16 else "primary-extension"
                    )
                    items.append((task, row, policy, run_kind, rollout))
            for policy in PHASE2_NEW_POLICIES:
                for rollout in SCREENING_ROLLOUTS:
                    items.append((task, row, policy, "screening", rollout))
                for rollout in EXTENSION_ROLLOUTS:
                    items.append(
                        (task, row, policy, "phase2-extension", rollout)
                    )
    return items


def _result_path(phase1_root: Path, item: tuple[str, int, str, str, int]) -> Path:
    task, row, policy, run_kind, rollout = item
    return (
        phase1_root
        / "phase1-v1"
        / "heldout"
        / run_kind
        / task
        / policy
        / f"q{row:04d}"
        / f"r{rollout:02d}.json"
    )


class Phase2RecordValidator:
    """Provenance checks for Phase 2 records; never reads outcome fields."""

    def __init__(
        self,
        *,
        manifest: dict[str, Any],
        phase2_freeze_sha256: str,
        phase1_schema_path: Path,
        phase2_schema_path: Path,
        phase2_runner_commit: str,
    ) -> None:
        self.phase1_validator = Draft202012Validator(
            json.loads(phase1_schema_path.read_text(encoding="utf-8"))
        )
        self.phase2_validator = Draft202012Validator(
            json.loads(phase2_schema_path.read_text(encoding="utf-8"))
        )
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
        self.phase2_freeze_sha256 = phase2_freeze_sha256
        self.phase2_runner_commit = phase2_runner_commit

    def validate(
        self, record: dict[str, Any], *, item: tuple[str, int, str, str, int]
    ) -> None:
        task, row, policy, run_kind, rollout = item
        if run_kind == "phase2-extension":
            self.phase2_validator.validate(record)
            expected_runner = self.phase2_runner_commit
            expected_freeze = self.phase2_freeze_sha256
        else:
            self.phase1_validator.validate(record)
            expected_runner = EXPECTED_PHASE1_RUNNER_COMMIT
            expected_freeze = None  # screening records carry the BCI freeze SHA
        checks = (
            record["run_kind"] == run_kind,
            record["policy"]["name"] == policy,
            int(record["rollout_index"]) == rollout,
            record["split_manifest_sha256"] == SPLIT_MANIFEST_SHA256,
            record["external_repository_commit"]
            == "1a2fddb5c6655597e63081c0af5ebb718a849f39",
            record["runner_repository"]["commit"] == expected_runner,
            record["runner_repository"]["dirty"] is False,
            record["source"]["split_role"] == "heldout",
            int(record["source"]["row_index"]) == row,
            record["source"]["prompt_sha256"] == self.prompt_hashes[(task, row)],
            record["source"]["dataset_id"] == self.dataset[task][0],
            record["source"]["dataset_revision"] == self.dataset[task][1],
            record["source"]["dataset_split"] == self.dataset[task][2],
        )
        if run_kind == "phase2-extension":
            checks += (record.get("bci_freeze_sha256") == expected_freeze,)
        if not all(checks):
            raise FrozenRuleError(f"provenance mismatch for {item}")

    def trace_hash_matches(self, result_path: Path, record: dict[str, Any]) -> bool:
        trace_path = result_path.with_name(result_path.stem + ".trace.jsonl.gz")
        if not trace_path.is_file():
            return False
        return sha256_file(trace_path) == record["trace_sha256"]


def _validate_phase2_artifacts(
    *, phase1_root: Path, manifest: dict[str, Any], validator: Phase2RecordValidator
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
            validator.validate(record, item=item)
            if not validator.trace_hash_matches(path, record):
                raise FrozenRuleError(f"trace hash mismatch for {item}")
        except FrozenRuleError as error:
            invalid.append(str(error))
            continue
        present += 1
    return {
        "expected": len(items),
        "present_and_valid": present,
        "missing": len(missing),
        "invalid": len(invalid),
        "first_missing": missing[:5],
        "first_invalid": invalid[:5],
    }


def _validate_phase2_markers(phase1_root: Path) -> dict[str, Any]:
    total = 0
    for shard in range(8):
        path = (
            phase1_root
            / "phase1-v1"
            / "manifests"
            / "heldout"
            / "phase2-extension"
            / "all"
            / f"shard-{shard:03d}-of-008.json"
        )
        if not path.is_file():
            raise FrozenRuleError(f"missing phase2 shard marker: {path}")
        marker = json.loads(path.read_text(encoding="utf-8"))
        if marker.get("expected_keys_sha256") != marker.get("observed_keys_sha256"):
            raise FrozenRuleError(f"phase2 marker key mismatch: {path}")
        if int(marker["result_count"]) != 2400:
            raise FrozenRuleError(f"phase2 marker count mismatch: {path}")
        total += int(marker["result_count"])
    return {"markers": 8, "total_result_count": total}


def _coverage_auc(successes: int, n: int) -> float:
    return mean(estimate_pass_at_k(n, successes, k) for k in COVERAGE_K_GRID)


def _beta_log2(c1: float, c8: float, c16: float, c32: float) -> float:
    return (-3 * c1 + 0 * c8 + 1 * c16 + 2 * c32) / 14


def _beta_ordinal(c1: float, c8: float, c16: float, c32: float) -> float:
    return (-3 * c1 - 1 * c8 + 1 * c16 + 3 * c32) / 20


def build_freeze_payload(
    *, analyzer_commit: str, runner_commit: str
) -> dict[str, Any]:
    return {
        "protocol_version": FREEZE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analyzer_commit": analyzer_commit,
        "runner_commit": runner_commit,
        "split_manifest_sha256": SPLIT_MANIFEST_SHA256,
        "bci_freeze_sha256": BCI_FREEZE_SHA256,
        "phase1_freeze_chain": list(PHASE1_FREEZE_CHAIN),
        "phase2_artifact_schema_sha256": PHASE2_ARTIFACT_SCHEMA_SHA256,
        "revision_history": {
            "supersedes_phase2_freeze_sha256": SUPERSEDED_PHASE2_FREEZE_SHA256,
            "reason": (
                "pre-launch mechanical schema correction: the generation-result "
                "schema already admitted phase2-extension, but the shard-completion "
                "branch accidentally omitted the same run kind"
            ),
            "phase2_outcomes_read_before_revision": False,
            "statistical_design_changed": False,
        },
        "phase2_policies": list(PHASE2_NEW_POLICIES),
        "rollout_budget": 64,
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "interval": "percentile 95%",
            "scheme": "task-stratified query bootstrap, as Phase 1",
        },
        "primary_endpoints": {
            "coverage_auc": (
                "mean over k in {4,8,16,32,64} of unbiased Pass@k at 64 rollouts"
            ),
            "p1": "paired contrast delta_q = CoverageAUC_q(B=32) - CoverageAUC_q(B=1); pooled and per-task mean over queries",
            "p2": (
                "per-query OLS slope of CoverageAUC on x=log2(B), x in {0,3,4,5}; "
                "beta_q = (-3*C1 + 0*C8 + 1*C16 + 2*C32)/14; pooled and per-task "
                "mean; ordinal-coding sensitivity (-3,-1,1,3)/20 with the "
                "disagreement rule (resolved-sign mismatch between codings => "
                "unresolved trend)"
            ),
            "p2_demotion": (
                f"decided at freeze: simulated log2 half-width "
                f"{P2_SIMULATION_HALF_WIDTH} < {P2_DEMOTION_HALF_WIDTH} "
                "demotion threshold, so P2 remains primary (recorded 2026-09-01)"
            ),
        },
        "verdict_taxonomy": {
            "p1": {
                "entirely_below_zero": (
                    "order-freedom cost confirmed at the endpoints of the "
                    "block-size axis"
                ),
                "straddles_zero": (
                    "no resolved endpoint effect; contradicts the Phase 1 "
                    "formal comparison; reported as an internal replication "
                    "failure with both estimates shown"
                ),
                "entirely_above_zero": (
                    "reversal; reported as-is; no mechanistic account is "
                    "pre-registered"
                ),
            },
            "p2": {
                "resolved_negative_both_codings": (
                    "monotone gradient resolved: cost scales with the scope "
                    "of order freedom"
                ),
                "straddles_or_disagrees": (
                    "no resolved gradient at this resolution"
                ),
                "resolved_positive_both_codings": "reversal; reported as-is",
            },
        },
        "condition_a_gate": {
            "rows": [
                {
                    "p1": "resolved_negative",
                    "p2": "resolved_negative",
                    "reading": "continuous freedom-cost gradient",
                    "run_condition_a": True,
                },
                {
                    "p1": "resolved_negative",
                    "p2": "unresolved",
                    "reading": (
                        "endpoints differ without a resolved smooth gradient: "
                        "discrete sequential-vs-block-parallel distinction"
                    ),
                    "run_condition_a": False,
                },
                {
                    "p1": "unresolved",
                    "p2": "unresolved",
                    "reading": (
                        "flat; paper rests on the diversity-precision account "
                        "plus resolved nulls with stated detection bounds"
                    ),
                    "run_condition_a": False,
                },
                {
                    "p1": "unresolved",
                    "p2": "resolved",
                    "reading": (
                        "internally inconsistent; both reported; no mechanistic "
                        "claim"
                    ),
                    "run_condition_a": False,
                },
            ],
            "rule": (
                "Condition A (entropy-first intervention) proceeds only on the "
                "first row; any other row blocks A"
            ),
        },
        "cardinality": {
            "new_generations": 19_200,
            "analysis_set": 51_200,
            "reused": 32_000,
            "shards": 8,
            "per_shard": 2_400,
        },
        "logging_spec": {
            "addition": (
                "top64_token_ids, top64_logits, full_logsumexp, full_entropy "
                "per unmasked position per step; makes the offline N_eff "
                "feasibility analysis exact; out of scope for every Phase 2 "
                "endpoint"
            ),
            "throughput_gate": (
                "smoke run measures logging overhead; above 10% of baseline "
                "generation rate, m is reduced to 32 (recorded before launch)"
            ),
        },
        "amendment_chain": list(PHASE1_FREEZE_CHAIN),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--phase1-artifact-schema", type=Path, required=True)
    parser.add_argument("--phase2-artifact-schema", type=Path, required=True)
    parser.add_argument("--phase1-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--phase2-freeze", type=Path, required=True)
    parser.add_argument("--analysis-repair-amendment", type=Path)
    parser.add_argument(
        "--expected-bci-freeze-sha256", default=BCI_FREEZE_SHA256
    )
    parser.add_argument("--expected-phase2-runner-commit", default=None)
    parser.add_argument(
        "--mode",
        choices=("emit-freeze", "validate-only", "analyze"),
        required=True,
    )
    parser.add_argument("--invocation-log", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output-validation", type=Path)
    parser.add_argument("--output-report-json", type=Path)
    parser.add_argument("--output-report-md", type=Path)
    parser.add_argument("--output-tables-csv", type=Path)
    return parser.parse_args(argv)


def _append_invocation_log(
    path: Path,
    *,
    mode: str,
    analyzer_commit: str,
    phase2_freeze_sha256: str,
    reads_heldout_correctness: bool,
    analysis_repair_amendment_sha256: str | None = None,
) -> None:
    record = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "analyzer_commit": analyzer_commit,
        "phase2_freeze_sha256": phase2_freeze_sha256,
        "reads_heldout_correctness": reads_heldout_correctness,
    }
    if analysis_repair_amendment_sha256 is not None:
        record["analysis_repair_amendment_sha256"] = (
            analysis_repair_amendment_sha256
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validate_analysis_repair_amendment(
    path: Path,
    *,
    phase2_freeze_sha256: str,
    frozen_analyzer_commit: str,
    repair_analyzer_commit: str,
) -> str:
    if not path.is_file():
        raise FrozenRuleError("missing analysis repair amendment")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol_version": ANALYSIS_REPAIR_VERSION,
        "original_phase2_freeze_sha256": phase2_freeze_sha256,
        "original_analyzer_commit": frozen_analyzer_commit,
        "repair_analyzer_commit": repair_analyzer_commit,
        "repair_scope": ANALYSIS_REPAIR_SCOPE,
        "statistical_definitions_changed": False,
        "experimental_artifacts_changed": False,
        "outcome_based_decisions": False,
    }
    mismatches = {
        key: {"expected": expected, "observed": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise FrozenRuleError(
            "invalid analysis repair amendment: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )
    for key in ("failed_invocation_utc", "failure_console_sha256", "reason"):
        if not payload.get(key):
            raise FrozenRuleError(f"analysis repair amendment lacks {key}")
    return sha256_file(path)


def _load_phase2_freeze(
    path: Path,
    *,
    analyzer_commit: str,
    analysis_repair_amendment: Path | None = None,
) -> tuple[dict[str, Any], str, str | None]:
    if not path.is_file():
        raise FrozenRuleError("missing Phase 2 freeze; emit it before analysis")
    freeze_sha = sha256_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_version") != FREEZE_VERSION:
        raise FrozenRuleError("Phase 2 freeze version mismatch")
    repair_sha: str | None = None
    frozen_analyzer_commit = payload.get("analyzer_commit")
    if frozen_analyzer_commit != analyzer_commit:
        if analysis_repair_amendment is None:
            raise FrozenRuleError(
                "Phase 2 freeze is bound to a different analyzer commit"
            )
        repair_sha = _validate_analysis_repair_amendment(
            analysis_repair_amendment,
            phase2_freeze_sha256=freeze_sha,
            frozen_analyzer_commit=frozen_analyzer_commit,
            repair_analyzer_commit=analyzer_commit,
        )
    elif analysis_repair_amendment is not None:
        raise FrozenRuleError(
            "analysis repair amendment supplied without an analyzer-commit mismatch"
        )
    if payload.get("bci_freeze_sha256") != BCI_FREEZE_SHA256:
        raise FrozenRuleError("Phase 2 freeze references a different BCI freeze")
    missing = [
        s for s in REQUIRED_FREEZE_SECTIONS if s not in payload
    ]
    if missing:
        raise FrozenRuleError(f"Phase 2 freeze lacks sections {missing}")
    return payload, freeze_sha, repair_sha


def _load_successes(
    phase1_root: Path, manifest: dict[str, Any]
) -> dict[tuple[str, int, int], int]:
    """Per (task, query, B) number of correct rollouts out of 64 (single pass)."""
    successes: dict[tuple[str, int, int], int] = defaultdict(int)
    for task in TASKS:
        rows = sorted(
            entry["row_index"] for entry in manifest["splits"][task]["heldout"]
        )
        for row in rows:
            for b in B_VALUES:
                # A cell with no correct rollouts is an observed zero, not a
                # missing dictionary key.
                successes[(task, row, b)] = 0
                policy = POLICY_BY_B[b]
                for rollout in range(64):
                    run_kind = (
                        "screening"
                        if rollout < 16
                        else "phase2-extension"
                        if policy in PHASE2_NEW_POLICIES
                        else "primary-extension"
                    )
                    path = _result_path(
                        phase1_root, (task, row, policy, run_kind, rollout)
                    )
                    record = json.loads(path.read_text(encoding="utf-8"))
                    if bool(record["correct"]):
                        successes[(task, row, b)] += 1
    return dict(successes)


def _analyze(
    phase1_root: Path,
    manifest: dict[str, Any],
    freeze_payload: dict[str, Any],
    validator: Phase2RecordValidator,
) -> dict[str, Any]:
    validation = _validate_phase2_artifacts(
        phase1_root=phase1_root, manifest=manifest, validator=validator
    )
    if validation["missing"] or validation["invalid"]:
        raise FrozenRuleError(
            f"Phase 2 artifacts incomplete: {validation['missing']} missing, "
            f"{validation['invalid']} invalid"
        )
    markers = _validate_phase2_markers(phase1_root)
    if markers["total_result_count"] != 19_200:
        raise FrozenRuleError("Phase 2 marker total != 19200")
    successes = _load_successes(phase1_root, manifest)
    coverage = {
        key: _coverage_auc(count, 64) for key, count in successes.items()
    }

    def query_values(statistic) -> dict[str, list[float]]:
        return {
            task: [
                statistic(
                    *[
                        coverage[(task, row, b)]
                        for b in B_VALUES
                    ]
                )
                for row in sorted(
                    entry["row_index"]
                    for entry in manifest["splits"][task]["heldout"]
                )
            ]
            for task in TASKS
        }

    p1_values = {
        task: [
            coverage[(task, row, 32)] - coverage[(task, row, 1)]
            for row in sorted(
                entry["row_index"] for entry in manifest["splits"][task]["heldout"]
            )
        ]
        for task in TASKS
    }
    p1_pooled = task_stratified_bootstrap(
        label="phase2|p1|pooled",
        tasks=p1_values,
        statistic=lambda s: mean(v for vs in s.values() for v in vs),
    )
    p1_by_task = {
        task: task_stratified_bootstrap(
            label=f"phase2|p1|{task}",
            tasks={task: values},
            statistic=lambda s: mean(v for vs in s.values() for v in vs),
        )
        for task, values in p1_values.items()
    }
    p2_log2_values = query_values(_beta_log2)
    p2_ordinal_values = query_values(_beta_ordinal)
    p2_log2_pooled = task_stratified_bootstrap(
        label="phase2|p2log2|pooled",
        tasks=p2_log2_values,
        statistic=lambda s: mean(v for vs in s.values() for v in vs),
    )
    p2_ordinal_pooled = task_stratified_bootstrap(
        label="phase2|p2ord|pooled",
        tasks=p2_ordinal_values,
        statistic=lambda s: mean(v for vs in s.values() for v in vs),
    )
    p2_log2_by_task = {
        task: task_stratified_bootstrap(
            label=f"phase2|p2log2|{task}",
            tasks={task: values},
            statistic=lambda s: mean(v for vs in s.values() for v in vs),
        )
        for task, values in p2_log2_values.items()
    }

    def resolved_sign(ci: dict[str, float]) -> str:
        if ci["ci_low"] > 0:
            return "resolved_positive"
        if ci["ci_high"] < 0:
            return "resolved_negative"
        return "unresolved"

    p1_class = _classify_ci(p1_pooled["ci_low"], p1_pooled["ci_high"])
    p1_sign = (
        "unresolved"
        if p1_class == "straddles_zero"
        else "resolved_negative"
        if p1_class == "entirely_below_zero"
        else "resolved_positive"
    )
    p2_log2_sign = resolved_sign(p2_log2_pooled)
    p2_ord_sign = resolved_sign(p2_ordinal_pooled)
    if p2_log2_sign == p2_ord_sign and p2_log2_sign != "unresolved":
        p2_sign = p2_log2_sign
    else:
        p2_sign = "unresolved"
    row_index = {
        ("resolved_negative", "resolved_negative"): 0,
        ("resolved_negative", "unresolved"): 1,
        ("unresolved", "unresolved"): 2,
    }.get((p1_sign, p2_sign), 3)
    gate_rows = freeze_payload["condition_a_gate"]["rows"]
    selected_row = gate_rows[min(row_index, len(gate_rows) - 1)]
    run_condition_a = bool(selected_row["run_condition_a"])

    adjacent = {}
    for left, right in ((1, 8), (8, 16), (16, 32)):
        values = {
            task: [
                coverage[(task, q, right)] - coverage[(task, q, left)]
                for q in sorted(
                    entry["row_index"]
                    for entry in manifest["splits"][task]["heldout"]
                )
            ]
            for task in TASKS
        }
        adjacent[f"{left}->{right}"] = task_stratified_bootstrap(
            label=f"phase2|adj|{left}_{right}",
            tasks=values,
            statistic=lambda s: mean(v for vs in s.values() for v in vs),
        )

    precision_diversity = {}
    for b in B_VALUES:
        cell = {}
        for k in (1, 16):
            cell[f"pass@{k}"] = mean(
                estimate_pass_at_k(64, successes[(task, row, b)], k)
                for task in TASKS
                for row in sorted(
                    entry["row_index"]
                    for entry in manifest["splits"][task]["heldout"]
                )
            )
        precision_diversity[str(b)] = cell

    per_task_p1 = {
        task: {
            "point": ci["point"],
            "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"],
        }
        for task, ci in p1_by_task.items()
    }
    per_task_p2 = {
        task: {
            "point": ci["point"],
            "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"],
        }
        for task, ci in p2_log2_by_task.items()
    }
    return {
        "validation": validation,
        "markers": markers,
        "p1": {
            "pooled": p1_pooled,
            "classification": p1_class,
            "per_task": per_task_p1,
        },
        "p2": {
            "log2_pooled": p2_log2_pooled,
            "ordinal_pooled": p2_ordinal_pooled,
            "classification": p2_sign,
            "per_task_log2": per_task_p2,
        },
        "adjacent_steps": adjacent,
        "precision_diversity": precision_diversity,
        "gate": {
            "p1_class": p1_class,
            "p2_class": p2_sign,
            "selected_row": selected_row,
            "run_condition_a": run_condition_a,
        },
    }


def render_report_markdown(report: dict[str, Any], freeze_payload: dict[str, Any]) -> str:
    taxonomy = freeze_payload["verdict_taxonomy"]
    lines = [
        "# Phase 2 Option B — Block-Size Dose-Response Report",
        "",
        f"- Phase 2 freeze SHA-256: `{report['freeze_sha256']}`",
        f"- BCI freeze (provenance): `{freeze_payload['bci_freeze_sha256']}`",
        f"- Runner commit: `{freeze_payload['runner_commit']}`",
        f"- Artifacts validated: {report['validation']['present_and_valid']}"
        f"/{report['validation']['expected']}",
        "",
        "## P1 — endpoint contrast CoverageAUC(B=32) − CoverageAUC(B=1)",
        "",
    ]
    p1 = report["p1"]
    lines.append(
        f"- Pooled: {p1['pooled']['point']:.5f} "
        f"(95% CI [{p1['pooled']['ci_low']:.5f}, {p1['pooled']['ci_high']:.5f}])"
    )
    for task in TASKS:
        t = p1["per_task"][task]
        lines.append(
            f"- {task}: {t['point']:.5f} "
            f"(95% CI [{t['ci_low']:.5f}, {t['ci_high']:.5f}])"
        )
    p1_reading = {
        "entirely_below_zero": taxonomy["p1"]["entirely_below_zero"],
        "straddles_zero": taxonomy["p1"]["straddles_zero"],
        "entirely_above_zero": taxonomy["p1"]["entirely_above_zero"],
    }[p1["classification"]]
    lines.extend(
        [
            f"- Pre-registered classification: `{p1['classification']}`",
            f"- Pre-registered reading: {p1_reading}",
            "",
            "## P2 — trend statistic (log2 coding with ordinal sensitivity)",
            "",
            f"- log2 beta pooled: {report['p2']['log2_pooled']['point']:.5f} "
            f"(95% CI [{report['p2']['log2_pooled']['ci_low']:.5f}, "
            f"{report['p2']['log2_pooled']['ci_high']:.5f}])",
            f"- ordinal beta pooled: {report['p2']['ordinal_pooled']['point']:.5f} "
            f"(95% CI [{report['p2']['ordinal_pooled']['ci_low']:.5f}, "
            f"{report['p2']['ordinal_pooled']['ci_high']:.5f}])",
            f"- Classification: `{report['p2']['classification']}`",
            "",
            "## Gate: combination table and Condition A",
            "",
            f"- P1 class: `{p1['classification']}`; P2 class: "
            f"`{report['p2']['classification']}`",
            f"- Selected row: {report['gate']['selected_row']['reading']}",
            f"- **Run Condition A: "
            f"{'YES' if report['gate']['run_condition_a'] else 'NO'}**",
            "",
            "## Adjacent-step contrasts (secondary, underpowered for a smooth gradient)",
            "",
            "| step | pooled diff | 95% CI |",
            "|---|---:|---|",
        ]
    )
    for step, ci in report["adjacent_steps"].items():
        lines.append(
            f"| {step} | {ci['point']:.5f} | "
            f"[{ci['ci_low']:.5f}, {ci['ci_high']:.5f}] |"
        )
    lines.extend(
        [
            "",
            "## Precision–diversity secondary",
            "",
            "| B | pass@1 | pass@16 |",
            "|---:|---:|---:|",
        ]
    )
    for b in B_VALUES:
        cell = report["precision_diversity"][str(b)]
        lines.append(f"| {b} | {cell['pass@1']:.4f} | {cell['pass@16']:.4f} |")
    lines.extend(
        [
            "",
            "Token counts are never treated as independent sample sizes; all "
            "intervals are task-stratified query bootstraps.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_split_manifest(args.split_manifest)
    analyzer_commit = _repo_commit(args.repo_root)
    if args.expected_phase2_runner_commit is not None:
        runner_commit = args.expected_phase2_runner_commit
    else:
        runner_commit = analyzer_commit
    if args.invocation_log is not None:
        freeze_sha = (
            sha256_file(args.phase2_freeze)
            if args.phase2_freeze.is_file()
            else "pending"
        )
        _append_invocation_log(
            args.invocation_log,
            mode=args.mode,
            analyzer_commit=analyzer_commit,
            phase2_freeze_sha256=freeze_sha,
            reads_heldout_correctness=(args.mode == "analyze"),
            analysis_repair_amendment_sha256=(
                sha256_file(args.analysis_repair_amendment)
                if args.analysis_repair_amendment is not None
                else None
            ),
        )

    if args.mode == "emit-freeze":
        if args.phase2_freeze.exists():
            raise RuntimeError(f"refusing to overwrite: {args.phase2_freeze}")
        payload = build_freeze_payload(
            analyzer_commit=analyzer_commit, runner_commit=runner_commit
        )
        write_json_atomic(args.phase2_freeze, payload)
        print(
            json.dumps(
                {
                    "phase2_freeze": str(args.phase2_freeze),
                    "sha256": sha256_file(args.phase2_freeze),
                    "analyzer_commit": analyzer_commit,
                    "runner_commit": runner_commit,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    freeze_payload, freeze_sha, repair_sha = _load_phase2_freeze(
        args.phase2_freeze,
        analyzer_commit=analyzer_commit,
        analysis_repair_amendment=args.analysis_repair_amendment,
    )
    validator = Phase2RecordValidator(
        manifest=manifest,
        phase2_freeze_sha256=freeze_sha,
        phase1_schema_path=args.phase1_artifact_schema,
        phase2_schema_path=args.phase2_artifact_schema,
        phase2_runner_commit=freeze_payload["runner_commit"],
    )

    if args.mode == "validate-only":
        validation = _validate_phase2_artifacts(
            phase1_root=args.phase1_root,
            manifest=manifest,
            validator=validator,
        )
        summary = {
            "validated_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_validation": validation,
            "outcome_fields_read": False,
        }
        if args.output_validation is not None:
            if args.output_validation.exists():
                raise RuntimeError(f"refusing to overwrite: {args.output_validation}")
            write_json_atomic(args.output_validation, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    report = _analyze(
        phase1_root=args.phase1_root,
        manifest=manifest,
        freeze_payload=freeze_payload,
        validator=validator,
    )
    report["freeze_sha256"] = freeze_sha
    report["analysis_repair_amendment_sha256"] = repair_sha
    for path in (args.output_report_json, args.output_report_md, args.output_tables_csv):
        if path is not None and path.exists():
            raise RuntimeError(f"refusing to overwrite: {path}")
    if args.output_report_json is not None:
        write_json_atomic(args.output_report_json, report)
    if args.output_report_md is not None:
        write_text_atomic(
            args.output_report_md, render_report_markdown(report, freeze_payload)
        )
    print(
        json.dumps(
            {
                "p1_pooled": report["p1"]["pooled"]["point"],
                "p1_ci": [report["p1"]["pooled"]["ci_low"], report["p1"]["pooled"]["ci_high"]],
                "p1_classification": report["p1"]["classification"],
                "p2_log2_pooled": report["p2"]["log2_pooled"]["point"],
                "p2_ci": [report["p2"]["log2_pooled"]["ci_low"], report["p2"]["log2_pooled"]["ci_high"]],
                "p2_classification": report["p2"]["classification"],
                "run_condition_a": report["gate"]["run_condition_a"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
