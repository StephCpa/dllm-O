"""Tests for the frozen Phase 1 held-out analyzer."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean

import pytest

from dllm_order_transmission import phase1_heldout_analyzer as analyzer
from dllm_order_transmission.metrics import estimate_pass_at_k

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_MANIFEST = REPO_ROOT / "protocols" / "phase1_split_manifest_v1.json"
ARTIFACT_SCHEMA = REPO_ROOT / "protocols" / "phase1_artifact_schema_v1.json"

TASK_PARAMETERS = {
    "gsm8k": {
        "candidate_threshold_q75": 0.4870048891794262,
        "entropy_iqr": 2.736059457063675,
        "entropy_median": 1.709262192249298,
        "low_margin_iqr": 2.5494451709226214,
        "low_margin_median": -0.11778303565727243,
    },
    "math500": {
        "candidate_threshold_q75": 0.3938236314724752,
        "entropy_iqr": 2.960707128047943,
        "entropy_median": 2.667424201965332,
        "low_margin_iqr": 2.3025850929904457,
        "low_margin_median": 0.4700036292441356,
    },
}
BASE_ENTROPY = {"gsm8k": 3.0, "math500": 3.2}
PREDICTOR_COMMON_STANDARDIZATION = {
    "mean_commit_confidence": {"mean": 0.9384, "scale": 0.0342},
    "mean_eligible_entropy": {"mean": 2.0725, "scale": 0.5948},
    "output_length": {"mean": 133.31, "scale": 23.893},
    "prompt_length": {"mean": 84.17, "scale": 53.819},
}
BASELINE_PREDICTOR = {
    "model": "ridge_regression",
    "selected_alpha": 100.0,
    "feature_order": [
        "intercept",
        "task_math500",
        "prompt_length",
        "output_length",
        "mean_eligible_entropy",
        "mean_commit_confidence",
    ],
    "coefficients": {
        "intercept": 0.0212,
        "task_math500": -0.0025,
        "prompt_length": 0.0041,
        "output_length": -0.0047,
        "mean_eligible_entropy": 0.0022,
        "mean_commit_confidence": 0.0065,
    },
    "standardization": PREDICTOR_COMMON_STANDARDIZATION,
}
AUGMENTED_PREDICTOR = {
    "model": "ridge_regression",
    "selected_alpha": 100.0,
    "feature_order": [
        "intercept",
        "task_math500",
        "prompt_length",
        "output_length",
        "mean_eligible_entropy",
        "mean_commit_confidence",
        "bci",
    ],
    "coefficients": {
        "intercept": 0.0211,
        "task_math500": -0.0024,
        "prompt_length": 0.0042,
        "output_length": -0.0047,
        "mean_eligible_entropy": 0.0027,
        "mean_commit_confidence": 0.0062,
        "bci": -0.0014,
    },
    "standardization": {
        **PREDICTOR_COMMON_STANDARDIZATION,
        "bci": {"mean": 3.1775, "scale": 0.3131},
    },
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _trace_event(task: str, query_position: int) -> dict:
    # Entropy slope kept small enough that the candidate score stays below the
    # frozen q75 threshold for every query position, so the fallback selects
    # exactly one candidate per rollout and the second position stays a
    # matched control.
    base_entropy = BASE_ENTROPY[task] + 0.002 * query_position
    return {
        "batch_index": 0,
        "global_step": 47,
        "block_index": 2,
        "step_in_block": 15,
        "active_block_start": 164,
        "active_block_end": 196,
        "eligible_positions": [170, 171],
        "entropy": [base_entropy - 1.0 - 0.01 * query_position, base_entropy - 0.8],
        "top1_token_ids": [5, 6],
        "top2_token_ids": [4, 4],
        "top1_logits": [1.5, 1.4],
        "top2_logits": [0.4, 0.4],
        "logit_margins": [1.5, 1.4],
        "probability_margins": [0.3, 0.3],
        "candidate_token_ids": [5, 6],
        "candidate_probabilities": [0.9, 0.9],
        "eligible_entropy": [base_entropy, base_entropy],
        "eligible_logit_margins": [1.0, 1.0],
        "selected_positions": [170, 171],
        "selected_token_ids": [5, 6],
        "state_checksum_before": "x",
        "state_checksum_after": "y",
    }


def _single_position_trace_event(task: str, query_position: int) -> dict:
    event = _trace_event(task, query_position)
    for key in (
        "entropy",
        "top1_token_ids",
        "top2_token_ids",
        "top1_logits",
        "top2_logits",
        "logit_margins",
        "probability_margins",
        "candidate_token_ids",
        "candidate_probabilities",
        "eligible_entropy",
        "eligible_logit_margins",
        "selected_positions",
        "selected_token_ids",
    ):
        event[key] = event[key][:1]
    event["eligible_positions"] = event["eligible_positions"][:1]
    return event


MANIFEST = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
DATASET = {
    task: (
        MANIFEST["sources"][task]["dataset_id"],
        MANIFEST["sources"][task]["revision"],
        MANIFEST["sources"][task]["split"],
    )
    for task in ("gsm8k", "math500")
}


def _record(
    *,
    task: str,
    row: int,
    policy: str,
    run_kind: str,
    rollout: int,
    correct: bool,
    freeze_sha: str,
    response: str = "the answer is 42",
) -> dict:
    return {
        "protocol_version": "phase1-v1",
        "run_kind": run_kind,
        "split_manifest_sha256": analyzer.SPLIT_MANIFEST_SHA256,
        "bci_freeze_sha256": freeze_sha,
        "source": {
            "dataset_id": DATASET[task][0],
            "dataset_revision": DATASET[task][1],
            "dataset_split": DATASET[task][2],
            "row_index": row,
            "split_role": "heldout",
            "prompt_sha256": PROMPT_HASHES[(task, row)],
        },
        "model_id": analyzer.MODEL_ID,
        "model_revision": analyzer.MODEL_REVISION,
        "policy": {
            "name": policy,
            "steps": 256,
            "generation_length": 256,
            "block_length": {"ar_b1": 1, "ao_b8": 8, "ao_b16": 16}.get(policy, 32),
            "temperature": 0.6,
            "cfg_scale": 0.0,
            "remasking": "random" if policy == "random_b32" else "low_confidence",
        },
        "rollout_index": rollout,
        "seed_key": f"{task}|q{row:04d}|{policy}|r{rollout:02d}",
        "seed": rollout,
        "response": response,
        "parsed_answer": "42" if correct else "7",
        "parser_status": "ok",
        "correct": correct,
        "wall_time_seconds": 1.0,
        "peak_memory_bytes": None,
        "trace_event_count": 1,
        "trace_sha256": "",
        "runner_repository": {
            "commit": analyzer.EXPECTED_RUNNER_COMMIT,
            "dirty": False,
        },
        "external_repository_commit": analyzer.JUSTGRPO_COMMIT,
        "environment_fingerprint": {"python": "test"},
        "completion_status": "complete",
    }


HELDOUT_ROWS = {
    task: sorted(
        entry["row_index"] for entry in MANIFEST["splits"][task]["heldout"]
    )
    for task in ("gsm8k", "math500")
}
PROMPT_HASHES = {
    (task, entry["row_index"]): entry["prompt_sha256"]
    for task in ("gsm8k", "math500")
    for entry in MANIFEST["splits"][task]["heldout"]
}


def build_tree(
    root: Path,
    *,
    single_token_queries: set[tuple[str, int]] = frozenset(),
    random_successes: int = 8,
) -> tuple[Path, str]:
    """Materialize a complete synthetic held-out run; returns freeze path + sha."""
    phase1_root = root / "outputs"
    freeze_payload = {
        "protocol_version": analyzer.EXPECTED_FREEZE_VERSION,
        "runner_commit": analyzer.EXPECTED_RUNNER_COMMIT,
        "split_manifest_sha256": analyzer.SPLIT_MANIFEST_SHA256,
        "candidate_boundary_rule": {"task_parameters": TASK_PARAMETERS},
        "baseline_predictor": BASELINE_PREDICTOR,
        "augmented_predictor": AUGMENTED_PREDICTOR,
        "target_policy": {
            "formal_heldout": {"k_grid": [4, 8, 16, 32, 64]},
        },
    }
    freeze_bytes = json.dumps(freeze_payload, sort_keys=True).encode("utf-8")
    freeze_path = root / "synthetic_freeze.json"
    freeze_path.write_bytes(freeze_bytes)
    freeze_sha = _sha256_bytes(freeze_bytes)

    empty_trace_bytes = gzip.compress(b"")
    trace_bytes_cache: dict[tuple[str, int, bool], bytes] = {}

    def trace_bytes(task: str, position: int, single: bool) -> bytes:
        key = (task, position, single)
        if key not in trace_bytes_cache:
            event = (
                _single_position_trace_event(task, position)
                if single
                else _trace_event(task, position)
            )
            trace_bytes_cache[key] = gzip.compress(
                (json.dumps(event) + "\n").encode("utf-8")
            )
        return trace_bytes_cache[key]

    for task in ("gsm8k", "math500"):
        rows = HELDOUT_ROWS[task]
        for position, row in enumerate(rows):
            for policy in analyzer.SCREENING_POLICIES:
                for rollout in range(16):
                    if policy == "ar_b1":
                        correct = rollout < (20 + position // 4)
                    elif policy == "ao_b32":
                        correct = rollout < 8
                    elif policy == "random_b32":
                        correct = rollout < random_successes
                    else:
                        correct = rollout % 2 == 0
                    _write_item(
                        phase1_root,
                        task=task,
                        row=row,
                        policy=policy,
                        run_kind="screening",
                        rollout=rollout,
                        correct=correct,
                        freeze_sha=freeze_sha,
                        trace=(
                            trace_bytes(
                                task,
                                position,
                                (task, row) in single_token_queries,
                            )
                            if policy == "ao_b32"
                            else empty_trace_bytes
                        ),
                    )
            for policy in analyzer.PRIMARY_POLICIES:
                for rollout in range(16, 64):
                    successes = 20 + position // 4 if policy == "ar_b1" else 8
                    _write_item(
                        phase1_root,
                        task=task,
                        row=row,
                        policy=policy,
                        run_kind="primary-extension",
                        rollout=rollout,
                        correct=rollout < successes,
                        freeze_sha=freeze_sha,
                        trace=empty_trace_bytes,
                    )

    markers_root = phase1_root / "phase1-v1" / "manifests" / "heldout"
    for run_kind, per_shard in (("screening", 2000), ("primary-extension", 2400)):
        for shard in range(8):
            marker = {
                "protocol_version": "phase1-v1",
                "split_role": "heldout",
                "run_kind": run_kind,
                "task_selection": "all",
                "shard_index": shard,
                "num_shards": 8,
                "expected_keys_sha256": "f" * 64,
                "observed_keys_sha256": "f" * 64,
                "result_count": per_shard,
                "completed_at_utc": "2026-08-27T00:00:00+00:00",
            }
            path = (
                markers_root
                / run_kind
                / "all"
                / f"shard-{shard:03d}-of-008.json"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(marker), encoding="utf-8")
    return freeze_path, freeze_sha


def _write_item(
    phase1_root: Path,
    *,
    task: str,
    row: int,
    policy: str,
    run_kind: str,
    rollout: int,
    correct: bool,
    freeze_sha: str,
    trace: bytes,
) -> None:
    directory = (
        phase1_root
        / "phase1-v1"
        / "heldout"
        / run_kind
        / task
        / policy
        / f"q{row:04d}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    record = _record(
        task=task,
        row=row,
        policy=policy,
        run_kind=run_kind,
        rollout=rollout,
        correct=correct,
        freeze_sha=freeze_sha,
    )
    record["trace_sha256"] = _sha256_bytes(trace)
    (directory / f"r{rollout:02d}.trace.jsonl.gz").write_bytes(trace)
    (directory / f"r{rollout:02d}.json").write_text(
        json.dumps(record), encoding="utf-8"
    )


@pytest.fixture(scope="session")
def synthetic_run(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("synthetic_run")
    freeze_path, freeze_sha = build_tree(root)
    return {
        "root": root,
        "phase1_root": root / "outputs",
        "freeze_path": freeze_path,
        "freeze_sha": freeze_sha,
    }


def _base_args(run: dict, tmp: Path) -> list[str]:
    return [
        "--split-manifest",
        str(SPLIT_MANIFEST),
        "--artifact-schema",
        str(ARTIFACT_SCHEMA),
        "--freeze",
        str(run["freeze_path"]),
        "--expected-freeze-sha256",
        run["freeze_sha"],
        "--phase1-root",
        str(run["phase1_root"]),
        "--repo-root",
        str(REPO_ROOT),
        "--analysis-freeze",
        str(tmp / "heldout_analysis_freeze.json"),
    ]


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_stratified_spearman_cancels_opposite_task_effects() -> None:
    value = analyzer.stratified_spearman(
        {
            "gsm8k": [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)],
            "math500": [(10.0, 30.0), (20.0, 20.0), (30.0, 10.0)],
        }
    )
    assert value == pytest.approx(0.0, abs=1e-12)


def test_stratified_spearman_uses_average_ranks_for_ties() -> None:
    value = analyzer.stratified_spearman(
        {"gsm8k": [(1.0, 5.0), (1.0, 5.0), (2.0, 7.0)]}
    )
    assert value == pytest.approx(1.0)


def test_stratified_spearman_zero_variance_aborts() -> None:
    with pytest.raises(analyzer.FrozenRuleError, match="zero variance"):
        analyzer.stratified_spearman(
            {"gsm8k": [(1.0, 1.0), (1.0, 2.0)]}
        )


def test_task_stratified_bootstrap_is_deterministic() -> None:
    statistic = lambda sample: sum(  # noqa: E731
        value for values in sample.values() for value in values
    ) / 6
    first = analyzer.task_stratified_bootstrap(
        label="unit",
        tasks={"gsm8k": [1.0, 2.0, 3.0], "math500": [4.0, 5.0, 6.0]},
        statistic=statistic,
    )
    second = analyzer.task_stratified_bootstrap(
        label="unit",
        tasks={"gsm8k": [1.0, 2.0, 3.0], "math500": [4.0, 5.0, 6.0]},
        statistic=statistic,
    )
    assert first == second


def test_task_stratified_bootstrap_aborts_on_degenerate_replicate() -> None:
    def statistic(sample: dict) -> float:
        return analyzer.stratified_spearman(
            {task: [(v, 1.0) for v in values] for task, values in sample.items()}
        )

    with pytest.raises(analyzer.FrozenRuleError, match="replicate"):
        analyzer.task_stratified_bootstrap(
            label="degenerate",
            tasks={"gsm8k": [(5.0, 1.0)] * 4},
            statistic=statistic,
        )


# ---------------------------------------------------------------------------
# emit-freeze and validate-only
# ---------------------------------------------------------------------------


def test_emit_freeze_then_refuses_overwrite(synthetic_run, tmp_path) -> None:
    args = _base_args(synthetic_run, tmp_path) + ["--mode", "emit-freeze"]
    assert analyzer.main(args) == 0
    assert (tmp_path / "heldout_analysis_freeze.json").is_file()
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        analyzer.main(args)


def test_validate_only_strict_passes(synthetic_run, tmp_path, capsys) -> None:
    args = _base_args(synthetic_run, tmp_path) + ["--mode", "validate-only"]
    assert analyzer.main(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["outcome_fields_read"] is False
    assert summary["artifact_validation"]["present_and_valid"] == 35200
    assert summary["artifact_validation"]["missing"] == 0


def test_validate_only_never_reads_outcome_fields(
    synthetic_run, tmp_path, capsys, monkeypatch
) -> None:
    original_loads = json.loads

    def sentinel_loads(text: str):
        record = original_loads(text)
        if isinstance(record, dict) and "completion_status" in record:
            record["correct"] = not record["correct"]
            record["parsed_answer"] = "sentinel"
            record["response"] = "sentinel response"
            record["parser_status"] = "error:Sentinel"
        return record

    plain_args = _base_args(synthetic_run, tmp_path) + ["--mode", "validate-only"]
    assert analyzer.main(plain_args) == 0
    plain = json.loads(capsys.readouterr().out)["artifact_validation"]

    monkeypatch.setattr(analyzer.json, "loads", sentinel_loads)
    assert analyzer.main(plain_args) == 0
    mutated = json.loads(capsys.readouterr().out)["artifact_validation"]
    monkeypatch.undo()

    assert plain == mutated


def test_validate_only_fails_on_missing_artifacts(synthetic_run, tmp_path) -> None:
    victim = (
        synthetic_run["phase1_root"]
        / "phase1-v1"
        / "heldout"
        / "screening"
        / "gsm8k"
        / "ar_b1"
        / f"q{HELDOUT_ROWS['gsm8k'][0]:04d}"
        / "r00.json"
    )
    original = victim.read_bytes()
    try:
        victim.unlink()
        args = _base_args(synthetic_run, tmp_path) + ["--mode", "validate-only"]
        with pytest.raises(analyzer.FrozenRuleError, match="missing"):
            analyzer.main(args)
        assert analyzer.main(args + ["--allow-partial"]) == 0
    finally:
        victim.write_bytes(original)


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def _expected_advantage(position: int) -> float:
    successes_ar = 20 + position // 4
    successes_ao = 8
    return mean(
        estimate_pass_at_k(64, successes_ar, k)
        - estimate_pass_at_k(64, successes_ao, k)
        for k in (4, 8, 16, 32, 64)
    )


def _run_analysis(run: dict, tmp_path: Path) -> dict:
    freeze_path = tmp_path / "heldout_analysis_freeze.json"
    if not freeze_path.is_file():
        assert (
            analyzer.main(_base_args(run, tmp_path) + ["--mode", "emit-freeze"])
            == 0
        )
    report_json = tmp_path / "gate1_report.json"
    assert (
        analyzer.main(
            _base_args(run, tmp_path)
            + [
                "--mode",
                "analyze",
                "--output-report-json",
                str(report_json),
                "--output-report-md",
                str(tmp_path / "gate1_report.md"),
                "--output-tables-csv",
                str(tmp_path / "gate1_tables.csv"),
            ]
        )
        == 0
    )
    return json.loads(report_json.read_text(encoding="utf-8"))


def test_analyze_end_to_end(synthetic_run, tmp_path) -> None:
    report = _run_analysis(synthetic_run, tmp_path)

    assert report["analyzer_commit"] == analyzer._repo_commit(REPO_ROOT)
    assert report["artifact_validation"]["present_and_valid"] == 35200

    for task in ("gsm8k", "math500"):
        closure = report["rq1_closure"]["by_task"][task]
        assert closure["matched_queries"] == 100
        assert closure["matched_pair_count"] == 1600
        assert closure["rq1_unresolved"] is False
        expected_mean = 1.0 + 0.01 * 49.5 - 0.8
        ci = closure["entropy_closure"]
        assert ci["point"] == pytest.approx(expected_mean)
        assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]
        assert ci["ci_high"] - ci["ci_low"] < 0.3
        margin = closure["margin_closure"]
        assert margin["point"] == pytest.approx(0.1)
        assert margin["ci_low"] == pytest.approx(0.1)
        assert margin["ci_high"] == pytest.approx(0.1)

    association = report["rq2_bci_association"]
    expected_assignments = {
        task: [
            (
                1.0 + 0.01 * position,
                _expected_advantage(position),
            )
            for position in range(100)
        ]
        for task in ("gsm8k", "math500")
    }
    expected_pooled = analyzer.stratified_spearman(expected_assignments)
    assert association["pooled"]["point"] == pytest.approx(expected_pooled)
    for task in ("gsm8k", "math500"):
        expected_task = analyzer.stratified_spearman(
            {task: expected_assignments[task]}
        )
        assert association["per_task"][task]["point"] == pytest.approx(expected_task)

    position = 10
    row = HELDOUT_ROWS["gsm8k"][position]
    observed = report["per_query"][f"gsm8k|q{row:04d}"]["formal_advantage"]
    assert observed == pytest.approx(_expected_advantage(position))

    ar_entry = next(
        entry
        for entry in report["policy_summaries"]
        if entry["task"] == "gsm8k" and entry["policy"] == "ar_b1"
    )
    # Schedule summaries report Pass@1 from the frozen 16-rollout screening
    # window; the 64-rollout primary extension contributes only to Pass@32,
    # Pass@64, and CoverageAUC.
    assert ar_entry["pass_at_1"] == pytest.approx(1.0)
    assert ar_entry["coverage_auc"] == pytest.approx(
        mean(
            mean(
                estimate_pass_at_k(64, 20 + position // 4, k)
                for k in (4, 8, 16, 32, 64)
            )
            for position in range(100)
        )
    )
    assert ar_entry["malformed_fraction"] == 0.0
    assert "coverage_auc" in ar_entry

    gate = report["gate1"]
    assert gate["condition_1_closure"]["passed"] is True
    assert gate["condition_2_bci"]["pooled_ci_above_zero"] is True
    assert gate["overall_pass"] == (
        gate["condition_1_closure"]["passed"]
        and gate["condition_2_bci"]["pooled_ci_above_zero"]
        and gate["condition_2_bci"]["pooled_delta_ci_above_zero"]
    )
    assert report["malformed_sensitivity"] is None

    markdown = (tmp_path / "gate1_report.md").read_text(encoding="utf-8")
    assert "RQ1" in markdown and "Gate 1" in markdown
    assert "## Amendment Invariants" in markdown
    assert analyzer.EXPECTED_FREEZE_SHA256 in markdown
    assert analyzer.EXPECTED_RUNNER_COMMIT in markdown
    csv_text = (tmp_path / "gate1_tables.csv").read_text(encoding="utf-8")
    assert "gate1,gate1_overall" in csv_text


def test_analyze_marks_task_unresolved_below_match_threshold(
    tmp_path_factory,
) -> None:
    root = tmp_path_factory.mktemp("unresolved_run")
    single = {( "gsm8k", row) for row in HELDOUT_ROWS["gsm8k"][:15]}
    freeze_path, freeze_sha = build_tree(root, single_token_queries=single)
    run = {
        "root": root,
        "phase1_root": root / "outputs",
        "freeze_path": freeze_path,
        "freeze_sha": freeze_sha,
    }
    report = _run_analysis(run, root / "work")

    gsm8k = report["rq1_closure"]["by_task"]["gsm8k"]
    math500 = report["rq1_closure"]["by_task"]["math500"]
    assert gsm8k["matched_queries"] == 85
    assert gsm8k["matched_fraction"] == 0.85
    assert gsm8k["rq1_unresolved"] is True
    assert math500["rq1_unresolved"] is False
    assert report["gate1"]["condition_1_closure"]["passed"] is False
    assert report["gate1"]["condition_1_closure"]["unresolved_tasks"] == ["gsm8k"]
    assert report["gate1"]["overall_pass"] is False


def test_analyze_refuses_freeze_sha_mismatch(synthetic_run, tmp_path) -> None:
    tampered = tmp_path / "tampered_freeze.json"
    tampered.write_text(
        json.dumps(
            json.loads(synthetic_run["freeze_path"].read_text(encoding="utf-8"))
            | {"runner_commit": "0" * 40}
        ),
        encoding="utf-8",
    )
    args = [
        "--split-manifest",
        str(SPLIT_MANIFEST),
        "--artifact-schema",
        str(ARTIFACT_SCHEMA),
        "--freeze",
        str(tampered),
        "--expected-freeze-sha256",
        synthetic_run["freeze_sha"],
        "--phase1-root",
        str(synthetic_run["phase1_root"]),
        "--repo-root",
        str(REPO_ROOT),
        "--analysis-freeze",
        str(tmp_path / "never.json"),
        "--mode",
        "validate-only",
        "--allow-partial",
    ]
    with pytest.raises(analyzer.FrozenRuleError, match="hash mismatch"):
        analyzer.main(args)


def test_analyze_refuses_wrong_runner_commit_in_record(
    synthetic_run, tmp_path
) -> None:
    victim = (
        synthetic_run["phase1_root"]
        / "phase1-v1"
        / "heldout"
        / "screening"
        / "gsm8k"
        / "ar_b1"
        / f"q{HELDOUT_ROWS['gsm8k'][0]:04d}"
        / "r00.json"
    )
    original = victim.read_bytes()
    record = json.loads(original)
    record["runner_repository"]["commit"] = "0" * 40
    victim.write_text(json.dumps(record), encoding="utf-8")
    try:
        with pytest.raises(analyzer.FrozenRuleError, match="provenance mismatch"):
            analyzer.main(
                _base_args(synthetic_run, tmp_path) + ["--mode", "validate-only"]
            )
    finally:
        victim.write_bytes(original)


def test_analyze_requires_analysis_freeze_bound_to_current_commit(
    synthetic_run, tmp_path
) -> None:
    assert (
        analyzer.main(_base_args(synthetic_run, tmp_path) + ["--mode", "emit-freeze"])
        == 0
    )
    freeze_path = tmp_path / "heldout_analysis_freeze.json"
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    payload["analyzer_commit"] = "0" * 40
    freeze_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(analyzer.FrozenRuleError, match="analyzer commit"):
        analyzer.main(
            _base_args(synthetic_run, tmp_path)
            + [
                "--mode",
                "analyze",
                "--output-report-json",
                str(tmp_path / "never.json"),
            ]
        )


# ---------------------------------------------------------------------------
# Pre-unblinding amendment sections, cardinality, and verdict taxonomy
# ---------------------------------------------------------------------------


def test_emitted_analysis_freeze_contains_amended_sections(
    synthetic_run, tmp_path
) -> None:
    assert (
        analyzer.main(_base_args(synthetic_run, tmp_path) + ["--mode", "emit-freeze"])
        == 0
    )
    payload = json.loads(
        (tmp_path / "heldout_analysis_freeze.json").read_text(encoding="utf-8")
    )
    taxonomy = payload["verdict_taxonomy"]
    assert taxonomy["gate_booleans_unchanged"] is True
    assert "resolved wrong-direction" in taxonomy["entirely_below_zero"]
    assert payload["freeze_amendments"][0][
        "superseded_analysis_freeze_sha256"
    ] == analyzer.SUPERSEDED_ANALYSIS_FREEZE_SHA256
    assert payload["freeze_amendments"][1][
        "superseded_analysis_freeze_sha256"
    ] == analyzer.SECOND_SUPERSEDED_ANALYSIS_FREEZE_SHA256
    assert payload["freeze_amendments"][1][
        "superseded_analyzer_commit"
    ] == analyzer.SECOND_SUPERSEDED_ANALYZER_COMMIT
    assert "instrumentation" in payload["freeze_amendments"][1]["reason"]
    assert payload["freeze_amendments"][2][
        "superseded_analysis_freeze_sha256"
    ] == analyzer.THIRD_SUPERSEDED_ANALYSIS_FREEZE_SHA256
    assert payload["freeze_amendments"][2][
        "superseded_analyzer_commit"
    ] == analyzer.THIRD_SUPERSEDED_ANALYZER_COMMIT
    assert payload["freeze_amendments"][3][
        "superseded_analysis_freeze_sha256"
    ] == analyzer.FOURTH_SUPERSEDED_ANALYSIS_FREEZE_SHA256
    assert payload["freeze_amendments"][3][
        "superseded_analyzer_commit"
    ] == analyzer.FOURTH_SUPERSEDED_ANALYZER_COMMIT
    assert "invariants" in payload["freeze_amendments"][3]["reason"]
    contrast_readings = payload["secondary_contrast_readings"]
    assert "order freedom" in contrast_readings["straddles_zero"]
    assert "confidence-guided" in contrast_readings["entirely_below_zero"]
    assert "protective" in contrast_readings["entirely_above_zero"]
    negative_readings = payload["negative_rq2_mechanistic_readings"]
    assert "competence" in negative_readings["account_a_competence"]
    assert "difficulty" in negative_readings["account_b_confound"]
    assert payload["unblinding_read_order"]["order"][0].startswith(
        "1. record the Gate 1 verdict"
    )
    power = payload["power_resolution"]
    assert power["per_task_correlation_mde_n100"] == pytest.approx(0.277, abs=0.002)
    assert power["pooled_correlation_mde_n200"] == pytest.approx(0.197, abs=0.002)
    cardinality = payload["design_cardinality"]
    assert cardinality["screening_generations"] == 16_000
    assert cardinality["primary_extension_generations"] == 19_200
    assert cardinality["heldout_generations_total"] == 35_200


def test_analyze_requires_amended_freeze_sections(synthetic_run, tmp_path) -> None:
    assert (
        analyzer.main(_base_args(synthetic_run, tmp_path) + ["--mode", "emit-freeze"])
        == 0
    )
    freeze_path = tmp_path / "heldout_analysis_freeze.json"
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    del payload["verdict_taxonomy"]
    freeze_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        analyzer.FrozenRuleError, match="required pre-unblinding sections"
    ):
        analyzer.main(
            _base_args(synthetic_run, tmp_path)
            + [
                "--mode",
                "analyze",
                "--output-report-json",
                str(tmp_path / "never.json"),
            ]
        )


def test_design_cardinality_matches_manifest() -> None:
    cardinality = analyzer._design_cardinality(MANIFEST)
    assert cardinality["heldout_queries_per_task"] == {
        "gsm8k": 100,
        "math500": 100,
    }
    assert cardinality["heldout_query_total"] == 200
    assert cardinality["screening_generations"] == 16_000
    assert cardinality["primary_extension_generations"] == 19_200
    assert cardinality["heldout_generations_total"] == 35_200
    assert cardinality["protocol_new_generation_total"] == 41_600


def test_formal_pass_at_64_scope_is_primary_policies_only(
    synthetic_run, tmp_path
) -> None:
    report = _run_analysis(synthetic_run, tmp_path)
    with_pass64 = {
        (entry["task"], entry["policy"])
        for entry in report["policy_summaries"]
        if "pass_at_64" in entry
    }
    expected = {
        (task, policy)
        for task in ("gsm8k", "math500")
        for policy in ("ar_b1", "ao_b32")
    }
    assert with_pass64 == expected
    secondary = [
        entry
        for entry in report["policy_summaries"]
        if entry["policy"] in ("ao_b8", "ao_b16", "random_b32")
    ]
    assert all("pass_at_64" not in entry for entry in secondary)
    assert all(entry["rollout_count"] == 16 for entry in secondary)
    assert report["design_cardinality"]["heldout_query_total"] == 200
    assert report["verdict_taxonomy"]["gate_booleans_unchanged"] is True
    classification = report["gate1"]["condition_2_bci"]["pooled_ci_classification"]
    assert classification in (
        "entirely_above_zero",
        "straddles_zero",
        "entirely_below_zero",
    )


# ---------------------------------------------------------------------------
# Confidence-guided versus random order contrast
# ---------------------------------------------------------------------------


def _confidence_vs_random_expected() -> float:
    return mean(
        estimate_pass_at_k(16, 8, k) - estimate_pass_at_k(16, 6, k)
        for k in (2, 4, 8, 16)
    )


def test_confidence_vs_random_contrast_values(tmp_path_factory) -> None:
    root = tmp_path_factory.mktemp("random_contrast_run")
    freeze_path, freeze_sha = build_tree(root, random_successes=6)
    run = {
        "root": root,
        "phase1_root": root / "outputs",
        "freeze_path": freeze_path,
        "freeze_sha": freeze_sha,
    }
    report = _run_analysis(run, root / "work")
    contrast = report["confidence_vs_random_contrast"]
    expected = _confidence_vs_random_expected()
    for task in ("gsm8k", "math500"):
        entry = contrast["per_task"][task]
        assert entry["point"] == pytest.approx(expected)
        assert entry["ci_low"] == pytest.approx(expected)
        assert entry["ci_high"] == pytest.approx(expected)
    assert contrast["pooled"]["point"] == pytest.approx(expected)
    assert contrast["k_grid"] == [2, 4, 8, 16]
    assert "causal" in contrast["scope"]
    assert contrast["pooled_ci_classification"] in (
        "entirely_above_zero",
        "straddles_zero",
        "entirely_below_zero",
    )
    assert "order freedom" in report["secondary_contrast_readings"][
        "straddles_zero"
    ]
    assert report["unblinding_read_order"]["order"][-1].startswith(
        "3. read the confidence-vs-random contrast"
    )
    csv_text = (root / "work" / "gate1_tables.csv").read_text(encoding="utf-8")
    assert "confidence_vs_random" in csv_text
    markdown = (root / "work" / "gate1_report.md").read_text(encoding="utf-8")
    assert "Pre-registered reading:" in markdown


def test_confidence_vs_random_ignores_extension_outcomes(
    synthetic_run, tmp_path
) -> None:
    report_before = _run_analysis(synthetic_run, tmp_path / "before")
    contrast_before = report_before["confidence_vs_random_contrast"]

    phase1_root = synthetic_run["phase1_root"]
    for task in ("gsm8k", "math500"):
        for row in HELDOUT_ROWS[task]:
            for policy in ("ar_b1", "ao_b32"):
                directory = (
                    phase1_root
                    / "phase1-v1"
                    / "heldout"
                    / "primary-extension"
                    / task
                    / policy
                    / f"q{row:04d}"
                )
                for rollout in range(16, 64):
                    path = directory / f"r{rollout:02d}.json"
                    record = json.loads(path.read_text(encoding="utf-8"))
                    record["correct"] = not record["correct"]
                    path.write_text(json.dumps(record), encoding="utf-8")

    report_after = _run_analysis(synthetic_run, tmp_path / "after")
    assert (
        report_after["confidence_vs_random_contrast"] == contrast_before
    )


# ---------------------------------------------------------------------------
# Screening malformed-output audit
# ---------------------------------------------------------------------------


def _set_malformed(
    phase1_root: Path, task: str, policy: str, row: int, rollout: int
) -> None:
    path = (
        phase1_root
        / "phase1-v1"
        / "heldout"
        / "screening"
        / task
        / policy
        / f"q{row:04d}"
        / f"r{rollout:02d}.json"
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    record["parser_status"] = "parse_error"
    record["parsed_answer"] = None
    path.write_text(json.dumps(record), encoding="utf-8")


def _run_audit(run: dict, tmp: Path) -> dict:
    args = _base_args(run, tmp) + [
        "--mode",
        "audit-malformed",
        "--output-audit-json",
        str(tmp / "audit.json"),
        "--output-audit-md",
        str(tmp / "audit.md"),
    ]
    assert analyzer.main(args) == 0
    return json.loads((tmp / "audit.json").read_text(encoding="utf-8"))


def test_malformed_audit_zero_case(synthetic_run, tmp_path) -> None:
    audit = _run_audit(synthetic_run, tmp_path)
    assert audit["correctness_read"] is False
    assert audit["counts"] == {
        "expected": 16_000,
        "present_and_valid": 16_000,
        "missing": 0,
        "invalid": 0,
    }
    assert audit["triggered"] == []
    for task in ("gsm8k", "math500"):
        differential = audit["differential_missingness"][task]
        assert differential["primary_ar_minus_ao_rate_difference"] == 0.0
        assert differential["flagged_for_confound_risk"] is False
    markdown = (tmp_path / "audit.md").read_text(encoding="utf-8")
    assert "Differential missingness" in markdown


def test_malformed_audit_exact_threshold_behavior(synthetic_run, tmp_path) -> None:
    phase1_root = synthetic_run["phase1_root"]
    first_row = HELDOUT_ROWS["gsm8k"][0]
    for rollout in range(16):
        _set_malformed(phase1_root, "gsm8k", "ao_b16", first_row, rollout)
    audit = _run_audit(synthetic_run, tmp_path / "at_threshold")
    cell = audit["by_cell"]["gsm8k|ao_b16"]
    assert cell["malformed_fraction"] == pytest.approx(0.01)
    assert cell["exceeds_sensitivity_threshold"] is False
    assert audit["triggered"] == []

    second_row = HELDOUT_ROWS["gsm8k"][1]
    _set_malformed(phase1_root, "gsm8k", "ao_b16", second_row, 0)
    audit = _run_audit(synthetic_run, tmp_path / "above_threshold")
    cell = audit["by_cell"]["gsm8k|ao_b16"]
    assert cell["malformed_fraction"] == pytest.approx(17 / 1600)
    assert cell["exceeds_sensitivity_threshold"] is True
    assert {"task": "gsm8k", "policy": "ao_b16"} in audit["triggered"]


def test_malformed_audit_differential_missingness(synthetic_run, tmp_path) -> None:
    phase1_root = synthetic_run["phase1_root"]
    row = HELDOUT_ROWS["math500"][0]
    for rollout in range(16):
        _set_malformed(phase1_root, "math500", "ar_b1", row, rollout)
    _set_malformed(phase1_root, "math500", "ar_b1", HELDOUT_ROWS["math500"][1], 0)
    audit = _run_audit(synthetic_run, tmp_path)
    differential = audit["differential_missingness"]["math500"]
    assert differential["primary_ar_minus_ao_rate_difference"] == pytest.approx(
        17 / 1600
    )
    assert differential["flagged_for_confound_risk"] is True
    worst = differential["max_abs_pairwise_rate_difference"]
    assert worst["left"] == "ar_b1"
    assert abs(worst["difference"]) == pytest.approx(17 / 1600)


def test_malformed_audit_sentinel_invariance(
    synthetic_run, tmp_path, monkeypatch
) -> None:
    plain_audit = _run_audit(synthetic_run, tmp_path / "plain")
    plain_audit.pop("created_at_utc")

    original_loads = json.loads

    def sentinel_loads(text: str):
        record = original_loads(text)
        if isinstance(record, dict) and "completion_status" in record:
            record["correct"] = not record["correct"]
            record["response"] = "sentinel response"
        return record

    monkeypatch.setattr(analyzer.json, "loads", sentinel_loads)
    mutated_audit = _run_audit(synthetic_run, tmp_path / "mutated")
    mutated_audit.pop("created_at_utc")
    monkeypatch.undo()

    assert plain_audit == mutated_audit


def test_malformed_audit_reports_missing(synthetic_run, tmp_path) -> None:
    victim = (
        synthetic_run["phase1_root"]
        / "phase1-v1"
        / "heldout"
        / "screening"
        / "math500"
        / "random_b32"
        / f"q{HELDOUT_ROWS['math500'][0]:04d}"
        / "r00.json"
    )
    original = victim.read_bytes()
    try:
        victim.unlink()
        audit = _run_audit(synthetic_run, tmp_path)
        assert audit["counts"]["missing"] == 1
        assert audit["counts"]["present_and_valid"] == 15_999
    finally:
        victim.write_bytes(original)


# ---------------------------------------------------------------------------
# Power-resolution simulation
# ---------------------------------------------------------------------------


def test_power_simulation_is_deterministic_and_bounded() -> None:
    first = analyzer.empirical_resolution_note(replicates=1_000)
    second = analyzer.empirical_resolution_note(replicates=1_000)
    assert first == second
    single_halves = [
        first["single_task"][task]["half_width"] for task in ("gsm8k", "math500")
    ]
    assert all(0.0 < half < 0.35 for half in single_halves)
    pooled_half = first["pooled"]["half_width"]
    assert 0.0 < pooled_half < min(single_halves)
    assert first["status"].startswith("pre-unblinding")


def test_power_simulation_cli(synthetic_run, tmp_path) -> None:
    args = _base_args(synthetic_run, tmp_path) + [
        "--mode",
        "power-simulation",
        "--output-simulation-json",
        str(tmp_path / "sim.json"),
        "--output-simulation-md",
        str(tmp_path / "sim.md"),
    ]
    assert analyzer.main(args) == 0
    note = json.loads((tmp_path / "sim.json").read_text(encoding="utf-8"))
    assert note["replicates"] == analyzer.RESOLUTION_SIMULATION_REPLICATES
    assert "half-widths" in (tmp_path / "sim.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Append-only invocation log
# ---------------------------------------------------------------------------


def test_invocation_log_records_modes_without_correctness_read(
    synthetic_run, tmp_path
) -> None:
    log = tmp_path / "invocations.jsonl"
    base = _base_args(synthetic_run, tmp_path) + ["--invocation-log", str(log)]
    assert analyzer.main(base + ["--mode", "emit-freeze"]) == 0
    assert analyzer.main(base + ["--mode", "audit-malformed"]) == 0
    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [entry["mode"] for entry in lines] == ["emit-freeze", "audit-malformed"]
    assert all(not entry["reads_heldout_correctness"] for entry in lines)
    assert lines[0]["analyzer_commit"] == analyzer._repo_commit(REPO_ROOT)
    assert lines[0]["bci_freeze_sha256"] == synthetic_run["freeze_sha"]
    assert "analysis_freeze" in lines[1]["output_paths"]
    assert "output_report_json" not in lines[1]["output_paths"]


def test_invocation_log_flags_first_correctness_read(
    synthetic_run, tmp_path
) -> None:
    log = tmp_path / "invocations.jsonl"
    base = _base_args(synthetic_run, tmp_path) + ["--invocation-log", str(log)]
    assert analyzer.main(base + ["--mode", "emit-freeze"]) == 0
    report_json = tmp_path / "report.json"
    assert (
        analyzer.main(
            base
            + [
                "--mode",
                "analyze",
                "--output-report-json",
                str(report_json),
            ]
        )
        == 0
    )
    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["reads_heldout_correctness"] is False
    assert lines[-1]["mode"] == "analyze"
    assert lines[-1]["reads_heldout_correctness"] is True
    assert lines[-1]["output_paths"]["output_report_json"] == str(report_json)


def test_invocation_log_appends_even_on_failure(synthetic_run, tmp_path) -> None:
    log = tmp_path / "invocations.jsonl"
    base = _base_args(synthetic_run, tmp_path) + [
        "--invocation-log",
        str(log),
        "--mode",
        "emit-freeze",
    ]
    assert analyzer.main(base) == 0
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        analyzer.main(base)
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["mode"] for line in lines] == [
        "emit-freeze",
        "emit-freeze",
    ]
