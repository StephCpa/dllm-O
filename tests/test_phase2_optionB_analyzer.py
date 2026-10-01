"""Tests for the Phase 2 Option B analyzer."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from statistics import mean

import pytest

from dllm_order_transmission import phase2_optionB_analyzer as analyzer
from dllm_order_transmission.metrics import estimate_pass_at_k

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_MANIFEST = REPO_ROOT / "protocols" / "phase1_split_manifest_v1.json"
PHASE1_SCHEMA = REPO_ROOT / "protocols" / "phase1_artifact_schema_v1.json"
PHASE2_SCHEMA = REPO_ROOT / "protocols" / "phase2_artifact_schema_v1.json"

TEST_RUNNER_COMMIT = "a" * 40

MANIFEST = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
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
DATASET = {
    task: (
        MANIFEST["sources"][task]["dataset_id"],
        MANIFEST["sources"][task]["revision"],
        MANIFEST["sources"][task]["split"],
    )
    for task in ("gsm8k", "math500")
}

# Correctness patterns chosen for deterministic expected values:
#   ar_b1 (B=1):  20 + position//4 correct of 64 (Phase 1 pattern)
#   ao_b8  (B=8):  8 screening + 20 extension = 28
#   ao_b16 (B=16): 8 screening + 12 extension = 20
#   ao_b32 (B=32): 8 screening + 0 extension  = 8  (Phase 1 build_tree marks
#                  ao_b32 extension rollouts 16-63 all incorrect)
AO_B8_EXTENSION_SUCCESSES = 20
AO_B16_EXTENSION_SUCCESSES = 12


def _phase2_extension_correct(policy: str, rollout: int) -> bool:
    if policy == "ao_b8":
        return rollout - 16 < AO_B8_EXTENSION_SUCCESSES
    return rollout - 16 < AO_B16_EXTENSION_SUCCESSES


def _write_phase2_record(
    phase1_root: Path,
    *,
    task: str,
    row: int,
    policy: str,
    rollout: int,
    correct: bool,
    freeze_sha: str,
) -> None:
    directory = (
        phase1_root / "phase1-v1" / "heldout" / "phase2-extension"
        / task / policy / f"q{row:04d}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "protocol_version": "phase1-v1",
        "run_kind": "phase2-extension",
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
        "model_id": "GSAI-ML/LLaDA-8B-Instruct",
        "model_revision": "08b83a6feb34df1a6011b80c3c00c7563e963b07",
        "policy": {
            "name": policy,
            "steps": 256,
            "generation_length": 256,
            "block_length": 8 if policy == "ao_b8" else 16,
            "temperature": 0.6,
            "cfg_scale": 0.0,
            "remasking": "low_confidence",
        },
        "rollout_index": rollout,
        "seed_key": f"{task}|q{row:04d}|{policy}|r{rollout:02d}",
        "seed": rollout,
        "response": "the answer is 42",
        "parsed_answer": "42" if correct else "7",
        "parser_status": "ok",
        "correct": correct,
        "wall_time_seconds": 1.0,
        "peak_memory_bytes": None,
        "trace_event_count": 1,
        "trace_sha256": "",
        "runner_repository": {"commit": TEST_RUNNER_COMMIT, "dirty": False},
        "external_repository_commit": "1a2fddb5c6655597e63081c0af5ebb718a849f39",
        "environment_fingerprint": {"python": "test"},
        "completion_status": "complete",
    }
    trace = gzip.compress(b"")
    record["trace_sha256"] = __import__("hashlib").sha256(trace).hexdigest()
    (directory / f"r{rollout:02d}.trace.jsonl.gz").write_bytes(trace)
    (directory / f"r{rollout:02d}.json").write_text(
        json.dumps(record), encoding="utf-8"
    )


def build_phase2_tree(root: Path, phase2_freeze_sha: str) -> Path:
    """Phase 1 synthetic tree plus ao_b8/ao_b16 phase2-extension records."""
    from test_phase1_heldout_analyzer import build_tree

    phase1_freeze_path, _ = build_tree(root)
    phase1_root = root / "outputs"
    for task in ("gsm8k", "math500"):
        for row in HELDOUT_ROWS[task]:
            for policy in ("ao_b8", "ao_b16"):
                for rollout in range(16, 64):
                    _write_phase2_record(
                        phase1_root,
                        task=task,
                        row=row,
                        policy=policy,
                        rollout=rollout,
                        correct=_phase2_extension_correct(policy, rollout),
                        freeze_sha=phase2_freeze_sha,
                    )
    markers_root = (
        phase1_root / "phase1-v1" / "manifests" / "heldout" / "phase2-extension" / "all"
    )
    markers_root.mkdir(parents=True, exist_ok=True)
    for shard in range(8):
        marker = {
            "protocol_version": "phase1-v1",
            "split_role": "heldout",
            "run_kind": "phase2-extension",
            "task_selection": "all",
            "shard_index": shard,
            "num_shards": 8,
            "expected_keys_sha256": "f" * 64,
            "observed_keys_sha256": "f" * 64,
            "result_count": 2400,
            "completed_at_utc": "2026-09-02T00:00:00+00:00",
        }
        (markers_root / f"shard-{shard:03d}-of-008.json").write_text(
            json.dumps(marker), encoding="utf-8"
        )
    return phase1_freeze_path


def _freeze_args(run: dict, freeze_path: Path) -> list[str]:
    return [
        "--split-manifest",
        str(SPLIT_MANIFEST),
        "--phase1-artifact-schema",
        str(PHASE1_SCHEMA),
        "--phase2-artifact-schema",
        str(PHASE2_SCHEMA),
        "--phase1-root",
        str(run["phase1_root"]),
        "--repo-root",
        str(REPO_ROOT),
        "--phase2-freeze",
        str(freeze_path),
        "--expected-phase2-runner-commit",
        TEST_RUNNER_COMMIT,
    ]


@pytest.fixture(scope="session")
def phase2_run(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("phase2_run")
    freeze_path = root / "session_phase2_freeze.json"
    phase1_root = root / "outputs"
    run = {"root": root, "phase1_root": phase1_root}
    assert (
        analyzer.main(
            _freeze_args(run, freeze_path) + ["--mode", "emit-freeze"]
        )
        == 0
    )
    freeze_sha = __import__("hashlib").sha256(
        freeze_path.read_bytes()
    ).hexdigest()
    build_phase2_tree(root, freeze_sha)
    return {**run, "freeze_path": freeze_path, "freeze_sha": freeze_sha}


def _base_args(run: dict, tmp: Path) -> list[str]:
    return _freeze_args(run, run["freeze_path"])


def _expected_coverage_auc(successes: int) -> float:
    return mean(
        estimate_pass_at_k(64, successes, k) for k in (4, 8, 16, 32, 64)
    )


def _expected_p1(position: int) -> float:
    s1 = 20 + position // 4
    return _expected_coverage_auc(8) - _expected_coverage_auc(s1)


def _expected_p2_log2(position: int) -> float:
    s1 = 20 + position // 4
    c1 = _expected_coverage_auc(s1)
    c8 = _expected_coverage_auc(28)
    c16 = _expected_coverage_auc(20)
    c32 = _expected_coverage_auc(8)
    return (-3 * c1 + 0 * c8 + 1 * c16 + 2 * c32) / 14


def test_emit_freeze_contains_required_sections(phase2_run, tmp_path) -> None:
    own_freeze = tmp_path / "own_freeze.json"
    args = _freeze_args(phase2_run, own_freeze) + ["--mode", "emit-freeze"]
    assert analyzer.main(args) == 0
    payload = json.loads(own_freeze.read_text(encoding="utf-8"))
    assert payload["protocol_version"] == "phase2-optionb-freeze-v1"
    assert payload["revision_history"] == {
        "supersedes_phase2_freeze_sha256": analyzer.SUPERSEDED_PHASE2_FREEZE_SHA256,
        "reason": (
            "pre-launch mechanical schema correction: the generation-result "
            "schema already admitted phase2-extension, but the shard-completion "
            "branch accidentally omitted the same run kind"
        ),
        "phase2_outcomes_read_before_revision": False,
        "statistical_design_changed": False,
    }
    assert payload["bci_freeze_sha256"] == analyzer.BCI_FREEZE_SHA256
    assert payload["runner_commit"] == TEST_RUNNER_COMMIT
    assert payload["primary_endpoints"]["p2_demotion"]
    assert len(payload["condition_a_gate"]["rows"]) == 4
    assert payload["cardinality"]["new_generations"] == 19_200
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        analyzer.main(args)


def test_validate_only_full_tree_and_sentinel(
    phase2_run, tmp_path, monkeypatch
) -> None:
    args = _base_args(phase2_run, tmp_path) + ["--mode", "validate-only"]
    assert analyzer.main(args) == 0
    import io
    import contextlib

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        analyzer.main(args)
    summary = json.loads(buffer.getvalue())
    assert summary["outcome_fields_read"] is False
    assert summary["artifact_validation"]["present_and_valid"] == 51_200
    assert summary["artifact_validation"]["missing"] == 0

    original_loads = json.loads

    def sentinel_loads(text: str):
        record = original_loads(text)
        if isinstance(record, dict) and "completion_status" in record:
            record["correct"] = not record["correct"]
            record["parsed_answer"] = "sentinel"
            record["parser_status"] = "error:Sentinel"
        return record

    monkeypatch.setattr(analyzer.json, "loads", sentinel_loads)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        analyzer.main(args)
    mutated = json.loads(buffer.getvalue())
    monkeypatch.undo()
    assert mutated["artifact_validation"] == summary["artifact_validation"]


def test_load_successes_preserves_all_zero_cells(monkeypatch) -> None:
    class AlwaysIncorrectPath:
        def read_text(self, *, encoding: str) -> str:
            assert encoding == "utf-8"
            return '{"correct": false}'

    monkeypatch.setattr(
        analyzer, "_result_path", lambda phase1_root, item: AlwaysIncorrectPath()
    )
    manifest = {
        "splits": {
            task: {"heldout": [{"row_index": 7}]} for task in analyzer.TASKS
        }
    }
    successes = analyzer._load_successes(Path("unused"), manifest)
    expected_keys = {
        (task, 7, b) for task in analyzer.TASKS for b in analyzer.B_VALUES
    }
    assert set(successes) == expected_keys
    assert all(successes[key] == 0 for key in expected_keys)


def test_analysis_repair_amendment_is_narrow_and_auditable(
    phase2_run, tmp_path
) -> None:
    original_freeze = json.loads(
        phase2_run["freeze_path"].read_text(encoding="utf-8")
    )
    original_freeze["analyzer_commit"] = "0" * 40
    mismatched_freeze = tmp_path / "mismatched_freeze.json"
    mismatched_freeze.write_text(json.dumps(original_freeze), encoding="utf-8")
    freeze_sha = __import__("hashlib").sha256(
        mismatched_freeze.read_bytes()
    ).hexdigest()
    repair = tmp_path / "repair.json"
    repair.write_text(
        json.dumps(
            {
                "protocol_version": analyzer.ANALYSIS_REPAIR_VERSION,
                "original_phase2_freeze_sha256": freeze_sha,
                "original_analyzer_commit": "0" * 40,
                "repair_analyzer_commit": analyzer._repo_commit(REPO_ROOT),
                "repair_scope": analyzer.ANALYSIS_REPAIR_SCOPE,
                "statistical_definitions_changed": False,
                "experimental_artifacts_changed": False,
                "outcome_based_decisions": False,
                "failed_invocation_utc": "2026-09-06T13:07:05.794369+00:00",
                "failure_console_sha256": "f" * 64,
                "reason": "All-zero success cells must remain explicit observations.",
            }
        ),
        encoding="utf-8",
    )
    payload, observed_freeze_sha, repair_sha = analyzer._load_phase2_freeze(
        mismatched_freeze,
        analyzer_commit=analyzer._repo_commit(REPO_ROOT),
        analysis_repair_amendment=repair,
    )
    assert payload["analyzer_commit"] == "0" * 40
    assert observed_freeze_sha == freeze_sha
    assert repair_sha == __import__("hashlib").sha256(repair.read_bytes()).hexdigest()

    bad = json.loads(repair.read_text(encoding="utf-8"))
    bad["statistical_definitions_changed"] = True
    repair.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(analyzer.FrozenRuleError, match="invalid analysis repair"):
        analyzer._load_phase2_freeze(
            mismatched_freeze,
            analyzer_commit=analyzer._repo_commit(REPO_ROOT),
            analysis_repair_amendment=repair,
        )


def test_analyze_p1_p2_and_gate(phase2_run, tmp_path) -> None:
    report_json = tmp_path / "report.json"
    out = analyzer.main(
        _base_args(phase2_run, tmp_path)
        + [
            "--mode",
            "analyze",
            "--output-report-json",
            str(report_json),
            "--output-report-md",
            str(tmp_path / "report.md"),
        ]
    )
    assert out == 0
    report = json.loads(report_json.read_text(encoding="utf-8"))

    expected_p1 = mean(
        _expected_p1(position) for position in range(100)
    )  # same pattern in both tasks
    assert report["p1"]["pooled"]["point"] == pytest.approx(expected_p1, abs=1e-9)
    expected_p2 = mean(_expected_p2_log2(position) for position in range(100))
    assert report["p2"]["log2_pooled"]["point"] == pytest.approx(
        expected_p2, abs=1e-9
    )
    assert report["p1"]["classification"] in (
        "entirely_above_zero",
        "straddles_zero",
        "entirely_below_zero",
    )
    assert report["gate"]["run_condition_a"] in (True, False)
    assert report["markers"]["total_result_count"] == 19_200
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "P1 — endpoint contrast" in markdown
    assert "Run Condition A" in markdown
    assert "Adjacent-step contrasts" in markdown


def test_analyze_requires_freeze_bound_to_current_commit(
    phase2_run, tmp_path
) -> None:
    tampered = tmp_path / "tampered_freeze.json"
    payload = json.loads(phase2_run["freeze_path"].read_text(encoding="utf-8"))
    payload["analyzer_commit"] = "0" * 40
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(analyzer.FrozenRuleError, match="analyzer commit"):
        analyzer.main(
            _freeze_args(phase2_run, tampered)
            + ["--mode", "analyze", "--output-report-json", str(tmp_path / "never.json")]
        )


def test_analyze_fails_on_missing_marker(phase2_run, tmp_path) -> None:
    victim = (
        phase2_run["phase1_root"]
        / "phase1-v1"
        / "manifests"
        / "heldout"
        / "phase2-extension"
        / "all"
        / "shard-007-of-008.json"
    )
    original = victim.read_bytes()
    try:
        victim.unlink()
        with pytest.raises(analyzer.FrozenRuleError, match="marker"):
            analyzer.main(
                _base_args(phase2_run, tmp_path)
                + ["--mode", "analyze", "--output-report-json", str(tmp_path / "never.json")]
            )
    finally:
        victim.write_bytes(original)
