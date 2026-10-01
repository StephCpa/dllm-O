import numpy as np

from dllm_order_transmission.phase1_calibration import (
    _annotate_candidates,
    _candidate_score,
    _fit_predictor,
    _match_candidates,
    _quantile,
    _robust_location_scale,
)


def test_quantile_and_robust_scale_are_deterministic():
    assert _quantile([0, 10, 20, 30], 0.75) == 22.5
    assert _robust_location_scale([1, 1, 1]) == (1.0, 1.0)


def test_candidate_score_rewards_entropy_and_low_margin():
    normalization = {
        "entropy_median": 1.0,
        "entropy_iqr": 1.0,
        "low_margin_median": 0.0,
        "low_margin_iqr": 1.0,
    }
    assert _candidate_score(2.0, 0.01, normalization) > _candidate_score(
        1.0, 1.0, normalization
    )


def test_candidate_fallback_and_matching_are_within_block():
    rows = []
    for position, score in enumerate([2.0, 0.0, -1.0]):
        rows.append(
            {
                "task": "gsm8k",
                "query": 0,
                "rollout": 0,
                "block": 0,
                "position": position,
                "normalized_position": position / 2,
                "normalized_commitment_rank": position / 2,
                "eligible_entropy": score,
                "eligible_margin": 1.0,
                "commit_confidence": 0.5,
                "entropy_closure": float(position),
                "margin_closure": float(position),
            }
        )
    fitted = {
        "gsm8k": {
            "entropy_median": 0.0,
            "entropy_iqr": 1.0,
            "low_margin_median": 0.0,
            "low_margin_iqr": 1.0,
            "candidate_threshold_q75": 0.75,
            "control_ceiling_q50": 0.0,
        }
    }
    _annotate_candidates(rows, fitted)
    # The fixed 0.25 caliper intentionally rejects these distant controls.
    assert _match_candidates(rows, fitted) == []
    assert sum(row["candidate"] for row in rows) == 1


def test_matching_uses_nearest_control_without_replacement():
    rows = []
    for position, score, closure in (
        (0, 2.0, 3.0),
        (1, 1.0, 1.0),
        (2, 0.5, 0.0),
    ):
        rows.append(
            {
                "task": "gsm8k",
                "query": 0,
                "rollout": 0,
                "block": 0,
                "position": position,
                "normalized_position": position / 10,
                "normalized_commitment_rank": position / 10,
                "eligible_entropy": score,
                "eligible_margin": 1.0,
                "commit_confidence": 0.5,
                "entropy_closure": closure,
                "margin_closure": closure,
            }
        )
    fitted = {
        "gsm8k": {
            "entropy_median": 0.0,
            "entropy_iqr": 10.0,
            "low_margin_median": 0.0,
            "low_margin_iqr": 1.0,
            "candidate_threshold_q75": 0.15,
            "control_ceiling_q50": 0.0,
        }
    }
    _annotate_candidates(rows, fitted)
    matches = _match_candidates(rows, fitted)
    assert len(matches) == 1
    assert matches[0]["control_position"] == 1
    assert matches[0]["entropy_closure_difference"] == 2.0


def test_ridge_freeze_contains_coefficients_and_screening_target():
    rows = []
    for task_index, task in enumerate(("gsm8k", "math500")):
        for query in range(10):
            value = query / 10
            rows.append(
                {
                    "task": task,
                    "task_math500": float(task_index),
                    "prompt_length": value,
                    "output_length": 2 * value,
                    "mean_eligible_entropy": 3 * value,
                    "mean_commit_confidence": 1 - value,
                    "bci": value + task_index,
                    "screening_coverage_auc_advantage": 0.2 * value + 0.1 * task_index,
                }
            )
    fitted = _fit_predictor(rows, ["prompt_length", "bci"])
    assert fitted["target"].startswith("screening_coverage_auc")
    assert fitted["feature_order"] == [
        "intercept",
        "task_math500",
        "prompt_length",
        "bci",
    ]
    assert np.isfinite(list(fitted["coefficients"].values())).all()
