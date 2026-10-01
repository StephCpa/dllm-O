from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).parents[1] / "scripts/random_extension_multiplicity_audit.py"
)
SPEC = importlib.util.spec_from_file_location("random_extension_multiplicity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_pass_at_k_boundaries() -> None:
    assert MODULE.pass_at_k(64, 0, 16) == 0.0
    assert MODULE.pass_at_k(64, 64, 16) == 1.0


def test_joint_reversal_and_simultaneous_bands() -> None:
    low = np.asarray([[-0.20, -0.10, 0, 0, 0.10, 0.20, 0]] * 8)
    differences = {"gsm8k": low, "math500": low}
    report = MODULE.summarize(differences, seed=7, replicates=100)
    assert report["joint_low_high_summary"]["joint_reversal_resolved"] is True
    assert report["per_k"]["1"]["point"] == pytest.approx(-0.20)
    assert report["per_k"]["32"]["point"] == pytest.approx(0.20)
