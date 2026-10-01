"""Validation and loading helpers for the pinned JustGRPO evaluator."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .protocol import JUSTGRPO_COMMIT


@dataclass(frozen=True)
class Gsm8kGrader:
    extract_reference: Callable[[str], str]
    extract_prediction: Callable[[str], str]
    equals: Callable[[str, str], bool]

    def grade(self, reference: str, response: str) -> tuple[str, bool]:
        prediction = self.extract_prediction(response)
        target = self.extract_reference(reference)
        return prediction, bool(self.equals(target, prediction))


@dataclass(frozen=True)
class Math500Grader:
    extract_answer: Callable[[str], str]
    equals_with_timeout: Callable[[str, str], bool]

    def grade(self, reference: str, response: str) -> tuple[str, bool]:
        prediction = self.extract_answer(response)
        target = self.extract_answer(reference)
        return prediction, bool(self.equals_with_timeout(target, prediction))


def _git_output(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def validate_justgrpo_root(root: str | Path) -> Path:
    path = Path(root).expanduser().resolve()
    if not (path / "utils" / "generate.py").is_file():
        raise FileNotFoundError(f"not a JustGRPO checkout: {path}")
    commit = _git_output(path, "rev-parse", "HEAD")
    if commit != JUSTGRPO_COMMIT:
        raise RuntimeError(
            f"JustGRPO commit mismatch: expected {JUSTGRPO_COMMIT}, found {commit}"
        )
    status_lines = _git_output(path, "status", "--porcelain").splitlines()
    substantive_changes = [
        line
        for line in status_lines
        if "__pycache__/" not in line and not line.rstrip().endswith(".pyc")
    ]
    if substantive_changes:
        raise RuntimeError(
            "JustGRPO checkout contains non-cache changes and is not frozen: "
            + "; ".join(substantive_changes[:5])
        )
    return path


def load_gsm8k_grader(root: str | Path) -> Gsm8kGrader:
    path = validate_justgrpo_root(root)
    sys.path.insert(0, str(path))
    try:
        from data.math import extract_answer_gsm8k
        from utils.grader import math_equal
        from utils.parser import extract_answer
    except ImportError as error:
        raise RuntimeError(
            "Unable to import the pinned JustGRPO grader. Install its requirements first."
        ) from error
    finally:
        if sys.path[0] == str(path):
            sys.path.pop(0)
    return Gsm8kGrader(
        extract_reference=extract_answer_gsm8k,
        extract_prediction=extract_answer,
        equals=math_equal,
    )


def load_math500_grader(root: str | Path) -> Math500Grader:
    path = validate_justgrpo_root(root)
    sys.path.insert(0, str(path))
    try:
        from utils.grader import math_equal
        from utils.parser import extract_answer
    except ImportError as error:
        raise RuntimeError(
            "Unable to import the pinned JustGRPO MATH-500 grader. "
            "Install its requirements first."
        ) from error
    finally:
        if sys.path[0] == str(path):
            sys.path.pop(0)

    def equals_with_timeout(target: str, prediction: str) -> bool:
        return bool(math_equal(target, prediction, timeout=True))

    return Math500Grader(
        extract_answer=extract_answer,
        equals_with_timeout=equals_with_timeout,
    )
