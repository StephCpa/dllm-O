from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "paper/scripts/build_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

RECORDED_TEXT = {"pages": 2, "page_sha256": ["a", "b"]}


def verdict(**overrides):
    arguments = {
        "figures_ok": True,
        "hash_match": False,
        "same_build": False,
        "recorded_text": RECORDED_TEXT,
        "current_text": dict(RECORDED_TEXT),
    }
    arguments.update(overrides)
    return MODULE.pdf_verdict(**arguments)[0]


def test_normalize_text_neutralizes_layout_but_keeps_signs() -> None:
    assert MODULE.normalize_text("de-\nploy ment") == MODULE.normalize_text("deployment")
    assert MODULE.normalize_text("ﬁrst") == "first"
    assert MODULE.normalize_text("−0.044") != MODULE.normalize_text("0.044")


def test_bit_identical_pdf_passes() -> None:
    assert verdict(hash_match=True) == MODULE.PASS


def test_any_pdf_change_fails_under_the_recorded_build() -> None:
    assert verdict(same_build=True) == MODULE.FAIL


def test_other_environment_with_matching_text_is_not_verified() -> None:
    assert verdict() == MODULE.NOT_VERIFIED


def test_other_environment_without_text_tools_is_not_verified() -> None:
    assert verdict(current_text=None) == MODULE.NOT_VERIFIED


def test_other_environment_with_changed_text_fails() -> None:
    assert verdict(current_text={"pages": 2, "page_sha256": ["a", "changed"]}) == MODULE.FAIL


def test_other_environment_with_changed_page_count_fails() -> None:
    assert verdict(current_text={"pages": 3, "page_sha256": ["a", "b", "c"]}) == MODULE.FAIL


def test_figure_drift_fails_before_the_pdf_is_judged() -> None:
    assert verdict(figures_ok=False, hash_match=True) == MODULE.FAIL


def test_write_refuses_missing_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(MODULE, "PAPER", tmp_path)
    monkeypatch.setattr(MODULE, "MANIFEST", tmp_path / "build_manifest.json")
    with pytest.raises(SystemExit, match="missing build outputs"):
        MODULE.write()
    assert not (tmp_path / "build_manifest.json").exists()


def test_check_fails_when_recorded_artifacts_are_missing(tmp_path, monkeypatch) -> None:
    manifest = {
        "environment": {"pdftex": "recorded", "pdftotext": "recorded", "source_date_epoch": "0"},
        "python_outputs": {path: "0" * 64 for path in MODULE.PYTHON_OUTPUTS},
        "tex_outputs": {MODULE.PDF: "0" * 64},
        "pdf_text": RECORDED_TEXT,
    }
    (tmp_path / "build_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(MODULE, "PAPER", tmp_path)
    monkeypatch.setattr(MODULE, "MANIFEST", tmp_path / "build_manifest.json")
    assert MODULE.check() == MODULE.FAIL
