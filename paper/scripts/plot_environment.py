"""Shared plotting environment for the manuscript figures.

Bit-identical figures need the same package versions and the same fonts on
every machine. This module

* resets Matplotlib to its built-in defaults, so a local ``matplotlibrc`` or
  style sheet cannot change the output;
* pins the fonts to the DejaVu families bundled with Matplotlib, so no system
  font lookup is involved; and
* compares the installed package versions with ``requirements-figures.txt``
  and reports any mismatch, because a different Matplotlib, FreeType, or
  Pillow build can change text extents and therefore the figure boxes.
"""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAPER = Path(__file__).resolve().parents[1]
REQUIREMENTS = PAPER / "requirements-figures.txt"

# Without a creation date, a rebuilt PDF is byte-identical to the previous one.
PDF_METADATA = {"CreationDate": None}

BASE_STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.hashsalt": "dllm-order-figures",
}


def pinned_versions() -> dict[str, str]:
    """Read ``name==version`` pins from requirements-figures.txt."""
    pins = {}
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, _, version = line.partition("==")
            pins[name.strip().lower()] = version.strip()
    return pins


def environment_mismatches() -> list[str]:
    """Return human-readable differences from the pinned plotting environment."""
    problems = []
    for name, expected in pinned_versions().items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = "not installed"
        if installed != expected:
            problems.append(f"{name}: installed {installed}, pinned {expected}")
    return problems


def warn_if_unpinned() -> None:
    problems = environment_mismatches()
    if problems:
        print(
            "WARNING: the plotting environment differs from requirements-figures.txt;\n"
            "figures will render, but they will not be bit-identical to the committed files:\n  "
            + "\n  ".join(problems),
            file=sys.stderr,
        )


def reset_style(overrides: dict | None = None) -> None:
    """Start from Matplotlib's defaults, then apply the pinned base style."""
    matplotlib.rcdefaults()
    plt.rcParams.update(BASE_STYLE)
    if overrides:
        plt.rcParams.update(overrides)
