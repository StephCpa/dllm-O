#!/usr/bin/env python3
"""Draw an optional empirical teaser for Figure 1(b).

This figure is intentionally not wired into the manuscript. It uses the
post-review random-extension report to replace the conceptual curves with
pooled T=0.6 observations. Full policy/task breakdowns and uncertainty
intervals remain in Figure 3.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "random_extension_results_20260914" / "random_extension_report.json"
OUT = Path(__file__).resolve().parent

COLORS = {
    "ar_b1": "#1F4E79",
    "ao_b32": "#D97732",
    "random_b32": "#2A8C82",
}
LABELS = {
    "ar_b1": "Sequential ($B=1$)",
    "ao_b32": "Confidence-first ($B=32$)",
    "random_b32": "Random ($B=32$)",
}
TASKS = ("gsm8k", "math500")
POLICIES = ("ar_b1", "ao_b32", "random_b32")
KS = np.array([1, 2, 4, 8, 16, 32, 64])


def pooled_values(report: dict, policy: str) -> np.ndarray:
    values = []
    for k in KS:
        values.append(
            np.mean(
                [report["summaries"][task][policy][f"pass_at_{int(k)}"] for task in TASKS]
            )
        )
    return np.asarray(values)


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if not np.all(KS > 0):
        raise ValueError("Log-scaled sampling budgets must be strictly positive")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 8.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=(5.35, 3.05), dpi=600)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for policy in POLICIES:
        values = pooled_values(report, policy)
        style = {"ar_b1": "-", "ao_b32": "-", "random_b32": "--"}[policy]
        marker = {"ar_b1": "o", "ao_b32": "o", "random_b32": "s"}[policy]
        ax.plot(
            KS,
            values,
            color=COLORS[policy],
            lw=1.8,
            ls=style,
            marker=marker,
            ms=4.4,
            mec="white",
            mew=0.65,
            label=LABELS[policy],
            zorder=3,
        )

    # Highlight the empirical change in the preferred policy without adding
    # an invented threshold or uncertainty claim to this teaser panel.
    ax.axvspan(0.85, 1.35, color="#F5E5D8", alpha=0.58, lw=0, zorder=0)
    ax.axvspan(1.35, 7.0, color="#E7F2F0", alpha=0.40, lw=0, zorder=0)
    ax.annotate(
        "ranking changes\nwith budget",
        xy=(8, 0.76),
        xytext=(4.6, 0.91),
        textcoords="data",
        color="#173B6C",
        fontsize=7.3,
        fontweight="bold",
        ha="center",
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#173B6C",
            "lw": 0.9,
            "shrinkA": 2,
            "shrinkB": 3,
        },
        zorder=5,
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(KS)
    ax.set_xticklabels([str(int(k)) for k in KS])
    ax.set_xlim(0.9, 72)
    ax.set_ylim(0.25, 1.0)
    ax.set_xlabel("Sampling budget $k$ (log scale)", labelpad=4)
    ax.set_ylabel("Pooled correct-solution coverage\nPass@$k$", labelpad=5)
    ax.grid(axis="y", color="#D8E0E8", lw=0.55, alpha=0.75)
    ax.tick_params(axis="both", length=3, colors="#1F3B68")
    for y, policy in zip((0.43, 0.35, 0.27), POLICIES):
        ax.text(
            0.70,
            y,
            LABELS[policy],
            transform=ax.transAxes,
            color=COLORS[policy],
            fontsize=7.3,
            ha="left",
            va="center",
            fontweight="bold",
        )
    ax.set_title(
        "Empirical teaser: policy preference changes with $k$",
        loc="left",
        pad=9,
        color="#173B6C",
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.17, top=0.84)
    save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}
    fig.savefig(OUT / "fig1b_empirical_candidate.pdf", **save_kwargs)
    fig.savefig(OUT / "fig1b_empirical_candidate.svg", **save_kwargs)
    fig.savefig(OUT / "fig1b_empirical_candidate.tiff", dpi=600, **save_kwargs)
    fig.savefig(OUT / "fig1b_empirical_candidate.png", dpi=600, **save_kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
