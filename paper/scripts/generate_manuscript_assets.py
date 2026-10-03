#!/usr/bin/env python3
"""Generate manuscript figures and LaTeX macros from committed result reports."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.ticker
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_environment import PDF_METADATA, plt, reset_style, warn_if_unpinned  # noqa: E402


PROJECT = Path(__file__).resolve().parents[2]
PAPER = Path(__file__).resolve().parents[1]
REPORTS = PROJECT / "reports"
FIGURES = PAPER / "figures"
GENERATED = PAPER / "generated"

COLORS = {
    "ar_b1": "#1F4E79",
    "ao_b32": "#D97732",
    "random_b32": "#2A8C82",
    "ef_b32": "#8A5FA8",
    "gsm8k": "#3D6FA6",
    "math500": "#B85C38",
}
LABELS = {
    "ar_b1": "Sequential ($B=1$)",
    "ao_b32": "Confidence-first ($B=32$)",
    "random_b32": "Random ($B=32$)",
    "ef_b32": "Entropy-first ($B=32$)",
}


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def phase2_tables() -> dict:
    text = (REPORTS / "phase2_optionB_results_20260906.md").read_text(encoding="utf-8")

    primary = re.search(
        r"\| Pooled \|\s*([-+0-9.]+)\s*\|\s*\[([-+0-9.]+),\s*([-+0-9.]+)\]",
        text,
    )
    if primary is None:
        raise ValueError("Could not parse Option B primary endpoint")

    p1_block = text.split("### P1: endpoint contrast", 1)[1].split("### P2", 1)[0]
    primary_by_task = {}
    for match in re.finditer(
        r"^\|\s*(Pooled|GSM8K|MATH500)\s*\|\s*([-+0-9.]+)\s*\|\s*\[([-+0-9.]+),\s*([-+0-9.]+)\]",
        p1_block,
        flags=re.MULTILINE,
    ):
        primary_by_task[match.group(1).lower()] = {
            "point": float(match.group(2)),
            "low": float(match.group(3)),
            "high": float(match.group(4)),
        }

    trend = re.search(
        r"^\|\s*`log2\(B\)`\s*\|\s*([-+0-9.]+)\s*\|\s*\[([-+0-9.]+),\s*([-+0-9.]+)\]",
        text,
        flags=re.MULTILINE,
    )
    if trend is None:
        raise ValueError("Could not parse Option B log2(B) trend endpoint")

    absolute_block = text.split("The absolute pooled CoverageAUC values are:", 1)[1].split(
        "The adjacent pooled contrasts are:", 1
    )[0]
    absolute = {}
    for match in re.finditer(
        r"^\|\s*(1|8|16|32)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|$",
        absolute_block,
        flags=re.MULTILINE,
    ):
        block = int(match.group(1))
        absolute[block] = {
            "pooled": float(match.group(2)),
            "gsm8k": float(match.group(3)),
            "math500": float(match.group(4)),
        }

    adjacent_block = text.split("The adjacent pooled contrasts are:", 1)[1].split(
        "Accordingly,", 1
    )[0]
    adjacent = []
    for match in re.finditer(
        r"^\|\s*B=(\d+) to B=(\d+)\s*\|\s*([-+0-9.]+)\s*\|\s*\[([-+0-9.]+),\s*([-+0-9.]+)\]",
        adjacent_block,
        flags=re.MULTILINE,
    ):
        adjacent.append(
            {
                "left": int(match.group(1)),
                "right": int(match.group(2)),
                "point": float(match.group(3)),
                "low": float(match.group(4)),
                "high": float(match.group(5)),
            }
        )

    precision_block = text.split("## 5. Precision-diversity trade-off", 1)[1].split(
        "Increasing order freedom", 1
    )[0]
    precision = {}
    for match in re.finditer(
        r"^\|\s*(1|8|16|32)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|$",
        precision_block,
        flags=re.MULTILINE,
    ):
        precision[int(match.group(1))] = {
            "pass1": float(match.group(2)),
            "pass16": float(match.group(3)),
        }

    if (
        len(absolute) != 4
        or len(adjacent) != 3
        or len(precision) != 4
        or set(primary_by_task) != {"pooled", "gsm8k", "math500"}
    ):
        raise ValueError("Option B tables were parsed incompletely")

    return {
        "primary": {
            "point": float(primary.group(1)),
            "low": float(primary.group(2)),
            "high": float(primary.group(3)),
        },
        "primary_by_task": primary_by_task,
        "trend": {
            "point": float(trend.group(1)),
            "low": float(trend.group(2)),
            "high": float(trend.group(3)),
        },
        "absolute": absolute,
        "adjacent": adjacent,
        "precision": precision,
    }


def pooled_policy_curves(random_report: dict) -> dict:
    curves = {}
    for policy in ("ar_b1", "ao_b32", "random_b32", "ef_b32"):
        curves[policy] = {}
        for k in (1, 2, 4, 8, 16, 32, 64):
            key = f"pass_at_{k}"
            curves[policy][k] = np.mean(
                [random_report["summaries"][task][policy][key] for task in ("gsm8k", "math500")]
            )
    return curves


def number_macro(name: str, value: float, digits: int = 3, signed: bool = True) -> str:
    """Return a macro that typesets correctly in text and math mode.

    Wrapping the value in ``\\ensuremath`` makes a negative estimate render
    with a true minus sign even when the macro is used in running text.
    """
    rendered = f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    return f"\\newcommand{{\\{name}}}{{\\ensuremath{{{rendered}}}}}"


def write_macros(
    phase2: dict,
    condition_a: dict,
    random_report: dict,
    gate1: dict,
    postreview: dict,
    robustness: dict,
    fragility: dict,
) -> None:
    p1 = phase2["primary"]
    trend = phase2["trend"]
    step = phase2["adjacent"][0]
    ca = condition_a["primary"]["pooled"]
    rand = random_report["primary"]["pooled"]

    gate_assoc = gate1["gate1"]["condition_2_bci"]["pooled"]
    gate_mae = gate1["gate1"]["condition_2_bci"]["mae_delta"]
    new48 = postreview["random_extension"]["new48_high_budget_mean_k16_32"]
    block_grids = postreview["block_size"]["b32_minus_b1_grid_sensitivity"]
    random_sequential_grids = random_report["secondary_pairwise"]["random_minus_ar_b1"]["grids"]
    leverage = robustness["phase2_leverage"]
    dependence = leverage["p1_p2_dependence"]
    step_share = leverage["p2_exact_decomposition"]["absolute_fraction_from_step"]

    def interval(stem: str, result: dict, low: str = "ci_low", high: str = "ci_high") -> list[str]:
        return [
            number_macro(f"{stem}Point", result["point"]),
            number_macro(f"{stem}Low", result[low]),
            number_macro(f"{stem}High", result[high]),
        ]

    lines = ["% Auto-generated by scripts/generate_manuscript_assets.py. Do not edit."]
    lines += interval("OptionB", p1, "low", "high")
    # The Markdown report rounds the slope to five decimals; use the exact
    # recomputed value for the point estimate to avoid double rounding.
    exact_trend = leverage["p2_exact_decomposition"]["mean_p2"]
    if abs(exact_trend - trend["point"]) > 1e-5:
        raise ValueError("Option B trend point disagrees with the robustness audit")
    lines += [
        number_macro("OptionBTrendPoint", exact_trend, digits=4),
        number_macro("OptionBTrendLow", trend["low"], digits=4),
        number_macro("OptionBTrendHigh", trend["high"], digits=4),
    ]
    lines += interval("BlockStepOneEight", step, "low", "high")
    lines += interval("ConditionA", ca)
    lines += interval("RandomPrimary", rand)
    lines += interval("BCIRho", gate_assoc)
    lines.append(number_macro("BCIMAE", gate_mae["point"], digits=6))
    lines += interval("RandomNew", new48)
    lines += interval("RandomSequentialLow", random_sequential_grids["low_k2_4_8_16"])
    lines += interval("RandomSequentialFull", random_sequential_grids["full_k4_8_16_32_64"])
    lines += interval("BlockLowGrid", block_grids["low_k1_2_4_8"])
    lines += interval("BlockHighGrid", block_grids["high_k16_32_64"])
    lines += [
        number_macro("BlockPassOneSequential", phase2["precision"][1]["pass1"], signed=False),
        number_macro("BlockPassOneWide", phase2["precision"][32]["pass1"], signed=False),
        number_macro("BlockPassSixteenSequential", phase2["precision"][1]["pass16"], signed=False),
        number_macro("BlockPassSixteenWide", phase2["precision"][32]["pass16"], signed=False),
        number_macro("BlockEndpointPearson", dependence["pearson"], signed=False),
        number_macro("BlockEndpointSpearman", dependence["spearman_tie_aware"], signed=False),
        f"\\newcommand{{\\BlockTrendStepShare}}{{{100 * step_share:.0f}\\%}}",
    ]
    concentration = fragility["concentration"]
    removal = fragility["most_negative_removed_first"]
    lines += [
        f"\\newcommand{{\\FragilitySmallEffectCount}}{{{concentration['abs_effect_below_0_01']}}}",
        f"\\newcommand{{\\FragilityTopTenShare}}"
        f"{{{100 * concentration['share_by_most_negative']['10']:.0f}\\%}}",
        f"\\newcommand{{\\FragilityRemovedForZero}}"
        f"{{{removal['first_removed_where_interval_reaches_zero']}}}",
        "",
    ]
    (GENERATED / "paper_numbers.tex").write_text("\n".join(lines), encoding="utf-8")


def write_postreview_tables(postreview: dict, temperature: dict) -> None:
    majority = postreview["majority_vote"]
    labels = {
        "ar_b1": "Sequential ($B=1$)",
        "ao_b32": "CF ($B=32$)",
        "random_b32": "RND ($B=32$)",
        "ef_b32": "EF ($B=32$)",
    }
    lines = [
        "% Auto-generated by scripts/generate_manuscript_assets.py. Do not edit.",
        "\\newcommand{\\PostreviewMajorityVoteRows}{%",
    ]
    for policy in labels:
        values = " & ".join(
            f"{majority[policy][str(k)]['pooled']:.3f}" for k in (1, 2, 4, 8, 16, 32, 64)
        )
        lines.append(f"{labels[policy]} & {values} \\\\")
    lines.append("}")
    (GENERATED / "postreview_majority_vote_rows.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    accounting = postreview["generation_accounting"]
    account_labels = {
        "phase0_signal": "Phase-0 signal",
        "phase1_calibration": "Phase-1 calibration",
        "phase1_heldout_screening": "Held-out screening",
        "phase1_primary_extension": "Primary extension",
        "phase2_block_size_extension": "Block-size extension",
        "condition_a_entropy_first": "Entropy-first intervention",
        "random_policy_extension": "Random-policy extension",
    }
    lines = [
        "% Auto-generated by scripts/generate_manuscript_assets.py. Do not edit.",
        "\\newcommand{\\GenerationAccountingRows}{%",
    ]
    for key, label in account_labels.items():
        lines.append(f"{label} & {accounting['rows'][key]:,} \\\\")
    new_generations = temperature["validation"]["new_artifacts"]["present_and_valid"]
    lines.append(f"Temperature extension & {new_generations:,} \\\\")
    lines.append(f"\\textbf{{Total}} & \\textbf{{{accounting['total'] + new_generations:,}}} \\\\")
    lines.append("}")
    (GENERATED / "generation_accounting_rows.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    difficulty = postreview["math_difficulty"]["levels"]
    lines = [
        "% Auto-generated by scripts/generate_manuscript_assets.py. Do not edit.",
        "\\newcommand{\\MathDifficultyRows}{%",
    ]
    for level, result in difficulty.items():
        policies = result["policies"]
        values = [
            policies[key]["coverage_auc_k4_8_16_32_64"]
            for key in ("ar_b1", "ao_b32", "random_b32", "ef_b32")
        ]
        formatted = " & ".join(f"{value:.3f}" for value in values)
        lines.append(f"{level.replace('Level ', '')} & {result['queries']} & {formatted} \\\\")
    lines.append("}")
    (GENERATED / "postreview_math_difficulty_rows.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


# All data figures are placed in one ACL column (3.03 in). Drawing them at
# that size keeps every label at roughly 6-7 pt in the printed PDF.
COLUMN_WIDTH = 3.1


def configure_plotting() -> None:
    reset_style(
        {
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7.2,
            "legend.fontsize": 6.3,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02, metadata=PDF_METADATA)
    fig.savefig(FIGURES / f"{stem}.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def figure_block_size(phase2: dict) -> None:
    fig, axes = plt.subplots(
        1, 2, figsize=(COLUMN_WIDTH, 1.95), gridspec_kw={"width_ratios": [1.25, 0.75]}
    )
    blocks = np.array([1, 8, 16, 32])
    x = np.log2(blocks)
    ax = axes[0]
    ax.axvspan(0.12, 2.88, facecolor="#F4F4F4", edgecolor="#D0D0D0", hatch="///", lw=0, zorder=-3)
    ax.text(
        1.5,
        -0.074,
        "$B=2,4$\nnot measured",
        ha="center",
        va="center",
        fontsize=5.8,
        color="#555555",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
    )
    ax.axhline(0, color="#777777", lw=0.6)
    series = (
        ("gsm8k", "GSM8K", COLORS["gsm8k"], "o", 0.9, 0.8),
        ("math500", "MATH-500", COLORS["math500"], "o", 0.9, 0.8),
        ("pooled", "pooled", "#222222", "D", 1.5, 1.0),
    )
    offsets = {"gsm8k": -0.16, "math500": 0.16, "pooled": 0.0}
    handles = []
    for key, label, color, marker, lw, alpha in series:
        base = phase2["absolute"][1][key]
        vals = np.array([phase2["absolute"][int(b)][key] - base for b in blocks])
        # The B=1 -> B=8 segment crosses unmeasured settings, so it is dotted.
        ax.plot(x[:2], vals[:2], ls=":", lw=lw, color=color, alpha=alpha)
        ax.plot(x[1:], vals[1:], ls="-", lw=lw, color=color, alpha=alpha)
        handle, = ax.plot(x, vals, ls="none", marker=marker, ms=3.0, color=color, alpha=alpha, label=label)
        handles.append(handle)
        primary = phase2["primary_by_task"][key]
        ax.errorbar(
            x[-1] + 0.62 + offsets[key],
            primary["point"],
            yerr=[[primary["point"] - primary["low"]], [primary["high"] - primary["point"]]],
            fmt=marker,
            ms=2.6,
            capsize=1.5,
            lw=0.8,
            color=color,
            alpha=alpha,
        )
    ax.text(
        x[-1] + 0.62,
        0.010,
        "frozen\nP1 CI",
        ha="center",
        va="top",
        fontsize=5.6,
        color="#444444",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
    )
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
    for tick in ax.get_xticklabels()[1:3]:
        tick.set_color("#9A9A9A")
    ax.set_xlim(-0.3, 6.05)
    ax.set_ylim(-0.087, 0.012)
    ax.set_xlabel("block size $B$ (log scale)")
    ax.set_ylabel(r"$\Delta$ CoverageAUC vs. $B=1$")
    ax.set_title("a  Change vs. sequential", loc="left", weight="bold")

    labels = [f"{row['left']}$\\to${row['right']}" for row in phase2["adjacent"]]
    points = np.array([row["point"] for row in phase2["adjacent"]])
    lows = np.array([row["low"] for row in phase2["adjacent"]])
    highs = np.array([row["high"] for row in phase2["adjacent"]])
    err = np.vstack([points - lows, highs - points])
    axes[1].axhline(0, color="#777777", lw=0.6)
    axes[1].errorbar(np.arange(3), points, yerr=err, fmt="o", ms=3.2, capsize=2, lw=0.9, color="#333333")
    axes[1].set_xticks(np.arange(3))
    axes[1].set_xticklabels(labels, fontsize=5.8, rotation=35, ha="right", rotation_mode="anchor")
    axes[1].set_xlim(-0.5, 2.5)
    axes[1].set_xlabel("adjacent change in $B$")
    axes[1].set_ylabel(r"$\Delta$ CoverageAUC")
    axes[1].set_title("b  Adjacent", loc="left", weight="bold")
    fig.legend(
        handles,
        [h.get_label() for h in handles],
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        handletextpad=0.3,
        columnspacing=1.2,
    )
    fig.tight_layout(w_pad=0.8, rect=(0, 0.07, 1, 1))
    save_figure(fig, "fig2_block_size")


def figure_policy_frontier(random_report: dict, multiplicity: dict) -> None:
    curves = pooled_policy_curves(random_report)
    k = np.array([1, 2, 4, 8, 16, 32, 64])
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH, 2.15))
    styles = {
        "ar_b1": ("D", "-"),
        "ao_b32": ("o", "-"),
        "random_b32": ("s", "-"),
        "ef_b32": ("^", "--"),
    }
    short = {"ar_b1": "Sequential ($B=1$)", "ao_b32": "CF ($B=32$)", "random_b32": "RND ($B=32$)", "ef_b32": "EF ($B=32$)"}
    handles = []
    for policy in styles:
        marker, line = styles[policy]
        handle, = axes[0].plot(
            k,
            [curves[policy][int(v)] for v in k],
            marker=marker,
            ls=line,
            lw=1.1,
            ms=2.6,
            color=COLORS[policy],
            label=short[policy],
        )
        handles.append(handle)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(k)
    axes[0].set_xticklabels([str(v) for v in k])
    axes[0].set_ylim(0.35, 0.91)
    axes[0].set_xlabel("sampling budget $k$")
    axes[0].set_ylabel("pooled Pass@$k$")
    axes[0].set_title("a  Frontiers", loc="left", weight="bold")

    # Both bands come from the same post-outcome multiplicity audit so that the
    # panel matches the appendix table exactly.
    per_k = multiplicity["per_k"]
    point = np.array([per_k[str(v)]["point"] for v in k])
    raw = np.array([per_k[str(v)]["raw_95_ci"] for v in k])
    simultaneous = np.array([per_k[str(v)]["max_deviation_95_simultaneous_ci"] for v in k])
    ax = axes[1]
    ax.axvspan(16 / 1.3, 64 * 1.3, color="#EDEDED", zorder=-3, lw=0)
    ax.axhline(0, color="#666666", lw=0.6)
    raw_band = ax.fill_between(k, raw[:, 0], raw[:, 1], color=COLORS["random_b32"], alpha=0.2, linewidth=0)
    sim_line, = ax.plot(k, simultaneous[:, 0], ls="--", lw=0.7, color=COLORS["random_b32"])
    ax.plot(k, simultaneous[:, 1], ls="--", lw=0.7, color=COLORS["random_b32"])
    ax.plot(k, point, marker="s", color=COLORS["random_b32"], lw=1.1, ms=2.6)
    ax.set_xscale("log", base=2)
    ax.set_xticks(k)
    ax.set_xticklabels([str(v) for v in k])
    ax.set_xlim(0.8, 83)
    ax.set_ylim(-0.15, 0.105)
    ax.set_xlabel("sampling budget $k$")
    ax.set_ylabel("RND $-$ CF Pass@$k$", labelpad=1)
    ax.set_title("b  RND $-$ CF", loc="left", weight="bold")
    ax.legend(
        [raw_band, sim_line],
        ["pointwise 95%", "simultaneous 95%"],
        frameon=False,
        loc="lower right",
        fontsize=5.8,
        handlelength=1.4,
        borderaxespad=0.2,
    )
    fig.legend(
        handles,
        [h.get_label() for h in handles],
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        columnspacing=1.2,
        handlelength=1.8,
    )
    fig.tight_layout(w_pad=0.6, rect=(0, 0.15, 1, 1))
    save_figure(fig, "fig3_policy_frontier")


def figure_diversity(random_report: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH, 1.7))
    short = {"ar_b1": "Seq.", "ao_b32": "CF", "random_b32": "RND", "ef_b32": "EF"}
    markers = {"ar_b1": "D", "ao_b32": "o", "random_b32": "s", "ef_b32": "^"}
    for ax, task, title in zip(axes, ("gsm8k", "math500"), ("GSM8K", "MATH-500")):
        for policy in ("ar_b1", "ao_b32", "random_b32", "ef_b32"):
            summary = random_report["summaries"][task][policy]
            entropy = summary["diversity"]["k64"]["mean_answer_entropy"]
            high_coverage = np.mean([summary[f"pass_at_{k}"] for k in (16, 32, 64)])
            ax.scatter(entropy, high_coverage, s=16, color=COLORS[policy], marker=markers[policy], zorder=3)
            ax.annotate(short[policy], (entropy, high_coverage), xytext=(3, 2), textcoords="offset points", fontsize=6.2)
        ax.grid(color="#DDDDDD", lw=0.4, alpha=0.65)
        ax.margins(x=0.2, y=0.22)
        ax.set_xlabel("answer entropy at $k=64$")
        ax.set_title(title, weight="bold")
    axes[0].set_ylabel("mean Pass@$k$, $k\\geq16$")
    fig.tight_layout(w_pad=0.8)
    save_figure(fig, "fig4_diversity")

def write_temperature_assets(report: dict, exploratory: dict, postreview: dict) -> None:
    lines = ["% Auto-generated from the audited temperature reports. Do not edit."]

    def macro(name: str, value: float, signed: bool = False) -> None:
        lines.append(number_macro(name, value, signed=signed))

    for key, label in (("t09", "Nine"), ("t12", "Twelve")):
        for endpoint, endpoint_label in (("low", "Low"), ("high", "High")):
            result = report["primary"]["endpoints"][f"{key}_{endpoint}"]
            stem = f"Temp{label}{endpoint_label}"
            macro(stem + "Point", result["point"], signed=True)
            macro(stem + "Lower", result["simultaneous_95_ci"][0], signed=True)
            macro(stem + "Upper", result["simultaneous_95_ci"][1], signed=True)
    result = exploratory["interactions"]["random_vs_cf_t1.2_minus_t0.6_low"]
    macro("TempInteractionPoint", result["point"], signed=True)
    macro("TempInteractionLower", result["raw_95_ci"][0], signed=True)
    macro("TempInteractionUpper", result["raw_95_ci"][1], signed=True)
    # Post-hoc direct interaction on the shared window k in {16, 32} with 32
    # rollouts per cell: [RND - CF]_T minus [RND - CF]_{T=0.6}.
    for temp, label in (("0.9", "Nine"), ("1.2", "Twelve")):
        result = exploratory["interactions"][f"random_vs_cf_t{temp}_minus_t0.6_high"]
        macro(f"TempInteractionHigh{label}Point", result["point"], signed=True)
        macro(f"TempInteractionHigh{label}Lower", result["raw_95_ci"][0], signed=True)
        macro(f"TempInteractionHigh{label}Upper", result["raw_95_ci"][1], signed=True)
    new_count = report["validation"]["new_artifacts"]["present_and_valid"]
    total = postreview["generation_accounting"]["total"] + new_count
    lines.append(f"\\newcommand{{\\TemperatureNewCount}}{{{new_count:,}}}")
    lines.append(f"\\newcommand{{\\StudyGenerationCount}}{{{total:,}}}")

    trace = exploratory["trace_sample"]["cells"]
    for temp, label in (("0.6", "Base"), ("0.9", "Mid"), ("1.2", "High")):
        macro(f"TempCFEntropy{label}", trace[temp]["confidence_first"]["selected_entropy"])
    for temp, label in (("0.6", "Base"), ("1.2", "High")):
        curve = report["secondary_curves"][temp]["random"]["equal_task_pooled"]
        vote = report["secondary_diagnostics"][temp]["random"]["by_k"]["32"]
        macro(f"TempRandomGap{label}", curve["32"] - vote["majority_vote_accuracy_ties_fractional"])

    # Descriptive T=0.6 reference under the temperature-extension functional:
    # the reused first 32 rollouts, Pass@1 and the mean of Pass@16 and Pass@32.
    # It is not part of the frozen four-endpoint family and has no interval.
    reference = report["secondary_within_temperature_contrasts"]["0.6"]["random_minus_confidence_first"]
    reference_low = reference["1"]
    reference_high = (reference["16"] + reference["32"]) / 2
    macro("TempSixLowPoint", reference_low, signed=True)
    macro("TempSixHighPoint", reference_high, signed=True)

    lines.append("\\newcommand{\\TemperaturePrimaryRows}{%")
    lines.append(
        f"0.6$^{{\\dagger}}$ & ${reference_low:+.3f}$ & \\multicolumn{{1}}{{c}}{{--}} "
        f"& ${reference_high:+.3f}$ & \\multicolumn{{1}}{{c}}{{--}} \\\\"
    )
    lines.append(r"\midrule")
    for temp, key in (("0.9", "t09"), ("1.2", "t12")):
        row = [temp]
        for endpoint in ("low", "high"):
            result = report["primary"]["endpoints"][f"{key}_{endpoint}"]
            low, high = result["simultaneous_95_ci"]
            row.extend([f"${result['point']:+.3f}$", f"$[{low:+.3f},{high:+.3f}]$"])
        lines.append(" & ".join(row) + r" \\")
    lines.append("}")

    labels = {"confidence_first": "CF", "random": "RND", "sequential": "Sequential"}
    lines.append("\\newcommand{\\TemperatureDiagnosticRows}{%")
    for temp in ("0.6", "0.9", "1.2"):
        for schedule, label in labels.items():
            curve = report["secondary_curves"][temp][schedule]["equal_task_pooled"]
            diagnostic = report["secondary_diagnostics"][temp][schedule]
            k32 = diagnostic["by_k"]["32"]
            row = [temp, label, f"{curve['1']:.3f}", f"{curve['32']:.3f}",
                   f"{k32['majority_vote_accuracy_ties_fractional']:.3f}",
                   f"{diagnostic['quality']['mean_whitespace_tokens']:.1f}",
                   f"{k32['mean_unique_parsed_answers']:.2f}"]
            lines.append(" & ".join(row) + r" \\")
    lines.append("}")
    lines.append("\\newcommand{\\TemperatureTraceRows}{%")
    for schedule, label in labels.items():
        row = [label]
        for temp in ("0.6", "0.9", "1.2"):
            cell = trace[temp][schedule]
            row.extend([f"{cell['selected_entropy']:.3f}", f"{cell['eligible_entropy']:.3f}"])
        lines.append(" & ".join(row) + r" \\")
    lines.append("}")
    (GENERATED / "temperature_numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def figure_temperature_frontier(report: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(COLUMN_WIDTH, 1.75), sharex=True, sharey=True)
    k = [1, 2, 4, 8, 16, 32]
    policies = [("confidence_first", "ao_b32", "o", "CF"),
                ("random", "random_b32", "s", "RND"),
                ("sequential", "ar_b1", "D", "Sequential")]
    for ax, temp in zip(axes, ("0.6", "0.9", "1.2")):
        # Shade the frozen high-budget window (mean of k=16 and k=32).
        ax.axvspan(16 / 1.3, 32 * 1.3, color="#EDEDED", zorder=-3, lw=0)
        for schedule, color_key, marker, label in policies:
            curve = report["secondary_curves"][temp][schedule]["equal_task_pooled"]
            ax.plot(k, [curve[str(v)] for v in k], color=COLORS[color_key],
                    marker=marker, ms=2.2, lw=1.0, label=label)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16], ["1", "4", "16"])
        ax.set_xticks(k, minor=True)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlim(0.8, 42)
        ax.set_ylim(0, 1)
        ax.set_title(f"$T={temp}$" + (" (reused)" if temp == "0.6" else ""))
        ax.grid(axis="y", color="#DDDDDD", lw=0.4)
    axes[1].set_xlabel("sampling budget $k$")
    axes[0].set_ylabel("pooled Pass@$k$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.53, 0.0), columnspacing=1.2)
    fig.tight_layout(w_pad=0.4, rect=(0, 0.09, 1, 1))
    save_figure(fig, "fig5_temperature_frontier")

def main() -> None:
    warn_if_unpinned()
    FIGURES.mkdir(parents=True, exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)
    phase2 = phase2_tables()
    condition_a = read_json(REPORTS / "heldout_analysis_v1" / "condition_a_report.json")
    random_report = read_json(
        REPORTS / "random_extension_results_20260914" / "random_extension_report.json"
    )
    gate1 = read_json(REPORTS / "heldout_analysis_v1" / "phase1_gate1_report.json")
    postreview = read_json(
        REPORTS / "postreview_cpu_audits_20260914" / "postreview_cpu_audits.json"
    )
    robustness = read_json(REPORTS / "posthoc_robustness_20260912" / "posthoc_robustness.json")
    fragility = read_json(
        REPORTS / "block_size_fragility_audit_20261002" / "block_size_fragility_audit.json"
    )
    multiplicity = read_json(
        REPORTS / "random_extension_results_20260914" / "random_extension_multiplicity_audit.json"
    )
    temperature_dir = REPORTS / "temperature_extension_results_20260927"
    temperature = read_json(temperature_dir / "temperature_extension_analysis_v1.json")
    exploratory = read_json(temperature_dir / "temperature_extension_exploratory_v2.json")
    configure_plotting()
    write_macros(phase2, condition_a, random_report, gate1, postreview, robustness, fragility)
    write_postreview_tables(postreview, temperature)
    write_temperature_assets(temperature, exploratory, postreview)
    # Figure 1 is a manually reviewed asset; do not overwrite it during builds.
    figure_block_size(phase2)
    figure_policy_frontier(random_report, multiplicity)
    figure_diversity(random_report)
    figure_temperature_frontier(temperature)
    print(f"Generated manuscript assets in {PAPER}")


if __name__ == "__main__":
    main()
