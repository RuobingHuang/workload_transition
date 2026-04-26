"""
draw_02c_posthoc_freq_within_intensity.py

Visualise the post-hoc pairwise freq comparisons within each intensity level.

Inputs
------
rq1_trial_aggregates_from_events.csv   – per-participant × condition means
rq1_posthoc_freq_within_intensity_holm.csv – Holm-corrected paired t-test results

Outputs (saved to figures/)
---------------------------
fig_02c_posthoc_<metric>_bars.png   – grouped bar charts (LI / HI × LF / MF / HF)
                                       with significance brackets
fig_02c_posthoc_effect_sizes.png    – dot-plot of Cohen's dz for all sig. contrasts
"""

import os
import itertools

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── constants ──────────────────────────────────────────────────────────────────
FREQ_ORDER  = ["LF", "MF", "HF"]
INTENSITIES = ["LI", "HI"]

METRIC_LABELS = {
    "recover_success_rate": "Recovery Success Rate",
    "delta_mean":           "Mean Deviation (delta_mean)",
    "resid_11_14s_mean":    "Residual 11-14 s (mean)",
}

INTENSITY_COLORS = {"LI": "#4C72B0", "HI": "#DD8452"}
FREQ_COLORS      = ["#5B8DB8", "#E08D4A", "#6AAB6A"]   # LF / MF / HF

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)


# ── helpers ────────────────────────────────────────────────────────────────────
def sig_stars(p_adj: float) -> str:
    if p_adj < 0.001:
        return "***"
    if p_adj < 0.01:
        return "**"
    if p_adj < 0.05:
        return "*"
    return "ns"


def draw_bracket(ax, x1, x2, y, h, text, color="black", fontsize=8):
    """Draw a significance bracket between two x-positions."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, color=color)
    ax.text((x1 + x2) / 2, y + h * 1.1, text,
            ha="center", va="bottom", fontsize=fontsize, color=color)


# ── Figure 1 – grouped bar charts per metric ──────────────────────────────────
def plot_bars_per_metric(df_agg: pd.DataFrame, df_ph: pd.DataFrame, metric: str):
    """
    One figure with two sub-panels (LI | HI).
    Each panel: 3 bars for LF / MF / HF, plus significance brackets.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)
    fig.suptitle(METRIC_LABELS.get(metric, metric), fontsize=13, fontweight="bold")

    for ax, intensity in zip(axes, INTENSITIES):
        sub = df_agg[df_agg["intensity"] == intensity].copy()
        means = (sub.groupby("freq")[metric].mean().reindex(FREQ_ORDER))
        sems  = (sub.groupby("freq")[metric].sem().reindex(FREQ_ORDER))

        xs = np.arange(len(FREQ_ORDER))
        bars = ax.bar(xs, means, yerr=sems, capsize=5,
                      color=FREQ_COLORS, edgecolor="white", linewidth=0.8,
                      error_kw=dict(ecolor="gray", lw=1.5))

        ax.set_xticks(xs)
        ax.set_xticklabels(FREQ_ORDER, fontsize=11)
        ax.set_title(f"Intensity: {intensity}", fontsize=11)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric) if ax == axes[0] else "")
        ax.spines[["top", "right"]].set_visible(False)

        # significance brackets
        ph_sub = df_ph[(df_ph["metric"] == metric) &
                       (df_ph["intensity"] == intensity) &
                       (df_ph["reject@0.05"] == True)].copy()

        if not ph_sub.empty:
            y_max  = (means + sems).max()
            y_step = (means + sems).max() * 0.12
            for lvl, row in enumerate(ph_sub.itertuples()):
                i1 = FREQ_ORDER.index(row.freq_a)
                i2 = FREQ_ORDER.index(row.freq_b)
                y_b = y_max + y_step * (lvl + 1)
                draw_bracket(ax, i1, i2, y_b, y_step * 0.25,
                             sig_stars(row.p_adj_holm))

    legend_patches = [mpatches.Patch(color=c, label=f)
                      for c, f in zip(FREQ_COLORS, FREQ_ORDER)]
    fig.legend(handles=legend_patches, title="Freq", loc="lower center",
               ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out_path = os.path.join(OUT_DIR, f"fig_02c_posthoc_{metric}_bars.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Figure 2 – Cohen's dz dot-plot ────────────────────────────────────────────
def plot_effect_sizes(df_ph: pd.DataFrame):
    """
    Horizontal dot-plot of Cohen's dz for every contrast, grouped by metric.
    Significant contrasts are filled; non-significant are open circles.
    """
    df = df_ph.copy()
    df["label"] = df["intensity"] + "  " + df["contrast"]

    metrics = list(df["metric"].unique())
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics,
                             figsize=(5 * n_metrics, max(4, len(df) // n_metrics * 0.55 + 1)),
                             sharey=False)
    if n_metrics == 1:
        axes = [axes]

    fig.suptitle("Post-hoc Effect Sizes (Cohen's dz)\nFreq pairwise contrasts within intensity",
                 fontsize=12, fontweight="bold")

    for ax, metric in zip(axes, metrics):
        sub = df[df["metric"] == metric].sort_values(
            ["intensity", "p_adj_holm"], ascending=[True, False])

        ys     = np.arange(len(sub))
        colors = [INTENSITY_COLORS[i] for i in sub["intensity"]]
        fills  = ["full" if r else "none" for r in sub["reject@0.05"]]

        for y, dz, color, fill, row in zip(ys, sub["cohens_dz"], colors, fills, sub.itertuples()):
            marker = "o"
            if fill == "full":
                ax.plot(dz, y, marker=marker, color=color, markersize=9,
                        markeredgecolor=color, zorder=3)
            else:
                ax.plot(dz, y, marker=marker, color=color, markersize=9,
                        markerfacecolor="white", markeredgecolor=color,
                        markeredgewidth=1.5, zorder=3)
            # p-value annotation
            stars = sig_stars(row.p_adj_holm)
            ax.text(dz + 0.04, y, stars, va="center", fontsize=7.5, color=color)

        ax.axvline(0, color="gray", lw=0.8, linestyle="--")
        ax.set_yticks(ys)
        ax.set_yticklabels(sub["label"].tolist(), fontsize=8)
        ax.set_xlabel("Cohen's dz", fontsize=10)
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)

    # legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=INTENSITY_COLORS["LI"],
               markersize=8, label="LI"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=INTENSITY_COLORS["HI"],
               markersize=8, label="HI"),
        Line2D([0], [0], marker="o", color="gray", markersize=8,
               markerfacecolor="white", markeredgecolor="gray",
               markeredgewidth=1.5, label="not sig."),
        Line2D([0], [0], marker="o", color="gray", markersize=8,
               markerfacecolor="gray", label="sig. (p_adj<.05)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_path = os.path.join(OUT_DIR, "fig_02c_posthoc_effect_sizes.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    df_agg = pd.read_csv("rq1_trial_aggregates_from_events.csv")
    df_ph  = pd.read_csv("rq1_posthoc_freq_within_intensity_holm.csv")

    metrics = df_ph["metric"].unique().tolist()

    for metric in metrics:
        if metric not in df_agg.columns:
            print(f"[skip] '{metric}' not in aggregates CSV")
            continue
        plot_bars_per_metric(df_agg, df_ph, metric)

    plot_effect_sizes(df_ph)

    print("\nAll figures saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
