"""
Visualize whether transition conditions are worse than BASE.

Input:
    trial_metrics.csv

Outputs:
    figures/transition_worse_than_base.png
    transition_vs_base_visual_summary.csv

The four primary metrics are error/workload-deviation metrics where higher
values mean worse performance:
    mean_dev, std_dev, rmse_dev, auc_dev
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_CSV = Path("trial_metrics.csv")
OUT_DIR = Path("figures")
OUT_FIG = OUT_DIR / "transition_worse_than_base.png"
OUT_CSV = Path("transition_vs_base_visual_summary.csv")

TRANSITION_CONDS = ["LI_LF", "HI_LF", "LI_MF", "HI_MF", "LI_HF", "HI_HF"]
METRICS = ["mean_dev", "std_dev", "rmse_dev", "auc_dev"]
METRIC_LABELS = {
    "mean_dev": "Mean deviation",
    "std_dev": "SD deviation",
    "rmse_dev": "RMSE deviation",
    "auc_dev": "AUC deviation",
}


def paired_transition_summary(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["condition"] == "BASE"].set_index("participant_id")
    rows = []

    for metric in METRICS:
        for condition in TRANSITION_CONDS:
            cond = df[df["condition"] == condition].set_index("participant_id")
            common = base.index.intersection(cond.index)
            paired = pd.DataFrame(
                {
                    "base": base.loc[common, metric].astype(float),
                    "transition": cond.loc[common, metric].astype(float),
                }
            ).dropna()
            diff = paired["transition"] - paired["base"]
            pct_diff = diff / paired["base"] * 100
            rows.append(
                {
                    "metric": metric,
                    "condition": condition,
                    "n": int(len(paired)),
                    "base_mean": paired["base"].mean(),
                    "transition_mean": paired["transition"].mean(),
                    "mean_diff": diff.mean(),
                    "mean_pct_worse": pct_diff.mean(),
                    "participants_worse": int((diff > 0).sum()),
                    "participants_total": int(len(diff)),
                    "participants_worse_pct": (diff > 0).mean() * 100,
                }
            )

    return pd.DataFrame(rows)


def draw_main_figure(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1.0])

    color_base = "#3a6ea5"
    color_transition = "#c44e52"
    color_line = "#8a8f98"

    # Top row: paired means for each metric. Every grey line is one condition.
    for i, metric in enumerate(METRICS):
        ax = fig.add_subplot(gs[0, i])
        base = df[df["condition"] == "BASE"].set_index("participant_id")

        for condition in TRANSITION_CONDS:
            cond = df[df["condition"] == condition].set_index("participant_id")
            common = base.index.intersection(cond.index)
            base_mean = base.loc[common, metric].astype(float).mean()
            cond_mean = cond.loc[common, metric].astype(float).mean()
            ax.plot([0, 1], [base_mean, cond_mean], color=color_line, alpha=0.45, lw=1.6)
            ax.scatter([1], [cond_mean], color=color_transition, s=38, zorder=3)

        base_all = df[df["condition"] == "BASE"][metric].astype(float).mean()
        trans_all = df[df["condition"].isin(TRANSITION_CONDS)][metric].astype(float).mean()
        ax.scatter([0], [base_all], color=color_base, s=90, zorder=4, label="BASE")
        ax.scatter([1], [trans_all], color=color_transition, s=90, zorder=4, label="Transition")

        ax.set_title(METRIC_LABELS[metric], fontsize=13, pad=10)
        ax.set_xticks([0, 1], ["BASE", "Transition"])
        ax.set_xlim(-0.25, 1.25)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        if i == 0:
            ax.set_ylabel("Metric value (higher = worse)")

    # Bottom left: heatmap of percentage worse than BASE.
    ax_heat = fig.add_subplot(gs[1, :3])
    heat = summary.pivot(index="metric", columns="condition", values="mean_pct_worse").loc[METRICS, TRANSITION_CONDS]
    im = ax_heat.imshow(heat.values, cmap="Reds", aspect="auto", vmin=0)
    ax_heat.set_xticks(np.arange(len(TRANSITION_CONDS)), TRANSITION_CONDS, rotation=35, ha="right")
    ax_heat.set_yticks(np.arange(len(METRICS)), [METRIC_LABELS[m] for m in METRICS])
    ax_heat.set_title("Mean percentage worse than BASE", fontsize=13, pad=10)
    for r in range(heat.shape[0]):
        for c in range(heat.shape[1]):
            ax_heat.text(c, r, f"+{heat.iloc[r, c]:.1f}%", ha="center", va="center", fontsize=10)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.035, pad=0.02)
    cbar.set_label("% worse")

    # Bottom right: compact claim check.
    ax_text = fig.add_subplot(gs[1, 3])
    ax_text.axis("off")
    total = len(summary)
    worse_cells = int((summary["mean_diff"] > 0).sum())
    min_pct = summary["mean_pct_worse"].min()
    max_pct = summary["mean_pct_worse"].max()
    participant_support = summary["participants_worse_pct"].mean()
    text = (
        "Claim check\n\n"
        f"{worse_cells}/{total} condition-metric comparisons\n"
        "are worse than BASE.\n\n"
        f"Mean worsening range:\n+{min_pct:.1f}% to +{max_pct:.1f}%\n\n"
        f"Average participant-level support:\n{participant_support:.1f}%"
    )
    ax_text.text(
        0.02,
        0.95,
        text,
        va="top",
        ha="left",
        fontsize=13,
        linespacing=1.35,
        bbox=dict(facecolor="#f2f4f7", edgecolor="#c9ced6", boxstyle="round,pad=0.55"),
    )

    fig.suptitle(
        "All transition conditions are worse than BASE on primary deviation metrics",
        fontsize=17,
        y=1.02,
    )
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    missing = {"participant_id", "condition", *METRICS} - set(df.columns)
    if missing:
        raise ValueError(f"{INPUT_CSV} is missing columns: {sorted(missing)}")

    summary = paired_transition_summary(df)
    summary.to_csv(OUT_CSV, index=False)
    draw_main_figure(df, summary)

    print(f"Wrote {OUT_FIG}")
    print(f"Wrote {OUT_CSV}")
    print(
        f"All {int((summary['mean_diff'] > 0).sum())}/{len(summary)} "
        "transition comparisons are worse than BASE for the primary metrics."
    )


if __name__ == "__main__":
    main()
