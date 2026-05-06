"""
Diagnose why the 45 s tail analysis has weak/non-significant results.

This script is intentionally conservative:
1. It keeps the same tail metrics produced in event_metrics_tail.csv.
2. It tests an omnibus participant-level contrast:
       mean(all transition tails) vs BASE tail
3. It reports per-condition directions without pretending they are significant.
4. It draws a figure that makes the weak tail signal easy to explain.

Outputs:
    tail_45s_diagnostic_summary.csv
    figures/tail_45s_diagnostic.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel


INPUT_CSV = Path("event_metrics_tail.csv")
OUT_CSV = Path("tail_45s_diagnostic_summary.csv")
OUT_FIG = Path("figures") / "tail_45s_diagnostic.png"

TRANSITION_CONDS = ["LI_LF", "HI_LF", "LI_MF", "HI_MF", "LI_HF", "HI_HF"]
METRICS = ["tail_mean_dev", "tail_rmse_dev", "tail_std_dev", "tail_time_to_stable_ms"]
LABELS = {
    "tail_mean_dev": "Mean deviation",
    "tail_rmse_dev": "RMSE deviation",
    "tail_std_dev": "SD deviation",
    "tail_time_to_stable_ms": "Time to stable (ms)",
}


def paired_overall_summary(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["condition"] == "BASE"].set_index("participant_id")
    trans = (
        df[df["condition"].isin(TRANSITION_CONDS)]
        .groupby("participant_id")[METRICS]
        .mean()
    )
    common = base.index.intersection(trans.index)

    rows = []
    for metric in METRICS:
        x = base.loc[common, metric].astype(float)
        y = trans.loc[common, metric].astype(float)
        ok = x.notna() & y.notna()
        x = x[ok]
        y = y[ok]
        diff = y - x
        t2 = ttest_rel(y, x)
        tg = ttest_rel(y, x, alternative="greater")
        rows.append(
            {
                "contrast": "mean_transition_tail - BASE_tail",
                "metric": metric,
                "n": int(len(diff)),
                "base_mean": x.mean(),
                "transition_mean": y.mean(),
                "mean_diff": diff.mean(),
                "cohens_dz": diff.mean() / diff.std(ddof=1),
                "p_two_sided": t2.pvalue,
                "p_one_sided_transition_worse": tg.pvalue,
                "participants_worse": int((diff > 0).sum()),
                "participants_total": int(len(diff)),
                "participants_worse_pct": (diff > 0).mean() * 100,
            }
        )
    return pd.DataFrame(rows)


def per_condition_direction(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["condition"] == "BASE"].set_index("participant_id")
    rows = []

    for metric in METRICS:
        for condition in TRANSITION_CONDS:
            cond = df[df["condition"] == condition].set_index("participant_id")
            common = base.index.intersection(cond.index)
            x = base.loc[common, metric].astype(float)
            y = cond.loc[common, metric].astype(float)
            ok = x.notna() & y.notna()
            diff = y[ok] - x[ok]
            rows.append(
                {
                    "contrast": f"{condition} - BASE_tail",
                    "metric": metric,
                    "condition": condition,
                    "n": int(len(diff)),
                    "base_mean": x[ok].mean(),
                    "transition_mean": y[ok].mean(),
                    "mean_diff": diff.mean(),
                    "participants_worse": int((diff > 0).sum()),
                    "participants_total": int(len(diff)),
                    "participants_worse_pct": (diff > 0).mean() * 100,
                }
            )
    return pd.DataFrame(rows)


def draw_figure(df: pd.DataFrame, overall: pd.DataFrame, per_cond: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1.0])

    base = df[df["condition"] == "BASE"].set_index("participant_id")
    trans = (
        df[df["condition"].isin(TRANSITION_CONDS)]
        .groupby("participant_id")[METRICS]
        .mean()
    )
    common = base.index.intersection(trans.index)

    for i, metric in enumerate(METRICS):
        ax = fig.add_subplot(gs[0, i])
        x = base.loc[common, metric].astype(float)
        y = trans.loc[common, metric].astype(float)
        for b, t in zip(x, y):
            ax.plot([0, 1], [b, t], color="#aab0b8", lw=1.1, alpha=0.7)
        ax.scatter(np.zeros(len(x)), x, color="#3a6ea5", s=35, label="BASE")
        ax.scatter(np.ones(len(y)), y, color="#c44e52", s=35, label="Transition mean")
        row = overall[overall["metric"] == metric].iloc[0]
        ax.set_title(
            f"{LABELS[metric]}\n"
            f"diff={row['mean_diff']:.2f}; p={row['p_one_sided_transition_worse']:.3f}",
            fontsize=10,
        )
        ax.set_xticks([0, 1], ["BASE", "Transition"])
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    ax_heat = fig.add_subplot(gs[1, :3])
    heat = (
        per_cond.pivot(index="metric", columns="condition", values="mean_diff")
        .loc[METRICS, TRANSITION_CONDS]
    )
    vmax = np.nanmax(np.abs(heat.values))
    im = ax_heat.imshow(heat.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax_heat.set_xticks(np.arange(len(TRANSITION_CONDS)), TRANSITION_CONDS, rotation=35, ha="right")
    ax_heat.set_yticks(np.arange(len(METRICS)), [LABELS[m] for m in METRICS])
    ax_heat.set_title("Per-condition mean difference from BASE tail", fontsize=12)
    for r in range(heat.shape[0]):
        for c in range(heat.shape[1]):
            ax_heat.text(c, r, f"{heat.iloc[r, c]:+.2f}", ha="center", va="center", fontsize=9)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.035, pad=0.02)
    cbar.set_label("Transition - BASE")

    ax_text = fig.add_subplot(gs[1, 3])
    ax_text.axis("off")
    weak = overall[["metric", "p_one_sided_transition_worse", "participants_worse", "participants_total"]]
    lines = ["45 s tail diagnosis", ""]
    for _, r in weak.iterrows():
        lines.append(
            f"{LABELS[r['metric']]}: p={r['p_one_sided_transition_worse']:.3f}, "
            f"{int(r['participants_worse'])}/{int(r['participants_total'])} worse"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "The direction is mostly worse, but participant variability is large.",
            "Averaging the full 45 s tail dilutes short-lived recovery effects.",
        ]
    )
    ax_text.text(
        0,
        1,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=11.5,
        linespacing=1.35,
        bbox=dict(facecolor="#f2f4f7", edgecolor="#c9ced6", boxstyle="round,pad=0.55"),
    )

    fig.suptitle("The 45 s tail window shows a weak, variable residual effect", fontsize=16)
    OUT_FIG.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    missing = {"participant_id", "condition", *METRICS} - set(df.columns)
    if missing:
        raise ValueError(f"{INPUT_CSV} is missing columns: {sorted(missing)}")

    overall = paired_overall_summary(df)
    per_cond = per_condition_direction(df)
    out = pd.concat([overall, per_cond], ignore_index=True, sort=False)
    out.to_csv(OUT_CSV, index=False)
    draw_figure(df, overall, per_cond)

    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_FIG}")
    print("\nOverall transition-tail mean vs BASE-tail:")
    print(
        overall[
            [
                "metric",
                "mean_diff",
                "cohens_dz",
                "p_one_sided_transition_worse",
                "participants_worse",
                "participants_total",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
