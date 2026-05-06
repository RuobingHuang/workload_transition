"""
Publication-style visualization for next-disturbance resilience.

This figure focuses on the strongest result from
next_disturbance_response_fixed.csv:
    1. High intensity produces larger next-disturbance responses than low intensity.
    2. Within high intensity, HI_LF shows the largest impairment, especially compared
       with HI_MF.

Inputs:
    next_disturbance_response_fixed.csv
    next_disturbance_rmanova_fixed.csv
    next_disturbance_pairwise_fixed.csv

Outputs:
    figures/next_disturbance_resilience_publication.png
    figures/next_disturbance_resilience_publication.pdf
    next_disturbance_resilience_publication_summary.csv
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESPONSE_CSV = Path("next_disturbance_response_fixed.csv")
ANOVA_CSV = Path("next_disturbance_rmanova_fixed.csv")
PAIRWISE_CSV = Path("next_disturbance_pairwise_fixed.csv")
OUT_DIR = Path("figures")
OUT_PNG = OUT_DIR / "next_disturbance_resilience_publication.png"
OUT_PDF = OUT_DIR / "next_disturbance_resilience_publication.pdf"
OUT_SUMMARY = Path("next_disturbance_resilience_publication_summary.csv")

FREQ_ORDER = ["LF", "MF", "HF"]
INTENSITY_ORDER = ["LI", "HI"]
COND_ORDER = ["LI_LF", "LI_MF", "LI_HF", "HI_LF", "HI_MF", "HI_HF"]

PRIMARY_METRICS = ["delta_rmse", "post_rmse", "peak_deviation", "post_auc"]
METRIC_LABELS = {
    "delta_rmse": "Delta RMSE",
    "post_rmse": "Post-transition RMSE",
    "peak_deviation": "Peak deviation",
    "post_auc": "Post-transition AUC",
}


def sem(x: pd.Series) -> float:
    x = x.dropna().astype(float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def p_text(p: float) -> str:
    if pd.isna(p):
        return "p = n/a"
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}".replace("0.", ".")


def add_sig_bar(ax, x1: float, x2: float, y: float, text: str, height: float) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color="black", lw=1.1)
    ax.text((x1 + x2) / 2, y + height, text, ha="center", va="bottom", fontsize=9)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    resp = pd.read_csv(RESPONSE_CSV)
    anova = pd.read_csv(ANOVA_CSV)
    pairwise = pd.read_csv(PAIRWISE_CSV)

    required = {"participant_id", "condition", "freq", "intensity", *PRIMARY_METRICS}
    missing = required - set(resp.columns)
    if missing:
        raise ValueError(f"{RESPONSE_CSV} is missing columns: {sorted(missing)}")

    resp["freq"] = pd.Categorical(resp["freq"], FREQ_ORDER, ordered=True)
    resp["intensity"] = pd.Categorical(resp["intensity"], INTENSITY_ORDER, ordered=True)
    resp["condition"] = pd.Categorical(resp["condition"], COND_ORDER, ordered=True)
    return resp, anova, pairwise


def make_summary(resp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in PRIMARY_METRICS:
        for condition in COND_ORDER:
            d = resp[resp["condition"] == condition][metric].dropna().astype(float)
            rows.append(
                {
                    "metric": metric,
                    "condition": condition,
                    "intensity": condition.split("_")[0],
                    "freq": condition.split("_")[1],
                    "mean": d.mean(),
                    "sem": sem(d),
                    "sd": d.std(ddof=1),
                    "n": len(d),
                }
            )
    return pd.DataFrame(rows)


def draw_figure(resp: pd.DataFrame, anova: pd.DataFrame, pairwise: pd.DataFrame, summary: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    colors = {"LI": "#4C78A8", "HI": "#D55E00"}
    x = np.arange(len(FREQ_ORDER))
    width = 0.34

    rng = np.random.default_rng(20260501)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)
    axes = axes.ravel()

    for ax, metric in zip(axes, PRIMARY_METRICS):
        metric_summary = summary[summary["metric"] == metric]

        for offset, intensity in [(-width / 2, "LI"), (width / 2, "HI")]:
            d = metric_summary[metric_summary["intensity"] == intensity].set_index("freq").loc[FREQ_ORDER]
            ax.bar(
                x + offset,
                d["mean"],
                width=width,
                color=colors[intensity],
                alpha=0.88,
                edgecolor="black",
                linewidth=0.6,
                label=intensity if metric == PRIMARY_METRICS[0] else None,
            )
            ax.errorbar(
                x + offset,
                d["mean"],
                yerr=d["sem"],
                color="black",
                fmt="none",
                capsize=3,
                lw=0.8,
            )

            # Participant-level points, kept faint so the mean pattern remains primary.
            for j, freq in enumerate(FREQ_ORDER):
                vals = resp[(resp["intensity"] == intensity) & (resp["freq"] == freq)][metric].dropna()
                jitter = rng.normal(0, 0.025, size=len(vals))
                ax.scatter(
                    np.full(len(vals), x[j] + offset) + jitter,
                    vals,
                    s=9,
                    color=colors[intensity],
                    alpha=0.25,
                    linewidths=0,
                    zorder=2,
                )

        ax.set_title(METRIC_LABELS[metric])
        ax.set_xticks(x, FREQ_ORDER)
        ax.set_xlabel("Transition frequency")
        ax.set_ylabel("Magnitude")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

        # RM-ANOVA annotation.
        rows = anova[anova["metric"] == metric]
        p_int = rows.loc[rows["effect"] == "intensity", "p"]
        p_ixf = rows.loc[rows["effect"] == "intensity:freq", "p"]
        ann = []
        if not p_int.empty:
            ann.append(f"Intensity: {p_text(float(p_int.iloc[0]))}")
        if not p_ixf.empty:
            ann.append(f"I x F: {p_text(float(p_ixf.iloc[0]))}")
        ax.text(
            0.02,
            0.98,
            "\n".join(ann),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(facecolor="white", edgecolor="#d0d0d0", boxstyle="round,pad=0.25", alpha=0.9),
        )

        # Mark the robust HI_LF > HI_MF pairwise result where available.
        hit = pairwise[
            (pairwise["metric"] == metric)
            & (pairwise["intensity"] == "HI")
            & (pairwise["cond_a"] == "HI_LF")
            & (pairwise["cond_b"] == "HI_MF")
        ]
        if not hit.empty and bool(hit.iloc[0]["reject@0.05"]):
            y_max = metric_summary["mean"].max() + metric_summary["sem"].max()
            y_min = min(0, metric_summary["mean"].min())
            height = (y_max - y_min) * 0.04
            add_sig_bar(
                ax,
                x[0] + width / 2,
                x[1] + width / 2,
                y_max + height,
                "*",
                height,
            )
            ax.set_ylim(top=y_max + height * 3.2)

    axes[0].legend(frameon=False, title="Intensity", loc="upper right")
    fig.suptitle("High-intensity transitions reduce resilience to the next disturbance", fontsize=12)

    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    resp, anova, pairwise = load_data()
    summary = make_summary(resp)
    summary.to_csv(OUT_SUMMARY, index=False)
    draw_figure(resp, anova, pairwise, summary)

    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
