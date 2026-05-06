"""
Visualize the U-shaped frequency pattern in next-disturbance resilience.

The U-shape is evaluated with the quadratic contrast:
    U = (LF + HF) / 2 - MF

Positive U means the middle-frequency condition has a lower response than the
average of low- and high-frequency conditions. In the current data this pattern
is mainly visible under high-intensity transitions.

Inputs:
    next_disturbance_response_fixed.csv

Outputs:
    figures/next_disturbance_u_shape.png
    figures/next_disturbance_u_shape.pdf
    next_disturbance_u_shape_summary.csv
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp


INPUT_CSV = Path("next_disturbance_response_fixed.csv")
OUT_DIR = Path("figures")
OUT_PNG = OUT_DIR / "next_disturbance_u_shape.png"
OUT_PDF = OUT_DIR / "next_disturbance_u_shape.pdf"
OUT_SUMMARY = Path("next_disturbance_u_shape_summary.csv")

FREQ_ORDER = ["LF", "MF", "HF"]
METRICS = ["delta_rmse", "peak_deviation"]
METRIC_LABELS = {
    "delta_rmse": "Delta RMSE",
    "peak_deviation": "Peak deviation",
}
COLORS = {"LI": "#4C78A8", "HI": "#D55E00"}


def sem(x: pd.Series) -> float:
    x = x.dropna().astype(float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def p_text(p: float) -> str:
    if pd.isna(p):
        return "p = n/a"
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}".replace("0.", ".")


def compute_u_shape_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        for intensity in ["LI", "HI"]:
            wide = (
                df[df["intensity"] == intensity]
                .pivot(index="participant_id", columns="freq", values=metric)
                .reindex(columns=FREQ_ORDER)
            )
            contrast = ((wide["LF"] + wide["HF"]) / 2 - wide["MF"]).dropna()
            t_res = ttest_1samp(contrast, 0)

            means = wide.mean()
            sems = wide.apply(sem)
            rows.append(
                {
                    "metric": metric,
                    "intensity": intensity,
                    "lf_mean": means["LF"],
                    "mf_mean": means["MF"],
                    "hf_mean": means["HF"],
                    "lf_sem": sems["LF"],
                    "mf_sem": sems["MF"],
                    "hf_sem": sems["HF"],
                    "u_contrast": contrast.mean(),
                    "u_t": t_res.statistic,
                    "u_p": t_res.pvalue,
                    "n": len(contrast),
                }
            )
    return pd.DataFrame(rows)


def draw_u_shape(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    rng = np.random.default_rng(20260501)
    x = np.arange(len(FREQ_ORDER))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)

    for ax, metric in zip(axes, METRICS):
        for intensity in ["LI", "HI"]:
            row = summary[(summary["metric"] == metric) & (summary["intensity"] == intensity)].iloc[0]
            means = np.array([row["lf_mean"], row["mf_mean"], row["hf_mean"]], dtype=float)
            errors = np.array([row["lf_sem"], row["mf_sem"], row["hf_sem"]], dtype=float)

            ax.plot(
                x,
                means,
                color=COLORS[intensity],
                marker="o",
                markersize=6,
                linewidth=2.2,
                label=intensity,
            )
            ax.errorbar(
                x,
                means,
                yerr=errors,
                color=COLORS[intensity],
                capsize=4,
                lw=1.1,
                fmt="none",
            )

            wide = (
                df[df["intensity"] == intensity]
                .pivot(index="participant_id", columns="freq", values=metric)
                .reindex(columns=FREQ_ORDER)
            )
            for _, vals in wide.iterrows():
                vals = vals.astype(float)
                if vals.notna().sum() == 3:
                    jitter = rng.normal(0, 0.025, size=3)
                    ax.plot(
                        x + jitter,
                        vals.values,
                        color=COLORS[intensity],
                        alpha=0.13,
                        lw=0.8,
                        zorder=0,
                    )

            ax.text(
                0.98,
                0.96 if intensity == "LI" else 0.82,
                f"{intensity} U: {p_text(row['u_p'])}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                color=COLORS[intensity],
                fontsize=9,
            )

        ax.set_title(METRIC_LABELS[metric])
        ax.set_xticks(x, FREQ_ORDER)
        ax.set_xlabel("Transition frequency")
        ax.set_ylabel("Next-disturbance response")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(frameon=False, title="Intensity", loc="upper left")
    fig.suptitle("U-shaped frequency pattern under high-intensity transitions", fontsize=12)

    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    required = {"participant_id", "freq", "intensity", *METRICS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{INPUT_CSV} is missing columns: {sorted(missing)}")

    df["freq"] = pd.Categorical(df["freq"], FREQ_ORDER, ordered=True)
    summary = compute_u_shape_summary(df)
    summary.to_csv(OUT_SUMMARY, index=False)
    draw_u_shape(df, summary)

    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_SUMMARY}")
    print(summary[["metric", "intensity", "lf_mean", "mf_mean", "hf_mean", "u_contrast", "u_p"]].to_string(index=False))


if __name__ == "__main__":
    main()
