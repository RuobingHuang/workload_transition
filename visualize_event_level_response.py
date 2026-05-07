"""
Publication-style event-level figure for immediate disruption and recovery.

Input:
    rq1_trial_aggregates_from_events_2_3.csv

Outputs:
    figures/event_level_response_publication.png
    figures/event_level_response_publication.pdf
    event_level_response_publication_summary.csv
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


INPUT_CSV = Path("rq1_trial_aggregates_from_events_2_3.csv")
OUT_DIR = Path("figures")
OUT_PNG = OUT_DIR / "event_level_response_publication.png"
OUT_PDF = OUT_DIR / "event_level_response_publication.pdf"
OUT_SUMMARY = Path("event_level_response_publication_summary.csv")
OUT_PAIRWISE = Path("event_level_response_publication_pairwise.csv")

FREQ_ORDER = ["LF", "MF", "HF"]
INTENSITY_ORDER = ["LI", "HI"]
METRICS = ["imm_mean", "peak_mean", "recover_success_rate", "resid_5_10s_mean"]
METRIC_LABELS = {
    "imm_mean": "Immediate deviation",
    "peak_mean": "Peak deviation",
    "recover_success_rate": "Recovery success rate",
    "resid_5_10s_mean": "Residual deviation (5-10 s)",
}
Y_LABELS = {
    "imm_mean": "Deviation",
    "peak_mean": "Deviation",
    "recover_success_rate": "Proportion",
    "resid_5_10s_mean": "Deviation",
}
COLORS = {"LI": "#4C78A8", "HI": "#D55E00"}
ANOVA_P = {
    "imm_mean": {"Frequency": 0.2889, "Intensity": 0.0009, "Interaction": 0.1012},
    "peak_mean": {"Frequency": 0.1243, "Intensity": 0.0009, "Interaction": 0.0663},
    "recover_success_rate": {"Frequency": 0.0004, "Intensity": 0.0009, "Interaction": 0.6325},
    "resid_5_10s_mean": {"Frequency": 0.0001, "Intensity": 0.1434, "Interaction": 0.5655},
}


def sem(values: pd.Series) -> float:
    values = values.dropna().astype(float)
    if len(values) <= 1:
        return float("nan")
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def p_text(p: float) -> str:
    if pd.isna(p):
        return "n.s."
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}".replace("0.", ".")


def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running_max = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        adj = min((m - rank) * p[idx], 1.0)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted.tolist()


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        for intensity in INTENSITY_ORDER:
            for freq in FREQ_ORDER:
                values = df[(df["intensity"] == intensity) & (df["freq"] == freq)][metric]
                values = values.dropna().astype(float)
                rows.append(
                    {
                        "metric": metric,
                        "intensity": intensity,
                        "freq": freq,
                        "mean": float(values.mean()),
                        "sem": sem(values),
                        "sd": float(values.std(ddof=1)),
                        "n": int(len(values)),
                    }
                )
    return pd.DataFrame(rows)


def paired_test(df: pd.DataFrame, metric: str, filters: dict, group_col: str, a: str, b: str) -> dict:
    sub = df.copy()
    for col, value in filters.items():
        sub = sub[sub[col] == value]
    wide = sub.pivot_table(index="participant_id", columns=group_col, values=metric, aggfunc="mean", observed=False)
    wide = wide[[a, b]].dropna()
    if len(wide) < 2:
        t_stat, p = np.nan, np.nan
    else:
        t_stat, p = stats.ttest_rel(wide[a], wide[b])
    return {
        "n": int(len(wide)),
        "t": float(t_stat) if not pd.isna(t_stat) else np.nan,
        "p": float(p) if not pd.isna(p) else np.nan,
        "mean_diff(a-b)": float((wide[a] - wide[b]).mean()) if len(wide) else np.nan,
    }


def build_pairwise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    freq_pairs = [("LF", "MF"), ("MF", "HF"), ("LF", "HF")]

    for metric in METRICS:
        for freq in FREQ_ORDER:
            result = paired_test(df, metric, {"freq": freq}, "intensity", "LI", "HI")
            rows.append(
                {
                    "metric": metric,
                    "comparison_family": "LI_vs_HI_within_frequency",
                    "freq": freq,
                    "intensity": "",
                    "group_a": "LI",
                    "group_b": "HI",
                    **result,
                }
            )

        for intensity in INTENSITY_ORDER:
            for freq_a, freq_b in freq_pairs:
                result = paired_test(
                    df,
                    metric,
                    {"intensity": intensity},
                    "freq",
                    freq_a,
                    freq_b,
                )
                rows.append(
                    {
                        "metric": metric,
                        "comparison_family": "frequency_within_intensity",
                        "freq": "",
                        "intensity": intensity,
                        "group_a": freq_a,
                        "group_b": freq_b,
                        **result,
                    }
                )

    pairwise = pd.DataFrame(rows)
    adjusted = []
    for _, group in pairwise.groupby(["metric", "comparison_family"], sort=False):
        adjusted.extend(holm_adjust(group["p"].tolist()))
    pairwise["p_adj_holm"] = adjusted
    pairwise["signif"] = pairwise["p_adj_holm"].map(stars)
    pairwise["reject@0.05"] = pairwise["p_adj_holm"] < 0.05
    return pairwise


def add_sig_bar(ax, x1: float, x2: float, y: float, text: str, height: float, color: str = "black") -> None:
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], lw=0.8, color=color, clip_on=False)
    ax.text((x1 + x2) / 2, y + height, text, ha="center", va="bottom", color=color, fontsize=9)


def anova_line(metric: str) -> str:
    p = ANOVA_P[metric]
    return " | ".join(
        [
            f"F: {p_text(p['Frequency'])}",
            f"I: {p_text(p['Intensity'])}",
            f"F x I: {p_text(p['Interaction'])}",
        ]
    )


def draw_figure(df: pd.DataFrame, summary: pd.DataFrame, pairwise: pd.DataFrame) -> None:
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

    rng = np.random.default_rng(20260507)
    x = np.arange(len(FREQ_ORDER))
    width = 0.34

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.0), constrained_layout=True)
    axes = axes.ravel()

    for ax, metric in zip(axes, METRICS):
        metric_summary = summary[summary["metric"] == metric]

        for offset, intensity in [(-width / 2, "LI"), (width / 2, "HI")]:
            s = (
                metric_summary[metric_summary["intensity"] == intensity]
                .set_index("freq")
                .loc[FREQ_ORDER]
            )
            ax.bar(
                x + offset,
                s["mean"],
                width=width,
                color=COLORS[intensity],
                alpha=0.88,
                edgecolor="black",
                linewidth=0.6,
                label=intensity if metric == METRICS[0] else None,
            )
            ax.errorbar(
                x + offset,
                s["mean"],
                yerr=s["sem"],
                color="black",
                fmt="none",
                capsize=3,
                lw=0.9,
                zorder=3,
            )

            for j, freq in enumerate(FREQ_ORDER):
                values = df[(df["intensity"] == intensity) & (df["freq"] == freq)][metric]
                values = values.dropna().astype(float)
                jitter = rng.normal(0, 0.025, size=len(values))
                ax.scatter(
                    np.full(len(values), x[j] + offset) + jitter,
                    values,
                    s=10,
                    color=COLORS[intensity],
                    alpha=0.28,
                    linewidths=0,
                    zorder=4,
                )

        ax.set_title(METRIC_LABELS[metric], fontsize=11)
        ax.set_xticks(x, FREQ_ORDER)
        ax.set_xlabel("Transition frequency")
        ax.set_ylabel(Y_LABELS[metric])
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

        metric_values = df[metric].dropna().astype(float)
        ymin = min(0, float(metric_values.min()))
        ymax = float(metric_values.max())
        span = max(ymax - ymin, 0.2)
        if metric == "recover_success_rate":
            ymin, ymax, span = 0, 1.0, 1.0

        sig_rows = pairwise[
            (pairwise["metric"] == metric)
            & (pairwise["reject@0.05"])
            & (pairwise["comparison_family"] == "LI_vs_HI_within_frequency")
        ]
        n_rows = len(sig_rows)
        ax.set_ylim(ymin, ymax + span * (0.15 + 0.06 * max(n_rows, 1)))
        bar_h = span * 0.025
        y_base = ymax + span * 0.04
        layer = 0

        for _, row in sig_rows.iterrows():
            freq_idx = FREQ_ORDER.index(row["freq"])
            add_sig_bar(
                ax,
                x[freq_idx] - width / 2,
                x[freq_idx] + width / 2,
                y_base + layer * span * 0.065,
                row["signif"],
                bar_h,
            )
            layer += 1

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, title="Intensity", loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))

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
    df["intensity"] = pd.Categorical(df["intensity"], INTENSITY_ORDER, ordered=True)

    summary = build_summary(df)
    pairwise = build_pairwise(df)
    summary.to_csv(OUT_SUMMARY, index=False)
    pairwise.to_csv(OUT_PAIRWISE, index=False)
    draw_figure(df, summary, pairwise)

    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_PAIRWISE}")


if __name__ == "__main__":
    main()
