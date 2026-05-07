"""
Analyze whether eye-tracking metrics support the performance mechanism story.

Inputs:
    eye_timecourse_200ms.csv
    rq1_trial_aggregates_from_events_2_3.csv
    data/expdata/index.csv

Outputs:
    eye_trial_window_metrics.csv
    eye_mechanism_merged_with_performance.csv
    eye_mechanism_correlations.csv
    figures/eye_mechanism_summary.png
    figures/eye_performance_correlation.png

The script summarizes Tobii time-course data into interpretable windows, then
links those eye metrics to event-level performance disruption and recovery.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.anova import AnovaRM


EYE_TIMECOURSE = Path("eye_timecourse_200ms.csv")
PERF_CSV = Path("rq1_trial_aggregates_from_events_2_3.csv")
INDEX_CSV = Path("data/expdata/index.csv")
OUT_DIR = Path("figures")

OUT_EYE_METRICS = Path("eye_trial_window_metrics.csv")
OUT_MERGED = Path("eye_mechanism_merged_with_performance.csv")
OUT_CORR = Path("eye_mechanism_correlations.csv")
OUT_ANOVA = Path("eye_mechanism_rmanova.csv")
OUT_FIG_SUMMARY = OUT_DIR / "eye_mechanism_summary.png"
OUT_FIG_CORR = OUT_DIR / "eye_performance_correlation.png"

TRANSITION_CONDITIONS = ["LI_LF", "HI_LF", "LI_MF", "HI_MF", "LI_HF", "HI_HF"]
FREQ_ORDER = ["LF", "MF", "HF"]
INTENSITY_ORDER = ["LI", "HI"]

WINDOWS = {
    "early": (0, 60),
    "middle": (60, 180),
    "late": (180, 280),
}

EYE_METRICS = [
    "pupil_mean",
    "fixation_prop",
    "saccade_rate_hz",
    "saccade_amplitude",
]

PERF_METRICS = [
    "imm_mean",
    "peak_mean",
    "recover_success_rate",
    "resid_5_10s_mean",
]

LABELS = {
    "pupil_mean": "Pupil diameter",
    "fixation_prop": "Fixation proportion",
    "saccade_rate_hz": "Saccade rate",
    "saccade_amplitude": "Saccade amplitude",
    "imm_mean": "Immediate deviation",
    "peak_mean": "Peak deviation",
    "recover_success_rate": "Recovery success rate",
    "resid_5_10s_mean": "Residual deviation",
}

COLORS = {"LI": "#4C78A8", "HI": "#D55E00"}


def condition_parts(condition: str) -> tuple[str, str]:
    if condition == "BASE":
        return "", ""
    intensity, freq = condition.split("_")
    return intensity, freq


def build_participant_map() -> pd.DataFrame:
    index = pd.read_csv(INDEX_CSV)
    return index[["participant_id", "participant_name"]].drop_duplicates()


def summarize_eye_windows(eye: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (participant_name, condition), group in eye.groupby(["participant_name", "condition"], sort=False):
        intensity, freq = condition_parts(condition)
        row = {
            "participant_name": participant_name,
            "condition": condition,
            "intensity": intensity,
            "freq": freq,
        }
        for window_name, (start_s, stop_s) in WINDOWS.items():
            sub = group[(group["time_s"] >= start_s) & (group["time_s"] < stop_s)]
            for metric in EYE_METRICS:
                row[f"{window_name}_{metric}"] = float(sub[metric].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def run_rmanova(eye_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    transition = eye_metrics[eye_metrics["condition"].isin(TRANSITION_CONDITIONS)].copy()
    transition["freq"] = pd.Categorical(transition["freq"], FREQ_ORDER, ordered=True)
    transition["intensity"] = pd.Categorical(transition["intensity"], INTENSITY_ORDER, ordered=True)

    for metric in [f"{window}_{eye_metric}" for window in WINDOWS for eye_metric in EYE_METRICS]:
        data = transition[["participant_name", "freq", "intensity", metric]].dropna()
        counts = data.groupby("participant_name")[metric].count()
        complete_subjects = counts[counts == 6].index
        data = data[data["participant_name"].isin(complete_subjects)]
        if data["participant_name"].nunique() < 3:
            continue
        try:
            result = AnovaRM(
                data,
                depvar=metric,
                subject="participant_name",
                within=["freq", "intensity"],
            ).fit()
            table = result.anova_table.reset_index().rename(columns={"index": "effect"})
            for rec in table.to_dict("records"):
                rows.append(
                    {
                        "metric": metric,
                        "effect": rec["effect"],
                        "F": rec["F Value"],
                        "num_df": rec["Num DF"],
                        "den_df": rec["Den DF"],
                        "p": rec["Pr > F"],
                        "n_subjects": data["participant_name"].nunique(),
                    }
                )
        except Exception as exc:
            rows.append({"metric": metric, "effect": "ERROR", "F": np.nan, "p": np.nan, "error": str(exc)})
    return pd.DataFrame(rows)


def merge_with_performance(eye_metrics: pd.DataFrame) -> pd.DataFrame:
    participant_map = build_participant_map()
    eye = eye_metrics.merge(participant_map, on="participant_name", how="left")
    perf = pd.read_csv(PERF_CSV)
    merged = perf.merge(eye, on=["participant_id", "condition", "freq", "intensity"], how="inner")
    return merged


def correlate_eye_performance(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    eye_cols = [f"{window}_{metric}" for window in WINDOWS for metric in EYE_METRICS]
    for eye_metric in eye_cols:
        for perf_metric in PERF_METRICS:
            sub = merged[[eye_metric, perf_metric]].dropna()
            if len(sub) < 5:
                continue
            r, p = stats.pearsonr(sub[eye_metric], sub[perf_metric])
            rows.append(
                {
                    "eye_metric": eye_metric,
                    "performance_metric": perf_metric,
                    "n": len(sub),
                    "r": r,
                    "p": p,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_adj_holm"] = holm_by_family(out["p"].to_numpy())
    return out


def holm_by_family(p_values: np.ndarray) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running_max = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        adj = min((m - rank) * p_values[idx], 1.0)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted.tolist()


def plot_eye_summary(eye_metrics: pd.DataFrame) -> None:
    plot_data = eye_metrics[eye_metrics["condition"].isin(TRANSITION_CONDITIONS)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.8), constrained_layout=True)
    axes = axes.ravel()
    metrics = [
        "middle_pupil_mean",
        "middle_fixation_prop",
        "middle_saccade_rate_hz",
        "middle_saccade_amplitude",
    ]

    x = np.arange(len(FREQ_ORDER))
    width = 0.34
    for ax, metric in zip(axes, metrics):
        for offset, intensity in [(-width / 2, "LI"), (width / 2, "HI")]:
            values = []
            errors = []
            for freq in FREQ_ORDER:
                sub = plot_data[(plot_data["freq"] == freq) & (plot_data["intensity"] == intensity)][metric].dropna()
                values.append(sub.mean())
                errors.append(sub.std(ddof=1) / np.sqrt(len(sub)) if len(sub) > 1 else np.nan)
            ax.bar(
                x + offset,
                values,
                width,
                yerr=errors,
                capsize=3,
                color=COLORS[intensity],
                edgecolor="black",
                linewidth=0.6,
                alpha=0.88,
                label=intensity if metric == metrics[0] else None,
            )
        ax.set_title(LABELS[metric.replace("middle_", "")])
        ax.set_xticks(x, FREQ_ORDER)
        ax.set_xlabel("Transition frequency")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, title="Intensity", loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_FIG_SUMMARY, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_correlation(merged: pd.DataFrame) -> None:
    x_metric = "middle_pupil_mean"
    y_metric = "imm_mean"
    sub = merged[[x_metric, y_metric, "intensity"]].dropna()
    fig, ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    for intensity in INTENSITY_ORDER:
        s = sub[sub["intensity"] == intensity]
        ax.scatter(s[x_metric], s[y_metric], color=COLORS[intensity], alpha=0.75, label=intensity)

    if len(sub) >= 3:
        slope, intercept, r, p, _ = stats.linregress(sub[x_metric], sub[y_metric])
        xs = np.linspace(sub[x_metric].min(), sub[x_metric].max(), 100)
        ax.plot(xs, intercept + slope * xs, color="black", lw=1.2)
        ax.text(0.04, 0.96, f"r = {r:.2f}, p = {p:.3f}", transform=ax.transAxes, ha="left", va="top")

    ax.set_xlabel("Middle-window pupil diameter")
    ax.set_ylabel("Immediate deviation")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, title="Intensity")
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_FIG_CORR, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    eye = pd.read_csv(EYE_TIMECOURSE)
    eye_metrics = summarize_eye_windows(eye)
    anova = run_rmanova(eye_metrics)
    merged = merge_with_performance(eye_metrics)
    corr = correlate_eye_performance(merged)

    eye_metrics.to_csv(OUT_EYE_METRICS, index=False)
    anova.to_csv(OUT_ANOVA, index=False)
    merged.to_csv(OUT_MERGED, index=False)
    corr.to_csv(OUT_CORR, index=False)

    plot_eye_summary(eye_metrics)
    plot_correlation(merged)

    print(f"Wrote {OUT_EYE_METRICS}")
    print(f"Wrote {OUT_ANOVA}")
    print(f"Wrote {OUT_MERGED}")
    print(f"Wrote {OUT_CORR}")
    print(f"Wrote {OUT_FIG_SUMMARY}")
    print(f"Wrote {OUT_FIG_CORR}")


if __name__ == "__main__":
    main()
