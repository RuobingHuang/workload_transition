import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis.io_log import read_trial_log
from analysis.preprocess import find_track_start_t0, extract_track_performance
from analysis.schedule import build_segments, build_transition_events_split
from analysis.config import SCHEDULES

SCHEDULE_OFFSET_SEC = 30.0  # 你的自动段长度
OUTDIR = Path("qc_plots")

LEVEL_COLOR = {
    "LOW":  (0.8, 0.9, 1.0, 0.18),
    "MID":  (0.8, 1.0, 0.8, 0.18),
    "HIGH": (1.0, 0.8, 0.8, 0.18),
    "TAIL": (0.9, 0.9, 0.9, 0.18),
}

def compute_schedule0(df_raw: pd.DataFrame, offset_sec: float = SCHEDULE_OFFSET_SEC) -> float:
    t_track0 = find_track_start_t0(df_raw)
    return t_track0 + offset_sec

def add_schedule_background(ax, condition: str):
    seg = build_segments(SCHEDULES[condition])
    for _, r in seg.iterrows():
        x0 = r["start_ms"] / 1000.0
        x1 = r["end_ms"] / 1000.0
        lvl = r["level"]
        ax.axvspan(x0, x1, color=LEVEL_COLOR.get(lvl, (0.9, 0.9, 0.9, 0.12)))

def add_transition_lines(ax, condition: str):
    reg, tail = build_transition_events_split(SCHEDULES[condition])
    all_ev = pd.concat([reg, tail], ignore_index=True) if (len(reg) or len(tail)) else pd.DataFrame()
    for _, ev in all_ev.iterrows():
        ax.axvline(ev["t_transition_ms"]/1000.0, color="k", lw=0.8, alpha=0.35)

def downsample_for_plot(perf: pd.DataFrame, target_hz: float = 10.0) -> pd.DataFrame:
    """
    perf is ~50Hz; downsample to ~target_hz for clearer plots.
    """
    if len(perf) < 10:
        return perf
    dt_ms = perf["elapsed_ms"].diff().median()
    if not np.isfinite(dt_ms) or dt_ms <= 0:
        return perf
    step = max(1, int(round((1000.0/target_hz) / dt_ms)))
    return perf.iloc[::step].copy()

def qc_one_trial(log_path: str, participant_id: str, condition: str, metric: str = "center_deviation"):
    df_raw = read_trial_log(log_path)
    t_schedule0 = compute_schedule0(df_raw, SCHEDULE_OFFSET_SEC)
    perf = extract_track_performance(df_raw, t_schedule0)  # t_sec=0 is schedule start

    # 基本采样检查
    dt = perf["elapsed_ms"].diff().median()
    n = len(perf)

    perf_plot = downsample_for_plot(perf, target_hz=10.0)

    fig, ax = plt.subplots(figsize=(14, 4))
    add_schedule_background(ax, condition)
    add_transition_lines(ax, condition)

    ax.plot(perf_plot["t_sec"], perf_plot[metric], lw=1.0, color="#1f77b4")
    ax.set_title(f"{participant_id} | {condition} | n={n} | median_dt_ms={dt:.1f}")
    ax.set_xlabel("Time since schedule start (s)")
    ax.set_ylabel(metric)
    ax.set_xlim(0, max(1, perf["t_sec"].max()))
    ax.grid(True, alpha=0.2)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"{participant_id}__{condition}__{metric}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)

def main():
    idx = pd.read_csv("E:\project\workload_transition\data\expdata\index.csv")

    # 你也可以只跑某个条件或某个被试先看看
    # idx = idx[idx["participant_id"].isin(["10_CYW"])]
    # idx = idx[idx["condition"].isin(["LI_HF"])]

    for r in idx.itertuples(index=False):
        qc_one_trial(r.log_path, r.participant_id, r.condition, metric="center_deviation")

    print(f"Saved QC plots to: {OUTDIR.resolve()}")

if __name__ == "__main__":
    main()