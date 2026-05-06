"""
debug_plot_next_disturbance_full_trial_fixed.py
────────────────────────────────────────────────────────────────
目的
----
检查正式分析脚本提取的目标扰动时间是否正确。

重要修正
--------
预处理后的 elapsed_ms 已经去掉了最开始 30s 自动化阶段：
  preprocessed elapsed time 0s = schedule start

因此本 debug 图直接使用 schedule transition time，不再 +30s。

功能
----
给定一个 participant_id，画该被试 6 个 transition conditions 的完整轨迹：

  x-axis: preprocessed elapsed time (s)
  y-axis: center_deviation

并在每个子图上标出：
  1. 所有 transition 时间 T1/T2/T3...
  2. 目标扰动 T2/T3/T4 的横坐标，例如 T2 = 105.0s
  3. pre / post window
  4. pre_rmse / post_rmse / delta_rmse

读取
----
  data/expdata/index.csv

输出
----
  next_disturbance_debug_fixed/
"""

import os
import sys
import argparse
import importlib.util

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


sys.path.insert(0, os.path.dirname(__file__))

from analysis.config import SCHEDULES
from analysis.io_log import read_trial_log
from analysis.preprocess import extract_track_performance, get_schedule_start_t0
from analysis.schedule import build_transition_events_split


def load_run07_module(module_path: str = "run_07_next_disturbance_resilience_fixed.py"):
    if not os.path.exists(module_path):
        raise FileNotFoundError(
            f"找不到 {module_path}。请把本脚本和 run_07_next_disturbance_resilience_fixed.py 放在同一目录，"
            f"或用 --run07_path 指定路径。"
        )

    spec = importlib.util.spec_from_file_location("run07_fixed", module_path)
    run07 = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(run07)
    return run07


def plot_one_subject_six_conditions(
    participant_id: str,
    index_csv: str = "data/expdata/index.csv",
    run07_path: str = "run_07_next_disturbance_resilience_fixed.py",
    save_dir: str = "next_disturbance_debug_fixed",
):
    os.makedirs(save_dir, exist_ok=True)

    run07 = load_run07_module(run07_path)

    transition_conds = getattr(
        run07,
        "TRANSITION_CONDS",
        ["LI_LF", "HI_LF", "LI_MF", "HI_MF", "LI_HF", "HI_HF"],
    )

    target_transition_index = run07.TARGET_TRANSITION_INDEX
    pre_window_ms = run07.PRE_WINDOW_MS
    post_window_ms = run07.POST_WINDOW_MS
    schedule_offset_ms = getattr(run07, "SCHEDULE_OFFSET_MS", 0)

    idx = pd.read_csv(index_csv)
    required = {"participant_id", "condition", "log_path"}
    missing = required - set(idx.columns)
    if missing:
        raise ValueError(f"index_csv 缺少列: {sorted(missing)}")

    sub_idx = idx[
        (idx["participant_id"].astype(str) == str(participant_id))
        & (idx["condition"].isin(transition_conds))
    ].copy()

    if sub_idx.empty:
        raise ValueError(f"没有找到 participant_id={participant_id} 的 transition condition 数据。")

    sub_idx["condition"] = pd.Categorical(
        sub_idx["condition"],
        categories=transition_conds,
        ordered=True,
    )
    sub_idx = sub_idx.sort_values("condition")

    n = len(sub_idx)
    fig, axes = plt.subplots(
        n,
        1,
        figsize=(15, max(3.2 * n, 8)),
        sharex=True,
    )

    if n == 1:
        axes = [axes]

    debug_rows = []

    global_max_s = 0.0

    for ax, row in zip(axes, sub_idx.itertuples(index=False)):
        cond = str(row.condition)

        try:
            df_log = read_trial_log(row.log_path)
            t0 = get_schedule_start_t0(df_log)
            perf = extract_track_performance(df_log, t0)

            max_elapsed_s = float(perf["elapsed_ms"].max()) / 1000.0
            global_max_s = max(global_max_s, max_elapsed_s)

            full = perf.sort_values("elapsed_ms").copy()

            t_sec = full["elapsed_ms"].to_numpy(dtype=float) / 1000.0
            dev = full["center_deviation"].to_numpy(dtype=float)

            ax.plot(t_sec, dev, linewidth=0.9, label="center_deviation")
            ax.axhline(0, linestyle=":", linewidth=1)

            transition_events, _ = build_transition_events_split(SCHEDULES[cond])
            transition_times_ms = transition_events["t_transition_ms"].to_numpy(dtype=float) + schedule_offset_ms

            # 所有 transition 时间点
            for k, t_ms in enumerate(transition_times_ms):
                x_s = t_ms / 1000.0
                ax.axvline(
                    x_s,
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.6,
                )

                ymin, ymax = ax.get_ylim()
                ax.text(
                    x_s,
                    ymax,
                    f"T{k + 1}\n{x_s:.1f}s",
                    rotation=90,
                    va="top",
                    ha="right",
                    fontsize=8,
                    alpha=0.85,
                )

            response = run07.compute_next_disturbance_response(
                perf=perf,
                condition=cond,
                pre_window_ms=pre_window_ms,
                post_window_ms=post_window_ms,
                schedule_offset_ms=schedule_offset_ms,
            )

            target_n = response.get("target_transition_number", np.nan)
            target_x_ms = response.get("analysis_target_time_ms", np.nan)

            if not np.isnan(target_x_ms):
                target_x_s = target_x_ms / 1000.0

                # 目标扰动：加粗虚线
                ax.axvline(
                    target_x_s,
                    linestyle="--",
                    linewidth=2.5,
                    label=f"target T{int(target_n)} = {target_x_s:.1f}s",
                )

                # pre/post window
                ax.axvspan(
                    (target_x_ms - pre_window_ms) / 1000.0,
                    target_x_s,
                    alpha=0.15,
                    label="pre window",
                )

                ax.axvspan(
                    target_x_s,
                    (target_x_ms + post_window_ms) / 1000.0,
                    alpha=0.15,
                    label="post window",
                )

                # 在图中间显著标出目标扰动横坐标
                ymin, ymax = ax.get_ylim()
                y_label = ymin + 0.12 * (ymax - ymin)

                ax.annotate(
                    f"T{int(target_n)} = {target_x_s:.1f}s",
                    xy=(target_x_s, y_label),
                    xytext=(target_x_s + 3, y_label),
                    arrowprops=dict(arrowstyle="->", linewidth=1.2),
                    fontsize=10,
                    bbox=dict(boxstyle="round", alpha=0.18),
                )

                debug_rows.append(
                    {
                        "participant_id": participant_id,
                        "condition": cond,
                        "max_elapsed_s": max_elapsed_s,
                        "target_transition_number": int(target_n),
                        "target_x_s": target_x_s,
                        "pre_window_start_s": (target_x_ms - pre_window_ms) / 1000.0,
                        "pre_window_end_s": target_x_s,
                        "post_window_start_s": target_x_s,
                        "post_window_end_s": (target_x_ms + post_window_ms) / 1000.0,
                        "pre_rmse": response.get("pre_rmse", np.nan),
                        "post_rmse": response.get("post_rmse", np.nan),
                        "delta_rmse": response.get("delta_rmse", np.nan),
                        "peak_deviation": response.get("peak_deviation", np.nan),
                        "post_auc": response.get("post_auc", np.nan),
                    }
                )

                text = (
                    f"{cond}\n"
                    f"max elapsed = {max_elapsed_s:.1f}s\n"
                    f"target: T{int(target_n)} = {target_x_s:.1f}s\n"
                    f"pre window: {(target_x_ms - pre_window_ms) / 1000:.1f}–{target_x_s:.1f}s\n"
                    f"post window: {target_x_s:.1f}–{(target_x_ms + post_window_ms) / 1000:.1f}s\n"
                    f"pre RMSE = {response.get('pre_rmse', np.nan):.3f}\n"
                    f"post RMSE = {response.get('post_rmse', np.nan):.3f}\n"
                    f"delta RMSE = {response.get('delta_rmse', np.nan):.3f}"
                )

                ax.text(
                    0.01,
                    0.95,
                    text,
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=9,
                    bbox=dict(boxstyle="round", alpha=0.15),
                )

            ax.set_ylabel(cond)
            ax.grid(True, alpha=0.25)

        except Exception as exc:
            ax.text(
                0.5,
                0.5,
                f"{cond} error:\n{exc}",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_ylabel(cond)

    axes[-1].set_xlabel("Preprocessed elapsed time / schedule time (s)")

    if global_max_s > 0:
        for ax in axes:
            ax.set_xlim(0, global_max_s)

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)

    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=4,
            frameon=False,
        )

    fig.suptitle(
        f"Full trajectory with target disturbance x-coordinate | participant: {participant_id}",
        y=0.995,
        fontsize=14,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.965])

    out_png = os.path.join(
        save_dir,
        f"{participant_id}_six_conditions_target_x_fixed.png",
    )
    plt.savefig(out_png, dpi=300)
    plt.close()

    debug_df = pd.DataFrame(debug_rows)
    out_csv = os.path.join(
        save_dir,
        f"{participant_id}_six_conditions_target_x_fixed.csv",
    )
    debug_df.to_csv(out_csv, index=False)

    print(f"[完成] 图已保存: {out_png}")
    print(f"[完成] 时间表已保存: {out_csv}")

    if not debug_df.empty:
        pd.set_option("display.width", 260)
        pd.set_option("display.max_columns", 40)
        pd.set_option("display.float_format", lambda x: f"{x:.3f}")
        print("\n==== 目标扰动横坐标检查表 ====")
        print(debug_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--participant_id",
        type=str,
        default="2_CYL",
        help="要检查的被试 ID，例如 2_CYL",
    )
    parser.add_argument(
        "--index_csv",
        type=str,
        default="data/expdata/index.csv",
        help="index.csv 路径",
    )
    parser.add_argument(
        "--run07_path",
        type=str,
        default="run_07_next_disturbance_resilience_fixed.py",
        help="修正后的 run_07 脚本路径",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="next_disturbance_debug_fixed",
        help="输出图和表的文件夹",
    )

    args = parser.parse_args()

    plot_one_subject_six_conditions(
        participant_id=args.participant_id,
        index_csv=args.index_csv,
        run07_path=args.run07_path,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
