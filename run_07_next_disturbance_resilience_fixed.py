"""
run_07_next_disturbance_resilience_fixed.py
────────────────────────────────────────────────────────────────
Frequency-conditioned next-disturbance response analysis

重要修正
--------
根据 debug 结果，extract_track_performance() 输出的 elapsed_ms 最大值约为 290s，
说明预处理后的时间轴已经去掉了最开始 30s 自动化阶段。

因此：
  preprocessed elapsed time 0s = schedule start

所以本脚本不再对 transition time 额外 +30s。

研究目标
--------
分析在不同 transition frequency history 之后，下一次相同强度扰动引发的偏移量。

核心逻辑
--------
在相同 intensity 下比较：
  LI_LF → 第 2 次扰动后的响应
  LI_MF → 第 3 次扰动后的响应
  LI_HF → 第 4 次扰动后的响应

  HI_LF → 第 2 次扰动后的响应
  HI_MF → 第 3 次扰动后的响应
  HI_HF → 第 4 次扰动后的响应

解释：
  - pre_rmse      : 扰动前窗口 RMSE
  - post_rmse     : 扰动后窗口 RMSE
  - delta_rmse    : post_rmse - pre_rmse，越大表示扰动造成的偏移越大
  - peak_deviation: 扰动后窗口内最大绝对偏移
  - post_auc      : 扰动后窗口内绝对偏移积分，越大表示扰动后的总体偏移负担越高

读取
----
  data/expdata/index.csv
  需要包含 participant_id / condition / log_path 三列

输出
----
  next_disturbance_response_fixed.csv
  next_disturbance_rmanova_fixed.csv
  next_disturbance_pairwise_fixed.csv
  next_disturbance_summary_fixed.csv
  next_disturbance_figures_fixed/
"""

import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ttest_rel
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

# ── 使 analysis 包可被直接运行 ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from analysis.config import CONDITION_META, SCHEDULES
from analysis.io_log import read_trial_log
from analysis.metrics_event import _window_slice_abs
from analysis.preprocess import extract_track_performance, get_schedule_start_t0
from analysis.schedule import build_transition_events_split


# ─────────────────────────────────────────────────────────────────────
# 参数
# ─────────────────────────────────────────────────────────────────────
TRANSITION_CONDS = ["LI_LF", "HI_LF", "LI_MF", "HI_MF", "LI_HF", "HI_HF"]

# 注意：这里必须为 0，因为 preprocessing 已经去掉了最开始 30s 自动化阶段
SCHEDULE_OFFSET_MS = 0

TARGET_TRANSITION_INDEX = {
    "LI_LF": 1,  # 第二次扰动，0-based index
    "LI_MF": 2,  # 第三次扰动
    "LI_HF": 3,  # 第四次扰动

    "HI_LF": 1,
    "HI_MF": 2,
    "HI_HF": 3,
}

PRE_WINDOW_MS = 3_000
POST_WINDOW_MS = 5_000

RESPONSE_METRICS = [
    "pre_rmse",
    "post_rmse",
    "delta_rmse",
    "peak_deviation",
    "post_auc",
]


# ─────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────
def _rmse(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return float("nan")
    return float(np.sqrt(np.mean(x ** 2)))


def _auc_abs(t_ms: np.ndarray, dev: np.ndarray) -> float:
    """
    计算绝对偏移量 AUC。
    单位约为 deviation × second。
    """
    t_ms = np.asarray(t_ms, dtype=float)
    dev = np.asarray(dev, dtype=float)

    ok = (~np.isnan(t_ms)) & (~np.isnan(dev))
    t_ms = t_ms[ok]
    dev = dev[ok]

    if len(t_ms) < 3:
        return float("nan")

    t_sec = (t_ms - t_ms[0]) / 1000.0
    return float(np.trapz(np.abs(dev), t_sec))


def _add_meta(df: pd.DataFrame) -> pd.DataFrame:
    """
    确保结果表中包含 freq 和 intensity 两列。
    """
    df = df.copy()

    if "freq" not in df.columns or "intensity" not in df.columns:

        def _parse_condition(c: str):
            inten, freq = str(c).split("_")
            return pd.Series({"intensity": inten, "freq": freq})

        parsed = df["condition"].apply(_parse_condition)

        for col in ["intensity", "freq"]:
            if col not in df.columns:
                df[col] = parsed[col]

    return df


# ─────────────────────────────────────────────────────────────────────
# 核心：下一次扰动响应
# ─────────────────────────────────────────────────────────────────────
def compute_next_disturbance_response(
    perf: pd.DataFrame,
    condition: str,
    pre_window_ms: int = PRE_WINDOW_MS,
    post_window_ms: int = POST_WINDOW_MS,
    schedule_offset_ms: int = SCHEDULE_OFFSET_MS,
) -> dict:
    """
    对单个 participant × condition 计算目标扰动前后的偏移响应。

    注意：
    ----
    schedule_offset_ms 默认为 0，因为预处理后的 elapsed_ms 已经以 schedule start 为 0。

    Returns
    -------
    dict
        target_transition_index
        schedule_target_time_ms
        analysis_target_time_ms
        pre_rmse
        post_rmse
        delta_rmse
        peak_deviation
        post_auc
        n_pre_samples
        n_post_samples
    """
    result = {
        "target_transition_index": np.nan,
        "target_transition_number": np.nan,
        "schedule_target_time_ms": np.nan,
        "analysis_target_time_ms": np.nan,
        "schedule_offset_ms": schedule_offset_ms,
        "pre_window_ms": pre_window_ms,
        "post_window_ms": post_window_ms,
        "pre_rmse": np.nan,
        "post_rmse": np.nan,
        "delta_rmse": np.nan,
        "peak_deviation": np.nan,
        "post_auc": np.nan,
        "n_pre_samples": 0,
        "n_post_samples": 0,
    }

    if condition not in TARGET_TRANSITION_INDEX:
        return result

    # transition_events 不包含 tail
    transition_events, _ = build_transition_events_split(SCHEDULES[condition])
    target_idx = TARGET_TRANSITION_INDEX[condition]

    if target_idx >= len(transition_events):
        print(f"[跳过] {condition}: target_idx={target_idx} 超过 transition 数量={len(transition_events)}")
        return result

    schedule_t_event = int(transition_events.iloc[target_idx]["t_transition_ms"])
    analysis_t_event = schedule_t_event + int(schedule_offset_ms)

    pre_df = _window_slice_abs(
        perf,
        analysis_t_event - pre_window_ms,
        analysis_t_event,
    )

    post_df = _window_slice_abs(
        perf,
        analysis_t_event,
        analysis_t_event + post_window_ms,
    )

    result["target_transition_index"] = int(target_idx)
    result["target_transition_number"] = int(target_idx + 1)
    result["schedule_target_time_ms"] = int(schedule_t_event)
    result["analysis_target_time_ms"] = int(analysis_t_event)
    result["n_pre_samples"] = int(len(pre_df))
    result["n_post_samples"] = int(len(post_df))

    if len(pre_df) < 5 or len(post_df) < 5:
        return result

    dev_pre = pre_df["center_deviation"].to_numpy(dtype=float)
    dev_post = post_df["center_deviation"].to_numpy(dtype=float)

    pre_rmse = _rmse(dev_pre)
    post_rmse = _rmse(dev_post)

    result["pre_rmse"] = pre_rmse
    result["post_rmse"] = post_rmse
    result["delta_rmse"] = (
        post_rmse - pre_rmse
        if not np.isnan(pre_rmse) and not np.isnan(post_rmse)
        else np.nan
    )
    result["peak_deviation"] = (
        float(np.nanmax(np.abs(dev_post)))
        if np.any(~np.isnan(dev_post))
        else np.nan
    )
    result["post_auc"] = _auc_abs(
        post_df["elapsed_ms"].to_numpy(dtype=float),
        dev_post,
    )

    return result


# ─────────────────────────────────────────────────────────────────────
# 可视化：单个 participant × condition 的局部扰动窗口
# ─────────────────────────────────────────────────────────────────────
def visualize_single_next_disturbance(
    perf: pd.DataFrame,
    participant_id: str,
    condition: str,
    response: dict,
    save_dir: str = "next_disturbance_figures_fixed",
    pre_window_ms: int = PRE_WINDOW_MS,
    post_window_ms: int = POST_WINDOW_MS,
) -> None:
    """
    可视化单个 participant × condition 的目标扰动响应局部窗口。
    """
    os.makedirs(save_dir, exist_ok=True)

    t_event = response.get("analysis_target_time_ms", np.nan)
    if np.isnan(t_event):
        return

    t_event = int(t_event)

    vis_start = t_event - pre_window_ms
    vis_end = t_event + post_window_ms

    win = (
        _window_slice_abs(perf, vis_start, vis_end)
        .sort_values("elapsed_ms")
        .copy()
    )

    if len(win) < 10:
        return

    t_sec_relative = (win["elapsed_ms"].to_numpy(dtype=float) - t_event) / 1000.0
    dev = win["center_deviation"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4.8))

    ax.plot(t_sec_relative, dev, linewidth=1.3, label="center_deviation")
    ax.axvline(0, linestyle="--", linewidth=1.5, label="target disturbance")

    ax.axvspan(
        -pre_window_ms / 1000.0,
        0,
        alpha=0.15,
        label=f"pre window ({pre_window_ms / 1000:.0f}s)",
    )
    ax.axvspan(
        0,
        post_window_ms / 1000.0,
        alpha=0.15,
        label=f"post window ({post_window_ms / 1000:.0f}s)",
    )

    ax.axhline(0, linestyle=":", linewidth=1)

    title = (
        f"Next Disturbance Response | {participant_id} | {condition} | "
        f"T{int(response['target_transition_number'])} = {response['analysis_target_time_ms'] / 1000:.1f}s"
    )
    ax.set_title(title)
    ax.set_xlabel("Time relative to target disturbance (s)")
    ax.set_ylabel("Center deviation")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    text = (
        f"T{int(response['target_transition_number'])} x = {response['analysis_target_time_ms'] / 1000:.1f}s\n"
        f"pre RMSE = {response['pre_rmse']:.3f}\n"
        f"post RMSE = {response['post_rmse']:.3f}\n"
        f"delta RMSE = {response['delta_rmse']:.3f}\n"
        f"peak deviation = {response['peak_deviation']:.3f}\n"
        f"post AUC = {response['post_auc']:.3f}"
    )

    ax.text(
        0.02,
        0.95,
        text,
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    plt.tight_layout()

    out_path = os.path.join(
        save_dir,
        f"{participant_id}_{condition}_next_disturbance_fixed.png",
    )
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[可视化完成] {out_path}")


# ─────────────────────────────────────────────────────────────────────
# 可视化：summary line plot
# ─────────────────────────────────────────────────────────────────────
def plot_summary_by_intensity(
    df: pd.DataFrame,
    metric: str = "delta_rmse",
    save_dir: str = "next_disturbance_figures_fixed",
) -> None:
    """
    按 intensity 分组，画 LF/MF/HF 的均值 ± SEM。
    """
    os.makedirs(save_dir, exist_ok=True)

    d = df.dropna(subset=[metric]).copy()
    if d.empty:
        return

    d = _add_meta(d)
    d["freq"] = pd.Categorical(d["freq"], ["LF", "MF", "HF"], ordered=True)
    d["intensity"] = pd.Categorical(d["intensity"], ["LI", "HI"], ordered=True)

    summary = (
        d.groupby(["intensity", "freq"], observed=True)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    x_labels = ["LF", "MF", "HF"]
    x = np.arange(len(x_labels))

    for intensity in ["LI", "HI"]:
        sub = summary[summary["intensity"] == intensity].set_index("freq").reindex(x_labels)
        ax.errorbar(
            x,
            sub["mean"].to_numpy(dtype=float),
            yerr=sub["sem"].to_numpy(dtype=float),
            marker="o",
            capsize=4,
            linewidth=1.8,
            label=intensity,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Transition frequency history")
    ax.set_ylabel(metric)
    ax.set_title(f"Frequency-conditioned next-disturbance response: {metric}")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Intensity")

    plt.tight_layout()

    out_path = os.path.join(save_dir, f"summary_{metric}_fixed.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[summary图完成] {out_path}")


# ─────────────────────────────────────────────────────────────────────
# 主分析流程
# ─────────────────────────────────────────────────────────────────────
def run_next_disturbance_analysis(index_csv: str) -> pd.DataFrame:
    """
    读取 index.csv，对所有 participant × condition 计算下一次扰动响应。
    """
    idx = pd.read_csv(index_csv)

    required = {"participant_id", "condition", "log_path"}
    missing = required - set(idx.columns)
    if missing:
        raise ValueError(f"index_csv 缺少列: {sorted(missing)}")

    rows = []

    print("正在计算不同 frequency history 后的下一次扰动响应（fixed: no +30s）…")

    for row in idx.itertuples(index=False):
        pid = str(row.participant_id)
        cond = str(row.condition)

        if cond not in TARGET_TRANSITION_INDEX:
            continue

        meta = CONDITION_META.get(cond, {})

        try:
            df_log = read_trial_log(row.log_path)
            t0 = get_schedule_start_t0(df_log)
            perf = extract_track_performance(df_log, t0)

            resp = compute_next_disturbance_response(
                perf=perf,
                condition=cond,
                pre_window_ms=PRE_WINDOW_MS,
                post_window_ms=POST_WINDOW_MS,
                schedule_offset_ms=SCHEDULE_OFFSET_MS,
            )

            rows.append(
                {
                    "participant_id": pid,
                    "condition": cond,
                    **meta,
                    **resp,
                }
            )

            print(
                f"  {pid} {cond}: "
                f"target=T{int(resp['target_transition_number']) if not np.isnan(resp['target_transition_number']) else 'NA'}  "
                f"x={resp['analysis_target_time_ms'] / 1000 if not np.isnan(resp['analysis_target_time_ms']) else np.nan:.1f}s  "
                f"pre={resp['pre_rmse']:.3f}  "
                f"post={resp['post_rmse']:.3f}  "
                f"delta={resp['delta_rmse']:.3f}  "
                f"peak={resp['peak_deviation']:.3f}"
            )

            # 默认每个 condition 画第一个成功被试，方便 debug
            existing_figs = (
                [
                    f for f in os.listdir("next_disturbance_figures_fixed")
                    if f.endswith("_next_disturbance_fixed.png")
                ]
                if os.path.exists("next_disturbance_figures_fixed")
                else []
            )

            already_has_cond = any(
                f"_{cond}_next_disturbance_fixed.png" in f
                for f in existing_figs
            )

            if not already_has_cond:
                visualize_single_next_disturbance(
                    perf=perf,
                    participant_id=pid,
                    condition=cond,
                    response=resp,
                )

        except Exception as exc:
            print(f"  [错误] {pid} {cond}: {exc}")

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# 统计分析
# ─────────────────────────────────────────────────────────────────────
def run_rmanova(df: pd.DataFrame) -> pd.DataFrame:
    """
    2(intensity) × 3(freq) RM-ANOVA。
    """
    d0 = df[df["condition"].isin(TRANSITION_CONDS)].copy()
    d0 = _add_meta(d0)
    d0["freq"] = pd.Categorical(d0["freq"], ["LF", "MF", "HF"], ordered=True)
    d0["intensity"] = pd.Categorical(d0["intensity"], ["LI", "HI"], ordered=True)

    summary_rows = []

    for metric in RESPONSE_METRICS:
        if metric not in d0.columns:
            continue

        d = d0[["participant_id", "freq", "intensity", metric]].dropna().copy()

        cell_counts = d.groupby(
            ["participant_id", "freq", "intensity"],
            observed=True,
        )[metric].count()

        complete_subs = (
            cell_counts[cell_counts == 1]
            .reset_index()["participant_id"]
            .unique()
        )

        d = d[d["participant_id"].isin(complete_subs)]

        if d["participant_id"].nunique() < 3:
            print(f"[跳过 RM-ANOVA] {metric}: 完整被试数 < 3")
            continue

        try:
            aov = AnovaRM(
                d,
                depvar=metric,
                subject="participant_id",
                within=["intensity", "freq"],
            ).fit()

            print(f"\n==== RM-ANOVA: {metric} ====")
            print(aov)

            for effect, row in aov.anova_table.iterrows():
                summary_rows.append(
                    {
                        "metric": metric,
                        "effect": effect,
                        "F": row.get("F Value", np.nan),
                        "num_df": row.get("Num DF", np.nan),
                        "den_df": row.get("Den DF", np.nan),
                        "p": row.get("Pr > F", np.nan),
                    }
                )

        except Exception as exc:
            print(f"[错误] RM-ANOVA on {metric}: {exc}")

    return pd.DataFrame(summary_rows)


def run_within_intensity_pairwise(df: pd.DataFrame) -> pd.DataFrame:
    """
    在相同 intensity 下比较 LF/MF/HF。
    """
    d0 = _add_meta(df)
    all_rows = []

    for metric in RESPONSE_METRICS:
        if metric not in d0.columns:
            continue

        metric_rows = []

        for intensity in ["LI", "HI"]:
            sub = d0[d0["intensity"] == intensity].copy()

            conds = [f"{intensity}_LF", f"{intensity}_MF", f"{intensity}_HF"]
            data_by_cond = {
                cond: sub[sub["condition"] == cond].set_index("participant_id")
                for cond in conds
            }

            for cond_a, cond_b in combinations(conds, 2):
                da = data_by_cond[cond_a]
                db = data_by_cond[cond_b]

                common = da.index.intersection(db.index)

                xa = da.loc[common, metric].astype(float)
                xb = db.loc[common, metric].astype(float)
                ok = xa.notna() & xb.notna()
                n = int(ok.sum())

                base = {
                    "metric": metric,
                    "intensity": intensity,
                    "contrast": f"{cond_b} - {cond_a}",
                    "cond_a": cond_a,
                    "cond_b": cond_b,
                    "n": n,
                    "mean_a": float(xa[ok].mean()) if ok.any() else np.nan,
                    "mean_b": float(xb[ok].mean()) if ok.any() else np.nan,
                }

                if n < 3:
                    metric_rows.append(
                        {
                            **base,
                            "mean_diff(b-a)": np.nan,
                            "cohens_dz": np.nan,
                            "t": np.nan,
                            "p_raw": np.nan,
                            "p_adj_holm": np.nan,
                            "reject@0.05": np.nan,
                            "note": "n<3, skipped",
                        }
                    )
                    continue

                diff = xb[ok] - xa[ok]
                mean_diff = float(diff.mean())
                sd = float(diff.std(ddof=1))
                dz = float(mean_diff / sd) if sd > 0 else np.nan

                t_val, p_val = ttest_rel(xb[ok], xa[ok])

                metric_rows.append(
                    {
                        **base,
                        "mean_diff(b-a)": mean_diff,
                        "cohens_dz": dz,
                        "t": float(t_val),
                        "p_raw": float(p_val),
                        "p_adj_holm": np.nan,
                        "reject@0.05": np.nan,
                        "note": "",
                    }
                )

        if metric_rows:
            tmp = pd.DataFrame(metric_rows)

            # 每个 metric × intensity 内做 Holm 校正
            for intensity in ["LI", "HI"]:
                mask = (
                    (tmp["metric"] == metric)
                    & (tmp["intensity"] == intensity)
                    & tmp["p_raw"].notna()
                )
                if mask.sum() > 0:
                    rej, p_adj, _, _ = multipletests(
                        tmp.loc[mask, "p_raw"].values,
                        method="holm",
                    )
                    tmp.loc[mask, "p_adj_holm"] = p_adj
                    tmp.loc[mask, "reject@0.05"] = rej

            all_rows.append(tmp)

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def summarize_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    """
    输出每个 condition 的描述性统计。
    """
    d = _add_meta(df)

    summary = (
        d.groupby(["intensity", "freq", "condition"], observed=True)[RESPONSE_METRICS]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    return summary


# ─────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────
def main(index_csv: str = "data/expdata/index.csv") -> None:
    os.makedirs("next_disturbance_figures_fixed", exist_ok=True)

    resp_df = run_next_disturbance_analysis(index_csv)

    if resp_df.empty:
        print("\n[警告] 没有得到任何结果，请检查 index.csv 和 log_path。")
        return

    resp_df = _add_meta(resp_df)

    out_resp = "next_disturbance_response_fixed.csv"
    resp_df.to_csv(out_resp, index=False)
    print(f"\nWrote: {out_resp} ({len(resp_df)} 行)")

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 45)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    show_cols = [
        "participant_id",
        "condition",
        "intensity",
        "freq",
        "target_transition_number",
        "schedule_target_time_ms",
        "analysis_target_time_ms",
        "pre_rmse",
        "post_rmse",
        "delta_rmse",
        "peak_deviation",
        "post_auc",
    ]

    print("\n==== 下一次扰动响应结果摘要 ====")
    print(resp_df[[c for c in show_cols if c in resp_df.columns]].to_string(index=False))

    summary_df = summarize_by_condition(resp_df)
    out_summary = "next_disturbance_summary_fixed.csv"
    summary_df.to_csv(out_summary, index=False)
    print(f"\nWrote: {out_summary}")

    anova_df = run_rmanova(resp_df)
    if not anova_df.empty:
        out_anova = "next_disturbance_rmanova_fixed.csv"
        anova_df.to_csv(out_anova, index=False)
        print(f"\nWrote: {out_anova}")

    pair_df = run_within_intensity_pairwise(resp_df)
    if not pair_df.empty:
        out_pair = "next_disturbance_pairwise_fixed.csv"
        pair_df.to_csv(out_pair, index=False)
        print(f"\nWrote: {out_pair}")

        print("\n==== 相同 intensity 下 LF/MF/HF 两两比较 ====")
        print(pair_df.to_string(index=False))

        trend = pair_df[pair_df["p_raw"] < 0.10]
        if not trend.empty:
            print("\n==== p_raw < 0.10 的趋势性差异 ====")
            print(trend.to_string(index=False))
        else:
            print("\n（无 p_raw < 0.10 的趋势性结果）")

    for metric in ["delta_rmse", "post_rmse", "peak_deviation", "post_auc"]:
        plot_summary_by_intensity(resp_df, metric=metric)

    print("\n分析完成。建议优先看：")
    print("  1. next_disturbance_response_fixed.csv")
    print("  2. next_disturbance_summary_fixed.csv")
    print("  3. next_disturbance_figures_fixed/summary_delta_rmse_fixed.png")
    print("  4. next_disturbance_pairwise_fixed.csv")


if __name__ == "__main__":
    main()
