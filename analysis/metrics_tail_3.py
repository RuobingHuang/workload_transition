import numpy as np
import pandas as pd

from .metrics_event import _window_slice_abs, _first_sustained_time
from .config import WIN

# TAIL 段标称时长（与所有 schedule 末尾的 45000 ms 对应）
TAIL_DURATION_MS = 45_000


def compute_tail_metrics(
    perf: pd.DataFrame,
    tail_event_row: pd.Series,
    reg_metrics_df: pd.DataFrame,
    tail_duration_ms: int = TAIL_DURATION_MS,
) -> dict:
    """
    计算 TAIL 段（试次最后约 45 s 低负荷段）的表现指标。

    Parameters
    ----------
    perf : DataFrame
        extract_track_performance 输出的完整试次性能表。
    tail_event_row : Series
        进入 TAIL 的转换事件行（来自 build_transition_events_split 的 tail_events.iloc[0]）。
    reg_metrics_df : DataFrame
        本次试次中所有常规转换事件的已计算指标（来自 compute_event_metrics）。
        用于派生遗留链接指标（最后事件恢复状态、累积扰动量等）。
    tail_duration_ms : int
        TAIL 窗口时长，默认 45 000 ms。

    返回指标
    --------
    绝对表现（抵抗力后效）:
        tail_mean_dev, tail_rmse_dev, tail_std_dev, tail_auc_dev, tail_hit_ratio

    段内趋势（恢复力后效）:
        tail_early_mean_dev  : TAIL 前 15 s 均值
        tail_late_mean_dev   : TAIL 后 15 s 均值
        tail_delta_early_late: late - early（负值 = TAIL 内仍改善，正值 = 疲劳加剧）
        tail_slope_dev       : OLS 斜率（deviation / second）
        tail_time_to_stable_ms: 进入 TAIL 后首次持续稳定（≤ late-TAIL 均值 + 1 SD）的时间

    遗留链接（event-level 历史）:
        last_event_recover_success : 最后一个常规 event 是否恢复成功（0/1）
        last_event_rec_mean_dev    : 最后一个常规 event 的 rec_mean_dev
        mean_recover_success_ratio : 本试次所有常规 event 的 recover_success 均值
        cumulative_delta_dev       : 本试次所有常规 event 的 delta_dev 均值（累积扰动负荷）
    """
    t_start = int(tail_event_row["t_transition_ms"])
    t_end = t_start + tail_duration_ms

    tail = _window_slice_abs(perf, t_start, t_end)

    out: dict = {}
    out["tail_t_start_ms"] = t_start
    out["tail_n_samples"] = len(tail)

    _nan = float("nan")

    if len(tail) == 0:
        for k in [
            "tail_mean_dev", "tail_rmse_dev", "tail_std_dev", "tail_auc_dev",
            "tail_hit_ratio",
            "tail_early_mean_dev", "tail_late_mean_dev", "tail_delta_early_late",
            "tail_slope_dev", "tail_time_to_stable_ms",
            "last_event_recover_success", "last_event_rec_mean_dev",
            "mean_recover_success_ratio", "cumulative_delta_dev",
        ]:
            out[k] = _nan
        return out

    dev = tail["center_deviation"].to_numpy(dtype=float)
    t_ms = tail["elapsed_ms"].to_numpy(dtype=float)
    t_sec_rel = (t_ms - t_start) / 1000.0  # 相对于 TAIL 起点的秒数

    # --- 绝对表现 ---
    out["tail_mean_dev"] = float(np.nanmean(dev))
    out["tail_rmse_dev"] = float(np.sqrt(np.nanmean(dev ** 2)))
    out["tail_std_dev"] = float(np.nanstd(dev))
    out["tail_auc_dev"] = float(np.trapezoid(np.nan_to_num(dev, nan=0.0), t_sec_rel))

    if "cursor_in_target" in tail.columns:
        hit = tail["cursor_in_target"].to_numpy(dtype=float)
        out["tail_hit_ratio"] = float(np.nanmean(hit))
    else:
        out["tail_hit_ratio"] = _nan

    # --- 前段（0-15 s）vs 后段（末尾 15 s）子窗口 ---
    early = _window_slice_abs(perf, t_start, t_start + 15_000)
    middle = _window_slice_abs(perf, t_start + 15_000,t_end - 15_000)
    late_start = max(t_start + 15_000, t_end - 15_000)
    late = _window_slice_abs(perf, late_start, t_end)

    out["tail_early_mean_dev"] = (
        float(np.nanmean(early["center_deviation"].to_numpy(dtype=float)))
        if len(early) else _nan
    )
    out["tail_mid_mean_dev"] = (
        float(np.nanmean(middle["center_deviation"].to_numpy(dtype=float)))
        if len(middle) else _nan
    )
    out["tail_late_mean_dev"] = (
        float(np.nanmean(late["center_deviation"].to_numpy(dtype=float)))
        if len(late) else _nan
    )
    early_v = out["tail_early_mean_dev"]
    late_v = out["tail_late_mean_dev"]
    mid_v = out["tail_mid_mean_dev"]
    out["tail_delta_mid_late"] = (
        late_v - mid_v
        if not (np.isnan(late_v) or np.isnan(mid_v))
        else _nan
    )
    out["tail_delta_early_late"] = (
        late_v - early_v
        if not (np.isnan(late_v) or np.isnan(early_v))
        else _nan
    )

    # --- TAIL 内 OLS 斜率（deviation 单位 / 秒） ---
    valid = ~np.isnan(dev)
    if valid.sum() >= 2:
        coeffs = np.polyfit(t_sec_rel[valid], dev[valid], 1)
        out["tail_slope_dev"] = float(coeffs[0])
    else:
        out["tail_slope_dev"] = _nan

    # --- 首次稳定时间 ---
    # 参考态 = TAIL 后段（末 15 s）均值 + 1 SD；
    # 找从 TAIL 起点开始，表现首次持续 sustain_ms 不超过该阈值的时刻
    if len(late) >= 2:
        late_dev = late["center_deviation"].to_numpy(dtype=float)
        thr = float(np.nanmean(late_dev)) + float(np.nanstd(late_dev))
    else:
        thr = out["tail_mean_dev"] + out["tail_std_dev"]

    post = tail.sort_values("elapsed_ms").copy()
    ok = post["center_deviation"].to_numpy(dtype=float) <= thr
    out["tail_time_to_stable_ms"] = _first_sustained_time(post, ok, WIN.sustain_ms)

    # --- 遗留链接（来自常规转换 event 的历史指标）---
    if reg_metrics_df is not None and len(reg_metrics_df) > 0:
        last = reg_metrics_df.iloc[-1]
        out["last_event_recover_success"] = (
            float(last["recover_success"]) if "recover_success" in last.index else _nan
        )
        out["last_event_rec_mean_dev"] = (
            float(last["rec_mean_dev"]) if "rec_mean_dev" in last.index else _nan
        )
        out["mean_recover_success_ratio"] = (
            float(reg_metrics_df["recover_success"].mean())
            if "recover_success" in reg_metrics_df.columns else _nan
        )
        out["cumulative_delta_dev"] = (
            float(reg_metrics_df["delta_dev"].mean())
            if "delta_dev" in reg_metrics_df.columns else _nan
        )
    else:
        out["last_event_recover_success"] = _nan
        out["last_event_rec_mean_dev"] = _nan
        out["mean_recover_success_ratio"] = _nan
        out["cumulative_delta_dev"] = _nan

    return out
