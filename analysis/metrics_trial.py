import numpy as np
import pandas as pd
#trial-level 指标（RQ1）
#对每个 trial（整段，不按事件）：
def auc_trapz(y: np.ndarray, t_sec: np.ndarray) -> float:

    # 用真实时间间隔做面积（更稳）
    if len(y) < 2:
        return float("nan")
    return float(np.trapezoid(y, t_sec))

def compute_trial_metrics(perf_wide: pd.DataFrame) -> dict:
    """
    perf_wide: output of extract_track_performance
    """
    dev = perf_wide["center_deviation"].to_numpy(dtype=float)
    t = perf_wide["t_sec"].to_numpy(dtype=float)

    out = {}
    out["n_samples"] = int(len(perf_wide))
    out["mean_dev"] = float(np.nanmean(dev))
    out["std_dev"] = float(np.nanstd(dev))
    out["rmse_dev"] = float(np.sqrt(np.nanmean(dev ** 2)))
    out["auc_dev"] = auc_trapz(np.nan_to_num(dev, nan=0.0), t)

    if "cursor_in_target" in perf_wide.columns:
        hit = perf_wide["cursor_in_target"].to_numpy(dtype=float)
        out["hit_ratio"] = float(np.nanmean(hit))
    return out