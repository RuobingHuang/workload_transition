import numpy as np
import pandas as pd
from .config import WIN
#event-level 指标（RQ2）

def _window_slice(perf: pd.DataFrame, t0_ms: int, start_ms: int, end_ms: int) -> pd.DataFrame:
    lo = t0_ms + start_ms
    hi = t0_ms + end_ms
    return perf[(perf["elapsed_ms"] >= lo) & (perf["elapsed_ms"] <= hi)].copy()

def _estimate_dt_ms(perf: pd.DataFrame) -> float:
    # 用中位数估计采样间隔（理论上约20ms）
    d = perf["elapsed_ms"].diff().dropna()
    if d.empty:
        return 20.0
    return float(d.median())

def _first_sustained_time(perf_post: pd.DataFrame, cond: np.ndarray, sustain_ms: int) -> float:
    """
    perf_post: must be sorted by elapsed_ms, starting at transition time.
    cond: boolean array, same length.
    return: time since transition (ms) when condition first holds for sustain_ms continuously.
            nan if not found.
    """
    if len(perf_post) == 0:
        return float("nan")

    dt = _estimate_dt_ms(perf_post)
    need = max(1, int(np.ceil(sustain_ms / dt)))

    # run-length scan
    run = 0
    start_idx = None
    for i, ok in enumerate(cond):
        if ok:
            if run == 0:
                start_idx = i
            run += 1
            if run >= need:
                t_ms = perf_post["elapsed_ms"].iloc[start_idx] - perf_post["elapsed_ms"].iloc[0]
                return float(t_ms)
        else:
            run = 0
            start_idx = None
    return float("nan")


def _window_slice_abs(perf: pd.DataFrame, start_ms: int, end_ms: int) -> pd.DataFrame:
    return perf[(perf["elapsed_ms"] >= start_ms) & (perf["elapsed_ms"] <= end_ms)].copy()


def compute_event_metrics(perf: pd.DataFrame, event_row: pd.Series, next_transition_ms: int | None) -> dict:
    ttr = int(event_row["t_transition_ms"])

    pre   = _window_slice_abs(perf, ttr + WIN.pre_start,  ttr + WIN.pre_end)
    imm   = _window_slice_abs(perf, ttr + WIN.imm_start,  ttr + WIN.imm_end)
    peakw = _window_slice_abs(perf, ttr,                 ttr + WIN.peak_end)

    # deviation arrays
    pre_dev = pre["center_deviation"].to_numpy(dtype=float) if len(pre) else np.array([np.nan])
    imm_dev = imm["center_deviation"].to_numpy(dtype=float) if len(imm) else np.array([np.nan])

    pre_mean = float(np.nanmean(pre_dev))
    pre_sd   = float(np.nanstd(pre_dev))

    # --- recovery window (do not cross next transition) ---
    rec_lo = ttr + WIN.rec_start
    rec_hi = ttr + WIN.rec_end
    if next_transition_ms is not None:
        rec_hi = min(rec_hi, int(next_transition_ms) - 1)

    rec = _window_slice_abs(perf, rec_lo, rec_hi) if rec_hi > rec_lo else perf.iloc[0:0].copy()

    out = {
        "t_transition_ms": ttr,
        "from_level": event_row["from_level"],
        "to_level": event_row["to_level"],
        "delta_multiplier": float(event_row["delta_multiplier"]),
        "direction": event_row["direction"],
        "next_transition_ms": (int(next_transition_ms) if next_transition_ms is not None else None),
        "n_pre": int(len(pre)),
        "n_imm": int(len(imm)),
        "n_rec": int(len(rec)),

        "pre_mean_dev": pre_mean,
        "pre_sd_dev": pre_sd,
        "imm_mean_dev": float(np.nanmean(imm_dev)),
        "delta_dev": float(np.nanmean(imm_dev)) - pre_mean,
        "rec_mean_dev": (
            float(np.nanmean(rec["center_deviation"].to_numpy(dtype=float))) - pre_mean
            if len(rec) else float("nan")
        ),
        "peak_dev_0_10s": (
            float(np.nanmax(peakw["center_deviation"].to_numpy(dtype=float)))
            if len(peakw) else float("nan")
        ),
    }

    # --- recovery time (restricted to rec_hi) ---
    thr = pre_mean + 1.0 * pre_sd
    post = perf[(perf["elapsed_ms"] >= ttr) & (perf["elapsed_ms"] <= rec_hi)].sort_values("elapsed_ms").copy()
    ok_dev = post["center_deviation"].to_numpy(dtype=float) <= thr
    out["t_recover_dev_ms"] = _first_sustained_time(post, ok_dev, WIN.sustain_ms)

    # --- fixed residual window (same relative time across conditions) ---
    # Option 1: 8-13s (safer)
    # fix_lo = ttr + 8_000
    # fix_hi = ttr + 13_000

    # Option 2: 9-14s (closer to end of 15s segment)
    fix_lo = ttr + 11_000
    fix_hi = ttr + 14_000

    if next_transition_ms is not None:
        fix_hi = min(fix_hi, int(next_transition_ms) - 1)

    if fix_hi > fix_lo:
        wfix = _window_slice_abs(perf, fix_lo, fix_hi)
        out["residual_dev_11_14s"] = float(
            np.nanmean(wfix["center_deviation"].to_numpy(dtype=float))) - pre_mean if len(wfix) else float("nan")
    else:
        out["residual_dev_11_14s"] = float("nan")



    out["recover_success"] = 0 if np.isnan(out["t_recover_dev_ms"]) else 1
    return out