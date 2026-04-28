"""
run_06_tail_mechanism_analysis.py
────────────────────────────────────────────────────────────────
TAIL 段恢复机制分析：自动将 45 s TAIL 段分割为三个恢复阶段

研究目标
--------
分析不同 transition schedules 后，最终 45 s TAIL 阶段的人类恢复机制，
将 TAIL 段自动分割为：
  1. Damage Phase   : TAIL 起点后 RMSE 仍 > 个人基线 + 1 SD 的阶段
  2. Recovery Phase : Damage 结束后波动率（滑动 std）仍偏高的阶段
  3. Final Steady   : 剩余稳定阶段

个人基线 (Personal Baseline)
-----------------------------
从 BASE 条件 TAIL 末尾 10 s 计算：
  baseline_rmse  = RMSE(center_deviation)
  baseline_std   = std(center_deviation)
  baseline_hit   = mean(cursor_in_target)

阶段分割判定
------------
  Damage end   : 连续 2 s 滑动窗口 RMSE ≤ baseline_rmse + 1 × baseline_std
  Recovery end : 从 Damage 结束后，连续 4 s 滑动窗口 std ≤ 1.5 × baseline_std

读取
----
  data/expdata/index.csv  ← 含 participant_id / condition / log_path 列

输出
----
  tail_phase_segmentation.csv  ← 每个 participant × condition 的三段时长与比例
  tail_phase_rmanova.csv       ← 2(intensity) × 3(freq) RM-ANOVA
  tail_phase_pairwise.csv      ← 6 条件两两配对 t 检验（Holm 校正）
"""

import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
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
# 参数常量
# ─────────────────────────────────────────────────────────────────────
TRANSITION_CONDS = ["LI_LF", "HI_LF", "LI_MF", "HI_MF", "LI_HF", "HI_HF"]
TAIL_DURATION_MS = 45_000

BASELINE_LAST_SEC = 10          # BASE TAIL 末尾多少秒用于个人基线
DAMAGE_SUSTAIN_MS = 2_000       # Damage 结束：RMSE 持续低于阈值的时长
RECOVERY_SUSTAIN_MS = 4_000     # Recovery 结束：波动率持续低于阈值的时长
ROLLING_RMSE_WINDOW_MS = 2_000  # 滑动 RMSE 窗口时长
ROLLING_STD_WINDOW_MS = 2_000   # 滑动 std 窗口时长
DAMAGE_THR_FACTOR = 1.0         # Damage 阈值 = baseline_rmse + factor × baseline_std
RECOVERY_THR_FACTOR = 1.5       # Recovery 阈值 = factor × baseline_std

PHASE_METRICS = [
    "damage_duration_ms",
    "recovery_duration_ms",
    "steady_duration_ms",
    "damage_ratio",
    "recovery_ratio",
    "steady_ratio",
]


# ─────────────────────────────────────────────────────────────────────
# 辅助：时间对齐滑动统计量
# ─────────────────────────────────────────────────────────────────────
def _rolling_rmse(t_ms: np.ndarray, dev: np.ndarray, window_ms: int) -> np.ndarray:
    """每个时间点取前 window_ms 内样本的 RMSE。"""
    idx = pd.to_datetime(t_ms, unit="ms")
    s = pd.Series(dev, index=idx)
    rolled = s.rolling(f"{window_ms}ms").apply(
        lambda x: float(np.sqrt(np.nanmean(x**2))), raw=True
    )
    return rolled.to_numpy(dtype=float)


def _rolling_std(t_ms: np.ndarray, dev: np.ndarray, window_ms: int) -> np.ndarray:
    """每个时间点取前 window_ms 内样本的标准差。"""
    idx = pd.to_datetime(t_ms, unit="ms")
    s = pd.Series(dev, index=idx)
    rolled = s.rolling(f"{window_ms}ms").std(ddof=1)
    return rolled.to_numpy(dtype=float)


def _find_first_sustained(
    t_ms: np.ndarray, condition: np.ndarray, sustain_ms: float
) -> float:
    """
    在时间序列中找第一个条件持续满足 sustain_ms 的起始时刻（ms）。

    Parameters
    ----------
    t_ms      : 时间数组（ms，升序）
    condition : bool ndarray，与 t_ms 等长，True 表示满足条件
    sustain_ms: 需要持续满足条件的最短时长（ms）

    Returns
    -------
    float: 该持续段的起始时刻 t_ms 值；未找到则返回 NaN。
    """
    run_start_idx: int | None = None
    for i in range(len(t_ms)):
        if not condition[i]:
            run_start_idx = None
            continue
        if run_start_idx is None:
            run_start_idx = i
        if t_ms[i] - t_ms[run_start_idx] >= sustain_ms:
            return float(t_ms[run_start_idx])
    return float("nan")


# ─────────────────────────────────────────────────────────────────────
# 个人基线（BASE 条件 TAIL 末尾）
# ─────────────────────────────────────────────────────────────────────
def compute_personal_baseline(
    perf: pd.DataFrame,
    t_tail_start_ms: int,
    tail_duration_ms: int = TAIL_DURATION_MS,
    last_sec: float = BASELINE_LAST_SEC,
) -> dict:
    """
    从 perf 中截取 TAIL 末尾 last_sec 秒计算个人基线。

    Returns
    -------
    dict with keys: baseline_rmse, baseline_std, baseline_hit
    """
    _nan = float("nan")
    t_end = t_tail_start_ms + tail_duration_ms
    t_base_start = t_end - int(last_sec * 1000)
    window = _window_slice_abs(perf, t_base_start, t_end)

    if len(window) < 5:
        return {"baseline_rmse": _nan, "baseline_std": _nan, "baseline_hit": _nan}

    dev = window["center_deviation"].to_numpy(dtype=float)
    dev_ok = dev[~np.isnan(dev)]

    bl: dict = {
        "baseline_rmse": float(np.sqrt(np.mean(dev_ok**2))) if len(dev_ok) >= 2 else _nan,
        "baseline_std": float(np.std(dev_ok, ddof=1)) if len(dev_ok) >= 2 else _nan,
    }

    if "cursor_in_target" in window.columns:
        hit = window["cursor_in_target"].to_numpy(dtype=float)
        bl["baseline_hit"] = float(np.nanmean(hit))
    else:
        bl["baseline_hit"] = _nan

    return bl


# ─────────────────────────────────────────────────────────────────────
# TAIL 三阶段自动分割
# ─────────────────────────────────────────────────────────────────────
def _fill_ratios(result: dict, tail_duration_ms: int) -> None:
    for key in ("damage", "recovery", "steady"):
        dur = result.get(f"{key}_duration_ms", float("nan"))
        result[f"{key}_ratio"] = (
            float(dur) / tail_duration_ms
            if not np.isnan(float(dur))
            else float("nan")
        )


def segment_tail_phases(
    perf: pd.DataFrame,
    t_tail_start_ms: int,
    tail_duration_ms: int = TAIL_DURATION_MS,
    baseline_rmse: float = float("nan"),
    baseline_std: float = float("nan"),
    damage_sustain_ms: int = DAMAGE_SUSTAIN_MS,
    recovery_sustain_ms: int = RECOVERY_SUSTAIN_MS,
    rolling_rmse_window_ms: int = ROLLING_RMSE_WINDOW_MS,
    rolling_std_window_ms: int = ROLLING_STD_WINDOW_MS,
    damage_thr_factor: float = DAMAGE_THR_FACTOR,
    recovery_thr_factor: float = RECOVERY_THR_FACTOR,
) -> dict:
    """
    将 TAIL 段自动分为三个阶段，返回各阶段时长及比例。

    Phase 1 – Damage Phase
        TAIL 起点起，到 RMSE 首次连续 damage_sustain_ms 低于
        (baseline_rmse + damage_thr_factor × baseline_std) 为止。

    Phase 2 – Recovery Phase
        Damage 结束后，到滑动 std 首次连续 recovery_sustain_ms 低于
        (recovery_thr_factor × baseline_std) 为止。

    Phase 3 – Final Steady State
        Recovery 结束后直至 TAIL 末尾的剩余部分。
    """
    _nan = float("nan")
    t_end = t_tail_start_ms + tail_duration_ms
    tail = (
        _window_slice_abs(perf, t_tail_start_ms, t_end)
        .sort_values("elapsed_ms")
        .copy()
    )

    result: dict = {
        "tail_t_start_ms": t_tail_start_ms,
        "damage_end_ms": _nan,
        "recovery_end_ms": _nan,
        "damage_duration_ms": _nan,
        "recovery_duration_ms": _nan,
        "steady_duration_ms": _nan,
        "damage_ratio": _nan,
        "recovery_ratio": _nan,
        "steady_ratio": _nan,
        "baseline_rmse": baseline_rmse,
        "baseline_std": baseline_std,
        "damage_threshold": _nan,
        "recovery_threshold": _nan,
        "n_tail_samples": len(tail),
    }

    if len(tail) < 10 or np.isnan(baseline_rmse) or np.isnan(baseline_std):
        return result

    t_ms = tail["elapsed_ms"].to_numpy(dtype=float)
    dev = tail["center_deviation"].to_numpy(dtype=float)

    dmg_thr = baseline_rmse + damage_thr_factor * baseline_std
    rec_thr = baseline_std * recovery_thr_factor
    result["damage_threshold"] = float(dmg_thr)
    result["recovery_threshold"] = float(rec_thr)

    # ── Phase 1: Damage Phase 结束判定 ──────────────────────────────
    roll_rmse = _rolling_rmse(t_ms, dev, rolling_rmse_window_ms)
    below_dmg = roll_rmse <= dmg_thr
    damage_end_ms = _find_first_sustained(t_ms, below_dmg, damage_sustain_ms)

    if np.isnan(damage_end_ms):
        # Damage 未结束，占满整个 TAIL
        result.update(
            {
                "damage_end_ms": float(t_end),
                "recovery_end_ms": float(t_end),
                "damage_duration_ms": float(tail_duration_ms),
                "recovery_duration_ms": 0.0,
                "steady_duration_ms": 0.0,
                "damage_ratio": 1.0,
                "recovery_ratio": 0.0,
                "steady_ratio": 0.0,
            }
        )
        return result

    result["damage_end_ms"] = float(damage_end_ms)
    result["damage_duration_ms"] = float(damage_end_ms - t_tail_start_ms)

    # ── Phase 2: Recovery Phase 结束判定 ────────────────────────────
    rec_mask = t_ms >= damage_end_ms
    if rec_mask.sum() < 5:
        result.update(
            {
                "recovery_end_ms": float(damage_end_ms),
                "recovery_duration_ms": 0.0,
                "steady_duration_ms": float(t_end - damage_end_ms),
            }
        )
        _fill_ratios(result, tail_duration_ms)
        return result

    t_rec = t_ms[rec_mask]
    dev_rec = dev[rec_mask]
    roll_std_rec = _rolling_std(t_rec, dev_rec, rolling_std_window_ms)
    below_rec = roll_std_rec <= rec_thr
    recovery_end_ms = _find_first_sustained(t_rec, below_rec, recovery_sustain_ms)

    if np.isnan(recovery_end_ms):
        # Recovery 未结束，占满 Damage 结束后的剩余部分
        result.update(
            {
                "recovery_end_ms": float(t_end),
                "recovery_duration_ms": float(t_end - damage_end_ms),
                "steady_duration_ms": 0.0,
            }
        )
    else:
        result.update(
            {
                "recovery_end_ms": float(recovery_end_ms),
                "recovery_duration_ms": float(recovery_end_ms - damage_end_ms),
                "steady_duration_ms": float(t_end - recovery_end_ms),
            }
        )

    _fill_ratios(result, tail_duration_ms)
    return result


# ─────────────────────────────────────────────────────────────────────
# 主流水线：读取所有试次并分割 TAIL
# ─────────────────────────────────────────────────────────────────────
def _get_tail_start_ms(condition: str) -> int:
    """从 SCHEDULES 中计算该条件 TAIL 段的起始时间（ms）。"""
    _, tail_ev = build_transition_events_split(SCHEDULES[condition])
    if tail_ev.empty:
        raise ValueError(f"条件 {condition!r} 没有找到 TAIL 事件")
    return int(tail_ev.iloc[0]["t_transition_ms"])


def run_segmentation(index_csv: str) -> pd.DataFrame:
    """
    对 index_csv 中所有 participant × condition 做 TAIL 三段分割。

    流程：
      1. 读取所有 BASE 条件，计算每人个人基线
      2. 对各过渡条件读取原始 log，提取 TAIL 性能数据并分割三段
    """
    idx = pd.read_csv(index_csv)
    required = {"participant_id", "condition", "log_path"}
    missing = required - set(idx.columns)
    if missing:
        raise ValueError(f"index_csv 缺少列: {sorted(missing)}")

    # ── Step 1: 个人基线（BASE 条件）─────────────────────────────────
    print("正在计算个人基线（BASE 条件 TAIL 末尾 10 s）…")
    baselines: dict[str, dict] = {}
    base_rows = idx[idx["condition"] == "BASE"]
    if base_rows.empty:
        print("[警告] 未找到 BASE 条件行，基线将全部为 NaN。")

    for row in base_rows.itertuples(index=False):
        pid = str(row.participant_id)
        try:
            df_log = read_trial_log(row.log_path)
            t0 = get_schedule_start_t0(df_log)
            perf = extract_track_performance(df_log, t0)
            t_tail = _get_tail_start_ms("BASE")
            bl = compute_personal_baseline(perf, t_tail)
            baselines[pid] = bl
            print(
                f"  {pid}  RMSE={bl['baseline_rmse']:.3f}  "
                f"std={bl['baseline_std']:.3f}  hit={bl['baseline_hit']:.3f}"
            )
        except Exception as exc:
            print(f"  [警告] {pid} BASE 基线计算失败: {exc}")
            baselines[pid] = {
                "baseline_rmse": float("nan"),
                "baseline_std": float("nan"),
                "baseline_hit": float("nan"),
            }

    _nan_bl = {
        "baseline_rmse": float("nan"),
        "baseline_std": float("nan"),
        "baseline_hit": float("nan"),
    }

    # ── Step 2: 过渡条件 TAIL 三段分割 ──────────────────────────────
    print("\n正在分割 TAIL 三阶段…")
    all_rows = []

    for row in idx.itertuples(index=False):
        pid = str(row.participant_id)
        cond = str(row.condition)
        if cond == "BASE":
            continue

        meta = CONDITION_META.get(cond, {})
        bl = baselines.get(pid, _nan_bl)

        try:
            df_log = read_trial_log(row.log_path)
            t0 = get_schedule_start_t0(df_log)
            perf = extract_track_performance(df_log, t0)
            t_tail = _get_tail_start_ms(cond)
            phases = segment_tail_phases(
                perf,
                t_tail_start_ms=t_tail,
                tail_duration_ms=TAIL_DURATION_MS,
                baseline_rmse=bl["baseline_rmse"],
                baseline_std=bl["baseline_std"],
            )
            all_rows.append(
                {
                    "participant_id": pid,
                    "condition": cond,
                    **meta,
                    **{k: bl[k] for k in ("baseline_rmse", "baseline_std", "baseline_hit")},
                    **phases,
                }
            )
            print(
                f"  {pid} {cond}: "
                f"damage={phases['damage_duration_ms']:.0f} ms  "
                f"recovery={phases['recovery_duration_ms']:.0f} ms  "
                f"steady={phases['steady_duration_ms']:.0f} ms"
            )
        except Exception as exc:
            print(f"  [错误] {pid} {cond}: {exc}")

    return pd.DataFrame(all_rows)


# ─────────────────────────────────────────────────────────────────────
# 统计分析
# ─────────────────────────────────────────────────────────────────────
def _add_meta(df: pd.DataFrame) -> pd.DataFrame:
    if "freq" not in df.columns or "intensity" not in df.columns:

        def _parse(c: str):
            inten, freq = c.split("_")
            return freq, inten

        df[["freq", "intensity"]] = df["condition"].apply(
            lambda x: pd.Series(_parse(x))
        )
    return df


def run_rmanova(df: pd.DataFrame) -> pd.DataFrame:
    """
    对 6 个过渡条件做 2(intensity) × 3(freq) 被试内 RM-ANOVA。
    返回包含所有指标各效应 F / p 的汇总 DataFrame。
    """
    trans = df[df["condition"].isin(TRANSITION_CONDS)].copy()
    trans = _add_meta(trans)
    trans["freq"] = pd.Categorical(trans["freq"], ["LF", "MF", "HF"], ordered=True)
    trans["intensity"] = pd.Categorical(
        trans["intensity"], ["LI", "HI"], ordered=True
    )

    summary_rows = []
    for metric in PHASE_METRICS:
        if metric not in trans.columns:
            continue

        d = trans[["participant_id", "freq", "intensity", metric]].dropna().copy()
        cell_counts = d.groupby(["participant_id", "freq", "intensity"])[metric].count()
        complete_subs = (
            cell_counts[cell_counts == 1].reset_index()["participant_id"].unique()
        )
        d = d[d["participant_id"].isin(complete_subs)]

        if d["participant_id"].nunique() < 3:
            print(f"[跳过 RM-ANOVA] {metric}: 完整被试数 < 3")
            continue

        try:
            aov = AnovaRM(
                d, depvar=metric, subject="participant_id", within=["intensity", "freq"]
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


def run_pairwise(df: pd.DataFrame) -> pd.DataFrame:
    """
    对 6 个过渡条件两两做配对 t 检验（双侧），在每个指标的 15 次比较内做 Holm 校正。
    """
    trans = {
        cond: df[df["condition"] == cond].set_index("participant_id")
        for cond in TRANSITION_CONDS
    }

    all_rows = []
    for metric in PHASE_METRICS:
        if metric not in df.columns:
            continue

        cond_rows = []
        for cond_a, cond_b in combinations(TRANSITION_CONDS, 2):
            da = trans[cond_a]
            db = trans[cond_b]
            if metric not in da.columns or metric not in db.columns:
                continue

            common = da.index.intersection(db.index)
            xa = da.loc[common, metric].astype(float)
            xb = db.loc[common, metric].astype(float)
            ok = xa.notna() & xb.notna()
            n = int(ok.sum())

            base_row = {
                "metric": metric,
                "contrast": f"{cond_b} - {cond_a}",
                "cond_a": cond_a,
                "cond_b": cond_b,
                "n": n,
                "mean_a": float(xa[ok].mean()) if ok.any() else np.nan,
                "mean_b": float(xb[ok].mean()) if ok.any() else np.nan,
            }

            if n < 3:
                cond_rows.append(
                    {
                        **base_row,
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

            cond_rows.append(
                {
                    **base_row,
                    "mean_diff(b-a)": mean_diff,
                    "cohens_dz": dz,
                    "t": float(t_val),
                    "p_raw": float(p_val),
                    "p_adj_holm": np.nan,
                    "reject@0.05": np.nan,
                    "note": "",
                }
            )

        if not cond_rows:
            continue

        res = pd.DataFrame(cond_rows)
        valid = res["p_raw"].notna()
        if valid.sum() > 0:
            rej, p_adj, _, _ = multipletests(
                res.loc[valid, "p_raw"].values, method="holm"
            )
            res.loc[valid, "p_adj_holm"] = p_adj
            res.loc[valid, "reject@0.05"] = rej

        all_rows.append(res)

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────
def main(index_csv: str = "data/expdata/index.csv") -> None:
    # ── 1. 三段分割 ──────────────────────────────────────────────────
    seg_df = run_segmentation(index_csv)

    if seg_df.empty:
        print("\n[警告] 未能得到任何分割结果，请检查数据路径与内容。")
        return

    out_seg = "tail_phase_segmentation.csv"
    seg_df.to_csv(out_seg, index=False)
    print(f"\nWrote: {out_seg}  ({len(seg_df)} 行)")

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda x: f"{x:.1f}")
    cols_show = [
        "participant_id",
        "condition",
        "damage_duration_ms",
        "recovery_duration_ms",
        "steady_duration_ms",
        "damage_ratio",
        "recovery_ratio",
        "steady_ratio",
    ]
    print("\n====  TAIL 三段时长摘要  ====")
    print(
        seg_df[[c for c in cols_show if c in seg_df.columns]].to_string(index=False)
    )

    # ── 2. RM-ANOVA ───────────────────────────────────────────────────
    anova_df = run_rmanova(seg_df)
    if not anova_df.empty:
        out_anova = "tail_phase_rmanova.csv"
        anova_df.to_csv(out_anova, index=False)
        print(f"\nWrote: {out_anova}")

    # ── 3. 两两配对 t 检验 ─────────────────────────────────────────────
    pair_df = run_pairwise(seg_df)
    if not pair_df.empty:
        pd.set_option("display.float_format", lambda x: f"{x:.4f}")
        print("\n====  TAIL 三段：6 条件两两配对 t 检验 + Holm 校正  ====")
        print(pair_df.to_string(index=False))
        out_pair = "tail_phase_pairwise.csv"
        pair_df.to_csv(out_pair, index=False)
        print(f"\nWrote: {out_pair}")

        trend = pair_df[pair_df["p_raw"] < 0.10]
        if not trend.empty:
            print("\n====  p_raw < 0.10 的趋势性差异  ====")
            print(trend.to_string(index=False))
        else:
            print("\n（无 p_raw < 0.10 的结果）")


if __name__ == "__main__":
    main()
