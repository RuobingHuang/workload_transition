"""
run_05_tail_pairwise.py
────────────────────────────────────────────────────────────────
TAIL 段 6 个过渡条件两两配对 t 检验

研究问题：
  6 个过渡条件（HI_HF / HI_LF / HI_MF / LI_HF / LI_LF / LI_MF）
  之间的 TAIL 指标是否存在显著差异（不与 BASE 比较）。

分析方法：
  - 每个指标对 C(6,2)=15 对条件做配对 t 检验（双侧）
  - 在每个指标的 15 次比较内做 Holm 校正
  - Cohen's dz 作为效应量

读取：
  event_metrics_tail.csv   ← main.py 生成

输出：
  - 屏幕打印
  - tail_pairwise_conditions.csv
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

# ── 条件 & 指标（与 run_04 保持一致） ─────────────────────────
TRANSITION_CONDS = ["LI_LF", "HI_LF", "LI_MF", "HI_MF", "LI_HF", "HI_HF"]

TAIL_METRICS = [
    "tail_mean_dev",
    "tail_rmse_dev",
    "tail_std_dev",
    "tail_auc_dev",
    "tail_early_mean_dev",
    "tail_late_mean_dev",
    "tail_time_to_stable_ms",
    "tail_hit_ratio",
    "recover_success",
    "tail_slope_dev",
    "tail_delta_early_late",
]


def load_tail_csv(path: str = "event_metrics_tail.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"participant_id", "condition"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(
            f"event_metrics_tail.csv 缺少列: {sorted(miss)}\n可用列: {sorted(df.columns)}"
        )
    return df


def compare_among_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """
    在 6 个过渡条件之间做两两配对 t 检验。
    每个指标共 15 对，在指标内做 Holm 校正。
    """
    trans = {
        cond: df[df["condition"] == cond].set_index("participant_id")
        for cond in TRANSITION_CONDS
    }

    all_rows = []

    for metric in TAIL_METRICS:
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

            if n < 3:
                cond_rows.append({
                    "metric": metric,
                    "contrast": f"{cond_b} - {cond_a}",
                    "cond_a": cond_a,
                    "cond_b": cond_b,
                    "n": n,
                    "mean_a": float(xa[ok].mean()) if ok.any() else np.nan,
                    "mean_b": float(xb[ok].mean()) if ok.any() else np.nan,
                    "mean_diff(b-a)": np.nan,
                    "cohens_dz": np.nan,
                    "t": np.nan,
                    "p_raw": np.nan,
                    "note": "n<3, skipped",
                    "p_adj_holm": np.nan,
                    "reject@0.05": np.nan,
                })
                continue

            diff = xb[ok] - xa[ok]
            mean_diff = float(diff.mean())
            sd = float(diff.std(ddof=1))
            dz = float(mean_diff / sd) if sd > 0 else np.nan

            t_val, p_val = ttest_rel(xb[ok], xa[ok])

            cond_rows.append({
                "metric": metric,
                "contrast": f"{cond_b} - {cond_a}",
                "cond_a": cond_a,
                "cond_b": cond_b,
                "n": n,
                "mean_a": float(xa[ok].mean()),
                "mean_b": float(xb[ok].mean()),
                "mean_diff(b-a)": mean_diff,
                "cohens_dz": dz,
                "t": float(t_val),
                "p_raw": float(p_val),
                "note": "",
                "p_adj_holm": np.nan,
                "reject@0.05": np.nan,
            })

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


def main():
    df = load_tail_csv("event_metrics_tail.csv")

    print(f"\n读取 event_metrics_tail.csv: {len(df)} 行，"
          f"{df['participant_id'].nunique()} 被试，"
          f"{sorted(df['condition'].unique())} 条件")

    result = compare_among_conditions(df)

    if result.empty:
        print("\n[警告] 没有得到任何结果，请检查数据。")
        return

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    print("\n====  TAIL：6 个过渡条件两两配对 t 检验 + Holm 校正  ====")
    print(result.to_string(index=False))

    out_path = "tail_pairwise_conditions.csv"
    result.to_csv(out_path, index=False)
    print(f"\nWrote: {out_path}")

    # ── 汇总：仅显示 p_raw < 0.10 的行，方便快速浏览 ─────────
    trend = result[result["p_raw"] < 0.10].copy()
    if not trend.empty:
        print("\n====  p_raw < 0.10 的趋势性差异  ====")
        print(trend.to_string(index=False))
    else:
        print("\n（无 p_raw < 0.10 的结果）")


if __name__ == "__main__":
    main()
