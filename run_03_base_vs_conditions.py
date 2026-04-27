"""
run_03_base_vs_conditions.py
────────────────────────────────────────────────────────────────
方案 A：Base vs 各过渡条件的直接配对比较

使用 trial_metrics.csv（由 main.py 生成）中的 trial-level 指标：
  mean_dev / std_dev / rmse_dev / auc_dev（以及 hit_ratio 如果存在）

对每个指标，对 BASE vs 每个过渡条件各跑一次双侧配对 t 检验
（因为有的条件可能比 BASE 更好，有的可能更差，方向不确定），
然后在 6 次检验内做 Holm 校正。

输出：
  - 屏幕打印
  - rq1_base_vs_conditions.csv
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

TRANSITION_CONDS = ["LI_LF", "HI_LF", "LI_MF", "HI_MF", "LI_HF", "HI_HF"]

TRIAL_METRICS = ["mean_dev", "std_dev", "rmse_dev", "auc_dev"]
OPTIONAL_METRICS = ["hit_ratio"]


def load_trial_metrics(path: str = "trial_metrics.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"participant_id", "condition"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"trial_metrics.csv 缺少列: {sorted(miss)}\n可用列: {sorted(df.columns)}")
    return df


def compare_base_vs_conditions(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """
    对每个 metric，分别检验 6 个过渡条件是否与 BASE 不同。
    配对：以 participant_id 对齐。
    双侧检验（因为有的条件可能比 BASE 更好，有的可能更差，方向不确定）。
    """
    base = df[df["condition"] == "BASE"].set_index("participant_id")
    rows = []

    for metric in metrics:
        if metric not in df.columns:
            continue

        # 是否 base 有该指标
        if metric not in base.columns or base[metric].isna().all():
            continue

        cond_rows = []

        for cond in TRANSITION_CONDS:
            cond_df = df[df["condition"] == cond].set_index("participant_id")
            if metric not in cond_df.columns:
                continue

            # 对齐被试
            common = base.index.intersection(cond_df.index)
            n = int(common.notna().sum())

            xa = base.loc[common, metric].astype(float)
            xb = cond_df.loc[common, metric].astype(float)
            ok = xa.notna() & xb.notna()
            n = int(ok.sum())

            if n < 3:
                cond_rows.append({
                    "metric": metric,
                    "contrast": f"{cond} - BASE",
                    "condition": cond,
                    "n": n,
                    "mean_base": float(xa[ok].mean()),
                    "mean_cond": float(xb[ok].mean()),
                    "mean_diff(cond-base)": float((xb[ok] - xa[ok]).mean()),
                    "cohens_dz": np.nan,
                    "t": np.nan,
                    "p_two_sided": np.nan,
                    "note": "n<3, skipped",
                })
                continue

            diff = xb[ok] - xa[ok]
            mean_diff = float(diff.mean())
            sd = float(diff.std(ddof=1))
            dz = float(mean_diff / sd) if sd > 0 else np.nan

            t, p = ttest_rel(xb[ok], xa[ok], alternative="two-sided")

            cond_rows.append({
                "metric": metric,
                "contrast": f"{cond} - BASE",
                "condition": cond,
                "n": n,
                "mean_base": float(xa[ok].mean()),
                "mean_cond": float(xb[ok].mean()),
                "mean_diff(cond-base)": mean_diff,
                "cohens_dz": dz,
                "t": float(t),
                "p_two_sided": float(p),
                "note": "",
            })

        if not cond_rows:
            continue

        res = pd.DataFrame(cond_rows)

        # Holm 校正只在有有效 p 的行上跑
        valid = res["p_two_sided"].notna()
        if valid.sum() > 0:
            rej, p_adj, _, _ = multipletests(res.loc[valid, "p_two_sided"].values, method="holm")
            res.loc[valid, "p_adj_holm"] = p_adj
            res.loc[valid, "reject@0.05"] = rej
        else:
            res["p_adj_holm"] = np.nan
            res["reject@0.05"] = False

        rows.append(res)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def main():
    df = load_trial_metrics("trial_metrics.csv")

    # 确定实际存在的指标
    available = [m for m in TRIAL_METRICS + OPTIONAL_METRICS if m in df.columns]
    if not available:
        raise ValueError(
            f"trial_metrics.csv 中未找到任何目标指标。\n"
            f"期望: {TRIAL_METRICS}\n实际列: {sorted(df.columns)}"
        )
    print(f"将比较的指标: {available}")

    out = compare_base_vs_conditions(df, available)

    if out.empty:
        print("未能计算任何比较（检查数据是否包含 BASE 条件及充足被试数）。")
        return

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\n====  Base vs 各过渡条件（双侧配对 t 检验 + Holm 校正）  ====")
    print(out.to_string(index=False))

    out.to_csv("rq1_base_vs_conditions.csv", index=False)
    print("\nWrote: rq1_base_vs_conditions.csv")


if __name__ == "__main__":
    main()
