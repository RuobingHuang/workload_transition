"""
run_04_tail_analysis.py
────────────────────────────────────────────────────────────────
TAIL 段分析（RQ-TAIL）：经历 transition 后的最终 45 s 表现

研究问题：
  人经历不同强度（LI/HI）× 频率（LF/MF/HF）的工作负荷转换后，
  进入低负荷尾段时是否表现出与生态系统类似的：
    - 抵抗力后效（Resistance aftereffect）：TAIL 均值偏差高于 BASE
    - 恢复力后效（Resilience aftereffect）：TAIL 内仍在恢复（斜率为负）或
      稳定所需时间更长

读取：
  event_metrics_tail.csv   ← main.py 生成，含 tail_df 列（包括 compute_tail_metrics 输出）

输出：
  - 屏幕打印
  - tail_analysis_base_vs_conditions.csv  （BASE vs 6 条件配对 t 检验）
  - tail_analysis_rmanova.csv             （intensity × freq RM-ANOVA）
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import AnovaRM

# ── 条件分组 ──────────────────────────────────────────────────
TRANSITION_CONDS = ["LI_LF", "HI_LF", "LI_MF", "HI_MF", "LI_HF", "HI_HF"]

# ── 感兴趣的指标及其方向（"higher_is_worse" → True 意味着数值越大表现越差）──
TAIL_METRICS_HIGHER_WORSE = [
    "tail_mean_dev",          # TAIL 段均值偏差
    "tail_rmse_dev",          # RMSE
    "tail_std_dev",           # 段内波动
    "tail_auc_dev",           # 偏差面积
    "tail_early_mean_dev",    # 前 15 s 均值
    "tail_late_mean_dev",     # 后 15 s 均值
    "tail_time_to_stable_ms", # 首次稳定所需时间（越长越差）
    # delta_dev / imm_mean_dev（进入 TAIL 时的即时冲击，来自 compute_event_metrics）
    "delta_dev",
    "imm_mean_dev",
    "rec_mean_dev",
    "t_recover_dev_ms",
]

# 越大越好的指标
TAIL_METRICS_LOWER_WORSE = [
    "tail_hit_ratio",
    "recover_success",        # 进入 TAIL 的转换本身是否完成恢复
]

# 无方向假设（双侧）
TAIL_METRICS_TWO_SIDED = [
    "tail_slope_dev",         # 负=仍改善，正=疲劳加剧
    "tail_delta_early_late",  # 负=段内改善
]


def _add_meta(df: pd.DataFrame) -> pd.DataFrame:
    """如果没有 freq/intensity 列，从 condition 字符串解析。"""
    if "freq" not in df.columns or "intensity" not in df.columns:
        def _parse(c):
            inten, freq = c.split("_")
            return freq, inten
        df[["freq", "intensity"]] = df["condition"].apply(lambda x: pd.Series(_parse(x)))
    return df


def load_tail_csv(path: str = "event_metrics_tail.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"participant_id", "condition"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(
            f"event_metrics_tail.csv 缺少列: {sorted(miss)}\n可用列: {sorted(df.columns)}"
        )
    return df


# ─────────────────────────────────────────────────────────────
# 1. BASE vs 6 过渡条件的配对 t 检验
# ─────────────────────────────────────────────────────────────
def compare_base_vs_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """
    对每个 TAIL 指标，分别检验 6 个过渡条件是否与 BASE 不同。
    双侧检验（因为有的条件可能比 BASE 更好，有的可能更差，方向不确定）。
    Holm 校正在每个指标的 6 次比较内进行。
    """
    base = df[df["condition"] == "BASE"].set_index("participant_id")
    all_metrics = (
        [(m, "two-sided") for m in TAIL_METRICS_HIGHER_WORSE]
        + [(m, "two-sided") for m in TAIL_METRICS_LOWER_WORSE]
        + [(m, "two-sided") for m in TAIL_METRICS_TWO_SIDED]
    )

    all_rows = []

    for metric, direction in all_metrics:
        if metric not in df.columns:
            continue
        if metric not in base.columns or base[metric].isna().all():
            continue

        cond_rows = []
        for cond in TRANSITION_CONDS:
            cond_df = df[df["condition"] == cond].set_index("participant_id")
            if metric not in cond_df.columns:
                continue

            common = base.index.intersection(cond_df.index)
            xa = base.loc[common, metric].astype(float)
            xb = cond_df.loc[common, metric].astype(float)
            ok = xa.notna() & xb.notna()
            n = int(ok.sum())

            if n < 3:
                cond_rows.append({
                    "metric": metric,
                    "direction": direction,
                    "contrast": f"{cond} - BASE",
                    "condition": cond,
                    "n": n,
                    "mean_base": float(xa[ok].mean()) if ok.any() else np.nan,
                    "mean_cond": float(xb[ok].mean()) if ok.any() else np.nan,
                    "mean_diff(cond-base)": np.nan,
                    "cohens_dz": np.nan,
                    "t": np.nan,
                    "p_raw": np.nan,
                    "note": "n<3, skipped",
                })
                continue

            diff = xb[ok] - xa[ok]
            mean_diff = float(diff.mean())
            sd = float(diff.std(ddof=1))
            dz = float(mean_diff / sd) if sd > 0 else np.nan

            t, p = ttest_rel(xb[ok], xa[ok], alternative=direction)

            cond_rows.append({
                "metric": metric,
                "direction": direction,
                "contrast": f"{cond} - BASE",
                "condition": cond,
                "n": n,
                "mean_base": float(xa[ok].mean()),
                "mean_cond": float(xb[ok].mean()),
                "mean_diff(cond-base)": mean_diff,
                "cohens_dz": dz,
                "t": float(t),
                "p_raw": float(p),
                "note": "",
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
        else:
            res["p_adj_holm"] = np.nan
            res["reject@0.05"] = False

        all_rows.append(res)

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# 2. 2(intensity) × 3(freq) RM-ANOVA（仅过渡条件）
# ─────────────────────────────────────────────────────────────
def run_rmanova_tail(df: pd.DataFrame) -> pd.DataFrame:
    """
    对 6 个过渡条件做 2 × 3 被试内 RM-ANOVA。
    自变量：intensity（LI/HI）× freq（LF/MF/HF）
    """
    trans = df[df["condition"].isin(TRANSITION_CONDS)].copy()
    trans = _add_meta(trans)
    trans["freq"] = pd.Categorical(trans["freq"], ["LF", "MF", "HF"], ordered=True)
    trans["intensity"] = pd.Categorical(trans["intensity"], ["LI", "HI"], ordered=True)

    all_metrics = (
        TAIL_METRICS_HIGHER_WORSE
        + TAIL_METRICS_LOWER_WORSE
        + TAIL_METRICS_TWO_SIDED
    )

    summary_rows = []

    for metric in all_metrics:
        if metric not in trans.columns:
            continue

        d = trans[["participant_id", "freq", "intensity", metric]].dropna().copy()
        # RM-ANOVA 要求每个 cell 都有值
        cell_counts = d.groupby(
            ["participant_id", "freq", "intensity"]
        )[metric].count()
        complete_subs = cell_counts[cell_counts == 1].reset_index()["participant_id"].unique()
        d = d[d["participant_id"].isin(complete_subs)]

        if len(d["participant_id"].unique()) < 3:
            print(f"[跳过] {metric}: 完整被试数 < 3")
            continue

        try:
            aov = AnovaRM(
                d, depvar=metric, subject="participant_id",
                within=["intensity", "freq"]
            ).fit()
            print(f"\n==== RM-ANOVA: {metric} ====")
            print(aov)
            # 提取三行（intensity, freq, interaction）加入汇总
            for _, row in aov.anova_table.iterrows():
                summary_rows.append({
                    "metric": metric,
                    "effect": row.name if hasattr(row, "name") else "",
                    "F": row.get("F Value", np.nan),
                    "num_df": row.get("Num DF", np.nan),
                    "den_df": row.get("Den DF", np.nan),
                    "p": row.get("Pr > F", np.nan),
                })
        except Exception as e:
            print(f"[错误] RM-ANOVA on {metric}: {e}")

    return pd.DataFrame(summary_rows)


# ─────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────
def main():
    df = load_tail_csv("event_metrics_tail.csv")

    print(f"\n读取 event_metrics_tail.csv: {len(df)} 行，"
          f"{df['participant_id'].nunique()} 被试，"
          f"{sorted(df['condition'].unique())} 条件")

    # --- BASE vs 各过渡条件 ---
    base_comp = compare_base_vs_conditions(df)
    if base_comp.empty:
        print("\n[警告] BASE 比较结果为空（检查是否有 BASE 条件数据）。")
    else:
        pd.set_option("display.width", 220)
        pd.set_option("display.max_columns", 20)
        pd.set_option("display.float_format", lambda x: f"{x:.4f}")
        print("\n====  TAIL: BASE vs 各过渡条件（配对 t 检验 + Holm 校正）  ====")
        print(base_comp.to_string(index=False))
        base_comp.to_csv("tail_analysis_base_vs_conditions.csv", index=False)
        print("\nWrote: tail_analysis_base_vs_conditions.csv")

    # --- intensity × freq RM-ANOVA ---
    anova_summary = run_rmanova_tail(df)
    if not anova_summary.empty:
        anova_summary.to_csv("tail_analysis_rmanova.csv", index=False)
        print("\nWrote: tail_analysis_rmanova.csv")


if __name__ == "__main__":
    main()
