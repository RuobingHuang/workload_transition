import itertools
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

FREQ_ORDER = ["LF", "MF", "HF"]
INTENSITIES = ["LI", "HI"]
METRICS = ["recover_success_rate", "delta_mean", "resid_11_14s_mean"]

def paired_tests_within_intensity(df, metric, intensity):
    """
    For a given metric and a given intensity (LI or HI),
    run paired t-tests across freq levels: LF–MF, LF–HF, MF–HF.
    Holm-correct within these 3 comparisons.
    """
    d = df[df["intensity"] == intensity].copy()

    # pivot: participant × freq
    piv = d.pivot_table(index="participant_id", columns="freq", values=metric, aggfunc="mean")

    # ensure columns exist
    missing_cols = [f for f in FREQ_ORDER if f not in piv.columns]
    if missing_cols:
        raise ValueError(f"Missing freq columns for intensity={intensity}, metric={metric}: {missing_cols}")

    pairs = list(itertools.combinations(FREQ_ORDER, 2))
    rows = []

    for a, b in pairs:
        xa = piv[a]
        xb = piv[b]
        ok = xa.notna() & xb.notna()
        n = int(ok.sum())
        if n < 3:
            continue

        t, p = ttest_rel(xa[ok], xb[ok], nan_policy="omit")

        diff = (xa[ok] - xb[ok])
        mean_diff = float(diff.mean())

        # paired effect size: Cohen's dz = mean(diff) / sd(diff)
        sd = float(diff.std(ddof=1))
        dz = float(mean_diff / sd) if sd > 0 else np.nan

        rows.append({
            "metric": metric,
            "intensity": intensity,
            "contrast": f"{a} - {b}",
            "freq_a": a,
            "freq_b": b,
            "n": n,
            "t": float(t),
            "p": float(p),
            "mean_diff(a-b)": mean_diff,
            "cohens_dz": dz,
        })

    res = pd.DataFrame(rows)
    if res.empty:
        return res

    # Holm correction within this family (3 tests)
    rej, p_adj, _, _ = multipletests(res["p"].values, method="holm")
    res["p_adj_holm"] = p_adj
    res["reject@0.05"] = rej
    return res.sort_values("p_adj_holm")

def main():
    # 这是你前一步聚合输出的文件名；如果不同就改这里
    df = pd.read_csv("rq1_trial_aggregates_from_events.csv")

    # 基本检查
    need = {"participant_id", "freq", "intensity"} | set(METRICS)
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns: {sorted(miss)}")

    all_res = []
    for metric in METRICS:
        for inten in INTENSITIES:
            res = paired_tests_within_intensity(df, metric, inten)
            if not res.empty:
                all_res.append(res)

    if not all_res:
        print("No post-hoc results computed (maybe too many missing values).")
        return

    out = pd.concat(all_res, ignore_index=True)

    # 打印到屏幕（可选）
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 50)
    print(out.to_string(index=False))

    # 导出CSV
    out.to_csv("rq1_posthoc_freq_within_intensity_holm.csv", index=False)
    print("\nWrote: rq1_posthoc_freq_within_intensity_holm.csv")

if __name__ == "__main__":
    main()