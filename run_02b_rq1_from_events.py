import numpy as np
import pandas as pd
from statsmodels.stats.anova import AnovaRM

TRANSITION_CONDS = ["LI_LF","HI_LF","LI_MF","HI_MF","LI_HF","HI_HF"]

# 你 event 表里应当有的列（按我们之前 pipeline 的命名）
NEEDED = {
    "participant_id", "condition",
    "delta_dev", "peak_dev_0_10s", "imm_mean_dev", "rec_mean_dev",
    "t_recover_dev_ms", "recover_success",
    "residual_dev_11_14s",
}

def add_meta(df: pd.DataFrame) -> pd.DataFrame:
    if "freq" not in df.columns or "intensity" not in df.columns:
        def parse(c):
            inten, freq = c.split("_")
            return (freq, inten)
        df[["freq","intensity"]] = df["condition"].apply(lambda x: pd.Series(parse(x)))
    return df

def check_cols(df, needed, name):
    miss = needed - set(df.columns)
    if miss:
        raise ValueError(f"{name} missing columns: {sorted(miss)}\nAvailable: {sorted(df.columns)}")

def build_trial_aggregates_from_regular_events(path="event_metrics_regular.csv") -> pd.DataFrame:
    ev = pd.read_csv(path)
    ev = ev[ev["condition"].isin(TRANSITION_CONDS)].copy()

    check_cols(ev, NEEDED, path)

    # 聚合：每个 participant×condition 一行
    g = ev.groupby(["participant_id", "condition"], as_index=False)

    out = g.agg(
        n_events=("t_transition_ms", "count") if "t_transition_ms" in ev.columns else ("delta_dev", "count"),

        peak_mean=("peak_dev_0_10s", "mean"),
        peak_p95=("peak_dev_0_10s", lambda x: np.nanpercentile(x, 95)),

        imm_mean=("imm_mean_dev", "mean"),
        rec_mean=("rec_mean_dev", "mean"),

        delta_mean=("delta_dev", "mean"),
        delta_med=("delta_dev", "median"),

        trec_med=("t_recover_dev_ms", "median"),
        trec_mean=("t_recover_dev_ms", "mean"),
        recover_success_rate=("recover_success", "mean"),

        # 主 residual（如果你改成固定窗口，就改这里的列名）
        resid_11_14s_mean=("residual_dev_11_14s", "mean"),

    )

    out = add_meta(out)
    out["freq"] = pd.Categorical(out["freq"], ["LF","MF","HF"], ordered=True)
    out["intensity"] = pd.Categorical(out["intensity"], ["LI","HI"], ordered=True)
    return out

def rm_anova(df: pd.DataFrame, metric: str):
    d = df[["participant_id","freq","intensity",metric]].dropna().copy()

    # AnovaRM 要求每个被试每个cell都有值；缺失会报错
    aov = AnovaRM(d, depvar=metric, subject="participant_id", within=["freq","intensity"]).fit()
    print(f"\n==== RM-ANOVA on {metric} ====")
    print(aov)

def main():
    df = build_trial_aggregates_from_regular_events("event_metrics_regular.csv")
    df.to_csv("rq1_trial_aggregates_from_events.csv", index=False)
    print("Wrote rq1_trial_aggregates_from_events.csv rows=", len(df))

    # 核心三指标：imm(0-5s均值), rec(5-10s均值), peak(0-10s最大值)，每个各做 2(intensity)×3(freq) RM-ANOVA
    for metric in ["imm_mean", "rec_mean", "peak_mean", "trec_med", "recover_success_rate", "delta_mean", "resid_11_14s_mean"]:
        rm_anova(df, metric)

if __name__ == "__main__":
    main()