import pandas as pd
from .pipeline_trial import run_one_trial

def run_all(index_csv: str,
            out_trial_csv: str = "trial_metrics.csv",
            out_event_regular_csv: str = "event_metrics_regular.csv",
            out_event_tail_csv: str = "event_metrics_tail.csv"):
    idx = pd.read_csv(index_csv)
    required = {"participant_id", "condition", "log_path"}
    if not required.issubset(set(idx.columns)):
        raise ValueError(f"index_csv must have columns {required}")

    trial_all, reg_all, tail_all = [], [], []

    for row in idx.itertuples(index=False):
        tdf, rdf, tldf = run_one_trial(row.log_path, row.participant_id, row.condition)
        trial_all.append(tdf)
        if len(rdf): reg_all.append(rdf)
        if len(tldf): tail_all.append(tldf)

    trial_df = pd.concat(trial_all, ignore_index=True)
    trial_df.to_csv(out_trial_csv, index=False)

    if reg_all:
        reg_df = pd.concat(reg_all, ignore_index=True)
        reg_df.to_csv(out_event_regular_csv, index=False)
    else:
        reg_df = pd.DataFrame()

    if tail_all:
        tail_df = pd.concat(tail_all, ignore_index=True)
        tail_df.to_csv(out_event_tail_csv, index=False)
    else:
        tail_df = pd.DataFrame()

    return trial_df, reg_df, tail_df