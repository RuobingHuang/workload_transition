import pandas as pd
from .io_log import read_trial_log
from .schedule import build_transition_events_split
from .metrics_trial import compute_trial_metrics
from .metrics_event import compute_event_metrics
from .metrics_tail import compute_tail_metrics
from .config import SCHEDULES, CONDITION_META
from .preprocess import get_schedule_start_t0, extract_track_performance



def run_one_trial(log_path: str, participant_id: str, condition: str):
    df = read_trial_log(log_path)
    t_schedule0 = get_schedule_start_t0(df, offset_sec=30.0)
    perf = extract_track_performance(df, t_schedule0)
    trial_metrics = compute_trial_metrics(perf)
    meta = CONDITION_META.get(condition, {})
    trial_df = pd.DataFrame([{
        "participant_id": participant_id,
        "condition": condition,
        **meta,
        **trial_metrics
    }])

    schedule = SCHEDULES[condition]
    reg_events, tail_events = build_transition_events_split(schedule)

    def compute_for_events(events_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        t_list = events_df["t_transition_ms"].tolist() if not events_df.empty else []
        for i, ev in events_df.iterrows():
            next_ms = t_list[i+1] if i+1 < len(t_list) else None
            m = compute_event_metrics(perf, ev, next_transition_ms=next_ms)
            rows.append({
                "participant_id": participant_id,
                "condition": condition,
                **meta,
                **m
            })
        return pd.DataFrame(rows)

    reg_df = compute_for_events(reg_events)
    tail_df = compute_for_events(tail_events)

    # TAIL 段体指标：描述整个 45 s 低负荷尾段的表现（抵抗力/恢复力后效）
    if not tail_df.empty and not tail_events.empty:
        tail_body = compute_tail_metrics(perf, tail_events.iloc[0], reg_df)
        for k, v in tail_body.items():
            tail_df[k] = v

    return trial_df, reg_df, tail_df