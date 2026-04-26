import pandas as pd

SCHEDULE_OFFSET_SEC = 30.0  # 你的自动模式时长（固定30s）

def find_track_start_t0(df: pd.DataFrame) -> float:
    mask = (
        (df["type"] == "event") &
        (df["module"] == "track") &
        (df["address"] == "self") &
        (df["value"] == "start")
    )
    starts = df.loc[mask, "scenario_time"].dropna()
    if starts.empty:
        raise ValueError("Cannot find track start event (event/track/self/start).")
    return float(starts.iloc[0])

def get_schedule_start_t0(df: pd.DataFrame, offset_sec: float = SCHEDULE_OFFSET_SEC) -> float:
    """
    Schedule starts offset_sec after track start.
    """
    t_track0 = find_track_start_t0(df)
    return t_track0 + float(offset_sec)

def extract_track_performance(df: pd.DataFrame, t_schedule0: float) -> pd.DataFrame:
    perf = df[(df["type"] == "performance") & (df["module"] == "track")].copy()
    perf = perf[perf["address"].isin(["center_deviation", "cursor_in_target"])].copy()
    perf["value_num"] = pd.to_numeric(perf["value"], errors="coerce")

    wide = perf.pivot_table(
        index="scenario_time",
        columns="address",
        values="value_num",
        aggfunc="last",
    ).reset_index()

    wide = wide.sort_values("scenario_time").reset_index(drop=True)

    # 以 schedule 起点作为0
    wide["t_sec"] = wide["scenario_time"] - t_schedule0
    wide = wide[wide["t_sec"] >= 0].copy()

    wide["elapsed_ms"] = (wide["t_sec"] * 1000).round().astype("int64")
    if "cursor_in_target" in wide.columns:
        wide["cursor_in_target"] = wide["cursor_in_target"].round()
    return wide