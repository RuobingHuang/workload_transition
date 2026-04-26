from dataclasses import dataclass
from typing import List, Tuple, Dict
import pandas as pd
from .config import LEVEL_TO_MULT

#从 schedule 生成 segments 与 transition events
@dataclass(frozen=True)
class TransitionEvent:
    t_transition_ms: int
    from_level: str
    to_level: str
    delta_multiplier: float
    direction: str  # "up" or "down"

def build_segments(schedule: List[Tuple[int, str]]) -> pd.DataFrame:
    """
    schedule: [(duration_ms, level), ...]
    return segments dataframe: start_ms, end_ms, level
    """
    rows = []
    cur = 0
    for dur, lvl in schedule:
        rows.append(dict(start_ms=cur, end_ms=cur + int(dur), level=str(lvl)))
        cur += int(dur)
    return pd.DataFrame(rows)

def build_transition_events(schedule: List[Tuple[int, str]]) -> pd.DataFrame:
    seg = build_segments(schedule)
    events = []
    for i in range(1, len(seg)):
        t = int(seg.loc[i, "start_ms"])
        from_level = seg.loc[i-1, "level"]
        to_level = seg.loc[i, "level"]
        dm = abs(LEVEL_TO_MULT[to_level] - LEVEL_TO_MULT[from_level])
        direction = "up" if LEVEL_TO_MULT[to_level] > LEVEL_TO_MULT[from_level] else "down"
        events.append(dict(
            t_transition_ms=t,
            from_level=from_level,
            to_level=to_level,
            delta_multiplier=dm,
            direction=direction,
        ))
    return pd.DataFrame(events)

def total_duration_ms(schedule: List[Tuple[int, str]]) -> int:
    return int(sum(d for d, _ in schedule))

def build_transition_events_split(schedule: List[Tuple[int, str]]):
    """
    Returns (regular_events_df, tail_event_df)
    - regular_events: transitions where to_level != 'TAIL'
    - tail_event: the single transition into TAIL (if exists), as a 1-row df (or empty)
    """
    seg = build_segments(schedule)
    events = []
    for i in range(1, len(seg)):
        t = int(seg.loc[i, "start_ms"])
        from_level = seg.loc[i-1, "level"]
        to_level = seg.loc[i, "level"]
        dm = abs(LEVEL_TO_MULT[to_level] - LEVEL_TO_MULT[from_level])
        direction = "up" if LEVEL_TO_MULT[to_level] > LEVEL_TO_MULT[from_level] else "down"
        events.append(dict(
            t_transition_ms=t,
            from_level=from_level,
            to_level=to_level,
            delta_multiplier=dm,
            direction=direction,
        ))
    evdf = pd.DataFrame(events)
    if evdf.empty:
        return evdf, evdf

    tail_df = evdf[evdf["to_level"] == "TAIL"].copy()
    reg_df = evdf[evdf["to_level"] != "TAIL"].copy()
    return reg_df.reset_index(drop=True), tail_df.reset_index(drop=True)