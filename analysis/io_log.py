import pandas as pd

REQUIRED_COLS = ["logtime", "scenario_time", "type", "module", "address", "value"]

def read_trial_log(path: str) -> pd.DataFrame:
    """
    Read one MATB trial log (TSV).
    Returns a DataFrame with required columns.
    """
    df = pd.read_csv(path, sep=None, engine="python", dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")

    # 类型转换：scenario_time/logtime 应是 float
    df["scenario_time"] = pd.to_numeric(df["scenario_time"], errors="coerce")
    df["logtime"] = pd.to_numeric(df["logtime"], errors="coerce")

    # value 有些是数字，有些是字符串；这里先保留原始str，后面按需要转
    return df