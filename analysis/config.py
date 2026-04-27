from dataclasses import dataclass
#统一配置（条件、schedule、窗口参数）
# 你在代码里写死的倍速（用于量化Δmultiplier）
LEVEL_TO_MULT = {
    "LOW": 1.5,
    "MID": 3.0,
    "HIGH": 4.5,
    "TAIL": 1.5,   # 按LOW处理
}

@dataclass(frozen=True)
class WindowsMs:
    pre_start: int = -10_000
    pre_end: int = -2_000
    imm_start: int = 0
    imm_end: int = 5_000
    peak_end: int = 10_000  # 0~10s 内最大偏差（独立 max 指标）
    rec_start: int = 5_000   # recovery 窗口起点（与 imm 紧接，不重叠）
    rec_end: int = 10_000    # recovery 窗口终点（对齐最短 10s 段）
    sustain_ms: int = 2_000  # “恢复需要持续满足阈值”的时长

WIN = WindowsMs()


# 你需要为7种模式各自放一个schedule（你之前已经列出来了）
# schedule = [(duration_ms, level_str), ...]
SCHEDULES = {}

SCHEDULES["LI_LF"] = [
    (60000, "LOW"),
    (60000, "MID"),
    (60000, "LOW"),
    (60000, "MID"),
    (45000, "TAIL"),
]
SCHEDULES["HI_LF"] = [
    (90000, "LOW"),
    (30000, "HIGH"),
    (90000, "LOW"),
    (30000, "HIGH"),
    (45000, "TAIL"),
]
SCHEDULES["LI_MF"] = [
    (30000, "LOW"), (30000, "MID"),
    (30000, "LOW"), (30000, "MID"),
    (30000, "LOW"), (30000, "MID"),
    (30000, "LOW"), (30000, "MID"),
    (45000, "TAIL"),
]
SCHEDULES["HI_MF"] = [
    (45000, "LOW"), (15000, "HIGH"),
    (45000, "LOW"), (15000, "HIGH"),
    (45000, "LOW"), (15000, "HIGH"),
    (45000, "LOW"), (15000, "HIGH"),
    (45000, "TAIL"),
]
SCHEDULES["LI_HF"] = (
    [(20000, "LOW"), (20000, "MID")] * 6
    + [(45000, "TAIL")]
)
SCHEDULES["HI_HF"] = (
    [(30000, "LOW"), (10000, "HIGH")] * 6
    + [(45000, "TAIL")]
)
SCHEDULES["BASE"] = [
    (240000, "LOW"),
    (45000, "TAIL"),
]



# ---- CONDITION_META: 用于 trial-level / event-level 的分组因子 ----
# freq: LF / MF / HF
# intensity: LI / HI
CONDITION_META = {
    "LI_LF": dict(freq="LF", intensity="LI"),
    "HI_LF": dict(freq="LF", intensity="HI"),
    "LI_MF": dict(freq="MF", intensity="LI"),
    "HI_MF": dict(freq="MF", intensity="HI"),
    "LI_HF": dict(freq="HF", intensity="LI"),
    "HI_HF": dict(freq="HF", intensity="HI"),
    "BASE":  dict(freq="BASE", intensity="NONE"),
}