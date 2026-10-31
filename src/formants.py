"""
formants.py — 共振峰提取

在给定的时间窗内测量 F1/F2 的中位数。

为什么要取中位数而不是单点:
  LPC 共振峰估计在单个时间点上容易跳变(尤其在浊音起始/结束处),
  在一段稳定窗口内取中位数可以抑制这种抖动.

为什么要限制取值范围:
  成人 F1 大致 200-1000 Hz, F2 大致 500-3500 Hz. 超出范围的读数几乎都是
  LPC 估计失败(例如把 F0 当成 F1), 必须剔除而不是"取个平均值凑数".
"""
from pathlib import Path

import numpy as np
import parselmouth
from parselmouth.praat import call


def load_sound(path, target_sr: int | None = None):
    snd = parselmouth.Sound(str(path))
    if target_sr and abs(snd.sampling_frequency - target_sr) > 1:
        snd = snd.resample(target_sr)
    return snd


def measure_formants(snd, t0: float, t1: float,
                     max_formant: float = 5500.0,
                     f1_range=(200, 1200), f2_range=(500, 3500)):
    """在 [t0, t1] 内估计 F1/F2 的中位数.

    返回 (F1, F2) 或 (nan, nan) 表示该窗口不可靠.
    """
    dur = snd.duration
    t0 = max(0.0, min(t0, dur - 0.01))
    t1 = max(t0 + 0.01, min(t1, dur))

    formant = snd.to_formant_burg(
        time_step=0.005, max_number_of_formants=5,
        maximum_formant=max_formant, window_length=0.025,
    )

    times = np.arange(t0, t1, 0.005)
    f1s, f2s = [], []
    for t in times:
        f1 = call(formant, "Get value at time", 1, float(t), "Hertz", "linear")
        f2 = call(formant, "Get value at time", 2, float(t), "Hertz", "linear")
        if f1 is None or f2 is None or np.isnan(f1) or np.isnan(f2):
            continue
        if not (f1_range[0] <= f1 <= f1_range[1]):
            continue
        if not (f2_range[0] <= f2 <= f2_range[1]) or f2 <= f1:
            continue
        f1s.append(f1)
        f2s.append(f2)

    if len(f1s) < 2:          # 少于 2 帧视为失败
        return float("nan"), float("nan"), 0
    return float(np.median(f1s)), float(np.median(f2s)), len(f1s)


def vowel_measurements(snd, segments, vowel_set, vowel_map, keep_stress=(1, 2)):
    """对每个元音段测量共振峰.

    segments: [(phone, t_start, t_end), ...]  来自强制对齐
    vowel_set: 目标元音集合 (ARPAbet, 不带重音数字)
    vowel_map: ARPAbet -> 便于统计的标签 (如 AA -> ɑː)
    keep_stress: 只保留这些重音等级的元音; 传 None 表示全部保留.
                 默认只保留重读 (1/2) —— 非重读元音会弱化(尤其 AH0 -> schwa),
                 其共振峰不代表目标元音.
    """
    import re

    rows = []
    for phone, t0, t1 in segments:
        raw = phone.upper()
        stress = None
        m = re.search(r"(\d)$", raw)
        if m:
            stress = int(m.group(1))
            raw = raw[:-1]
        if raw not in vowel_set:
            continue
        if keep_stress is not None and stress is not None and stress not in keep_stress:
            continue
        dur = t1 - t0
        if dur < 0.04:                    # 太短, 测不准
            continue
        # 取中间 50% 作为稳定段, 避开过渡
        mid0 = t0 + dur * 0.25
        mid1 = t0 + dur * 0.75
        f1, f2, n = measure_formants(snd, mid0, mid1)
        rows.append({
            "phone": raw, "vowel": vowel_map.get(raw, raw),
            "t0": t0, "t1": t1, "dur": dur,
            "F1": f1, "F2": f2, "n_frames": n,
        })
    return rows
