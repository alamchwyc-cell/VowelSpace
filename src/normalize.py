"""
normalize.py — 说话人归一化与频率标度转换

正确的顺序（这一点很容易搞错）:
    Hz --(Bark)--> Bark --(Lobanov)--> 归一化坐标

    先转 Bark 再归一化，理由:
      - Bark 是人耳感知标度，距离与面积才有心理声学意义；
      - 归一化是线性变换，先做哪个在数学上都可以，
        但"面积"必须在同一个线性空间里算，所以统一到 Bark 后再 z-score。

为什么要归一化:
    共振峰频率随声道长度变化。女性 F1/F2 系统性高于男性约 15-20%，
    远大于母语背景带来的差异。不归一化，得到的"组间差异"其实是性别差异。
"""
import numpy as np
import pandas as pd


def hz_to_bark(f):
    """Traunmüller (1990) 的 Hz -> Bark 近似."""
    f = np.asarray(f, dtype=float)
    return 26.81 * f / (1960.0 + f) - 0.53


def hz_to_erb(f):
    """Glasberg & Moore 的 ERB-rate."""
    f = np.asarray(f, dtype=float)
    return 21.4 * np.log10(1 + 0.00437 * f)


def add_bark(df: pd.DataFrame, formants=("F1", "F2")) -> pd.DataFrame:
    out = df.copy()
    for f in formants:
        out[f"{f}_bark"] = hz_to_bark(out[f])
    return out


def lobanov(df: pd.DataFrame, cols=("F1_bark", "F2_bark"),
            speaker="speaker", suffix="_n") -> pd.DataFrame:
    """Lobanov 归一化: 每个说话人、每个维度做 z-score.

    默认作用在 Bark 坐标上, 输出 <col><suffix>.
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        g = out.groupby(speaker)[c]
        mu = g.transform("mean")
        sd = g.transform("std")
        out[f"{c}{suffix}"] = (out[c] - mu) / sd.replace(0, np.nan)
    return out


def nearey(df: pd.DataFrame, cols=("F1_bark", "F2_bark"),
           speaker="speaker", suffix="_nearey") -> pd.DataFrame:
    """Nearey 的对数均值归一化 (稳健性检验用).

    在 Bark 上做 log 再减去说话人均值.
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        lg = np.log(out[c] - out[c].min() + 1.0)
        out[f"{c}{suffix}"] = lg - lg.groupby(out[speaker]).transform("mean")
    return out
