"""
vowelspace.py — 元音空间几何量

两个核心指标:
  1) 元音空间面积 (vowel space area, VSA): 一个说话人所有单元音的
     (F1, F2) 点在 Bark 平面上的凸包面积. 面积越小, 说明各元音越挤.
  2) 元音对距离: 特定元音之间的欧氏距离, 用于检验"混音"假设
     (如 /ɪ/-/iː/ 距离变小).
"""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform


def convex_hull_area(points: np.ndarray) -> float:
    """二维点集的凸包面积. 点少于 3 个或共线时返回 nan."""
    pts = np.asarray(points, dtype=float)
    pts = pts[~np.isnan(pts).any(axis=1)]
    if len(pts) < 3:
        return float("nan")
    try:
        return float(ConvexHull(pts).volume)      # 二维下 volume 即面积
    except Exception:
        return float("nan")


def speaker_vsa(df: pd.DataFrame, speaker_col="speaker",
                f1="F1_bark_n", f2="F2_bark_n", vowel_col="vowel") -> pd.DataFrame:
    """每个说话人的元音空间面积.

    每个元音先取该说话人的中位数 (消除同一元音多次出现的重复计数),
    再对中位数点求凸包.
    """
    rows = []
    for spk, g in df.groupby(speaker_col):
        med = g.groupby(vowel_col)[[f1, f2]].median().dropna()
        area = convex_hull_area(med.values)
        rows.append({speaker_col: spk, "vsa": area,
                     "n_vowels": len(med), "n_tokens": len(g)})
    return pd.DataFrame(rows)


def pairwise_distances(df: pd.DataFrame, pairs, speaker_col="speaker",
                       f1="F1_bark_n", f2="F2_bark_n", vowel_col="vowel") -> pd.DataFrame:
    """指定元音对在每个说话人内的欧氏距离 (用中位数点)."""
    rows = []
    for spk, g in df.groupby(speaker_col):
        med = g.groupby(vowel_col)[[f1, f2]].median()
        for a, b in pairs:
            if a in med.index and b in med.index:
                d = float(np.linalg.norm(med.loc[a].values - med.loc[b].values))
                rows.append({speaker_col: spk, "pair": f"{a}-{b}", "distance": d})
    return pd.DataFrame(rows)


def distance_matrix(df: pd.DataFrame, speaker: str, speaker_col="speaker",
                    f1="F1_bark_n", f2="F2_bark_n", vowel_col="vowel"):
    """单个说话人的元音间距离矩阵 (便于画热图)."""
    g = df[df[speaker_col] == speaker]
    med = g.groupby(vowel_col)[[f1, f2]].median().dropna()
    if len(med) < 2:
        return None
    d = squareform(pdist(med.values))
    return pd.DataFrame(d, index=med.index, columns=med.index)
