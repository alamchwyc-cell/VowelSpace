"""
reference.py — 构建母语者参照分布

诊断工具的核心: 给定一个学习者的元音测量, 判断它偏离母语者分布多远.

参照分布的定义:
  对每个元音 v,
    μ_v = 7 位母语者各自中位数点的均值 (归一化 Bark 空间)
    Σ_v = 这 7 个点的协方差 (+ 正则化, 防止奇异)

评分:
  马氏距离 d = sqrt((x-μ)ᵀ Σ⁻¹ (x-μ))
  在原假设(该发音者与母语者同分布)下 d² 近似服从 χ²(2),
  于是 score = 100 × (1 - CDF_χ²(d²))  —— 可解释为"比多少比例母语者更偏离".
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from normalize import add_bark, lobanov  # noqa: E402

VOWELS = ["iː", "ɪ", "e", "æ", "ɑː", "ɔː", "ʊ", "uː", "ʌ", "ɜː"]
F1, F2 = "F1_bark_n", "F2_bark_n"
RIDGE = 0.05          # 协方差正则化系数 (加到对角线上)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for spk, g in df.groupby("speaker"):
        m = pd.Series(True, index=g.index)
        for f in ["F1", "F2"]:
            q1, q3 = g[f].quantile([0.25, 0.75])
            iqr = q3 - q1
            m &= g[f].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        keep.append(g[m])
    return pd.concat(keep, ignore_index=True)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    return lobanov(add_bark(df))


def build(df: pd.DataFrame | None = None, group: str = "Native"):
    if df is None:
        df = normalize(clean(pd.read_csv(DATA / "formants.csv")))
    d = df[df["group"] == group]

    ref = {}
    for v in VOWELS:
        g = d[d["vowel"] == v]
        if g.empty:
            continue
        med = g.groupby("speaker")[[F1, F2]].median().dropna()
        if len(med) < 3:
            continue
        mu = med.values.mean(axis=0)
        cov = np.cov(med.values.T) + np.eye(2) * RIDGE
        ref[v] = {
            "mu": mu.tolist(),
            "cov": cov.tolist(),
            "n_speakers": int(len(med)),
            "speaker_points": med.values.tolist(),
        }
    out = {"vowels": ref, "ridge": RIDGE,
           "n_reference_speakers": int(d["speaker"].nunique())}
    (MODELS / "vowel_reference.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"参照分布已保存: {len(ref)} 个元音, {out['n_reference_speakers']} 位母语者")
    for v, r in ref.items():
        print(f"  {v:3s} μ=({r['mu'][0]:+.2f}, {r['mu'][1]:+.2f})  n={r['n_speakers']}")
    return out


def score_point(x, ref_v):
    mu = np.array(ref_v["mu"])
    cov = np.array(ref_v["cov"])
    diff = np.asarray(x, float) - mu
    d = float(np.sqrt(diff @ np.linalg.inv(cov) @ diff))
    p = 1 - chi2.cdf(d ** 2, df=2)
    return d, float(p) * 100


def score_speaker(med: pd.DataFrame, ref: dict):
    """med: 该说话人每个元音的 (F1_bark_n, F2_bark_n) 中位数 (DataFrame, index=vowel)."""
    rows = []
    for v in VOWELS:
        if v not in ref["vowels"] or v not in med.index:
            continue
        x = med.loc[v, [F1, F2]].values
        d, s = score_point(x, ref["vowels"][v])
        rows.append({"vowel": v, "F1": float(x[0]), "F2": float(x[1]),
                     "distance": d, "score": s})
    df = pd.DataFrame(rows)
    total = float(df["score"].mean()) if len(df) else float("nan")
    return df.sort_values("distance", ascending=False), total


if __name__ == "__main__":
    build()
