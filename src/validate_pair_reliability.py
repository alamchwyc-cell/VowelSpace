"""
validate_pair_reliability.py — 验证"元音对距离"能否用于个体诊断

背景:
  元音对距离是本研究的核心指标(组间 d 达 -3.83)。直觉上应直接搬进工具。
  但组均值与个体测量是两回事: 个体只有几十个 token, 中位数估计噪声大。

做法:
  对每位说话人, 以全样本算出的"元音对距离类别"为基准(弱/中/正常),
  再随机抽取 n 个 token 重算, 统计类别一致率。

判据:
  三分类随机水平为 33%。若一致率在可用录音长度(n≈50-80)下仍低于 80%,
  则认为该指标不适合作为个体诊断输出。

结果(见 results/pair_reliability.csv):
  ɪ-iː   n=20 0.645 / n=30 0.696 / n=50 0.772 / n=80 0.832
  ɑː-ʌ  n=20 0.393 / n=30 0.430 / n=50 0.484 / n=80 0.489

结论: 不稳定, 因此工具不输出元音对距离, 只输出逐元音的单点偏离。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from normalize import add_bark, lobanov  # noqa: E402

PAIRS = [("ɪ", "iː"), ("ɑː", "ʌ")]
NS = (20, 30, 50, 80)
REPS = 40


def pair_distance(g, a, b):
    med = g.groupby("vowel")[["F1_bark_n", "F2_bark_n"]].median()
    if a not in med.index or b not in med.index:
        return np.nan
    return float(np.linalg.norm(med.loc[a] - med.loc[b]))


def category(d, ref):
    if np.isnan(d):
        return None
    x = d / ref
    return "weak" if x < 0.5 else ("mid" if x < 0.8 else "normal")


def main():
    raw = pd.read_csv(DATA / "formants_all.csv").dropna(subset=["F1", "F2"])
    df = lobanov(add_bark(raw))

    nat = df[df.group == "Native"]
    ref = {p: float(np.median([pair_distance(g, *p) for _, g in nat.groupby("speaker")]))
           for p in PAIRS}
    print("母语者中位数:", {f"{a}-{b}": round(v, 2) for (a, b), v in ref.items()})

    rng = np.random.default_rng(0)
    rows = []
    for spk, g in df.groupby("speaker"):
        for p in PAIRS:
            full = pair_distance(g, *p)
            if np.isnan(full):
                continue
            base = category(full, ref[p])
            for n in NS:
                if len(g) < n:
                    continue
                agree = []
                for _ in range(REPS):
                    sub = g.sample(n=n, random_state=int(rng.integers(1e9)))
                    if sub.vowel.nunique() < 8:
                        continue
                    c = category(pair_distance(sub, *p), ref[p])
                    if c is not None:
                        agree.append(c == base)
                if agree:
                    rows.append({"pair": f"{p[0]}-{p[1]}", "n": n,
                                 "agreement": float(np.mean(agree))})

    res = pd.DataFrame(rows)
    out = res.groupby(["pair", "n"])["agreement"].mean().round(3).unstack()
    print("\n类别一致率(与全样本比较):")
    print(out.to_string())
    out.to_csv(RESULTS / "pair_reliability.csv")
    print(f"\n-> {RESULTS/'pair_reliability.csv'}")
    print("参照: 三分类随机水平 = 0.333")


if __name__ == "__main__":
    main()
