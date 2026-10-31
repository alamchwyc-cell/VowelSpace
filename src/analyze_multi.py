"""
analyze_multi.py — 跨母语背景的全量分析

核心问题:
  /ɪ/–/iː/ 的范畴合并, 是普通话母语者特有的迁移, 还是各母语背景共有的现象?

设计:
  7 组说话人(6 种 L2 母语 + 英语母语), 朗读同一批 250 句.
  以"每位说话人的元音对距离"为单位, 做组间比较(说话人是分析单位, 不是 token).

输出:
  results/multi_*.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from normalize import add_bark, lobanov  # noqa: E402
from stats import cohens_d  # noqa: E402

PAIRS = [("ɪ", "iː"), ("ɑː", "ʌ"), ("ʊ", "uː"), ("æ", "e")]
ORDER = ["English", "Mandarin", "Arabic", "Hindi", "Korean", "Spanish", "Vietnamese"]


def clean(df):
    out = []
    for spk, g in df.groupby("speaker"):
        m = pd.Series(True, index=g.index)
        for f in ["F1", "F2"]:
            q1, q3 = g[f].quantile([0.25, 0.75])
            iqr = q3 - q1
            m &= g[f].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        out.append(g[m])
    return pd.concat(out, ignore_index=True)


def main():
    raw = pd.read_csv(DATA / "formants_all.csv")
    raw = raw.dropna(subset=["F1", "F2"])          # 只保留重读元音(已有 F1/F2)
    print(f"重读元音 token: {len(raw)}, 说话人 {raw.speaker.nunique()}")
    df = lobanov(add_bark(clean(raw)))

    label = df.groupby("speaker")[["group", "l1", "gender"]].first()
    label["l1"] = np.where(label["group"] == "Native", "English", label["l1"])

    # ---------- 每位说话人的元音对距离 ----------
    rows = []
    for spk, g in df.groupby("speaker"):
        med = g.groupby("vowel")[["F1_bark_n", "F2_bark_n"]].median()
        row = {"speaker": spk, "l1": label.loc[spk, "l1"],
               "gender": label.loc[spk, "gender"]}
        for a, b in PAIRS:
            row[f"{a}-{b}"] = (float(np.linalg.norm(med.loc[a] - med.loc[b]))
                               if a in med.index and b in med.index else np.nan)
        # 元音空间面积
        from scipy.spatial import ConvexHull
        pts = med.dropna().values
        row["vsa"] = float(ConvexHull(pts).volume) if len(pts) >= 3 else np.nan
        rows.append(row)
    spk_df = pd.DataFrame(rows)
    spk_df["l1"] = pd.Categorical(spk_df["l1"], categories=ORDER, ordered=True)
    spk_df = spk_df.sort_values("l1")
    spk_df.to_csv(RESULTS / "multi_speaker_level.csv", index=False)

    print("\n=== 每位说话人 (说话人为分析单位) ===")
    print(spk_df[["l1", "speaker", "ɪ-iː", "ɑː-ʌ", "vsa"]]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # ---------- 组间比较 ----------
    print("\n=== 各组均值 ± SD ===")
    summ = spk_df.groupby("l1", observed=True).agg(
        n=("speaker", "count"),
        ii_min=("ɪ-iː", "mean"), ii_sd=("ɪ-iː", "std"),
        aa_ah=("ɑː-ʌ", "mean"), vsa=("vsa", "mean")).round(3)
    print(summ.to_string())
    summ.to_csv(RESULTS / "multi_group_summary.csv")

    # Kruskal-Wallis: 所有组是否相同
    groups = [spk_df[spk_df.l1 == g]["ɪ-iː"].dropna().values for g in ORDER
              if (spk_df.l1 == g).sum() > 0]
    H, p = st.kruskal(*groups)
    print(f"\nKruskal-Wallis (ɪ-iː 距离 × 7 组): H = {H:.2f}, p = {p:.2e}")

    # 普通话 vs 其他 L2 (合并) vs 母语者
    man = spk_df[spk_df.l1 == "Mandarin"]["ɪ-iː"].dropna().values
    other = spk_df[spk_df.l1.isin(["Arabic", "Hindi", "Korean", "Spanish", "Vietnamese"])]["ɪ-iː"].dropna().values
    eng = spk_df[spk_df.l1 == "English"]["ɪ-iː"].dropna().values
    print(f"Mandarin {man.mean():.2f} (n={len(man)}) | "
          f"其他 L2 {other.mean():.2f} (n={len(other)}) | "
          f"母语者 {eng.mean():.2f} (n={len(eng)})")
    print(f"  Mandarin vs 母语者: d = {cohens_d(man, eng):.2f}, "
          f"Mann-Whitney p = {st.mannwhitneyu(man, eng).pvalue:.2e}")
    print(f"  Mandarin vs 其他 L2: d = {cohens_d(man, other):.2f}, "
          f"Mann-Whitney p = {st.mannwhitneyu(man, other).pvalue:.3f}")
    print(f"  其他 L2 vs 母语者: d = {cohens_d(other, eng):.2f}, "
          f"Mann-Whitney p = {st.mannwhitneyu(other, eng).pvalue:.2e}")

    # ---------- 时长 ----------
    dur = raw.dropna(subset=["dur"])
    piv = dur.groupby(["speaker", "vowel"])["dur"].median().unstack()
    piv["tense_ratio"] = piv["iː"] / piv["ɪ"]
    dur_df = piv.reset_index()[["speaker", "tense_ratio"]].merge(
        spk_df[["speaker", "l1"]], on="speaker")
    print("\n=== 松紧时长比 ===")
    print(dur_df.groupby("l1", observed=True)["tense_ratio"]
          .agg(["count", "mean", "std"]).round(3).to_string())
    dur_df.to_csv(RESULTS / "multi_duration.csv", index=False)

    # ---------- 语速 ----------
    pro = pd.read_csv(DATA / "prosody_all.csv")
    pro = pro.merge(spk_df[["speaker", "l1"]], on="speaker")
    print("\n=== 语速 ===")
    print(pro.groupby(["speaker"]).agg(rate=("rate", "median")).reset_index()
          .merge(spk_df[["speaker", "l1"]], on="speaker")
          .groupby("l1", observed=True)["rate"].agg(["count", "mean", "std"])
          .round(2).to_string())

    (RESULTS / "multi_summary.json").write_text(json.dumps({
        "n_speakers": int(spk_df.shape[0]),
        "kruskal_H": float(H), "kruskal_p": float(p),
        "mandarin_mean": float(man.mean()), "other_l2_mean": float(other.mean()),
        "native_mean": float(eng.mean()),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果 -> {RESULTS}")


if __name__ == "__main__":
    main()
