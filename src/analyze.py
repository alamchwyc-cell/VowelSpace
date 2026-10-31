"""
analyze.py — 主分析

输入: data/formants.csv (每个元音 token 的 F1/F2)
输出:
  results/vsa_by_speaker.csv      每位说话人的元音空间面积
  results/pair_distances.csv      元音对距离
  results/mixed_model.txt         混合效应模型摘要
  results/summary.json            关键指标汇总

分析顺序:
  1) 清洗: 剔除异常值 (按说话人内 IQR)
  2) 归一化: Lobanov (主) + Nearey (稳健性)
  3) 换标度: Bark
  4) 指标: 元音空间面积、元音对距离
  5) 统计: 混合效应模型 (token 级) + 置换检验 (说话人级)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGS = ROOT / "figs"
for d in (RESULTS, FIGS):
    d.mkdir(exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize import add_bark, lobanov, nearey  # noqa: E402
from stats import cohens_d, mixed_model, permutation_test  # noqa: E402
from vowelspace import pairwise_distances, speaker_vsa  # noqa: E402

PAIRS = [("ɪ", "iː"), ("æ", "e"), ("ʊ", "uː"), ("ɑː", "ʌ"), ("e", "ɪ")]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """按说话人剔除极端值 (F1/F2 超出 1.5×IQR), 处理 LPC 估计失败."""
    out = []
    for spk, g in df.groupby("speaker"):
        keep = pd.Series(True, index=g.index)
        for f in ["F1", "F2"]:
            q1, q3 = g[f].quantile([0.25, 0.75])
            iqr = q3 - q1
            keep &= g[f].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        out.append(g[keep])
    return pd.concat(out, ignore_index=True)


def main():
    df = pd.read_csv(DATA / "formants.csv")
    print(f"原始 token: {len(df)}, 说话人 {df.speaker.nunique()}")

    df = clean(df)
    print(f"清洗后 token: {len(df)}")

    # 顺序: Hz -> Bark -> Lobanov(在 Bark 上)
    df = add_bark(df)
    df = lobanov(df)
    df = nearey(df)

    # ---------- 元音空间面积 ----------
    vsa = speaker_vsa(df, f1="F1_bark_n", f2="F2_bark_n")
    meta = df.groupby("speaker")[["group", "gender"]].first()
    vsa = vsa.merge(meta, left_on="speaker", right_index=True)
    vsa.to_csv(RESULTS / "vsa_by_speaker.csv", index=False)
    print("\n=== 元音空间面积 (Bark²) ===")
    print(vsa[["speaker", "group", "gender", "n_vowels", "vsa"]]
          .sort_values(["group", "vsa"], ascending=[True, False]).to_string(index=False))

    l2 = vsa[vsa.group == "L2"]["vsa"].values
    na = vsa[vsa.group == "Native"]["vsa"].values
    perm = permutation_test(vsa["vsa"].values, vsa["group"].values, n_perm=10000)
    d = cohens_d(l2, na)
    print(f"\nL2 均值 {np.mean(l2):.2f} vs 母语者 {np.mean(na):.2f}")
    print(f"差值 {perm['observed_diff']:.2f}, 置换 p = {perm['p_value']:.4f}, Cohen's d = {d:.2f}")

    # ---------- 元音对距离 ----------
    dist = pairwise_distances(df, PAIRS, f1="F1_bark_n", f2="F2_bark_n")
    dist = dist.merge(meta, left_on="speaker", right_index=True)
    dist.to_csv(RESULTS / "pair_distances.csv", index=False)

    pair_summary = []
    for p in sorted(dist["pair"].unique()):
        a = dist[(dist.pair == p) & (dist.group == "L2")]["distance"].values
        b = dist[(dist.pair == p) & (dist.group == "Native")]["distance"].values
        pair_summary.append({"pair": p, "L2_mean": np.mean(a), "Native_mean": np.mean(b),
                             "diff": np.mean(a) - np.mean(b), "d": cohens_d(a, b)})
    ps = pd.DataFrame(pair_summary)
    ps.to_csv(RESULTS / "pair_summary.csv", index=False)
    print("\n=== 元音对距离 (Bark) ===")
    print(ps.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---------- 混合效应模型 ----------
    # 母语者性别（CMU ARCTIC 说话人）
    GENDER = {"bdl": "M", "slt": "F", "clb": "F", "rms": "M",
              "jmk": "M", "awb": "M", "ksp": "F"}
    df["gender"] = df["gender"].fillna(df["speaker"].map(GENDER))

    print("\n=== 混合效应模型 (原始 F1, Hz) ===")
    m1 = mixed_model(df, "F1")
    with open(RESULTS / "mixed_model.txt", "w") as f:
        f.write(str(m1.summary()))
    pd.DataFrame({"coef": m1.params, "se": m1.bse, "p": m1.pvalues}) \
        .to_csv(RESULTS / "mixed_coefs.csv")
    print(m1.summary().tables[1].to_string()[:1200])

    summary = {
        "n_tokens": int(len(df)),
        "n_speakers": {k: int(v) for k, v in df.groupby("group")["speaker"].nunique().items()},
        "vsa_mean_L2": float(np.mean(l2)),
        "vsa_mean_Native": float(np.mean(na)),
        "vsa_diff": float(perm["observed_diff"]),
        "vsa_perm_p": float(perm["p_value"]),
        "vsa_cohens_d": float(d),
        "pairs": pair_summary,
    }
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {RESULTS}")


if __name__ == "__main__":
    main()
