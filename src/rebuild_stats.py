"""
rebuild_stats.py — 用全量数据(31 位说话人)重算"普通话 vs 母语者"的各项统计

输出到 results/ 下与旧脚本同名的文件, 使既有图件可直接复用。
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from normalize import add_bark, lobanov  # noqa: E402
from stats import cohens_d, permutation_test  # noqa: E402
from vowelspace import pairwise_distances, speaker_vsa  # noqa: E402

PAIRS = [("ɪ", "iː"), ("ɑː", "ʌ"), ("ʊ", "uː"), ("æ", "e")]
GENDER = {"bdl": "M", "slt": "F", "clb": "F", "rms": "M",
          "jmk": "M", "awb": "M", "ksp": "F"}


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
    raw = pd.read_csv(DATA / "formants_all.csv").dropna(subset=["F1", "F2"])
    sub = raw[((raw.group == "L2") & (raw.l1 == "Mandarin")) | (raw.group == "Native")]
    sub = sub.copy()
    sub["gender"] = sub["gender"].fillna(sub["speaker"].map(GENDER))
    print(f"子集: {len(sub)} token, {sub.speaker.nunique()} 位说话人")

    df = lobanov(add_bark(clean(sub)))

    # 元音空间面积
    vsa = speaker_vsa(df, f1="F1_bark_n", f2="F2_bark_n")
    meta = df.groupby("speaker")[["group", "gender"]].first()
    vsa = vsa.merge(meta, left_on="speaker", right_index=True)
    vsa.to_csv(RESULTS / "vsa_by_speaker.csv", index=False)

    l2 = vsa[vsa.group == "L2"]["vsa"].values
    na = vsa[vsa.group == "Native"]["vsa"].values
    perm = permutation_test(vsa["vsa"].values, vsa["group"].values, n_perm=10000)
    print(f"VSA: L2 {l2.mean():.2f} vs Native {na.mean():.2f} "
          f"(d={cohens_d(l2, na):.2f}, p={perm['p_value']:.3f})")

    # 元音对距离
    dist = pairwise_distances(df, PAIRS, f1="F1_bark_n", f2="F2_bark_n")
    dist = dist.merge(meta, left_on="speaker", right_index=True)
    dist.to_csv(RESULTS / "pair_distances.csv", index=False)

    rows = []
    for p in sorted(dist["pair"].unique()):
        a = dist[(dist.pair == p) & (dist.group == "L2")]["distance"].values
        b = dist[(dist.pair == p) & (dist.group == "Native")]["distance"].values
        rows.append({"pair": p, "L2_mean": float(np.mean(a)),
                     "Native_mean": float(np.mean(b)),
                     "diff": float(np.mean(a) - np.mean(b)),
                     "d": cohens_d(a, b)})
    pd.DataFrame(rows).to_csv(RESULTS / "pair_summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # 混合效应模型
    from stats import mixed_model
    m = mixed_model(df, "F1")
    with open(RESULTS / "mixed_model.txt", "w") as f:
        f.write(str(m.summary()))
    pd.DataFrame({"coef": m.params, "se": m.bse, "p": m.pvalues}) \
        .to_csv(RESULTS / "mixed_coefs.csv")

    # 时长(仅重读元音, 与共振峰同口径)
    dur = sub.dropna(subset=["dur"])
    piv = dur.groupby(["speaker", "vowel"])["dur"].median().unstack()
    piv["tense_ratio"] = piv["iː"] / piv["ɪ"]
    td = piv.reset_index()[["speaker", "tense_ratio"]].merge(
        meta.reset_index()[["speaker", "group"]], on="speaker")
    td.to_csv(RESULTS / "duration_tense_ratio.csv", index=False)
    a = td[td.group == "L2"]["tense_ratio"].values
    b = td[td.group == "Native"]["tense_ratio"].values
    print(f"时长比: L2 {a.mean():.2f} vs Native {b.mean():.2f} "
          f"(d={cohens_d(a, b):.2f}, MWU p={st.mannwhitneyu(a, b).pvalue:.4f})")

    # 语速
    pro = pd.read_csv(DATA / "prosody_all.csv")
    pro = pro.drop(columns=[c for c in ["group", "l1", "gender"] if c in pro.columns])
    pro = pro.merge(meta.reset_index()[["speaker", "group"]], on="speaker")
    sp = pro.groupby(["speaker", "group"])["rate"].median().reset_index()
    sp.to_csv(RESULTS / "prosody_rate.csv", index=False)
    a = sp[sp.group == "L2"]["rate"].values
    b = sp[sp.group == "Native"]["rate"].values
    print(f"语速: L2 {a.mean():.2f} vs Native {b.mean():.2f} "
          f"(d={cohens_d(a, b):.2f}, MWU p={st.mannwhitneyu(a, b).pvalue:.2e})")

    json.dump({"n_tokens": int(len(sub)),
               "n_speakers": int(sub.speaker.nunique())},
              open(RESULTS / "summary.json", "w"), indent=2)
    print("完成")


if __name__ == "__main__":
    main()
