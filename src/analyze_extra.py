"""
analyze_extra.py — 时长 / 擦音频谱 / 韵律 三组分析

每组都是"说话人层面的中位数 + 组间比较 + 效应量"，
与元音空间分析保持同一套统计口径。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stats import cohens_d, permutation_test  # noqa: E402

GENDER = {"bdl": "M", "slt": "F", "clb": "F", "rms": "M",
          "jmk": "M", "awb": "M", "ksp": "F"}


def summarize(df, value, by, label):
    """每个说话人的中位数 -> 组间比较."""
    spk = df.groupby(["speaker", "group"])[value].median().reset_index()
    L2 = spk[spk.group == "L2"][value].values
    NA = spk[spk.group == "Native"][value].values
    perm = permutation_test(spk[value].values, spk["group"].values, n_perm=10000)
    return {
        "指标": label,
        "L2_mean": float(np.mean(L2)),
        "Native_mean": float(np.mean(NA)),
        "diff": float(perm["observed_diff"]),
        "p": float(perm["p_value"]),
        "d": cohens_d(L2, NA),
    }, spk


def main():
    out = {}

    # ---------------- 1. 元音时长 ----------------
    dur = pd.read_csv(DATA / "durations.csv")
    dur["gender"] = dur["gender"].fillna(dur["speaker"].map(GENDER))

    # 松紧对立: /iː/ 与 /ɪ/ 的时长比
    piv = dur.pivot_table(index=["speaker", "group"], columns="vowel",
                          values="dur", aggfunc="median")
    piv["tense_ratio"] = piv["iː"] / piv["ɪ"]
    piv = piv.reset_index()
    piv.to_csv(RESULTS / "duration_tense_ratio.csv", index=False)

    r1, spk_dur = summarize(piv, "tense_ratio", None, "/iː/ : /ɪ/ 时长比")
    r2, _ = summarize(dur, "dur", None, "元音平均时长 (s)")
    print("=== 时长 ===")
    print(piv[["speaker", "group", "tense_ratio"]].sort_values(["group", "tense_ratio"])
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"/iː/:/ɪ/ 时长比  L2 {r1['L2_mean']:.2f} vs 母语 {r1['Native_mean']:.2f}"
          f"  d={r1['d']:.2f}  p={r1['p']:.4f}")
    out["duration"] = [r1, r2]

    # ---------------- 2. 擦音频谱矩 ----------------
    fr = pd.read_csv(DATA / "fricatives.csv")
    fr["gender"] = fr["gender"].fillna(fr["speaker"].map(GENDER))
    rows = []
    for ph in ["TH", "DH"]:
        d = fr[fr.phone == ph]
        for metric in ["centroid", "sd", "skew", "kurt"]:
            r, _ = summarize(d, metric, None, f"{ph} {metric}")
            rows.append(r)
    fr_res = pd.DataFrame(rows)
    fr_res.to_csv(RESULTS / "fricative_summary.csv", index=False)
    print("\n=== 擦音 (/θ/ /ð/) 频谱矩 ===")
    print(fr_res.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    out["fricative"] = rows

    spk_fr = fr.groupby(["speaker", "group", "phone"])["centroid"].median().unstack()
    spk_fr.to_csv(RESULTS / "fricative_centroid_by_speaker.csv")

    # ---------------- 3. 韵律 ----------------
    pr = pd.read_csv(DATA / "prosody.csv")
    pr["gender"] = pr["gender"].fillna(pr["speaker"].map(GENDER))
    rows = []
    for metric, label in [("f0_range_st", "基频范围 (半音)"),
                          ("rate", "语速 (音节/秒)"),
                          ("f0_median", "基频中位数 (Hz)")]:
        r, spk = summarize(pr, metric, None, label)
        rows.append(r)
        spk.to_csv(RESULTS / f"prosody_{metric}.csv", index=False)
    print("\n=== 韵律 ===")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    out["prosody"] = rows

    (RESULTS / "summary_extra.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果 -> {RESULTS}")


if __name__ == "__main__":
    main()
