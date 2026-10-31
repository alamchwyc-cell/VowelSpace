"""
make_figs.py — 生成报告用图
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGS = ROOT / "figs"
RESULTS = ROOT / "results"
FIGS.mkdir(exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figures import plot_pairwise, plot_permutation, plot_vowel_space  # noqa: E402
from normalize import add_bark, lobanov, nearey  # noqa: E402
from stats import permutation_test  # noqa: E402
from vowelspace import speaker_vsa  # noqa: E402


def main():
    df = pd.read_csv(DATA / "formants.csv")

    # 与 analyze.py 相同的清洗
    keep = []
    for spk, g in df.groupby("speaker"):
        m = pd.Series(True, index=g.index)
        for f in ["F1", "F2"]:
            q1, q3 = g[f].quantile([0.25, 0.75])
            iqr = q3 - q1
            m &= g[f].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        keep.append(g[m])
    df = pd.concat(keep, ignore_index=True)

    df = add_bark(df)
    df = lobanov(df)

    plot_vowel_space(df, FIGS / "fig1_vowel_space.png",
                     f1="F1_bark_n", f2="F2_bark_n")

    dist = pd.read_csv(RESULTS / "pair_distances.csv")
    plot_pairwise(dist, FIGS / "fig2_pair_distances.png")

    vsa = pd.read_csv(RESULTS / "vsa_by_speaker.csv")
    res = permutation_test(vsa["vsa"].values, vsa["group"].values, n_perm=10000)
    rng = np.random.default_rng(42)
    vals, labs = vsa["vsa"].values, vsa["group"].values
    n_l2 = (labs == "L2").sum()
    null = []
    for _ in range(5000):
        idx = rng.permutation(len(vals))
        null.append(vals[idx[:n_l2]].mean() - vals[idx[n_l2:]].mean())
    plot_permutation(np.array(null), res["observed_diff"], FIGS / "fig4_permutation.png")

    print("图已生成")


if __name__ == "__main__":
    main()


def plot_wer(out: Path):
    """元音偏离度 vs ASR 词错率."""
    from scipy.stats import spearmanr
    p = RESULTS / "wer_vs_vowel.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    rho = spearmanr(df["mean_dist"], df["wer_mean"]).statistic

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for grp, c, lab in [("Native", "#2E5A88", "英语母语者"),
                        ("L2", "#C0504D", "中国英语学习者")]:
        d = df[df["group"] == grp]
        ax.scatter(d["mean_dist"], d["wer_mean"], s=55, color=c, label=lab,
                   edgecolors="white", linewidths=0.8, zorder=3)
        for _, r in d.iterrows():
            ax.annotate(r["speaker"], (r["mean_dist"], r["wer_mean"]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("平均元音偏离（马氏距离）")
    ax.set_ylabel("ASR 词错率")
    ax.set_title("图 5  元音偏离度与可懂度", fontsize=10)
    ax.text(0.03, 0.93, f"Spearman ρ = {rho:.3f}", transform=ax.transAxes,
            fontsize=9, color="#444")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)


def plot_duration(out: Path):
    p = RESULTS / "duration_tense_ratio.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for i, (grp, c, lab) in enumerate([("Native", "#2E5A88", "英语母语者"),
                                       ("L2", "#C0504D", "中国英语学习者")]):
        d = df[df["group"] == grp]
        x = np.full(len(d), i) + np.random.default_rng(0).normal(0, 0.04, len(d))
        ax.scatter(x, d["tense_ratio"], s=60, color=c, edgecolors="white",
                   linewidths=0.8, zorder=3, label=lab)
        ax.hlines(d["tense_ratio"].mean(), i - 0.18, i + 0.18, color=c, lw=2)
    ax.axhline(1.0, ls=":", color="gray", lw=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["英语母语者", "中国英语学习者"], fontsize=8)
    ax.set_ylabel("长元音与短元音的时长比")
    ax.set_title("图 6  松紧元音的时长区分", fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)


def plot_rate(out: Path):
    p = RESULTS / "prosody_rate.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for i, (grp, c) in enumerate([("Native", "#2E5A88"), ("L2", "#C0504D")]):
        d = df[df["group"] == grp]["rate"].values
        ax.scatter(np.full(len(d), i) + np.random.default_rng(1).normal(0, 0.04, len(d)),
                   d, s=60, color=c, edgecolors="white", linewidths=0.8, zorder=3)
        ax.hlines(d.mean(), i - 0.18, i + 0.18, color=c, lw=2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["英语母语者", "中国英语学习者"], fontsize=8)
    ax.set_ylabel("语速（音节/秒）")
    ax.set_title("图 7  朗读语速", fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)
