"""
figures.py — 论文/报告用图

图1 元音空间对比: 两组在 (F1, F2) 平面上的凸包
图2 元音间距离: 关键元音对的组间对比 (带误差棒)
图3 混合效应模型系数森林图
图4 置换检验零分布
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "figs"
FIGS.mkdir(exist_ok=True)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402

style.setup()

COLORS = {"L2": "#C0504D", "Native": "#2E5A88"}
LABELS = {"L2": "中国英语学习者", "Native": "英语母语者"}


def plot_vowel_space(df: pd.DataFrame, out: Path, f1="F1_bark", f2="F2_bark"):
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    for grp in ["Native", "L2"]:
        g = df[df["group"] == grp]
        if g.empty:
            continue
        med = g.groupby("vowel")[[f1, f2]].median().dropna()
        pts = med.values
        if len(pts) >= 3:
            hull = ConvexHull(pts)
            poly = pts[np.append(hull.vertices, hull.vertices[0])]
            ax.fill(poly[:, 1], poly[:, 0], color=COLORS[grp], alpha=0.12)
            ax.plot(poly[:, 1], poly[:, 0], color=COLORS[grp], lw=1.2, alpha=0.7)
        ax.scatter(med[f2], med[f1], s=28, color=COLORS[grp],
                   label=f"{LABELS[grp]}（面积 {ConvexHull(pts).volume:.1f}）"
                   if len(pts) >= 3 else LABELS[grp], zorder=3)
        for v, row in med.iterrows():
            ax.annotate(v, (row[f2], row[f1]), fontsize=8,
                        xytext=(3, 3), textcoords="offset points",
                        fontfamily="DejaVu Sans")   # IPA 字形

    ax.invert_xaxis()          # 惯例: F2 从大到小 (前元音在左)
    ax.invert_yaxis()          # 惯例: F1 从大到小 (高元音在上)
    ax.set_xlabel("F2（归一化 Bark）")
    ax.set_ylabel("F1（归一化 Bark）")
    ax.set_title("元音空间对比", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)


def plot_pairwise(dist_df: pd.DataFrame, out: Path):
    """dist_df: speaker, pair, group"""
    pairs = sorted(dist_df["pair"].unique())
    fig, axes = plt.subplots(1, len(pairs), figsize=(2.2 * len(pairs), 3.2),
                             sharey=True)
    if len(pairs) == 1:
        axes = [axes]
    for ax, p in zip(axes, pairs):
        d = dist_df[dist_df["pair"] == p]
        data, labels, colors = [], [], []
        for grp in ["Native", "L2"]:
            v = d[d["group"] == grp]["distance"].dropna().values
            if len(v):
                data.append(v)
                labels.append(LABELS[grp])
                colors.append(COLORS[grp])
        bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                        medianprops={"color": "black", "lw": 1})
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.35)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=7, rotation=15)
        ax.set_title(p, fontsize=9, fontfamily="DejaVu Sans")
    axes[0].set_ylabel("元音间距离 (Bark)")
    fig.suptitle("关键元音对的声学距离", fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)


def plot_permutation(null_diffs, obs, out: Path):
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.hist(null_diffs, bins=50, color="#9AA5B1", alpha=0.8)
    ax.axvline(obs, color="#C0504D", lw=2, label=f"观测差值 {obs:.2f}")
    ax.set_xlabel("打乱分组后的组间差值")
    ax.set_ylabel("频数")
    ax.set_title("置换检验：零分布与观测值", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)


def plot_forest(coef_csv: Path, out: Path):
    """混合效应模型中 group×vowel 交互项的森林图."""
    import pandas as pd
    df = pd.read_csv(coef_csv, index_col=0)
    d = df[df.index.str.contains("group.*vowel", regex=True)].copy()
    if d.empty:
        print("没有交互项可画")
        return
    d["vowel"] = [x.split("[T.")[-1].rstrip("]") for x in d.index]
    d = d.sort_values("coef")

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    y = np.arange(len(d))
    err = 1.96 * d["se"].values
    ax.errorbar(d["coef"].values, y, xerr=err, fmt="o", color="#2E5A88",
                ecolor="#9AA5B1", capsize=3, ms=5)
    ax.axvline(0, color="#C0504D", lw=1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(d["vowel"], fontfamily="DejaVu Sans")
    ax.set_xlabel("母语者相对学习者的 F1 差异 (Hz)")
    ax.set_title("图 3  混合效应模型的组×元音交互项", fontsize=10)
    for yi, (c, p) in enumerate(zip(d["coef"].values, d["p"].values)):
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        if star:
            ax.text(c + (10 if c > 0 else -10), yi, star, va="center",
                    ha="left" if c > 0 else "right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)
