"""
figs_pub.py — 生成发表级图（中英两套）

用法:
    python3 src/figs_pub.py            # 两套都生成
    python3 src/figs_pub.py --lang en
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
import figstyle as fs  # noqa: E402
from normalize import add_bark, lobanov  # noqa: E402

VOWELS = ["iː", "ɪ", "e", "æ", "ɑː", "ɔː", "ʊ", "uː", "ʌ", "ɜː"]

TXT = {
    "en": {
        "native": "English natives", "l2": "Mandarin learners",
        "f1": "F1 (normalized Bark)", "f2": "F2 (normalized Bark)",
        "vsa": "Vowel space", "pair": "Vowel-pair distance (Bark)",
        "coef": "Native − L2 difference in F1 (Hz)",
        "perm_x": "Group difference under permuted labels (Bark²)",
        "perm_y": "Count", "perm_t": "Exact permutation test",
        "wer_x": "Mean vowel deviation (Mahalanobis distance)",
        "wer_y": "ASR word error rate", "das": "DAS",
        "dur_y": "Duration ratio (long / short)",
        "rate_y": "Speech rate (syllables / s)",
        "note_n": "n = {}",
    },
    "zh": {
        "native": "英语母语者", "l2": "中国英语学习者",
        "f1": "F1（归一化 Bark）", "f2": "F2（归一化 Bark）",
        "vsa": "元音空间", "pair": "元音对距离（Bark）",
        "coef": "母语者 − 学习者 的 F1 差值（Hz）",
        "perm_x": "打乱分组后的组间差值（Bark²）",
        "perm_y": "频数", "perm_t": "精确置换检验",
        "wer_x": "平均元音偏离（马氏距离）",
        "wer_y": "ASR 词误率", "das": "元音偏离",
        "dur_y": "长元音／短元音时长比",
        "rate_y": "语速（音节／秒）",
        "note_n": "n = {}",
    },
}

def mun(s, lang):
    """英文图用真减号 U+2212, 中文图用 ASCII 连字符 (SimSun 缺 U+2212)."""
    return s.replace("-", "\u2212") if lang == "en" else s


IPA_ZH = {"iː": "IY", "ɪ": "IH", "e": "EH", "æ": "AE", "ɑː": "AA",
          "ɔː": "AO", "ʊ": "UH", "uː": "UW", "ʌ": "AH", "ɜː": "ER"}


def load_vowel_data():
    df = pd.read_csv(DATA / "formants.csv")
    keep = []
    for spk, g in df.groupby("speaker"):
        m = pd.Series(True, index=g.index)
        for f in ["F1", "F2"]:
            q1, q3 = g[f].quantile([0.25, 0.75])
            iqr = q3 - q1
            m &= g[f].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        keep.append(g[m])
    df = pd.concat(keep, ignore_index=True)
    df = lobanov(add_bark(df))
    return df


# ---------------------------------------------------------------- 图 1

def fig1_vowel_space(lang):
    t = TXT[lang]
    df = load_vowel_data()
    from scipy.spatial import ConvexHull

    fig, ax = plt.subplots(figsize=(fs.SINGLE, fs.SINGLE * 1.02))
    med = {}
    for grp, c, lab in [("Native", fs.C_NATIVE, t["native"]),
                        ("L2", fs.C_L2, t["l2"])]:
        g = df[df.group == grp]
        m = g.groupby("vowel")[["F1_bark_n", "F2_bark_n"]].median().dropna()
        med[grp] = m
        pts = m.values
        if len(pts) >= 3:
            hull = ConvexHull(pts)
            poly = pts[np.append(hull.vertices, hull.vertices[0])]
            ax.fill(poly[:, 1], poly[:, 0], color=c, alpha=0.10, lw=0)
            ax.plot(poly[:, 1], poly[:, 0], color=c, lw=0.9, alpha=0.75)
        n = g.speaker.nunique()
        ax.scatter(m["F2_bark_n"], m["F1_bark_n"], s=22, color=c,
                   edgecolors="white", linewidths=0.5, zorder=3,
                   label=f"{lab} ({t['note_n'].format(n)})")

    # 标签从空间中心沿径向向外偏移, 避免与散点和其他标签重叠
    allpts = np.vstack([med[g].values for g in ("Native", "L2")])
    center = allpts.mean(axis=0)
    for v in VOWELS:
        if v in med["Native"].index:
            p = med["Native"].loc[v].values
            d = p - center
            d = d / (np.linalg.norm(d) + 1e-9) * 11
            lbl = v if lang == "en" else IPA_ZH.get(v, v)
            ax.annotate(lbl, (p[1], p[0]), fontsize=7.5, color="#222222",
                        xytext=(d[1], d[0]), textcoords="offset points",
                        ha="center", va="center", clip_on=False,
                        fontstyle="normal")

    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel(t["f2"])
    ax.set_ylabel(t["f1"])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=2, handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout()
    fs.save(fig, "fig1_vowel_space", lang)


# ---------------------------------------------------------------- 图 2

def fig2_pair(lang):
    t = TXT[lang]
    d = pd.read_csv(RESULTS / "pair_distances.csv")
    s = pd.read_csv(RESULTS / "pair_summary.csv")
    s = s.reindex(s["d"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    pairs = list(s["pair"])
    x = np.arange(len(pairs))
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(fs.DOUBLE * 0.78, 2.5))
    for i, p in enumerate(pairs):
        for off, grp, c in [(-0.17, "Native", fs.C_NATIVE), (0.17, "L2", fs.C_L2)]:
            v = d[(d.pair == p) & (d.group == grp)]["distance"].values
            ax.scatter(i + off + rng.normal(0, 0.032, len(v)), v, s=15,
                       color=c, edgecolors="white", linewidths=0.4, zorder=3)
            n = len(v)
            from scipy.stats import t as tdist
            ci = tdist.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n) if n > 1 else 0
            ax.errorbar(i + off, v.mean(), yerr=ci, fmt="_", color=c,
                        capsize=2.5, elinewidth=0.9, markersize=10, zorder=4)

    def lab(pair):
        a, b = pair.split("-")
        return f"{a}–{b}" if lang == "en" else f"{IPA_ZH.get(a, a)}–{IPA_ZH.get(b, b)}"

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab(p)}\nd = {s.loc[i,'d']:.2f}" for i, p in enumerate(pairs)],
                       fontsize=7)
    ax.set_ylabel(t["pair"])
    ax.set_xlim(-0.6, len(pairs) - 0.4)
    ax.set_ylim(bottom=0)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=fs.C_NATIVE,
                              label=t["native"]),
                       Line2D([], [], marker="o", ls="", color=fs.C_L2,
                              label=t["l2"])],
              loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2,
              handletextpad=0.3, columnspacing=1.5)
    fig.tight_layout()
    fs.save(fig, "fig2_pair_distances", lang)


# ---------------------------------------------------------------- 图 3

def fig3_forest(lang):
    t = TXT[lang]
    d = pd.read_csv(RESULTS / "mixed_coefs.csv", index_col=0)
    d = d[d.index.str.contains(":C\\(vowel\\)")].copy()
    d["vowel"] = [i.split("[T.")[-1].rstrip("]") for i in d.index]
    d = d.sort_values("coef")

    fig, ax = plt.subplots(figsize=(fs.SINGLE, 2.6))
    y = np.arange(len(d))
    ax.errorbar(d["coef"], y, xerr=1.96 * d["se"], fmt="o", ms=3.5,
                color=fs.C_L2, ecolor="#9AA5B1", elinewidth=0.8, capsize=2)
    ax.axvline(0, color="#666666", lw=0.7, ls="--")
    ax.set_yticks(y)
    labels = [IPA_ZH[v] if lang == "zh" else v for v in d["vowel"]]
    ax.set_yticklabels(labels)
    ax.set_xlabel(t["coef"])
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0, x1 + 0.20 * (x1 - x0))
    for yi, p in zip(y, d["p"]):
        mark = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax.text(0.985, yi, mark, transform=ax.get_yaxis_transform(),
                fontsize=6.5, va="center", ha="right", color="#555555")
    fig.tight_layout()
    fs.save(fig, "fig3_forest", lang)


# ---------------------------------------------------------------- 图 4

def fig4_perm(lang):
    t = TXT[lang]
    vsa = pd.read_csv(RESULTS / "vsa_by_speaker.csv")
    vals = vsa["vsa"].values
    labs = vsa["group"].values
    obs = vals[labs == "L2"].mean() - vals[labs == "Native"].mean()

    n_l2 = int((labs == "L2").sum())
    null = []
    for comb in itertools.combinations(range(len(vals)), n_l2):
        rest = [i for i in range(len(vals)) if i not in comb]
        null.append(vals[list(comb)].mean() - vals[rest].mean())
    null = np.array(null)
    p_exact = float((np.abs(null) >= abs(obs)).mean())

    fig, ax = plt.subplots(figsize=(fs.SINGLE, 2.5))
    ax.hist(null, bins=18, color="#B8C2CC", edgecolor="white", linewidth=0.5)
    ax.axvline(obs, color=fs.C_L2, lw=1.2, zorder=4)
    ax.text(0.03, 0.95, mun(f"observed = {obs:.2f}\nexact p = {p_exact:.3f}", lang),
            transform=ax.transAxes, fontsize=6.5, color=fs.C_L2, va="top")
    ax.set_xlabel(t["perm_x"])
    ax.set_ylabel(t["perm_y"])
    ax.set_title(t["perm_t"])
    fig.tight_layout()
    fs.save(fig, "fig4_permutation", lang)


# ---------------------------------------------------------------- 图 5

def fig5_wer(lang):
    t = TXT[lang]
    from scipy.stats import spearmanr
    d = pd.read_csv(RESULTS / "wer_vs_vowel.csv")
    rho = spearmanr(d["mean_dist"], d["wer_mean"]).statistic

    fig, ax = plt.subplots(figsize=(fs.SINGLE, 2.8))
    for grp, c, lab in [("Native", fs.C_NATIVE, t["native"]),
                        ("L2", fs.C_L2, t["l2"])]:
        g = d[d.group == grp]
        ax.scatter(g["mean_dist"], g["wer_mean"], s=26, color=c,
                   edgecolors="white", linewidths=0.5, zorder=3, label=lab)

    # 只标离群点, 避免重叠
    for _, r in d.iterrows():
        if r["speaker"] in ("BWC", "LXC", "NCC", "ksp", "awb"):
            ax.annotate(r["speaker"], (r["mean_dist"], r["wer_mean"]),
                        fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(t["wer_x"])
    ax.set_ylabel(t["wer_y"])
    ax.text(0.03, 0.94, f"Spearman ρ = {rho:.3f}", transform=ax.transAxes,
            fontsize=7, va="top")
    ax.margins(x=0.06, y=0.16)
    ax.legend(loc="lower right", handletextpad=0.4, borderpad=0.2)
    fig.tight_layout()
    fs.save(fig, "fig5_wer", lang)


# ---------------------------------------------------------------- 图 6 / 7

def _dotplot(df, col, ylab, title, name, lang, dval, pval):
    fig, ax = plt.subplots(figsize=(fs.SINGLE * 0.88, 2.4))
    rng = np.random.default_rng(1)
    for i, (grp, c, lab) in enumerate([("Native", fs.C_NATIVE, TXT[lang]["native"]),
                                       ("L2", fs.C_L2, TXT[lang]["l2"])]):
        v = df[df.group == grp][col].values
        ax.scatter(np.full(len(v), i) + rng.normal(0, 0.055, len(v)), v,
                   s=26, color=c, edgecolors="white", linewidths=0.5, zorder=3)
        from scipy.stats import t as tdist
        ci = tdist.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        ax.errorbar(i, v.mean(), yerr=ci, fmt="none", color=c,
                    capsize=3, elinewidth=0.9, zorder=4)
        ax.plot([i - 0.10, i + 0.10], [v.mean()] * 2, color=c, lw=1.6, zorder=5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([TXT[lang]["native"], TXT[lang]["l2"]], fontsize=7.5)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.text(0.97, 0.95, mun(f"d = {dval:.2f}, p {pval}", lang), transform=ax.transAxes,
            va="top", ha="right", fontsize=6.5, color="#555555")
    fig.tight_layout()
    fs.save(fig, name, lang)


def fig6_duration(lang):
    d = pd.read_csv(RESULTS / "duration_tense_ratio.csv")
    _dotplot(d, "tense_ratio", TXT[lang]["dur_y"],
             "Tense–lax duration contrast" if lang == "en" else "松紧元音的时长区分",
             "fig6_duration", lang, -2.94, "= 0.008")


def fig7_rate(lang):
    d = pd.read_csv(RESULTS / "prosody_rate.csv")
    _dotplot(d, "rate", TXT[lang]["rate_y"],
             "Speech rate" if lang == "en" else "朗读语速",
             "fig7_rate", lang, -2.95, "< 0.001")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="both")
    a = ap.parse_args()
    langs = ["en", "zh"] if a.lang == "both" else [a.lang]
    for lang in langs:
        fs.setup(lang)
        fig1_vowel_space(lang)
        fig2_pair(lang)
        fig3_forest(lang)
        fig4_perm(lang)
        fig5_wer(lang)
        fig6_duration(lang)
        fig7_rate(lang)


if __name__ == "__main__":
    main()
