"""
figstyle.py — 发表级图的统一样式

规范:
  字体  Times New Roman (西文/IPA) + SimSun (中文)
  字号  8 pt 轴标签, 7 pt 刻度/图例, 9 pt 面板标号
  尺寸  单栏 85 mm, 双栏 170 mm
  导出  PDF (矢量, 字体嵌入) + PNG 600 dpi + SVG
  刻度  朝内, 只留左/下边框, 无网格
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "figs"

FONT_EN = "Times New Roman"
FONT_ZH = "SimSun"

# 全项目统一配色
C_L2 = "#C0504D"
C_NATIVE = "#2E5A88"
C_NEUTRAL = "#8C8C8C"
C_ACCENT = "#7F9A3C"

SINGLE = 85 / 25.4      # 3.35 in
DOUBLE = 170 / 25.4     # 6.69 in


def setup(lang: str = "en"):
    """lang: 'en' 全英文 (投稿); 'zh' 中文标签 (结题报告)."""
    fam = FONT_EN if lang == "en" else FONT_ZH
    plt.rcParams.update({
        "font.family": fam,
        "font.serif": [FONT_EN],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "lines.linewidth": 1.0,
        "legend.frameon": False,
        "axes.unicode_minus": (lang == "en"),   # SimSun 缺 U+2212
        "pdf.fonttype": 42,     # TrueType 嵌入, 期刊可编辑
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def panel(ax, letter, lang="en"):
    ax.text(-0.16, 1.06, f"({letter})", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left")


def save(fig, name: str, sub: str, formats=("pdf", "png", "svg")):
    out = FIGS / sub
    out.mkdir(parents=True, exist_ok=True)
    for f in formats:
        fig.savefig(out / f"{name}.{f}")
    plt.close(fig)
    print("saved", out / name)
