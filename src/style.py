"""
style.py — 统一绘图风格与中文字体注册

matplotlib 默认字体不含中文, 这里显式注册系统字体, 避免图里出现方块.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
]


def setup(font_size: int = 9) -> str:
    chosen = None
    for p in CANDIDATES:
        if Path(p).exists():
            try:
                font_manager.fontManager.addfont(p)
                chosen = font_manager.FontProperties(fname=p).get_name()
                break
            except Exception:
                continue
    if chosen is None:
        chosen = "DejaVu Sans"

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [chosen, "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": font_size,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    return chosen


if __name__ == "__main__":
    print("使用字体:", setup())
