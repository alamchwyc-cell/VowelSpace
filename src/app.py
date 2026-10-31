"""
app.py — 元音发音诊断（Streamlit）

启动:
    streamlit run src/app.py
"""
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402
from diagnose import diagnose  # noqa: E402
from reference import F1, F2  # noqa: E402

style.setup()

REF = json.loads((ROOT / "models/vowel_reference.json").read_text(encoding="utf-8"))

# MFA 是单进程重度计算: 用全局锁把并发请求串行化, 避免互相干扰
_ANALYSIS_LOCK = threading.Lock()

st.set_page_config(page_title="英语元音发音诊断", layout="wide")
st.title("英语元音发音诊断")
st.caption("读一段英文，看看你的元音和英语母语者差在哪里。")

SAMPLE_TEXT = ("Author of the danger trail, Philip Steels, etc. "
               "The birch canoe slid on the smooth planks. "
               "Glue the sheet to the dark blue background. "
               "It's easy to tell the depth of a well.")


def plot_diagnosis(points: dict, problems=None, key="v1"):
    fig, ax = plt.subplots(figsize=(5.6, 5.0))

    # 母语者参照: 每个元音画 1σ/2σ 椭圆
    for v, r in REF["vowels"].items():
        mu = np.array(r["mu"])
        cov = np.array(r["cov"])
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        for k, alpha in [(2.0, 0.10)]:
            w, h = 2 * k * np.sqrt(vals)
            e = matplotlib.patches.Ellipse(mu, w, h, angle=ang,
                                           facecolor="#2E5A88", alpha=alpha,
                                           edgecolor="#2E5A88", lw=0.8)
            ax.add_patch(e)
        ax.plot(mu[0], mu[1], "o", color="#2E5A88", ms=4, alpha=0.8)
        ax.annotate(v, mu, fontsize=8, xytext=(4, 4),
                    textcoords="offset points", color="#2E5A88",
                    fontfamily="DejaVu Sans")

    # 用户
    for v, xy in points.items():
        bad = problems and v in [p["vowel"] for p in problems]
        ax.plot(xy[0], xy[1], "o", color="#C0504D" if bad else "#E39B94",
                ms=7, mec="white", mew=0.8, zorder=5)
        ax.annotate(v, xy, fontsize=8, xytext=(5, -10),
                    textcoords="offset points", color="#C0504D",
                    fontfamily="DejaVu Sans", weight="bold")

    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel("F2（归一化）")
    ax.set_ylabel("F1（归一化）")
    ax.set_title("你的元音（红）vs 母语者分布（蓝）", fontsize=10)
    fig.tight_layout()
    return fig


with st.sidebar:
    st.header("怎么用")
    st.markdown(
        "1. 朗读一段英文（至少 30 句）\n"
        "2. 上传录音（wav / mp3 / m4a 均可）\n"
        "3. 粘贴你读的原文\n"
        "4. 点「开始诊断」\n\n"
        "录音要求：安静环境、正常语速、不要刻意放慢。"
    )
    st.divider()
    st.caption("参照：7 位英语母语者，17,281 个元音样本")

col1, col2 = st.columns([1, 1])
with col1:
    up = st.file_uploader("上传录音", type=["wav", "mp3", "m4a", "aac", "flac", "ogg", "opa", "opus", "wma", "amr"])
with col2:
    st.text_area("朗读示例文本（可选）", SAMPLE_TEXT, height=100, disabled=True)

text = st.text_area("你朗读的原文", height=140, placeholder="把录音对应的英文原文粘贴到这里……")

if st.button("开始诊断", type="primary", disabled=(up is None or not text.strip())):
    suffix = Path(up.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(up.getbuffer())
        wav_path = f.name

    if not _ANALYSIS_LOCK.acquire(blocking=False):
        st.info("当前有另一位用户正在分析，已进入队列，请稍候……")
        _ANALYSIS_LOCK.acquire()
    try:
        with st.spinner("对齐 + 测量中，大约 20–60 秒……"):
            res = diagnose(wav_path, text.strip())
    finally:
        _ANALYSIS_LOCK.release()
        try:
            os.unlink(wav_path)          # 隐私: 处理完立即删除上传文件
        except OSError:
            pass

    if not res["ok"]:
        st.error(res["reason"])
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("总分", f"{res['total_score']:.0f} / 100")
        c2.metric("元音 token", res["n_tokens"])
        c3.metric("覆盖元音", f"{res['n_vowels_found']} / 10")

        left, right = st.columns([1, 1])
        with left:
            st.pyplot(plot_diagnosis(res["vowel_points"], res["problems"]))
        with right:
            if res["problems"]:
                st.subheader("问题元音")
                for p in res["problems"]:
                    st.markdown(f"**/{p['vowel']}/** — 偏离 {p['distance']:.2f}σ，"
                                f"得分 {p['score']:.0f}")
                    if p.get("direction"):
                        st.caption(f"方向：{p['direction']}")
                    st.caption(p["advice"])
            else:
                st.success("没有发现明显偏离的元音。")

        st.subheader("各元音得分")
        df = pd.DataFrame(res["per_vowel"])[["vowel", "distance", "score"]]
        df.columns = ["元音", "偏离（σ）", "得分"]
        st.dataframe(df, hide_index=True)

st.divider()
st.caption("分数含义：与母语者分布的偏离程度，100 = 落在母语者分布中心，"
           "0 = 远离母语者分布。评分基于马氏距离与 χ² 分布，公式固定，未做人工调参。")
st.caption("隐私：上传的录音仅用于本次分析，处理完成后立即从服务器删除，不保存任何音频与结果。")
