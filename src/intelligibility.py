"""
intelligibility.py — 用 ASR 词错率作为"可懂度"代理

思路:
  真人听辨实验成本高, 但语音识别器可以当作一个"挑剔的听者":
  如果 ASR 把某个词听错, 说明该发音在声学上偏离了标准英语的典型分布.

  这不是完美的可懂度测量 (ASR 有自己的偏置), 但它:
    - 完全可复现
    - 无需被试与伦理审批
    - 与真人听辨在文献中呈中等以上相关

用法:
    python3 src/intelligibility.py --per-speaker 50
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

MODEL_NAME = "facebook/wav2vec2-base-960h"


def norm_text(s: str) -> str:
    s = s.upper()
    s = re.sub(r"[^A-Z' ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def wer(ref: str, hyp: str) -> float:
    """词错率 = (替换 + 删除 + 插入) / 参考词数."""
    r, h = ref.split(), hyp.split()
    if not r:
        return float("nan")
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return float(d[len(r), len(h)] / len(r))


def transcribe(paths, batch_size=8):
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    import soundfile as sf

    proc = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).eval()

    outs = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i + batch_size]
        wavs = []
        for p in batch:
            x, sr = sf.read(p)
            if x.ndim > 1:
                x = x.mean(1)
            if sr != 16000:
                from scipy.signal import resample_poly
                from math import gcd
                g = gcd(sr, 16000)
                x = resample_poly(x, 16000 // g, sr // g)
            wavs.append(x)
        # transformers 5.x: Wav2Vec2Processor 返回 input_values 而非 input_ids,
        # 直接解包字典交给模型最稳妥
        inp = proc(wavs, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inp).logits
        ids = torch.argmax(logits, dim=-1)
        outs.extend(proc.batch_decode(ids))
        if i % 80 == 0:
            print(f"  {i}/{len(paths)}", flush=True)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-speaker", type=int, default=50)
    a = ap.parse_args()

    l2 = pd.read_csv(DATA / "l2_meta.csv")
    cmu = []
    for spk_dir in sorted((DATA / "cmu").glob("*")):
        if spk_dir.is_dir():
            for wav in sorted(spk_dir.glob("*.wav")):
                cmu.append({"speaker": spk_dir.name, "group": "Native",
                            "utt": wav.stem, "wav": str(wav)})
    ref_text = dict(zip(l2["utt"], l2["text"]))

    rows = []
    for spk, g in l2.groupby("speaker"):
        for _, r in g.head(a.per_speaker).iterrows():
            rows.append({"speaker": spk, "group": "L2", "utt": r["utt"],
                         "wav": r["wav"], "text": r["text"]})
    # 母语者也按同样数量抽样, 保证两组样本量可比
    cmu_df = pd.DataFrame([c for c in cmu if c["utt"] in ref_text])
    for _, c in cmu_df.groupby("speaker").head(a.per_speaker).iterrows():
        rows.append({"speaker": c["speaker"], "group": "Native", "utt": c["utt"],
                     "wav": c["wav"], "text": ref_text[c["utt"]]})

    df = pd.DataFrame(rows)
    print(f"待识别 {len(df)} 条")
    hyps = transcribe(df["wav"].tolist())

    df["hyp"] = hyps
    df["wer"] = [wer(norm_text(r), norm_text(h))
                 for r, h in zip(df["text"], df["hyp"])]
    df.to_csv(RESULTS / "asr_wer_tokens.csv", index=False)

    summ = df.groupby(["group", "speaker"]).agg(
        n=("wer", "size"), wer_mean=("wer", "mean"), wer_median=("wer", "median")
    ).reset_index()
    summ.to_csv(RESULTS / "asr_wer_by_speaker.csv", index=False)
    print("\n=== 每位说话人的 ASR 词错率 ===")
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n组均值:", summ.groupby("group")["wer_mean"].mean().round(3).to_dict())


if __name__ == "__main__":
    main()
