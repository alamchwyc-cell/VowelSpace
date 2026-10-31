"""
prepare_l2.py — 从 parquet 中提取 L2-ARCTIC 普通话说话人的音频

L2-ARCTIC 在 HF 上以 parquet 形式发布(audio 列是 wav 字节流).
这里解码、统一为 16 kHz 单声道, 并按 <speaker>/<utt>.wav 落盘, 供 MFA 使用.

为什么统一 16 kHz:
  1) CMU ARCTIC 原生就是 16 kHz, 两组必须一致;
  2) 共振峰分析只看 5.5 kHz 以下, 16 kHz 的 Nyquist(8 kHz) 足够;
  3) 文件小, 对齐快.
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "l2_wav"
TARGET_SR = 16000
MANDARIN_L1 = 3          # 数据集中 l1 的编码: 0=Arabic 1=Hindi 2=Korean 3=Mandarin ...


def resample(x: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return x
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(sr, target)
    return resample_poly(x, target // g, sr // g)


def main():
    shards = sorted(DATA.glob("l2_*.parquet"))
    shards = [p for p in shards if p.name != "l2arctic.parquet"]
    print(f"读取 {len(shards)} 个分片")

    frames = []
    for p in shards:
        d = pd.read_parquet(p, columns=["speaker", "l1", "gender", "transcript",
                                        "file_id", "duration", "audio"])
        frames.append(d[d["l1"] == MANDARIN_L1])
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["speaker", "file_id"])
    print(f"普通话样本 {len(df)}, 说话人 {sorted(df.speaker.unique())}")

    OUT.mkdir(parents=True, exist_ok=True)
    meta = []
    for i, r in df.iterrows():
        spk = r["speaker"]
        utt = "arctic_" + str(r["file_id"])
        d = OUT / spk
        d.mkdir(exist_ok=True)
        dst = d / f"{utt}.wav"
        if not dst.exists():
            data, sr = sf.read(io.BytesIO(r["audio"]["bytes"]))
            if data.ndim > 1:
                data = data.mean(axis=1)
            data = resample(data, sr, TARGET_SR).astype(np.float32)
            sf.write(dst, data, TARGET_SR, subtype="PCM_16")
        meta.append({"speaker": spk, "group": "L2", "l1": "Mandarin",
                     "gender": "M" if r["gender"] == 1 else "F",
                     "utt": utt, "text": r["transcript"], "wav": str(dst)})
        if len(meta) % 250 == 0:
            print(f"  {len(meta)}/{len(df)}", flush=True)

    m = pd.DataFrame(meta)
    m.to_csv(DATA / "l2_meta.csv", index=False)
    print(f"完成: {len(m)} 个文件 -> {DATA/'l2_meta.csv'}")
    print(m.groupby(["speaker", "gender"]).size().to_string())


if __name__ == "__main__":
    main()
