"""
extract_more.py — 在已有对齐结果上补充提取三类测量

不需要重新对齐: MFA 的 TextGrid 里已经有每个音素的起止时间,
只是之前我们只取了元音的中点共振峰.

新增:
  1) 元音时长      —— 松紧元音对立的一半信息
  2) 擦音频谱矩    —— /θ/ /ð/ 的频谱重心/偏度, 用于检测"读成 s/z"
  3) 韵律          —— 基频范围(半音)、语速(音节/秒)

输出:
  data/durations.csv   每个元音 token 的时长
  data/fricatives.csv  每个 /θ/ /ð/ token 的频谱矩
  data/prosody.csv     每句话的 F0 范围与语速
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from align import collect_textgrids, read_phone_tiers  # noqa: E402

FRICATIVES = {"TH", "DH", "S", "Z", "SH", "ZH", "F", "V"}
VOWEL_SET = {"IY", "IH", "EH", "AE", "AA", "AO", "UH", "UW", "AH", "ER"}
VOWEL_MAP = {"IY": "iː", "IH": "ɪ", "EH": "e", "AE": "æ", "AA": "ɑː",
             "AO": "ɔː", "UH": "ʊ", "UW": "uː", "AH": "ʌ", "ER": "ɜː"}


def spectral_moments(snd, t0, t1):
    """摩擦段的频谱矩 (重心/标准差/偏度/峰度), 只用 1-8 kHz.

    用 numpy 的 rFFT 直接算, 避免依赖 Praat 的频谱对象 API.
    """
    dur = t1 - t0
    if dur < 0.03:
        return None
    # 取中间 40%, 避开与相邻元音/塞音的过渡
    a, b = t0 + dur * 0.3, t0 + dur * 0.7
    seg = snd.extract_part(from_time=a, to_time=b, preserve_times=False)
    x = np.asarray(seg.values[0], dtype=float)
    if len(x) < 64:
        return None
    sr = seg.sampling_frequency
    w = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * w))
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    p = spec ** 2
    m = (freqs >= 1000) & (freqs <= 8000)
    f, p = freqs[m], p[m]
    if p.sum() <= 0:
        return None
    p = p / p.sum()
    centroid = float((f * p).sum())
    sd = float(np.sqrt(((f - centroid) ** 2 * p).sum()))
    if sd <= 0:
        return None
    skew = float((((f - centroid) ** 3) * p).sum() / sd ** 3)
    kurt = float((((f - centroid) ** 4) * p).sum() / sd ** 4)
    return centroid, sd, skew, kurt


def f0_stats(snd):
    """基频范围: 用 5%-95% 分位, 换算成半音."""
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
    vals = pitch.selected_array["frequency"]
    vals = vals[vals > 0]
    if len(vals) < 10:
        return None
    p5, p50, p95 = np.percentile(vals, [5, 50, 95])
    return float(p50), float(12 * np.log2(p95 / p5)), float(p95 - p5)


def process(task):
    (spk, utt), tg, info = task
    import parselmouth

    try:
        segs = read_phone_tiers(tg)
        snd = parselmouth.Sound(info["wav"])
        if abs(snd.sampling_frequency - 16000) > 1:
            snd = snd.resample(16000)
    except Exception:
        return [], [], []

    base = {"speaker": spk, "group": info["group"], "utt": utt,
            "gender": info["gender"]}
    dur_rows, fric_rows = [], []

    n_vowels = 0
    for phone, t0, t1 in segs:
        p = phone.upper().rstrip("012")
        if p in VOWEL_SET:
            n_vowels += 1
            dur_rows.append({**base, "vowel": VOWEL_MAP[p], "phone": p,
                             "dur": float(t1 - t0)})
        elif p in FRICATIVES:
            m = spectral_moments(snd, t0, t1)
            if m:
                fric_rows.append({**base, "phone": p, "t0": t0, "t1": t1,
                                  "centroid": m[0], "sd": m[1],
                                  "skew": m[2], "kurt": m[3]})

    # 韵律: 以语音段(首音素起点到末音素终点)为分母
    pro_rows = []
    if segs and n_vowels >= 2:
        t_start, t_end = segs[0][1], segs[-1][2]
        span = t_end - t_start
        if span > 0.5:
            f = f0_stats(snd)
            if f:
                pro_rows.append({**base, "n_vowels": n_vowels,
                                 "span": float(span),
                                 "rate": float(n_vowels / span),
                                 "f0_median": f[0], "f0_range_st": f[1],
                                 "f0_range_hz": f[2]})
    return dur_rows, fric_rows, pro_rows


def main():
    tgs = collect_textgrids(DATA / "aligned")
    meta = {}
    d = pd.read_csv(DATA / "l2_meta.csv")
    for _, r in d.iterrows():
        meta[(r["speaker"], r["utt"])] = {"group": "L2", "gender": r["gender"],
                                          "wav": r["wav"]}
    for spk_dir in sorted((DATA / "cmu").glob("*")):
        if spk_dir.is_dir():
            for wav in spk_dir.glob("*.wav"):
                meta[(spk_dir.name, wav.stem)] = {"group": "Native", "gender": "",
                                                  "wav": str(wav)}

    tasks = [(k, tg, meta[k]) for k, tg in tgs.items() if k in meta]
    print(f"处理 {len(tasks)} 个文件", flush=True)

    D, F, P = [], [], []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, (a, b, c) in enumerate(ex.map(process, tasks, chunksize=16)):
            D += a; F += b; P += c
            if i % 400 == 0:
                print(f"  {i}/{len(tasks)}", flush=True)

    pd.DataFrame(D).to_csv(DATA / "durations.csv", index=False)
    pd.DataFrame(F).to_csv(DATA / "fricatives.csv", index=False)
    pd.DataFrame(P).to_csv(DATA / "prosody.csv", index=False)
    print(f"元音 token {len(D)}, 摩擦音 token {len(F)}, 句子 {len(P)}")


if __name__ == "__main__":
    main()
