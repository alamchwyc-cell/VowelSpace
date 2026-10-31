"""
extract_all.py — 从官方语料 + 母语者对齐结果提取全部测量

数据源:
  L2     : data/official/<spk>/wav/*.wav + textgrid/*.TextGrid  (官方标注)
  Native : data/cmu/<spk>/*.wav + data/aligned/<spk>/*.TextGrid  (MFA 对齐)

输出(均含 stress 列, 便于统一口径):
  data/formants_all.csv   元音 token 的 F1/F2/时长
  data/prosody_all.csv    每句的 F0 范围与语速

用法:
    python3 src/extract_all.py
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(Path(__file__).resolve().parent))

VOWELS = {"IY": "iː", "IH": "ɪ", "EH": "e", "AE": "æ", "AA": "ɑː",
          "AO": "ɔː", "UH": "ʊ", "UW": "uː", "AH": "ʌ", "ER": "ɜː"}


def _stress(phone: str):
    import re
    m = re.search(r"(\d)$", phone)
    return int(m.group(1)) if m else None


def _process(task):
    spk, group, l1, gender, utt, wav, tg = task
    import parselmouth
    from align import read_phone_tiers
    from formants import measure_formants

    try:
        segs = read_phone_tiers(Path(tg))
        snd = parselmouth.Sound(str(wav))
        if abs(snd.sampling_frequency - 16000) > 1:
            snd = snd.resample(16000)
    except Exception:
        return [], []

    base = {"speaker": spk, "group": group, "l1": l1, "gender": gender, "utt": utt}
    rows = []
    n_vowels = 0
    for phone, t0, t1 in segs:
        raw = phone.upper().rstrip("012")
        if raw not in VOWELS:
            continue
        n_vowels += 1
        st = _stress(phone)
        dur = t1 - t0
        f1 = f2 = float("nan")
        if st in (1, 2) and dur >= 0.04:
            a, b = t0 + dur * 0.3, t0 + dur * 0.7
            f1, f2, _ = measure_formants(snd, a, b)
        rows.append({**base, "vowel": VOWELS[raw], "phone": raw, "stress": st,
                     "dur": float(dur), "F1": f1, "F2": f2})

    pro = []
    if segs and n_vowels >= 2:
        t_start, t_end = segs[0][1], segs[-1][2]
        span = t_end - t_start
        if span > 0.5:
            pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
            v = pitch.selected_array["frequency"]
            v = v[v > 0]
            if len(v) >= 10:
                p5, p50, p95 = np.percentile(v, [5, 50, 95])
                pro.append({**base, "n_vowels": n_vowels, "span": float(span),
                            "rate": float(n_vowels / span),
                            "f0_median": float(p50),
                            "f0_range_st": float(12 * np.log2(p95 / p5))})
    return rows, pro


def build_meta():
    rows = []

    off = pd.read_csv(DATA / "official_meta.csv")
    for _, r in off.iterrows():
        rows.append((r["speaker"], "L2", r["l1"], r["gender"], r["utt"],
                     r["wav"], r["textgrid"]))

    GENDER = {"bdl": "M", "slt": "F", "clb": "F", "rms": "M",
              "jmk": "M", "awb": "M", "ksp": "F"}
    for spk_dir in sorted((DATA / "cmu").glob("*")):
        if not spk_dir.is_dir():
            continue
        spk = spk_dir.name
        for wav in sorted(spk_dir.glob("*.wav")):
            tg = DATA / "aligned" / spk / f"{wav.stem}.TextGrid"
            if tg.exists():
                rows.append((spk, "Native", "English", GENDER.get(spk, ""),
                             wav.stem, str(wav), str(tg)))
    return rows


def main():
    tasks = build_meta()
    print(f"共 {len(tasks)} 条; 说话人 {len({t[0] for t in tasks})}", flush=True)

    F, P = [], []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, (a, b) in enumerate(ex.map(_process, tasks, chunksize=16)):
            F += a
            P += b
            if i % 800 == 0:
                print(f"  {i}/{len(tasks)}", flush=True)

    df = pd.DataFrame(F)
    df.to_csv(DATA / "formants_all.csv", index=False)
    pd.DataFrame(P).to_csv(DATA / "prosody_all.csv", index=False)
    print(f"元音 token {len(df)}; 句子 {len(P)}")
    print(df.groupby(["group", "speaker"]).size().to_string())


if __name__ == "__main__":
    main()
