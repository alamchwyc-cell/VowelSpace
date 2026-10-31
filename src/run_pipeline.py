"""
run_pipeline.py — 主流程：语料准备 → 强制对齐 → 共振峰提取

步骤:
  1) 把 L2 与母语者录音按 MFA 要求的目录结构组织 (软链接, 不复制)
  2) 调用 MFA 做音素级对齐
  3) 解析 TextGrid, 在每个元音的中段测量 F1/F2
  4) 输出 data/formants.csv

用法:
  python3 src/run_pipeline.py --build-corpus
  python3 src/run_pipeline.py --align
  python3 src/run_pipeline.py --extract
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 目标单元音 (ARPAbet) 与便于阅读的标签
VOWELS = {
    "IY": "iː", "IH": "ɪ", "EH": "e", "AE": "æ",
    "AA": "ɑː", "AO": "ɔː", "UH": "ʊ", "UW": "uː",
    "AH": "ʌ", "ER": "ɜː",
}
# 双元音与儿化音暂不纳入(其共振峰在滑动, 中点不代表目标)
EXCLUDE = {"EY", "AY", "OY", "AW", "OW", "AX", "AXR", "IX"}


def build_corpus():
    from align import prepare_corpus

    l2 = pd.read_csv(DATA / "l2_meta.csv")
    records = []
    for _, r in l2.iterrows():
        records.append({"speaker": r["speaker"], "utt_id": r["utt"],
                        "wav_path": r["wav"], "text": r["text"]})

    # 母语者: 用同一 utt 的文本
    text_map = dict(zip(l2["utt"], l2["text"]))
    cmu_root = DATA / "cmu"
    n_cmu = 0
    for spk_dir in sorted(cmu_root.glob("*")):
        if not spk_dir.is_dir():
            continue
        for wav in sorted(spk_dir.glob("*.wav")):
            utt = wav.stem
            if utt not in text_map:
                continue
            records.append({"speaker": spk_dir.name, "utt_id": utt,
                            "wav_path": str(wav), "text": text_map[utt]})
            n_cmu += 1

    print(f"L2 {len(l2)} 条, 母语者 {n_cmu} 条, 合计 {len(records)}")
    corpus = DATA / "corpus"
    n = prepare_corpus(records, corpus)
    print(f"语料目录 {corpus} (新增链接 {n})")


def do_align(dictionary="english_us_arpa", acoustic="english_us_arpa", jobs=4):
    from align import run_mfa
    run_mfa(DATA / "corpus", dictionary, acoustic, DATA / "aligned", jobs=jobs)


def _extract_one(task):
    """模块级函数: 供多进程调用 (局部函数无法 pickle)."""
    from align import read_phone_tiers
    from formants import load_sound, vowel_measurements

    (spk, utt), tg, info = task
    if info is None:
        return []
    try:
        segs = read_phone_tiers(tg)
        snd = load_sound(info["wav"], target_sr=16000)
        rows = vowel_measurements(snd, segs, set(VOWELS), VOWELS)
    except Exception:
        return []
    return [{"speaker": spk, "group": info["group"], "l1": info["l1"],
             "gender": info["gender"], "utt": utt,
             "vowel": r["vowel"], "phone": r["phone"],
             "dur": r["dur"], "F1": r["F1"], "F2": r["F2"],
             "n_frames": r["n_frames"]} for r in rows]


def do_extract(max_workers: int = 4):
    from concurrent.futures import ProcessPoolExecutor
    from align import collect_textgrids

    tgs = collect_textgrids(DATA / "aligned")
    print(f"TextGrid {len(tgs)} 个")

    meta = {}
    d = pd.read_csv(DATA / "l2_meta.csv")
    for _, r in d.iterrows():
        meta[(r["speaker"], r["utt"])] = {"group": "L2", "l1": r["l1"],
                                          "gender": r["gender"], "wav": r["wav"]}
    for spk_dir in sorted((DATA / "cmu").glob("*")):
        if spk_dir.is_dir():
            for wav in spk_dir.glob("*.wav"):
                meta[(spk_dir.name, wav.stem)] = {"group": "Native", "l1": "English",
                                                  "gender": "", "wav": str(wav)}

    tasks = [(k, tg, meta.get(k)) for k, tg in tgs.items()]

    rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for i, chunk in enumerate(ex.map(_extract_one, tasks, chunksize=16)):
            rows.extend(chunk)
            if i % 500 == 0:
                print(f"  {i}/{len(tasks)}", flush=True)

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["F1", "F2"])
    df.to_csv(DATA / "formants.csv", index=False)
    print(f"共振峰测量 {len(df)} 条 -> {DATA/'formants.csv'}")
    print(df.groupby(["group", "speaker"]).size().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-corpus", action="store_true")
    ap.add_argument("--align", action="store_true")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    if a.build_corpus:
        build_corpus()
    if a.align:
        do_align(jobs=a.jobs)
    if a.extract:
        do_extract(max_workers=a.jobs)
