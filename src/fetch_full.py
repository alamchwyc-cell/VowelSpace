"""
fetch_full.py — 从官方 L2-ARCTIC v5.0 抓取全部说话人

与之前用的 HF parquet 镜像相比, 官方包有两个好处:
  1) 自带音素级 TextGrid 标注, 不必再跑 MFA;
  2) 24 位说话人齐全(镜像只有部分, 且母语标签有误).

策略: 逐个下载 -> 只留 250 句的 wav + textgrid -> 删 zip 与临时目录,
峰值磁盘占用约 0.5 GB.

用法:
    python3 src/fetch_full.py                # 全部 24 位
    python3 src/fetch_full.py --speakers TNI TXHC
"""
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "official"
BASE = "https://huggingface.co/datasets/chikingsley/l2-arctic-release-v5.0/resolve/main/{spk}.zip"

# 官方语料 24 位说话人的母语背景(Zhao et al., Interspeech 2018)
L1 = {
    "Arabic":    ["ABA", "MBMPS", "YBAA", "ZHAA"],
    "Hindi":     ["ASI", "RRBI", "SKA", "SVBI"],
    "Korean":    ["HJK", "HKK", "YDCK", "YKWK"],
    "Mandarin":  ["BWC", "LXC", "NCC", "TNI", "TXHC"],
    "Spanish":   ["EBVS", "ERMS", "NJS", "PNV"],
    "Vietnamese": ["HQTV", "THV", "TLV"],
}
GENDER = {
    "ABA": "M", "MBMPS": "M", "YBAA": "M", "ZHAA": "F",
    "ASI": "M", "RRBI": "M", "SKA": "F", "SVBI": "F",
    "HJK": "F", "HKK": "F", "YDCK": "F", "YKWK": "M",
    "BWC": "M", "LXC": "F", "NCC": "F", "TNI": "M", "TXHC": "M",
    "EBVS": "M", "ERMS": "M", "NJS": "F", "PNV": "F",
    "HQTV": "M", "THV": "F", "TLV": "F",
}
ALL = [s for v in L1.values() for s in v]


def get(speaker: str, n_utts: int = 250):
    dst = OUT / speaker
    if (dst / "wav").exists() and len(list((dst / "wav").glob("*.wav"))) >= n_utts:
        print(f"  {speaker} 已存在, 跳过")
        return

    zip_path = DATA / f"{speaker}.zip"
    if not zip_path.exists():
        print(f"  下载 {speaker}.zip ...", flush=True)
        subprocess.run(["curl", "-sL", "--retry", "3", "-o", str(zip_path),
                        BASE.format(spk=speaker)], check=True)
    print(f"  解压 {speaker} ({zip_path.stat().st_size/1e6:.0f} MB) ...", flush=True)

    with tempfile.TemporaryDirectory(dir=DATA) as td:
        subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", td], check=True)
        src = Path(td) / speaker
        (dst / "wav").mkdir(parents=True, exist_ok=True)
        (dst / "textgrid").mkdir(parents=True, exist_ok=True)
        (dst / "transcript").mkdir(parents=True, exist_ok=True)
        n = 0
        for i in range(1, n_utts + 1):
            utt = f"arctic_a{i:04d}"
            w = src / "wav" / f"{utt}.wav"
            t = src / "textgrid" / f"{utt}.TextGrid"
            r = src / "transcript" / f"{utt}.txt"
            if w.exists() and t.exists():
                shutil.copy(w, dst / "wav" / f"{utt}.wav")
                shutil.copy(t, dst / "textgrid" / f"{utt}.TextGrid")
                if r.exists():
                    shutil.copy(r, dst / "transcript" / f"{utt}.txt")
                n += 1
        print(f"    保留 {n} 句")
    zip_path.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speakers", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=250)
    a = ap.parse_args()

    targets = a.speakers or ALL
    print(f"目标 {len(targets)} 位说话人")
    for i, spk in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] {spk}")
        get(spk, a.n)

    import pandas as pd
    rows = []
    for l1, spks in L1.items():
        for s in spks:
            for w in sorted((OUT / s / "wav").glob("*.wav")):
                rows.append({"speaker": s, "l1": l1, "gender": GENDER.get(s, ""),
                             "utt": w.stem, "wav": str(w),
                             "textgrid": str(OUT / s / "textgrid" / f"{w.stem}.TextGrid")})
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "official_meta.csv", index=False)
    print(f"\n元数据: {len(df)} 条 -> data/official_meta.csv")
    print(df.groupby(["l1", "speaker"]).size().to_string())


if __name__ == "__main__":
    main()
