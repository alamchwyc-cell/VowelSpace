"""
fetch_cmu.py — 下载 CMU ARCTIC 母语者录音

L2-ARCTIC 的说话人朗读的正是 CMU ARCTIC 的同一批提示句(arctic_a0001 ...),
所以只要拿到同一批 utt_id 的母语者录音, 就得到了"同一文本、不同母语背景"的对照.

用法:
    python3 src/fetch_cmu.py --utts arctic_a0001 arctic_a0002 ...
    python3 src/fetch_cmu.py --from-list data/l2_subset.csv
"""
import argparse
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "cmu"

# CMU ARCTIC 的母语者说话人 (4 女 3 男, 这里取 4 个代表性说话人)
NATIVE_SPEAKERS = ["bdl", "slt", "clb", "rms"]
BASE = "http://festvox.org/cmu_arctic/cmu_arctic/cmu_us_{spk}_arctic/wav/{utt}.wav"


def download(utts, speakers=NATIVE_SPEAKERS, force=False):
    ok, skip, fail = 0, 0, 0
    for spk in speakers:
        d = OUT / spk
        d.mkdir(parents=True, exist_ok=True)
        for utt in utts:
            dst = d / f"{utt}.wav"
            if dst.exists() and not force:
                skip += 1
                continue
            url = BASE.format(spk=spk, utt=utt)
            try:
                urlretrieve(url, dst)
                ok += 1
            except Exception as e:
                fail += 1
                if fail <= 5:
                    print(f"  失败 {url}: {e}")
    print(f"下载完成: 新增 {ok}, 跳过 {skip}, 失败 {fail}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--utts", nargs="*", default=[])
    ap.add_argument("--from-list", help="CSV, 含 utt 列")
    ap.add_argument("--speakers", nargs="*", default=NATIVE_SPEAKERS)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    utts = list(a.utts)
    if a.from_list:
        import pandas as pd
        df = pd.read_csv(a.from_list)
        col = "utt" if "utt" in df.columns else df.columns[0]
        utts += df[col].astype(str).tolist()
    utts = sorted(set(utts))
    print(f"待下载 {len(utts)} 句 × {len(a.speakers)} 位说话人")
    download(utts, a.speakers, a.force)
