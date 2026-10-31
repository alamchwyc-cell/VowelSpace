"""
align.py — 强制对齐（Montreal Forced Aligner 封装）

为什么必须做音素级对齐:
  元音空间分析要求"知道每个元音在哪个时间点". 整段平均会把辅音、过渡段、
  静音全部混进来, 得到的不是元音共振峰.
  MMS/wav2vec2 那类对齐是字符级的, 英文拼写与读音不一一对应,
  无法可靠定位元音, 所以用 MFA(音素级, 带发音词典).

流程:
  1) 准备语料目录: <speaker>/<utt>.wav + <speaker>/<utt>.lab
  2) mfa align  →  TextGrid (音素级区间)
  3) 解析 TextGrid → [(phone, t0, t1), ...]
"""
import os
import shutil
import subprocess
from pathlib import Path

from praatio import textgrid


def prepare_corpus(records, out_dir: Path):
    """records: [{'speaker','utt_id','wav_path','text'}, ...]

    写成 MFA 要求的目录结构. 音频用**软链接**而非拷贝:
    既省空间, 也保证源文件修复后语料目录自动同步.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for r in records:
        spk = out_dir / r["speaker"]
        spk.mkdir(exist_ok=True)
        wav_dst = spk / f"{r['utt_id']}.wav"
        lab_dst = spk / f"{r['utt_id']}.lab"
        if wav_dst.is_symlink() and wav_dst.resolve() == Path(r["wav_path"]).resolve():
            pass
        else:
            if wav_dst.exists() or wav_dst.is_symlink():
                wav_dst.unlink()
            os.symlink(Path(r["wav_path"]).resolve(), wav_dst)
            n += 1
        if not lab_dst.exists():
            lab_dst.write_text(r["text"].strip(), encoding="utf-8")
    return n


def run_mfa(corpus: Path, dictionary: str, acoustic: str, out_dir: Path,
            jobs: int = 4, mfa_bin: str = "mfa"):
    """调用 MFA 命令行. dictionary/acoustic 可以是模型名, 也可以是路径."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [mfa_bin, "align", str(corpus), dictionary, acoustic, str(out_dir),
           "--clean", "--jobs", str(jobs), "--quiet"]
    print("运行:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return out_dir


def read_phone_tiers(textgrid_path: Path):
    """读取 TextGrid 中的 phones 层, 返回 [(phone, t0, t1), ...]."""
    tg = textgrid.openTextgrid(str(textgrid_path), includeEmptyIntervals=False)
    name = "phones" if "phones" in tg.tierNames else tg.tierNames[0]
    entries = tg.getTier(name).entries
    return [(lab, float(s), float(e)) for s, e, lab in entries if lab]


def collect_textgrids(align_dir: Path):
    """返回 {(speaker, utt): textgrid_path}.

    注意: 不能只用文件名做键 —— 不同说话人读的是同一批句子, 文件名完全相同,
    只用 stem 会把 2500 个文件压成 250 个.
    """
    out = {}
    for p in align_dir.rglob("*.TextGrid"):
        out[(p.parent.name, p.stem)] = p
    return out
