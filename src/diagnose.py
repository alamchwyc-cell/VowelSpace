"""
diagnose.py — 元音发音诊断

输入: 一段朗读录音 + 对应文本
输出: 每个元音的偏离程度、总分、问题元音与改进建议

流程:
  1) MFA align_one 做音素级对齐（单文件约 20 秒）
  2) 元音中点提取 F1/F2
  3) 该说话人内部的 Lobanov 归一化（需要足够多的元音 token）
  4) 与母语者参照分布比较（马氏距离 + χ² 百分位）
  5) 规则化生成建议

用法:
    python3 src/diagnose.py --wav my.wav --text "the text you read"
    python3 src/diagnose.py --wav my.wav --text-file sentence.txt
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from align import read_phone_tiers  # noqa: E402
from formants import load_sound, vowel_measurements  # noqa: E402
from normalize import add_bark, lobanov  # noqa: E402
from reference import F1, F2, VOWELS, score_speaker  # noqa: E402
from run_pipeline import VOWELS as VOWEL_MAP  # noqa: E402

DICT = MODELS / "dict/english_us_arpa.dict"
ACOUSTIC = MODELS / "acoustic"
REF = MODELS / "vowel_reference.json"
DUR_REF = MODELS / "duration_reference.json"
MFA_ENV = os.environ.get("MFA_ENV", "mfa")


def mfa_command():
    """定位 mfa 可执行文件.

    三种情况:
      1) 容器里 MFA 直接装在 base 环境 -> PATH 上有 mfa
      2) 本地用 conda 环境 mfa -> conda run -n mfa mfa
      3) 显式指定 MFA_CMD
    """
    import shutil
    from pathlib import Path as _P

    explicit = os.environ.get("MFA_CMD")
    if explicit:
        return explicit.split()
    # 本地 conda 环境 mfa 优先(base 里可能存在一个缺 _kalpy 的坏副本)
    conda_bin = _P.home() / "miniconda3/envs" / MFA_ENV / "bin/mfa"
    if conda_bin.exists():
        return ["conda", "run", "-n", MFA_ENV, "mfa"]
    # 容器里通常直接装在 PATH 上
    if shutil.which("mfa"):
        return ["mfa"]
    return ["conda", "run", "-n", MFA_ENV, "mfa"]
MIN_TOKENS = 25           # 少于这个数量, 归一化不可靠


def align_one(wav: Path, text: str, out_tg: Path):
    with tempfile.NamedTemporaryFile("w", suffix=".lab", delete=False,
                                     encoding="utf-8") as f:
        f.write(text)
        lab = f.name
    try:
        subprocess.run(
            mfa_command() + ["align_one",
                             str(wav), lab, str(DICT), str(ACOUSTIC), str(out_tg)],
            check=True, capture_output=True, text=True)
    finally:
        os.unlink(lab)
    return out_tg


def extract(wav: Path, tg: Path) -> pd.DataFrame:
    segs = read_phone_tiers(tg)
    snd = load_sound(wav, target_sr=16000)
    rows = vowel_measurements(snd, segs, set(VOWEL_MAP), VOWEL_MAP)
    return pd.DataFrame(rows).dropna(subset=["F1", "F2"])


ADVICE = {
    "iː": "长元音、紧元音：舌位高而前，时长约为 /ɪ/ 的 1.3 倍以上。对比练习：sheep–ship、beat–bit。",
    "ɪ": "短元音、松元音：比 /iː/ 更短更松，舌位略低、略中央。对比练习：ship–sheep、sit–seat。",
    "e": "半高前元音，开口度介于 /ɪ/ 与 /æ/ 之间。对比练习：bed–bad、pen–pan。",
    "æ": "低前元音，需要明显的下颌下降。对比练习：bad–bed、man–men。",
    "ɑː": "后低长元音，舌后缩并压低，口型张开。对比练习：cart–cut、heart–hut。",
    "ɔː": "后中长元音，圆唇且舌位靠后。对比练习：caught–cot、walk–wok。",
    "ʊ": "短后元音，圆唇但放松、时长短。对比练习：full–fool、pull–pool。",
    "uː": "长后元音，圆唇紧、时长长。对比练习：fool–full、pool–pull。",
    "ʌ": "中央短元音，舌位居中、时长短促。对比练习：cut–cart、hut–heart。",
    "ɜː": "中央长元音，舌位居中、唇形中性。对比练习：bird–bud、work–walk。",
}


def direction_text(vowel: str, user_xy, ref_mu, thr: float = 0.15) -> str:
    """根据本次测量与母语者中心的偏移方向生成动作指示.

    归一化 Bark 空间中:
      F1 变小 = 舌位偏高(开口不足); F1 变大 = 舌位偏低(开口过大)
      F2 变小 = 舌位偏后;           F2 变大 = 舌位偏前
    """
    d1 = float(user_xy[0] - ref_mu[0])
    d2 = float(user_xy[1] - ref_mu[1])
    parts = []
    if abs(d1) >= thr:
        parts.append("舌位偏高、开口不足" if d1 < 0 else "舌位偏低、开口过大")
    if abs(d2) >= thr:
        parts.append("舌位偏后" if d2 < 0 else "舌位偏前")
    if not parts:
        return "位置已在母语者范围内，无需调整"
    return "、".join(parts) + f"（F1 {d1:+.2f}，F2 {d2:+.2f}）"


def diagnose(wav_path, text: str, ref_path: Path = REF) -> dict:
    from audio_io import ensure_wav16k

    ref = json.loads(Path(ref_path).read_text(encoding="utf-8"))

    # 任意常见格式(m4a/mp3/...)统一转成 16 kHz 单声道 wav, 再交给 Praat/MFA
    wav, tmp_dir = ensure_wav16k(wav_path)
    try:
        with tempfile.TemporaryDirectory() as td:
            tg = Path(td) / "out.TextGrid"
            align_one(wav, text, tg)
            toks = extract(wav, tg)
    finally:
        if tmp_dir is not None:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if len(toks) < MIN_TOKENS:
        return {"ok": False,
                "reason": f"只检测到 {len(toks)} 个元音，少于 {MIN_TOKENS} 个，"
                          f"归一化不可靠。请朗读更长的材料。"}

    toks = toks.copy()
    toks["speaker"] = "user"          # Lobanov 需要一个说话人列
    df = lobanov(add_bark(toks))
    med = df.groupby("vowel")[[F1, F2]].median().dropna()
    per_vowel, total = score_speaker(med, ref)

    # ---- 时长: 松紧对立 ----
    dur_ref = json.loads(DUR_REF.read_text(encoding="utf-8")) if DUR_REF.exists() else None
    dur_info = None
    if dur_ref:
        med_dur = toks.groupby("vowel")["dur"].median()
        if "iː" in med_dur.index and "ɪ" in med_dur.index:
            ratio = float(med_dur["iː"] / med_dur["ɪ"])
            dur_info = {
                "ratio": ratio,
                "ref_mean": dur_ref["tense_ratio_mean"],
                "ref_sd": dur_ref["tense_ratio_sd"],
                "z": (ratio - dur_ref["tense_ratio_mean"]) /
                     max(1e-6, dur_ref["tense_ratio_sd"]),
            }

    problems = per_vowel.head(3).to_dict("records")
    for p in problems:
        p["advice"] = ADVICE.get(p["vowel"], "")
        rv = ref["vowels"].get(p["vowel"])
        if rv is not None and p["vowel"] in med.index:
            p["direction"] = direction_text(
                p["vowel"], med.loc[p["vowel"]].values, np.array(rv["mu"]))
        else:
            p["direction"] = ""

    return {
        "ok": True,
        "n_tokens": int(len(df)),
        "n_vowels_found": int(len(per_vowel)),
        "total_score": total,
        "per_vowel": per_vowel.to_dict("records"),
        "problems": [p for p in problems if p["score"] < 50],
        "duration": dur_info,
        "vowel_points": {v: med.loc[v, [F1, F2]].tolist()
                         for v in med.index},
    }


def render(res: dict) -> str:
    if not res["ok"]:
        return f"无法评估：{res['reason']}"
    L = ["=" * 58,
         f"  总分：{res['total_score']:.0f} / 100",
         f"  （检测到 {res['n_tokens']} 个元音 token，覆盖 {res['n_vowels_found']} 个元音）",
         "=" * 58]
    if res["problems"]:
        L.append("  问题元音（按偏离程度排序）")
        for i, p in enumerate(res["problems"], 1):
            L.append(f"    {i}. /{p['vowel']}/  偏离 {p['distance']:.2f}σ   "
                     f"得分 {p['score']:.0f}")
            if p.get("direction"):
                L.append(f"       方向：{p['direction']}")
            L.append(f"       {p['advice']}")
    else:
        L.append("  没有发现明显偏离的元音。")
    if res.get("duration"):
        d = res["duration"]
        flag = "正常" if abs(d["z"]) < 1 else ("偏短" if d["z"] < 0 else "偏长")
        L.append("")
        L.append(f"  松紧时长：/iː/ : /ɪ/ = {d['ratio']:.2f}"
                 f"（母语者 {d['ref_mean']:.2f}±{d['ref_sd']:.2f}，{flag}）")
        if d["z"] < -1:
            L.append("    → 长元音拉得不够长。试着把 /iː/ 的时长放慢到 /ɪ/ 的 1.5 倍。")
    L.append("")
    L.append("  各元音得分：")
    for p in res["per_vowel"]:
        L.append(f"    /{p['vowel']:3s}/  {p['score']:5.1f}")
    L.append("=" * 58)
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", "--audio", dest="wav", required=True,
                    help="录音文件，支持 wav / mp3 / m4a / flac / ogg 等")
    ap.add_argument("--text")
    ap.add_argument("--text-file")
    a = ap.parse_args()

    text = a.text or (Path(a.text_file).read_text(encoding="utf-8") if a.text_file else None)
    if not text:
        ap.error("需要 --text 或 --text-file")

    print(render(diagnose(a.wav, text.strip())))
