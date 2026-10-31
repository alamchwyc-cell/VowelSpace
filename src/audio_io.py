"""
audio_io.py — 统一的音频读取与格式转换

问题:
  处理链下游用的是 Praat(parselmouth), 它只能读 WAV/AIFF/FLAC 等少数格式,
  读不了 mp3 与 m4a——而手机录音默认就是 m4a。

策略(两级):
  1) 先用 soundfile 解码(wav / mp3 / flac / ogg / aiff ...), 失败则
  2) 调 ffmpeg 转码(m4a / aac / wma / amr / 3gp ...)

  统一输出 16 kHz 单声道 WAV(临时文件), 供 Praat 读取.

用法:
    from audio_io import ensure_wav16k
    wav, tmp = ensure_wav16k("我的录音.m4a")
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

TARGET_SR = 16000
# 支持的扩展名(用于界面提示)
SUPPORTED = ["wav", "mp3", "m4a", "aac", "flac", "ogg", "opus", "wma", "amr", "aiff", "aif"]


# ffmpeg 可能不在 PATH 上: 依次查找(项目约定非 PATH 工具放 tools/bin)
_FFMPEG_CANDIDATES = [
    "ffmpeg",
    str(Path.home() / "miniconda3/envs/audio/bin/ffmpeg"),
    str(Path.home() / "workspace/tools/bin/ffmpeg"),
]


def ffmpeg_path():
    for c in _FFMPEG_CANDIDATES:
        p = shutil.which(c) if "/" not in c else (c if Path(c).exists() else None)
        if p:
            return p
    return None


def _ffmpeg_available() -> bool:
    return ffmpeg_path() is not None


def _resample(x, sr: int, target: int = TARGET_SR):
    import numpy as np
    if sr == target:
        return x
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(sr, target)
    return resample_poly(x, target // g, sr // g)


def _read_with_soundfile(path: Path):
    import soundfile as sf
    data, sr = sf.read(str(path), always_2d=True)
    mono = data.mean(axis=1)
    return _resample(mono, sr), TARGET_SR


def _read_with_ffmpeg(path: Path):
    """用 ffmpeg 解码为 16 kHz 单声道 PCM, 走管道避免中间文件."""
    import numpy as np
    cmd = [ffmpeg_path() or "ffmpeg", "-v", "error", "-i", str(path),
           "-f", "s16le", "-acodec", "pcm_s16le",
           "-ac", "1", "-ar", str(TARGET_SR), "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0, TARGET_SR


def ensure_wav16k(path, target_sr: int = TARGET_SR):
    """把任意常见格式转成 16 kHz 单声道 wav.

    返回 (wav_path, tmp_dir_or_None). tmp_dir 不为 None 时由调用方负责清理.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower().lstrip(".")
    data = None

    # 已是标准 wav 且采样率合适: 直接返回原路径, 不起临时文件
    if suffix == "wav":
        try:
            import soundfile as sf
            info = sf.info(str(path))
            if abs(info.samplerate - target_sr) < 1 and info.channels == 1:
                return path, None
        except Exception:
            pass

    # 1) soundfile
    try:
        data, sr = _read_with_soundfile(path)
    except Exception:
        data = None

    # 2) ffmpeg 兜底
    if data is None:
        if not _ffmpeg_available():
            raise RuntimeError(
                f"无法读取 {path.name}: soundfile 不支持该格式, 且系统中没有 ffmpeg.\n"
                f"请安装 ffmpeg(conda install -c conda-forge ffmpeg)或改用 wav/mp3.")
        data, sr = _read_with_ffmpeg(path)

    import numpy as np
    import soundfile as sf
    td = tempfile.mkdtemp(prefix="vw_")
    out = Path(td) / "converted.wav"
    sf.write(str(out), np.asarray(data, dtype=np.float32), target_sr, subtype="PCM_16")
    return out, Path(td)


def probe(path) -> dict:
    """返回音频的基本信息, 供界面显示与质量检查."""
    path = Path(path)
    info = {"file": path.name, "size_mb": round(path.stat().st_size / 1e6, 2)}
    try:
        import soundfile as sf
        i = sf.info(str(path))
        info.update(samplerate=i.samplerate, channels=i.channels,
                    duration=round(i.duration, 1), format=i.format)
    except Exception:
        if _ffmpeg_available():
            try:
                probe = str(Path(ffmpeg_path()).with_name("ffprobe")) if ffmpeg_path() else None
                dur = subprocess.run(
                    [probe or "ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=nw=1:nk=1", str(path)],
                    capture_output=True, text=True, check=True).stdout.strip()
                info.update(duration=round(float(dur), 1), format=path.suffix.lstrip("."))
            except Exception:
                pass
    return info
