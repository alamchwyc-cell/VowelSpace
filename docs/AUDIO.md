# 音频格式支持

## 支持的格式

| 格式 | 解码方式 | 说明 |
|---|---|---|
| wav | soundfile | 推荐，无损 |
| mp3 | soundfile（libsndfile ≥1.1） | 无需 ffmpeg |
| flac / ogg / aiff | soundfile | |
| **m4a / aac** | **ffmpeg** | iPhone/Android 录音默认格式 |
| wma / amr | ffmpeg | |
| silk（微信语音） | ❌ | 需专用解码器，不支持 |

## 处理链

```
任意格式 → soundfile 解码 ─┬→ 16 kHz 单声道 wav（临时）→ Praat / MFA
            ↓ 失败          │
        ffmpeg 解码 ────────┘
```

统一转成 16 kHz 单声道，是因为：
1. CMU ARCTIC 原生就是 16 kHz；
2. 共振峰只看到 5.5 kHz 以下，16 kHz 的 Nyquist（8 kHz）足够；
3. 统一采样率避免重采样引入的差异。

## ffmpeg 的安装位置

conda 装 base 环境会与 `requests` 冲突，因此单独建环境：

```bash
conda create -n audio -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ ffmpeg -y
```

`src/audio_io.py` 按以下顺序查找 ffmpeg：

1. `PATH` 中的 `ffmpeg`
2. `~/miniconda3/envs/audio/bin/ffmpeg`
3. `~/workspace/tools/bin/ffmpeg`

找不到时，只有 m4a/aac/wma/amr 会报错；wav/mp3/flac/ogg 仍可正常使用。

## 手机录音建议

- iPhone：语音备忘录默认 m4a，可直接上传；
- Android：录音机多为 m4a 或 3gp，可直接上传；
- 微信语音：格式特殊，**请改用系统录音机**。
