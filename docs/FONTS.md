# 字体说明

本项目所有图使用**正式字体**，不使用度量替代品：

| 用途 | 字体 | 文件 |
|---|---|---|
| 西文 / 数字 / IPA | **Times New Roman** | `times.ttf`、`timesbd.ttf`、`timesi.ttf` |
| 中文 | **宋体（SimSun）** | `simsun.ttc` |
| 中文标题（预留） | 黑体（SimHei） | `simhei.ttf` |

## 为什么仓库里没有字体文件

Times New Roman 与宋体均为**商业字体**，只授权在已购许可的机器上使用，
不能随代码分发。因此：

- `.gitignore` 已排除 `*.ttf` / `*.ttc`
- 仓库只保留本说明，使用者需自行准备字体

## 获取方式

### 本机（Windows + WSL）

Windows 自带这些字体，直接复制即可：

```bash
mkdir -p ~/.fonts
cp /mnt/c/Windows/Fonts/{times.ttf,timesbd.ttf,timesi.ttf,simsun.ttc,simhei.ttf} ~/.fonts/
fc-cache -f ~/.fonts
rm -rf ~/.cache/matplotlib      # 关键：否则 matplotlib 仍用旧缓存
```

### 纯 Linux 环境

以下为**开源可分发**的替代方案（风格接近，非同一字体）：

| 目标 | 替代字体 | 说明 |
|---|---|---|
| Times New Roman | **Liberation Serif** | 度量兼容，排版宽度一致 |
| 宋体 | **Noto Serif CJK SC**（思源宋体） | Adobe/Google 开源，可商用 |
| 黑体 | **Noto Sans CJK SC**（思源黑体） | 同上 |

安装：

```bash
sudo apt install fonts-liberation fonts-noto-cjk
```

在本项目的 `src/figstyle.py` 中把 `FONT_EN` / `FONT_ZH` 改成对应名称即可。

## 验证

```python
from matplotlib import font_manager as fm
for name in ["Times New Roman", "SimSun"]:
    print(name, "->", fm.findfont(fm.FontProperties(family=name),
                                  fallback_to_default=False))
```

返回的必须是 `~/.fonts/` 下的真实文件；若返回 `DejaVuSans.ttf`，说明仍是 fallback。

## 已知限制

- **SimSun 不含 IPA 字形**。因此中文版图的元音标注使用 ARPAbet（`IY`、`IH`、`AE`…），
  英文版使用 IPA（`iː`、`ɪ`、`æ`…）。两套标签通过 `IPA_ZH` 映射保持一一对应。
- **SimSun 缺 U+2212（真减号）**。中文版图内数字使用 ASCII 连字符（`axes.unicode_minus=False`），
  英文版使用 U+2212。
- Times New Roman 的 IPA 覆盖经逐字符检查为 **27/27**（含 `ɪ æ ɑ ɔ ʊ ɜ ʌ ð θ ʃ ʒ ŋ ɹ ː ə ɛ`）。

## 导出规格

- 矢量：PDF，`pdf.fonttype = 42`（TrueType 子集嵌入，可在 Illustrator 中编辑）
- 位图：PNG，600 dpi
- 已验证：PDF 内为 `/FontFile2`，**无 Type 3 字体**（多数期刊不接受 Type 3）
