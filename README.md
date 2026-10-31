# VowelSpace：中国英语学习者元音空间的声学分析

> 用同一批朗读材料，比较普通话母语者与英语母语者的元音声学空间，
> 检验"母语迁移导致元音空间收缩"这一假设，并把结论做成一个可用的发音诊断工具。

## 一句话结论

**母语迁移不是"整体发音空间变小"，而是特定音位范畴的合并。**
元音空间面积差异不显著（p = 0.31），但 `/ɪ/–/iː/`（d = −3.03）
与 `/ɑː/–/ʌ/`（d = −4.36）几乎被合并。

## 主要结果

| 指标 | 中国学习者 | 英语母语者 | 效应量 |
|---|---|---|---|
| 元音空间面积（归一化 Bark²） | 4.36 | 5.04 | d = −0.77（p = 0.31） |
| **/ɪ/–/iː/ 距离** | **0.27** | **1.20** | **d = −3.03** |
| **/ɑː/–/ʌ/ 距离** | **0.42** | **1.00** | **d = −4.36** |
| /ʊ/–/uː/ 距离 | 0.54 | 0.78 | d = −0.82 |
| **/iː/ 与 /ɪ/ 时长比** | **1.12** | **1.41** | **d = −2.94（p = 0.008）** |
| **语速（音节/秒）** | **2.80** | **3.64** | **d = −2.95** |

混合效应模型：组别主效应 **p = 0.43（不显著）**，但 6 个元音的
组×元音交互项显著 —— 说明差异是**局部的**，整体指标检测不到。

## 数据

| 语料 | 说话人 | 数量 |
|---|---|---|
| L2-ARCTIC（普通话） | BWC、LXC、NCC | 750 条 |
| CMU ARCTIC | bdl、slt、clb、rms、jmk、awb、ksp | 1750 条 |

两组朗读**完全相同的 250 个提示句**。

## 流程

```
音频 → MFA 音素级对齐 → 元音中点 F1/F2
     → Hz→Bark → Lobanov 归一化
     → 元音空间面积 / 元音对距离
     → 混合效应模型 + 置换检验 + 效应量
```

## 目录

```
VowelSpace/
├── src/
│   ├── prepare_l2.py       parquet → wav（L2-ARCTIC）
│   ├── fetch_cmu.py        下载母语者录音（同一批句子）
│   ├── align.py            MFA 封装 + TextGrid 解析
│   ├── formants.py         元音中点共振峰（LPC，中位数）
│   ├── normalize.py        Bark + Lobanov / Nearey
│   ├── vowelspace.py       凸包面积、元音对距离
│   ├── stats.py            混合效应模型、置换检验、效应量
│   ├── analyze.py          主分析
│   ├── make_figs.py        出图
│   ├── reference.py        母语者参照分布（诊断用）
│   ├── diagnose.py         单段录音诊断
│   ├── intelligibility.py  ASR 词错率（可懂度代理）
│   ├── extract_more.py     时长 / 擦音频谱 / 韵律提取
│   ├── analyze_extra.py    三组补充分析
│   └── app.py              网页界面
├── figs/    结果图
├── results/ 结果表
├── report/  研究设计 + 结果分析
└── run_all.sh
```

## 发音诊断工具

```bash
python3 src/diagnose.py --wav my.wav --text "the text you read"
```

```
==========================================================
  总分：51 / 100
  （检测到 70 个元音 token，覆盖 10 个元音）
==========================================================
  问题元音（按偏离程度排序）
    1. /uː/  偏离 2.38σ   得分 6
    2. /ɪ/   偏离 2.33σ   得分 7
       与 /iː/ 区分不足。试着把舌位放低一点、肌肉放松，缩短时长。
```

网页版：

```bash
streamlit run src/app.py
```

**评分方法**：把用户的元音中位数与母语者参照分布比较，用马氏距离度量偏离，
再按 χ²(2) 转换为百分位。公式固定，未做人工调参。

## 环境

```bash
pip install -r requirements.txt
# 强制对齐
conda create -n mfa -c conda-forge montreal-forced-aligner python=3.10
```

MFA 模型（国内 GitHub 限流时可直接从 HuggingFace 取）：

```bash
mkdir -p models/acoustic models/dict
for f in final.mdl final.alimdl lda.mat phones.txt tree meta.json; do
  curl -L "https://huggingface.co/MontrealCorpusTools/english_us_arpa/resolve/main/acoustic/$f" -o models/acoustic/$f
done
curl -L "https://huggingface.co/MontrealCorpusTools/english_us_arpa/resolve/main/dictionary/english_us_arpa.dict" \
     -o models/dict/english_us_arpa.dict
```

## 复现

```bash
bash run_all.sh
```

## 方法与严谨性

| 设计 | 排除的混淆 |
|---|---|
| 两组读同一批句子 | 文本、音素环境 |
| Bark 标度 + Lobanov 归一化 | 声道长度、性别 |
| 混合效应模型（说话人随机截距） | 同一人多条样本的伪重复 |
| 置换检验（说话人层面） | 正态假设 |
| 双归一化方法 | 结论对方法的依赖 |
| 效应量与事先声明判据 | 只报 p 值 |

**一个曾经踩过的坑**：初版把 Lobanov 作用在 Hz 上，却用未归一化的 Bark 算面积，
导致结论被性别差异污染。修正后元音空间面积的方向反转。这个教训写在
`report/02_结果与分析.md` 第 0 节。

## 局限

- 只有 3 位普通话说话人（公开镜像仅含 BWC/LXC/NCC；官方语料的 TNI、TXHC 未收录）。
- 朗读语料，非自然口语。
- 说话人层面的统计功效有限，主结论依赖 token 级模型。
