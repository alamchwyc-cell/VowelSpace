# 图注（Figure Captions）

中英两套图分别在 `figs/en/` 与 `figs/zh/`，各含 PDF（矢量）、PNG（600 dpi）、SVG。
本文件给出与期刊/学位论文格式一致的图注，可直接粘入正文。

统计约定：误差线一律为 **95% 置信区间**（t 分布，df = n − 1）；
d 为 **Cohen's d**（学习者 − 母语者）；p 值来自 **置换检验**（说话人层面，10,000 次重抽样；
图 4 为精确枚举）。n 指**说话人数**，不是 token 数。

---

## 图 1 元音空间

**中文**

> **图 1　中国英语学习者与英语母语者的元音空间对比。** 坐标为说话人内 Lobanov 归一化后的
> Bark 值（F1 向下增大，F2 向左增大）。实心点为各元音在该组内的中位数，阴影为凸包，
> 面积即元音空间面积（VSA）。英语母语者 n = 7，中国英语学习者 n = 3。

**English**

> **Figure 1. Vowel space of Mandarin learners and English natives.**
> Coordinates are Lobanov-normalised Bark values (F1 increases downward, F2 increases leftward).
> Dots mark group medians for each vowel; shaded regions are convex hulls, whose area is the
> vowel space area (VSA). n = 7 English natives and n = 3 Mandarin learners.

## 图 2 元音对距离

**中文**

> **图 2　关键元音对的声学距离。** 每个点代表一位说话人（说话人内各元音中位数之间的
> 欧氏距离，单位为归一化 Bark）。横线为组均值，误差线为 95% 置信区间。
> 各组按 |d| 降序排列。d 为 Cohen's d（学习者 − 母语者）。

**English**

> **Figure 2. Acoustic distance for selected vowel pairs.**
> Each dot is one speaker (Euclidean distance between within-speaker vowel medians in
> normalised Bark). Horizontal bars show group means; whiskers are 95% CIs.
> Pairs are ordered by |d|. d = Cohen's d (learners − natives).

## 图 3 混合效应模型

**中文**

> **图 3　组别 × 元音交互项。** 混合效应模型 `F1 ~ group × vowel + gender + (1|speaker)`
> （13,490 个 token）。点为母语者相对学习者的 F1 差值，误差线为 95% 置信区间；
> 虚线为 0（无差异）。显著性标记：\*\*\* p < 0.001，\*\* p < 0.01，\* p < 0.05，n.s. 不显著。

**English**

> **Figure 3. Group × vowel interaction terms.**
> Mixed-effects model `F1 ~ group × vowel + gender + (1|speaker)` over 13,490 tokens.
> Dots show the native-minus-learner difference in F1; whiskers are 95% CIs; the dashed line
> marks zero. \*\*\* p < 0.001, \*\* p < 0.01, \* p < 0.05, n.s. not significant.

## 图 4 置换检验

**中文**

> **图 4　元音空间面积的精确置换检验。** 对 10 位说话人的分组标签做全部
> C(10, 3) = 120 种重排，得到零分布。红色竖线为观测到的组间差值（学习者 − 母语者），
> 双侧精确 p = 0.308。

**English**

> **Figure 4. Exact permutation test for vowel space area.**
> All C(10, 3) = 120 relabellings of the ten speakers yield the null distribution.
> The red line marks the observed group difference (learners − natives); two-sided exact
> p = 0.308.

## 图 5 元音偏离与可懂度

**中文**

> **图 5　元音偏离度与 ASR 词错率的关系。** 每位说话人一个点：横轴为该说话人各元音
> 到母语者参照分布的平均马氏距离，纵轴为自动语音识别（wav2vec2-base-960h）在该说话人
> 40 句朗读上的词错率。Spearman ρ = 0.988（n = 10，p < 0.001）；仅母语者内部 ρ = 1.000（n = 7）。

**English**

> **Figure 5. Vowel deviation and ASR word error rate.**
> One point per speaker. x-axis: mean Mahalanobis distance of the speaker's vowels from the
> native reference distribution; y-axis: word error rate of an ASR system
> (wav2vec2-base-960h) on 40 sentences per speaker. Spearman ρ = 0.988 (n = 10, p < 0.001);
> within natives only, ρ = 1.000 (n = 7).

## 图 6 松紧时长

**中文**

> **图 6　松紧元音的时长区分。** 纵轴为 `/iː/` 与 `/ɪ/` 的说话人内时长中位数之比。
> 每点一位说话人，横线与误差线为组均值与 95% 置信区间。
> 学习者 1.12 vs 母语者 1.41，d = −2.94，p = 0.008。

**English**

> **Figure 6. Tense–lax duration contrast.**
> Within-speaker median duration ratio of /iː/ to /ɪ/. Each dot is one speaker; the horizontal
> bar and whiskers show the group mean and 95% CI. Learners 1.12 vs natives 1.41,
> d = −2.94, p = 0.008.

## 图 7 语速

**中文**

> **图 7　朗读语速。** 纵轴为每句话的音节数除以语音段时长（音节/秒）。
> 学习者 2.80 vs 母语者 3.64，d = −2.95，p < 0.001。

**English**

> **Figure 7. Speech rate.**
> Syllables per second, computed per utterance over the voiced span.
> Learners 2.80 vs natives 3.64, d = −2.95, p < 0.001.

---

## 未采用的图

擦音频谱矩（`/θ/` `/ð/`）曾实现并分析，但三条独立检验均未通过
（组间 p = 0.25；ASR 在含 th 词上的错误率与其它词无差异；测量本身不稳定），
故未纳入图集。数据保留在 `results/fricative_*.csv` 备查。
