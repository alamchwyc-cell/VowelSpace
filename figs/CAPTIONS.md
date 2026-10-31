# 图注（Figure Captions）

中英两套图分别在 `figs/en/` 与 `figs/zh/`，各含 PDF（矢量）、PNG（600 dpi）、SVG。

统计约定：误差线一律为 **95% 置信区间**（t 分布，df = n − 1）；
d 为 **Cohen's d**（学习者 − 母语者）；p 值来自置换检验或非参数检验，
图注中逐条注明。n 指**说话人数**，不是 token 数。

> 数据：31 位说话者，7,750 条录音，78,293 个元音 token。
> 图 1–7 为普通话（n = 5）与英语母语者（n = 7）的对比；图 8 为全部七个组。

---

## 图 1 元音空间

**中文**

> **图 1　中国英语学习者与英语母语者的元音空间对比。** 坐标为说话人内 Lobanov 归一化后的
> Bark 值（F1 向下增大，F2 向左增大）。圆点为英语母语者（n = 7），三角为中国英语学习者（n = 5）；
> 阴影为各组凸包，其面积即元音空间面积（VSA）。

**English**

> **Figure 1. Vowel space of Mandarin learners and English natives.**
> Coordinates are Lobanov-normalised Bark values (F1 increases downward, F2 increases leftward).
> Circles denote English natives (n = 7), triangles Mandarin learners (n = 5);
> shaded regions are group convex hulls, whose area is the vowel space area.

## 图 2 元音对距离

**中文**

> **图 2　关键元音对的声学距离。** 每个点代表一位说话人（说话人内各元音中位数之间的
> 欧氏距离，单位为归一化 Bark）。横线为组均值，误差线为 95% 置信区间。各组按 |d| 降序。

**English**

> **Figure 2. Acoustic distance for selected vowel pairs.**
> Each dot is one speaker (Euclidean distance between within-speaker vowel medians in
> normalised Bark). Horizontal bars are group means; whiskers are 95% CIs. Pairs are ordered by |d|.

## 图 3 混合效应模型

**中文**

> **图 3　组别 × 元音交互项。** 混合效应模型 `F1 ~ group × vowel + gender + (1|speaker)`。
> 点为母语者相对学习者的 F1 差值，误差线为 95% 置信区间，虚线为 0。
> \*\*\* p < 0.001，\*\* p < 0.01，\* p < 0.05，n.s. 不显著。

**English**

> **Figure 3. Group × vowel interaction terms.**
> Mixed-effects model `F1 ~ group × vowel + gender + (1|speaker)`. Dots show the
> native-minus-learner difference in F1; whiskers are 95% CIs; the dashed line marks zero.
> \*\*\* p < 0.001, \*\* p < 0.01, \* p < 0.05, n.s. not significant.

## 图 4 置换检验

**中文**

> **图 4　元音空间面积的精确置换检验。** 对 12 位说话人的分组标签做全部
> C(12, 5) = 792 种重排，得到零分布。红竖线为观测到的组间差值（学习者 − 母语者），
> 双侧精确 p = 0.095。

**English**

> **Figure 4. Exact permutation test for vowel space area.**
> All C(12, 5) = 792 relabellings of the twelve speakers yield the null distribution.
> The red line marks the observed difference (learners − natives); two-sided exact p = 0.095.

## 图 5 元音偏离与可懂度

**中文**

> **图 5　元音偏离度与 ASR 词错率的关系。** 每位说话人一个点：横轴为该说话人各元音
> 到母语者参照分布的平均马氏距离，纵轴为自动语音识别（wav2vec2-base-960h）在其 40 句
> 朗读上的词错率。Spearman ρ = 0.937（n = 12，p < 0.001）；仅母语者内部 ρ = 1.000（n = 7）。

**English**

> **Figure 5. Vowel deviation and ASR word error rate.**
> One point per speaker. x-axis: mean Mahalanobis distance of the speaker's vowels from the
> native reference distribution; y-axis: word error rate of an ASR system (wav2vec2-base-960h)
> on 40 sentences per speaker. Spearman ρ = 0.937 (n = 12, p < 0.001); within natives only,
> ρ = 1.000 (n = 7).

## 图 6 松紧时长

**中文**

> **图 6　松紧元音的时长区分。** 纵轴为 `/iː/` 与 `/ɪ/` 的说话人内时长中位数之比
> （仅重读元音）。学习者 1.13 vs 母语者 1.27，d = −1.06，p = 0.074（趋势，未达显著）。

**English**

> **Figure 6. Tense–lax duration contrast.**
> Within-speaker median duration ratio of /iː/ to /ɪ/, stressed vowels only.
> Learners 1.13 vs natives 1.27, d = −1.06, p = 0.074 (trend, not significant).

## 图 7 语速

**中文**

> **图 7　朗读语速。** 纵轴为每句话的音节数除以语音段时长（音节/秒）。
> 学习者 2.89 vs 母语者 3.64，d = −2.51，p = 0.005。

**English**

> **Figure 7. Speech rate.**
> Syllables per second, computed per utterance over the voiced span.
> Learners 2.89 vs natives 3.64, d = −2.51, p = 0.005.

## 图 8 跨母语比较（核心图）

**中文**

> **图 8　不同母语背景下 `/ɪ/` 与 `/iː/` 的声学距离。** 每个点代表一位说话人
> （说话人内两元音中位数之间的欧氏距离，归一化 Bark）。横线为组均值，n 为组内说话人数。
> 圆点为英语母语者，三角为二语学习者，各组按均值降序排列。
> 七组整体差异 Kruskal–Wallis p = 0.002；但**普通话与其他二语组无差异**（p = 0.73），
> 西班牙语与越南语的合并程度强于普通话。

**English**

> **Figure 8. Acoustic distance between /ɪ/ and /iː/ across first-language backgrounds.**
> Each dot is one speaker (Euclidean distance between the within-speaker medians of the two
> vowels, in normalised Bark). Horizontal bars are group means; n = number of speakers.
> Circles: English natives; triangles: L2 learners. Groups are ordered by descending mean.
> Overall Kruskal–Wallis p = 0.002, but Mandarin does not differ from the other L2 groups
> (p = 0.73); Spanish and Vietnamese show even greater merging than Mandarin.

---

## 未采用的图

擦音频谱矩（`/θ/` `/ð/`）曾实现并分析，但三条独立检验均未通过
（组间 p = 0.25；ASR 在含 th 词上的错误率与其它词无差异；测量本身不稳定），
故未纳入图集。数据保留在 `results/fricative_*.csv` 备查。
