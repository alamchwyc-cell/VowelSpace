"""
stats.py — 统计检验

三个层次:
  1) 混合效应模型: 处理"同一说话人说多句"导致的非独立性
  2) 置换检验: 在说话人层面打乱分组标签, 不依赖正态假设
  3) 效应量: 只报 p 值不够, 必须报"差多少"

为什么必须用混合效应模型:
  每位说话人贡献 250 条发音, 这些样本彼此不独立. 若直接做 t 检验,
  等于把 11 个人当成 2750 个独立个体, p 值会严重虚低(伪重复).
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as st


def mixed_model(df: pd.DataFrame, dv: str, group_col="group", vowel_col="vowel",
                speaker_col="speaker", gender_col="gender"):
    """dv ~ group * vowel + gender + (1|speaker).

    注意: dv 应当用**原始共振峰**(或对数), 不能用 Lobanov 归一化后的值.
    Lobanov 已把每个说话人的均值归零, 随机截距随之退化为 0,
    模型会报 "random effects covariance is singular".
    归一化值只用于描述性的元音空间图.
    """
    d = df.copy()
    d = d.rename(columns={group_col: "group", vowel_col: "vowel",
                          speaker_col: "speaker", gender_col: "gender"})
    d["gender"] = d["gender"].fillna("U")
    formula = f"{dv} ~ C(group) * C(vowel) + C(gender)"
    model = smf.mixedlm(formula, d, groups=d["speaker"])
    for method in ("lbfgs", "powell", "bfgs"):
        try:
            res = model.fit(method=method, maxiter=500)
            if np.isfinite(res.params).all() and (res.bse < 1e4).all():
                return res
        except Exception:
            continue
    return res


def cohens_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def permutation_test(values, labels, n_perm=10000, seed=42):
    """在"说话人"层面打乱分组标签, 检验两组均值差.

    values: 每个说话人的指标值 (如元音空间面积)
    labels: 对应的组别 (L2 / Native)
    """
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    obs = values[labels == "L2"].mean() - values[labels == "Native"].mean()

    rng = np.random.default_rng(seed)
    n_l2 = int((labels == "L2").sum())
    cnt = 0
    for _ in range(n_perm):
        idx = rng.permutation(len(values))
        a = values[idx[:n_l2]].mean()
        b = values[idx[n_l2:]].mean()
        if abs(a - b) >= abs(obs):
            cnt += 1
    return {"observed_diff": float(obs), "p_value": (cnt + 1) / (n_perm + 1),
            "n_perm": n_perm}


def welch_ttest(a, b):
    t, p = st.ttest_ind(a, b, equal_var=False)
    return {"t": float(t), "p": float(p),
            "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
            "d": cohens_d(a, b)}
