"""
shrinkage.py — 小样本鲁棒的说话人归一化

问题:
  标准 Lobanov 归一化用说话人自身的均值/方差做 z-score。
  当样本少(例如只读了 12 句)时, 这两个量的估计噪声很大,
  同一个人的诊断结果会随录音长度大幅波动——实测中同一位说话人
  12 句给出时长比 1.60, 250 句给出 1.14, 结论相反。

方法:
  把说话人的均值/方差向同性别母语者总体收缩(经验贝叶斯 / James-Stein):

      μ̂ = (n·μ_s + k·μ_pop) / (n + k)
      σ̂ = (n·σ_s + k·σ_pop) / (n + k)

  n 是该说话人的 token 数, k 是收缩强度。
  n 大 -> 接近标准 Lobanov; n 小 -> 更多依赖总体先验。

  先验按性别分组: 女性与男性的共振峰基线不同, 用性别先验比用全体先验更合理。

验证:
  对每位说话人抽样 n = 10/20/30/50/100 个 token, 用不同方法归一化后
  计算元音中位数, 与"全样本"结果比较, 看偏差是否更小。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FORMS = ["F1_bark", "F2_bark"]


def population_prior(df: pd.DataFrame, gender_col="gender"):
    """按性别估计总体先验 (均值与标准差)."""
    prior = {}
    for g, sub in df.groupby(gender_col):
        prior[g] = {
            "mu": {f: float(sub[f].mean()) for f in FORMS},
            "sd": {f: float(sub[f].std()) for f in FORMS},
            "n": int(len(sub)),
        }
    # 性别缺失时用全体
    prior["_all"] = {
        "mu": {f: float(df[f].mean()) for f in FORMS},
        "sd": {f: float(df[f].std()) for f in FORMS},
        "n": int(len(df)),
    }
    return prior


def _prior_for(prior, g):
    return prior.get(g) or prior.get("_all")


def lobanov(df: pd.DataFrame, speaker="speaker", suffix="_n") -> pd.DataFrame:
    out = df.copy()
    for f in FORMS:
        g = out.groupby(speaker)[f]
        out[f + suffix] = (out[f] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    return out


def shrink_normalize(df: pd.DataFrame, prior, k: float = 50.0,
                     speaker="speaker", gender_col="gender", suffix="_s") -> pd.DataFrame:
    out = df.copy()
    n_by_spk = out.groupby(speaker).size()
    g_by_spk = out.groupby(speaker)[gender_col].first() if gender_col in out else None

    for f in FORMS:
        mu_s = out.groupby(speaker)[f].transform("mean")
        sd_s = out.groupby(speaker)[f].transform("std")
        n = out[speaker].map(n_by_spk).astype(float)

        if g_by_spk is not None:
            mu_pop = out[speaker].map({s: _prior_for(prior, g)["mu"][f]
                                       for s, g in g_by_spk.items()})
            sd_pop = out[speaker].map({s: _prior_for(prior, g)["sd"][f]
                                       for s, g in g_by_spk.items()})
        else:
            p = prior["_all"]
            mu_pop = pd.Series(p["mu"][f], index=out.index)
            sd_pop = pd.Series(p["sd"][f], index=out.index)

        mu_hat = (n * mu_s + k * mu_pop) / (n + k)
        sd_hat = (n * sd_s + k * sd_pop) / (n + k)
        out[f + suffix] = (out[f] - mu_hat) / sd_hat.replace(0, np.nan)
    return out


def stability_test(df: pd.DataFrame, prior, speaker, ns=(10, 20, 30, 50, 100),
                   n_rep=30, seed=0, vowel_col="vowel"):
    """比较两种归一化在小样本下与全样本结果的平均绝对偏差."""
    rng = np.random.default_rng(seed)
    rows = []
    for spk, g in df.groupby(speaker):
        full_l = lobanov(g.copy())
        full_s = shrink_normalize(g.copy(), prior)
        # 以"全样本 Lobanov"为参照基准, 看小样本估计偏离多远
        base = full_l.groupby(vowel_col)[["F1_bark_n", "F2_bark_n"]].median()
        if base.isna().all().all():
            continue
        for n in ns:
            if len(g) < n:
                continue
            err_l, err_s = [], []
            for _ in range(n_rep):
                sub = g.sample(n=n, random_state=int(rng.integers(1e9)))
                if sub[vowel_col].nunique() < 5:
                    continue
                ml = lobanov(sub.copy()).groupby(vowel_col)[["F1_bark_n", "F2_bark_n"]].median()
                ms = shrink_normalize(sub.copy(), prior).groupby(vowel_col)[["F1_bark_s", "F2_bark_s"]].median()
                ms.columns = ["F1_bark_n", "F2_bark_n"]
                common = base.index.intersection(ml.index)
                if len(common) < 5:
                    continue
                err_l.append(np.nanmean(np.abs(ml.loc[common] - base.loc[common]).values))
                common2 = base.index.intersection(ms.index)
                err_s.append(np.nanmean(np.abs(ms.loc[common2] - base.loc[common2]).values))
            if err_l:
                rows.append({"speaker": spk, "n": n,
                             "lobanov": float(np.nanmean(err_l)),
                             "shrinkage": float(np.nanmean(err_s))})
    return pd.DataFrame(rows)


def distance_stability(df: pd.DataFrame, prior, ns=(10, 20, 30, 50),
                       n_rep=40, seed=1, pair=("ɪ", "iː"), k=50.0):
    """以"/ɪ/-/iː/ 距离"为下游指标, 比较两种归一化在小样本下的稳定性.

    对每位说话人:
      1) 用全样本算距离(各方法各自的标准)
      2) 抽 n 个 token 重算, 记录与全样本的绝对偏差
    稳定的方法 = 偏差小.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for spk, g in df.groupby("speaker"):
        if not ({pair[0], pair[1]} <= set(g["vowel"].unique())):
            continue
        full = {}
        for name, fn in [("lobanov", lambda d: lobanov(d)),
                         ("shrinkage", lambda d: shrink_normalize(d, prior, k=k))]:
            dd = fn(g.copy())
            med = dd.groupby("vowel")[[f + "_n" for f in FORMS]].median() \
                if name == "lobanov" else dd.groupby("vowel")[[f + "_s" for f in FORMS]].median()
            med.columns = [f + "_n" for f in FORMS]
            full[name] = float(np.linalg.norm(med.loc[pair[0]] - med.loc[pair[1]]))
        for n in ns:
            if len(g) < n:
                continue
            dev = {"lobanov": [], "shrinkage": []}
            for _ in range(n_rep):
                sub = g.sample(n=n, random_state=int(rng.integers(1e9)))
                if not ({pair[0], pair[1]} <= set(sub["vowel"].unique())):
                    continue
                for name, fn in [("lobanov", lambda d: lobanov(d)),
                                 ("shrinkage", lambda d: shrink_normalize(d, prior, k=k))]:
                    dd = fn(sub.copy())
                    suf = "_n" if name == "lobanov" else "_s"
                    med = dd.groupby("vowel")[[f + suf for f in FORMS]].median()
                    med.columns = [f + "_n" for f in FORMS]
                    dist = float(np.linalg.norm(med.loc[pair[0]] - med.loc[pair[1]]))
                    dev[name].append(abs(dist - full[name]))
            if dev["lobanov"]:
                rows.append({"speaker": spk, "n": n,
                             "lobanov": float(np.mean(dev["lobanov"])),
                             "shrinkage": float(np.mean(dev["shrinkage"]))})
    return pd.DataFrame(rows)
