"""情境 18 分析工具 —— 單調性 / AUC / 相關 / 複合異常分數。

統計呈現規格(config `statistics` 節):
  * 所有 AUC 附 bootstrap 95% CI(train LO 僅 65 段,點估計不可單獨呈現)
  * 單調性用 permutation p 值(不依賴常態或大樣本近似)
  * 所有函數回傳 n,供表格強制標註

NaN policy(config `nan_policy`):複合分數的 top-3 選擇只在**非 NaN**特徵中進行,
分母隨實際可用特徵數調整;可用特徵不足 `min_available_features` 時該 run 為 NaN。
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

LEVEL_ORDER = ["LN", "LO", "MED", "HI"]
LEVEL_ORDINAL = {lab: i for i, lab in enumerate(LEVEL_ORDER)}


# --------------------------------------------------------------------- AUC


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann–Whitney U 形式的 AUC(對 tie 取 0.5)。"""
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    u = stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u / (len(pos) * len(neg)))


def auc_with_ci(pos, neg, n_boot: int = 2000, seed: int = 42,
                alpha: float = 0.05) -> Dict[str, float]:
    """AUC + bootstrap 百分位 CI。兩組各自重抽,保留原始組別大小。"""
    pos = np.asarray(pd.Series(pos).dropna(), dtype=float)
    neg = np.asarray(pd.Series(neg).dropna(), dtype=float)
    point = auc(pos, neg)
    if not np.isfinite(point) or len(pos) < 2 or len(neg) < 2:
        return {"auc": point, "ci_low": float("nan"), "ci_high": float("nan"),
                "n_pos": int(len(pos)), "n_neg": int(len(neg))}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = auc(rng.choice(pos, len(pos), replace=True),
                       rng.choice(neg, len(neg), replace=True))
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"auc": point, "ci_low": float(lo), "ci_high": float(hi),
            "n_pos": int(len(pos)), "n_neg": int(len(neg))}


# ------------------------------------------------------------- 單調性 / 趨勢


def spearman_permutation(x, y, n_perm: int = 2000, seed: int = 42) -> Dict[str, float]:
    """Spearman rho + permutation p(雙尾)。x/y 對齊後才丟進來。"""
    s = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(s) < 3:
        return {"rho": float("nan"), "p_perm": float("nan"), "n": int(len(s))}
    xv, yv = s["x"].to_numpy(), s["y"].to_numpy()
    rho = float(stats.spearmanr(xv, yv).statistic)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = stats.spearmanr(xv, rng.permutation(yv)).statistic
    p = float((np.sum(np.abs(null) >= abs(rho)) + 1) / (n_perm + 1))
    return {"rho": rho, "p_perm": p, "n": int(len(s))}


def monotonicity(df: pd.DataFrame, feature: str, label_col: str = "ylabel",
                 n_perm: int = 2000, seed: int = 42) -> Dict[str, float]:
    """特徵對 LN<LO<MED<HI 序位的趨勢檢定(Spearman + permutation p)。"""
    sub = df[[feature, label_col]].dropna()
    ordinal = sub[label_col].map(LEVEL_ORDINAL)
    res = spearman_permutation(sub[feature], ordinal, n_perm=n_perm, seed=seed)
    res["direction"] = "up" if res["rho"] > 0 else "down"
    res["coverage"] = float(len(sub) / len(df)) if len(df) else float("nan")
    return res


def partial_spearman(df: pd.DataFrame, feature: str, target: str,
                     covariate: str, n_perm: int = 2000,
                     seed: int = 42) -> Dict[str, float]:
    """控制 `covariate` 後,`feature` 與 `target` 的偏相關(Spearman 殘差法)。

    做法:三者各自轉秩,feature 與 target 分別對 covariate 秩做線性回歸取殘差,
    再算殘差間的 Spearman + permutation p。
    """
    s = df[[feature, target, covariate]].dropna()
    if len(s) < 5:
        return {"rho_partial": float("nan"), "p_perm": float("nan"),
                "n": int(len(s))}
    r = s.rank()
    c = r[covariate].to_numpy()
    design = np.column_stack([np.ones_like(c), c])

    def resid(col):
        y = r[col].to_numpy()
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        return y - design @ beta

    out = spearman_permutation(resid(feature), resid(target),
                               n_perm=n_perm, seed=seed)
    return {"rho_partial": out["rho"], "p_perm": out["p_perm"], "n": out["n"]}


# --------------------------------------------------- 健康基線 + 複合異常分數


def ln_baseline(train_df: pd.DataFrame, features: Sequence[str],
                label_col: str = "ylabel") -> pd.DataFrame:
    """基線統計**只從 train 的 LN 段**計算(防洩漏鐵律 2)。"""
    ln = train_df[train_df[label_col] == "LN"]
    if ln.empty:
        raise ValueError("train 中找不到 LN 段,無法建基線")
    return pd.DataFrame({
        "mean": ln[list(features)].mean(),
        "std": ln[list(features)].std(ddof=1),
        "n": ln[list(features)].notna().sum(),
    })


def zscores(df: pd.DataFrame, baseline: pd.DataFrame,
            features: Sequence[str]) -> pd.DataFrame:
    """對每個 run 算各特徵的 z-score(NaN 保持 NaN,不填補)。"""
    out = {}
    for f in features:
        sd = baseline.loc[f, "std"]
        out[f] = (df[f] - baseline.loc[f, "mean"]) / sd if sd > 0 else np.nan
    return pd.DataFrame(out, index=df.index)


def composite_score(z: pd.DataFrame, top_k: int = 3,
                    min_available: int = 3) -> pd.Series:
    """複合分數 = mean(top-k |z|),只在非 NaN 特徵中選 top-k。

    可用特徵數 < `min_available` 時回傳 NaN(不以較少特徵硬算)。
    """
    a = z.abs().to_numpy()
    avail = np.sum(~np.isnan(a), axis=1)
    out = np.full(len(z), np.nan)
    for i in range(len(z)):
        if avail[i] < min_available:
            continue
        row = a[i][~np.isnan(a[i])]
        k = min(top_k, len(row))
        out[i] = float(np.mean(np.sort(row)[::-1][:k]))
    return pd.Series(out, index=z.index, name="composite")


def within_noisy(df: pd.DataFrame, features: Sequence[str],
                 label_col: str = "ylabel", n_perm: int = 2000,
                 n_boot: int = 2000, seed: int = 42) -> Dict[str, Dict]:
    """**主要推論軌**:只在 LO/MED/HI 三類內做趨勢與兩兩 AUC。

    LN 來自零負載檔案,與其餘三類差了一個負載條件;排除 LN 後,
    三類同屬 `*_noisy`,負載條件一致,故此軌無負載混淆。
    """
    levels = ["LO", "MED", "HI"]
    sub = df[df[label_col].isin(levels)]
    ordinal = sub[label_col].map({l: i for i, l in enumerate(levels)})
    out: Dict[str, Dict] = {}
    for f in features:
        s = pd.DataFrame({"f": sub[f], "o": ordinal}).dropna()
        trend = spearman_permutation(s["f"], s["o"], n_perm=n_perm, seed=seed)
        pairs = {}
        for i in range(len(levels)):
            for j in range(i + 1, len(levels)):
                a, b = levels[i], levels[j]
                pairs[f"{a}_vs_{b}"] = auc_with_ci(
                    sub[sub[label_col] == b][f], sub[sub[label_col] == a][f],
                    n_boot=n_boot, seed=seed)
        out[f] = {"trend": trend, "pairwise_auc": pairs,
                  "n_by_level": sub.groupby(label_col)[f].count()
                                   .reindex(levels).fillna(0).astype(int).to_dict()}
    return out


def trigger_rates(scores: pd.Series, labels: pd.Series,
                  threshold: float) -> pd.DataFrame:
    """各等級的觸發率(複合分數 > 門檻)。"""
    d = pd.DataFrame({"s": scores, "y": labels}).dropna()
    g = d.groupby("y")["s"]
    return pd.DataFrame({
        "n": g.size(),
        "n_triggered": g.apply(lambda x: int((x > threshold).sum())),
        "trigger_rate": g.apply(lambda x: float((x > threshold).mean())),
    }).reindex([l for l in LEVEL_ORDER if l in d["y"].unique()])
