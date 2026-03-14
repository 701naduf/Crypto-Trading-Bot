"""
非线性分析模块

评价体系的第二层 API——非线性关系分析。

为什么需要非线性分析:
    IC (Spearman 相关) 只能捕捉单调关系。
    部分因子（如波动率因子）与收益存在 U 形或其他非线性关系:
        - 低波动 → 正收益（均值回复）
        - 高波动 → 正收益（趋势延续）
        - 中等波动 → 无明显收益
    此时 IC ≈ 0，但因子实际上有预测力。

分析工具:
    1. 互信息 (MI): 捕捉任意统计依赖，不限于单调
    2. 因子特征曲线: LOWESS 非参数回归拟合 factor_value → return
    3. 条件 IC（分段）: factor 值低/中/高三段各自的 IC
    4. 分段收益: 因子各分位区间的平均收益

依赖: evaluation.metrics, scipy, statsmodels (可选)
"""

import numpy as np
import pandas as pd

from .metrics import (
    compute_forward_returns_panel,
    mutual_information,
    spearman_ic,
)


def nonlinear_analysis(
    factor_panel: pd.DataFrame,
    price_panel: pd.DataFrame,
    horizon: int = 1,
    n_bins: int = 10,
) -> dict:
    """
    非线性分析（第二层 API 入口）

    Args:
        factor_panel: 因子面板
        price_panel:  价格面板
        horizon:      前瞻窗口（bar 数）
        n_bins:       分段数量（用于因子特征曲线和条件 IC）

    Returns:
        dict: {
            "mutual_info":    互信息值（全局非线性依赖强度）,
            "factor_profile": 因子特征曲线 DataFrame（分段的 factor → return 关系）,
            "conditional_ic": 条件 IC dict（低/中/高三段各自的 IC）,
            "bin_returns":    各分段的平均收益 DataFrame,
        }
    """
    forward_ret = compute_forward_returns_panel(price_panel, horizon)

    # 对齐
    common_idx = factor_panel.index.intersection(forward_ret.index)
    common_cols = factor_panel.columns.intersection(forward_ret.columns)
    f = factor_panel.loc[common_idx, common_cols]
    r = forward_ret.loc[common_idx, common_cols]

    # 展平
    f_flat = f.stack().dropna()
    r_flat = r.stack().dropna()
    common_flat = f_flat.index.intersection(r_flat.index)
    f_flat = f_flat.loc[common_flat]
    r_flat = r_flat.loc[common_flat]

    if len(f_flat) < 20:
        return _empty_nonlinear_result()

    # 1. 互信息
    mi = mutual_information(f_flat.values, r_flat.values)

    # 2. 因子特征曲线（分箱版本，不依赖 LOWESS）
    profile = _factor_profile(f_flat, r_flat, n_bins=n_bins)

    # 3. 条件 IC（三段）
    cond_ic = _conditional_ic(f_flat, r_flat)

    # 4. 各段收益
    bin_ret = _bin_returns(f_flat, r_flat, n_bins=n_bins)

    return {
        "mutual_info": float(mi) if not np.isnan(mi) else np.nan,
        "factor_profile": profile,
        "conditional_ic": cond_ic,
        "bin_returns": bin_ret,
    }


def _factor_profile(
    factor: pd.Series,
    returns: pd.Series,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    因子特征曲线（分箱版本）

    将因子值等频分为 n_bins 组，计算每组的:
        - 因子值中位数
        - 收益均值
        - 收益标准差
        - 观测数量

    这是 LOWESS 的简化替代方案，不需要 statsmodels 依赖，
    且对 outlier 更稳健。

    Returns:
        pd.DataFrame: index=bin (1..n_bins), columns=[factor_median, return_mean, return_std, count]
    """
    # 等频分箱
    try:
        labels = pd.qcut(factor, q=n_bins, labels=False, duplicates="drop") + 1
    except ValueError:
        # 因子值分布太集中，无法等频分箱
        return pd.DataFrame(columns=["factor_median", "return_mean", "return_std", "count"])

    df = pd.DataFrame({"factor": factor.values, "return": returns.values, "bin": labels.values})

    result = df.groupby("bin").agg(
        factor_median=("factor", "median"),
        return_mean=("return", "mean"),
        return_std=("return", "std"),
        count=("return", "count"),
    )

    return result


def _conditional_ic(
    factor: pd.Series,
    returns: pd.Series,
) -> dict:
    """
    条件 IC（三段分析）

    将因子值按 30/70 分位数分为低、中、高三段，
    分别计算每段内的 IC。

    如果三段 IC 符号不同 → 因子存在非线性关系。
    例如: 低段 IC > 0, 高段 IC > 0, 中段 IC ≈ 0 → U 形关系。

    Returns:
        dict: {
            "low_ic":  低段 IC（factor < p30）,
            "mid_ic":  中段 IC（p30 ≤ factor ≤ p70）,
            "high_ic": 高段 IC（factor > p70）,
        }
    """
    p30 = factor.quantile(0.3)
    p70 = factor.quantile(0.7)

    low_mask = factor < p30
    mid_mask = (factor >= p30) & (factor <= p70)
    high_mask = factor > p70

    return {
        "low_ic": spearman_ic(factor[low_mask], returns[low_mask]),
        "mid_ic": spearman_ic(factor[mid_mask], returns[mid_mask]),
        "high_ic": spearman_ic(factor[high_mask], returns[high_mask]),
    }


def _bin_returns(
    factor: pd.Series,
    returns: pd.Series,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    各段平均收益

    将因子值等频分箱，计算每箱的平均收益和命中率。

    Returns:
        pd.DataFrame: index=bin, columns=[return_mean, hit_rate, count]
    """
    try:
        labels = pd.qcut(factor, q=n_bins, labels=False, duplicates="drop") + 1
    except ValueError:
        return pd.DataFrame(columns=["return_mean", "hit_rate", "count"])

    df = pd.DataFrame({"factor": factor.values, "return": returns.values, "bin": labels.values})

    result = df.groupby("bin").agg(
        return_mean=("return", "mean"),
        count=("return", "count"),
    )
    # 命中率: return 方向与 factor 方向一致的比例
    hits = df.groupby("bin").apply(
        lambda g: (np.sign(g["factor"]) == np.sign(g["return"])).mean(),
        include_groups=False,
    )
    result["hit_rate"] = hits

    return result


def _empty_nonlinear_result() -> dict:
    """空数据时返回的默认结果"""
    return {
        "mutual_info": np.nan,
        "factor_profile": pd.DataFrame(columns=["factor_median", "return_mean", "return_std", "count"]),
        "conditional_ic": {"low_ic": np.nan, "mid_ic": np.nan, "high_ic": np.nan},
        "bin_returns": pd.DataFrame(columns=["return_mean", "hit_rate", "count"]),
    }
