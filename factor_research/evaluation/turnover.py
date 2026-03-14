"""
换手率分析模块

评价体系的第二层 API——因子换手率分析。

为什么关心换手率:
    因子值的变化速度直接决定了策略的交易频率。
    高换手 → 频繁调仓 → 高交易成本 → 可能吃掉 alpha。
    低换手 → 信号稳定 → 低交易成本 → 但可能反应迟钝。

    在 crypto 市场中，交易成本（maker/taker fee + 滑点）
    是影响策略净收益的关键因素。

分析维度:
    1. 因子自相关: 相邻两期因子值的相关性（信号持续性）
    2. 排名变化率: 截面排名的变化速度
    3. 信号翻转频率: 因子符号（方向）的改变频率

依赖: evaluation.metrics
"""

import numpy as np
import pandas as pd


def turnover_analysis(
    factor_panel: pd.DataFrame,
) -> dict:
    """
    换手率分析（第二层 API 入口）

    Args:
        factor_panel: 因子面板 (index=timestamp, columns=symbols)

    Returns:
        dict: {
            "autocorrelation":    因子自相关（各标的的均值）,
            "rank_change_rate":   排名变化率（各标的的均值）,
            "signal_flip_rate":   信号翻转频率（各标的的均值）,
            "autocorr_by_symbol": 按标的的自相关 dict,
            "flip_by_symbol":     按标的的翻转频率 dict,
        }
    """
    if factor_panel.empty or len(factor_panel) < 3:
        return _empty_turnover_result()

    # 1. 因子自相关
    autocorr_by_symbol = {}
    for col in factor_panel.columns:
        series = factor_panel[col].dropna()
        if len(series) > 1:
            autocorr_by_symbol[col] = float(series.autocorr(lag=1))
        else:
            autocorr_by_symbol[col] = np.nan

    avg_autocorr = np.nanmean(list(autocorr_by_symbol.values()))

    # 2. 排名变化率: |rank_t - rank_{t-1}| / N
    ranks = factor_panel.rank(axis=1, pct=True)
    rank_diff = ranks.diff().abs()
    n_symbols = len(factor_panel.columns)
    rank_change = rank_diff.mean(axis=1).mean() if n_symbols > 0 else np.nan

    # 3. 信号翻转频率: sign 变化次数 / 总 bar 数
    flip_by_symbol = {}
    for col in factor_panel.columns:
        series = factor_panel[col].dropna()
        if len(series) > 1:
            signs = np.sign(series)
            changes = (signs != signs.shift(1)).sum() - 1  # 第一个 NaN 不算
            flip_by_symbol[col] = float(max(0, changes) / (len(series) - 1))
        else:
            flip_by_symbol[col] = np.nan

    avg_flip = np.nanmean(list(flip_by_symbol.values()))

    return {
        "autocorrelation": float(avg_autocorr) if not np.isnan(avg_autocorr) else np.nan,
        "rank_change_rate": float(rank_change) if not np.isnan(rank_change) else np.nan,
        "signal_flip_rate": float(avg_flip) if not np.isnan(avg_flip) else np.nan,
        "autocorr_by_symbol": autocorr_by_symbol,
        "flip_by_symbol": flip_by_symbol,
    }


def _empty_turnover_result() -> dict:
    """空数据时返回的默认结果"""
    return {
        "autocorrelation": np.nan,
        "rank_change_rate": np.nan,
        "signal_flip_rate": np.nan,
        "autocorr_by_symbol": {},
        "flip_by_symbol": {},
    }
