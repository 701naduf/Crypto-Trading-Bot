"""
波动率目标 (Vol Targeting)

动态调整组合杠杆，使组合的预期年化波动率稳定在目标水平。

原理:
    realized_vol = 组合最近 N 行的年化波动率
    scale_factor = target_vol / realized_vol
    adjusted_weights = original_weights * scale_factor

    波动率高于目标 → 缩小仓位
    波动率低于目标 → 放大仓位

受 leverage_cap 约束：scale_factor 不超过使 Σ|w| = leverage_cap 的值。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_model.config import MINUTES_PER_YEAR


def apply_vol_target(
    weights: pd.DataFrame,
    price_panel: pd.DataFrame,
    vol_target: float = 0.15,
    lookback: int = 60,
    leverage_cap: float = 2.0,
) -> pd.DataFrame:
    """
    波动率目标调整

    对每个时刻的权重进行缩放，使组合预期年化波动率接近 vol_target。

    Args:
        weights:      目标权重面板 (timestamp × symbol)
        price_panel:  价格面板 (timestamp × symbol)，用于计算组合收益率
        vol_target:   年化波动率目标（如 0.15 = 15%）
        lookback:     波动率估计窗口（bar 数）
        leverage_cap: 最大杠杆倍数

    Returns:
        调整后的权重面板
    """
    if vol_target <= 0:
        raise ValueError(f"vol_target 必须 > 0, 收到 {vol_target}")

    # 计算收益率
    returns_panel = price_panel.pct_change().reindex(weights.index)

    # 计算组合收益率: r_p = Σ(w_{i,t-1} × r_{i,t})
    # 使用滞后一期的权重
    shifted_weights = weights.shift(1)
    portfolio_returns = (shifted_weights * returns_panel).sum(axis=1)

    # 滚动年化波动率
    rolling_vol = portfolio_returns.rolling(
        window=lookback, min_periods=max(lookback // 2, 2),
    ).std() * np.sqrt(MINUTES_PER_YEAR)

    # 缩放因子
    scale = vol_target / rolling_vol.replace(0, np.nan)

    # 限制缩放因子，不超过 leverage_cap
    adjusted = weights.copy()
    for i, ts in enumerate(weights.index):
        if pd.isna(scale.iloc[i]) or np.isinf(scale.iloc[i]):
            continue

        s = scale.iloc[i]
        w = weights.iloc[i].values
        scaled_w = w * s

        # 检查杠杆上限
        current_leverage = np.nansum(np.abs(scaled_w))
        if current_leverage > leverage_cap:
            # 进一步缩放以满足杠杆上限
            scaled_w = scaled_w * (leverage_cap / current_leverage)

        adjusted.iloc[i] = scaled_w

    return adjusted
