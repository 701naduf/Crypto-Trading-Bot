"""
滚动 Beta 估计

估计每个标的相对于市场（BTC/USDT）的时变 beta。
用于 beta-neutral 约束。

方法: OLS 滚动回归
    r_i,t = alpha + beta_i × r_BTC,t + epsilon
    使用最近 lookback 行的收益率
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_beta(
    returns_panel: pd.DataFrame,
    market_symbol: str = "BTC/USDT",
    lookback: int = 60,
) -> pd.DataFrame:
    """
    滚动 beta 估计

    对每个标的，用 OLS 回归估计其相对于市场标的的时变 beta:
        r_i,t = alpha_i + beta_i × r_market,t + epsilon

    beta_i = Cov(r_i, r_market) / Var(r_market)

    Args:
        returns_panel:  收益率面板 (timestamp × symbol)
        market_symbol:  市场标的（beta 的基准），默认 BTC/USDT
        lookback:       滚动窗口大小（bar 数）

    Returns:
        beta 面板 (timestamp × symbol)
        市场标的自身的 beta 恒为 1.0

    Raises:
        ValueError: market_symbol 不在 returns_panel 中
    """
    if market_symbol not in returns_panel.columns:
        raise ValueError(
            f"market_symbol '{market_symbol}' 不在 returns_panel 的列中。"
            f"可用: {returns_panel.columns.tolist()}"
        )

    r_market = returns_panel[market_symbol]
    market_var = r_market.rolling(window=lookback, min_periods=lookback).var()

    beta_dict = {}
    for symbol in returns_panel.columns:
        if symbol == market_symbol:
            # 市场自身的 beta 恒为 1.0
            beta_dict[symbol] = pd.Series(1.0, index=returns_panel.index)
        else:
            r_asset = returns_panel[symbol]
            cov = r_asset.rolling(window=lookback, min_periods=lookback).cov(r_market)
            # beta = cov / var，var==0 时返回 NaN
            beta_dict[symbol] = cov / market_var.replace(0, np.nan)

    return pd.DataFrame(beta_dict, index=returns_panel.index)
