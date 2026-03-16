"""
向量化回测

用目标权重和价格面板直接计算 P&L。
用途: 快速筛选策略，不替代 Phase 3 的事件驱动回测。

P&L 计算:
    portfolio_return_t = Σ(w_{i,t-1} × r_{i,t})
    其中 w 是 t-1 时刻的目标权重，r 是 t 时刻的实际收益

交易成本模型（两层）:
    1. 固定手续费: fee = turnover × fee_rate  (通常 0.04% taker)
    2. 市场冲击 (size-dependent):
       impact_i = impact_coeff × σ_i × sqrt(|Δw_i| × portfolio_value / ADV_i)
       这是 Almgren-Chriss 模型的简化版（square-root model）。

    total_cost_t = Σ(|Δw_i,t| × fee_rate + impact_i,t)
    net_return_t = portfolio_return_t - total_cost_t

为什么需要市场冲击模型？
    固定费率假设忽略了交易量对价格的影响。
    对于流动性较差的标的（如 DOGE/USDT），大单交易会显著推高成本。
    sqrt-model 是量化行业最常用的市场冲击估计方法。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from alpha_model.backtest.performance import BacktestResult

from alpha_model.config import (
    DEFAULT_FEE_RATE,
    DEFAULT_IMPACT_COEFF,
    DEFAULT_PORTFOLIO_VALUE,
    MINUTES_PER_YEAR,
)

from factor_research.evaluation.metrics import cumulative_returns

logger = logging.getLogger(__name__)


def estimate_market_impact(
    delta_weights: pd.DataFrame,
    adv_panel: pd.DataFrame,
    volatility_panel: pd.DataFrame,
    portfolio_value: float,
    impact_coeff: float = DEFAULT_IMPACT_COEFF,
) -> pd.DataFrame:
    """
    Square-root 市场冲击估计

    公式: impact_i = impact_coeff × σ_i × sqrt(|ΔV_i| / ADV_i)
    其中 ΔV_i = |Δw_i| × portfolio_value

    Args:
        delta_weights:    权重变化面板 (timestamp × symbol)
        adv_panel:        日均成交量面板 (timestamp × symbol, USDT)
        volatility_panel: 滚动波动率面板 (timestamp × symbol)
        portfolio_value:  组合总资金
        impact_coeff:     冲击系数

    Returns:
        市场冲击成本面板 (timestamp × symbol)，单位为收益率
    """
    # 交易金额: |Δw_i| × portfolio_value
    trade_value = delta_weights.abs() * portfolio_value

    # 参与率: trade_value / ADV
    participation = trade_value / adv_panel.replace(0, np.nan)

    # 诊断: ADV=0 意味着该标的无流动性，冲击无法估计
    zero_adv_ratio = (adv_panel == 0).sum() / max(len(adv_panel), 1)
    for sym in zero_adv_ratio.index:
        if zero_adv_ratio[sym] > 0.1:
            logger.warning(
                "标的 '%s' ADV=0 占比 %.1f%%，流动性不足，市场冲击估计不可靠",
                sym, zero_adv_ratio[sym] * 100,
            )

    # 冲击: coeff × σ × √participation
    impact = impact_coeff * volatility_panel * np.sqrt(participation.clip(lower=0))

    return impact.fillna(0)


def vectorized_backtest(
    weights: pd.DataFrame,
    price_panel: pd.DataFrame,
    fee_rate: float = DEFAULT_FEE_RATE,
    impact_coeff: float = DEFAULT_IMPACT_COEFF,
    adv_panel: pd.DataFrame | None = None,
    portfolio_value: float = DEFAULT_PORTFOLIO_VALUE,
) -> BacktestResult:
    """
    向量化回测

    Args:
        weights:         目标权重面板 (timestamp × symbol)
        price_panel:     价格面板 (timestamp × symbol)
        fee_rate:        单边手续费率（0.0004 = 0.04% Binance taker）
        impact_coeff:    市场冲击系数（默认 0.1，可根据回测校准）
        adv_panel:       日均成交量面板 (timestamp × symbol, 单位 USDT)
                         None 则退化为纯手续费模型（无市场冲击）
        portfolio_value: 组合总资金（用于将权重转为实际交易金额）

    Returns:
        BacktestResult
    """
    # 对齐时间索引
    common_idx = weights.index.intersection(price_panel.index)
    if len(common_idx) == 0:
        raise ValueError("weights 和 price_panel 没有共同时间索引")

    symbols = weights.columns.tolist()
    w = weights.reindex(common_idx)[symbols]
    prices = price_panel.reindex(common_idx)[symbols]

    # 收益率
    returns_panel = prices.pct_change()

    # 组合毛收益: r_p = Σ(w_{i,t-1} × r_{i,t})
    shifted_w = w.shift(1)
    portfolio_gross = (shifted_w * returns_panel).sum(axis=1)

    # 换手率: turnover_t = Σ|Δw_{i,t}|
    delta_w = w.diff()
    turnover = delta_w.abs().sum(axis=1)

    # --- 交易成本 ---
    # 1. 固定手续费
    fee_cost = turnover * fee_rate

    # 2. 市场冲击（如果提供了 ADV）
    impact_cost = pd.Series(0.0, index=common_idx)
    if adv_panel is not None:
        # 滚动波动率（60 bar）
        vol_panel = returns_panel.rolling(60, min_periods=10).std() * np.sqrt(MINUTES_PER_YEAR)
        adv_aligned = adv_panel.reindex(common_idx)[symbols]

        impact = estimate_market_impact(
            delta_w, adv_aligned, vol_panel,
            portfolio_value, impact_coeff,
        )
        impact_cost = impact.sum(axis=1)

    total_cost_series = fee_cost + impact_cost
    total_cost_scalar = total_cost_series.sum()

    # 净收益
    net_returns = portfolio_gross - total_cost_series

    # 净值曲线
    equity = cumulative_returns(net_returns.dropna())

    return BacktestResult(
        equity_curve=equity,
        returns=net_returns,
        turnover=turnover,
        weights_history=w,
        gross_returns=portfolio_gross,
        total_cost=total_cost_scalar,
    )
