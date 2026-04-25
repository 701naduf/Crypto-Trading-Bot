"""
向量化回测

用目标权重和价格面板直接计算 P&L。
用途: 快速筛选策略，不替代 Phase 3 的事件驱动回测。

P&L 计算:
    portfolio_return_t = Σ(w_{i,t-1} × r_{i,t})
    其中 w 是 t-1 时刻的目标权重，r 是 t 时刻的实际收益

交易成本模型（三层，与 execution_optimizer.cost 口径一致）:
    1. 固定手续费: fee = turnover × fee_rate  (通常 0.04% taker)
    2. 买卖价差: spread_i = spread_i / 2 × |Δw_i|          （需传 spread_panel）
    3. 市场冲击 (size-dependent, Almgren-Chriss 总冲击成本):
       边际冲击率 marginal_impact(q) = σ × √(q/ADV)
       总成本 = ∫₀^Q σ × √(q/ADV) dq = (2/3) × σ × Q^1.5 / √ADV
       代入 Q = |Δw_i| × V 并 ÷ V 归一化为收益率空间:
       impact_i = (2/3) × impact_coeff × σ_i × √(V/ADV_i) × |Δw_i|^1.5

    total_cost_t = Σ(|Δw_i,t| × fee_rate) + spread_cost_t + impact_t
    net_return_t = portfolio_return_t - total_cost_t

    spread 和 impact 分量各自可选（仅传入对应面板时启用），支持"仅 fee"、
    "fee+impact"（Phase 2b 旧行为）、"完整三分量"三种模式。

为什么需要市场冲击模型？
    固定费率假设忽略了交易量对价格的影响。
    对于流动性较差的标的（如 DOGE/USDT），大单交易会显著推高成本。
    sqrt-model 是量化行业最常用的市场冲击估计方法。

2/3 系数:
    来自边际冲击率对 q 的积分，显式保留以使 impact_coeff 成为纯校准量
    （从成交数据回归时与 Almgren-Chriss 文献系数直接对应）。
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
    Almgren-Chriss 总冲击成本估计（收益率空间）

    公式: impact_i = (2/3) × impact_coeff × σ_i × √(V / ADV_i) × |Δw_i|^1.5

    ADV 兜底（Z4 + Z12，跨四处统一）:
      ADV < 1.0 USD 或 NaN 时强制按 ADV=1.0 计算，与 execution_optimizer.cost /
      backtest_engine.Rebalancer / backtest_engine.attribution.compute_per_symbol_cost
      四处兜底策略**完全一致**。NaN 触发 logger.warning（Option B：silent + warning，
      与 N1 funding NaN 风格一致）。

    NaN 处理（Z6 + Z8）:
      vol_panel warmup 期 NaN（rolling std min_periods 不足）和 delta_weights 首行
      NaN（pandas diff 首行恒 NaN）经 fillna(0) 显式归零。这是合法 NaN，但 fillna(0)
      会**低估成本**（impact=0 → ed gain 更高 → 乐观偏差，**非保守**）；warmup 占比
      ~0.07%（10 bar / 30240 bar），影响微小。用户应在 21 天热身期外评估 P&L 主体。

      与 PnLTracker.record 的 sum(skipna=False) 防御**不同源**：
        - 此处 NaN：合法的"未启动状态"
        - PnLTracker NaN：运行时数据/公式 bug → fail-fast

    Args:
        delta_weights:    权重变化面板 (timestamp × symbol)
        adv_panel:        日均成交量面板 (timestamp × symbol, USDT)
        volatility_panel: 滚动波动率面板 (timestamp × symbol)，**日化** σ（非年化）。
                          Almgren-Chriss 约定 σ 单位为 1/√day，与 ADV 日级尺度配套。
                          传入年化 σ 会导致 impact 被高估 √365.25 ≈ 19 倍。
        portfolio_value:  组合总资金
        impact_coeff:     冲击系数（纯校准量，不含 2/3 prefactor）

    Returns:
        市场冲击成本面板 (timestamp × symbol)，单位为收益率
    """
    from alpha_model.backtest.adv_helpers import safe_adv_panel

    # ADV 兜底（Z4/Z12 跨四处统一）
    adv_safe = safe_adv_panel(adv_panel, context="estimate_market_impact")

    # Almgren-Chriss 总冲击成本: (2/3) × coeff × σ × √(V/ADV) × |Δw|^1.5
    impact = (2.0 / 3.0) * impact_coeff * volatility_panel \
             * np.sqrt(portfolio_value / adv_safe) \
             * delta_weights.abs().pow(1.5)

    # vol_panel warmup 期 NaN + delta_weights 首行 NaN 经 fillna(0) 显式归零（Z6/Z8）
    return impact.fillna(0)


def vectorized_backtest(
    weights: pd.DataFrame,
    price_panel: pd.DataFrame,
    fee_rate: float = DEFAULT_FEE_RATE,
    impact_coeff: float = DEFAULT_IMPACT_COEFF,
    adv_panel: pd.DataFrame | None = None,
    spread_panel: pd.DataFrame | None = None,
    vol_panel: pd.DataFrame | None = None,
    portfolio_value: float = DEFAULT_PORTFOLIO_VALUE,
    periods_per_year: float = MINUTES_PER_YEAR,
) -> BacktestResult:
    """
    向量化回测

    交易成本三分量（与 execution_optimizer.cost 口径一致）:
        1. 手续费  fee      = fee_rate × Σ|Δw|
        2. 买卖价差 spread   = Σ(spread_i/2 × |Δw_i|)    ← 需传入 spread_panel
        3. 市场冲击 impact   = (2/3) × coeff × σ × √(V/ADV) × |Δw|^1.5   ← 需传入 adv_panel

    spread_panel 和 adv_panel 分别独立可选，支持三种组合：
        - 仅 fee（spread/adv 都不传）：最快，但会显著低估成本
        - fee + impact（只传 adv）：Phase 2b 历史行为，向后兼容
        - fee + spread + impact（都传）：与 execution_optimizer 三分量一致，
          适合 Phase 3 事件驱动回测的"完美执行 ≈ 向量化"可比性校验

    Args:
        weights:          目标权重面板 (timestamp × symbol)
        price_panel:      价格面板 (timestamp × symbol)
        fee_rate:         单边手续费率（0.0004 = 0.04% Binance taker）
        impact_coeff:     市场冲击系数（默认 0.1，可根据回测校准）
        adv_panel:        日均成交量面板 (timestamp × symbol, 单位 USDT)
                          None 则退化为纯手续费模型（无市场冲击）
        spread_panel:     买卖价差面板 (timestamp × symbol)，比率形式 (ask-bid)/mid
                          None 则不计入价差成本（默认，向后兼容 Phase 2b 旧行为）
        vol_panel:        日化 σ 面板 (timestamp × symbol)。
                          - None（默认）：内部用 rolling(60, min_periods=10).std()
                            × √bars_per_day 估算 60-min σ
                          - 非 None：尊重外部 σ，跳过内部计算（用于 Phase 3 跨模式
                            σ 一致性，避免 vectorized 60-min σ 与事件驱动模式
                            20-day σ 不一致）
                          ⚠️ 两种用法窗口长度差 1-2 个数量级，impact 估计随之差
                             √480 ≈ 22 倍。用户应在同一回测内一致用法。
        portfolio_value:  组合总资金（用于将权重转为实际交易金额）
        periods_per_year: 年化系数。默认 `MINUTES_PER_YEAR` (525960) 对应 1m bar。
                          内部用于把 bar 级波动率转为 impact 公式所需的**日化** σ
                          （`bars_per_day = periods_per_year / 365.25`）。
                          其他频率：5m→105192, 15m→35064, 1h→8766。

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
    # 1. 固定手续费: fee_rate × Σ|Δw|
    fee_cost = turnover * fee_rate

    # 2. 买卖价差: Σ(spread_i/2 × |Δw_i|)（与 execution_optimizer.cost 一致）
    spread_cost = pd.Series(0.0, index=common_idx)
    if spread_panel is not None:
        spread_aligned = spread_panel.reindex(common_idx)[symbols]
        spread_cost = (spread_aligned / 2.0 * delta_w.abs()).sum(axis=1).fillna(0.0)

    # 3. 市场冲击（如果提供了 ADV）
    impact_cost = pd.Series(0.0, index=common_idx)
    if adv_panel is not None:
        # σ 来源：传入则用外部 vol_panel（Phase 3 跨模式一致性），否则内部估算 60-min σ
        if vol_panel is not None:
            vol = vol_panel.reindex(common_idx)[symbols]
        else:
            # 滚动波动率（60 bar），**日化** σ，与 Almgren-Chriss 约定配套
            # bars_per_day 由 periods_per_year 推导：1m→1440, 5m→288, 1h→24
            bars_per_day = periods_per_year / 365.25
            vol = returns_panel.rolling(60, min_periods=10).std() * np.sqrt(bars_per_day)
        adv_aligned = adv_panel.reindex(common_idx)[symbols]

        impact = estimate_market_impact(
            delta_w, adv_aligned, vol,
            portfolio_value, impact_coeff,
        )
        # skipna=False 防御：estimate_market_impact 已 fillna(0)，下游 sum 应无 NaN；
        # skipna=False 让任何意外 NaN 暴露而非静默吞掉
        impact_cost = impact.sum(axis=1, skipna=False)

    total_cost_series = fee_cost + spread_cost + impact_cost
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
