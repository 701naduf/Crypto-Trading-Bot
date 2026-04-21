"""
动态执行成本函数（收益率空间）

返回无量纲 cvxpy 表达式，可直接加入 cp.Minimize 目标函数。

成本在收益率空间表达，与目标函数中 w'Σw 和 λ×α'w 量纲一致，可直接相加。

量纲推导（Almgren-Chriss 总冲击成本）:
    边际冲击率（每单位交易量导致的价格偏离）:
      marginal_impact(q) = σ × √(q / ADV)

    执行总量 Q 的总美元成本（对 q 从 0 到 Q 积分）:
      total_impact_USD = ∫₀^Q σ × √(q/ADV) dq = (2/3) × σ × Q^1.5 / √ADV

    代入 Q = V × |Δw|，并 ÷ V 归一化:
      commission = fee_rate × Σ|Δw_i|                                       （无量纲）
      spread     = Σ(spread_i/2 × |Δw_i|)                                   （无量纲）
      impact     = (2/3) × Σ(impact_coeff × σ_i × √(V/ADV_i) × |Δw_i|^1.5) （无量纲）

    说明:
      - 2/3 系数来自边际冲击率的积分，显式保留以便 impact_coeff 成为纯校准量
        （从成交数据回归时，其值与 Almgren-Chriss 文献系数直接对应）
      - √(V/ADV_i) 项体现组合规模效应：组合越大，相对冲击越高

DCP 合规说明:
    - cp.abs(delta_w) → convex, nonneg
    - cp.power(nonneg_convex, 1.5) → convex（p≥1 且参数 nonneg，符合 DCP composition rule）
    - 线性组合保凸

求解器说明:
    1.5 次幂项超出 OSQP（QP 求解器）能力范围，cvxpy 将自动选择 ECOS（SOCP 求解器）。
"""
from __future__ import annotations
import numpy as np
import cvxpy as cp
import pandas as pd

from execution_optimizer.config import (
    MarketContext, DEFAULT_IMPACT_COEFF, DEFAULT_FEE_RATE,
)


def build_cost_expression(
    delta_w: cp.Variable,
    context: MarketContext,
    impact_coeff: float | pd.Series = DEFAULT_IMPACT_COEFF,
    fee_rate: float = DEFAULT_FEE_RATE,
) -> cp.Expression:
    """
    构建无量纲的总执行成本表达式

    Args:
        delta_w:      权重变化向量 cp.Variable, shape=(N,)
        context:      当前时刻的市场状态
        impact_coeff: sqrt-model 校准系数。
                      float → 全标的共享；
                      pd.Series(index=symbols) → 逐标的独立校准
        fee_rate:     taker 手续费率

    Returns:
        cvxpy scalar Expression（凸），无量纲
    """
    V = context.portfolio_value
    symbols = context.symbols
    spread_arr = context.spread.reindex(symbols).values.astype(float)
    sigma_arr  = context.volatility.reindex(symbols).values.astype(float)
    adv_arr    = context.adv.reindex(symbols).values.astype(float)

    # 逐标的冲击系数
    if isinstance(impact_coeff, pd.Series):
        coeff_arr = impact_coeff.reindex(symbols).values.astype(float)
    else:
        coeff_arr = np.full(len(symbols), impact_coeff)

    delta_abs = cp.abs(delta_w)   # |Δw_i|, nonneg, shape=(N,)

    # ① 手续费: fee_rate × Σ|Δw_i|
    commission = fee_rate * cp.sum(delta_abs)

    # ② 买卖价差: Σ(spread_i/2 × |Δw_i|)
    half_spread = spread_arr / 2.0
    spread_cost = half_spread @ delta_abs

    # ③ 市场冲击: (2/3) × Σ(eff_coeff_i × |Δw_i|^1.5)
    #    eff_coeff_i = coeff_i × σ_i × √(V / ADV_i)
    #    2/3 来自 Almgren-Chriss 边际冲击率对 q 的积分，显式保留
    eff_coeff = (2.0 / 3.0) * coeff_arr * sigma_arr * np.sqrt(
        V / np.maximum(adv_arr, 1.0)
    )
    impact_cost = eff_coeff @ cp.power(delta_abs, 1.5)

    return commission + spread_cost + impact_cost
