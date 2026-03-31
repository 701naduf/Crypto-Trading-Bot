"""
执行成本函数测试 — T1~T3

T1: 成本分量数值正确性（手工计算比对）
T2: 成本关于 |Δw| 单调递增
T3: 零交易成本为零
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import cvxpy as cp
import pytest

from execution_optimizer.config import MarketContext
from execution_optimizer.cost import build_cost_expression


# ---------------------------------------------------------------------------
# 测试夹具：构造已知 MarketContext
# ---------------------------------------------------------------------------

def _make_context(
    spread: list[float] | None = None,
    volatility: list[float] | None = None,
    adv: list[float] | None = None,
    portfolio_value: float = 10_000.0,
) -> MarketContext:
    """构造双标的 MarketContext（BTC, ETH）"""
    symbols = ["BTC/USDT", "ETH/USDT"]
    return MarketContext(
        timestamp=pd.Timestamp("2026-01-01 00:00"),
        symbols=symbols,
        spread=pd.Series(
            spread or [0.0002, 0.0004], index=symbols,
        ),
        volatility=pd.Series(
            volatility or [0.02, 0.03], index=symbols,
        ),
        adv=pd.Series(
            adv or [1_000_000.0, 500_000.0], index=symbols,
        ),
        portfolio_value=portfolio_value,
    )


def _eval_cost_at(delta_w_val: np.ndarray, context: MarketContext,
                  impact_coeff: float = 0.1, fee_rate: float = 0.0004) -> float:
    """
    用 cvxpy 在指定 Δw 值处求值成本表达式。

    构建一个最小化问题：min cost，其中 delta_w 被 fix 到给定值。
    这样 prob.value 就是成本值。
    """
    n = len(delta_w_val)
    delta_w = cp.Variable(n)
    cost_expr = build_cost_expression(
        delta_w, context, impact_coeff=impact_coeff, fee_rate=fee_rate,
    )
    # 固定 delta_w = delta_w_val，求解 min cost（即求值）
    prob = cp.Problem(cp.Minimize(cost_expr), [delta_w == delta_w_val])
    prob.solve()
    assert prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE), (
        f"求解失败: {prob.status}"
    )
    return prob.value


# ---------------------------------------------------------------------------
# T1: 成本分量数值正确性
# ---------------------------------------------------------------------------

class TestT1CostNumericalCorrectness:
    """
    构造已知 MarketContext（spread=[0.0002, 0.0004], σ=[0.02, 0.03],
    ADV=[1e6, 5e5], V=10000），固定 Δw=[0.1, -0.05]，
    将 build_cost_expression 在该点求值，逐项与手工计算比对。
    """

    def setup_method(self):
        self.context = _make_context()
        self.delta_w_val = np.array([0.1, -0.05])
        self.fee_rate = 0.0004
        self.impact_coeff = 0.1

        # 手工计算各分量
        abs_dw = np.abs(self.delta_w_val)  # [0.1, 0.05]

        # ① commission = fee_rate × Σ|Δw_i| = 0.0004 × (0.1 + 0.05) = 0.00006
        self.expected_commission = self.fee_rate * np.sum(abs_dw)

        # ② spread = Σ(spread_i/2 × |Δw_i|)
        #   = 0.0002/2 × 0.1 + 0.0004/2 × 0.05
        #   = 0.0001 × 0.1 + 0.0002 × 0.05
        #   = 0.00001 + 0.00001 = 0.00002
        spread_arr = np.array([0.0002, 0.0004])
        self.expected_spread = np.sum(spread_arr / 2.0 * abs_dw)

        # ③ impact = Σ(coeff_i × σ_i × √(V/ADV_i) × |Δw_i|^1.5)
        #   BTC: 0.1 × 0.02 × √(10000/1000000) × 0.1^1.5
        #      = 0.1 × 0.02 × √0.01 × 0.031623
        #      = 0.1 × 0.02 × 0.1 × 0.031623
        #      = 0.000006325
        #   ETH: 0.1 × 0.03 × √(10000/500000) × 0.05^1.5
        #      = 0.1 × 0.03 × √0.02 × 0.011180
        #      = 0.1 × 0.03 × 0.14142 × 0.011180
        #      = 0.000004743
        sigma_arr = np.array([0.02, 0.03])
        adv_arr = np.array([1_000_000.0, 500_000.0])
        V = 10_000.0
        eff_coeff = self.impact_coeff * sigma_arr * np.sqrt(V / adv_arr)
        self.expected_impact = np.sum(eff_coeff * abs_dw ** 1.5)

        self.expected_total = (
            self.expected_commission + self.expected_spread + self.expected_impact
        )

    def test_total_cost_matches_hand_calculation(self):
        """总成本与手工计算一致"""
        actual = _eval_cost_at(
            self.delta_w_val, self.context,
            impact_coeff=self.impact_coeff, fee_rate=self.fee_rate,
        )
        np.testing.assert_allclose(actual, self.expected_total, rtol=1e-4)

    def test_commission_component(self):
        """单独验证手续费分量：设 spread=0, impact_coeff=0"""
        ctx = _make_context(spread=[0.0, 0.0])
        actual = _eval_cost_at(
            self.delta_w_val, ctx,
            impact_coeff=0.0, fee_rate=self.fee_rate,
        )
        np.testing.assert_allclose(actual, self.expected_commission, atol=1e-10)

    def test_spread_component(self):
        """单独验证价差分量：设 fee_rate=0, impact_coeff=0"""
        actual = _eval_cost_at(
            self.delta_w_val, self.context,
            impact_coeff=0.0, fee_rate=0.0,
        )
        np.testing.assert_allclose(actual, self.expected_spread, atol=1e-10)

    def test_impact_component(self):
        """单独验证冲击分量：设 fee_rate=0, spread=0"""
        ctx = _make_context(spread=[0.0, 0.0])
        actual = _eval_cost_at(
            self.delta_w_val, ctx,
            impact_coeff=self.impact_coeff, fee_rate=0.0,
        )
        np.testing.assert_allclose(actual, self.expected_impact, rtol=1e-4)

    def test_per_symbol_impact_coeff(self):
        """逐标的冲击系数：pd.Series 形式"""
        symbols = self.context.symbols
        coeff_series = pd.Series([0.05, 0.2], index=symbols)

        # 手工计算
        abs_dw = np.abs(self.delta_w_val)
        sigma_arr = np.array([0.02, 0.03])
        adv_arr = np.array([1_000_000.0, 500_000.0])
        V = 10_000.0
        coeff_arr = np.array([0.05, 0.2])
        eff_coeff = coeff_arr * sigma_arr * np.sqrt(V / adv_arr)
        expected_impact = np.sum(eff_coeff * abs_dw ** 1.5)

        # 清除其他分量
        ctx = _make_context(spread=[0.0, 0.0])
        actual = _eval_cost_at(
            self.delta_w_val, ctx,
            impact_coeff=coeff_series, fee_rate=0.0,
        )
        np.testing.assert_allclose(actual, expected_impact, rtol=1e-4)


# ---------------------------------------------------------------------------
# T2: 成本关于 |Δw| 单调递增
# ---------------------------------------------------------------------------

class TestT2CostMonotonicity:
    """
    固定 MarketContext，依次传入 |Δw|=[0.01, 0.05, 0.1, 0.5]，
    验证 cost 严格单调递增。
    """

    def test_cost_monotonically_increasing(self):
        context = _make_context()
        magnitudes = [0.01, 0.05, 0.1, 0.5]
        costs = []

        for mag in magnitudes:
            dw = np.array([mag, mag])
            cost = _eval_cost_at(dw, context)
            costs.append(cost)

        # 验证严格递增
        for i in range(1, len(costs)):
            assert costs[i] > costs[i - 1], (
                f"cost({magnitudes[i]})={costs[i]:.10f} "
                f"不大于 cost({magnitudes[i-1]})={costs[i-1]:.10f}"
            )

    def test_cost_monotonic_single_asset(self):
        """单标的方向上递增"""
        context = _make_context()
        magnitudes = [0.01, 0.05, 0.1, 0.5]
        costs = []

        for mag in magnitudes:
            dw = np.array([mag, 0.0])  # 只交易 BTC
            cost = _eval_cost_at(dw, context)
            costs.append(cost)

        for i in range(1, len(costs)):
            assert costs[i] > costs[i - 1]


# ---------------------------------------------------------------------------
# T3: 零交易成本为零
# ---------------------------------------------------------------------------

class TestT3ZeroTradeCost:
    """Δw = zeros(N) → cost ≈ 0（atol=1e-10）"""

    def test_zero_delta_w_gives_zero_cost(self):
        context = _make_context()
        dw = np.zeros(2)
        cost = _eval_cost_at(dw, context)
        np.testing.assert_allclose(cost, 0.0, atol=1e-10)

    def test_zero_delta_w_different_contexts(self):
        """不同 MarketContext 下零交易成本仍为零"""
        for V in [1_000, 100_000, 1_000_000]:
            context = _make_context(portfolio_value=V)
            dw = np.zeros(2)
            cost = _eval_cost_at(dw, context)
            np.testing.assert_allclose(cost, 0.0, atol=1e-10)
