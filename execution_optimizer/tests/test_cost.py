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

        # ③ impact = (2/3) × Σ(coeff_i × σ_i × √(V/ADV_i) × |Δw_i|^1.5)
        #   2/3 来自 Almgren-Chriss 边际冲击率对 q 的积分
        sigma_arr = np.array([0.02, 0.03])
        adv_arr = np.array([1_000_000.0, 500_000.0])
        V = 10_000.0
        eff_coeff = (2.0 / 3.0) * self.impact_coeff * sigma_arr * np.sqrt(V / adv_arr)
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

        # 手工计算（含 2/3 系数）
        abs_dw = np.abs(self.delta_w_val)
        sigma_arr = np.array([0.02, 0.03])
        adv_arr = np.array([1_000_000.0, 500_000.0])
        V = 10_000.0
        coeff_arr = np.array([0.05, 0.2])
        eff_coeff = (2.0 / 3.0) * coeff_arr * sigma_arr * np.sqrt(V / adv_arr)
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


# ---------------------------------------------------------------------------
# 跨模块一致性护栏：execution_optimizer.cost 与 alpha_model.backtest.vectorized
# 的 impact 公式必须给出相同数值
# ---------------------------------------------------------------------------

class TestImpactFormulaConsistency:
    """
    护栏测试：cost.build_cost_expression 的 impact 分量
    与 vectorized.estimate_market_impact 在相同输入下应精确相等。

    这是 B.1 修复后的一致性保证——任一侧公式被改动都会挂。
    支持 "完美执行下事件驱动 ≈ 向量化" 的可比性校验。
    """

    def test_impact_matches_vectorized_single_step(self):
        """单步场景：cost.py impact 分量 == vectorized impact 分量"""
        from alpha_model.backtest.vectorized import estimate_market_impact

        # 固定输入
        delta_w_val = np.array([0.1, -0.05])
        ctx = _make_context(spread=[0.0, 0.0])  # 关掉 spread 和 fee，只测 impact
        impact_coeff = 0.1

        # cost.py 路径
        cost_py_impact = _eval_cost_at(
            delta_w_val, ctx, impact_coeff=impact_coeff, fee_rate=0.0,
        )

        # vectorized.py 路径（构造单行面板）
        symbols = ctx.symbols
        ts = pd.Timestamp("2026-01-01 00:00")
        delta_w_panel = pd.DataFrame([delta_w_val], index=[ts], columns=symbols)
        adv_panel = pd.DataFrame([ctx.adv.values], index=[ts], columns=symbols)
        vol_panel = pd.DataFrame([ctx.volatility.values], index=[ts], columns=symbols)

        vectorized_impact_panel = estimate_market_impact(
            delta_weights=delta_w_panel,
            adv_panel=adv_panel,
            volatility_panel=vol_panel,
            portfolio_value=ctx.portfolio_value,
            impact_coeff=impact_coeff,
        )
        vectorized_impact_total = vectorized_impact_panel.sum(axis=1).iloc[0]

        np.testing.assert_allclose(
            cost_py_impact, vectorized_impact_total, rtol=1e-6,
            err_msg="cost.py 与 vectorized.py 的 impact 公式数值不一致"
        )

    def test_impact_consistency_across_magnitudes(self):
        """跨多个 |Δw| 量级，两侧仍一致（抓变化时的相对偏差）"""
        from alpha_model.backtest.vectorized import estimate_market_impact

        ctx = _make_context(spread=[0.0, 0.0])
        impact_coeff = 0.1
        symbols = ctx.symbols

        for mag in [0.001, 0.01, 0.1, 0.5]:
            dw = np.array([mag, -mag / 2])
            cost_py_val = _eval_cost_at(dw, ctx, impact_coeff=impact_coeff, fee_rate=0.0)

            ts = pd.Timestamp("2026-01-01 00:00")
            dw_panel = pd.DataFrame([dw], index=[ts], columns=symbols)
            adv_panel = pd.DataFrame([ctx.adv.values], index=[ts], columns=symbols)
            vol_panel = pd.DataFrame([ctx.volatility.values], index=[ts], columns=symbols)

            vec_val = estimate_market_impact(
                delta_weights=dw_panel, adv_panel=adv_panel,
                volatility_panel=vol_panel, portfolio_value=ctx.portfolio_value,
                impact_coeff=impact_coeff,
            ).sum(axis=1).iloc[0]

            np.testing.assert_allclose(
                cost_py_val, vec_val, rtol=1e-6,
                err_msg=f"|Δw|={mag} 下两侧 impact 不一致: "
                        f"cost.py={cost_py_val}, vectorized={vec_val}"
            )
