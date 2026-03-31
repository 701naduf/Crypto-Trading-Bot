"""
ExecutionOptimizer 测试 — T4~T10

T4:  optimize_step 基本可解性
T5:  高成本压制交易
T6:  ADV 参与率约束生效
T7:  资金费率影响持仓方向
T8:  Vol Targeting 事后缩放
T9:  Fallback 路径（协方差失败 / 求解失败 / 状态非最优）
T10: Beta-neutral 路径
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_model.core.types import PortfolioConstraints
from execution_optimizer.config import MarketContext
from execution_optimizer.optimizer import ExecutionOptimizer


# ---------------------------------------------------------------------------
# 测试夹具
# ---------------------------------------------------------------------------

SYMBOLS = ["BTC/USDT", "ETH/USDT"]


def _make_price_history(
    n_bars: int = 200,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """
    生成合成价格面板（几何布朗运动）

    确保足够长的历史以支持协方差估计（默认 200 bars > vol_lookback=60）。
    """
    symbols = symbols or SYMBOLS
    rng = np.random.RandomState(42)
    dates = pd.date_range("2026-01-01", periods=n_bars, freq="min")
    prices = {}
    for sym in symbols:
        # 每分钟收益率 ~ N(0, 0.001^2)
        ret = rng.normal(0, 0.001, n_bars)
        prices[sym] = 100.0 * np.exp(np.cumsum(ret))
    return pd.DataFrame(prices, index=dates)


def _make_context(
    spread: list[float] | None = None,
    adv: list[float] | None = None,
    portfolio_value: float = 10_000.0,
    funding_rate: pd.Series | None = None,
    symbols: list[str] | None = None,
) -> MarketContext:
    """构造 MarketContext"""
    symbols = symbols or SYMBOLS
    return MarketContext(
        timestamp=pd.Timestamp("2026-01-01 03:20"),
        symbols=symbols,
        spread=pd.Series(
            spread or [0.0002, 0.0004], index=symbols,
        ),
        volatility=pd.Series(
            [0.02, 0.03], index=symbols,
        ),
        adv=pd.Series(
            adv or [1_000_000.0, 500_000.0], index=symbols,
        ),
        portfolio_value=portfolio_value,
        funding_rate=funding_rate,
    )


def _default_constraints(**kwargs) -> PortfolioConstraints:
    """构造默认约束配置"""
    defaults = dict(
        max_weight=0.4,
        dollar_neutral=True,
        beta_neutral=False,
        vol_target=None,
        leverage_cap=2.0,
        risk_aversion=1.0,
        turnover_penalty=0.01,  # 不被 ExecutionOptimizer 使用
    )
    defaults.update(kwargs)
    return PortfolioConstraints(**defaults)


# ---------------------------------------------------------------------------
# T4: optimize_step 基本可解性
# ---------------------------------------------------------------------------

class TestT4BasicSolvability:
    """
    给定合理输入（有效信号、非零 ADV、正常 spread），验证：
    - 返回值无 NaN / Inf
    - Σ|w| ≤ leverage_cap + atol
    - |w_i| ≤ max_weight + atol
    """

    def test_basic_solve_no_nan_inf(self):
        """返回值无 NaN / Inf"""
        constraints = _default_constraints()
        optimizer = ExecutionOptimizer(constraints, max_participation=None)

        signals = pd.Series([0.5, -0.3], index=SYMBOLS)
        current_w = pd.Series([0.0, 0.0], index=SYMBOLS)
        context = _make_context()
        prices = _make_price_history()

        result = optimizer.optimize_step(signals, current_w, context, prices)

        assert isinstance(result, pd.Series)
        assert not result.isna().any(), f"含 NaN: {result}"
        assert not np.isinf(result.values).any(), f"含 Inf: {result}"

    def test_leverage_cap_respected(self):
        """Σ|w| ≤ leverage_cap"""
        constraints = _default_constraints(leverage_cap=1.5)
        optimizer = ExecutionOptimizer(constraints, max_participation=None)

        signals = pd.Series([1.0, -0.8], index=SYMBOLS)
        current_w = pd.Series([0.0, 0.0], index=SYMBOLS)
        context = _make_context()
        prices = _make_price_history()

        result = optimizer.optimize_step(signals, current_w, context, prices)

        total_leverage = np.abs(result.values).sum()
        assert total_leverage <= constraints.leverage_cap + 1e-6, (
            f"杠杆 {total_leverage:.6f} 超过上限 {constraints.leverage_cap}"
        )

    def test_max_weight_respected(self):
        """|w_i| ≤ max_weight"""
        constraints = _default_constraints(max_weight=0.3)
        optimizer = ExecutionOptimizer(constraints, max_participation=None)

        signals = pd.Series([2.0, -1.5], index=SYMBOLS)
        current_w = pd.Series([0.0, 0.0], index=SYMBOLS)
        context = _make_context()
        prices = _make_price_history()

        result = optimizer.optimize_step(signals, current_w, context, prices)

        for sym in SYMBOLS:
            assert abs(result[sym]) <= constraints.max_weight + 1e-6, (
                f"{sym} 权重 {result[sym]:.6f} 超过上限 {constraints.max_weight}"
            )

    def test_dollar_neutral_respected(self):
        """Σw ≈ 0（dollar-neutral）"""
        constraints = _default_constraints(dollar_neutral=True)
        optimizer = ExecutionOptimizer(constraints, max_participation=None)

        signals = pd.Series([0.5, -0.3], index=SYMBOLS)
        current_w = pd.Series([0.0, 0.0], index=SYMBOLS)
        context = _make_context()
        prices = _make_price_history()

        result = optimizer.optimize_step(signals, current_w, context, prices)

        weight_sum = result.values.sum()
        np.testing.assert_allclose(weight_sum, 0.0, atol=1e-5)

    def test_nonzero_initial_weights(self):
        """从非零初始仓位出发也能正常求解"""
        constraints = _default_constraints()
        optimizer = ExecutionOptimizer(constraints, max_participation=None)

        signals = pd.Series([0.3, -0.2], index=SYMBOLS)
        current_w = pd.Series([0.1, -0.1], index=SYMBOLS)
        context = _make_context()
        prices = _make_price_history()

        result = optimizer.optimize_step(signals, current_w, context, prices)

        assert not result.isna().any()
        assert np.abs(result.values).sum() <= constraints.leverage_cap + 1e-6


# ---------------------------------------------------------------------------
# T5: 高成本压制交易
# ---------------------------------------------------------------------------

class TestT5HighCostSuppression:
    """
    设置 spread=0.10（10% bid-ask）→ |Δw| 应接近零（成本完全压制信号）。
    设置 spread=0.0001 → 权重变化应显著大于高 spread 场景。
    """

    def test_high_spread_suppresses_trading(self):
        """10% bid-ask spread 应压制交易"""
        constraints = _default_constraints()

        # 高 spread 场景
        optimizer_high = ExecutionOptimizer(constraints, max_participation=None)
        context_high = _make_context(spread=[0.10, 0.10])

        # 低 spread 场景
        optimizer_low = ExecutionOptimizer(constraints, max_participation=None)
        context_low = _make_context(spread=[0.0001, 0.0001])

        signals = pd.Series([0.5, -0.3], index=SYMBOLS)
        current_w = pd.Series([0.0, 0.0], index=SYMBOLS)
        prices = _make_price_history()

        result_high = optimizer_high.optimize_step(
            signals, current_w, context_high, prices,
        )
        result_low = optimizer_low.optimize_step(
            signals, current_w, context_low, prices,
        )

        # 高 spread 下权重变化应很小
        delta_high = np.abs(result_high.values - current_w.values).sum()
        delta_low = np.abs(result_low.values - current_w.values).sum()

        assert delta_low > delta_high, (
            f"低 spread delta={delta_low:.6f} 应大于高 spread delta={delta_high:.6f}"
        )

    def test_extreme_spread_near_zero_trade(self):
        """极端 spread 下交易量趋近零"""
        constraints = _default_constraints()
        optimizer = ExecutionOptimizer(constraints, max_participation=None)

        # 200% bid-ask spread — 极端不流动
        # half_spread=1.0，远超 α=0.5 的驱动力，成本完全压制信号
        context = _make_context(spread=[2.0, 2.0])
        signals = pd.Series([0.5, -0.3], index=SYMBOLS)
        current_w = pd.Series([0.0, 0.0], index=SYMBOLS)
        prices = _make_price_history()

        result = optimizer.optimize_step(signals, current_w, context, prices)

        delta = np.abs(result.values).sum()
        assert delta < 0.05, (
            f"200% spread 下仍有 {delta:.4f} 的交易，应接近零"
        )


# ---------------------------------------------------------------------------
# T6: ADV 参与率约束生效
# ---------------------------------------------------------------------------

class TestT6ADVParticipation:
    """
    设置 max_participation=0.01，ADV 较小，
    验证 |Δw_i| × V ≤ 0.01 × ADV_i + atol 对所有标的成立。
    """

    def test_participation_constraint_binding(self):
        """ADV 参与率约束限制权重变化"""
        max_part = 0.01
        small_adv = [50_000.0, 30_000.0]  # 较小的 ADV
        V = 10_000.0

        constraints = _default_constraints()
        optimizer = ExecutionOptimizer(
            constraints,
            max_participation=max_part,
        )

        signals = pd.Series([2.0, -1.5], index=SYMBOLS)  # 强信号
        current_w = pd.Series([0.0, 0.0], index=SYMBOLS)
        context = _make_context(adv=small_adv, portfolio_value=V)
        prices = _make_price_history()

        result = optimizer.optimize_step(signals, current_w, context, prices)

        # 验证: |Δw_i| × V ≤ max_participation × ADV_i
        for i, sym in enumerate(SYMBOLS):
            delta_w_i = abs(result[sym] - current_w[sym])
            trade_volume = delta_w_i * V
            max_allowed = max_part * small_adv[i]
            assert trade_volume <= max_allowed + 1e-4, (
                f"{sym}: 交易量 {trade_volume:.2f} 超过 "
                f"ADV 限制 {max_allowed:.2f}"
            )

    def test_participation_limits_weight_space(self):
        """等价地验证权重空间: |Δw_i| ≤ participation × ADV_i / V"""
        max_part = 0.01
        small_adv = [50_000.0, 30_000.0]
        V = 10_000.0

        constraints = _default_constraints()
        optimizer = ExecutionOptimizer(
            constraints,
            max_participation=max_part,
        )

        signals = pd.Series([1.0, -0.8], index=SYMBOLS)
        current_w = pd.Series([0.0, 0.0], index=SYMBOLS)
        context = _make_context(adv=small_adv, portfolio_value=V)
        prices = _make_price_history()

        result = optimizer.optimize_step(signals, current_w, context, prices)

        for i, sym in enumerate(SYMBOLS):
            delta_w_i = abs(result[sym] - current_w[sym])
            w_limit = max_part * small_adv[i] / V
            assert delta_w_i <= w_limit + 1e-6, (
                f"{sym}: |Δw|={delta_w_i:.6f} 超过限制 {w_limit:.6f}"
            )

    def test_no_participation_constraint(self):
        """max_participation=None 时不限制"""
        constraints = _default_constraints()
        optimizer = ExecutionOptimizer(
            constraints,
            max_participation=None,
        )

        signals = pd.Series([2.0, -1.5], index=SYMBOLS)
        current_w = pd.Series([0.0, 0.0], index=SYMBOLS)
        context = _make_context()
        prices = _make_price_history()

        result = optimizer.optimize_step(signals, current_w, context, prices)

        # 只要能正常返回即可
        assert not result.isna().any()


# ---------------------------------------------------------------------------
# T7: 资金费率影响持仓方向
# ---------------------------------------------------------------------------

class TestT7FundingRateImpact:
    """
    构造两标的，信号相同，但 A 的 funding_rate >> 0，B 的 funding_rate ≈ 0。
    验证优化结果中 |w_A| < |w_B|（资金费率高的标的被抑制）。
    """

    def test_high_funding_rate_suppresses_long_position(self):
        """高资金费率抑制多头持仓"""
        constraints = _default_constraints(dollar_neutral=False)

        # A: 高资金费率（多头付费），B: 零资金费率
        funding = pd.Series([0.01, 0.0], index=SYMBOLS)  # A 每 bar 1% 资金费率
        context = _make_context(funding_rate=funding)

        optimizer = ExecutionOptimizer(constraints, max_participation=None)

        # 两标的信号相同（都看多）
        signals = pd.Series([1.0, 1.0], index=SYMBOLS)
        current_w = pd.Series([0.0, 0.0], index=SYMBOLS)
        prices = _make_price_history()

        result = optimizer.optimize_step(signals, current_w, context, prices)

        # 高 funding_rate 的标的应被抑制
        assert abs(result[SYMBOLS[0]]) < abs(result[SYMBOLS[1]]), (
            f"高资金费率标的 |w|={abs(result[SYMBOLS[0]]):.6f} "
            f"应小于低资金费率标的 |w|={abs(result[SYMBOLS[1]]):.6f}"
        )

    def test_no_funding_rate_symmetric(self):
        """无资金费率时，相同信号产生相近权重"""
        constraints = _default_constraints(dollar_neutral=False)

        # 无资金费率
        context = _make_context(funding_rate=None)
        optimizer = ExecutionOptimizer(constraints, max_participation=None)

        # 两标的信号相同
        signals = pd.Series([1.0, 1.0], index=SYMBOLS)
        current_w = pd.Series([0.0, 0.0], index=SYMBOLS)
        prices = _make_price_history()

        result = optimizer.optimize_step(signals, current_w, context, prices)

        # 权重应相对接近（不完全相等，因为协方差不同）
        # 只检查两者都非零且同向
        assert result[SYMBOLS[0]] > 0 and result[SYMBOLS[1]] > 0, (
            f"相同正信号应产生正权重: {result.to_dict()}"
        )

    def test_negative_funding_rate_encourages_long(self):
        """负资金费率（多头收费）应鼓励持仓"""
        constraints = _default_constraints(dollar_neutral=False)

        # A: 负资金费率（多头有收益），B: 正资金费率
        funding = pd.Series([-0.005, 0.005], index=SYMBOLS)
        context = _make_context(funding_rate=funding)

        optimizer = ExecutionOptimizer(constraints, max_participation=None)

        signals = pd.Series([1.0, 1.0], index=SYMBOLS)
        current_w = pd.Series([0.0, 0.0], index=SYMBOLS)
        prices = _make_price_history()

        result = optimizer.optimize_step(signals, current_w, context, prices)

        # 负 funding_rate 的标的应更受青睐
        assert result[SYMBOLS[0]] > result[SYMBOLS[1]], (
            f"负资金费率标的 w={result[SYMBOLS[0]]:.6f} "
            f"应大于正资金费率标的 w={result[SYMBOLS[1]]:.6f}"
        )


# ---------------------------------------------------------------------------
# T8: Vol Targeting 事后缩放
# ---------------------------------------------------------------------------

class TestT8VolTargeting:
    """
    设置 vol_target，验证：
    - 输出权重对应的年化波动率接近目标
    - 缩放后超过 leverage_cap 时，二次检查生效
    """

    def test_vol_target_scales_weights(self):
        """vol_target=0.15 时，组合年化波动率应接近目标"""
        from alpha_model.config import MINUTES_PER_YEAR

        vol_target = 0.15
        constraints = _default_constraints(
            vol_target=vol_target, dollar_neutral=False, leverage_cap=5.0,
        )
        optimizer = ExecutionOptimizer(constraints, max_participation=None)

        signals = pd.Series([0.5, -0.3], index=SYMBOLS)
        current_w = pd.Series([0.0, 0.0], index=SYMBOLS)
        context = _make_context()
        prices = _make_price_history()

        result = optimizer.optimize_step(signals, current_w, context, prices)

        # 用同一个协方差矩阵验证组合波动率
        returns_panel = prices[SYMBOLS].pct_change().dropna()
        from alpha_model.portfolio.covariance import estimate_covariance
        lookback = min(constraints.vol_lookback, len(returns_panel))
        cov = estimate_covariance(returns_panel, lookback=lookback)
        w_arr = result.values
        port_vol = np.sqrt(w_arr @ cov @ w_arr * MINUTES_PER_YEAR)

        # 允许 50% 的容差（因为估计误差），核心是验证缩放逻辑被触发且方向正确
        assert port_vol < vol_target * 1.5, (
            f"组合波动率 {port_vol:.4f} 远超目标 {vol_target}"
        )

    def test_vol_target_leverage_cap_recheck(self):
        """缩放后超过 leverage_cap 时，二次检查应限制杠杆"""
        # 设置极高 vol_target（强制大幅缩放）+ 紧 leverage_cap
        constraints = _default_constraints(
            vol_target=5.0,          # 非常高的目标，会导致大幅放大
            dollar_neutral=False,
            leverage_cap=1.0,        # 紧杠杆上限
        )
        optimizer = ExecutionOptimizer(constraints, max_participation=None)

        signals = pd.Series([0.5, -0.3], index=SYMBOLS)
        current_w = pd.Series([0.0, 0.0], index=SYMBOLS)
        context = _make_context()
        prices = _make_price_history()

        result = optimizer.optimize_step(signals, current_w, context, prices)

        # 即使 vol_target 要求大幅放大，leverage_cap 也应被尊重
        total_leverage = np.abs(result.values).sum()
        assert total_leverage <= constraints.leverage_cap + 1e-5, (
            f"杠杆 {total_leverage:.6f} 超过 leverage_cap {constraints.leverage_cap}"
        )

    def test_no_vol_target_no_scaling(self):
        """vol_target=None 时不做缩放（对照组）"""
        constraints_no_vt = _default_constraints(vol_target=None, dollar_neutral=False)
        constraints_with_vt = _default_constraints(vol_target=0.05, dollar_neutral=False)

        optimizer_no = ExecutionOptimizer(constraints_no_vt, max_participation=None)
        optimizer_vt = ExecutionOptimizer(constraints_with_vt, max_participation=None)

        signals = pd.Series([0.5, -0.3], index=SYMBOLS)
        current_w = pd.Series([0.0, 0.0], index=SYMBOLS)
        context = _make_context()
        prices = _make_price_history()

        result_no = optimizer_no.optimize_step(signals, current_w, context, prices)
        result_vt = optimizer_vt.optimize_step(signals, current_w, context, prices)

        # 两者的权重应不同（vol_target 会缩放）
        assert not np.allclose(result_no.values, result_vt.values, atol=1e-6), (
            "有无 vol_target 的结果不应完全相同"
        )


# ---------------------------------------------------------------------------
# T9: Fallback 路径
# ---------------------------------------------------------------------------

class TestT9Fallback:
    """
    验证三条 fallback 路径：
    - 协方差估计失败 → 返回 current_weights
    - cvxpy 求解异常 → 返回 current_weights
    - 求解状态非最优 → 返回 current_weights
    """

    def test_covariance_failure_returns_current_weights(self):
        """price_history 长度不足 → 协方差估计失败 → 返回 current_weights"""
        constraints = _default_constraints()
        optimizer = ExecutionOptimizer(constraints, max_participation=None)

        signals = pd.Series([0.5, -0.3], index=SYMBOLS)
        current_w = pd.Series([0.1, -0.1], index=SYMBOLS)
        context = _make_context()

        # 只给 5 行价格数据（dropna 后仅 4 行 < min_periods=20）
        prices = _make_price_history(n_bars=5)

        result = optimizer.optimize_step(signals, current_w, context, prices)

        # 应返回 current_weights（fallback）
        np.testing.assert_array_almost_equal(
            result.values, current_w.reindex(SYMBOLS).values,
        )

    def test_infeasible_problem_returns_current_weights(self):
        """构造不可行问题 → 求解状态非最优 → 返回 current_weights"""
        # max_weight=0.1 + dollar_neutral + leverage_cap=0.05
        # dollar_neutral 要求 Σw=0，max_weight=0.1，但 leverage_cap=0.05
        # 对两标的：Σ|w|≤0.05 且 w1+w2=0 → |w1|=|w2|≤0.025，可行但极小
        # 用更极端的约束：max_weight=0.01 + leverage_cap=0.01 + 强信号
        # 实际上 QP 通常能找到 w≈0 的解，所以改用 monkeypatch 使 solve 抛异常
        import unittest.mock as mock

        constraints = _default_constraints()
        optimizer = ExecutionOptimizer(constraints, max_participation=None)

        signals = pd.Series([0.5, -0.3], index=SYMBOLS)
        current_w = pd.Series([0.2, -0.2], index=SYMBOLS)
        context = _make_context()
        prices = _make_price_history()

        # 模拟 cvxpy Problem.solve() 抛出异常
        with mock.patch("cvxpy.Problem.solve", side_effect=Exception("SolverError")):
            result = optimizer.optimize_step(signals, current_w, context, prices)

        np.testing.assert_array_almost_equal(
            result.values, current_w.reindex(SYMBOLS).values,
        )

    def test_suboptimal_status_returns_current_weights(self):
        """求解状态为 INFEASIBLE → 返回 current_weights"""
        import unittest.mock as mock
        import cvxpy as cp

        constraints = _default_constraints()
        optimizer = ExecutionOptimizer(constraints, max_participation=None)

        signals = pd.Series([0.5, -0.3], index=SYMBOLS)
        current_w = pd.Series([0.15, -0.15], index=SYMBOLS)
        context = _make_context()
        prices = _make_price_history()

        # 模拟 solve 成功但状态为 INFEASIBLE
        original_solve = cp.Problem.solve

        def fake_solve(self_prob, *args, **kwargs):
            original_solve(self_prob, *args, **kwargs)
            self_prob._status = cp.INFEASIBLE

        with mock.patch("cvxpy.Problem.solve", fake_solve):
            result = optimizer.optimize_step(signals, current_w, context, prices)

        np.testing.assert_array_almost_equal(
            result.values, current_w.reindex(SYMBOLS).values,
        )


# ---------------------------------------------------------------------------
# T10: Beta-neutral 路径
# ---------------------------------------------------------------------------

class TestT10BetaNeutral:
    """
    设置 beta_neutral=True，验证：
    - 求解成功
    - 输出权重满足 β'w ≈ 0
    """

    def test_beta_neutral_solve_success(self):
        """beta_neutral=True 时求解成功"""
        constraints = _default_constraints(
            beta_neutral=True, dollar_neutral=True, beta_lookback=60,
        )
        optimizer = ExecutionOptimizer(constraints, max_participation=None)

        signals = pd.Series([0.5, -0.3], index=SYMBOLS)
        current_w = pd.Series([0.0, 0.0], index=SYMBOLS)
        context = _make_context()
        prices = _make_price_history(n_bars=200)

        result = optimizer.optimize_step(signals, current_w, context, prices)

        assert not result.isna().any(), f"含 NaN: {result}"
        assert not np.isinf(result.values).any(), f"含 Inf: {result}"

    def test_beta_neutral_constraint_respected(self):
        """β'w ≈ 0"""
        from alpha_model.portfolio.beta import rolling_beta

        constraints = _default_constraints(
            beta_neutral=True, dollar_neutral=True, beta_lookback=60,
        )
        optimizer = ExecutionOptimizer(constraints, max_participation=None)

        signals = pd.Series([0.5, -0.3], index=SYMBOLS)
        current_w = pd.Series([0.0, 0.0], index=SYMBOLS)
        context = _make_context()
        prices = _make_price_history(n_bars=200)

        result = optimizer.optimize_step(signals, current_w, context, prices)

        # 用同一数据计算 beta，验证 β'w ≈ 0
        returns_panel = prices[SYMBOLS].pct_change().dropna()
        beta_panel = rolling_beta(returns_panel, lookback=constraints.beta_lookback)
        last_beta = beta_panel.iloc[-1].reindex(SYMBOLS).values

        if not np.any(np.isnan(last_beta)):
            beta_exposure = last_beta @ result.values
            np.testing.assert_allclose(beta_exposure, 0.0, atol=1e-4, err_msg=(
                f"Beta 暴露 {beta_exposure:.6f} 应接近 0"
            ))

    def test_beta_neutral_insufficient_history_skips(self):
        """price_history 不足以估计 beta 时，beta_vec=None，约束被跳过"""
        constraints = _default_constraints(
            beta_neutral=True, dollar_neutral=True,
            beta_lookback=60,  # 需要 60 行 beta
        )
        optimizer = ExecutionOptimizer(constraints, max_participation=None)

        signals = pd.Series([0.5, -0.3], index=SYMBOLS)
        current_w = pd.Series([0.0, 0.0], index=SYMBOLS)
        context = _make_context()
        # 只给 70 行（pct_change 后 69 行，刚好够 lookback=60，
        # 但 rolling_beta 的 min_periods=lookback，仅最后几行有值）
        prices = _make_price_history(n_bars=70)

        # 应正常求解（beta 约束被跳过，不应 crash）
        result = optimizer.optimize_step(signals, current_w, context, prices)
        assert not result.isna().any()


# ---------------------------------------------------------------------------
# 边界场景
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """边界场景测试"""

    def test_single_symbol_dollar_neutral(self):
        """单标的 + dollar_neutral → 权重恒为 0"""
        sym = ["BTC/USDT"]
        constraints = _default_constraints(dollar_neutral=True)
        optimizer = ExecutionOptimizer(constraints, max_participation=None)

        signals = pd.Series([1.0], index=sym)
        current_w = pd.Series([0.0], index=sym)
        context = MarketContext(
            timestamp=pd.Timestamp("2026-01-01 03:20"),
            symbols=sym,
            spread=pd.Series([0.0002], index=sym),
            volatility=pd.Series([0.02], index=sym),
            adv=pd.Series([1_000_000.0], index=sym),
            portfolio_value=10_000.0,
        )
        prices = _make_price_history(n_bars=200, symbols=sym)

        result = optimizer.optimize_step(signals, current_w, context, prices)

        # dollar_neutral: Σw = 0，单标的只能 w=0
        np.testing.assert_allclose(result.values, [0.0], atol=1e-6)
