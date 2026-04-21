"""
backtest/ 模块的单元测试

测试:
    - performance.py: 绩效指标
    - vectorized.py: 向量化回测
"""

import pytest
import numpy as np
import pandas as pd

from alpha_model.backtest.performance import (
    BacktestResult,
    sortino_ratio,
    calmar_ratio,
    max_drawdown_duration,
)
from alpha_model.backtest.vectorized import vectorized_backtest, estimate_market_impact

from factor_research.evaluation.metrics import cumulative_returns


def _make_price_panel(n_rows=200, symbols=None, seed=42):
    """生成价格面板"""
    if symbols is None:
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    rng = np.random.RandomState(seed)
    returns = rng.randn(n_rows, len(symbols)) * 0.001
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    idx = pd.date_range("2024-01-01", periods=n_rows, freq="1min", tz="UTC")
    return pd.DataFrame(prices, index=idx, columns=symbols)


# ---------------------------------------------------------------------------
# performance.py
# ---------------------------------------------------------------------------

class TestSortinoRatio:
    """Sortino ratio"""

    def test_positive_returns(self):
        """正收益序列的 Sortino > 0"""
        returns = pd.Series(np.random.RandomState(42).randn(1000) * 0.001 + 0.0001)
        s = sortino_ratio(returns)
        assert s > 0

    def test_negative_returns(self):
        """负收益序列的 Sortino < 0"""
        returns = pd.Series(np.random.RandomState(42).randn(1000) * 0.001 - 0.001)
        s = sortino_ratio(returns)
        assert s < 0


class TestCalmarRatio:
    """Calmar ratio"""

    def test_positive_returns(self):
        """正收益且有回撤"""
        returns = pd.Series(np.random.RandomState(42).randn(1000) * 0.001 + 0.0001)
        c = calmar_ratio(returns)
        # 应为有限数
        assert np.isfinite(c)


class TestMaxDrawdownDuration:
    """最长回撤持续期"""

    def test_no_drawdown(self):
        """持续上涨无回撤"""
        returns = pd.Series([0.01] * 100)
        assert max_drawdown_duration(returns) == 0

    def test_with_drawdown(self):
        """有回撤的序列"""
        returns = pd.Series([0.01] * 50 + [-0.02] * 10 + [0.01] * 50)
        duration = max_drawdown_duration(returns)
        assert duration > 0


class TestBacktestResult:
    """BacktestResult"""

    def test_summary_keys(self):
        """summary() 应包含所有关键指标"""
        returns = pd.Series(
            np.random.RandomState(42).randn(500) * 0.001,
            index=pd.date_range("2024-01-01", periods=500, freq="1min"),
        )
        equity = cumulative_returns(returns)
        turnover = pd.Series(0.1, index=returns.index)
        weights = pd.DataFrame(
            {"A": [0.5] * 500, "B": [-0.5] * 500},
            index=returns.index,
        )

        result = BacktestResult(
            equity_curve=equity,
            returns=returns,
            turnover=turnover,
            weights_history=weights,
        )
        summary = result.summary()
        expected_keys = [
            "annual_return", "annual_volatility", "sharpe_ratio",
            "sortino_ratio", "calmar_ratio", "max_drawdown",
            "max_drawdown_duration", "avg_turnover", "total_cost",
            "win_rate", "n_periods", "total_return",
        ]
        for key in expected_keys:
            assert key in summary, f"缺少 {key}"


# ---------------------------------------------------------------------------
# vectorized.py
# ---------------------------------------------------------------------------

class TestVectorizedBacktest:
    """向量化回测"""

    def test_basic_backtest(self):
        """基本回测"""
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        n = 200
        idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")

        price_panel = _make_price_panel(n, symbols)
        weights = pd.DataFrame(
            {
                "BTC/USDT": [0.3] * n,
                "ETH/USDT": [0.2] * n,
                "SOL/USDT": [-0.5] * n,
            },
            index=idx,
        )

        result = vectorized_backtest(weights, price_panel)
        assert isinstance(result, BacktestResult)
        assert len(result.returns) > 0
        assert len(result.equity_curve) > 0

    def test_zero_weights_zero_returns(self):
        """全零权重 → 零收益"""
        symbols = ["BTC/USDT", "ETH/USDT"]
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")

        price_panel = _make_price_panel(n, symbols)
        weights = pd.DataFrame(0.0, index=idx, columns=symbols)

        result = vectorized_backtest(weights, price_panel)
        # 毛收益应为 0
        assert result.gross_returns.abs().max() < 1e-10

    def test_fee_reduces_returns(self):
        """手续费应降低净收益"""
        symbols = ["BTC/USDT", "ETH/USDT"]
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        rng = np.random.RandomState(42)

        price_panel = _make_price_panel(n, symbols)
        # 每期都换仓以产生手续费
        w = rng.randn(n, len(symbols)) * 0.3
        weights = pd.DataFrame(w, index=idx, columns=symbols)

        # 高费率 vs 低费率
        result_low = vectorized_backtest(weights, price_panel, fee_rate=0.0001)
        result_high = vectorized_backtest(weights, price_panel, fee_rate=0.001)

        # 高费率的总成本应更高
        assert result_high.total_cost > result_low.total_cost

    def test_no_common_index_raises(self):
        """无共同索引应报错"""
        idx1 = pd.date_range("2024-01-01", periods=10, freq="1min", tz="UTC")
        idx2 = pd.date_range("2025-01-01", periods=10, freq="1min", tz="UTC")
        weights = pd.DataFrame({"A": [0.5] * 10}, index=idx1)
        prices = pd.DataFrame({"A": [100] * 10}, index=idx2)
        with pytest.raises(ValueError, match="共同时间索引"):
            vectorized_backtest(weights, prices)


class TestMarketImpact:
    """市场冲击估计"""

    def test_zero_delta_zero_impact(self):
        """零权重变化 → 零冲击"""
        idx = pd.date_range("2024-01-01", periods=10, freq="1min", tz="UTC")
        delta = pd.DataFrame(0.0, index=idx, columns=["A", "B"])
        adv = pd.DataFrame(1e6, index=idx, columns=["A", "B"])
        vol = pd.DataFrame(0.01, index=idx, columns=["A", "B"])
        impact = estimate_market_impact(delta, adv, vol, 10000)
        assert impact.abs().max().max() < 1e-10

    def test_larger_delta_larger_impact(self):
        """更大的权重变化 → 更大的冲击"""
        idx = pd.date_range("2024-01-01", periods=10, freq="1min", tz="UTC")
        delta_small = pd.DataFrame(0.1, index=idx, columns=["A"])
        delta_large = pd.DataFrame(0.5, index=idx, columns=["A"])
        adv = pd.DataFrame(1e6, index=idx, columns=["A"])
        vol = pd.DataFrame(0.01, index=idx, columns=["A"])

        impact_small = estimate_market_impact(delta_small, adv, vol, 10000)
        impact_large = estimate_market_impact(delta_large, adv, vol, 10000)

        assert impact_large.values.mean() > impact_small.values.mean()

    def test_no_adv_only_fee_cost(self):
        """[T9] adv_panel=None 时只有手续费成本，无市场冲击"""
        symbols = ["BTC/USDT", "ETH/USDT"]
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        rng = np.random.RandomState(42)

        price_panel = _make_price_panel(n, symbols)
        weights = pd.DataFrame(
            rng.randn(n, len(symbols)) * 0.3,
            index=idx, columns=symbols,
        )

        result_no_adv = vectorized_backtest(
            weights, price_panel, fee_rate=0.001, adv_panel=None,
        )
        assert result_no_adv.total_cost > 0
        # 毛收益和净收益的差值应等于手续费总成本
        gross_sum = result_no_adv.gross_returns.sum()
        net_sum = result_no_adv.returns.sum()
        assert abs((gross_sum - net_sum) - result_no_adv.total_cost) < 1e-10


# ---------------------------------------------------------------------------
# 跨模块端到端一致性护栏：vectorized_backtest 全链路 impact 计算
# 必须与 execution_optimizer.cost.build_cost_expression 在同输入下一致
# ---------------------------------------------------------------------------

class TestImpactEndToEndConsistency:
    """
    端到端护栏：从 vectorized_backtest 入口出发计算 impact cost，
    与 cost.py 在同输入下的 impact 分量精确相等。

    这层护栏比 test_cost.py 的 TestImpactFormulaConsistency 更严格——
    它覆盖了 vectorized_backtest 内部 σ 尺度、ADV 对齐、delta_w.shift 等
    调用细节，能抓到"公式对但传入参数尺度错"这类 bug（历史上 σ 年化 vs
    日化就是这类，commit 3.5 修复）。
    """

    def test_vectorized_impact_matches_cost_py_single_trade(self):
        """
        精确数值比对：构造"仅在某一个 bar 发生一次调仓"的场景，
        断言该 bar 的 vectorized impact cost == cost.py 的 impact 值。
        """
        from execution_optimizer.config import MarketContext
        from execution_optimizer.cost import build_cost_expression
        import cvxpy as cp

        symbols = ["BTC/USDT", "ETH/USDT"]
        n = 120
        idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        rng = np.random.RandomState(7)

        rets = rng.randn(n, len(symbols)) * 0.001
        prices = 100 * np.exp(np.cumsum(rets, axis=0))
        price_panel = pd.DataFrame(prices, index=idx, columns=symbols)

        # weights: 前 79 行全 0；第 80 行开始 [0.1, -0.05]
        # 这样 delta_w 仅在第 80 行非零，其他行 impact=0
        trade_bar = 80
        w = np.zeros((n, len(symbols)))
        w[trade_bar:, 0] = 0.1
        w[trade_bar:, 1] = -0.05
        weights = pd.DataFrame(w, index=idx, columns=symbols)

        adv_value = 1_000_000.0
        adv_panel = pd.DataFrame(adv_value, index=idx, columns=symbols)
        V = 10_000.0
        impact_coeff = 0.1

        # --- 路径 1: vectorized_backtest ---
        result = vectorized_backtest(
            weights, price_panel,
            fee_rate=0.0, impact_coeff=impact_coeff,
            adv_panel=adv_panel, portfolio_value=V,
        )
        # total_cost 对应的就是 impact_cost（fee=0），且只发生在 trade_bar
        # net_returns.loc[trade_bar] = gross - impact
        impact_at_trade_vec = (
            result.gross_returns.iloc[trade_bar] - result.returns.iloc[trade_bar]
        )

        # --- 路径 2: cost.py，使用 vectorized 内部同样估出的 σ ---
        # 复现 vectorized 内部的 σ 计算：rolling(60).std() × √1440（日化）
        returns_panel = price_panel.pct_change()
        vol_panel = returns_panel.rolling(60, min_periods=10).std() * np.sqrt(1440)
        sigma_at_trade = vol_panel.iloc[trade_bar]  # pd.Series(index=symbols)

        ctx = MarketContext(
            timestamp=idx[trade_bar],
            symbols=symbols,
            spread=pd.Series([0.0, 0.0], index=symbols),   # 关 spread
            volatility=sigma_at_trade,                      # 日化 σ
            adv=pd.Series([adv_value, adv_value], index=symbols),
            portfolio_value=V,
        )

        delta_w_at_trade = np.array([0.1, -0.05])
        n_sym = len(symbols)
        dw_var = cp.Variable(n_sym)
        cost_expr = build_cost_expression(
            dw_var, ctx, impact_coeff=impact_coeff, fee_rate=0.0,
        )
        prob = cp.Problem(cp.Minimize(cost_expr), [dw_var == delta_w_at_trade])
        prob.solve()
        impact_at_trade_costpy = prob.value

        # --- 比对 ---
        np.testing.assert_allclose(
            impact_at_trade_vec, impact_at_trade_costpy, rtol=1e-4,
            err_msg=(
                f"vectorized_backtest 与 cost.py 的 impact 不一致:\n"
                f"  vectorized = {impact_at_trade_vec:.10f}\n"
                f"  cost.py    = {impact_at_trade_costpy:.10f}\n"
                f"  ratio      = {impact_at_trade_vec / impact_at_trade_costpy:.4f}\n"
                f"  (若 ratio ≈ 19.1，说明 σ 尺度问题回归)"
            )
        )


# ---------------------------------------------------------------------------
# 缩放护栏：periods_per_year 参数被正确透传到年化/日化系数
# ---------------------------------------------------------------------------

class TestPeriodsPerYearParameterization:
    """
    方案 2（periods_per_year 参数化）的护栏测试。

    核心思想：相同 returns / weights / prices 输入，改变 periods_per_year
    参数时，年化或日化相关的输出应按已知公式精确缩放。

    覆盖三个对外入口:
      - BacktestResult.summary()
      - vectorized_backtest()
      - apply_vol_target()  (ExecutionOptimizer 的测试在 test_optimizer.py)
    """

    def test_summary_annual_volatility_scales_with_sqrt_periods(self):
        """summary 的 annual_volatility 应按 √periods_per_year 缩放"""
        rng = np.random.RandomState(42)
        idx = pd.date_range("2024-01-01", periods=500, freq="1min", tz="UTC")
        returns = pd.Series(rng.randn(500) * 0.001, index=idx)

        result = BacktestResult(
            equity_curve=cumulative_returns(returns),
            returns=returns,
            turnover=pd.Series(0.0, index=idx),
            weights_history=pd.DataFrame(index=idx),
        )

        s_minute = result.summary(periods_per_year=525960.0)   # 1m
        s_fiveminute = result.summary(periods_per_year=105192.0)  # 5m

        # annual_vol = std × √periods → ratio = √(525960/105192) = √5 ≈ 2.236
        expected_ratio = np.sqrt(525960.0 / 105192.0)
        actual_ratio = s_minute["annual_volatility"] / s_fiveminute["annual_volatility"]
        np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=1e-6)

    def test_summary_default_matches_explicit_minutes_per_year(self):
        """默认参数行为等同于显式传 MINUTES_PER_YEAR（向后兼容）"""
        from alpha_model.config import MINUTES_PER_YEAR
        rng = np.random.RandomState(7)
        idx = pd.date_range("2024-01-01", periods=200, freq="1min", tz="UTC")
        returns = pd.Series(rng.randn(200) * 0.001, index=idx)

        result = BacktestResult(
            equity_curve=cumulative_returns(returns),
            returns=returns,
            turnover=pd.Series(0.0, index=idx),
            weights_history=pd.DataFrame(index=idx),
        )

        s_default = result.summary()
        s_explicit = result.summary(periods_per_year=MINUTES_PER_YEAR)

        for key in ("annual_return", "annual_volatility", "sharpe_ratio"):
            np.testing.assert_allclose(s_default[key], s_explicit[key], rtol=1e-12)

    def test_vectorized_backtest_impact_scales_with_bars_per_day(self):
        """
        vectorized_backtest 的 impact 使用 √(periods_per_year / 365.25) 作为
        bars_per_day 换算日化 σ。相同输入下，改变 periods_per_year 应按
        √ratio 缩放 impact。
        """
        symbols = ["BTC/USDT"]
        n = 120
        idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        rng = np.random.RandomState(11)

        rets = rng.randn(n, 1) * 0.001
        prices = 100 * np.exp(np.cumsum(rets, axis=0))
        price_panel = pd.DataFrame(prices, index=idx, columns=symbols)

        trade_bar = 80
        w = np.zeros((n, 1))
        w[trade_bar:, 0] = 0.1
        weights = pd.DataFrame(w, index=idx, columns=symbols)
        adv_panel = pd.DataFrame(1_000_000.0, index=idx, columns=symbols)

        result_1m = vectorized_backtest(
            weights, price_panel, fee_rate=0.0, adv_panel=adv_panel,
            portfolio_value=10_000.0, periods_per_year=525960.0,
        )
        result_5m = vectorized_backtest(
            weights, price_panel, fee_rate=0.0, adv_panel=adv_panel,
            portfolio_value=10_000.0, periods_per_year=105192.0,
        )

        impact_1m = (result_1m.gross_returns.iloc[trade_bar]
                     - result_1m.returns.iloc[trade_bar])
        impact_5m = (result_5m.gross_returns.iloc[trade_bar]
                     - result_5m.returns.iloc[trade_bar])

        # impact ∝ √(bars_per_day) = √(periods_per_year / 365.25)
        # ratio = √(525960/105192) = √5
        expected_ratio = np.sqrt(525960.0 / 105192.0)
        actual_ratio = impact_1m / impact_5m
        np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=1e-4)


class TestApplyVolTargetPeriodsPerYear:
    """apply_vol_target 的 periods_per_year 参数化护栏"""

    def test_apply_vol_target_scales_with_sqrt_periods(self):
        """
        apply_vol_target 的 rolling_vol 按 √periods_per_year 缩放。
        相同 vol_target 下，较大 periods_per_year 估出的 port_vol 更大，
        scale 因子较小，导致 weights 被更多压缩。
        """
        from alpha_model.portfolio.risk_budget import apply_vol_target

        symbols = ["A", "B"]
        n = 200
        idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        rng = np.random.RandomState(23)

        rets = rng.randn(n, 2) * 0.01  # 较大波动以触发 vol target
        prices = 100 * np.exp(np.cumsum(rets, axis=0))
        price_panel = pd.DataFrame(prices, index=idx, columns=symbols)
        weights = pd.DataFrame(
            np.tile([0.5, -0.5], (n, 1)), index=idx, columns=symbols,
        )

        # 同样的 vol_target=0.2 (年化 20%), 不同 periods_per_year
        adj_1m = apply_vol_target(
            weights, price_panel, vol_target=0.2, lookback=60,
            periods_per_year=525960.0,
        )
        adj_5m = apply_vol_target(
            weights, price_panel, vol_target=0.2, lookback=60,
            periods_per_year=105192.0,
        )

        # 取某个有效行（rolling_vol 已稳定）
        check_idx = idx[150]

        w_1m = adj_1m.loc[check_idx].abs().sum()
        w_5m = adj_5m.loc[check_idx].abs().sum()

        # 1m 声称年化系数更大 → port_vol 估得更大 → scale 更小 → |w| 更小
        # 具体 ratio: scale ∝ 1 / √periods → |w| ∝ 1/√periods
        # ratio = w_5m / w_1m = √(525960/105192) = √5 ≈ 2.236
        # 前提：未触发 leverage_cap
        expected_ratio = np.sqrt(525960.0 / 105192.0)
        actual_ratio = w_5m / w_1m
        # 若触发 leverage_cap 二次缩放，比例会偏离；测试构造保证未触发
        np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=1e-4,
            err_msg=f"w_1m={w_1m}, w_5m={w_5m}, expected ratio={expected_ratio}")
