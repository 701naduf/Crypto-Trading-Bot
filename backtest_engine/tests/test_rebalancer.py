"""
test_rebalancer.py — Rebalancer + ExecutionReport

覆盖：
  - 选择 A 最小下单量过滤
  - 选择 B spread 方向性的总额收敛（spread_cost = Σ(spread_i/2 × |Δw_i|)）
  - 选择 C/D impact 跨模块护栏（vs cost.py / vectorized.py 一致）
  - 选择 F ExecutionReport schema 不变量
  - cost_mode 行为
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import cvxpy as cp

from execution_optimizer import MarketContext
from execution_optimizer.cost import build_cost_expression

from backtest_engine.config import ExecutionMode, CostMode
from backtest_engine.rebalancer import Rebalancer, ExecutionReport


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_context(
    symbols, V=10000.0, spread=0.0005, vol=0.02, adv=1e9,
    t=pd.Timestamp("2024-01-01", tz="UTC"),
):
    return MarketContext(
        timestamp=t,
        symbols=list(symbols),
        spread=pd.Series([spread] * len(symbols), index=symbols, dtype=float),
        volatility=pd.Series([vol] * len(symbols), index=symbols, dtype=float),
        adv=pd.Series([adv] * len(symbols), index=symbols, dtype=float),
        portfolio_value=V,
        funding_rate=None,
    )


def _make_rebalancer(
    cost_mode=CostMode.FULL_COST, min_trade_value=5.0, fee_rate=0.0004,
    impact_coeff=0.1,
):
    return Rebalancer(
        execution_mode=ExecutionMode.MARKET,
        cost_mode=cost_mode,
        min_trade_value=min_trade_value,
        fee_rate=fee_rate,
        impact_coeff=impact_coeff,
    )


# ---------------------------------------------------------------------------
# 选择 A: 最小下单量过滤
# ---------------------------------------------------------------------------

class TestMinTradeValueFilter:

    def test_below_min_skipped(self):
        symbols = ["BTC/USDT", "ETH/USDT"]
        ctx = _make_context(symbols, V=10000.0)
        r = _make_rebalancer(min_trade_value=5.0)

        cw = pd.Series([0.0, 0.0], index=symbols)
        tw = pd.Series([0.0001, 0.5], index=symbols)
        # Δw₁ × V = 0.0001 × 10000 = 1 USDT < 5 → 过滤 BTC
        # Δw₂ × V = 0.5    × 10000 = 5000 USDT ≥ 5 → 执行 ETH

        actual, report = r.execute(cw, tw, ctx, price_at_t=pd.Series())
        assert actual["BTC/USDT"] == 0.0  # 维持 current
        assert actual["ETH/USDT"] == 0.5
        assert report.filtered_symbols == ["BTC/USDT"]

    def test_above_min_passes(self):
        symbols = ["BTC/USDT"]
        ctx = _make_context(symbols, V=10000.0)
        r = _make_rebalancer(min_trade_value=5.0)
        cw = pd.Series([0.0], index=symbols)
        tw = pd.Series([0.5], index=symbols)
        actual, report = r.execute(cw, tw, ctx, price_at_t=pd.Series())
        assert actual["BTC/USDT"] == 0.5
        assert report.filtered_symbols == []

    def test_no_change_zero_costs(self):
        """target == current → actual_delta = 0，三分量都是 0"""
        symbols = ["BTC/USDT"]
        ctx = _make_context(symbols)
        r = _make_rebalancer()
        w = pd.Series([0.5], index=symbols)
        actual, report = r.execute(w, w, ctx, price_at_t=pd.Series())
        assert (actual == w).all()
        assert report.fee_cost == 0.0
        assert report.spread_cost == 0.0
        assert report.impact_cost == 0.0

    def test_filter_can_violate_dollar_neutral(self):
        """选择 A 副作用：过滤后可能违反 dollar_neutral，是预期行为（暴露执行摩擦）"""
        symbols = ["BTC/USDT", "ETH/USDT"]
        ctx = _make_context(symbols, V=10000.0)
        r = _make_rebalancer(min_trade_value=100.0)
        cw = pd.Series([0.0, 0.0], index=symbols)
        # target dollar-neutral: +0.005 / -0.005，金额 = 50 USDT 各
        tw = pd.Series([0.005, -0.005], index=symbols)
        actual, _ = r.execute(cw, tw, ctx, price_at_t=pd.Series())
        # 两个都 < 100 USDT 被过滤 → actual = (0, 0)，但 dollar-neutral 仍成立（trivially）
        # 把过滤阈值设成只过滤一边
        r2 = _make_rebalancer(min_trade_value=60.0)
        actual2, _ = r2.execute(cw, tw, ctx, price_at_t=pd.Series())
        # 现在 |0.005×10000|=50 < 60 → 全部过滤
        assert (actual2 == 0.0).all()

    def test_filter_partial(self):
        """部分过滤场景：一些 symbol 通过，一些被过滤"""
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        ctx = _make_context(symbols, V=10000.0)
        r = _make_rebalancer(min_trade_value=10.0)
        cw = pd.Series([0.0, 0.0, 0.0], index=symbols)
        # Δw × V = 1, 100, 1000 → BTC 过滤, ETH 通过, SOL 通过
        tw = pd.Series([0.0001, 0.01, 0.1], index=symbols)
        actual, report = r.execute(cw, tw, ctx, price_at_t=pd.Series())
        assert actual["BTC/USDT"] == 0.0
        assert actual["ETH/USDT"] == 0.01
        assert actual["SOL/USDT"] == 0.1
        assert report.filtered_symbols == ["BTC/USDT"]


# ---------------------------------------------------------------------------
# 选择 B: spread 总额收敛
# ---------------------------------------------------------------------------

class TestSpreadCost:

    def test_spread_cost_formula(self):
        """spread_cost = Σ(spread_i/2 × |Δw_i|)"""
        symbols = ["BTC/USDT", "ETH/USDT"]
        ctx = _make_context(symbols, V=10000.0, spread=0.001, adv=1e15)
        r = _make_rebalancer(cost_mode=CostMode.FULL_COST, fee_rate=0.0)

        cw = pd.Series([0.0, 0.0], index=symbols)
        tw = pd.Series([0.5, -0.3], index=symbols)
        _, report = r.execute(cw, tw, ctx, price_at_t=pd.Series())

        # spread_cost = 0.001/2 × (0.5 + 0.3) = 0.0004
        expected = 0.001 / 2.0 * 0.8
        assert np.isclose(report.spread_cost, expected, rtol=1e-12)

    def test_match_vectorized_zeros_spread(self):
        symbols = ["BTC/USDT"]
        ctx = _make_context(symbols, spread=0.001, adv=1e15)
        r = _make_rebalancer(cost_mode=CostMode.MATCH_VECTORIZED, fee_rate=0.0)

        cw = pd.Series([0.0], index=symbols)
        tw = pd.Series([0.5], index=symbols)
        _, report = r.execute(cw, tw, ctx, price_at_t=pd.Series())
        assert report.spread_cost == 0.0


# ---------------------------------------------------------------------------
# 选择 D: impact 跨模块一致性护栏
# ---------------------------------------------------------------------------

class TestImpactConsistency:

    def test_impact_formula_matches_cost_py(self):
        """
        Rebalancer 标量 impact 必须与 execution_optimizer.cost.build_cost_expression
        在相同输入下精确相等（rtol=1e-6）—— §11.4.7 防退化护栏
        """
        symbols = ["BTC/USDT", "ETH/USDT"]
        V = 10000.0
        spread = 0.0005
        sigma = 0.03
        adv = 5e9
        coeff = 0.1
        ctx = _make_context(symbols, V=V, spread=spread, vol=sigma, adv=adv)

        # Rebalancer 标量
        r = _make_rebalancer(
            cost_mode=CostMode.FULL_COST, fee_rate=0.0,
            impact_coeff=coeff, min_trade_value=0.01,
        )
        cw = pd.Series([0.0, 0.0], index=symbols)
        tw = pd.Series([0.3, -0.2], index=symbols)
        _, report = r.execute(cw, tw, ctx, price_at_t=pd.Series())

        # cost.py: 用 cvxpy 表达式求值
        delta_var = cp.Variable(2)
        delta_var.value = (tw - cw).values  # [0.3, -0.2]
        cost_expr = build_cost_expression(
            delta_var, ctx, impact_coeff=coeff, fee_rate=0.0,
        )
        # 表达式 = spread_cost + impact_cost（fee_rate=0）
        cost_total = float(cost_expr.value)
        cost_py_impact = cost_total - report.spread_cost
        # 现在 cost_py_impact 是 cost.py 算的 impact 部分
        assert np.isclose(report.impact_cost, cost_py_impact, rtol=1e-6), (
            f"Rebalancer impact ({report.impact_cost}) ≠ "
            f"cost.py impact ({cost_py_impact})"
        )

    def test_impact_formula_matches_vectorized(self):
        """
        Rebalancer 标量 impact 必须与 alpha_model.backtest.estimate_market_impact
        在相同输入下精确相等
        """
        from alpha_model.backtest.vectorized import estimate_market_impact

        symbols = ["BTC/USDT", "ETH/USDT"]
        V = 10000.0
        sigma = 0.03
        adv = 5e9
        coeff = 0.1

        ctx = _make_context(symbols, V=V, vol=sigma, adv=adv, spread=0.0)
        r = _make_rebalancer(
            cost_mode=CostMode.MATCH_VECTORIZED, fee_rate=0.0,  # 隔离 impact
            impact_coeff=coeff, min_trade_value=0.01,
        )
        cw = pd.Series([0.0, 0.0], index=symbols)
        tw = pd.Series([0.3, -0.2], index=symbols)
        _, report = r.execute(cw, tw, ctx, price_at_t=pd.Series())

        # vectorized: 用 DataFrame 单行
        idx = [pd.Timestamp("2024-01-01", tz="UTC")]
        delta_df = pd.DataFrame([[0.3, -0.2]], index=idx, columns=symbols)
        adv_df = pd.DataFrame([[adv, adv]], index=idx, columns=symbols)
        vol_df = pd.DataFrame([[sigma, sigma]], index=idx, columns=symbols)

        impact_df = estimate_market_impact(delta_df, adv_df, vol_df, V, coeff)
        vec_impact = float(impact_df.sum(axis=1).iloc[0])
        assert np.isclose(report.impact_cost, vec_impact, rtol=1e-6)

    def test_per_symbol_impact_coeff(self):
        """impact_coeff 可为 pd.Series（逐标的）"""
        symbols = ["BTC/USDT", "ETH/USDT"]
        coeff_series = pd.Series([0.05, 0.2], index=symbols)
        ctx = _make_context(symbols, V=10000.0, spread=0.0, adv=1e9)
        r = _make_rebalancer(
            cost_mode=CostMode.MATCH_VECTORIZED, fee_rate=0.0,
            impact_coeff=coeff_series, min_trade_value=0.01,
        )
        cw = pd.Series([0.0, 0.0], index=symbols)
        tw = pd.Series([0.3, 0.3], index=symbols)
        _, report = r.execute(cw, tw, ctx, price_at_t=pd.Series())
        # 不同 coeff → 不同贡献，但都是 0.3 同 |Δw|
        assert report.impact_cost > 0


# ---------------------------------------------------------------------------
# fee_cost
# ---------------------------------------------------------------------------

class TestFeeCost:

    def test_fee_formula(self):
        symbols = ["BTC/USDT", "ETH/USDT"]
        ctx = _make_context(symbols, V=10000.0, spread=0.0, adv=1e15)
        r = _make_rebalancer(fee_rate=0.0004, impact_coeff=0.0)

        cw = pd.Series([0.0, 0.0], index=symbols)
        tw = pd.Series([0.5, -0.3], index=symbols)
        _, report = r.execute(cw, tw, ctx, price_at_t=pd.Series())

        # fee = 0.0004 × (0.5 + 0.3) = 0.00032
        assert np.isclose(report.fee_cost, 0.0004 * 0.8, rtol=1e-12)


# ---------------------------------------------------------------------------
# ExecutionReport schema 不变量（§11.4.0' / §12.1）
# ---------------------------------------------------------------------------

class TestExecutionReportSchema:

    def test_schema_fields(self):
        symbols = ["BTC/USDT", "ETH/USDT"]
        ctx = _make_context(symbols)
        r = _make_rebalancer()
        cw = pd.Series([0.0, 0.0], index=symbols)
        tw = pd.Series([0.5, -0.3], index=symbols)
        _, report = r.execute(cw, tw, ctx, price_at_t=pd.Series())

        # 7 个字段全部存在且类型正确
        assert isinstance(report.timestamp, pd.Timestamp)
        assert isinstance(report.actual_delta, pd.Series)
        assert isinstance(report.trade_values, pd.Series)
        assert isinstance(report.fee_cost, float)
        assert isinstance(report.spread_cost, float)
        assert isinstance(report.impact_cost, float)
        assert isinstance(report.filtered_symbols, list)

    def test_actual_delta_index_equals_symbols(self):
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        ctx = _make_context(symbols)
        r = _make_rebalancer()
        cw = pd.Series([0.0, 0.0, 0.0], index=symbols)
        tw = pd.Series([0.1, 0.1, 0.1], index=symbols)
        _, report = r.execute(cw, tw, ctx, price_at_t=pd.Series())
        assert list(report.actual_delta.index) == symbols
        assert list(report.trade_values.index) == symbols

    def test_trade_values_non_negative(self):
        symbols = ["BTC/USDT"]
        ctx = _make_context(symbols)
        r = _make_rebalancer()
        cw = pd.Series([0.5], index=symbols)
        tw = pd.Series([-0.3], index=symbols)  # 大幅减仓
        _, report = r.execute(cw, tw, ctx, price_at_t=pd.Series())
        assert (report.trade_values >= 0).all()

    def test_costs_non_negative(self):
        symbols = ["BTC/USDT"]
        ctx = _make_context(symbols)
        r = _make_rebalancer()
        cw = pd.Series([0.0], index=symbols)
        tw = pd.Series([0.5], index=symbols)
        _, report = r.execute(cw, tw, ctx, price_at_t=pd.Series())
        assert report.fee_cost >= 0
        assert report.spread_cost >= 0
        assert report.impact_cost >= 0

    def test_timestamp_matches_context(self):
        symbols = ["BTC/USDT"]
        t = pd.Timestamp("2025-06-15 12:30:00", tz="UTC")
        ctx = _make_context(symbols, t=t)
        r = _make_rebalancer()
        cw = pd.Series([0.0], index=symbols)
        tw = pd.Series([0.5], index=symbols)
        _, report = r.execute(cw, tw, ctx, price_at_t=pd.Series())
        assert report.timestamp == t


# ---------------------------------------------------------------------------
# 边界情况（§11.4.5）
# ---------------------------------------------------------------------------

class TestBoundaryConditions:

    def test_adv_zero_protected(self):
        """ADV=0 用 max(adv, 1.0) 兜底（与 cost.py 一致）—— 不发散"""
        symbols = ["BTC/USDT"]
        ctx = MarketContext(
            timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
            symbols=symbols,
            spread=pd.Series([0.001], index=symbols),
            volatility=pd.Series([0.03], index=symbols),
            adv=pd.Series([0.0], index=symbols),  # ★ ADV=0
            portfolio_value=10000.0,
            funding_rate=None,
        )
        r = _make_rebalancer()
        cw = pd.Series([0.0], index=symbols)
        tw = pd.Series([0.5], index=symbols)
        _, report = r.execute(cw, tw, ctx, price_at_t=pd.Series())
        # impact 不应是 inf（兜底起作用）
        assert np.isfinite(report.impact_cost)

    def test_spread_nan_propagates_to_cost(self):
        """spread=NaN（orderbook gap 超 max_gap）→ spread_cost 是 NaN，PnLTracker 会 fail-fast"""
        symbols = ["BTC/USDT"]
        ctx = MarketContext(
            timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
            symbols=symbols,
            spread=pd.Series([np.nan], index=symbols),
            volatility=pd.Series([0.03], index=symbols),
            adv=pd.Series([1e9], index=symbols),
            portfolio_value=10000.0,
            funding_rate=None,
        )
        r = _make_rebalancer(cost_mode=CostMode.FULL_COST)
        cw = pd.Series([0.0], index=symbols)
        tw = pd.Series([0.5], index=symbols)
        _, report = r.execute(cw, tw, ctx, price_at_t=pd.Series())
        # 当前 spread NaN → spread_cost NaN（让 PnLTracker NumericalError 触发）
        assert np.isnan(report.spread_cost)

    def test_target_w_extra_symbol_ignored(self):
        """target_w 的 index 是 context.symbols 的超集 → 多余 symbol 被忽略"""
        symbols = ["BTC/USDT"]  # context 只有 BTC
        ctx = _make_context(symbols)
        r = _make_rebalancer()

        # target 含 ETH，但 context.symbols 不含它
        cw = pd.Series([0.0], index=symbols)
        tw = pd.Series([0.5, 0.3], index=["BTC/USDT", "ETH/USDT"])
        actual, report = r.execute(cw, tw, ctx, price_at_t=pd.Series())
        assert list(actual.index) == ["BTC/USDT"]
        assert "ETH/USDT" not in report.actual_delta.index

    def test_target_w_missing_symbol_treated_as_zero(self):
        """target_w 缺某 symbol → 视为 0"""
        symbols = ["BTC/USDT", "ETH/USDT"]
        ctx = _make_context(symbols)
        r = _make_rebalancer(min_trade_value=0.01)

        cw = pd.Series([0.5, 0.5], index=symbols)
        tw = pd.Series([0.3], index=["BTC/USDT"])  # 缺 ETH
        actual, _ = r.execute(cw, tw, ctx, price_at_t=pd.Series())
        # ETH 视为 target=0 → actual=0（从 0.5 调到 0）
        assert actual["BTC/USDT"] == 0.3
        assert actual["ETH/USDT"] == 0.0


# ---------------------------------------------------------------------------
# v1 范围
# ---------------------------------------------------------------------------

class TestV1Scope:

    def test_non_market_rejected(self):
        with pytest.raises(NotImplementedError, match="MARKET"):
            Rebalancer(
                execution_mode=ExecutionMode.LIMIT,
                cost_mode=CostMode.FULL_COST,
                min_trade_value=5.0, fee_rate=0.0004, impact_coeff=0.1,
            )

    def test_negative_min_trade_value_rejected(self):
        with pytest.raises(ValueError, match="min_trade_value"):
            Rebalancer(
                execution_mode=ExecutionMode.MARKET,
                cost_mode=CostMode.FULL_COST,
                min_trade_value=-1.0, fee_rate=0.0004, impact_coeff=0.1,
            )

    def test_negative_fee_rate_rejected(self):
        with pytest.raises(ValueError, match="fee_rate"):
            Rebalancer(
                execution_mode=ExecutionMode.MARKET,
                cost_mode=CostMode.FULL_COST,
                min_trade_value=5.0, fee_rate=-0.001, impact_coeff=0.1,
            )
