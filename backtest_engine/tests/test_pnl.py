"""
test_pnl.py — PnLTracker

覆盖 §11.5 的所有核心设计选择 + §11.5.5 v3 修订（NumericalError vs is_bankrupt 分流）+
§11.5.9 与 vectorized_backtest 的零摩擦一致性护栏。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_model.backtest.performance import BacktestResult
from alpha_model.backtest.vectorized import vectorized_backtest

from backtest_engine.pnl import PnLTracker, NumericalError
from backtest_engine.rebalancer import ExecutionReport


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _zero_exec_report(symbols, t):
    return ExecutionReport(
        timestamp=t,
        actual_delta=pd.Series(0.0, index=symbols, dtype=float),
        trade_values=pd.Series(0.0, index=symbols, dtype=float),
        fee_cost=0.0, spread_cost=0.0, impact_cost=0.0,
        filtered_symbols=[],
    )


def _exec_report(symbols, t, fee=0.0, spread=0.0, impact=0.0,
                 actual_delta=None, filtered=None):
    if actual_delta is None:
        actual_delta = pd.Series(0.0, index=symbols, dtype=float)
    return ExecutionReport(
        timestamp=t,
        actual_delta=actual_delta,
        trade_values=actual_delta.abs() * 10000.0,
        fee_cost=fee, spread_cost=spread, impact_cost=impact,
        filtered_symbols=filtered or [],
    )


# ---------------------------------------------------------------------------
# 基本初始化
# ---------------------------------------------------------------------------

class TestInit:

    def test_default_state(self):
        t = PnLTracker(initial_portfolio_value=10000.0)
        assert t.portfolio_value == 10000.0
        assert t.is_bankrupt is False
        assert t.bankruptcy_timestamp is None
        assert len(t.fee_series) == 0
        assert len(t.funding_events) == 0

    def test_negative_initial_v_raises(self):
        with pytest.raises(ValueError):
            PnLTracker(initial_portfolio_value=-1.0)

    def test_zero_initial_v_raises(self):
        with pytest.raises(ValueError):
            PnLTracker(initial_portfolio_value=0.0)


# ---------------------------------------------------------------------------
# record() 基本行为
# ---------------------------------------------------------------------------

class TestRecord:

    def test_first_record_gross_zero(self):
        """首 bar gross_rate = 0（_prev_actual_w is None）"""
        symbols = ["BTC/USDT"]
        t = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        tracker = PnLTracker(10000.0)
        actual_w = pd.Series([0.5], index=symbols)
        price = pd.Series([100.0], index=symbols)
        report = _zero_exec_report(symbols, t)

        tracker.record(t, actual_w, price, report)
        # V 不变（gross=0, cost=0, funding=0）
        assert tracker.portfolio_value == 10000.0

    def test_v_updates_with_gross(self):
        """V_t = V_{t-1} × (1 + gross - cost - funding)"""
        symbols = ["BTC/USDT"]
        ts = pd.date_range("2024-01-01", periods=2, freq="1min", tz="UTC")
        tracker = PnLTracker(10000.0)

        # bar 0: 持有 0.5 BTC
        tracker.record(
            ts[0], pd.Series([0.5], index=symbols),
            pd.Series([100.0], index=symbols), _zero_exec_report(symbols, ts[0]),
        )
        # bar 1: 价格涨 1%
        tracker.record(
            ts[1], pd.Series([0.5], index=symbols),
            pd.Series([101.0], index=symbols), _zero_exec_report(symbols, ts[1]),
        )
        # gross = 0.5 × 0.01 = 0.005 → V = 10000 × 1.005 = 10050
        assert np.isclose(tracker.portfolio_value, 10050.0, rtol=1e-12)

    def test_cost_subtracted_from_v(self):
        symbols = ["BTC/USDT"]
        ts = pd.date_range("2024-01-01", periods=2, freq="1min", tz="UTC")
        tracker = PnLTracker(10000.0)

        tracker.record(
            ts[0], pd.Series([0.5], index=symbols),
            pd.Series([100.0], index=symbols), _zero_exec_report(symbols, ts[0]),
        )
        # 0.001 cost
        tracker.record(
            ts[1], pd.Series([0.5], index=symbols),
            pd.Series([100.0], index=symbols),
            _exec_report(symbols, ts[1], fee=0.0005, spread=0.0003, impact=0.0002),
        )
        assert np.isclose(tracker.portfolio_value, 10000.0 * (1 - 0.001), rtol=1e-12)

    def test_compute_result_requires_records(self):
        tracker = PnLTracker(10000.0)
        with pytest.raises(RuntimeError):
            tracker.compute_backtest_result(pd.DataFrame(), 525960)


# ---------------------------------------------------------------------------
# Funding（选择 B + C）
# ---------------------------------------------------------------------------

class TestFunding:

    def test_apply_funding_settlement_records_rate(self):
        symbols = ["BTC/USDT"]
        t = pd.Timestamp("2024-01-01 08:00:00", tz="UTC")
        tracker = PnLTracker(10000.0)

        cw = pd.Series([0.5], index=symbols)
        rates = pd.Series([0.0001], index=symbols)
        tracker.apply_funding_settlement(t, cw, rates)

        # rate_total = 0.5 × 0.0001 = 5e-5
        assert np.isclose(tracker.funding_events.iloc[0], 5e-5, rtol=1e-12)
        assert np.isclose(tracker.portfolio_value, 10000.0 * (1 - 5e-5), rtol=1e-12)

    def test_funding_negative_for_short(self):
        """Short 持仓在 positive funding 下收款"""
        symbols = ["BTC/USDT"]
        t = pd.Timestamp("2024-01-01 08:00:00", tz="UTC")
        tracker = PnLTracker(10000.0)

        cw = pd.Series([-0.5], index=symbols)  # short
        rates = pd.Series([0.0001], index=symbols)
        tracker.apply_funding_settlement(t, cw, rates)
        # rate_total = -0.5 × 0.0001 = -5e-5（负值=收款）
        assert tracker.funding_events.iloc[0] < 0
        assert tracker.portfolio_value > 10000.0  # V 增加

    def test_funding_then_record_uses_v_before_funding(self):
        """选择 B 方案 Z：record 用 v_before_funding 做基准（精确公式）"""
        symbols = ["BTC/USDT"]
        ts = pd.date_range("2024-01-01", periods=2, freq="1min", tz="UTC")
        tracker = PnLTracker(10000.0)

        # bar 0
        tracker.record(
            ts[0], pd.Series([0.5], index=symbols),
            pd.Series([100.0], index=symbols), _zero_exec_report(symbols, ts[0]),
        )
        v_before = tracker.portfolio_value  # 10000

        # bar 1: funding event + price unchanged + zero cost
        # apply_funding_settlement first: V 减小
        rates = pd.Series([0.0001], index=symbols)
        tracker.apply_funding_settlement(
            ts[1], pd.Series([0.5], index=symbols), rates,
        )
        # 然后 record(同 bar)
        tracker.record(
            ts[1], pd.Series([0.5], index=symbols),
            pd.Series([100.0], index=symbols), _zero_exec_report(symbols, ts[1]),
        )

        # 精确公式: V = v_before × (1 + 0 - 5e-5 - 0)
        expected = v_before * (1.0 - 5e-5)
        assert np.isclose(tracker.portfolio_value, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# 破产 vs NumericalError（§11.5.5 v3 修订）
# ---------------------------------------------------------------------------

class TestBankruptcyVsNumerical:

    def test_v_le_zero_finite_sets_flag(self):
        """V ≤ 0 但 finite → is_bankrupt=True，不抛异常（保留 equity_curve 截断）"""
        symbols = ["BTC/USDT"]
        ts = pd.date_range("2024-01-01", periods=2, freq="1min", tz="UTC")
        tracker = PnLTracker(10000.0)

        tracker.record(
            ts[0], pd.Series([0.5], index=symbols),
            pd.Series([100.0], index=symbols), _zero_exec_report(symbols, ts[0]),
        )

        # 极端亏损：100% 成本
        tracker.record(
            ts[1], pd.Series([0.5], index=symbols),
            pd.Series([100.0], index=symbols),
            _exec_report(symbols, ts[1], fee=2.0),  # cost=200%，V 为负
        )
        assert tracker.is_bankrupt
        assert tracker.bankruptcy_timestamp == ts[1]
        assert np.isfinite(tracker.portfolio_value)

    def test_nan_v_raises_numerical(self):
        """V = NaN → 抛 NumericalError，不进入 is_bankrupt 通道"""
        symbols = ["BTC/USDT"]
        ts = pd.date_range("2024-01-01", periods=2, freq="1min", tz="UTC")
        tracker = PnLTracker(10000.0)

        tracker.record(
            ts[0], pd.Series([0.5], index=symbols),
            pd.Series([100.0], index=symbols), _zero_exec_report(symbols, ts[0]),
        )
        # spread_cost=NaN（模拟 orderbook gap 超 max_gap 的真实场景，§11.4.5）→ V=NaN
        with pytest.raises(NumericalError):
            tracker.record(
                ts[1], pd.Series([0.5], index=symbols),
                pd.Series([100.0], index=symbols),
                _exec_report(symbols, ts[1], spread=np.nan),
            )
        # is_bankrupt 仍是 False（NumericalError 通道）
        assert not tracker.is_bankrupt

    def test_inf_v_raises_numerical(self):
        symbols = ["BTC/USDT"]
        ts = pd.date_range("2024-01-01", periods=2, freq="1min", tz="UTC")
        tracker = PnLTracker(10000.0)

        tracker.record(
            ts[0], pd.Series([0.5], index=symbols),
            pd.Series([100.0], index=symbols), _zero_exec_report(symbols, ts[0]),
        )
        # impact_cost = -inf 让 V 爆 inf
        with pytest.raises(NumericalError):
            tracker.record(
                ts[1], pd.Series([0.5], index=symbols),
                pd.Series([100.0], index=symbols),
                _exec_report(symbols, ts[1], impact=-np.inf),
            )


# ---------------------------------------------------------------------------
# 选择 D: 紧凑存储 + properties
# ---------------------------------------------------------------------------

class TestPropertiesAndStorage:

    def test_fee_spread_impact_series_separate(self):
        symbols = ["BTC/USDT"]
        ts = pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC")
        tracker = PnLTracker(10000.0)

        for i, t in enumerate(ts):
            tracker.record(
                t, pd.Series([0.5], index=symbols),
                pd.Series([100.0], index=symbols),
                _exec_report(symbols, t, fee=0.001 + i * 0.0001),
            )

        fees = tracker.fee_series
        assert len(fees) == 3
        assert np.isclose(fees.iloc[0], 0.001, rtol=1e-12)
        assert np.isclose(fees.iloc[2], 0.0012, rtol=1e-12)

    def test_filter_statistics_accumulate(self):
        symbols = ["BTC/USDT", "ETH/USDT"]
        ts = pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC")
        tracker = PnLTracker(10000.0)

        for i, t in enumerate(ts):
            filtered = ["BTC/USDT"] if i % 2 == 0 else []
            tracker.record(
                t, pd.Series([0.5, 0.5], index=symbols),
                pd.Series([100.0, 100.0], index=symbols),
                _exec_report(symbols, t, filtered=filtered),
            )
        stats = tracker.filter_statistics
        assert stats["BTC/USDT"] == 2  # bar 0 + 2

    def test_portfolio_value_history(self):
        symbols = ["BTC/USDT"]
        ts = pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC")
        tracker = PnLTracker(10000.0)
        for t in ts:
            tracker.record(
                t, pd.Series([0.5], index=symbols),
                pd.Series([100.0], index=symbols), _zero_exec_report(symbols, t),
            )
        hist = tracker.portfolio_value_history
        assert len(hist) == 3
        # V 不变（gross=0, cost=0, funding=0）
        assert (hist == 10000.0).all()


# ---------------------------------------------------------------------------
# §11.5.9 零摩擦 vs vectorized_backtest 一致性护栏
# ---------------------------------------------------------------------------

class TestZeroFrictionMatchesVectorized:

    def test_zero_friction_matches_vectorized_one_symbol(self):
        """单 symbol，weights 恒定，无 funding/cost → equity 与 vectorized 数值一致"""
        symbols = ["BTC/USDT"]
        n = 100
        ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        rng = np.random.default_rng(0)
        prices = pd.DataFrame(
            {"BTC/USDT": 100.0 * (1 + rng.normal(0, 0.001, n)).cumprod()},
            index=ts,
        )
        weights = pd.DataFrame({"BTC/USDT": np.full(n, 0.5)}, index=ts)

        # vectorized
        vec_result = vectorized_backtest(
            weights=weights, price_panel=prices,
            fee_rate=0.0, impact_coeff=0.0, portfolio_value=10000.0,
        )

        # event-driven 模拟（等价于 FIXED_GAMMA + 完美执行 + 无 funding）
        tracker = PnLTracker(10000.0)
        for i, t in enumerate(ts):
            actual_w = weights.iloc[i]
            price_t = prices.iloc[i]
            tracker.record(t, actual_w, price_t, _zero_exec_report(symbols, t))

        ed_result = tracker.compute_backtest_result(prices, 525960)

        # equity_curve 数值一致（rtol=1e-10）
        # vec_result.equity_curve 是 1.0-起始；ed_result.equity_curve 是 V-起始
        # 所以比较时归一化
        vec_eq = vec_result.equity_curve / vec_result.equity_curve.iloc[0]
        ed_eq = ed_result.equity_curve / ed_result.equity_curve.iloc[0]
        # 两者 returns 应一致
        # 注意：vectorized 的 portfolio_gross 在 t=0 是 NaN，pct_change 自然
        common = vec_result.returns.dropna().index.intersection(ed_result.returns.index)
        pd.testing.assert_series_equal(
            vec_result.returns.loc[common].fillna(0.0),
            ed_result.returns.loc[common],
            rtol=1e-10, check_names=False, check_freq=False,
        )


# ---------------------------------------------------------------------------
# compute_backtest_result schema
# ---------------------------------------------------------------------------

class TestBacktestResultSchema:

    def test_returns_backtestresult_instance(self):
        symbols = ["BTC/USDT"]
        ts = pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC")
        tracker = PnLTracker(10000.0)
        for t in ts:
            tracker.record(
                t, pd.Series([0.5], index=symbols),
                pd.Series([100.0], index=symbols), _zero_exec_report(symbols, t),
            )
        prices = pd.DataFrame({"BTC/USDT": [100.0, 100.0, 100.0]}, index=ts)
        result = tracker.compute_backtest_result(prices, 525960)
        assert isinstance(result, BacktestResult)
        assert isinstance(result.equity_curve, pd.Series)
        assert isinstance(result.weights_history, pd.DataFrame)
        assert isinstance(result.gross_returns, pd.Series)
