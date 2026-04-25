"""
test_weights_source.py — Protocol + 两实现

覆盖 §11.3.4 PrecomputedWeights、§11.3.5 OnlineOptimizer、cost_mode 行为。
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from alpha_model.core.types import PortfolioConstraints
from execution_optimizer import ExecutionOptimizer, MarketContext

from backtest_engine.config import CostMode
from backtest_engine.weights_source import (
    WeightsSource, PrecomputedWeights, OnlineOptimizer,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_context(symbols, t=None, spread_value=0.0005):
    return MarketContext(
        timestamp=t if t else pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
        symbols=list(symbols),
        spread=pd.Series([spread_value] * len(symbols), index=symbols),
        volatility=pd.Series([0.03] * len(symbols), index=symbols),
        adv=pd.Series([1e9] * len(symbols), index=symbols),
        portfolio_value=10000.0,
        funding_rate=None,
    )


def _make_weights_panel(symbols, n_bars=100):
    idx = pd.date_range("2024-01-01", periods=n_bars, freq="1min", tz="UTC")
    rng = np.random.default_rng(0)
    data = rng.uniform(-0.3, 0.3, (n_bars, len(symbols)))
    return pd.DataFrame(data, index=idx, columns=symbols)


# ---------------------------------------------------------------------------
# Protocol 契约
# ---------------------------------------------------------------------------

class TestProtocol:

    def test_precomputed_satisfies_protocol(self):
        symbols = ["BTC/USDT", "ETH/USDT"]
        impl = PrecomputedWeights(_make_weights_panel(symbols))
        assert isinstance(impl, WeightsSource)

    def test_online_satisfies_protocol(self):
        symbols = ["BTC/USDT", "ETH/USDT"]
        opt = ExecutionOptimizer(constraints=PortfolioConstraints())
        impl = OnlineOptimizer(opt, _make_weights_panel(symbols), CostMode.FULL_COST)
        assert isinstance(impl, WeightsSource)


# ---------------------------------------------------------------------------
# PrecomputedWeights
# ---------------------------------------------------------------------------

class TestPrecomputedWeights:

    def test_returns_panel_row(self):
        symbols = ["BTC/USDT", "ETH/USDT"]
        panel = _make_weights_panel(symbols)
        impl = PrecomputedWeights(panel)

        t = panel.index[10]
        ctx = _make_context(symbols, t=t)
        target = impl.get_target_weights(
            t, current_weights=pd.Series(0.0, index=symbols),
            context=ctx, price_history=pd.DataFrame(),
        )
        pd.testing.assert_series_equal(
            target, panel.loc[t].astype(float), check_names=False,
        )

    def test_ignores_current_weights(self):
        """如实重放：不同 current_w 不影响 target_w（§11.3.2）"""
        symbols = ["BTC/USDT", "ETH/USDT"]
        panel = _make_weights_panel(symbols)
        impl = PrecomputedWeights(panel)
        t = panel.index[10]
        ctx = _make_context(symbols, t=t)

        a = impl.get_target_weights(
            t, current_weights=pd.Series(0.0, index=symbols),
            context=ctx, price_history=pd.DataFrame(),
        )
        b = impl.get_target_weights(
            t, current_weights=pd.Series([0.5, -0.5], index=symbols),
            context=ctx, price_history=pd.DataFrame(),
        )
        pd.testing.assert_series_equal(a, b)

    def test_missing_t_raises_keyerror(self):
        symbols = ["BTC/USDT"]
        panel = _make_weights_panel(symbols)
        impl = PrecomputedWeights(panel)
        ctx = _make_context(symbols)

        bad_t = pd.Timestamp("2099-01-01", tz="UTC")
        with pytest.raises(KeyError, match="找不到"):
            impl.get_target_weights(
                bad_t, pd.Series(0.0, index=symbols), ctx, pd.DataFrame(),
            )

    def test_missing_symbol_raises_keyerror(self):
        panel = _make_weights_panel(["BTC/USDT"])
        impl = PrecomputedWeights(panel)
        # context 含 panel 中不存在的 symbol
        ctx = _make_context(["BTC/USDT", "DOGE/USDT"])
        t = panel.index[0]
        with pytest.raises(KeyError, match="DOGE"):
            impl.get_target_weights(
                t, pd.Series(0.0, index=["BTC/USDT", "DOGE/USDT"]),
                ctx, pd.DataFrame(),
            )

    def test_nan_filled_with_zero(self):
        """缺失的 symbol 值填 0.0（预计算阶段缺失=未进入组合）"""
        idx = pd.date_range("2024-01-01", periods=5, freq="1min", tz="UTC")
        panel = pd.DataFrame(
            {"BTC/USDT": [0.5, np.nan, 0.5, 0.5, 0.5],
             "ETH/USDT": [0.3, 0.3, 0.3, 0.3, 0.3]},
            index=idx,
        )
        impl = PrecomputedWeights(panel)
        ctx = _make_context(["BTC/USDT", "ETH/USDT"], t=idx[1])
        target = impl.get_target_weights(
            idx[1], pd.Series(0.0, index=["BTC/USDT", "ETH/USDT"]),
            ctx, pd.DataFrame(),
        )
        assert target["BTC/USDT"] == 0.0  # NaN -> 0
        assert target["ETH/USDT"] == 0.3

    def test_empty_panel_raises(self):
        with pytest.raises(ValueError, match="为空"):
            PrecomputedWeights(pd.DataFrame())

    def test_non_datetime_index_raises(self):
        panel = pd.DataFrame({"BTC/USDT": [0.1, 0.2]}, index=[0, 1])
        with pytest.raises(ValueError, match="DatetimeIndex"):
            PrecomputedWeights(panel)

    def test_multiindex_columns_rejected(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC")
        cols = pd.MultiIndex.from_tuples([("BTC", "x"), ("BTC", "y")])
        panel = pd.DataFrame(np.zeros((3, 2)), index=idx, columns=cols)
        with pytest.raises(ValueError, match="MultiIndex"):
            PrecomputedWeights(panel)

    def test_duplicate_index_rejected(self):
        idx = pd.DatetimeIndex(
            ["2024-01-01", "2024-01-01", "2024-01-02"], tz="UTC",
        )
        panel = pd.DataFrame({"BTC/USDT": [0.1, 0.2, 0.3]}, index=idx)
        with pytest.raises(ValueError, match="unique"):
            PrecomputedWeights(panel)


# ---------------------------------------------------------------------------
# OnlineOptimizer
# ---------------------------------------------------------------------------

class TestOnlineOptimizer:

    def test_delegates_to_optimizer(self):
        symbols = ["BTC/USDT", "ETH/USDT"]
        opt = MagicMock(spec=ExecutionOptimizer)
        expected = pd.Series([0.5, -0.5], index=symbols)
        opt.optimize_step.return_value = expected

        signals = _make_weights_panel(symbols)
        impl = OnlineOptimizer(opt, signals, CostMode.FULL_COST)

        t = signals.index[5]
        ctx = _make_context(symbols, t=t)
        result = impl.get_target_weights(
            t, pd.Series(0.0, index=symbols), ctx, pd.DataFrame(),
        )

        pd.testing.assert_series_equal(result, expected)
        opt.optimize_step.assert_called_once()

    def test_full_cost_passes_raw_spread(self):
        """FULL_COST 下 optimizer 收到的 context.spread = 真实 spread"""
        symbols = ["BTC/USDT", "ETH/USDT"]
        opt = MagicMock(spec=ExecutionOptimizer)
        opt.optimize_step.return_value = pd.Series(0.0, index=symbols)

        signals = _make_weights_panel(symbols)
        impl = OnlineOptimizer(opt, signals, CostMode.FULL_COST)

        t = signals.index[5]
        ctx = _make_context(symbols, t=t, spread_value=0.001)
        impl.get_target_weights(
            t, pd.Series(0.0, index=symbols), ctx, pd.DataFrame(),
        )

        ctx_passed = opt.optimize_step.call_args.kwargs["market_context"]
        assert (ctx_passed.spread == 0.001).all()

    def test_match_vectorized_zeros_spread(self):
        """MATCH_VECTORIZED 下 optimizer 收到的 context.spread = 0"""
        symbols = ["BTC/USDT", "ETH/USDT"]
        opt = MagicMock(spec=ExecutionOptimizer)
        opt.optimize_step.return_value = pd.Series(0.0, index=symbols)

        signals = _make_weights_panel(symbols)
        impl = OnlineOptimizer(opt, signals, CostMode.MATCH_VECTORIZED)

        t = signals.index[5]
        ctx = _make_context(symbols, t=t, spread_value=0.001)
        impl.get_target_weights(
            t, pd.Series(0.0, index=symbols), ctx, pd.DataFrame(),
        )

        ctx_passed = opt.optimize_step.call_args.kwargs["market_context"]
        assert (ctx_passed.spread == 0.0).all()

    def test_match_vectorized_does_not_mutate_input(self):
        """clone 而非 mutate：原始 context 不变"""
        symbols = ["BTC/USDT"]
        opt = MagicMock(spec=ExecutionOptimizer)
        opt.optimize_step.return_value = pd.Series(0.0, index=symbols)

        signals = _make_weights_panel(symbols)
        impl = OnlineOptimizer(opt, signals, CostMode.MATCH_VECTORIZED)

        t = signals.index[5]
        ctx = _make_context(symbols, t=t, spread_value=0.001)
        original_spread = ctx.spread.copy()
        impl.get_target_weights(
            t, pd.Series(0.0, index=symbols), ctx, pd.DataFrame(),
        )
        pd.testing.assert_series_equal(ctx.spread, original_spread)

    def test_missing_t_raises(self):
        symbols = ["BTC/USDT"]
        opt = ExecutionOptimizer(constraints=PortfolioConstraints())
        signals = _make_weights_panel(symbols)
        impl = OnlineOptimizer(opt, signals, CostMode.FULL_COST)

        ctx = _make_context(symbols)
        bad_t = pd.Timestamp("2099-01-01", tz="UTC")
        with pytest.raises(KeyError, match="找不到"):
            impl.get_target_weights(
                bad_t, pd.Series(0.0, index=symbols), ctx, pd.DataFrame(),
            )

    def test_non_optimizer_rejected(self):
        with pytest.raises(TypeError, match="ExecutionOptimizer"):
            OnlineOptimizer("not an optimizer", _make_weights_panel(["BTC/USDT"]),
                            CostMode.FULL_COST)

    def test_empty_signals_rejected(self):
        opt = ExecutionOptimizer(constraints=PortfolioConstraints())
        with pytest.raises(ValueError, match="为空"):
            OnlineOptimizer(opt, pd.DataFrame(), CostMode.FULL_COST)

    def test_non_datetime_signals_index_rejected(self):
        opt = ExecutionOptimizer(constraints=PortfolioConstraints())
        signals = pd.DataFrame({"BTC/USDT": [0.1, 0.2]}, index=[0, 1])
        with pytest.raises(ValueError, match="DatetimeIndex"):
            OnlineOptimizer(opt, signals, CostMode.FULL_COST)
