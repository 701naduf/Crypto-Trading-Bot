"""
test_config.py — BacktestConfig 校验逻辑测试

覆盖 §11.1.3 校验清单的全部 16 条 + 默认值 + 字段类型契约。
"""
from __future__ import annotations

import logging

import pandas as pd
import pytest

from alpha_model.core.types import PortfolioConstraints

from backtest_engine.config import (
    BacktestConfig, RunMode, ExecutionMode, CostMode,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_kwargs(**overrides):
    """构造一个合法的最小 BacktestConfig 参数集"""
    base = dict(
        strategy_name="test_strategy",
        symbols=["BTC/USDT", "ETH/USDT"],
        start=pd.Timestamp("2024-01-01", tz="UTC"),
        end=pd.Timestamp("2024-02-01", tz="UTC"),
        run_mode=RunMode.VECTORIZED,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 默认值与基本构造
# ---------------------------------------------------------------------------

class TestDefaults:

    def test_minimal_vectorized_config_works(self):
        cfg = BacktestConfig(**_make_kwargs())
        assert cfg.bar_freq == "1m"
        assert cfg.execution_mode == ExecutionMode.MARKET
        assert cfg.cost_mode == CostMode.FULL_COST
        assert cfg.optimize_every_n_bars == 1
        assert cfg.time_convention == "bar_close"
        assert cfg.constraints is None
        assert cfg.regime_series is None

    def test_dynamic_cost_with_constraints(self):
        cons = PortfolioConstraints(max_weight=0.4)
        cfg = BacktestConfig(**_make_kwargs(
            run_mode=RunMode.EVENT_DRIVEN_DYNAMIC_COST,
            constraints=cons,
        ))
        assert cfg.constraints is cons


# ---------------------------------------------------------------------------
# §11.1.3 校验项 1-2 — constraints
# ---------------------------------------------------------------------------

class TestConstraintsValidation:

    def test_dynamic_cost_without_constraints_raises(self):
        with pytest.raises(ValueError, match="DYNAMIC_COST"):
            BacktestConfig(**_make_kwargs(
                run_mode=RunMode.EVENT_DRIVEN_DYNAMIC_COST,
                constraints=None,
            ))

    def test_non_dynamic_with_constraints_warns(self, caplog):
        cons = PortfolioConstraints()
        with caplog.at_level(logging.WARNING):
            BacktestConfig(**_make_kwargs(
                run_mode=RunMode.VECTORIZED,
                constraints=cons,
            ))
        assert any("不使用 constraints" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# §11.1.3 校验项 3 / 7 / 8 / 9 / 13 — 数值合法性
# ---------------------------------------------------------------------------

class TestNumericValidation:

    def test_start_ge_end_raises(self):
        with pytest.raises(ValueError, match="start"):
            BacktestConfig(**_make_kwargs(
                start=pd.Timestamp("2024-02-01", tz="UTC"),
                end=pd.Timestamp("2024-01-01", tz="UTC"),
            ))

    def test_initial_portfolio_value_zero_raises(self):
        with pytest.raises(ValueError, match="initial_portfolio_value"):
            BacktestConfig(**_make_kwargs(initial_portfolio_value=0.0))

    def test_initial_portfolio_value_negative_raises(self):
        with pytest.raises(ValueError, match="initial_portfolio_value"):
            BacktestConfig(**_make_kwargs(initial_portfolio_value=-100.0))

    def test_periods_per_year_negative_raises(self):
        with pytest.raises(ValueError, match="periods_per_year"):
            BacktestConfig(**_make_kwargs(periods_per_year=-1.0))

    def test_min_trade_value_zero_raises(self):
        with pytest.raises(ValueError, match="min_trade_value"):
            BacktestConfig(**_make_kwargs(min_trade_value=0.0))

    def test_optimize_every_n_bars_zero_raises(self):
        with pytest.raises(ValueError, match="optimize_every_n_bars"):
            BacktestConfig(**_make_kwargs(optimize_every_n_bars=0))


# ---------------------------------------------------------------------------
# §11.1.3 校验项 4 — symbols 非空
# ---------------------------------------------------------------------------

class TestSymbolsValidation:

    def test_empty_symbols_raises(self):
        with pytest.raises(ValueError, match="symbols"):
            BacktestConfig(**_make_kwargs(symbols=[]))


# ---------------------------------------------------------------------------
# §11.1.3 校验项 5 / 6 / 10 — v1 范围限制
# ---------------------------------------------------------------------------

class TestNotImplementedV1:

    def test_non_1m_bar_freq_raises(self):
        with pytest.raises(NotImplementedError, match="bar_freq"):
            BacktestConfig(**_make_kwargs(bar_freq="5m"))

    def test_non_market_execution_mode_raises(self):
        with pytest.raises(NotImplementedError, match="MARKET"):
            BacktestConfig(**_make_kwargs(execution_mode=ExecutionMode.LIMIT))

    def test_non_bar_close_convention_raises(self):
        with pytest.raises(NotImplementedError, match="bar_close"):
            BacktestConfig(**_make_kwargs(time_convention="bar_open"))


# ---------------------------------------------------------------------------
# §11.1.3 校验项 11 / 12 / 14 — 静默冗余 warning
# ---------------------------------------------------------------------------

class TestSilentRedundancyWarnings:

    def test_vectorized_match_vectorized_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            BacktestConfig(**_make_kwargs(
                run_mode=RunMode.VECTORIZED,
                cost_mode=CostMode.MATCH_VECTORIZED,
            ))
        assert any("MATCH_VECTORIZED" in rec.message for rec in caplog.records)

    def test_non_dynamic_with_n_neq_1_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            BacktestConfig(**_make_kwargs(
                run_mode=RunMode.VECTORIZED,
                optimize_every_n_bars=5,
            ))
        assert any("optimize_every_n_bars" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# §11.1.3 校验项 15 / 16 — v3 修订的 tz 校验
# ---------------------------------------------------------------------------

class TestTzAwareValidation:

    def test_start_naive_raises(self):
        with pytest.raises(ValueError, match="tz-aware"):
            BacktestConfig(**_make_kwargs(
                start=pd.Timestamp("2024-01-01"),  # naive
            ))

    def test_end_naive_raises(self):
        with pytest.raises(ValueError, match="tz-aware"):
            BacktestConfig(**_make_kwargs(
                end=pd.Timestamp("2024-02-01"),  # naive
            ))

    def test_regime_series_naive_index_raises(self):
        # naive DatetimeIndex
        idx = pd.date_range("2024-01-01", periods=10, freq="1min")
        regime = pd.Series(["bull"] * 10, index=idx)
        with pytest.raises(ValueError, match="tz-aware"):
            BacktestConfig(**_make_kwargs(regime_series=regime))

    def test_regime_series_non_datetime_index_raises(self):
        regime = pd.Series(["bull", "bear"], index=[0, 1])
        with pytest.raises(ValueError, match="DatetimeIndex"):
            BacktestConfig(**_make_kwargs(regime_series=regime))

    def test_regime_series_tz_aware_passes(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="1min", tz="UTC")
        regime = pd.Series(["bull"] * 10, index=idx)
        cfg = BacktestConfig(**_make_kwargs(regime_series=regime))
        assert cfg.regime_series is regime


# ---------------------------------------------------------------------------
# 枚举字段类型契约
# ---------------------------------------------------------------------------

class TestEnumValues:
    """枚举值字符串与文档（§11.1.5）一致"""

    def test_run_mode_values(self):
        assert RunMode.VECTORIZED.value == "vectorized"
        assert RunMode.EVENT_DRIVEN_FIXED_GAMMA.value == "event_driven_fixed_gamma"
        assert RunMode.EVENT_DRIVEN_DYNAMIC_COST.value == "event_driven_dynamic_cost"

    def test_execution_mode_values(self):
        assert ExecutionMode.MARKET.value == "market"
        assert ExecutionMode.LIMIT.value == "limit"
        assert ExecutionMode.TWAP.value == "twap"

    def test_cost_mode_values(self):
        assert CostMode.FULL_COST.value == "full_cost"
        assert CostMode.MATCH_VECTORIZED.value == "match_vectorized"


# ---------------------------------------------------------------------------
# kw_only 强制
# ---------------------------------------------------------------------------

class TestKwOnly:

    def test_positional_args_rejected(self):
        with pytest.raises(TypeError):
            BacktestConfig(  # type: ignore[misc]
                "test", ["BTC/USDT"],
                pd.Timestamp("2024-01-01", tz="UTC"),
                pd.Timestamp("2024-02-01", tz="UTC"),
                RunMode.VECTORIZED,
            )
