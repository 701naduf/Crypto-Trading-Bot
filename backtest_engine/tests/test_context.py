"""
test_context.py — MarketContextBuilder

覆盖 §11.2.5 关键约束、§11.2.6 边界情况、§11.2.8 双接口契约。
所有测试用 synthetic FakeDataReader（conftest.py），不依赖真实数据。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from execution_optimizer.config import MarketContext

from backtest_engine.config import BacktestConfig, RunMode
from backtest_engine.context import MarketContextBuilder


# 测试用：min_periods 设小一点，让 30 天合成数据足以产生非 NaN vol
SHORT_VOL_MIN_PERIODS = 200


@pytest.fixture
def builder(fake_reader, base_config):
    return MarketContextBuilder(
        fake_reader, base_config, vol_min_periods=SHORT_VOL_MIN_PERIODS,
    )


# ---------------------------------------------------------------------------
# 基本构造与索引
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_panels_are_built(self, builder):
        panels = builder.build_panels()
        assert set(panels.keys()) == {"price_panel", "spread_panel", "adv_panel", "vol_panel"}
        for name, p in panels.items():
            assert isinstance(p, pd.DataFrame), f"{name} 不是 DataFrame"
            assert list(p.columns) == ["BTC/USDT", "ETH/USDT"]

    def test_eval_index_within_start_end(self, builder, base_config):
        idx = builder.eval_bar_index
        assert idx.min() >= base_config.start
        assert idx.max() <= base_config.end
        assert idx.tz is not None

    def test_full_index_includes_lookback(self, builder, base_config):
        full = builder.bar_index
        # 热身期 21 天
        expected_earliest = base_config.start - pd.Timedelta(days=21)
        assert full.min() == expected_earliest


# ---------------------------------------------------------------------------
# 字段构造约束（§11.2.5）
# ---------------------------------------------------------------------------

class TestFieldSemantics:

    def test_vol_is_daily_scaled(self, builder):
        """日化 σ 应在合理范围（合成数据 annual_vol=0.6 → daily ≈ 0.6/√365 ≈ 0.031）"""
        panels = builder.build_panels()
        vol = panels["vol_panel"].dropna()
        # 至少有一些非 NaN 行
        assert len(vol) > 0
        # 日化 σ 量级应在 0.005~0.2 之间
        median_vol = vol.median().median()
        assert 0.005 < median_vol < 0.2, f"日化 σ 中位数 {median_vol} 偏离合理范围"

    def test_adv_is_usd_scale(self, builder):
        """ADV 单位是 USD：close × volume × bars_per_day"""
        panels = builder.build_panels()
        adv = panels["adv_panel"].dropna()
        assert len(adv) > 0
        # 合成 close ≈ 30000, volume ≈ 100, bars_per_day=1440 → ADV ≈ 4.32B
        median_adv = adv.median().median()
        assert 1e8 < median_adv < 1e11, f"ADV 中位数 {median_adv:.2e} 偏离合理范围"

    def test_spread_is_ratio(self, builder):
        """spread = (ask-bid)/mid 是比率，5 bps → 5e-4"""
        panels = builder.build_panels()
        spread = panels["spread_panel"].dropna()
        # 合成 spread_bps=5 → 0.0005
        median_spread = spread.median().median()
        assert 1e-5 < median_spread < 1e-2, f"spread 中位数 {median_spread} 偏离合理范围"

    def test_funding_is_amortized_per_bar(self, fake_reader, base_config):
        """funding_rate 是摊销值 = 真实 8h 费率 / bars_per_8h（1m → 480）"""
        builder = MarketContextBuilder(
            fake_reader, base_config, vol_min_periods=SHORT_VOL_MIN_PERIODS,
        )
        # 取一个评估区间内的 bar 测试
        t = builder.eval_bar_index[100]
        ctx = builder.build(t, current_weights=pd.Series(0.0, index=base_config.symbols),
                            portfolio_value=10000.0)
        assert ctx.funding_rate is not None
        # 合成 rate_per_8h=1e-4，摊销 / 480 ≈ 2.08e-7
        expected = 1e-4 / 480.0
        assert np.isclose(ctx.funding_rate.iloc[0], expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# build() 接口契约
# ---------------------------------------------------------------------------

class TestBuild:

    def test_build_returns_market_context(self, builder, base_config):
        t = builder.eval_bar_index[100]
        cw = pd.Series(0.0, index=base_config.symbols)
        ctx = builder.build(t, current_weights=cw, portfolio_value=10000.0)
        assert isinstance(ctx, MarketContext)
        assert ctx.timestamp == t
        assert list(ctx.symbols) == base_config.symbols
        assert isinstance(ctx.spread, pd.Series)
        assert isinstance(ctx.volatility, pd.Series)
        assert isinstance(ctx.adv, pd.Series)
        assert ctx.portfolio_value == 10000.0

    def test_build_ignores_current_weights(self, builder, base_config):
        """D1: current_weights 是前瞻保留参数，build() 不消费"""
        t = builder.eval_bar_index[100]
        ctx_a = builder.build(
            t, current_weights=pd.Series(0.0, index=base_config.symbols),
            portfolio_value=10000.0,
        )
        ctx_b = builder.build(
            t, current_weights=pd.Series([0.5, -0.5], index=base_config.symbols),
            portfolio_value=10000.0,
        )
        # 两次输出的非 V 字段应完全一致
        pd.testing.assert_series_equal(ctx_a.spread, ctx_b.spread)
        pd.testing.assert_series_equal(ctx_a.volatility, ctx_b.volatility)

    def test_build_panels_eval_range(self, builder, base_config):
        """build_panels 切到 [start, end] 评估区间"""
        panels = builder.build_panels()
        for name, p in panels.items():
            assert p.index.min() >= base_config.start
            assert p.index.max() <= base_config.end


# ---------------------------------------------------------------------------
# 边界情况（§11.2.6）
# ---------------------------------------------------------------------------

class TestBoundaryConditions:

    def test_lookback_insufficient_raises(self, fake_reader, synthetic_symbols):
        """start 距数据库最早数据 < 21 天 → ValueError"""
        # 合成数据从 2024-01-01 开始，把 lookback_days 设到 2024-01-30 之前会不够
        cfg = BacktestConfig(
            strategy_name="t",
            symbols=synthetic_symbols,
            start=pd.Timestamp("2024-01-02", tz="UTC"),  # 数据从 2024-01-01 开始，前面只有 1 天
            end=pd.Timestamp("2024-01-05", tz="UTC"),
            run_mode=RunMode.VECTORIZED,
        )
        with pytest.raises(ValueError, match="价格数据起点"):
            MarketContextBuilder(
                fake_reader, cfg, lookback_days=21,
                vol_min_periods=SHORT_VOL_MIN_PERIODS,
            )

    def test_missing_symbol_raises(self, fake_reader, synthetic_period):
        """symbols 中含 DataReader 没有的 symbol → KeyError"""
        _, start, end = synthetic_period
        cfg = BacktestConfig(
            strategy_name="t",
            symbols=["BTC/USDT", "UNKNOWN/USDT"],
            start=start,
            end=end,
            run_mode=RunMode.VECTORIZED,
        )
        with pytest.raises(KeyError, match="UNKNOWN"):
            MarketContextBuilder(
                fake_reader, cfg, vol_min_periods=SHORT_VOL_MIN_PERIODS,
            )

    def test_orderbook_gap_exceeds_max(self, synthetic_symbols, synthetic_period):
        """orderbook gap > spread_ffill_max_minutes → spread 该段为 NaN"""
        from .conftest import (
            FakeDataReader, _make_synthetic_ohlcv, _make_synthetic_funding,
        )

        earliest, start, end = synthetic_period
        ohlcv = _make_synthetic_ohlcv(synthetic_symbols, earliest, end)
        funding = _make_synthetic_funding(synthetic_symbols, earliest, end)

        # 故意创造 30 分钟空缺的 orderbook（对每个 symbol）
        orderbook = {}
        for sym in synthetic_symbols:
            ts1 = pd.date_range(earliest, start, freq="10s", tz="UTC")
            ts2 = pd.date_range(
                start + pd.Timedelta(minutes=30), end, freq="10s", tz="UTC",
            )
            ts = ts1.append(ts2)
            n = len(ts)
            orderbook[sym] = pd.DataFrame({
                "timestamp": ts,
                "bid_price_0": np.full(n, 30000.0),
                "ask_price_0": np.full(n, 30015.0),
                "bid_qty_0": np.full(n, 1.0),
                "ask_qty_0": np.full(n, 1.0),
            })

        reader = FakeDataReader(ohlcv, orderbook, funding)
        cfg = BacktestConfig(
            strategy_name="t", symbols=synthetic_symbols,
            start=start, end=end, run_mode=RunMode.VECTORIZED,
        )
        builder = MarketContextBuilder(
            reader, cfg, spread_ffill_max_minutes=5,
            vol_min_periods=SHORT_VOL_MIN_PERIODS,
        )
        spread = builder.build_panels()["spread_panel"]

        # gap 区间（start ~ start+30min）内大部分 bar 应为 NaN
        gap_window = spread.loc[start:start + pd.Timedelta(minutes=29)]
        nan_ratio = gap_window.isna().mean().mean()
        assert nan_ratio > 0.5, f"gap 期间 NaN 比例仅 {nan_ratio}（预期 > 0.5）"


# ---------------------------------------------------------------------------
# 其他不变量
# ---------------------------------------------------------------------------

class TestInvariants:

    def test_panels_align_index(self, builder):
        """所有 panel 的 index 严格相等（评估区间）"""
        panels = builder.build_panels()
        ref = panels["price_panel"].index
        for name, p in panels.items():
            assert (p.index == ref).all(), f"{name} index 不一致"

    def test_panels_align_columns(self, builder):
        panels = builder.build_panels()
        ref = panels["price_panel"].columns
        for name, p in panels.items():
            assert (p.columns == ref).all(), f"{name} columns 不一致"

    def test_no_reader_after_init(self, fake_reader, base_config):
        """A1 取舍：__init__ 后不再访问 reader（间接验证：清空 reader 的 dict 不影响 build_panels）"""
        builder = MarketContextBuilder(
            fake_reader, base_config, vol_min_periods=SHORT_VOL_MIN_PERIODS,
        )
        # 清空 reader（破坏其状态）
        fake_reader.ohlcv_by_symbol.clear()
        fake_reader.orderbook_by_symbol.clear()
        # build_panels / build 应仍能工作
        panels = builder.build_panels()
        assert len(panels) == 4
