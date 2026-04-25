"""
backtest_engine 测试公共 fixtures

设计：所有测试用 synthetic fixture，不依赖 db/ 真实数据。
FakeDataReader 与 DataReader 接口契约一致（duck-typed）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# FakeDataReader — 鸭子类型替代真实 DataReader
# ---------------------------------------------------------------------------

@dataclass
class FakeDataReader:
    """
    内存版 DataReader，用 dict 持有合成面板。

    与真实 DataReader 接口契约一致：
      get_ohlcv(symbol, timeframe, start, end) → DataFrame[timestamp, open, high, low, close, volume]
      get_orderbook(symbol, start, end, levels) → DataFrame[timestamp, bid_price_0, ask_price_0, ...]
      get_funding_rate(symbol, start, end) → DataFrame[symbol, timestamp, funding_rate]
    """

    ohlcv_by_symbol: dict[str, pd.DataFrame]
    orderbook_by_symbol: dict[str, pd.DataFrame]
    funding_by_symbol: dict[str, pd.DataFrame]

    def get_ohlcv(self, symbol, timeframe, start=None, end=None):
        df = self.ohlcv_by_symbol.get(symbol)
        if df is None:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        return _slice_by_timestamp(df, start, end)

    def get_orderbook(self, symbol, start=None, end=None, levels=None):
        df = self.orderbook_by_symbol.get(symbol)
        if df is None:
            return pd.DataFrame(columns=["timestamp", "bid_price_0", "ask_price_0"])
        return _slice_by_timestamp(df, start, end)

    def get_funding_rate(self, symbol, start=None, end=None):
        df = self.funding_by_symbol.get(symbol)
        if df is None:
            return pd.DataFrame(columns=["symbol", "timestamp", "funding_rate"])
        return _slice_by_timestamp(df, start, end)


def _slice_by_timestamp(df: pd.DataFrame, start, end) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        return df
    out = df
    if start is not None:
        out = out[out["timestamp"] >= start]
    if end is not None:
        out = out[out["timestamp"] <= end]
    return out.copy()


# ---------------------------------------------------------------------------
# 合成行情数据生成
# ---------------------------------------------------------------------------

def _make_synthetic_ohlcv(
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    bar_freq: str = "1min",
    seed: int = 0,
    base_price: float = 30000.0,
    annual_vol: float = 0.6,
) -> dict[str, pd.DataFrame]:
    """
    合成 OHLCV 面板：几何布朗运动（GBM）+ 固定 volume

    Args:
        symbols:    每个 symbol 独立随机种子（用 hash 分散）
        start, end: tz-aware UTC Timestamp
        bar_freq:   bar 频率（pandas freq string）
        seed:       基础随机种子
        annual_vol: 年化波动率（用于推导 1m bar std）
        base_price: 初始价格
    """
    idx = pd.date_range(start, end, freq=bar_freq, tz="UTC")
    bars_per_year = 365.25 * 24 * 60  # 1m 假设
    sigma_per_bar = annual_vol / np.sqrt(bars_per_year)

    out = {}
    for i, sym in enumerate(symbols):
        rng = np.random.default_rng(seed + i)
        log_ret = rng.normal(0, sigma_per_bar, len(idx))
        log_price = np.log(base_price * (1 + 0.01 * i)) + np.cumsum(log_ret)
        close = np.exp(log_price)
        open_ = np.concatenate([[close[0]], close[:-1]])
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, sigma_per_bar / 2, len(idx))))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, sigma_per_bar / 2, len(idx))))
        volume = rng.uniform(50, 200, len(idx))  # 单位 base asset

        df = pd.DataFrame({
            "timestamp": idx,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })
        out[sym] = df
    return out


def _make_synthetic_orderbook(
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    snapshot_freq: str = "10s",
    spread_bps: float = 5.0,
    seed: int = 100,
) -> dict[str, pd.DataFrame]:
    """合成 orderbook（10s 快照），固定中间价 + 固定 spread"""
    idx = pd.date_range(start, end, freq=snapshot_freq, tz="UTC")
    out = {}
    for i, sym in enumerate(symbols):
        rng = np.random.default_rng(seed + i)
        mid = 30000.0 * (1 + 0.01 * i) + rng.normal(0, 5, len(idx)).cumsum() * 0.01
        half_spread = mid * spread_bps / 1e4 / 2.0
        bid = mid - half_spread
        ask = mid + half_spread

        df = pd.DataFrame({
            "timestamp": idx,
            "bid_price_0": bid,
            "ask_price_0": ask,
            "bid_qty_0": rng.uniform(1, 5, len(idx)),
            "ask_qty_0": rng.uniform(1, 5, len(idx)),
        })
        out[sym] = df
    return out


def _make_synthetic_funding(
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    rate_per_8h: float = 0.0001,
) -> dict[str, pd.DataFrame]:
    """合成 funding rate：每 8 小时一次，固定费率"""
    settle_ts = pd.date_range(
        start.normalize(), end, freq="8h", tz="UTC",
    )
    settle_ts = settle_ts[settle_ts >= start]
    out = {}
    for sym in symbols:
        df = pd.DataFrame({
            "symbol": [sym] * len(settle_ts),
            "timestamp": settle_ts,
            "funding_rate": [rate_per_8h] * len(settle_ts),
        })
        out[sym] = df
    return out


# ---------------------------------------------------------------------------
# 标准 fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_symbols():
    return ["BTC/USDT", "ETH/USDT"]


@pytest.fixture(scope="module")
def synthetic_period():
    """30 天回测期：足够热身（21 天）+ 评估区间（约 9 天）"""
    start = pd.Timestamp("2024-01-22", tz="UTC")
    end = pd.Timestamp("2024-01-31", tz="UTC")
    earliest = start - pd.Timedelta(days=21)
    return earliest, start, end


@pytest.fixture(scope="module")
def fake_reader(synthetic_symbols, synthetic_period):
    """完整合成数据：OHLCV 含 21 天热身 + 9 天评估"""
    earliest, _start, end = synthetic_period
    return FakeDataReader(
        ohlcv_by_symbol=_make_synthetic_ohlcv(synthetic_symbols, earliest, end),
        orderbook_by_symbol=_make_synthetic_orderbook(synthetic_symbols, earliest, end),
        funding_by_symbol=_make_synthetic_funding(synthetic_symbols, earliest, end),
    )


@pytest.fixture
def base_config(synthetic_symbols, synthetic_period):
    """适合事件驱动测试的最小 BacktestConfig（默认 VECTORIZED）"""
    from backtest_engine.config import BacktestConfig, RunMode

    _, start, end = synthetic_period
    return BacktestConfig(
        strategy_name="synthetic_test",
        symbols=synthetic_symbols,
        start=start,
        end=end,
        run_mode=RunMode.VECTORIZED,
    )
