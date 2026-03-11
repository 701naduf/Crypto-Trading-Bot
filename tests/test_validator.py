"""
validator 模块测试

纯本地测试，不需要网络连接。
"""

import pandas as pd
import pytest

from data.validator import (
    validate_ohlcv,
    validate_orderbook,
    validate_ticks,
    validate_market_data,
)


class TestValidateOhlcv:
    def _make_df(self, **overrides):
        """创建一个合法的 OHLCV DataFrame"""
        data = {
            "timestamp": [pd.Timestamp("2024-01-15 10:00:00", tz="UTC")],
            "open": [42000.0],
            "high": [42100.0],
            "low": [41900.0],
            "close": [42050.0],
            "volume": [100.0],
        }
        data.update(overrides)
        return pd.DataFrame(data)

    def test_valid(self):
        df = self._make_df()
        valid, invalid = validate_ohlcv(df)
        assert len(valid) == 1
        assert len(invalid) == 0

    def test_negative_price(self):
        df = self._make_df(open=[-1.0])
        valid, invalid = validate_ohlcv(df)
        assert len(valid) == 0
        assert len(invalid) == 1

    def test_negative_volume(self):
        df = self._make_df(volume=[-10.0])
        valid, invalid = validate_ohlcv(df)
        assert len(valid) == 0

    def test_high_less_than_open(self):
        """high 低于 open 应被过滤"""
        df = self._make_df(high=[41500.0])  # high < open(42000)
        valid, invalid = validate_ohlcv(df)
        assert len(valid) == 0

    def test_low_greater_than_close(self):
        """low 高于 close 应被过滤"""
        df = self._make_df(low=[42500.0])  # low > close(42050)
        valid, invalid = validate_ohlcv(df)
        assert len(valid) == 0

    def test_empty_df(self):
        df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        valid, invalid = validate_ohlcv(df)
        assert len(valid) == 0


class TestValidateTicks:
    def _make_df(self, **overrides):
        data = {
            "trade_id": [12345],
            "timestamp": [pd.Timestamp("2024-01-15 10:00:00", tz="UTC")],
            "price": [42000.0],
            "amount": [0.5],
            "side": ["buy"],
        }
        data.update(overrides)
        return pd.DataFrame(data)

    def test_valid(self):
        valid, invalid = validate_ticks(self._make_df())
        assert len(valid) == 1

    def test_invalid_side(self):
        valid, invalid = validate_ticks(self._make_df(side=["unknown"]))
        assert len(valid) == 0

    def test_zero_trade_id(self):
        valid, invalid = validate_ticks(self._make_df(trade_id=[0]))
        assert len(valid) == 0

    def test_negative_price(self):
        valid, invalid = validate_ticks(self._make_df(price=[-1.0]))
        assert len(valid) == 0


class TestValidateOrderbook:
    def _make_snapshot(self, depth=10):
        return {
            "timestamp": pd.Timestamp("2024-01-15 10:00:00", tz="UTC"),
            "bids": [[42000 - i * 10, 1.0] for i in range(depth)],
            "asks": [[42001 + i * 10, 1.0] for i in range(depth)],
        }

    def test_valid(self):
        assert validate_orderbook(self._make_snapshot(), depth=10)

    def test_wrong_depth(self):
        snapshot = self._make_snapshot(depth=5)
        assert not validate_orderbook(snapshot, depth=10)

    def test_crossed_book(self):
        """买一 >= 卖一 应无效"""
        snapshot = self._make_snapshot()
        snapshot["bids"][0][0] = 50000  # bid > ask
        assert not validate_orderbook(snapshot, depth=10)

    def test_negative_price(self):
        snapshot = self._make_snapshot()
        snapshot["asks"][0][0] = -1
        assert not validate_orderbook(snapshot, depth=10)


class TestValidateMarketData:
    def test_funding_rate_valid(self):
        df = pd.DataFrame({"funding_rate": [0.0001, -0.0003]})
        valid, invalid = validate_market_data(df, "funding_rate")
        assert len(valid) == 2

    def test_funding_rate_extreme(self):
        df = pd.DataFrame({"funding_rate": [0.5]})  # 超出 ±0.1
        valid, invalid = validate_market_data(df, "funding_rate")
        assert len(valid) == 0

    def test_unknown_type(self):
        df = pd.DataFrame({"col": [1]})
        valid, invalid = validate_market_data(df, "unknown_type")
        assert len(valid) == 1  # 未知类型跳过校验，全部通过
