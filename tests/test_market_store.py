"""
market_store 模块测试

使用临时数据库，纯本地测试。
"""

from datetime import datetime, timezone

import pandas as pd
import pytest

from data.market_store import MarketStore


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test_market.db")
    return MarketStore(db_path=db_path)


class TestMarketStore:
    def test_write_funding_rate(self, store):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-15 08:00", periods=3, freq="8h", tz="UTC"),
            "funding_rate": [0.0001, -0.0002, 0.0003],
        })
        new = store.write_funding_rate(df, "BTC/USDT")
        assert new == 3

    def test_write_open_interest(self, store):
        data = {
            "timestamp": datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            "open_interest": 50000.0,
            "open_interest_value": 2100000000.0,
        }
        new = store.write_open_interest("BTC/USDT", data)
        assert new == 1

    def test_write_long_short_ratio(self, store):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-15 10:00", periods=2, freq="5min", tz="UTC"),
            "long_ratio": [0.52, 0.48],
            "short_ratio": [0.48, 0.52],
            "long_short_ratio": [1.08, 0.92],
        })
        new = store.write_long_short_ratio(df, "BTC/USDT")
        assert new == 2

    def test_write_taker_buy_sell(self, store):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-15 10:00", periods=2, freq="5min", tz="UTC"),
            "buy_vol": [100.0, 120.0],
            "sell_vol": [90.0, 130.0],
            "buy_sell_ratio": [1.11, 0.92],
        })
        new = store.write_taker_buy_sell(df, "BTC/USDT")
        assert new == 2

    def test_read(self, store):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-15 08:00", periods=3, freq="8h", tz="UTC"),
            "funding_rate": [0.0001, -0.0002, 0.0003],
        })
        store.write_funding_rate(df, "BTC/USDT")

        result = store.read("funding_rates", "BTC/USDT")
        assert len(result) == 3
        assert "funding_rate" in result.columns

    def test_get_latest_timestamp(self, store):
        assert store.get_latest_timestamp("funding_rates", "BTC/USDT") is None

        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-15 08:00", periods=3, freq="8h", tz="UTC"),
            "funding_rate": [0.0001, -0.0002, 0.0003],
        })
        store.write_funding_rate(df, "BTC/USDT")

        latest = store.get_latest_timestamp("funding_rates", "BTC/USDT")
        assert latest is not None
        assert latest.day == 16  # 08:00 + 16h = 次日 00:00

    def test_idempotent(self, store):
        """重复写入不产生重复"""
        df = pd.DataFrame({
            "timestamp": [datetime(2024, 1, 15, 8, 0, 0, tzinfo=timezone.utc)],
            "funding_rate": [0.0001],
        })
        store.write_funding_rate(df, "BTC/USDT")
        new = store.write_funding_rate(df, "BTC/USDT")
        assert new == 0

    def test_invalid_table(self, store):
        with pytest.raises(ValueError):
            store.read("invalid_table", "BTC/USDT")
