"""
aggregator 模块测试

纯本地测试，不需要网络连接。
"""

import pandas as pd
import pytest

from data_infra.data.aggregator import aggregate_ticks_to_ohlcv, resample_ohlcv


class TestAggregateTicks:
    def test_basic(self):
        """基本 tick → OHLCV 聚合"""
        ticks = pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2024-01-15 10:00:01",
                "2024-01-15 10:00:05",
                "2024-01-15 10:00:08",
                "2024-01-15 10:00:12",
                "2024-01-15 10:00:18",
            ], utc=True),
            "price": [42000, 42100, 41900, 42050, 42200],
            "amount": [0.1, 0.2, 0.15, 0.3, 0.25],
        })

        ohlcv = aggregate_ticks_to_ohlcv(ticks, "10s")
        assert len(ohlcv) >= 1
        assert "open" in ohlcv.columns
        assert "volume" in ohlcv.columns

    def test_empty_input(self):
        ticks = pd.DataFrame(columns=["timestamp", "price", "amount"])
        ohlcv = aggregate_ticks_to_ohlcv(ticks, "10s")
        assert ohlcv.empty

    def test_single_tick(self):
        """单笔成交也能生成一根K线"""
        ticks = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-15 10:00:05"], utc=True),
            "price": [42000.0],
            "amount": [1.0],
        })
        ohlcv = aggregate_ticks_to_ohlcv(ticks, "10s")
        assert len(ohlcv) == 1
        assert ohlcv["open"].iloc[0] == 42000.0
        assert ohlcv["high"].iloc[0] == 42000.0
        assert ohlcv["low"].iloc[0] == 42000.0
        assert ohlcv["close"].iloc[0] == 42000.0


class TestResampleOhlcv:
    @pytest.fixture
    def df_1m(self):
        """10 根 1m K线"""
        return pd.DataFrame({
            "timestamp": pd.date_range("2024-01-15 10:00", periods=10, freq="1min", tz="UTC"),
            "open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            "high": [110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
            "low": [90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
            "close": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            "volume": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        })

    def test_1m_to_5m(self, df_1m):
        result = resample_ohlcv(df_1m, "1m", "5m")
        assert len(result) == 2  # 10根1m → 2根5m

        # 第一根 5m: 聚合 10:00 ~ 10:04
        first = result.iloc[0]
        assert first["open"] == 100        # 第一根的 open
        assert first["high"] == 114        # 5根中的最大 high
        assert first["low"] == 90          # 5根中的最小 low
        assert first["close"] == 105       # 最后一根的 close
        assert first["volume"] == 150      # 10+20+30+40+50

    def test_same_timeframe(self, df_1m):
        """同周期不变"""
        result = resample_ohlcv(df_1m, "1m", "1m")
        assert len(result) == 10

    def test_invalid_ratio(self, df_1m):
        """非整数倍应报错"""
        with pytest.raises(ValueError):
            resample_ohlcv(df_1m, "5m", "7m")  # 420s % 300s != 0

    def test_empty(self):
        df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        result = resample_ohlcv(df, "1m", "5m")
        assert result.empty
