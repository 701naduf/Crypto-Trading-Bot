"""
reader 模块测试

测试 DataReader 的路由逻辑。使用临时存储，纯本地测试。
"""

from datetime import datetime, timezone

import pandas as pd
import pytest

from data.kline_store import KlineStore
from data.reader import DataReader


@pytest.fixture
def reader(tmp_path):
    """创建使用临时存储的 DataReader"""
    reader = DataReader()
    # 替换为临时数据库
    reader._kline_store = KlineStore(db_path=str(tmp_path / "kline.db"))
    return reader


@pytest.fixture
def reader_with_data(reader):
    """带有预填数据的 DataReader"""
    # 写入 60 根 1m K线 (一小时)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-15 10:00", periods=60, freq="1min", tz="UTC"),
        "open": [42000 + i for i in range(60)],
        "high": [42100 + i for i in range(60)],
        "low": [41900 + i for i in range(60)],
        "close": [42050 + i for i in range(60)],
        "volume": [100 + i for i in range(60)],
    })
    reader._kline_store.write(df, "BTC/USDT", "1m")
    return reader


class TestDataReader:
    def test_get_ohlcv_1m(self, reader_with_data):
        """直接读取 1m"""
        df = reader_with_data.get_ohlcv("BTC/USDT", "1m")
        assert len(df) == 60

    def test_get_ohlcv_5m(self, reader_with_data):
        """5m 从 1m 降采样"""
        df = reader_with_data.get_ohlcv("BTC/USDT", "5m")
        assert len(df) == 12  # 60 / 5

    def test_get_ohlcv_15m(self, reader_with_data):
        """15m 从 1m 降采样"""
        df = reader_with_data.get_ohlcv("BTC/USDT", "15m")
        assert len(df) == 4  # 60 / 15

    def test_get_ohlcv_1h(self, reader_with_data):
        """1h 从 1m 降采样"""
        df = reader_with_data.get_ohlcv("BTC/USDT", "1h")
        assert len(df) == 1  # 60 / 60

    def test_get_ohlcv_empty(self, reader):
        """无数据时返回空 DataFrame"""
        df = reader.get_ohlcv("BTC/USDT", "1m")
        assert df.empty

    def test_get_ohlcv_with_range(self, reader_with_data):
        """带时间范围的读取"""
        start = datetime(2024, 1, 15, 10, 10, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, 10, 20, 0, tzinfo=timezone.utc)
        df = reader_with_data.get_ohlcv("BTC/USDT", "1m", start=start, end=end)
        assert len(df) == 11  # 10:10 ~ 10:20 包含两端
