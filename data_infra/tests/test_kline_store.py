"""
kline_store 模块测试

使用临时数据库，纯本地测试。
"""

import os
import tempfile
from datetime import datetime, timezone

import pandas as pd
import pytest

from data_infra.data.kline_store import KlineStore


@pytest.fixture
def store(tmp_path):
    """创建使用临时数据库的 KlineStore"""
    db_path = str(tmp_path / "test_kline.db")
    return KlineStore(db_path=db_path)


@pytest.fixture
def sample_df():
    """创建示例 K线 DataFrame"""
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-15 10:00", periods=5, freq="1min", tz="UTC"),
        "open": [42000, 42100, 42050, 42200, 42150],
        "high": [42100, 42200, 42100, 42300, 42200],
        "low": [41950, 42050, 42000, 42100, 42100],
        "close": [42100, 42050, 42100, 42150, 42180],
        "volume": [100, 150, 80, 200, 120],
    })


class TestKlineStore:
    def test_write_and_read(self, store, sample_df):
        """写入后能正确读取"""
        new = store.write(sample_df, "BTC/USDT", "1m")
        assert new == 5

        df = store.read("BTC/USDT", "1m")
        assert len(df) == 5
        assert df["open"].iloc[0] == 42000

    def test_idempotent_write(self, store, sample_df):
        """重复写入不产生重复数据"""
        store.write(sample_df, "BTC/USDT", "1m")
        new = store.write(sample_df, "BTC/USDT", "1m")
        assert new == 0

        df = store.read("BTC/USDT", "1m")
        assert len(df) == 5

    def test_read_with_time_range(self, store, sample_df):
        """按时间范围读取"""
        store.write(sample_df, "BTC/USDT", "1m")

        start = datetime(2024, 1, 15, 10, 1, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, 10, 3, 0, tzinfo=timezone.utc)
        df = store.read("BTC/USDT", "1m", start=start, end=end)
        assert len(df) == 3  # 10:01, 10:02, 10:03

    def test_get_latest_timestamp(self, store, sample_df):
        """获取最新时间"""
        assert store.get_latest_timestamp("BTC/USDT", "1m") is None

        store.write(sample_df, "BTC/USDT", "1m")
        latest = store.get_latest_timestamp("BTC/USDT", "1m")
        assert latest == datetime(2024, 1, 15, 10, 4, 0, tzinfo=timezone.utc)

    def test_different_symbols(self, store, sample_df):
        """不同币对的数据互不影响"""
        store.write(sample_df, "BTC/USDT", "1m")
        store.write(sample_df, "ETH/USDT", "1m")

        btc = store.read("BTC/USDT", "1m")
        eth = store.read("ETH/USDT", "1m")
        assert len(btc) == 5
        assert len(eth) == 5

    def test_count(self, store, sample_df):
        """统计行数"""
        assert store.count("BTC/USDT", "1m") == 0
        store.write(sample_df, "BTC/USDT", "1m")
        assert store.count("BTC/USDT", "1m") == 5

    def test_empty_read(self, store):
        """读取空数据返回空 DataFrame"""
        df = store.read("BTC/USDT", "1m")
        assert df.empty
        assert "timestamp" in df.columns
