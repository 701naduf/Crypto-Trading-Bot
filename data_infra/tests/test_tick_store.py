"""
tick_store 模块测试

使用临时目录，纯本地测试。
"""

import os

import pandas as pd
import pytest

from data_infra.data.tick_store import TickStore


@pytest.fixture
def store(tmp_path):
    return TickStore(data_dir=str(tmp_path / "ticks"))


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "trade_id": [100, 101, 102, 103, 104],
        "timestamp": pd.date_range("2024-01-15 10:00:00", periods=5, freq="1s", tz="UTC"),
        "price": [42000.0, 42010.0, 42005.0, 42020.0, 42015.0],
        "amount": [0.1, 0.2, 0.15, 0.3, 0.25],
        "side": ["buy", "sell", "buy", "buy", "sell"],
    })


class TestTickStore:
    def test_write_and_read(self, store, sample_df):
        new = store.write(sample_df, "BTC/USDT")
        assert new == 5

        df = store.read("BTC/USDT")
        assert len(df) == 5
        assert df["trade_id"].tolist() == [100, 101, 102, 103, 104]

    def test_dedup(self, store, sample_df):
        """重复写入自动去重"""
        store.write(sample_df, "BTC/USDT")
        new = store.write(sample_df, "BTC/USDT")
        assert new == 0

        df = store.read("BTC/USDT")
        assert len(df) == 5

    def test_get_latest_trade_id(self, store, sample_df):
        assert store.get_latest_trade_id("BTC/USDT") is None

        store.write(sample_df, "BTC/USDT")
        assert store.get_latest_trade_id("BTC/USDT") == 104

    def test_cross_day_data(self, store):
        """跨天数据自动分到不同文件"""
        df = pd.DataFrame({
            "trade_id": [1, 2, 3],
            "timestamp": pd.to_datetime([
                "2024-01-15 23:59:00",
                "2024-01-16 00:00:00",
                "2024-01-16 00:01:00",
            ], utc=True),
            "price": [42000, 42100, 42200],
            "amount": [0.1, 0.2, 0.3],
            "side": ["buy", "sell", "buy"],
        })

        store.write(df, "BTC/USDT")

        # 两天的文件应该都存在
        sym_dir = store._symbol_dir("BTC/USDT")
        files = sorted(os.listdir(sym_dir))
        assert len(files) == 2
        assert "2024-01-15.parquet" in files
        assert "2024-01-16.parquet" in files

    def test_atomic_write_no_tmp(self, store, sample_df):
        """写入完成后不应有 .tmp 文件残留"""
        store.write(sample_df, "BTC/USDT")

        sym_dir = store._symbol_dir("BTC/USDT")
        for f in os.listdir(sym_dir):
            assert not f.endswith(".tmp"), f"残留 .tmp 文件: {f}"

    def test_different_symbols(self, store, sample_df):
        store.write(sample_df, "BTC/USDT")
        store.write(sample_df, "ETH/USDT")

        btc = store.read("BTC/USDT")
        eth = store.read("ETH/USDT")
        assert len(btc) == 5
        assert len(eth) == 5
