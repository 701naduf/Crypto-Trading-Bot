"""
orderbook_store 模块测试

使用临时目录，纯本地测试。
"""

import os
from datetime import datetime, timezone

import pandas as pd
import pytest

from data.orderbook_store import OrderbookStore


@pytest.fixture
def store(tmp_path):
    return OrderbookStore(data_dir=str(tmp_path / "orderbook"), buffer_size=5)


def make_snapshot(ts=None, depth=10):
    """创建一个模拟订单簿快照"""
    if ts is None:
        ts = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    return {
        "timestamp": ts,
        "bids": [[42000 - i * 10, 1.0 + i * 0.1] for i in range(depth)],
        "asks": [[42001 + i * 10, 1.0 + i * 0.1] for i in range(depth)],
    }


class TestOrderbookStore:
    def test_append_and_flush(self, store):
        """追加数据到缓冲后刷盘"""
        for i in range(3):
            ts = datetime(2024, 1, 15, 10, 0, i, tzinfo=timezone.utc)
            store.append("BTC/USDT", make_snapshot(ts))

        assert store.get_buffer_size("BTC/USDT") == 3

        store.flush("BTC/USDT")
        assert store.get_buffer_size("BTC/USDT") == 0

    def test_auto_flush(self, store):
        """缓冲满时自动刷盘（buffer_size=5）"""
        for i in range(6):
            ts = datetime(2024, 1, 15, 10, 0, i, tzinfo=timezone.utc)
            store.append("BTC/USDT", make_snapshot(ts))

        # 5个时触发了自动刷盘，第6个在缓冲中
        assert store.get_buffer_size("BTC/USDT") == 1

    def test_read(self, store):
        """刷盘后能正确读取"""
        for i in range(3):
            ts = datetime(2024, 1, 15, 10, 0, i, tzinfo=timezone.utc)
            store.append("BTC/USDT", make_snapshot(ts))

        store.flush()
        df = store.read("BTC/USDT")
        assert len(df) == 3
        assert "bid_price_0" in df.columns
        assert "ask_price_9" in df.columns

    def test_read_with_levels(self, store):
        """读取指定档位数"""
        store.append("BTC/USDT", make_snapshot())
        store.flush()

        df = store.read("BTC/USDT", levels=3)
        # 应只有 timestamp + 3*2(bid) + 3*2(ask) = 13 列
        assert "bid_price_0" in df.columns
        assert "bid_price_2" in df.columns
        assert "bid_price_3" not in df.columns

    def test_flush_and_close(self, store):
        """退出前刷盘"""
        for i in range(3):
            ts = datetime(2024, 1, 15, 10, 0, i, tzinfo=timezone.utc)
            store.append("BTC/USDT", make_snapshot(ts))

        store.flush_and_close()
        assert store.get_buffer_size() == 0

        df = store.read("BTC/USDT")
        assert len(df) == 3
