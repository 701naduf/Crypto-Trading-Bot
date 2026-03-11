"""
KlineFetcher 测试

mock ccxt.binance.fetch_ohlcv，测试:
    - 返回 DataFrame 列名和类型
    - 空数据处理
    - batch 分页逻辑
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data_infra.data.fetcher import KlineFetcher


@pytest.fixture
def fetcher():
    """创建 KlineFetcher，mock 掉 ccxt 交易所实例"""
    with patch("data_infra.data.fetcher.ccxt") as mock_ccxt:
        mock_exchange = MagicMock()
        mock_ccxt.binance.return_value = mock_exchange
        f = KlineFetcher()
        f.exchange = mock_exchange
        yield f


class TestFetchOhlcv:
    def test_returns_correct_columns(self, fetcher):
        """返回 DataFrame 应包含标准 OHLCV 列"""
        fetcher.exchange.fetch_ohlcv.return_value = [
            [1705305600000, 42000.0, 42100.0, 41900.0, 42050.0, 100.5],
            [1705305660000, 42050.0, 42200.0, 42000.0, 42150.0, 200.3],
        ]

        df = fetcher.fetch_ohlcv("BTC/USDT", "1m")

        assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
        assert len(df) == 2

    def test_timestamp_is_utc_datetime(self, fetcher):
        """timestamp 列应为 UTC datetime"""
        fetcher.exchange.fetch_ohlcv.return_value = [
            [1705305600000, 42000.0, 42100.0, 41900.0, 42050.0, 100.5],
        ]

        df = fetcher.fetch_ohlcv("BTC/USDT", "1m")

        assert df["timestamp"].dt.tz is not None  # 有时区信息
        assert str(df["timestamp"].dt.tz) == "UTC"

    def test_empty_response(self, fetcher):
        """API 返回空列表时应返回空 DataFrame"""
        fetcher.exchange.fetch_ohlcv.return_value = []

        df = fetcher.fetch_ohlcv("BTC/USDT", "1m")

        assert df.empty
        assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]

    def test_none_response(self, fetcher):
        """API 返回 None 时应返回空 DataFrame"""
        fetcher.exchange.fetch_ohlcv.return_value = None

        df = fetcher.fetch_ohlcv("BTC/USDT", "1m")

        assert df.empty


class TestFetchOhlcvBatch:
    def test_single_page(self, fetcher):
        """数据量不足一页时只请求一次"""
        fetcher.exchange.fetch_ohlcv.return_value = [
            [1705305600000, 42000.0, 42100.0, 41900.0, 42050.0, 100.5],
            [1705305660000, 42050.0, 42200.0, 42000.0, 42150.0, 200.3],
        ]

        start = datetime(2024, 1, 15, 8, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, 8, 5, tzinfo=timezone.utc)

        df = fetcher.fetch_ohlcv_batch("BTC/USDT", "1m", start, end)

        assert len(df) == 2
        assert fetcher.exchange.fetch_ohlcv.call_count == 1

    def test_pagination(self, fetcher):
        """数据量超过一页时自动分页"""
        # 第一页: 满额（模拟 KLINE_LIMIT=1000，这里用3条模拟）
        page1 = [
            [1705305600000 + i * 60000, 42000.0, 42100.0, 41900.0, 42050.0, 100.0]
            for i in range(3)
        ]
        # 第二页: 不满额，表示到头
        page2 = [
            [1705305600000 + 3 * 60000, 42000.0, 42100.0, 41900.0, 42050.0, 100.0],
        ]

        fetcher.exchange.fetch_ohlcv.side_effect = [page1, page2]

        start = datetime(2024, 1, 15, 8, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc)

        with patch("data_infra.data.fetcher.settings") as mock_settings:
            mock_settings.KLINE_LIMIT = 3
            mock_settings.EXCHANGE_ID = "binance"
            mock_settings.API_KEY = ""
            mock_settings.API_SECRET = ""
            mock_settings.REQUEST_TIMEOUT = 30
            mock_settings.PROXY_HOST = None
            mock_settings.PROXY_PORT = None
            df = fetcher.fetch_ohlcv_batch("BTC/USDT", "1m", start, end)

        assert fetcher.exchange.fetch_ohlcv.call_count == 2
        assert len(df) == 4

    def test_empty_range(self, fetcher):
        """无数据时返回空 DataFrame"""
        fetcher.exchange.fetch_ohlcv.return_value = []

        start = datetime(2024, 1, 15, 8, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc)

        df = fetcher.fetch_ohlcv_batch("BTC/USDT", "1m", start, end)

        assert df.empty
