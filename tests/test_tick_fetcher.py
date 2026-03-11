"""
TickFetcher 测试

mock ccxt.async_support.binance.fetch_trades，测试:
    - 返回 DataFrame 列名和格式
    - trade_id 排序
    - 冷启动两种模式
    - close 资源释放
    - fetch_until_latest 追赶逻辑
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from data.tick_fetcher import TickFetcher


@pytest.fixture
def fetcher():
    """创建 TickFetcher，mock 掉 ccxt async 交易所实例"""
    with patch("data.tick_fetcher.ccxt_async") as mock_ccxt:
        mock_exchange = MagicMock()
        mock_exchange.fetch_trades = AsyncMock()
        mock_exchange.close = AsyncMock()
        mock_ccxt.binance.return_value = mock_exchange
        f = TickFetcher()
        f.exchange = mock_exchange
        yield f


def _make_trades(start_id, count):
    """生成模拟 trade 数据"""
    return [
        {
            "id": str(start_id + i),
            "timestamp": 1705305600000 + i * 100,
            "symbol": "BTC/USDT",
            "side": "buy" if i % 2 == 0 else "sell",
            "price": 42000.0 + i,
            "amount": 0.001 * (i + 1),
        }
        for i in range(count)
    ]


class TestFetchTrades:
    def test_returns_correct_columns(self, fetcher):
        """返回 DataFrame 应包含标准列"""
        fetcher.exchange.fetch_trades.return_value = _make_trades(100, 3)

        df = asyncio.run(fetcher.fetch_trades("BTC/USDT", from_id=99))

        assert list(df.columns) == ["trade_id", "timestamp", "price", "amount", "side"]
        assert len(df) == 3

    def test_trade_id_sorted(self, fetcher):
        """trade_id 应按升序排列"""
        # 故意乱序返回
        trades = _make_trades(100, 5)
        trades.reverse()
        fetcher.exchange.fetch_trades.return_value = trades

        df = asyncio.run(fetcher.fetch_trades("BTC/USDT"))

        assert df["trade_id"].is_monotonic_increasing

    def test_empty_response(self, fetcher):
        """API 返回空列表时应返回空 DataFrame"""
        fetcher.exchange.fetch_trades.return_value = []

        df = asyncio.run(fetcher.fetch_trades("BTC/USDT"))

        assert df.empty
        assert list(df.columns) == ["trade_id", "timestamp", "price", "amount", "side"]

    def test_from_id_passed_to_params(self, fetcher):
        """from_id 参数应正确传递给 ccxt"""
        fetcher.exchange.fetch_trades.return_value = _make_trades(200, 1)

        asyncio.run(fetcher.fetch_trades("BTC/USDT", from_id=199))

        call_kwargs = fetcher.exchange.fetch_trades.call_args
        assert call_kwargs[1]["params"]["fromId"] == 199


class TestFetchUntilLatest:
    def test_single_batch(self, fetcher):
        """不满额时一次即完成"""
        fetcher.exchange.fetch_trades.return_value = _make_trades(100, 3)

        with patch("data.tick_fetcher.settings") as mock_settings:
            mock_settings.TICK_FETCH_LIMIT = 1000
            df = asyncio.run(fetcher.fetch_until_latest("BTC/USDT", from_id=99))

        assert len(df) == 3

    def test_multi_batch(self, fetcher):
        """满额时自动继续拉取"""
        # 第一批: 满额（5条，假设 limit=5）
        # 第二批: 不满额（2条），停止
        batch1 = _make_trades(100, 5)
        batch2 = _make_trades(105, 2)
        fetcher.exchange.fetch_trades.side_effect = [batch1, batch2]

        with patch("data.tick_fetcher.settings") as mock_settings:
            mock_settings.TICK_FETCH_LIMIT = 5
            df = asyncio.run(fetcher.fetch_until_latest("BTC/USDT", from_id=99))

        assert len(df) == 7
        assert fetcher.exchange.fetch_trades.call_count == 2

    def test_empty_from_start(self, fetcher):
        """起始就无数据时返回空 DataFrame"""
        fetcher.exchange.fetch_trades.return_value = []

        with patch("data.tick_fetcher.settings") as mock_settings:
            mock_settings.TICK_FETCH_LIMIT = 1000
            df = asyncio.run(fetcher.fetch_until_latest("BTC/USDT", from_id=99))

        assert df.empty


class TestColdStart:
    def test_latest_mode(self, fetcher):
        """latest 模式: 取最新 trades 的最小 id"""
        fetcher.exchange.fetch_trades.return_value = _make_trades(500, 10)

        with patch("data.tick_fetcher.settings") as mock_settings:
            mock_settings.TICK_COLD_START_MODE = "latest"
            result = asyncio.run(fetcher.resolve_cold_start("BTC/USDT"))

        assert result == 500  # 最小的 trade_id

    def test_from_date_mode(self, fetcher):
        """from_date 模式: 通过 startTime 获取第一批"""
        fetcher.exchange.fetch_trades.return_value = _make_trades(1000, 10)

        with patch("data.tick_fetcher.settings") as mock_settings:
            mock_settings.TICK_COLD_START_MODE = "from_date"
            mock_settings.TICK_COLD_START_DATE = "2024-01-01"
            result = asyncio.run(fetcher.resolve_cold_start("BTC/USDT"))

        assert result == 1000

    def test_latest_mode_empty(self, fetcher):
        """latest 模式无数据时返回 None"""
        fetcher.exchange.fetch_trades.return_value = []

        with patch("data.tick_fetcher.settings") as mock_settings:
            mock_settings.TICK_COLD_START_MODE = "latest"
            result = asyncio.run(fetcher.resolve_cold_start("BTC/USDT"))

        assert result is None


class TestClose:
    def test_close_releases_resources(self, fetcher):
        """close 应调用交易所 close"""
        asyncio.run(fetcher.close())

        fetcher.exchange.close.assert_called_once()
