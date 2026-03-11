"""
OrderbookFetcher 测试

测试:
    - _process_message 解析 WebSocket 消息
    - _make_stream_name 流名称生成
    - _stream_to_symbol 映射
    - _get_reconnect_delay 指数退避计算
"""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from data.orderbook_fetcher import OrderbookFetcher


@pytest.fixture
def fetcher():
    """创建 OrderbookFetcher 实例"""
    with patch("data.orderbook_fetcher.settings") as mock_settings:
        mock_settings.SYMBOLS = ["BTC/USDT", "ETH/USDT"]
        mock_settings.ORDERBOOK_DEPTH = 10
        mock_settings.ORDERBOOK_UPDATE_SPEED = "100ms"
        mock_settings.WS_RECONNECT_DELAY = 5
        mock_settings.WS_PING_INTERVAL = 20
        f = OrderbookFetcher()
    return f


class TestMakeStreamName:
    def test_basic(self, fetcher):
        """标准交易对转 stream 名称"""
        assert fetcher._make_stream_name("BTC/USDT") == "btcusdt@depth10@100ms"

    def test_other_pair(self, fetcher):
        assert fetcher._make_stream_name("ETH/USDT") == "ethusdt@depth10@100ms"

    def test_lowercase(self, fetcher):
        """输出应为小写"""
        result = fetcher._make_stream_name("DOGE/USDT")
        assert result == result.lower()


class TestStreamToSymbol:
    def test_mapping(self, fetcher):
        """stream 名称应正确映射回交易对"""
        assert fetcher._stream_to_symbol["btcusdt@depth10@100ms"] == "BTC/USDT"
        assert fetcher._stream_to_symbol["ethusdt@depth10@100ms"] == "ETH/USDT"

    def test_unknown_stream(self, fetcher):
        """未知 stream 不在映射中"""
        assert "unknown@depth10@100ms" not in fetcher._stream_to_symbol


class TestProcessMessage:
    def test_valid_message(self, fetcher):
        """正确解析 Binance combined stream 消息"""
        callback = MagicMock()

        msg = json.dumps({
            "stream": "btcusdt@depth10@100ms",
            "data": {
                "lastUpdateId": 123456,
                "bids": [["42000.00", "1.5"], ["41999.00", "0.8"]],
                "asks": [["42001.00", "2.0"], ["42002.00", "1.2"]],
            }
        })

        fetcher._process_message(msg, callback)

        callback.assert_called_once()
        symbol, snapshot = callback.call_args[0]

        assert symbol == "BTC/USDT"
        assert isinstance(snapshot["timestamp"], datetime)
        assert snapshot["bids"] == [[42000.0, 1.5], [41999.0, 0.8]]
        assert snapshot["asks"] == [[42001.0, 2.0], [42002.0, 1.2]]

    def test_string_to_float_conversion(self, fetcher):
        """Binance 返回的字符串价格/数量应转为 float"""
        callback = MagicMock()

        msg = json.dumps({
            "stream": "ethusdt@depth10@100ms",
            "data": {
                "bids": [["2500.50", "10.123"]],
                "asks": [["2501.00", "5.456"]],
            }
        })

        fetcher._process_message(msg, callback)

        _, snapshot = callback.call_args[0]
        assert isinstance(snapshot["bids"][0][0], float)
        assert isinstance(snapshot["bids"][0][1], float)

    def test_invalid_json(self, fetcher):
        """无效 JSON 不应崩溃"""
        callback = MagicMock()
        fetcher._process_message("not json", callback)
        callback.assert_not_called()

    def test_unknown_stream_ignored(self, fetcher):
        """未知 stream 名的消息被忽略"""
        callback = MagicMock()

        msg = json.dumps({
            "stream": "unknown@depth10@100ms",
            "data": {"bids": [], "asks": []},
        })

        fetcher._process_message(msg, callback)
        callback.assert_not_called()


class TestReconnectDelay:
    def test_exponential_backoff(self, fetcher):
        """重连延迟应指数增长"""
        # WS_RECONNECT_DELAY = 5
        fetcher._reconnect_count = 1
        assert fetcher._get_reconnect_delay() == 5.0  # 5 * 2^0

        fetcher._reconnect_count = 2
        assert fetcher._get_reconnect_delay() == 10.0  # 5 * 2^1

        fetcher._reconnect_count = 3
        assert fetcher._get_reconnect_delay() == 20.0  # 5 * 2^2

    def test_max_delay(self, fetcher):
        """延迟上限为 60 秒"""
        fetcher._reconnect_count = 10
        assert fetcher._get_reconnect_delay() == 60.0

    def test_first_reconnect(self, fetcher):
        """第一次重连延迟为基础值"""
        fetcher._reconnect_count = 1
        assert fetcher._get_reconnect_delay() == 5.0


class TestBuildWsUrl:
    def test_combined_stream_url(self, fetcher):
        """URL 应包含所有 stream 名"""
        url = fetcher._build_ws_url()
        assert "btcusdt@depth10@100ms" in url
        assert "ethusdt@depth10@100ms" in url
        assert url.startswith("wss://")
