"""
MarketFetcher 测试

mock ccxt 合约 API，测试:
    - 4 个 fetch 方法的返回格式
    - _to_futures_symbol / _to_binance_symbol 辅助函数
    - 空数据处理
    - 返回值包含 symbol 字段
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data_infra.data.market_fetcher import MarketFetcher, _to_futures_symbol, _to_binance_symbol


# =========================================================================
# 辅助函数测试
# =========================================================================

class TestHelpers:
    def test_to_futures_symbol_basic(self):
        """现货格式转合约格式"""
        assert _to_futures_symbol("BTC/USDT") == "BTC/USDT:USDT"

    def test_to_futures_symbol_already_futures(self):
        """已是合约格式时不变"""
        assert _to_futures_symbol("BTC/USDT:USDT") == "BTC/USDT:USDT"

    def test_to_binance_symbol(self):
        """标准格式转 Binance API 格式"""
        assert _to_binance_symbol("BTC/USDT") == "BTCUSDT"
        assert _to_binance_symbol("ETH/USDT") == "ETHUSDT"
        assert _to_binance_symbol("DOGE/USDT") == "DOGEUSDT"


# =========================================================================
# MarketFetcher 测试
# =========================================================================

@pytest.fixture
def fetcher():
    """创建 MarketFetcher，mock 掉 ccxt 交易所实例"""
    with patch("data_infra.data.market_fetcher.ccxt") as mock_ccxt:
        mock_exchange = MagicMock()
        mock_ccxt.binance.return_value = mock_exchange
        f = MarketFetcher()
        f.exchange = mock_exchange
        yield f


class TestFetchFundingRate:
    def test_returns_dataframe_with_symbol(self, fetcher):
        """返回 DataFrame 应包含 symbol 列"""
        fetcher.exchange.fetch_funding_rate_history.return_value = [
            {"timestamp": 1705305600000, "fundingRate": 0.0001},
            {"timestamp": 1705334400000, "fundingRate": -0.0002},
        ]

        df = fetcher.fetch_funding_rate("BTC/USDT")

        assert "symbol" in df.columns
        assert "timestamp" in df.columns
        assert "funding_rate" in df.columns
        assert len(df) == 2
        assert (df["symbol"] == "BTC/USDT").all()

    def test_empty_response(self, fetcher):
        """API 返回空列表时应返回空 DataFrame 且包含正确列"""
        fetcher.exchange.fetch_funding_rate_history.return_value = []

        df = fetcher.fetch_funding_rate("BTC/USDT")

        assert df.empty
        assert "symbol" in df.columns
        assert "timestamp" in df.columns
        assert "funding_rate" in df.columns


class TestFetchOpenInterest:
    def test_returns_dict_with_symbol(self, fetcher):
        """返回 dict 应包含 symbol 键"""
        fetcher.exchange.fapiPublicGetOpenInterest.return_value = {
            "openInterest": "1234.56",
        }
        fetcher.exchange.fetch_ticker.return_value = {"last": 42000.0}

        result = fetcher.fetch_open_interest("BTC/USDT")

        assert result["symbol"] == "BTC/USDT"
        assert "timestamp" in result
        assert result["open_interest"] == 1234.56
        assert result["open_interest_value"] == 1234.56 * 42000.0

    def test_zero_price(self, fetcher):
        """价格为 0 时持仓价值应为 0"""
        fetcher.exchange.fapiPublicGetOpenInterest.return_value = {
            "openInterest": "100.0",
        }
        fetcher.exchange.fetch_ticker.return_value = {"last": 0}

        result = fetcher.fetch_open_interest("ETH/USDT")

        assert result["open_interest_value"] == 0


class TestFetchLongShortRatio:
    def test_returns_dataframe_with_symbol(self, fetcher):
        """返回 DataFrame 应包含 symbol 列"""
        fetcher.exchange.fapiDataGetTopLongShortPositionRatio.return_value = [
            {
                "timestamp": "1705305600000",
                "longAccount": "0.55",
                "shortAccount": "0.45",
                "longShortRatio": "1.222",
            },
        ]

        df = fetcher.fetch_long_short_ratio("BTC/USDT")

        assert "symbol" in df.columns
        assert (df["symbol"] == "BTC/USDT").all()
        assert "long_ratio" in df.columns
        assert "short_ratio" in df.columns
        assert "long_short_ratio" in df.columns

    def test_empty_response(self, fetcher):
        """空响应返回空 DataFrame 且列正确"""
        fetcher.exchange.fapiDataGetTopLongShortPositionRatio.return_value = []

        df = fetcher.fetch_long_short_ratio("ETH/USDT")

        assert df.empty
        assert "symbol" in df.columns


class TestFetchTakerBuySellVolume:
    def test_returns_dataframe_with_symbol(self, fetcher):
        """返回 DataFrame 应包含 symbol 列"""
        fetcher.exchange.fapiDataGetTakerlongshortRatio.return_value = [
            {
                "timestamp": "1705305600000",
                "buyVol": "100.5",
                "sellVol": "200.3",
                "buySellRatio": "0.502",
            },
        ]

        df = fetcher.fetch_taker_buy_sell_volume("BTC/USDT")

        assert "symbol" in df.columns
        assert (df["symbol"] == "BTC/USDT").all()
        assert "buy_vol" in df.columns
        assert "sell_vol" in df.columns
        assert "buy_sell_ratio" in df.columns

    def test_empty_response(self, fetcher):
        """空响应返回空 DataFrame 且列正确"""
        fetcher.exchange.fapiDataGetTakerlongshortRatio.return_value = []

        df = fetcher.fetch_taker_buy_sell_volume("SOL/USDT")

        assert df.empty
        assert "symbol" in df.columns

    def test_missing_fields_default_to_zero(self, fetcher):
        """字段缺失时应使用默认值 0"""
        fetcher.exchange.fapiDataGetTakerlongshortRatio.return_value = [
            {"timestamp": "1705305600000"},  # 缺少 buyVol, sellVol, buySellRatio
        ]

        df = fetcher.fetch_taker_buy_sell_volume("BTC/USDT")

        assert df["buy_vol"].iloc[0] == 0
        assert df["sell_vol"].iloc[0] == 0
        assert df["buy_sell_ratio"].iloc[0] == 0
