"""core/types.py 的单元测试"""

import pytest

from factor_research.core.types import (
    DataRequest,
    DataType,
    FactorMeta,
    FactorType,
)


class TestFactorType:
    """FactorType 枚举测试"""

    def test_enum_values(self):
        assert FactorType.TIME_SERIES.value == "time_series"
        assert FactorType.CROSS_SECTIONAL.value == "cross_sectional"
        assert FactorType.CROSS_ASSET.value == "cross_asset"

    def test_enum_from_value(self):
        assert FactorType("time_series") == FactorType.TIME_SERIES


class TestDataType:
    """DataType 枚举测试"""

    def test_all_data_types_exist(self):
        expected = ["ohlcv", "tick", "orderbook", "funding_rate",
                     "open_interest", "long_short_ratio", "taker_buy_sell"]
        actual = [dt.value for dt in DataType]
        assert sorted(actual) == sorted(expected)


class TestDataRequest:
    """DataRequest 数据类测试"""

    def test_minimal_creation(self):
        req = DataRequest(DataType.OHLCV)
        assert req.data_type == DataType.OHLCV
        assert req.timeframe is None
        assert req.orderbook_levels is None
        assert req.lookback_bars == 0
        assert req.symbols is None

    def test_full_creation(self):
        req = DataRequest(
            data_type=DataType.ORDERBOOK,
            orderbook_levels=10,
            lookback_bars=100,
            symbols=["BTC/USDT"],
        )
        assert req.orderbook_levels == 10
        assert req.lookback_bars == 100
        assert req.symbols == ["BTC/USDT"]

    def test_ohlcv_with_timeframe(self):
        req = DataRequest(DataType.OHLCV, timeframe="5m", lookback_bars=60)
        assert req.timeframe == "5m"
        assert req.lookback_bars == 60


class TestFactorMeta:
    """FactorMeta 数据类测试"""

    def test_creation(self):
        meta = FactorMeta(
            name="test_factor",
            display_name="测试因子",
            factor_type=FactorType.TIME_SERIES,
            category="test",
            description="测试用因子",
            data_requirements=[DataRequest(DataType.OHLCV, timeframe="1m")],
            output_freq="1m",
        )
        assert meta.name == "test_factor"
        assert meta.factor_type == FactorType.TIME_SERIES
        assert meta.params == {}
        assert meta.version == "1.0"

    def test_with_params(self):
        meta = FactorMeta(
            name="test",
            display_name="test",
            factor_type=FactorType.TIME_SERIES,
            category="test",
            description="test",
            data_requirements=[],
            output_freq="1m",
            params={"window": 60, "decay": 0.94},
            author="test_author",
            version="2.0",
        )
        assert meta.params["window"] == 60
        assert meta.author == "test_author"
        assert meta.version == "2.0"
