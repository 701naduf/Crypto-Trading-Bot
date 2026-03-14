"""core/base.py 的单元测试"""

import numpy as np
import pandas as pd
import pytest

from factor_research.core.base import (
    CrossAssetFactor,
    CrossSectionalFactor,
    Factor,
    TimeSeriesFactor,
)
from factor_research.core.types import (
    DataRequest,
    DataType,
    FactorMeta,
    FactorType,
)


# =========================================================================
# 测试用的具体因子实现
# =========================================================================

class DummyTimeSeriesFactor(TimeSeriesFactor):
    """测试用时序因子: 简单地返回收盘价的百分比变化"""

    def meta(self):
        return FactorMeta(
            name="dummy_ts",
            display_name="Dummy TS",
            factor_type=FactorType.TIME_SERIES,
            category="test",
            description="Test time series factor",
            data_requirements=[DataRequest(DataType.OHLCV, timeframe="1m")],
            output_freq="1m",
        )

    def compute_single(self, symbol, data):
        ohlcv = data[DataType.OHLCV]
        ret = ohlcv["close"].pct_change()
        ret.index = pd.to_datetime(ohlcv["timestamp"], utc=True)
        return ret


class DummyCrossSectionalFactor(CrossSectionalFactor):
    """测试用截面因子: 截面排名"""

    def meta(self):
        return FactorMeta(
            name="dummy_cs",
            display_name="Dummy CS",
            factor_type=FactorType.CROSS_SECTIONAL,
            category="test",
            description="Test cross sectional factor",
            data_requirements=[DataRequest(DataType.OHLCV, timeframe="1m")],
            output_freq="1m",
        )

    def compute(self, data):
        ohlcv_dict = data[DataType.OHLCV]
        returns = {}
        for sym, df in ohlcv_dict.items():
            ret = df["close"].pct_change()
            ret.index = pd.to_datetime(df["timestamp"], utc=True)
            returns[sym] = ret
        panel = pd.DataFrame(returns)
        return panel.rank(axis=1, pct=True)


class DummyCrossAssetFactor(CrossAssetFactor):
    """测试用跨标的因子: BTC 收益 → 其他标的"""

    @property
    def input_symbols(self):
        return ["BTC/USDT"]

    @property
    def output_symbols(self):
        return ["ETH/USDT"]

    def meta(self):
        return FactorMeta(
            name="dummy_ca",
            display_name="Dummy CA",
            factor_type=FactorType.CROSS_ASSET,
            category="test",
            description="Test cross asset factor",
            data_requirements=[DataRequest(DataType.OHLCV, timeframe="1m")],
            output_freq="1m",
        )

    def compute(self, data):
        btc = data[DataType.OHLCV]["BTC/USDT"]
        btc_ret = btc["close"].pct_change()
        btc_ret.index = pd.to_datetime(btc["timestamp"], utc=True)
        return pd.DataFrame({"ETH/USDT": btc_ret})


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def sample_ohlcv():
    """生成测试用 OHLCV 数据"""
    timestamps = pd.date_range("2024-01-01", periods=100, freq="1min", tz="UTC")
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.1)
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": prices,
        "high": prices + 0.1,
        "low": prices - 0.1,
        "close": prices,
        "volume": np.random.randint(100, 1000, 100).astype(float),
    })


@pytest.fixture
def sample_ohlcv_eth():
    timestamps = pd.date_range("2024-01-01", periods=100, freq="1min", tz="UTC")
    np.random.seed(123)
    prices = 50 + np.cumsum(np.random.randn(100) * 0.05)
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": prices,
        "high": prices + 0.05,
        "low": prices - 0.05,
        "close": prices,
        "volume": np.random.randint(50, 500, 100).astype(float),
    })


# =========================================================================
# 测试
# =========================================================================

class TestFactor:
    """Factor 基类测试"""

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            Factor()


class TestTimeSeriesFactor:
    """TimeSeriesFactor 测试"""

    def test_meta(self):
        factor = DummyTimeSeriesFactor()
        meta = factor.meta()
        assert meta.name == "dummy_ts"
        assert meta.factor_type == FactorType.TIME_SERIES

    def test_compute(self, sample_ohlcv, sample_ohlcv_eth):
        factor = DummyTimeSeriesFactor()

        data = {
            "BTC/USDT": {DataType.OHLCV: sample_ohlcv},
            "ETH/USDT": {DataType.OHLCV: sample_ohlcv_eth},
        }

        panel = factor.compute(data)

        assert isinstance(panel, pd.DataFrame)
        assert "BTC/USDT" in panel.columns
        assert "ETH/USDT" in panel.columns
        assert len(panel) > 0
        # 第一行是 NaN（pct_change 的第一个值）
        assert panel.iloc[0].isna().all()
        assert panel.iloc[1].notna().all()

    def test_compute_single_symbol(self, sample_ohlcv):
        factor = DummyTimeSeriesFactor()

        data = {"BTC/USDT": {DataType.OHLCV: sample_ohlcv}}
        panel = factor.compute(data)

        assert isinstance(panel, pd.DataFrame)
        assert list(panel.columns) == ["BTC/USDT"]


class TestCrossSectionalFactor:
    """CrossSectionalFactor 测试"""

    def test_compute(self, sample_ohlcv, sample_ohlcv_eth):
        factor = DummyCrossSectionalFactor()

        data = {
            DataType.OHLCV: {
                "BTC/USDT": sample_ohlcv,
                "ETH/USDT": sample_ohlcv_eth,
            }
        }

        panel = factor.compute(data)

        assert isinstance(panel, pd.DataFrame)
        assert "BTC/USDT" in panel.columns
        assert "ETH/USDT" in panel.columns
        # 排名值应该在 0-1 之间
        valid = panel.dropna()
        assert (valid >= 0).all().all()
        assert (valid <= 1).all().all()


class TestCrossAssetFactor:
    """CrossAssetFactor 测试"""

    def test_properties(self):
        factor = DummyCrossAssetFactor()
        assert factor.input_symbols == ["BTC/USDT"]
        assert factor.output_symbols == ["ETH/USDT"]

    def test_compute(self, sample_ohlcv):
        factor = DummyCrossAssetFactor()

        data = {DataType.OHLCV: {"BTC/USDT": sample_ohlcv}}
        panel = factor.compute(data)

        assert isinstance(panel, pd.DataFrame)
        assert "ETH/USDT" in panel.columns
        assert "BTC/USDT" not in panel.columns
