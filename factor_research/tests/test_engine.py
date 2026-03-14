"""core/engine.py 的单元测试

覆盖: FactorEngine 的完整编排流程
    - 三种因子类型的数据准备和计算
    - save/no-save 模式
    - 批量计算、分类过滤、容错
    - 数据路由的格式验证
    - 边界场景（未注册因子、空数据）

Mock 策略: MockDataReader 替代真实数据库读取
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from factor_research.core.base import (
    CrossAssetFactor,
    CrossSectionalFactor,
    TimeSeriesFactor,
)
from factor_research.core.engine import FactorEngine
from factor_research.core.registry import FactorRegistry
from factor_research.core.types import (
    DataRequest,
    DataType,
    FactorMeta,
    FactorType,
)
from factor_research.store.factor_store import FactorStore


# =========================================================================
# 测试用因子实现
# =========================================================================

class _TSFactor(TimeSeriesFactor):
    """测试用时序因子"""

    def meta(self):
        return FactorMeta(
            name="test_ts",
            display_name="Test TS",
            factor_type=FactorType.TIME_SERIES,
            category="test",
            description="test",
            data_requirements=[DataRequest(DataType.OHLCV, timeframe="1m")],
            output_freq="1m",
        )

    def compute_single(self, symbol, data):
        ohlcv = data[DataType.OHLCV]
        ret = ohlcv["close"].pct_change()
        ret.index = pd.to_datetime(ohlcv["timestamp"], utc=True)
        return ret


class _CSFactor(CrossSectionalFactor):
    """测试用截面因子"""

    def meta(self):
        return FactorMeta(
            name="test_cs",
            display_name="Test CS",
            factor_type=FactorType.CROSS_SECTIONAL,
            category="test_cs_cat",
            description="test",
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


class _CAFactor(CrossAssetFactor):
    """测试用跨标的因子"""

    @property
    def input_symbols(self):
        return ["BTC/USDT"]

    @property
    def output_symbols(self):
        return ["ETH/USDT"]

    def meta(self):
        return FactorMeta(
            name="test_ca",
            display_name="Test CA",
            factor_type=FactorType.CROSS_ASSET,
            category="test_ca_cat",
            description="test",
            data_requirements=[DataRequest(DataType.OHLCV, timeframe="1m")],
            output_freq="1m",
        )

    def compute(self, data):
        btc = data[DataType.OHLCV]["BTC/USDT"]
        btc_ret = btc["close"].pct_change()
        btc_ret.index = pd.to_datetime(btc["timestamp"], utc=True)
        return pd.DataFrame({"ETH/USDT": btc_ret})


class _FailingFactor(TimeSeriesFactor):
    """测试用: 计算时抛异常"""

    def meta(self):
        return FactorMeta(
            name="test_fail",
            display_name="Test Fail",
            factor_type=FactorType.TIME_SERIES,
            category="test",
            description="always fails",
            data_requirements=[DataRequest(DataType.OHLCV, timeframe="1m")],
            output_freq="1m",
        )

    def compute_single(self, symbol, data):
        raise RuntimeError("模拟计算失败")


class _MomentumFactor(TimeSeriesFactor):
    """测试用: 不同 category 的因子"""

    def meta(self):
        return FactorMeta(
            name="test_momentum",
            display_name="Test Momentum",
            factor_type=FactorType.TIME_SERIES,
            category="momentum",
            description="test momentum",
            data_requirements=[DataRequest(DataType.OHLCV, timeframe="1m")],
            output_freq="1m",
        )

    def compute_single(self, symbol, data):
        ohlcv = data[DataType.OHLCV]
        ret = ohlcv["close"].pct_change()
        ret.index = pd.to_datetime(ohlcv["timestamp"], utc=True)
        return ret


# =========================================================================
# Fixtures
# =========================================================================

def _make_ohlcv(n=50):
    """生成合成 OHLCV DataFrame"""
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(n) * 0.1)
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": prices,
        "high": prices + 0.1,
        "low": prices - 0.1,
        "close": prices,
        "volume": np.random.randint(100, 1000, n).astype(float),
    })


@pytest.fixture
def mock_reader():
    """Mock DataReader: 各 get_* 方法返回合成数据"""
    reader = MagicMock()
    ohlcv_btc = _make_ohlcv(50)
    np.random.seed(123)
    ohlcv_eth = _make_ohlcv(50)

    def get_ohlcv(symbol, timeframe, **kwargs):
        return ohlcv_btc.copy() if "BTC" in symbol else ohlcv_eth.copy()

    reader.get_ohlcv = MagicMock(side_effect=get_ohlcv)
    return reader


@pytest.fixture
def registry():
    """每个测试用全新的注册表"""
    reg = FactorRegistry()
    reg.clear()
    return reg


@pytest.fixture
def engine(mock_reader, registry, tmp_path):
    """使用 Mock 组件的 FactorEngine"""
    store = FactorStore(base_dir=str(tmp_path / "factors"))
    return FactorEngine(reader=mock_reader, store=store, registry=registry)


@pytest.fixture
def store(tmp_path):
    return FactorStore(base_dir=str(tmp_path / "factors"))


# =========================================================================
# 测试
# =========================================================================

class TestComputeFactorTimeSeries:

    def test_compute_ts_factor(self, engine, registry):
        """时序因子端到端: 注册→计算→返回面板"""
        registry.register(_TSFactor)
        panel = engine.compute_factor(
            "test_ts",
            symbols=["BTC/USDT", "ETH/USDT"],
            save=False,
        )
        assert isinstance(panel, pd.DataFrame)
        assert "BTC/USDT" in panel.columns
        assert "ETH/USDT" in panel.columns
        assert len(panel) > 0

    def test_compute_cs_factor(self, engine, registry):
        """截面因子端到端"""
        registry.register(_CSFactor)
        panel = engine.compute_factor(
            "test_cs",
            symbols=["BTC/USDT", "ETH/USDT"],
            save=False,
        )
        assert isinstance(panel, pd.DataFrame)
        assert "BTC/USDT" in panel.columns
        assert "ETH/USDT" in panel.columns

    def test_compute_ca_factor(self, engine, registry):
        """跨标的因子端到端: 输出列只含 output_symbols"""
        registry.register(_CAFactor)
        panel = engine.compute_factor(
            "test_ca",
            symbols=["BTC/USDT", "ETH/USDT"],
            save=False,
        )
        assert isinstance(panel, pd.DataFrame)
        assert "ETH/USDT" in panel.columns
        assert "BTC/USDT" not in panel.columns


class TestComputeFactorSave:

    def test_save_true(self, engine, registry, store):
        """save=True 时因子被持久化"""
        registry.register(_TSFactor)
        engine._store = store
        engine.compute_factor(
            "test_ts",
            symbols=["BTC/USDT"],
            save=True,
        )
        assert store.exists("test_ts")

    def test_save_false(self, engine, registry, store):
        """save=False 时因子不持久化"""
        registry.register(_TSFactor)
        engine._store = store
        engine.compute_factor(
            "test_ts",
            symbols=["BTC/USDT"],
            save=False,
        )
        assert not store.exists("test_ts")


class TestComputeAll:

    def test_compute_all_batch(self, engine, registry):
        """批量计算所有已注册因子"""
        registry.register(_TSFactor)
        registry.register(_MomentumFactor)
        results = engine.compute_all(symbols=["BTC/USDT"])
        assert len(results) == 2
        assert "test_ts" in results
        assert "test_momentum" in results

    def test_compute_all_categories_filter(self, engine, registry):
        """按 category 过滤批量计算"""
        registry.register(_TSFactor)       # category="test"
        registry.register(_MomentumFactor) # category="momentum"
        results = engine.compute_all(
            symbols=["BTC/USDT"],
            categories=["momentum"],
        )
        assert len(results) == 1
        assert "test_momentum" in results

    def test_compute_all_single_failure_no_block(self, engine, registry):
        """单因子失败不阻断批量计算"""
        registry.register(_TSFactor)
        registry.register(_FailingFactor)
        results = engine.compute_all(symbols=["BTC/USDT"])
        # _FailingFactor 失败，_TSFactor 成功
        assert "test_ts" in results
        assert "test_fail" not in results


class TestPrepareData:

    def test_prepare_ts_format(self, engine, registry, mock_reader):
        """时序因子数据格式: {symbol: {DataType: DataFrame}}"""
        registry.register(_TSFactor)
        factor = registry.get("test_ts")
        data = engine._prepare_data(
            factor, ["BTC/USDT", "ETH/USDT"], None, None
        )
        assert "BTC/USDT" in data
        assert "ETH/USDT" in data
        assert DataType.OHLCV in data["BTC/USDT"]
        assert isinstance(data["BTC/USDT"][DataType.OHLCV], pd.DataFrame)

    def test_prepare_cs_format(self, engine, registry, mock_reader):
        """截面因子数据格式: {DataType: {symbol: DataFrame}}"""
        registry.register(_CSFactor)
        factor = registry.get("test_cs")
        data = engine._prepare_data(
            factor, ["BTC/USDT", "ETH/USDT"], None, None
        )
        assert DataType.OHLCV in data
        assert "BTC/USDT" in data[DataType.OHLCV]
        assert "ETH/USDT" in data[DataType.OHLCV]

    def test_prepare_ca_only_reads_input_symbols(self, engine, registry, mock_reader):
        """跨标的因子只读取 input_symbols 的数据"""
        registry.register(_CAFactor)
        factor = registry.get("test_ca")
        data = engine._prepare_data(
            factor, ["BTC/USDT", "ETH/USDT"], None, None
        )
        # 只有 BTC/USDT 的数据（input_symbols）
        assert "BTC/USDT" in data[DataType.OHLCV]
        # ETH/USDT 不在数据中（它是 output_symbol，不是 input_symbol）
        assert "ETH/USDT" not in data[DataType.OHLCV]

    def test_read_data_passes_timeframe(self, engine, registry, mock_reader):
        """OHLCV 数据读取传递 timeframe 参数"""
        registry.register(_TSFactor)
        factor = registry.get("test_ts")
        engine._prepare_data(factor, ["BTC/USDT"], None, None)
        # 验证 get_ohlcv 被调用时传递了 timeframe="1m"
        mock_reader.get_ohlcv.assert_called()
        call_args = mock_reader.get_ohlcv.call_args
        assert call_args[0][1] == "1m"  # 第二个位置参数是 timeframe


class _FamilyFactor(TimeSeriesFactor):
    """测试用: 参数化因子族"""

    def __init__(self, lookback: int = 5):
        self.lookback = lookback

    def meta(self):
        return FactorMeta(
            name=f"fam_{self.lookback}",
            display_name=f"Fam {self.lookback}",
            factor_type=FactorType.TIME_SERIES,
            category="test_fam",
            description="test family",
            data_requirements=[DataRequest(DataType.OHLCV, timeframe="1m")],
            output_freq="1m",
            params={"lookback": self.lookback},
            family="test_family",
        )

    def compute_single(self, symbol, data):
        ohlcv = data[DataType.OHLCV]
        ret = ohlcv["close"].pct_change()
        ret.index = pd.to_datetime(ohlcv["timestamp"], utc=True)
        return ret


class TestComputeFactorInstance:

    def test_compute_factor_instance(self, engine, registry):
        """compute_factor_instance 接受因子实例直接计算"""
        factor = _TSFactor()
        panel = engine.compute_factor_instance(
            factor,
            symbols=["BTC/USDT"],
            save=False,
        )
        assert isinstance(panel, pd.DataFrame)
        assert "BTC/USDT" in panel.columns
        assert len(panel) > 0

    def test_compute_factor_instance_default_no_save(self, engine, registry, store):
        """compute_factor_instance 默认 save=False"""
        engine._store = store
        factor = _TSFactor()
        engine.compute_factor_instance(factor, symbols=["BTC/USDT"])
        assert not store.exists("test_ts")


class TestPrepareDataPublic:

    def test_prepare_data_public(self, engine, registry, mock_reader):
        """prepare_data() 公开接口返回正确格式数据"""
        factor = _TSFactor()
        data = engine.prepare_data(
            factor, symbols=["BTC/USDT", "ETH/USDT"]
        )
        assert "BTC/USDT" in data
        assert "ETH/USDT" in data
        assert DataType.OHLCV in data["BTC/USDT"]


class TestComputeFamily:

    def test_compute_family(self, engine, registry, mock_reader):
        """compute_family 共享数据计算族内所有变体"""
        registry.register(_FamilyFactor, lookback=5)
        registry.register(_FamilyFactor, lookback=10)
        results = engine.compute_family(
            "test_family",
            symbols=["BTC/USDT"],
            save=False,
        )
        assert len(results) == 2
        assert "fam_5" in results
        assert "fam_10" in results

    def test_compute_family_shared_data(self, engine, registry, mock_reader):
        """compute_family 数据准备只调用一次"""
        registry.register(_FamilyFactor, lookback=5)
        registry.register(_FamilyFactor, lookback=10)
        registry.register(_FamilyFactor, lookback=30)
        mock_reader.get_ohlcv.reset_mock()
        engine.compute_family(
            "test_family",
            symbols=["BTC/USDT"],
            save=False,
        )
        # 数据只准备一次: 1 symbol × 1 次 = 1 次调用
        assert mock_reader.get_ohlcv.call_count == 1

    def test_compute_family_nonexistent_raises(self, engine, registry):
        """不存在的族名 → KeyError"""
        with pytest.raises(KeyError, match="不存在"):
            engine.compute_family("nonexistent", symbols=["BTC/USDT"])


class TestComputeAllFamilyGrouped:

    def test_compute_all_groups_families(self, engine, registry, mock_reader):
        """compute_all 按族分组共享数据"""
        # 注册一个族（2个变体）和一个独立因子
        registry.register(_FamilyFactor, lookback=5)
        registry.register(_FamilyFactor, lookback=10)
        registry.register(_TSFactor)  # 独立因子 (family="")
        mock_reader.get_ohlcv.reset_mock()
        results = engine.compute_all(symbols=["BTC/USDT"])
        assert len(results) == 3
        assert "fam_5" in results
        assert "fam_10" in results
        assert "test_ts" in results
        # 族内共享: 1 次 (族) + 1 次 (独立) = 2 次调用
        assert mock_reader.get_ohlcv.call_count == 2


class TestEdgeCases:

    def test_unregistered_factor_raises(self, engine):
        """未注册因子 → KeyError"""
        with pytest.raises(KeyError, match="未注册"):
            engine.compute_factor("nonexistent", symbols=["BTC/USDT"])