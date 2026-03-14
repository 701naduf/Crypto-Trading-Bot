"""示例因子实现的单元测试

覆盖:
    - momentum/returns.py: MultiScaleReturns 因子家族（参数化）
    - microstructure/imbalance.py: OrderbookImbalance 因子

测试维度:
    - meta() 元数据正确性
    - 参数化构造
    - 注册表中可查到
    - family 字段正确性
    - compute_single 基本计算
    - 已知答案验证
    - 空数据 / 数据不足 / 边界场景
"""

import numpy as np
import pandas as pd
import pytest

from factor_research.core.registry import FactorRegistry
from factor_research.core.types import DataType, FactorType

# 导入因子类（导入即触发 @register_factor_family / @register_factor 注册到默认注册表）
from factor_research.factors.momentum.returns import MultiScaleReturns
from factor_research.factors.microstructure.imbalance import OrderbookImbalance


# =========================================================================
# Fixtures
# =========================================================================

def _make_ohlcv(n=100, seed=42):
    """生成合成 OHLCV DataFrame"""
    np.random.seed(seed)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    prices = 100 + np.cumsum(np.random.randn(n) * 0.1)
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": prices,
        "high": prices + 0.1,
        "low": prices - 0.1,
        "close": prices,
        "volume": np.random.randint(100, 1000, n).astype(float),
    })


def _make_orderbook(n=100, levels=10, seed=42):
    """生成合成订单簿 DataFrame"""
    np.random.seed(seed)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="100ms", tz="UTC")
    data = {"timestamp": timestamps}
    for i in range(levels):
        data[f"bid_price_{i}"] = 100 - i * 0.01 + np.random.randn(n) * 0.001
        data[f"bid_qty_{i}"] = np.random.rand(n) * 10 + 0.1
        data[f"ask_price_{i}"] = 100 + (i + 1) * 0.01 + np.random.randn(n) * 0.001
        data[f"ask_qty_{i}"] = np.random.rand(n) * 10 + 0.1
    return pd.DataFrame(data)


# =========================================================================
# MultiScaleReturns 测试
# =========================================================================

class TestMultiScaleReturns:

    def test_meta_default(self):
        """默认构造 (lookback=5) 的 meta 正确"""
        factor = MultiScaleReturns()
        meta = factor.meta()
        assert meta.name == "returns_5m"
        assert meta.category == "momentum"
        assert meta.factor_type == FactorType.TIME_SERIES
        assert meta.params["lookback"] == 5

    def test_meta_parameterized(self):
        """参数化构造 (lookback=30) 的 meta 正确"""
        factor = MultiScaleReturns(lookback=30)
        meta = factor.meta()
        assert meta.name == "returns_30m"
        assert meta.params["lookback"] == 30

    def test_family_field(self):
        """family 字段正确设置"""
        factor = MultiScaleReturns()
        assert factor.meta().family == "multi_scale_returns"
        factor2 = MultiScaleReturns(lookback=60)
        assert factor2.meta().family == "multi_scale_returns"

    def test_all_variants_registered(self):
        """4 个收益率因子全部可在注册表中找到"""
        reg = FactorRegistry()
        for lb in [5, 10, 30, 60]:
            reg.register(MultiScaleReturns, lookback=lb)
        for name in ["returns_5m", "returns_10m", "returns_30m", "returns_60m"]:
            assert name in reg, f"因子 {name} 未注册"

    def test_registered_instance_has_correct_params(self):
        """注册表 get() 返回的实例参数正确"""
        reg = FactorRegistry()
        reg.register(MultiScaleReturns, lookback=10)
        factor = reg.get("returns_10m")
        assert isinstance(factor, MultiScaleReturns)
        assert factor.lookback == 10

    def test_compute_single_basic(self):
        """100 根 K线 → 返回非空 Series"""
        factor = MultiScaleReturns()
        ohlcv = _make_ohlcv(100)
        result = factor.compute_single("BTC/USDT", {DataType.OHLCV: ohlcv})
        assert isinstance(result, pd.Series)
        assert len(result) > 0

    def test_compute_single_known_answer(self):
        """已知答案: close = [100, 101, ..., 109] → returns_5m[5] = 105/100 - 1 = 0.05"""
        timestamps = pd.date_range("2024-01-01", periods=10, freq="1min", tz="UTC")
        close = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]
        ohlcv = pd.DataFrame({
            "timestamp": timestamps,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [1000.0] * 10,
        })
        factor = MultiScaleReturns()
        result = factor.compute_single("BTC/USDT", {DataType.OHLCV: ohlcv})
        # t=5: 105/100 - 1 = 0.05
        assert result.iloc[0] == pytest.approx(0.05, abs=1e-10)

    def test_known_answer_different_lookback(self):
        """已知答案: lookback=N 时 returns = close / close.shift(N) - 1"""
        timestamps = pd.date_range("2024-01-01", periods=15, freq="1min", tz="UTC")
        close = list(range(100, 115))
        ohlcv = pd.DataFrame({
            "timestamp": timestamps,
            "open": close,
            "high": close,
            "low": close,
            "close": [float(c) for c in close],
            "volume": [1000.0] * 15,
        })
        factor = MultiScaleReturns(lookback=10)
        result = factor.compute_single("BTC/USDT", {DataType.OHLCV: ohlcv})
        # t=10: 110/100 - 1 = 0.1
        assert result.iloc[0] == pytest.approx(0.1, abs=1e-10)

    def test_compute_single_empty_data(self):
        """空 DataFrame → 返回空 Series"""
        factor = MultiScaleReturns()
        empty_ohlcv = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        result = factor.compute_single("BTC/USDT", {DataType.OHLCV: empty_ohlcv})
        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_compute_single_insufficient_data(self):
        """数据不足 (len < lookback+1) → 返回空 Series"""
        factor = MultiScaleReturns()
        ohlcv = _make_ohlcv(3)  # 只有 3 根 K线, lookback=5 需要至少 6
        result = factor.compute_single("BTC/USDT", {DataType.OHLCV: ohlcv})
        assert len(result) == 0

    def test_compute_single_datetime_index(self):
        """返回的 Series 具有 DatetimeIndex (UTC)"""
        factor = MultiScaleReturns()
        ohlcv = _make_ohlcv(100)
        result = factor.compute_single("BTC/USDT", {DataType.OHLCV: ohlcv})
        assert hasattr(result.index, "tz")
        assert str(result.index.tz) == "UTC"


# =========================================================================
# OrderbookImbalance 测试
# =========================================================================

class TestOrderbookImbalance:

    def test_meta(self):
        """meta 正确"""
        factor = OrderbookImbalance()
        meta = factor.meta()
        assert meta.name == "orderbook_imbalance"
        assert meta.category == "microstructure"
        assert meta.factor_type == FactorType.TIME_SERIES

    def test_parameterized_init(self):
        """参数化构造正常工作"""
        factor = OrderbookImbalance(levels=5, resample_freq="2s")
        meta = factor.meta()
        assert meta.params["levels"] == 5
        assert meta.params["resample_freq"] == "2s"

    def test_registered(self):
        """注册表中可查到"""
        reg = FactorRegistry()
        reg.register(OrderbookImbalance)
        assert "orderbook_imbalance" in reg

    def test_compute_single_basic(self):
        """合成 10 档订单簿 → 返回非空 Series"""
        factor = OrderbookImbalance()
        ob = _make_orderbook(100)
        result = factor.compute_single("BTC/USDT", {DataType.ORDERBOOK: ob})
        assert isinstance(result, pd.Series)
        assert len(result) > 0

    def test_value_range(self):
        """所有值在 [-1, 1]"""
        factor = OrderbookImbalance()
        ob = _make_orderbook(200)
        result = factor.compute_single("BTC/USDT", {DataType.ORDERBOOK: ob})
        valid = result.dropna()
        assert valid.min() >= -1.0
        assert valid.max() <= 1.0

    def test_known_answer_all_bid(self):
        """bid 全 100, ask 全 0 → imbalance = 1.0"""
        timestamps = pd.date_range("2024-01-01", periods=10, freq="100ms", tz="UTC")
        data = {"timestamp": timestamps}
        for i in range(10):
            data[f"bid_price_{i}"] = [100.0] * 10
            data[f"bid_qty_{i}"] = [100.0] * 10
            data[f"ask_price_{i}"] = [101.0] * 10
            data[f"ask_qty_{i}"] = [0.0] * 10
        ob = pd.DataFrame(data)
        factor = OrderbookImbalance()
        result = factor.compute_single("BTC/USDT", {DataType.ORDERBOOK: ob})
        # bid=1000, ask=0, total=1000 → imbalance = 1.0
        valid = result.dropna()
        assert len(valid) > 0
        assert all(v == pytest.approx(1.0) for v in valid)

    def test_known_answer_all_ask(self):
        """bid 全 0, ask 全 100 → imbalance = -1.0"""
        timestamps = pd.date_range("2024-01-01", periods=10, freq="100ms", tz="UTC")
        data = {"timestamp": timestamps}
        for i in range(10):
            data[f"bid_price_{i}"] = [100.0] * 10
            data[f"bid_qty_{i}"] = [0.0] * 10
            data[f"ask_price_{i}"] = [101.0] * 10
            data[f"ask_qty_{i}"] = [100.0] * 10
        ob = pd.DataFrame(data)
        factor = OrderbookImbalance()
        result = factor.compute_single("BTC/USDT", {DataType.ORDERBOOK: ob})
        valid = result.dropna()
        assert len(valid) > 0
        assert all(v == pytest.approx(-1.0) for v in valid)

    def test_divide_by_zero_protection(self):
        """bid=0, ask=0 → NaN (不崩溃)"""
        timestamps = pd.date_range("2024-01-01", periods=10, freq="100ms", tz="UTC")
        data = {"timestamp": timestamps}
        for i in range(10):
            data[f"bid_price_{i}"] = [100.0] * 10
            data[f"bid_qty_{i}"] = [0.0] * 10
            data[f"ask_price_{i}"] = [101.0] * 10
            data[f"ask_qty_{i}"] = [0.0] * 10
        ob = pd.DataFrame(data)
        factor = OrderbookImbalance()
        # 不应崩溃
        result = factor.compute_single("BTC/USDT", {DataType.ORDERBOOK: ob})
        assert isinstance(result, pd.Series)
        # 所有值被 dropna, 返回空 series（因为全是 NaN）
        assert len(result) == 0

    def test_empty_data(self):
        """空 DataFrame → 返回空 Series"""
        factor = OrderbookImbalance()
        empty_ob = pd.DataFrame()
        result = factor.compute_single("BTC/USDT", {DataType.ORDERBOOK: empty_ob})
        assert isinstance(result, pd.Series)
        assert len(result) == 0
