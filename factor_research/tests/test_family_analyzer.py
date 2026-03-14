"""evaluation/family_analyzer.py 的单元测试

覆盖:
    - sweep() 表结构、行数、指标范围
    - select() 过滤、top_n、空结果
    - robustness() 返回结构
    - detail() 返回完整报告
    - plot_sensitivity() / plot_heatmap() 不报错
    - 自定义 param_grid 覆盖
    - 无 _param_grid 因子的异常

Mock 策略: 使用合成数据的参数化 DummyFactor，不依赖真实 DataReader
"""

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")  # 非交互后端，避免弹窗

from factor_research.core.base import TimeSeriesFactor
from factor_research.core.types import (
    DataRequest,
    DataType,
    FactorMeta,
    FactorType,
)
from factor_research.evaluation.family_analyzer import FamilyAnalyzer


# =========================================================================
# 测试用因子
# =========================================================================

class _DummySingleParam(TimeSeriesFactor):
    """单参数测试因子"""

    _param_grid = {"window": [3, 5, 10]}

    def __init__(self, window: int = 3):
        self.window = window

    def meta(self):
        return FactorMeta(
            name=f"dummy_{self.window}",
            display_name=f"Dummy ({self.window})",
            factor_type=FactorType.TIME_SERIES,
            category="test",
            description="test",
            data_requirements=[DataRequest(DataType.OHLCV, timeframe="1m")],
            output_freq="1m",
            params={"window": self.window},
            family="dummy_family",
        )

    def compute_single(self, symbol, data):
        ohlcv = data[DataType.OHLCV]
        if ohlcv.empty or len(ohlcv) < self.window + 1:
            return pd.Series(dtype=float)
        close = ohlcv["close"]
        result = close.rolling(self.window).mean() / close - 1
        if "timestamp" in ohlcv.columns:
            result.index = pd.to_datetime(ohlcv["timestamp"], utc=True)
        return result.dropna()


class _DummyMultiParam(TimeSeriesFactor):
    """双参数测试因子"""

    _param_grid = {"window": [3, 5], "method": ["sma", "ema"]}

    def __init__(self, window: int = 3, method: str = "sma"):
        self.window = window
        self.method = method

    def meta(self):
        return FactorMeta(
            name=f"dummy_{self.method}_{self.window}",
            display_name=f"Dummy ({self.method}, {self.window})",
            factor_type=FactorType.TIME_SERIES,
            category="test",
            description="test",
            data_requirements=[DataRequest(DataType.OHLCV, timeframe="1m")],
            output_freq="1m",
            params={"window": self.window, "method": self.method},
            family="dummy_multi_family",
        )

    def compute_single(self, symbol, data):
        ohlcv = data[DataType.OHLCV]
        if ohlcv.empty or len(ohlcv) < self.window + 1:
            return pd.Series(dtype=float)
        close = ohlcv["close"]
        if self.method == "ema":
            result = close.ewm(span=self.window).mean() / close - 1
        else:
            result = close.rolling(self.window).mean() / close - 1
        if "timestamp" in ohlcv.columns:
            result.index = pd.to_datetime(ohlcv["timestamp"], utc=True)
        return result.dropna()


class _DummyNoGrid(TimeSeriesFactor):
    """无 _param_grid 的因子"""

    def meta(self):
        return FactorMeta(
            name="dummy_no_grid",
            display_name="No Grid",
            factor_type=FactorType.TIME_SERIES,
            category="test",
            description="test",
            data_requirements=[DataRequest(DataType.OHLCV, timeframe="1m")],
            output_freq="1m",
        )

    def compute_single(self, symbol, data):
        return pd.Series(dtype=float)


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def synth_data():
    """
    合成数据: 时序因子格式 {symbol: {DataType: DataFrame}}
    以及对应的价格面板
    """
    n = 200
    np.random.seed(42)
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")

    data = {}
    price_dict = {}
    for sym in symbols:
        prices = 100 + np.cumsum(np.random.randn(n) * 0.1)
        ohlcv = pd.DataFrame({
            "timestamp": timestamps,
            "open": prices,
            "high": prices + 0.1,
            "low": prices - 0.1,
            "close": prices,
            "volume": np.random.randint(100, 1000, n).astype(float),
        })
        data[sym] = {DataType.OHLCV: ohlcv}
        price_dict[sym] = pd.Series(prices, index=timestamps)

    price_panel = pd.DataFrame(price_dict)
    return data, price_panel


@pytest.fixture
def single_analyzer(synth_data):
    """单参数 FamilyAnalyzer"""
    data, price_panel = synth_data
    return FamilyAnalyzer(
        factor_class=_DummySingleParam,
        data=data,
        price_panel=price_panel,
        horizons=[1, 5],
    )


@pytest.fixture
def multi_analyzer(synth_data):
    """双参数 FamilyAnalyzer"""
    data, price_panel = synth_data
    return FamilyAnalyzer(
        factor_class=_DummyMultiParam,
        data=data,
        price_panel=price_panel,
        horizons=[1, 5],
    )


# =========================================================================
# sweep 测试
# =========================================================================

class TestSweep:

    def test_sweep_columns(self, single_analyzer):
        """T-1: sweep 返回表包含参数名 + horizon + 指标名"""
        df = single_analyzer.sweep()
        assert "window" in df.columns
        assert "horizon" in df.columns
        assert "ic_mean" in df.columns
        assert "ic_ir" in df.columns
        assert "turnover_autocorr" in df.columns
        assert "monotonicity" in df.columns

    def test_sweep_row_count(self, single_analyzer):
        """T-2: sweep 行数 = |param_grid 笛卡尔积| × |horizons|"""
        df = single_analyzer.sweep()
        # 3 window values × 2 horizons = 6
        assert len(df) == 3 * 2

    def test_sweep_ic_range(self, single_analyzer):
        """T-3: ic_mean 在 [-1, 1] 范围内"""
        df = single_analyzer.sweep()
        valid_ic = df["ic_mean"].dropna()
        if len(valid_ic) > 0:
            assert valid_ic.min() >= -1.0
            assert valid_ic.max() <= 1.0

    def test_sweep_custom_param_grid(self, synth_data):
        """T-4: 手动覆盖 param_grid 生效"""
        data, price_panel = synth_data
        custom_grid = {"window": [3, 7, 15]}
        fa = FamilyAnalyzer(
            factor_class=_DummySingleParam,
            data=data,
            price_panel=price_panel,
            param_grid=custom_grid,
            horizons=[1],
        )
        df = fa.sweep()
        assert len(df) == 3  # 3 windows × 1 horizon
        assert set(df["window"]) == {3, 7, 15}

    def test_sweep_multi_param(self, multi_analyzer):
        """双参数 sweep: 笛卡尔积 2×2=4 组 × 2 horizons = 8 行"""
        df = multi_analyzer.sweep()
        assert len(df) == 4 * 2

    def test_sweep_caches_result(self, single_analyzer):
        """sweep 结果被缓存"""
        df1 = single_analyzer.sweep()
        assert single_analyzer._sweep_df is not None
        assert len(single_analyzer._sweep_df) == len(df1)


# =========================================================================
# select 测试
# =========================================================================

class TestSelect:

    def test_select_filters(self, single_analyzer):
        """T-5: select 按条件过滤"""
        single_analyzer.sweep()
        # 设置一个极低阈值，应该返回所有行
        result = single_analyzer.select(min_ic_mean=-10, horizon=1)
        assert len(result) > 0

    def test_select_top_n(self, single_analyzer):
        """T-6: top_n 限制返回行数"""
        single_analyzer.sweep()
        result = single_analyzer.select(top_n=2, horizon=1)
        assert len(result) <= 2

    def test_select_empty_result(self, single_analyzer):
        """T-7: 阈值过高 → 返回空 DataFrame"""
        single_analyzer.sweep()
        result = single_analyzer.select(min_ic_ir=999)
        assert len(result) == 0

    def test_select_before_sweep_raises(self, single_analyzer):
        """未运行 sweep 就 select → ValueError"""
        with pytest.raises(ValueError, match="sweep"):
            single_analyzer.select()


# =========================================================================
# robustness 测试
# =========================================================================

class TestRobustness:

    def test_robustness_structure(self, single_analyzer):
        """T-8: robustness 返回包含 robustness_score 列"""
        single_analyzer.sweep()
        result = single_analyzer.robustness(metric="ic_ir", horizon=1)
        assert isinstance(result, pd.DataFrame)
        assert "robustness_score" in result.columns

    def test_robustness_multi_param(self, multi_analyzer):
        """双参数的 robustness 也有 robustness_score"""
        multi_analyzer.sweep()
        result = multi_analyzer.robustness(metric="ic_ir", horizon=1)
        assert "robustness_score" in result.columns


# =========================================================================
# detail 测试
# =========================================================================

class TestDetail:

    def test_detail_report_keys(self, single_analyzer):
        """T-9: detail 返回完整报告，包含 6 个维度"""
        single_analyzer.sweep()  # 不是必须，但先 sweep 确保数据可计算
        report = single_analyzer.detail(window=5)
        assert isinstance(report, dict)
        for key in ["ic", "quantile", "tail", "stability", "nonlinear", "turnover"]:
            assert key in report, f"报告缺少 '{key}' 键"


# =========================================================================
# plot 测试
# =========================================================================

class TestPlots:

    def test_plot_sensitivity(self, single_analyzer):
        """T-10: plot_sensitivity 返回 Figure 不报错"""
        import matplotlib.pyplot as plt
        single_analyzer.sweep()
        fig = single_analyzer.plot_sensitivity(metric="ic_mean")
        assert fig is not None
        plt.close(fig)

    def test_plot_heatmap_dual_param(self, multi_analyzer):
        """T-11: 双参数 plot_heatmap 返回 Figure"""
        import matplotlib.pyplot as plt
        multi_analyzer.sweep()
        fig = multi_analyzer.plot_heatmap(metric="ic_ir", horizon=1)
        assert fig is not None
        plt.close(fig)

    def test_plot_heatmap_single_param(self, single_analyzer):
        """T-12: 单参数 plot_heatmap 退化为柱状图，不报错"""
        import matplotlib.pyplot as plt
        single_analyzer.sweep()
        fig = single_analyzer.plot_heatmap(metric="ic_ir", horizon=1)
        assert fig is not None
        plt.close(fig)

    def test_plot_before_sweep_raises(self, single_analyzer):
        """未运行 sweep 就 plot → ValueError"""
        with pytest.raises(ValueError, match="sweep"):
            single_analyzer.plot_sensitivity()


# =========================================================================
# 边界场景
# =========================================================================

class TestEdgeCases:

    def test_no_param_grid_raises(self, synth_data):
        """T-13: 无 _param_grid 且未传 param_grid → ValueError"""
        data, price_panel = synth_data
        with pytest.raises(ValueError, match="param_grid"):
            FamilyAnalyzer(
                factor_class=_DummyNoGrid,
                data=data,
                price_panel=price_panel,
            )

    def test_no_param_grid_with_manual_grid(self, synth_data):
        """无 _param_grid 但手动传入 param_grid → 正常工作"""
        data, price_panel = synth_data
        # _DummyNoGrid 没有 __init__ 参数，所以我们用 _DummySingleParam
        fa = FamilyAnalyzer(
            factor_class=_DummySingleParam,
            data=data,
            price_panel=price_panel,
            param_grid={"window": [3]},
            horizons=[1],
        )
        df = fa.sweep()
        assert len(df) == 1
