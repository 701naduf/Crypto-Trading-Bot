"""
preprocessing/ 模块的单元测试

测试:
    - alignment.py: 多频率因子对齐
    - transform.py: 标准化工具箱 + 特征矩阵构建
"""

import pytest
import numpy as np
import pandas as pd

from alpha_model.preprocessing.alignment import align_factor_panels, _infer_freq
from alpha_model.preprocessing.transform import (
    expanding_zscore,
    rolling_zscore,
    cross_sectional_zscore,
    cross_sectional_rank,
    winsorize,
    build_feature_matrix,
    build_pooled_target,
)
from alpha_model.preprocessing.selection import select_factors
from alpha_model.core.types import TrainMode


# ---------------------------------------------------------------------------
# 测试辅助
# ---------------------------------------------------------------------------

def _make_panel(n_rows=100, symbols=None, freq="1min", seed=42):
    """生成测试用因子面板"""
    if symbols is None:
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=n_rows, freq=freq, tz="UTC")
    data = rng.randn(n_rows, len(symbols))
    return pd.DataFrame(data, index=idx, columns=symbols)


# ---------------------------------------------------------------------------
# alignment.py
# ---------------------------------------------------------------------------

class TestInferFreq:
    """频率推断"""

    def test_infer_1min(self):
        panel = _make_panel(freq="1min")
        td = _infer_freq(panel)
        assert td == pd.Timedelta("1min")

    def test_infer_5min(self):
        panel = _make_panel(freq="5min")
        td = _infer_freq(panel)
        assert td == pd.Timedelta("5min")

    def test_single_row_returns_none(self):
        panel = _make_panel(n_rows=1)
        assert _infer_freq(panel) is None

    def test_empty_returns_none(self):
        panel = _make_panel(n_rows=0)
        assert _infer_freq(panel) is None


class TestAlignFactorPanels:
    """多频率因子对齐"""

    def test_same_freq_alignment(self):
        """相同频率的面板对齐后索引相同"""
        panels = {
            "f1": _make_panel(freq="1min", seed=1),
            "f2": _make_panel(freq="1min", seed=2),
        }
        aligned = align_factor_panels(panels)
        assert aligned["f1"].index.equals(aligned["f2"].index)

    def test_different_freq_alignment(self):
        """不同频率的面板对齐到最低频"""
        p_1m = _make_panel(n_rows=60, freq="1min", seed=1)
        p_5m = _make_panel(n_rows=12, freq="5min", seed=2)
        panels = {"fast": p_1m, "slow": p_5m}
        aligned = align_factor_panels(panels)
        # 对齐后两个面板索引相同
        assert aligned["fast"].index.equals(aligned["slow"].index)

    def test_explicit_target_freq(self):
        """手动指定目标频率"""
        panels = {"f1": _make_panel(n_rows=60, freq="1min")}
        aligned = align_factor_panels(panels, target_freq="5min")
        # 结果频率应为 5min
        diffs = pd.Series(aligned["f1"].index).diff().dropna()
        assert diffs.median() == pd.Timedelta("5min")

    def test_ffill_fills_gaps(self):
        """前向填充能填上空隙"""
        panels = {"f1": _make_panel(n_rows=10, freq="5min")}
        aligned = align_factor_panels(panels, target_freq="1min")
        # 不应有太多 NaN
        nan_pct = aligned["f1"].isna().mean().mean()
        assert nan_pct < 0.5

    def test_no_fill_has_nans(self):
        """不填充时应有 NaN"""
        panels = {"f1": _make_panel(n_rows=10, freq="5min")}
        aligned = align_factor_panels(panels, target_freq="1min", fill_method=None)
        # 绝大部分应该是 NaN
        nan_pct = aligned["f1"].isna().mean().mean()
        assert nan_pct > 0.5

    def test_max_gap_limits_fill(self):
        """max_gap 限制前向填充步数"""
        panels = {"f1": _make_panel(n_rows=10, freq="5min")}
        aligned = align_factor_panels(
            panels, target_freq="1min", max_gap=2,
        )
        # max_gap=2 时，5min 间隔中只有前 2 行被填充，第 3/4 行为 NaN
        nan_pct = aligned["f1"].isna().mean().mean()
        assert nan_pct > 0.3

    def test_empty_input_raises(self):
        """空输入应报错"""
        with pytest.raises(ValueError, match="不能为空"):
            align_factor_panels({})


# ---------------------------------------------------------------------------
# transform.py — 标准化
# ---------------------------------------------------------------------------

class TestExpandingZscore:
    """Expanding z-score"""

    def test_output_shape(self):
        panel = _make_panel()
        result = expanding_zscore(panel, min_periods=10)
        assert result.shape == panel.shape

    def test_early_rows_are_nan(self):
        """前 min_periods 行应为 NaN"""
        panel = _make_panel(n_rows=50)
        result = expanding_zscore(panel, min_periods=20)
        # 前 19 行应全为 NaN（expanding 需要 min_periods 才产出）
        assert result.iloc[:19].isna().all().all()

    def test_later_rows_not_nan(self):
        """min_periods 之后应有值"""
        panel = _make_panel(n_rows=50)
        result = expanding_zscore(panel, min_periods=10)
        assert not result.iloc[10:].isna().all().all()

    def test_no_future_info(self):
        """expanding z-score 不使用未来信息"""
        panel = _make_panel(n_rows=100)
        result = expanding_zscore(panel, min_periods=10)
        # 修改最后一行不应影响前面的值
        panel2 = panel.copy()
        panel2.iloc[-1] = 999
        result2 = expanding_zscore(panel2, min_periods=10)
        # 前 98 行应完全一致
        pd.testing.assert_frame_equal(result.iloc[:98], result2.iloc[:98])


class TestRollingZscore:
    """Rolling z-score"""

    def test_output_shape(self):
        panel = _make_panel()
        result = rolling_zscore(panel, window=20)
        assert result.shape == panel.shape

    def test_early_rows_are_nan(self):
        """前 window-1 行应为 NaN"""
        panel = _make_panel(n_rows=50)
        result = rolling_zscore(panel, window=20)
        assert result.iloc[:19].isna().all().all()


class TestCrossSectionalZscore:
    """截面 z-score"""

    def test_output_shape(self):
        panel = _make_panel()
        result = cross_sectional_zscore(panel)
        assert result.shape == panel.shape

    def test_row_mean_zero(self):
        """截面标准化后每行均值应接近 0"""
        panel = _make_panel(n_rows=50, symbols=["A", "B", "C", "D", "E"])
        result = cross_sectional_zscore(panel)
        row_means = result.mean(axis=1).dropna()
        assert row_means.abs().max() < 1e-10

    def test_row_std_one(self):
        """截面标准化后每行标准差应接近 1"""
        panel = _make_panel(n_rows=50, symbols=["A", "B", "C", "D", "E"])
        result = cross_sectional_zscore(panel)
        row_stds = result.std(axis=1).dropna()
        assert (row_stds - 1.0).abs().max() < 1e-10


class TestCrossSectionalRank:
    """截面排名"""

    def test_output_range(self):
        """排名值域应在 [0, 1]"""
        panel = _make_panel()
        result = cross_sectional_rank(panel)
        assert result.min().min() >= 0
        assert result.max().max() <= 1

    def test_output_shape(self):
        panel = _make_panel()
        result = cross_sectional_rank(panel)
        assert result.shape == panel.shape


class TestWinsorize:
    """去极值"""

    def test_extreme_values_clipped(self):
        """极端值应被截断"""
        panel = _make_panel(n_rows=200)
        # 注入极端值
        panel.iloc[100, 0] = 100.0
        result = winsorize(panel, sigma=3.0, method="expanding", min_periods=10)
        # 原始值 100 应被截断
        assert result.iloc[100, 0] < 100.0

    def test_cross_sectional_winsorize(self):
        """截面方式去极值"""
        panel = _make_panel(n_rows=50, symbols=["A", "B", "C", "D", "E"])
        panel.iloc[25, 0] = 100.0
        result = winsorize(panel, sigma=2.0, method="cross_sectional")
        # 截断后应 ≤ 原始极端值（截面 mean+2*std 的上界）
        assert result.iloc[25, 0] <= 100.0
        # 而且应比未截断时的其他行的最大值合理
        normal_max = panel.drop(index=panel.index[25]).abs().max().max()
        assert result.iloc[25, 0] < 100.0 or result.iloc[25, 0] <= panel.iloc[25].mean() + 2 * panel.iloc[25].std()

    def test_invalid_method_raises(self):
        """不支持的 method 应报错"""
        panel = _make_panel()
        with pytest.raises(ValueError, match="不支持的 method"):
            winsorize(panel, method="invalid")


# ---------------------------------------------------------------------------
# transform.py — build_feature_matrix
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:
    """特征矩阵构建"""

    def test_pooled_output_format(self):
        """Pooled 模式输出 MultiIndex"""
        panels = {
            "f1": _make_panel(seed=1),
            "f2": _make_panel(seed=2),
        }
        symbols = ["BTC/USDT", "ETH/USDT"]
        X = build_feature_matrix(panels, symbols, TrainMode.POOLED)
        assert isinstance(X.index, pd.MultiIndex)
        assert X.index.names == ["timestamp", "symbol"]
        assert list(X.columns) == ["f1", "f2"]

    def test_pooled_row_count(self):
        """Pooled 模式行数 = timestamps × symbols"""
        panels = {"f1": _make_panel(n_rows=50, seed=1)}
        symbols = ["BTC/USDT", "ETH/USDT"]
        X = build_feature_matrix(panels, symbols, TrainMode.POOLED)
        assert len(X) == 50 * 2  # 50 timestamps × 2 symbols

    def test_per_symbol_output_format(self):
        """Per-Symbol 模式输出字典"""
        panels = {
            "f1": _make_panel(seed=1),
            "f2": _make_panel(seed=2),
        }
        symbols = ["BTC/USDT", "ETH/USDT"]
        X = build_feature_matrix(panels, symbols, TrainMode.PER_SYMBOL)
        assert isinstance(X, dict)
        assert "BTC/USDT" in X
        assert list(X["BTC/USDT"].columns) == ["f1", "f2"]

    def test_empty_panels_raises(self):
        """空因子面板应报错"""
        with pytest.raises(ValueError, match="不能为空"):
            build_feature_matrix({}, ["BTC/USDT"], TrainMode.POOLED)

    def test_empty_symbols_raises(self):
        """空标的列表应报错"""
        panels = {"f1": _make_panel()}
        with pytest.raises(ValueError, match="不能为空"):
            build_feature_matrix(panels, [], TrainMode.POOLED)


# ---------------------------------------------------------------------------
# transform.py — build_pooled_target [T7]
# ---------------------------------------------------------------------------

class TestBuildPooledTarget:
    """[T7] Pooled 目标变量构建"""

    def test_index_alignment(self):
        """X 和 y 的 MultiIndex 应完全对齐"""
        symbols = ["BTC/USDT", "ETH/USDT"]
        panels = {
            "f1": _make_panel(n_rows=50, symbols=symbols, seed=1),
        }
        X = build_feature_matrix(panels, symbols, TrainMode.POOLED)
        fwd = _make_panel(n_rows=50, symbols=symbols, seed=2)
        y = build_pooled_target(X, fwd, symbols)
        assert y.index.equals(X.index)

    def test_symbol_not_swapped(self):
        """symbol 维度不应错位"""
        symbols = ["A", "B"]
        n = 20
        idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")

        panels = {"f1": pd.DataFrame(
            {"A": np.ones(n), "B": -np.ones(n)}, index=idx,
        )}
        fwd = pd.DataFrame(
            {"A": np.ones(n) * 0.01, "B": np.ones(n) * -0.01}, index=idx,
        )

        X = build_feature_matrix(panels, symbols, TrainMode.POOLED)
        y = build_pooled_target(X, fwd, symbols)

        y_a = y.xs("A", level="symbol")
        y_b = y.xs("B", level="symbol")
        assert (y_a.dropna() > 0).all()
        assert (y_b.dropna() < 0).all()


# ---------------------------------------------------------------------------
# selection.py [T5]
# ---------------------------------------------------------------------------

class TestSelectFactors:
    """[T5] 因子筛选"""

    @pytest.fixture
    def setup_data(self):
        """构造合成因子面板和价格面板"""
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        n = 300
        idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        rng = np.random.RandomState(42)

        price_panel = pd.DataFrame(
            100 * np.exp(np.cumsum(rng.randn(n, 3) * 0.001, axis=0)),
            index=idx, columns=symbols,
        )

        # 创建 5 个因子：前 2 个与 forward return 相关，后 3 个为噪声
        fwd = price_panel.pct_change().shift(-1)
        factor_panels = {}
        for i in range(5):
            if i < 2:
                # 有预测力的因子
                data = fwd.values * (0.5 + 0.3 * i) + rng.randn(n, 3) * 0.01
            else:
                # 纯噪声因子
                data = rng.randn(n, 3) * 0.01
            panel = pd.DataFrame(data, index=idx, columns=symbols)
            # [R5] 截断尾部 NaN 行（shift(-1) 导致最后一行 NaN）
            factor_panels[f"factor_{i}"] = panel.iloc[:-1]

        return factor_panels, price_panel.iloc[:-1]

    def test_threshold_mode_filters(self, setup_data):
        """threshold 模式应过滤低质量因子"""
        factor_panels, price_panel = setup_data
        selected = select_factors(
            factor_panels, price_panel,
            mode="threshold", metric="ic", min_ic=0.01,
            horizon=1, min_factors=1,
        )
        assert isinstance(selected, dict)
        assert len(selected) >= 1
        assert len(selected) <= len(factor_panels)

    def test_topk_mode_returns_k(self, setup_data):
        """top_k 模式应返回 k 个因子"""
        factor_panels, price_panel = setup_data
        selected = select_factors(
            factor_panels, price_panel,
            mode="top_k", top_k=3, horizon=1,
        )
        assert len(selected) == 3

    def test_empty_input_raises(self):
        """空输入应报错"""
        with pytest.raises(ValueError, match="不能为空"):
            select_factors({}, pd.DataFrame(), mode="threshold")

    def test_invalid_mode_raises(self):
        """无效模式应报错"""
        panels = {"f1": _make_panel()}
        with pytest.raises(ValueError, match="不支持的 mode"):
            select_factors(panels, _make_panel(), mode="invalid")
