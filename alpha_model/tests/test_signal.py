"""
signal/ 模块的单元测试

测试:
    - generator.py: 信号生成
    - smoother.py: 信号平滑
"""

import pytest
import numpy as np
import pandas as pd

from alpha_model.signal.generator import generate_signal
from alpha_model.signal.smoother import ema_smooth


def _make_predictions(n_rows=100, symbols=None, seed=42):
    """生成测试用预测值面板"""
    if symbols is None:
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=n_rows, freq="1min", tz="UTC")
    data = rng.randn(n_rows, len(symbols))
    return pd.DataFrame(data, index=idx, columns=symbols)


# ---------------------------------------------------------------------------
# generator.py
# ---------------------------------------------------------------------------

class TestGenerateSignal:
    """信号生成"""

    def test_zscore_output_shape(self):
        """zscore 模式输出形状正确"""
        preds = _make_predictions()
        signal = generate_signal(preds, method="cross_sectional_zscore")
        assert signal.shape == preds.shape

    def test_zscore_row_mean_zero(self):
        """zscore 后每行均值接近 0"""
        preds = _make_predictions(
            n_rows=50, symbols=["A", "B", "C", "D", "E"],
        )
        signal = generate_signal(preds, method="cross_sectional_zscore")
        row_means = signal.mean(axis=1).dropna()
        assert row_means.abs().max() < 1e-10

    def test_zscore_row_std_one(self):
        """zscore 后每行标准差接近 1"""
        preds = _make_predictions(
            n_rows=50, symbols=["A", "B", "C", "D", "E"],
        )
        signal = generate_signal(preds, method="cross_sectional_zscore")
        row_stds = signal.std(axis=1).dropna()
        assert (row_stds - 1.0).abs().max() < 1e-10

    def test_rank_output_range(self):
        """rank 模式输出值域 [0, 1]"""
        preds = _make_predictions()
        signal = generate_signal(preds, method="cross_sectional_rank")
        assert signal.min().min() >= 0
        assert signal.max().max() <= 1

    def test_no_clip_by_default(self):
        """默认不截断"""
        preds = _make_predictions(symbols=["A", "B", "C", "D", "E"])
        # 注入极端值（相对于 5 个标的的截面，100 应该足够极端）
        preds.iloc[50, 0] = 100.0
        signal = generate_signal(preds, method="cross_sectional_zscore")
        # 极端值经过 zscore 后应仍然很大（5 标的截面 zscore）
        assert signal.iloc[50, 0] > 1.5

    def test_clip_sigma(self):
        """指定 clip_sigma 时应截断"""
        preds = _make_predictions()
        preds.iloc[50, 0] = 100.0
        signal = generate_signal(
            preds, method="cross_sectional_zscore", clip_sigma=3.0,
        )
        assert signal.iloc[50, 0] <= 3.0

    def test_empty_input(self):
        """空输入返回空 DataFrame"""
        empty = pd.DataFrame()
        result = generate_signal(empty)
        assert result.empty

    def test_invalid_method(self):
        """不支持的方法应报错"""
        preds = _make_predictions()
        with pytest.raises(ValueError, match="不支持的 method"):
            generate_signal(preds, method="invalid")


# ---------------------------------------------------------------------------
# smoother.py
# ---------------------------------------------------------------------------

class TestEmaSmooth:
    """EMA 平滑"""

    def test_output_shape(self):
        """输出形状正确"""
        signal = _make_predictions()
        smoothed = ema_smooth(signal, halflife=5)
        assert smoothed.shape == signal.shape

    def test_smoothing_reduces_variance(self):
        """平滑应降低方差"""
        signal = _make_predictions(n_rows=200)
        smoothed = ema_smooth(signal, halflife=10)
        # 平滑后每列的方差应小于原始
        for col in signal.columns:
            assert smoothed[col].var() < signal[col].var()

    def test_larger_halflife_more_smooth(self):
        """halflife 越大越平滑"""
        signal = _make_predictions(n_rows=200)
        s5 = ema_smooth(signal, halflife=5)
        s20 = ema_smooth(signal, halflife=20)
        # halflife=20 的方差应更小
        assert s20.var().mean() < s5.var().mean()

    def test_invalid_halflife(self):
        """halflife < 1 应报错"""
        signal = _make_predictions()
        with pytest.raises(ValueError, match="halflife"):
            ema_smooth(signal, halflife=0)
