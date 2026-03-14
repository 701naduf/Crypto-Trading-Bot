"""alignment 模块的单元测试

覆盖: grid, refresh_time, hayashi_yoshida
"""

import numpy as np
import pandas as pd
import pytest

from factor_research.alignment.grid import grid_align
from factor_research.alignment.refresh_time import refresh_time_align
from factor_research.alignment.hayashi_yoshida import (
    hy_correlation,
    hy_correlation_matrix,
    hy_covariance,
    hy_covariance_matrix,
)


# =========================================================================
# Grid Alignment 测试
# =========================================================================

class TestGridAlign:

    def test_basic(self):
        btc = pd.Series(
            [0.1, 0.2, 0.15],
            index=pd.to_datetime([
                "2024-01-01 10:00:00",
                "2024-01-01 10:00:01",
                "2024-01-01 10:00:02",
            ], utc=True),
        )
        eth = pd.Series(
            [0.3, -0.1],
            index=pd.to_datetime([
                "2024-01-01 10:00:00",
                "2024-01-01 10:00:02",
            ], utc=True),
        )

        panel = grid_align({"BTC": btc, "ETH": eth}, freq="1s")
        assert isinstance(panel, pd.DataFrame)
        assert "BTC" in panel.columns
        assert "ETH" in panel.columns
        assert len(panel) == 3  # 3 秒

    def test_forward_fill(self):
        """ETH 在 t=1s 时没有数据，应该用 t=0s 的值填充"""
        btc = pd.Series(
            [1.0, 2.0, 3.0],
            index=pd.to_datetime([
                "2024-01-01 10:00:00",
                "2024-01-01 10:00:01",
                "2024-01-01 10:00:02",
            ], utc=True),
        )
        eth = pd.Series(
            [10.0, 30.0],
            index=pd.to_datetime([
                "2024-01-01 10:00:00",
                "2024-01-01 10:00:02",
            ], utc=True),
        )

        panel = grid_align({"BTC": btc, "ETH": eth}, freq="1s")
        # ETH 在 t=1s 应该被 ffill 为 10.0
        assert panel.loc[pd.Timestamp("2024-01-01 10:00:01", tz="UTC"), "ETH"] == 10.0

    def test_empty_dict(self):
        panel = grid_align({})
        assert panel.empty

    def test_max_gap(self):
        """max_gap 限制填充间隔"""
        series = pd.Series(
            [1.0, 2.0],
            index=pd.to_datetime([
                "2024-01-01 10:00:00",
                "2024-01-01 10:00:10",
            ], utc=True),
        )
        panel = grid_align({"A": series}, freq="1s", max_gap=3)
        # t=0s 和 t=10s 有值，中间 gap > 3 的应该是 NaN
        assert not np.isnan(panel.iloc[0, 0])  # t=0s 有数据
        assert np.isnan(panel.iloc[5, 0])  # t=5s gap 太长


# =========================================================================
# Refresh Time Alignment 测试
# =========================================================================

class TestRefreshTimeAlign:

    def test_basic(self):
        btc = pd.Series(
            [1.0, 2.0, 3.0],
            index=pd.to_datetime(["10:00:00.1", "10:00:00.5", "10:00:01.0"]),
        )
        eth = pd.Series(
            [4.0, 5.0],
            index=pd.to_datetime(["10:00:00.3", "10:00:00.8"]),
        )

        panel = refresh_time_align({"BTC": btc, "ETH": eth})
        assert isinstance(panel, pd.DataFrame)
        assert "BTC" in panel.columns
        assert "ETH" in panel.columns
        assert len(panel) >= 1

    def test_single_symbol(self):
        btc = pd.Series([1.0, 2.0], index=pd.to_datetime(["10:00:00", "10:00:01"]))
        panel = refresh_time_align({"BTC": btc})
        assert len(panel) == 2

    def test_empty(self):
        panel = refresh_time_align({})
        assert panel.empty

    def test_no_overlap(self):
        """两个标的完全不重叠"""
        btc = pd.Series([1.0], index=pd.to_datetime(["10:00:00"]))
        eth = pd.Series([2.0], index=pd.to_datetime(["10:00:01"]))
        panel = refresh_time_align({"BTC": btc, "ETH": eth})
        # 在 10:00:01 时 BTC 有历史数据（10:00:00），ETH 有首次数据
        # 所以刷新时刻是 10:00:01（前提是两者都至少有一次更新）
        assert len(panel) >= 1


# =========================================================================
# Hayashi-Yoshida 测试
# =========================================================================

class TestHayashiYoshida:

    def test_covariance_identical(self):
        """同一序列的 HY 协方差 = 已实现方差"""
        prices = pd.Series(
            [100, 101, 100.5, 102, 101.5],
            index=pd.to_datetime([
                "10:00:00", "10:00:01", "10:00:02", "10:00:03", "10:00:04"
            ]),
        )
        var = hy_covariance(prices, prices)
        # 已实现方差 = sum(Δp²)
        diffs = np.diff(prices.values)
        expected = np.sum(diffs ** 2)
        assert var == pytest.approx(expected, abs=1e-10)

    def test_covariance_positive(self):
        """正相关的两个序列"""
        x = pd.Series(
            [100, 101, 102, 103],
            index=pd.to_datetime(["10:00:00", "10:00:01", "10:00:02", "10:00:03"]),
        )
        y = pd.Series(
            [50, 51, 52, 53],
            index=pd.to_datetime(["10:00:00.5", "10:00:01.5", "10:00:02.5", "10:00:03.5"]),
        )
        cov = hy_covariance(x, y)
        assert cov > 0

    def test_correlation_range(self):
        """相关系数应在 [-1, 1] 之间"""
        np.random.seed(42)
        x = pd.Series(
            100 + np.cumsum(np.random.randn(50) * 0.1),
            index=pd.date_range("2024-01-01 10:00:00", periods=50, freq="1s"),
        )
        y = pd.Series(
            50 + np.cumsum(np.random.randn(50) * 0.05),
            index=pd.date_range("2024-01-01 10:00:00.5", periods=50, freq="1s"),
        )
        corr = hy_correlation(x, y)
        assert -1 <= corr <= 1

    def test_correlation_perfect(self):
        """完美正相关"""
        x = pd.Series([100, 101, 102, 103, 104],
                       index=pd.date_range("10:00:00", periods=5, freq="1s"))
        y = pd.Series([100, 101, 102, 103, 104],
                       index=pd.date_range("10:00:00", periods=5, freq="1s"))
        corr = hy_correlation(x, y)
        assert corr == pytest.approx(1.0, abs=1e-6)

    def test_covariance_insufficient_data(self):
        x = pd.Series([100], index=pd.to_datetime(["10:00:00"]))
        y = pd.Series([50, 51], index=pd.to_datetime(["10:00:00", "10:00:01"]))
        assert np.isnan(hy_covariance(x, y))

    def test_covariance_matrix(self):
        x = pd.Series([100, 101, 102], index=pd.date_range("10:00:00", periods=3, freq="1s"))
        y = pd.Series([50, 51, 52], index=pd.date_range("10:00:00.5", periods=3, freq="1s"))
        cov = hy_covariance_matrix({"A": x, "B": y})
        assert cov.shape == (2, 2)
        # 对称性
        assert cov.loc["A", "B"] == pytest.approx(cov.loc["B", "A"])

    def test_correlation_matrix(self):
        x = pd.Series([100, 101, 102], index=pd.date_range("10:00:00", periods=3, freq="1s"))
        y = pd.Series([50, 51, 52], index=pd.date_range("10:00:00.5", periods=3, freq="1s"))
        corr = hy_correlation_matrix({"A": x, "B": y})
        # 对角线应为 1
        assert corr.loc["A", "A"] == pytest.approx(1.0)
        assert corr.loc["B", "B"] == pytest.approx(1.0)
