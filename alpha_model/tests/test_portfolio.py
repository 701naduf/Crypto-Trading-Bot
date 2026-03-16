"""
portfolio/ 模块的单元测试

测试:
    - beta.py: 滚动 beta 估计
    - covariance.py: 协方差矩阵估计
    - constraints.py: cvxpy 约束生成
    - risk_budget.py: 波动率目标
    - constructor.py: 组合构建器
"""

import pytest
import numpy as np
import pandas as pd

from alpha_model.core.types import PortfolioConstraints
from alpha_model.portfolio.beta import rolling_beta
from alpha_model.portfolio.covariance import estimate_covariance, rolling_covariance
from alpha_model.portfolio.constraints import build_constraints
from alpha_model.portfolio.risk_budget import apply_vol_target
from alpha_model.portfolio.constructor import PortfolioConstructor


def _make_returns(n_rows=200, symbols=None, seed=42):
    """生成收益率面板"""
    if symbols is None:
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=n_rows, freq="1min", tz="UTC")
    data = rng.randn(n_rows, len(symbols)) * 0.001
    # BTC 作为市场，其他标的部分相关
    market = data[:, 0]
    for i in range(1, len(symbols)):
        data[:, i] = 0.6 * market + 0.4 * data[:, i]
    return pd.DataFrame(data, index=idx, columns=symbols)


def _make_price_panel(n_rows=200, symbols=None, seed=42):
    """生成价格面板"""
    returns = _make_returns(n_rows, symbols, seed)
    prices = 100 * np.exp(returns.cumsum())
    return prices


# ---------------------------------------------------------------------------
# beta.py
# ---------------------------------------------------------------------------

class TestRollingBeta:
    """滚动 beta"""

    def test_market_beta_is_one(self):
        """市场自身的 beta 恒为 1"""
        returns = _make_returns()
        betas = rolling_beta(returns, market_symbol="BTC/USDT", lookback=60)
        # BTC 列全为 1
        assert (betas["BTC/USDT"].dropna() == 1.0).all()

    def test_correlated_asset_positive_beta(self):
        """正相关标的的 beta > 0"""
        returns = _make_returns(n_rows=200)
        betas = rolling_beta(returns, lookback=60)
        # ETH 与 BTC 正相关
        eth_beta = betas["ETH/USDT"].dropna()
        assert eth_beta.mean() > 0

    def test_output_shape(self):
        """输出形状正确"""
        returns = _make_returns()
        betas = rolling_beta(returns, lookback=60)
        assert betas.shape == returns.shape
        assert list(betas.columns) == list(returns.columns)

    def test_invalid_market_symbol(self):
        """不存在的市场标的应报错"""
        returns = _make_returns()
        with pytest.raises(ValueError, match="market_symbol"):
            rolling_beta(returns, market_symbol="INVALID")


# ---------------------------------------------------------------------------
# covariance.py
# ---------------------------------------------------------------------------

class TestEstimateCovariance:
    """协方差矩阵估计"""

    def test_ledoit_wolf_psd(self):
        """Ledoit-Wolf 输出应为正半定矩阵"""
        returns = _make_returns(n_rows=200)
        cov = estimate_covariance(returns, lookback=100, method="ledoit_wolf")
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= -1e-10)

    def test_sample_covariance(self):
        """样本协方差应为方阵"""
        returns = _make_returns(n_rows=200)
        cov = estimate_covariance(returns, lookback=100, method="sample")
        n = len(returns.columns)
        assert cov.shape == (n, n)

    def test_symmetric(self):
        """协方差矩阵应对称"""
        returns = _make_returns(n_rows=200)
        cov = estimate_covariance(returns, lookback=100)
        np.testing.assert_allclose(cov, cov.T, atol=1e-10)

    def test_insufficient_data_raises(self):
        """数据不足应报错"""
        returns = _make_returns(n_rows=5)
        with pytest.raises(ValueError, match="有效数据"):
            estimate_covariance(returns, lookback=100, min_periods=20)

    def test_rolling_covariance(self):
        """滚动协方差应返回字典"""
        returns = _make_returns(n_rows=200)
        result = rolling_covariance(returns, lookback=60)
        assert isinstance(result, dict)
        assert len(result) > 0
        # 每个值应为方阵
        for cov in result.values():
            assert cov.shape[0] == cov.shape[1]

    def test_rolling_covariance_output_count(self):
        """[T6] 滚动协方差的输出数量应与可用窗口数一致"""
        returns = _make_returns(n_rows=200)
        result = rolling_covariance(returns, lookback=60, min_periods=20)
        assert len(result) > 100
        assert len(result) <= 140

    def test_rolling_covariance_matrix_shape(self):
        """[T6] 每个输出矩阵形状应为 N×N"""
        returns = _make_returns(n_rows=200)
        result = rolling_covariance(returns, lookback=60)
        n_symbols = len(returns.columns)
        for cov in result.values():
            assert cov.shape == (n_symbols, n_symbols)


# ---------------------------------------------------------------------------
# constraints.py
# ---------------------------------------------------------------------------

class TestBuildConstraints:
    """约束生成"""

    def test_basic_constraints(self):
        """基本约束生成"""
        import cvxpy as cp
        n = 5
        w = cp.Variable(n)
        config = PortfolioConstraints(
            max_weight=0.4, dollar_neutral=True, leverage_cap=2.0,
        )
        constraints = build_constraints(w, config)
        # 应至少有仓位上限 + dollar-neutral + 杠杆上限
        assert len(constraints) >= 3

    def test_beta_neutral_without_beta_raises(self):
        """beta-neutral 约束缺 beta 应报错"""
        import cvxpy as cp
        w = cp.Variable(3)
        config = PortfolioConstraints(beta_neutral=True)
        with pytest.raises(ValueError, match="beta"):
            build_constraints(w, config, beta=None)

    def test_beta_neutral_with_beta(self):
        """beta-neutral 约束正常生成"""
        import cvxpy as cp
        w = cp.Variable(3)
        config = PortfolioConstraints(beta_neutral=True)
        beta = np.array([1.0, 1.2, 0.8])
        constraints = build_constraints(w, config, beta=beta)
        assert len(constraints) >= 4  # 仓位 + dollar + beta + 杠杆


# ---------------------------------------------------------------------------
# constructor.py
# ---------------------------------------------------------------------------

class TestPortfolioConstructor:
    """组合构建器"""

    def test_basic_construct(self):
        """基本组合构建"""
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        n = 200
        idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        rng = np.random.RandomState(42)

        signal = pd.DataFrame(
            rng.randn(n, len(symbols)), index=idx, columns=symbols,
        )
        price_panel = _make_price_panel(n, symbols)

        constraints = PortfolioConstraints(
            max_weight=0.4, dollar_neutral=True,
            vol_target=None,  # 先不加 vol target
        )
        constructor = PortfolioConstructor(constraints)
        weights = constructor.construct(signal.iloc[100:], price_panel)

        assert isinstance(weights, pd.DataFrame)
        assert set(weights.columns) == set(symbols)

    def test_dollar_neutral_constraint(self):
        """dollar-neutral 约束: 权重之和接近 0"""
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        n = 200
        idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        rng = np.random.RandomState(42)

        signal = pd.DataFrame(
            rng.randn(n, len(symbols)), index=idx, columns=symbols,
        )
        price_panel = _make_price_panel(n, symbols)

        constraints = PortfolioConstraints(
            dollar_neutral=True, vol_target=None,
        )
        constructor = PortfolioConstructor(constraints)
        weights = constructor.construct(signal.iloc[100:], price_panel)

        # 每行权重之和应接近 0
        row_sums = weights.sum(axis=1).dropna()
        non_zero_rows = row_sums[weights.abs().sum(axis=1) > 1e-6]
        if len(non_zero_rows) > 0:
            assert non_zero_rows.abs().max() < 0.05

    def test_weight_cap_constraint(self):
        """仓位上限约束: |w_i| ≤ max_weight"""
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        n = 200
        idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        rng = np.random.RandomState(42)

        signal = pd.DataFrame(
            rng.randn(n, len(symbols)) * 2, index=idx, columns=symbols,
        )
        price_panel = _make_price_panel(n, symbols)

        constraints = PortfolioConstraints(
            max_weight=0.3, vol_target=None,
        )
        constructor = PortfolioConstructor(constraints)
        weights = constructor.construct(signal.iloc[100:], price_panel)

        # 所有权重绝对值应 ≤ 0.3 + 小误差
        assert weights.abs().max().max() < 0.3 + 0.01

    def test_infeasible_fallback(self):
        """[T3] 约束矛盾导致 infeasible 时应退化为安全权重"""
        symbols = ["BTC/USDT", "ETH/USDT"]
        n = 200
        idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        rng = np.random.RandomState(42)

        signal = pd.DataFrame(
            rng.randn(n, len(symbols)), index=idx, columns=symbols,
        )
        price_panel = _make_price_panel(n, symbols)

        # 矛盾约束: max_weight=0.01 + dollar_neutral + leverage_cap=0.01
        constraints = PortfolioConstraints(
            max_weight=0.01,
            dollar_neutral=True,
            leverage_cap=0.01,
            vol_target=None,
        )
        constructor = PortfolioConstructor(constraints)
        # 不应抛异常，应退化
        weights = constructor.construct(signal.iloc[100:], price_panel)
        assert isinstance(weights, pd.DataFrame)
        # 退化权重应为有限值
        assert np.all(np.isfinite(weights.fillna(0).values))

    def test_leverage_cap_constraint(self):
        """[T8] 杠杆上限约束: Σ|w_i| ≤ leverage_cap"""
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        n = 200
        idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        rng = np.random.RandomState(42)

        # 强信号，诱导优化器分配大权重
        signal = pd.DataFrame(
            rng.randn(n, len(symbols)) * 5, index=idx, columns=symbols,
        )
        price_panel = _make_price_panel(n, symbols)

        constraints = PortfolioConstraints(
            max_weight=0.4,
            leverage_cap=0.5,
            dollar_neutral=False,
            vol_target=None,
        )
        constructor = PortfolioConstructor(constraints)
        weights = constructor.construct(signal.iloc[100:], price_panel)

        # 每行 Σ|w_i| 应 ≤ 0.5 + 求解误差
        row_leverage = weights.abs().sum(axis=1)
        non_zero = row_leverage[row_leverage > 1e-6]
        if len(non_zero) > 0:
            assert non_zero.max() < 0.5 + 0.02


# ---------------------------------------------------------------------------
# risk_budget.py
# ---------------------------------------------------------------------------

class TestVolTargeting:
    """[T2] Vol targeting"""

    def test_high_vol_scales_down(self):
        """高波动率 → 缩小仓位"""
        symbols = ["BTC/USDT", "ETH/USDT"]
        n = 300
        idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        rng = np.random.RandomState(42)

        high_vol_returns = rng.randn(n, len(symbols)) * 0.05
        prices = 100 * np.exp(np.cumsum(high_vol_returns, axis=0))
        price_panel = pd.DataFrame(prices, index=idx, columns=symbols)

        weights = pd.DataFrame(0.5, index=idx, columns=symbols)

        adjusted = apply_vol_target(
            weights, price_panel,
            vol_target=0.15, lookback=60, leverage_cap=2.0,
        )
        # 高波动率下，调整后的权重绝对值应 ≤ 原始
        tail = adjusted.iloc[100:]
        assert tail.abs().mean().mean() < weights.iloc[100:].abs().mean().mean()

    def test_leverage_cap_respected(self):
        """vol targeting 后不应超过 leverage_cap"""
        symbols = ["BTC/USDT", "ETH/USDT"]
        n = 300
        idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        rng = np.random.RandomState(42)

        low_vol_returns = rng.randn(n, len(symbols)) * 0.0001
        prices = 100 * np.exp(np.cumsum(low_vol_returns, axis=0))
        price_panel = pd.DataFrame(prices, index=idx, columns=symbols)

        weights = pd.DataFrame(0.3, index=idx, columns=symbols)

        adjusted = apply_vol_target(
            weights, price_panel,
            vol_target=0.50, lookback=60, leverage_cap=1.0,
        )
        row_leverage = adjusted.abs().sum(axis=1).iloc[100:]
        assert row_leverage.max() <= 1.0 + 0.01

    def test_vol_target_invalid_raises(self):
        """vol_target <= 0 应报错"""
        with pytest.raises(ValueError):
            apply_vol_target(
                pd.DataFrame(), pd.DataFrame(),
                vol_target=-0.1,
            )
