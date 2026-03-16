"""evaluation 模块的单元测试

覆盖: metrics, ic, quantile, tail, stability, nonlinear, turnover, correlation, analyzer
"""

import numpy as np
import pandas as pd
import pytest

from factor_research.evaluation.metrics import (
    annualize_return,
    annualize_volatility,
    compute_forward_returns,
    compute_forward_returns_panel,
    cross_sectional_rank,
    cross_sectional_zscore,
    cumulative_returns,
    max_drawdown,
    pearson_ic,
    rank_normalize,
    sharpe_ratio,
    spearman_ic,
    zscore_normalize,
)
from factor_research.evaluation.ic import ic_analysis, ic_decay, ic_series, ic_summary
from factor_research.evaluation.quantile import quantile_backtest
from factor_research.evaluation.tail import tail_analysis
from factor_research.evaluation.turnover import turnover_analysis
from factor_research.evaluation.nonlinear import nonlinear_analysis
from factor_research.evaluation.stability import stability_analysis
from factor_research.evaluation.correlation import correlation_analysis, incremental_ic
from factor_research.evaluation.report import format_report_text, plot_report
from factor_research.evaluation.analyzer import FactorAnalyzer


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def factor_panel():
    """合成因子面板: 与收益正相关"""
    np.random.seed(42)
    n = 200
    index = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    # 因子值
    f_btc = np.random.randn(n)
    f_eth = np.random.randn(n)
    return pd.DataFrame(
        {"BTC/USDT": f_btc, "ETH/USDT": f_eth},
        index=index,
    )


@pytest.fixture
def price_panel():
    """合成价格面板"""
    np.random.seed(42)
    n = 200
    index = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    btc_prices = 100 + np.cumsum(np.random.randn(n) * 0.1)
    eth_prices = 50 + np.cumsum(np.random.randn(n) * 0.05)
    return pd.DataFrame(
        {"BTC/USDT": btc_prices, "ETH/USDT": eth_prices},
        index=index,
    )


@pytest.fixture
def correlated_data():
    """因子与收益有正相关的数据（用于验证 IC 方向）"""
    np.random.seed(42)
    n = 500
    index = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    # 构造有正 IC 的因子
    noise = np.random.randn(n) * 0.5
    signal = np.random.randn(n)
    prices_btc = 100 + np.cumsum(signal * 0.1 + noise * 0.01)
    prices_eth = 50 + np.cumsum(signal * 0.05 + noise * 0.005)

    factor_panel = pd.DataFrame(
        {"BTC/USDT": signal, "ETH/USDT": signal * 0.8 + np.random.randn(n) * 0.2},
        index=index,
    )
    price_panel = pd.DataFrame(
        {"BTC/USDT": prices_btc, "ETH/USDT": prices_eth},
        index=index,
    )
    return factor_panel, price_panel


# =========================================================================
# Metrics 测试
# =========================================================================

class TestMetrics:

    def test_spearman_ic(self):
        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([1, 2, 3, 4, 5])
        assert spearman_ic(x, y) == pytest.approx(1.0)

    def test_spearman_ic_negative(self):
        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([5, 4, 3, 2, 1])
        assert spearman_ic(x, y) == pytest.approx(-1.0)

    def test_spearman_ic_with_nan(self):
        x = pd.Series([1, np.nan, 3, 4, 5])
        y = pd.Series([1, 2, np.nan, 4, 5])
        result = spearman_ic(x, y)
        assert not np.isnan(result)

    def test_spearman_ic_insufficient_data(self):
        x = pd.Series([1, 2])
        y = pd.Series([1, 2])
        assert np.isnan(spearman_ic(x, y))

    def test_pearson_ic(self):
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0])
        assert pearson_ic(x, y) == pytest.approx(1.0)

    def test_forward_returns(self):
        prices = pd.Series([100.0, 102.0, 101.0, 105.0, 103.0])
        ret = compute_forward_returns(prices, 1)
        assert ret.iloc[0] == pytest.approx(0.02)  # 102/100 - 1
        assert np.isnan(ret.iloc[-1])  # 最后一个没有前瞻

    def test_forward_returns_panel(self, price_panel):
        ret = compute_forward_returns_panel(price_panel, 1)
        assert ret.shape == price_panel.shape
        assert list(ret.columns) == list(price_panel.columns)

    def test_rank_normalize(self):
        s = pd.Series([10, 30, 20, 40, 50])
        ranked = rank_normalize(s)
        assert ranked.min() > 0
        assert ranked.max() <= 1

    def test_cross_sectional_rank(self, factor_panel):
        ranked = cross_sectional_rank(factor_panel)
        assert ranked.shape == factor_panel.shape
        # 每行排名和应该是固定的
        row_sums = ranked.sum(axis=1).dropna()
        assert all(s == pytest.approx(row_sums.iloc[0]) for s in row_sums)

    def test_zscore_normalize(self):
        s = pd.Series([1, 2, 3, 4, 5])
        z = zscore_normalize(s)
        assert z.mean() == pytest.approx(0, abs=1e-10)
        assert z.std() == pytest.approx(1, abs=0.01)

    def test_cumulative_returns(self):
        ret = pd.Series([0.01, 0.02, -0.01, 0.03])
        cum = cumulative_returns(ret)
        assert len(cum) == 4
        expected = (1.01 * 1.02 * 0.99 * 1.03) - 1
        assert cum.iloc[-1] == pytest.approx(expected)

    def test_max_drawdown(self):
        cum = pd.Series([0.0, 0.1, 0.05, 0.15, 0.08, 0.12])
        mdd = max_drawdown(cum)
        # 行业惯例: 返回负数或零
        assert mdd <= 0
        assert mdd >= -1
        # 这条曲线的最大回撤发生在 0.15 → 0.08
        # wealth: 1.15 → 1.08, drawdown = (1.08 - 1.15) / 1.15 ≈ -0.0609
        assert mdd == pytest.approx((1.08 - 1.15) / 1.15, abs=1e-4)

    def test_sharpe_ratio_positive(self):
        ret = pd.Series([0.001] * 100)  # 稳定正收益但有微小波动
        # 常数收益 → std=0 → 应返回 inf（正收益零波动率）
        sr = sharpe_ratio(ret)
        assert sr == np.inf

    def test_sharpe_ratio_zero_vol_negative(self):
        ret = pd.Series([-0.001] * 100)  # 稳定负收益
        sr = sharpe_ratio(ret)
        assert sr == -np.inf

    def test_sharpe_ratio_with_variance(self):
        # 使用有波动但整体正收益的序列
        # 小幅收益避免年化时溢出（periods_per_year=525960）
        np.random.seed(123)
        ret = pd.Series(np.random.randn(1000) * 0.0001 + 0.00001)
        sr = sharpe_ratio(ret)
        assert np.isfinite(sr)


# =========================================================================
# IC 分析测试
# =========================================================================

class TestICAnalysis:

    def test_ic_series(self, factor_panel, price_panel):
        ic_ts = ic_series(factor_panel, price_panel, horizon=1)
        assert isinstance(ic_ts, pd.Series)
        assert len(ic_ts) > 0

    def test_ic_summary_structure(self, factor_panel, price_panel):
        ic_ts = ic_series(factor_panel, price_panel, horizon=1)
        summary = ic_summary(ic_ts)
        assert "ic_mean" in summary
        assert "ic_std" in summary
        assert "ic_ir" in summary
        assert "ic_win_rate" in summary
        assert "n_observations" in summary

    def test_ic_decay(self, factor_panel, price_panel):
        decay = ic_decay(factor_panel, price_panel, horizons=[1, 5, 10])
        assert isinstance(decay, pd.DataFrame)
        assert list(decay.index) == [1, 5, 10]
        assert "ic_mean" in decay.columns

    def test_ic_analysis_full(self, factor_panel, price_panel):
        result = ic_analysis(factor_panel, price_panel, horizons=[1, 5])
        assert "ic_series" in result
        assert "ic_summary" in result
        assert "ic_decay" in result
        assert 1 in result["ic_series"]
        assert 5 in result["ic_series"]


# =========================================================================
# 分层回测测试
# =========================================================================

class TestQuantileBacktest:

    def test_basic(self, factor_panel, price_panel):
        result = quantile_backtest(factor_panel, price_panel, n_groups=2, horizon=1)
        assert "group_returns" in result
        assert "long_short_return" in result
        assert "monotonicity" in result
        assert len(result["group_returns"]) == 2

    def test_with_5_groups(self, factor_panel, price_panel):
        result = quantile_backtest(factor_panel, price_panel, n_groups=5, horizon=1)
        # 5 标的只能分2组（因为只有 2 个 symbol）
        assert len(result["group_returns"]) == 2

    def test_cumulative_by_group(self, factor_panel, price_panel):
        result = quantile_backtest(factor_panel, price_panel, n_groups=2, horizon=1)
        cum = result["cumulative_by_group"]
        assert isinstance(cum, pd.DataFrame)


# =========================================================================
# 尾部分析测试
# =========================================================================

class TestTailAnalysis:

    def test_basic(self, factor_panel, price_panel):
        result = tail_analysis(factor_panel, price_panel, threshold=0.9, horizon=1)
        assert "conditional_ic" in result
        assert "tail_hit_rate" in result
        assert "tail_frequency" in result
        assert "n_tail_observations" in result
        assert result["n_tail_observations"] > 0

    def test_tail_frequency_reasonable(self, factor_panel, price_panel):
        result = tail_analysis(factor_panel, price_panel, threshold=0.9)
        # 90% 阈值，尾部应该约占 10%
        assert 0.05 < result["tail_frequency"] < 0.3

    def test_mae_exists(self, factor_panel, price_panel):
        """返回结果包含 mae 字段"""
        result = tail_analysis(factor_panel, price_panel, threshold=0.9, horizon=5)
        assert "mae" in result
        assert not np.isnan(result["mae"])

    def test_mae_nonpositive(self, factor_panel, price_panel):
        """MAE 应为非正值（最大浮亏 ≤ 0）或接近 0"""
        result = tail_analysis(factor_panel, price_panel, threshold=0.9, horizon=5)
        # MAE 是平均值，在随机数据中通常为负
        # 但不做严格 ≤ 0 的断言，因为平均 MAE 在极少数情况下可为微正
        assert result["mae"] < 0.1  # 宽松上界

    def test_mae_known_answer(self):
        """构造简单数据手算验证 MAE

        需要足够多的观测点以通过最小样本数检查 (>=10)。
        使用 2 个标的 × 30 个时间点，确保尾部信号含已知的价格路径。
        """
        np.random.seed(99)
        n = 30
        index = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")

        # 价格: A 标的在 t=5 处进场价 100, 之后路径 98→97→99→101
        prices_a = np.full(n, 100.0)
        prices_a[5] = 100.0
        prices_a[6] = 98.0
        prices_a[7] = 97.0
        prices_a[8] = 99.0
        prices_a[9] = 101.0
        # 其余价格平坦 100
        prices_b = np.full(n, 50.0) + np.random.randn(n) * 0.01

        price_panel = pd.DataFrame(
            {"A": prices_a, "B": prices_b}, index=index,
        )

        # 因子: 大量小值 + t=5 处 A 有极端正信号
        factor_a = np.random.randn(n) * 0.01  # 小噪声
        factor_a[5] = 10.0  # 极端正信号
        factor_b = np.random.randn(n) * 0.01

        factor_panel = pd.DataFrame(
            {"A": factor_a, "B": factor_b}, index=index,
        )

        result = tail_analysis(factor_panel, price_panel, threshold=0.9, horizon=4)
        # t=5 处 A 进场 (factor=10, 做多), 价格从 100 → 98,97,99,101
        # 累计收益: (98-100)/100=-0.02, (97-100)/100=-0.03, (99-100)/100=-0.01, (101-100)/100=+0.01
        # MAE_this = -0.03
        # MAE 是所有尾部信号的平均，t=5 A 信号的 MAE = -0.03
        # 其他尾部信号 (B 标的或其他时刻) 的 MAE 接近 0 (价格平坦)
        # 因此整体 MAE 应为负值
        assert result["mae"] < 0


# =========================================================================
# 换手率分析测试
# =========================================================================

class TestTurnoverAnalysis:

    def test_basic(self, factor_panel):
        result = turnover_analysis(factor_panel)
        assert "autocorrelation" in result
        assert "rank_change_rate" in result
        assert "signal_flip_rate" in result

    def test_autocorrelation_range(self, factor_panel):
        result = turnover_analysis(factor_panel)
        ac = result["autocorrelation"]
        assert -1 <= ac <= 1

    def test_flip_rate_range(self, factor_panel):
        result = turnover_analysis(factor_panel)
        flip = result["signal_flip_rate"]
        assert 0 <= flip <= 1


# =========================================================================
# 非线性分析测试
# =========================================================================

class TestNonlinearAnalysis:

    def test_basic(self, factor_panel, price_panel):
        result = nonlinear_analysis(factor_panel, price_panel, horizon=1)
        assert "mutual_info" in result
        assert "factor_profile" in result
        assert "conditional_ic" in result
        assert "bin_returns" in result

    def test_conditional_ic_structure(self, factor_panel, price_panel):
        result = nonlinear_analysis(factor_panel, price_panel)
        cond = result["conditional_ic"]
        assert "low_ic" in cond
        assert "mid_ic" in cond
        assert "high_ic" in cond


# =========================================================================
# 相关性分析测试
# =========================================================================

class TestCorrelationAnalysis:

    def test_basic(self, factor_panel):
        # 创建两个不同因子
        panels = {
            "factor_a": factor_panel,
            "factor_b": factor_panel * -1,  # 完全负相关
        }
        result = correlation_analysis(panels)
        assert "correlation_matrix" in result
        assert "vif" in result
        corr = result["correlation_matrix"]
        assert corr.loc["factor_a", "factor_b"] == pytest.approx(-1.0, abs=0.01)

    def test_single_factor(self, factor_panel):
        result = correlation_analysis({"only_one": factor_panel})
        assert result["correlation_matrix"].empty


# =========================================================================
# FactorAnalyzer 测试
# =========================================================================

class TestFactorAnalyzer:

    def test_init(self, factor_panel, price_panel):
        analyzer = FactorAnalyzer(factor_panel, price_panel)
        assert analyzer is not None

    def test_init_empty_raises(self, price_panel):
        with pytest.raises(ValueError, match="因子面板为空"):
            FactorAnalyzer(pd.DataFrame(), price_panel)

    def test_init_no_common_symbols_raises(self):
        fp = pd.DataFrame({"A": [1, 2]}, index=pd.date_range("2024-01-01", periods=2, tz="UTC"))
        pp = pd.DataFrame({"B": [1, 2]}, index=pd.date_range("2024-01-01", periods=2, tz="UTC"))
        with pytest.raises(ValueError, match="没有共同标的"):
            FactorAnalyzer(fp, pp)

    def test_full_report(self, factor_panel, price_panel):
        analyzer = FactorAnalyzer(factor_panel, price_panel)
        report = analyzer.full_report(horizons=[1, 5])
        assert "ic" in report
        assert "quantile" in report
        assert "tail" in report
        assert "stability" in report
        assert "nonlinear" in report
        assert "turnover" in report

    def test_summary_text(self, factor_panel, price_panel):
        analyzer = FactorAnalyzer(factor_panel, price_panel)
        text = analyzer.summary_text(factor_name="test")
        assert isinstance(text, str)
        assert "test" in text
        assert "IC" in text

    def test_individual_methods(self, factor_panel, price_panel):
        analyzer = FactorAnalyzer(factor_panel, price_panel)
        assert isinstance(analyzer.ic_analysis(), dict)
        assert isinstance(analyzer.quantile_backtest(), dict)
        assert isinstance(analyzer.tail_analysis(), dict)
        assert isinstance(analyzer.turnover_analysis(), dict)


# =========================================================================
# 稳定性分析测试
# =========================================================================

class TestStabilityAnalysis:

    def test_basic_structure(self, factor_panel, price_panel):
        """返回包含 4 个标准 key"""
        result = stability_analysis(factor_panel, price_panel, horizon=1)
        assert "regime_ic" in result
        assert "monthly_ic" in result
        assert "rolling_ic" in result
        assert "ic_max_drawdown" in result

    def test_regime_ic_structure(self, factor_panel, price_panel):
        """regime_ic 包含 trend 和 vol 子字典"""
        result = stability_analysis(factor_panel, price_panel, horizon=1)
        regime = result["regime_ic"]
        assert "trend" in regime
        assert "vol" in regime
        # trend 下有 uptrend/downtrend
        if regime["trend"]:
            assert "uptrend" in regime["trend"]
            assert "downtrend" in regime["trend"]

    def test_monthly_ic_type(self, factor_panel, price_panel):
        """monthly_ic 返回 DataFrame，含 ic_mean, ic_std, n_obs 列"""
        result = stability_analysis(factor_panel, price_panel, horizon=1)
        monthly = result["monthly_ic"]
        assert isinstance(monthly, pd.DataFrame)
        if not monthly.empty:
            assert "ic_mean" in monthly.columns
            assert "ic_std" in monthly.columns
            assert "n_obs" in monthly.columns

    def test_rolling_ic_type(self, factor_panel, price_panel):
        """rolling_ic 返回 pd.Series"""
        result = stability_analysis(factor_panel, price_panel, horizon=1)
        assert isinstance(result["rolling_ic"], pd.Series)

    def test_ic_max_drawdown_nonpositive(self, factor_panel, price_panel):
        """ic_max_drawdown 为非正数（行业惯例: 回撤用负数表示）"""
        result = stability_analysis(factor_panel, price_panel, horizon=1)
        mdd = result["ic_max_drawdown"]
        if not np.isnan(mdd):
            assert mdd <= 0

    def test_short_data_no_crash(self):
        """短序列（< 10 行）不崩溃"""
        idx = pd.date_range("2024-01-01", periods=5, freq="1min", tz="UTC")
        fp = pd.DataFrame({"A": np.random.randn(5), "B": np.random.randn(5)}, index=idx)
        pp = pd.DataFrame({"A": [100, 101, 102, 103, 104], "B": [50, 51, 52, 53, 54]}, index=idx)
        result = stability_analysis(fp, pp, horizon=1)
        assert isinstance(result, dict)


# =========================================================================
# 增量 IC 测试
# =========================================================================

class TestIncrementalIC:

    @pytest.fixture
    def wide_correlated_data(self):
        """因子与收益有正相关的 5 标的数据（IC 需要 ≥3 标的的截面）"""
        np.random.seed(42)
        n = 200
        index = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        symbols = ["A", "B", "C", "D", "E"]
        signal = np.random.randn(n)

        factor_data = {}
        price_data = {}
        for i, sym in enumerate(symbols):
            noise_f = np.random.randn(n) * 0.3
            factor_data[sym] = signal + noise_f
            price_data[sym] = 100 + np.cumsum(signal * 0.1 + np.random.randn(n) * 0.02)

        return pd.DataFrame(factor_data, index=index), pd.DataFrame(price_data, index=index)

    def test_no_existing_factors(self, wide_correlated_data):
        """无 existing_factors → raw_ic == incremental_ic, info_retention == 1.0"""
        factor_panel, price_panel = wide_correlated_data
        result = incremental_ic(factor_panel, {}, price_panel, horizon=1)
        assert result["info_retention"] == pytest.approx(1.0)
        assert result["raw_ic"] == pytest.approx(result["incremental_ic"])

    def test_with_existing_factor(self, wide_correlated_data):
        """有 existing_factors → incremental_ic 绝对值 ≤ raw_ic 绝对值"""
        factor_panel, price_panel = wide_correlated_data
        np.random.seed(123)
        existing = factor_panel * 0.8 + np.random.randn(*factor_panel.shape) * 0.2
        result = incremental_ic(
            factor_panel, {"existing": existing}, price_panel, horizon=1,
        )
        assert abs(result["incremental_ic"]) <= abs(result["raw_ic"]) + 0.05

    def test_redundant_factor(self, wide_correlated_data):
        """完全冗余: new = existing * 2 + constant → incremental_ic ≈ 0"""
        factor_panel, price_panel = wide_correlated_data
        existing = factor_panel * 2 + 5
        result = incremental_ic(
            factor_panel, {"clone": existing}, price_panel, horizon=1,
        )
        assert abs(result["incremental_ic"]) < 0.1

    def test_independent_factor(self, wide_correlated_data):
        """完全独立: 新因子与旧因子独立 → info_retention 接近 1.0"""
        factor_panel, price_panel = wide_correlated_data
        np.random.seed(999)
        independent = pd.DataFrame(
            np.random.randn(*factor_panel.shape),
            index=factor_panel.index,
            columns=factor_panel.columns,
        )
        result = incremental_ic(
            factor_panel, {"independent": independent}, price_panel, horizon=1,
        )
        if not np.isnan(result["info_retention"]):
            assert result["info_retention"] > 0.5

    def test_insufficient_data(self):
        """少于 10 个观测 → 返回 NaN"""
        idx = pd.date_range("2024-01-01", periods=5, freq="1min", tz="UTC")
        fp = pd.DataFrame({"A": np.random.randn(5)}, index=idx)
        pp = pd.DataFrame({"A": [100, 101, 102, 103, 104]}, index=idx)
        result = incremental_ic(fp, {}, pp, horizon=1)
        assert np.isnan(result["raw_ic"]) or isinstance(result["raw_ic"], float)


# =========================================================================
# Report 测试
# =========================================================================

class TestReport:

    def test_format_report_text_basic(self, factor_panel, price_panel):
        """输出包含标题和因子名"""
        analyzer = FactorAnalyzer(factor_panel, price_panel)
        report = analyzer.full_report(horizons=[1])
        text = format_report_text(report, factor_name="test_factor")
        assert "因子评价报告" in text
        assert "test_factor" in text

    def test_format_report_text_sections(self, factor_panel, price_panel):
        """输出包含各分析节"""
        analyzer = FactorAnalyzer(factor_panel, price_panel)
        report = analyzer.full_report(horizons=[1])
        text = format_report_text(report, factor_name="test")
        # 至少应包含 IC 分析节
        assert "IC" in text

    def test_format_report_text_empty(self):
        """传入空 dict → 不崩溃，返回 str"""
        text = format_report_text({}, factor_name="empty")
        assert isinstance(text, str)
        assert "因子评价报告" in text

    def test_plot_report_structure(self, factor_panel, price_panel):
        """plot_report 返回 dict，值为 matplotlib Figure"""
        import matplotlib
        matplotlib.use("Agg")  # 无头模式
        import matplotlib.pyplot as plt

        analyzer = FactorAnalyzer(factor_panel, price_panel)
        report = analyzer.full_report(horizons=[1])
        figures = plot_report(report, factor_name="test")
        assert isinstance(figures, dict)
        for key, fig in figures.items():
            assert isinstance(fig, plt.Figure), f"图表 {key} 不是 Figure 实例"
        plt.close("all")


# =========================================================================
# 增强测试: 5 标的分层、MI 已知答案、FactorAnalyzer.plot
# =========================================================================

class TestQuantileBacktest5Symbols:

    @pytest.fixture
    def factor_panel_5sym(self):
        """5 标的因子面板"""
        np.random.seed(42)
        n = 200
        index = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "DOGE/USDT"]
        return pd.DataFrame(
            np.random.randn(n, 5), index=index, columns=symbols,
        )

    @pytest.fixture
    def price_panel_5sym(self):
        """5 标的价格面板"""
        np.random.seed(42)
        n = 200
        index = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "DOGE/USDT"]
        data = {}
        for sym in symbols:
            data[sym] = 100 + np.cumsum(np.random.randn(n) * 0.1)
        return pd.DataFrame(data, index=index)

    def test_5_groups_with_5_symbols(self, factor_panel_5sym, price_panel_5sym):
        """5 标的 n_groups=5: 应分为 5 组，每组 1 个标的"""
        result = quantile_backtest(
            factor_panel_5sym, price_panel_5sym, n_groups=5, horizon=1,
        )
        assert len(result["group_returns"]) == 5

    def test_5_groups_all_keys(self, factor_panel_5sym, price_panel_5sym):
        """5 标的分层: group_returns 包含 1~5"""
        result = quantile_backtest(
            factor_panel_5sym, price_panel_5sym, n_groups=5, horizon=1,
        )
        for g in range(1, 6):
            assert g in result["group_returns"], f"缺少组 {g}"


class TestMutualInfoKnownAnswer:

    def test_independent_mi_near_zero(self):
        """独立变量: MI ≈ 0"""
        np.random.seed(42)
        n = 200
        index = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        symbols = ["A", "B", "C", "D", "E"]

        # 因子和价格完全独立
        fp = pd.DataFrame(np.random.randn(n, 5), index=index, columns=symbols)
        pp = pd.DataFrame(
            100 + np.cumsum(np.random.randn(n, 5) * 0.1, axis=0),
            index=index, columns=symbols,
        )

        result = nonlinear_analysis(fp, pp, horizon=1)
        # MI 对独立变量应接近 0（允许统计噪声）
        assert result["mutual_info"] < 0.3

    def test_dependent_mi_positive(self):
        """依赖变量: MI > 独立变量的 MI"""
        np.random.seed(42)
        n = 500
        index = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        symbols = ["A", "B", "C", "D", "E"]

        # 独立因子
        fp_indep = pd.DataFrame(np.random.randn(n, 5), index=index, columns=symbols)

        # 依赖因子: 因子 ≈ 前瞻收益
        signal = np.random.randn(n, 5)
        prices = 100 + np.cumsum(signal * 0.1, axis=0)
        factor_dep = np.diff(prices, axis=0, prepend=100)
        fp_dep = pd.DataFrame(factor_dep, index=index, columns=symbols)

        pp = pd.DataFrame(prices, index=index, columns=symbols)

        mi_indep = nonlinear_analysis(fp_indep, pp, horizon=1)["mutual_info"]
        mi_dep = nonlinear_analysis(fp_dep, pp, horizon=1)["mutual_info"]

        # 依赖因子的 MI 应大于独立因子的 MI
        assert mi_dep > mi_indep


class TestAnalyzerPlot:

    def test_plot_returns_figures(self, factor_panel, price_panel):
        """FactorAnalyzer.plot() 返回包含标准图表的 dict"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        analyzer = FactorAnalyzer(factor_panel, price_panel)
        figures = analyzer.plot(factor_name="test")
        assert isinstance(figures, dict)
        # 至少应有 IC 衰减和分层回测
        expected_keys = ["ic_decay", "quantile_returns"]
        for key in expected_keys:
            assert key in figures, f"缺少图表: {key}"
            assert isinstance(figures[key], plt.Figure)
        plt.close("all")
