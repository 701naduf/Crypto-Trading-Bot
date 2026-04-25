"""
test_plot.py — plot.py 8 张图 + plot_all

§11.9.8 设计：测以"不抛异常 + 返回类型正确"为主，不做像素级对比。
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from alpha_model.backtest.performance import BacktestResult

from backtest_engine.config import BacktestConfig, RunMode
from backtest_engine.plot import (
    plot_equity_curve, plot_drawdown, plot_returns_distribution,
    plot_cost_breakdown, plot_weights_history, plot_regime_stats,
    plot_rolling_sharpe, plot_deviation_attribution, plot_all,
)
from backtest_engine.report import BacktestReport, SCHEMA_VERSION


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _make_report(with_regime=True, with_deviation=True, bankrupt=False):
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.001, n), index=idx)
    equity = (1 + returns).cumprod() * 10000
    weights = pd.DataFrame(
        rng.uniform(-0.3, 0.3, (n, 3)), index=idx,
        columns=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
    )

    base = BacktestResult(
        equity_curve=equity, returns=returns,
        turnover=pd.Series(np.abs(rng.normal(0, 0.01, n)), index=idx),
        weights_history=weights, gross_returns=returns,
        total_cost=0.001,
    )
    cfg = BacktestConfig(
        strategy_name="test",
        symbols=list(weights.columns),
        start=idx[0], end=idx[-1],
        run_mode=RunMode.EVENT_DRIVEN_FIXED_GAMMA,
    )

    deviation = (
        pd.DataFrame({
            "bias_source": ["funding 事件", "min_trade_value 过滤", "总差（实测）"],
            "delta_terminal_return": [-0.001, np.nan, -0.005],
            "delta_sharpe": [np.nan, np.nan, -0.1],
            "quantified": [True, False, True],
            "method": ["first_order_approx", "not_quantified", "direct"],
            "note": ["", "", ""],
        }) if with_deviation else None
    )
    regime = (
        pd.DataFrame({
            "n_bars": [50, 50],
            "total_return": [0.01, -0.005],
            "sharpe": [1.5, -0.5],
            "max_drawdown": [-0.01, -0.02],
            "turnover_mean": [0.01, 0.012],
            "cost_rate_bp": [200, 250],
        }, index=pd.Index(["bull", "bear"], name="regime"))
        if with_regime else None
    )

    return BacktestReport(
        base=base, config=cfg,
        cost_breakdown={
            "absolute": {"fee": 0.001, "spread": 0.0005, "impact": 0.0008,
                         "funding": -0.0002, "total": 0.0021},
            "annualized_bp": {"fee": 100, "spread": 50, "impact": 80,
                              "funding": -20, "total": 210},
            "share": {"fee": 0.476, "spread": 0.238, "impact": 0.381, "funding": -0.095},
        },
        deviation=deviation, regime_stats=regime,
        funding_settlements={
            "n_events": 3, "total_rate": -0.0002,
            "mean_rate_per_event": -6.67e-5,
            "first_event": idx[10], "last_event": idx[80],
        },
        bankruptcy_timestamp=(idx[50] if bankrupt else None),
        run_metadata={
            "run_mode": cfg.run_mode.value, "cost_mode": "full_cost",
            "execution_mode": "market", "start": idx[0], "end": idx[-1],
            "n_bars": n, "n_bars_planned": n,
            "walltime_seconds": 1.0, "schema_version": SCHEMA_VERSION,
        },
    )


@pytest.fixture
def report():
    return _make_report()


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# 8 个独立 plot 函数
# ---------------------------------------------------------------------------

class TestEachPlot:

    def test_plot_equity_curve(self, report):
        fig = plot_equity_curve(report)
        assert isinstance(fig, plt.Figure)

    def test_plot_drawdown(self, report):
        fig = plot_drawdown(report)
        assert isinstance(fig, plt.Figure)

    def test_plot_returns_distribution(self, report):
        fig = plot_returns_distribution(report)
        assert isinstance(fig, plt.Figure)

    def test_plot_cost_breakdown(self, report):
        fig = plot_cost_breakdown(report)
        assert isinstance(fig, plt.Figure)

    def test_plot_weights_history(self, report):
        fig = plot_weights_history(report)
        assert isinstance(fig, plt.Figure)

    def test_plot_regime_stats(self, report):
        fig = plot_regime_stats(report)
        assert isinstance(fig, plt.Figure)

    def test_plot_rolling_sharpe(self, report):
        fig = plot_rolling_sharpe(report)
        assert isinstance(fig, plt.Figure)

    def test_plot_deviation_attribution(self, report):
        fig = plot_deviation_attribution(report)
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# ax 参数化路径
# ---------------------------------------------------------------------------

class TestAxParameter:

    def test_with_ax_returns_same_figure(self, report):
        fig, ax = plt.subplots()
        out = plot_equity_curve(report, ax=ax)
        assert out is fig

    def test_with_ax_does_not_create_new(self, report):
        n_before = len(plt.get_fignums())
        fig, ax = plt.subplots()
        plot_drawdown(report, ax=ax)
        # 仅创建了一次 figure
        assert len(plt.get_fignums()) - n_before == 1


# ---------------------------------------------------------------------------
# Optional 字段缺失行为
# ---------------------------------------------------------------------------

class TestOptionalDataMissing:

    def test_plot_regime_raises_when_none(self):
        rep = _make_report(with_regime=False)
        with pytest.raises(ValueError, match="regime_stats"):
            plot_regime_stats(rep)

    def test_plot_deviation_raises_when_none(self):
        rep = _make_report(with_deviation=False)
        with pytest.raises(ValueError, match="deviation"):
            plot_deviation_attribution(rep)


# ---------------------------------------------------------------------------
# 破产标记
# ---------------------------------------------------------------------------

class TestBankruptcyMarker:

    def test_equity_marks_bankruptcy(self, monkeypatch):
        rep = _make_report(bankrupt=True)
        # axvline 调用次数计数器
        calls = []
        original_axvline = plt.Axes.axvline

        def counting_axvline(self, *args, **kwargs):
            calls.append(kwargs.get("color", None))
            return original_axvline(self, *args, **kwargs)

        monkeypatch.setattr(plt.Axes, "axvline", counting_axvline)
        plot_equity_curve(rep)
        # 应至少有一次红色竖线
        assert "red" in calls


# ---------------------------------------------------------------------------
# plot_all
# ---------------------------------------------------------------------------

class TestPlotAll:

    def test_plot_all_returns_figure(self, report):
        fig = plot_all(report)
        assert isinstance(fig, plt.Figure)

    def test_plot_all_in_vectorized_mode_no_funding(self, report):
        from dataclasses import replace
        rep_vec = replace(
            report,
            run_metadata={**report.run_metadata, "run_mode": "vectorized"},
            funding_settlements=None, deviation=None,
        )
        fig = plot_all(rep_vec)
        assert isinstance(fig, plt.Figure)

    def test_plot_all_no_regime_no_deviation(self):
        rep = _make_report(with_regime=False, with_deviation=False)
        fig = plot_all(rep)
        assert isinstance(fig, plt.Figure)
