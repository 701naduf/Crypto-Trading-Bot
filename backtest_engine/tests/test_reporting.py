"""
test_reporting.py — Markdown 报告生成

§11.10.7 测试设计：文件存在 / overwrite / 跳过缺失 section / bankruptcy warning /
summary 表格内容 / headless backend 自动启用。
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from alpha_model.backtest.performance import BacktestResult

from backtest_engine.config import BacktestConfig, RunMode
from backtest_engine.report import BacktestReport, SCHEMA_VERSION
from backtest_engine.reporting import to_markdown


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _make_report(with_regime=True, with_deviation=True, bankrupt=False,
                 run_mode=RunMode.EVENT_DRIVEN_FIXED_GAMMA):
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.001, n), index=idx)
    equity = (1 + returns).cumprod() * 10000
    weights = pd.DataFrame(
        rng.uniform(-0.3, 0.3, (n, 2)), index=idx,
        columns=["BTC/USDT", "ETH/USDT"],
    )
    base = BacktestResult(
        equity_curve=equity, returns=returns,
        turnover=pd.Series(np.abs(rng.normal(0, 0.01, n)), index=idx),
        weights_history=weights, gross_returns=returns,
        total_cost=0.001,
    )
    cfg = BacktestConfig(
        strategy_name="test_strategy",
        symbols=list(weights.columns),
        start=idx[0], end=idx[-1],
        run_mode=run_mode,
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
        deviation=(
            pd.DataFrame({
                "bias_source": ["funding 事件", "总差（实测）"],
                "delta_terminal_return": [-0.001, -0.005],
                "delta_sharpe": [np.nan, -0.1],
                "quantified": [True, True],
                "method": ["first_order_approx", "direct"],
                "note": ["", ""],
            }) if with_deviation else None
        ),
        regime_stats=(
            pd.DataFrame({
                "n_bars": [50, 50],
                "total_return": [0.01, -0.005],
                "sharpe": [1.5, -0.5],
                "max_drawdown": [-0.01, -0.02],
                "turnover_mean": [0.01, 0.012],
                "cost_rate_bp": [200, 250],
            }, index=pd.Index(["bull", "bear"], name="regime"))
            if with_regime else None
        ),
        funding_settlements={
            "n_events": 3, "total_rate": -0.0002,
            "mean_rate_per_event": -6.67e-5,
            "first_event": idx[10], "last_event": idx[80],
        },
        bankruptcy_timestamp=(idx[50] if bankrupt else None),
        run_metadata={
            "run_mode": run_mode.value, "cost_mode": "full_cost",
            "execution_mode": "market", "start": idx[0], "end": idx[-1],
            "n_bars": n, "n_bars_planned": n,
            "walltime_seconds": 1.5, "schema_version": SCHEMA_VERSION,
        },
    )


@pytest.fixture(autouse=True)
def close_figs():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# 基础生成
# ---------------------------------------------------------------------------

class TestBasic:

    def test_to_markdown_creates_files(self, tmp_path):
        rep = _make_report()
        path = tmp_path / "report"
        out = to_markdown(rep, path)
        assert out == path / "report.md"
        assert (path / "report.md").exists()
        # 6 个常规图（regime + deviation 也有 → 8）
        png_files = list((path / "figures").glob("*.png"))
        assert len(png_files) >= 6

    def test_returns_md_path(self, tmp_path):
        rep = _make_report()
        out = to_markdown(rep, tmp_path / "rep")
        assert out.suffix == ".md"


# ---------------------------------------------------------------------------
# overwrite
# ---------------------------------------------------------------------------

class TestOverwrite:

    def test_refuses_existing_dir(self, tmp_path):
        rep = _make_report()
        path = tmp_path / "rep"
        to_markdown(rep, path)
        with pytest.raises(FileExistsError):
            to_markdown(rep, path)

    def test_overwrite_works(self, tmp_path):
        rep = _make_report()
        path = tmp_path / "rep"
        to_markdown(rep, path)
        # overwrite=True 不抛
        to_markdown(rep, path, overwrite=True)
        assert (path / "report.md").exists()


# ---------------------------------------------------------------------------
# 内容校验
# ---------------------------------------------------------------------------

class TestContent:

    def test_summary_table_present(self, tmp_path):
        rep = _make_report()
        to_markdown(rep, tmp_path / "rep")
        text = (tmp_path / "rep" / "report.md").read_text(encoding="utf-8")
        assert "## Performance Summary" in text
        assert "| Sharpe ratio |" in text
        assert "| Total return |" in text

    def test_cost_breakdown_table_present(self, tmp_path):
        rep = _make_report()
        to_markdown(rep, tmp_path / "rep")
        text = (tmp_path / "rep" / "report.md").read_text(encoding="utf-8")
        assert "## Cost Breakdown" in text
        assert "| Fee |" in text
        assert "| Funding |" in text
        assert "| **Total** |" in text

    def test_skips_optional_sections(self, tmp_path):
        rep = _make_report(with_regime=False, with_deviation=False)
        to_markdown(rep, tmp_path / "rep")
        text = (tmp_path / "rep" / "report.md").read_text(encoding="utf-8")
        assert "## Regime Analysis" not in text
        assert "## Deviation Attribution" not in text

    def test_includes_regime_section(self, tmp_path):
        rep = _make_report(with_regime=True, with_deviation=False)
        to_markdown(rep, tmp_path / "rep")
        text = (tmp_path / "rep" / "report.md").read_text(encoding="utf-8")
        assert "## Regime Analysis" in text
        assert "bull" in text
        assert "bear" in text

    def test_includes_deviation_section(self, tmp_path):
        rep = _make_report(with_regime=False, with_deviation=True)
        to_markdown(rep, tmp_path / "rep")
        text = (tmp_path / "rep" / "report.md").read_text(encoding="utf-8")
        assert "## Deviation Attribution" in text

    def test_bankruptcy_warning(self, tmp_path):
        rep = _make_report(bankrupt=True)
        to_markdown(rep, tmp_path / "rep")
        text = (tmp_path / "rep" / "report.md").read_text(encoding="utf-8")
        assert "Bankruptcy" in text
        assert "⚠️" in text

    def test_vectorized_omits_funding_warning(self, tmp_path):
        rep = _make_report(run_mode=RunMode.VECTORIZED)
        to_markdown(rep, tmp_path / "rep")
        text = (tmp_path / "rep" / "report.md").read_text(encoding="utf-8")
        # VECTORIZED 模式下不输出 funding 警告
        assert "Funding 是合约持仓成本" not in text

    def test_configuration_section_includes_json(self, tmp_path):
        rep = _make_report()
        to_markdown(rep, tmp_path / "rep")
        text = (tmp_path / "rep" / "report.md").read_text(encoding="utf-8")
        assert "## Configuration" in text
        # 含 strategy_name
        assert "test_strategy" in text


# ---------------------------------------------------------------------------
# 图片
# ---------------------------------------------------------------------------

class TestFigures:

    def test_all_figures_present_in_full_mode(self, tmp_path):
        rep = _make_report(with_regime=True, with_deviation=True)
        to_markdown(rep, tmp_path / "rep")
        figs = list((tmp_path / "rep" / "figures").glob("*.png"))
        names = {f.stem for f in figs}
        assert "equity_curve" in names
        assert "drawdown" in names
        assert "regime_stats" in names
        assert "deviation_attribution" in names

    def test_no_optional_figures_when_missing(self, tmp_path):
        rep = _make_report(with_regime=False, with_deviation=False)
        to_markdown(rep, tmp_path / "rep")
        figs = list((tmp_path / "rep" / "figures").glob("*.png"))
        names = {f.stem for f in figs}
        assert "regime_stats" not in names
        assert "deviation_attribution" not in names


# ---------------------------------------------------------------------------
# Headless backend
# ---------------------------------------------------------------------------

class TestHeadless:

    def test_headless_works(self, tmp_path):
        """to_markdown 内部强制 Agg backend，无需用户提前 setup"""
        # 不显式设置 backend，调 to_markdown 应不报错
        rep = _make_report()
        to_markdown(rep, tmp_path / "rep")
        assert (tmp_path / "rep" / "report.md").exists()
