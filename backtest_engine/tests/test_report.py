"""
test_report.py — BacktestReport

覆盖 §11.8 字段规范 + summary/__repr__ + save/load roundtrip + schema_version 兼容性 +
context_panels 嵌入 + config 序列化（PortfolioConstraints / impact_coeff Series / Timestamp）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha_model.backtest.performance import BacktestResult
from alpha_model.core.types import PortfolioConstraints

from backtest_engine.config import (
    BacktestConfig, RunMode, ExecutionMode, CostMode,
)
from backtest_engine.report import (
    BacktestReport, SCHEMA_VERSION, _serialize_config, _deserialize_config,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_base_result(n=20):
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.001, n), index=idx)
    equity = (1 + returns).cumprod() * 10000
    return BacktestResult(
        equity_curve=equity,
        returns=returns,
        turnover=pd.Series(np.abs(rng.normal(0, 0.01, n)), index=idx),
        weights_history=pd.DataFrame(
            rng.uniform(-0.5, 0.5, (n, 2)),
            index=idx, columns=["BTC/USDT", "ETH/USDT"],
        ),
        gross_returns=returns + 0.0001,
        total_cost=0.001,
    )


def _make_config(run_mode=RunMode.EVENT_DRIVEN_FIXED_GAMMA, **overrides):
    base = dict(
        strategy_name="test_strategy",
        symbols=["BTC/USDT", "ETH/USDT"],
        start=pd.Timestamp("2024-01-01", tz="UTC"),
        end=pd.Timestamp("2024-02-01", tz="UTC"),
        run_mode=run_mode,
    )
    base.update(overrides)
    return BacktestConfig(**base)


def _make_report(run_mode=RunMode.EVENT_DRIVEN_FIXED_GAMMA, with_deviation=False,
                 with_regime=False, bankrupt=False) -> BacktestReport:
    return BacktestReport(
        base=_make_base_result(),
        config=_make_config(run_mode=run_mode),
        cost_breakdown={
            "absolute": {"fee": 0.001, "spread": 0.0005, "impact": 0.0008,
                         "funding": -0.0002, "total": 0.0021},
            "annualized_bp": {"fee": 100, "spread": 50, "impact": 80,
                              "funding": -20, "total": 210},
            "share": {"fee": 0.476, "spread": 0.238, "impact": 0.381, "funding": -0.095},
        },
        deviation=(
            pd.DataFrame({
                "bias_source": ["min_trade_value 过滤", "总差（实测）"],
                "delta_terminal_return": [0.0, -0.001],
                "delta_sharpe": [np.nan, -0.05],
                "quantified": [False, True],
                "method": ["not_quantified", "direct"],
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
        funding_settlements=(
            None if run_mode == RunMode.VECTORIZED else {
                "n_events": 3,
                "total_rate": -0.0002,
                "mean_rate_per_event": -6.67e-5,
                "first_event": pd.Timestamp("2024-01-01 08:00", tz="UTC"),
                "last_event": pd.Timestamp("2024-01-01 16:00", tz="UTC"),
            }
        ),
        bankruptcy_timestamp=(
            pd.Timestamp("2024-01-15", tz="UTC") if bankrupt else None
        ),
        run_metadata={
            "run_mode": run_mode.value,
            "cost_mode": "full_cost",
            "execution_mode": "market",
            "start": pd.Timestamp("2024-01-01", tz="UTC"),
            "end": pd.Timestamp("2024-02-01", tz="UTC"),
            "n_bars": 20,
            "n_bars_planned": 20,
            "walltime_seconds": 1.5,
            "schema_version": SCHEMA_VERSION,
        },
    )


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------

class TestSummary:

    def test_contains_base_keys(self):
        s = _make_report().summary()
        for key in [
            "annual_return", "annual_volatility", "sharpe_ratio", "sortino_ratio",
            "calmar_ratio", "max_drawdown", "max_drawdown_duration", "avg_turnover",
            "total_cost", "win_rate", "n_periods", "total_return",
        ]:
            assert key in s, f"missing base key {key}"

    def test_contains_cost_bp_keys(self):
        s = _make_report().summary()
        for key in ["fee_bp", "spread_bp", "impact_bp", "funding_bp", "cost_total_bp"]:
            assert key in s, f"missing cost_bp key {key}"

    def test_contains_status_keys(self):
        s = _make_report().summary()
        assert "bankruptcy_flag" in s
        assert "bankruptcy_timestamp" in s

    def test_includes_deviation_only_when_present(self):
        s_no_dev = _make_report(with_deviation=False).summary()
        assert "total_deviation_terminal" not in s_no_dev

        s_with_dev = _make_report(with_deviation=True).summary()
        assert "total_deviation_terminal" in s_with_dev
        assert np.isclose(s_with_dev["total_deviation_terminal"], -0.001, rtol=1e-12)

    def test_bankruptcy_flag_derives_from_timestamp(self):
        s_a = _make_report(bankrupt=False).summary()
        assert s_a["bankruptcy_flag"] is False
        s_b = _make_report(bankrupt=True).summary()
        assert s_b["bankruptcy_flag"] is True


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

class TestRepr:

    def test_repr_returns_string(self):
        r = _make_report()
        text = repr(r)
        assert isinstance(text, str)
        assert "BacktestReport" in text

    def test_repr_under_25_lines(self):
        text = repr(_make_report(with_deviation=True, with_regime=True, bankrupt=False))
        assert len(text.splitlines()) <= 25

    def test_repr_omits_funding_in_vectorized(self):
        text = repr(_make_report(run_mode=RunMode.VECTORIZED))
        assert "Funding events:" not in text
        assert "Bankruptcy:" not in text

    def test_repr_shows_bankruptcy(self):
        text = repr(_make_report(bankrupt=True))
        assert "YES" in text


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------

class TestToDict:

    def test_to_dict_basic(self):
        d = _make_report(with_deviation=True, with_regime=True).to_dict()
        assert "schema_version" in d
        assert "config" in d
        assert "summary" in d
        assert "base" in d
        assert d["deviation"] is not None
        assert d["regime_stats"] is not None

    def test_to_dict_handles_nan(self):
        rep = _make_report()
        # 注入 NaN 测试 _json_safe
        rep.cost_breakdown["absolute"]["nan_field"] = float("nan")
        d = rep.to_dict()
        # 整体可 json.dumps（无 NaN 报错）
        s = json.dumps(d)
        assert "NaN" not in s


# ---------------------------------------------------------------------------
# save / load roundtrip
# ---------------------------------------------------------------------------

class TestSaveLoadRoundtrip:

    def test_basic_roundtrip(self, tmp_path):
        rep1 = _make_report(with_regime=True)
        path = tmp_path / "report1"
        rep1.save(path)
        rep2 = BacktestReport.load(path)

        # base 字段一致
        pd.testing.assert_series_equal(
            rep1.base.equity_curve, rep2.base.equity_curve, check_freq=False,
        )
        pd.testing.assert_frame_equal(
            rep1.base.weights_history, rep2.base.weights_history, check_freq=False,
        )
        assert rep1.base.total_cost == rep2.base.total_cost

        # cost_breakdown / regime_stats
        assert rep1.cost_breakdown == rep2.cost_breakdown
        pd.testing.assert_frame_equal(rep1.regime_stats, rep2.regime_stats)

        # config 关键字段
        assert rep1.config.strategy_name == rep2.config.strategy_name
        assert rep1.config.run_mode == rep2.config.run_mode
        assert rep1.config.start == rep2.config.start

    def test_save_refuses_existing_dir(self, tmp_path):
        rep = _make_report()
        path = tmp_path / "existing"
        path.mkdir()
        with pytest.raises(FileExistsError):
            rep.save(path)

    def test_save_overwrite(self, tmp_path):
        rep1 = _make_report()
        path = tmp_path / "rep"
        rep1.save(path)
        # overwrite=True 不抛
        rep1.save(path, overwrite=True)

    def test_load_rejects_unknown_schema(self, tmp_path):
        rep = _make_report()
        path = tmp_path / "rep"
        rep.save(path)
        # 改 metadata.json 把 schema_version 设到 2.0
        meta_path = path / "metadata.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["schema_version"] = "2.0"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        with pytest.raises(ValueError, match="不兼容"):
            BacktestReport.load(path)

    def test_load_rejects_missing_schema(self, tmp_path):
        rep = _make_report()
        path = tmp_path / "rep"
        rep.save(path)
        meta_path = path / "metadata.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        del meta["schema_version"]
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        with pytest.raises(ValueError, match="schema_version"):
            BacktestReport.load(path)

    def test_load_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            BacktestReport.load(tmp_path / "nonexistent")

    def test_atomic_no_partial_state_on_failure(self, tmp_path, monkeypatch):
        """v3 修订：写入失败时不留半成品"""
        rep = _make_report()
        path = tmp_path / "rep"

        # 通过 monkeypatch 让 to_parquet 抛异常
        original = pd.Series.to_frame
        def fake_to_frame(*args, **kwargs):
            raise RuntimeError("disk full")

        monkeypatch.setattr(pd.Series, "to_frame", fake_to_frame)
        with pytest.raises(RuntimeError, match="disk full"):
            rep.save(path)

        # tmp_dir 应被清理；report_dir 不应存在（半成品）
        assert not path.exists()


# ---------------------------------------------------------------------------
# context_panels 嵌入
# ---------------------------------------------------------------------------

class TestContextEmbedding:

    def test_save_without_context_no_subdir(self, tmp_path):
        rep = _make_report()
        path = tmp_path / "rep"
        rep.save(path)
        assert not (path / "context").exists()
        rep2 = BacktestReport.load(path)
        assert rep2.context_panels is None

    def test_save_with_context_writes_subdir(self, tmp_path, monkeypatch):
        rep = _make_report()

        # 构造 mock context_builder
        class MockBuilder:
            def build_panels(self):
                idx = pd.date_range("2024-01-01", periods=5, freq="1min", tz="UTC")
                return {
                    "spread_panel": pd.DataFrame({"BTC/USDT": [0.001] * 5}, index=idx),
                    "adv_panel": pd.DataFrame({"BTC/USDT": [1e9] * 5}, index=idx),
                    "vol_panel": pd.DataFrame({"BTC/USDT": [0.03] * 5}, index=idx),
                    "price_panel": pd.DataFrame({"BTC/USDT": [100.0] * 5}, index=idx),
                }

        path = tmp_path / "rep_ctx"
        rep.save(path, context_builder=MockBuilder())
        assert (path / "context" / "spread_panel.parquet").exists()
        assert (path / "context" / "adv_panel.parquet").exists()
        assert (path / "context" / "vol_panel.parquet").exists()
        # price_panel 不嵌入（§11.8.3.3）
        assert not (path / "context" / "price_panel.parquet").exists()

        # load 自动检测
        rep2 = BacktestReport.load(path)
        assert rep2.context_panels is not None
        assert set(rep2.context_panels.keys()) == {"spread_panel", "adv_panel", "vol_panel"}


# ---------------------------------------------------------------------------
# config 序列化
# ---------------------------------------------------------------------------

class TestConfigSerialization:

    def test_basic_roundtrip(self):
        cfg = _make_config()
        d = _serialize_config(cfg)
        assert isinstance(d, dict)
        cfg2 = _deserialize_config(d)
        assert cfg.strategy_name == cfg2.strategy_name
        assert cfg.run_mode == cfg2.run_mode
        assert cfg.start == cfg2.start

    def test_constraints_roundtrip(self):
        cons = PortfolioConstraints(max_weight=0.3, vol_target=0.2, leverage_cap=1.5)
        cfg = _make_config(
            run_mode=RunMode.EVENT_DRIVEN_DYNAMIC_COST, constraints=cons,
        )
        d = _serialize_config(cfg)
        cfg2 = _deserialize_config(d)
        assert cfg2.constraints.max_weight == 0.3
        assert cfg2.constraints.vol_target == 0.2
        assert cfg2.constraints.leverage_cap == 1.5

    def test_impact_coeff_series_roundtrip(self):
        ic = pd.Series([0.05, 0.2], index=["BTC/USDT", "ETH/USDT"])
        cfg = _make_config(impact_coeff=ic)
        d = _serialize_config(cfg)
        cfg2 = _deserialize_config(d)
        assert isinstance(cfg2.impact_coeff, pd.Series)
        pd.testing.assert_series_equal(
            cfg2.impact_coeff.astype(float),
            ic.astype(float),
            check_names=False, check_index_type=False,
        )

    def test_impact_coeff_float_roundtrip(self):
        cfg = _make_config(impact_coeff=0.15)
        d = _serialize_config(cfg)
        cfg2 = _deserialize_config(d)
        assert isinstance(cfg2.impact_coeff, float)
        assert cfg2.impact_coeff == 0.15

    def test_regime_series_separate_storage(self, tmp_path):
        regime = pd.Series(
            ["bull", "bear"] * 10,
            index=pd.date_range("2024-01-01", periods=20, freq="1h", tz="UTC"),
        )
        cfg = _make_config(regime_series=regime)
        rep = BacktestReport(
            base=_make_base_result(),
            config=cfg,
            cost_breakdown={"absolute": {}, "annualized_bp": {}, "share": {}},
            deviation=None, regime_stats=None,
            funding_settlements=None, bankruptcy_timestamp=None,
            run_metadata={
                "run_mode": cfg.run_mode.value, "cost_mode": "full_cost",
                "execution_mode": "market", "start": cfg.start, "end": cfg.end,
                "n_bars": 20, "n_bars_planned": 20, "walltime_seconds": 1.0,
                "schema_version": SCHEMA_VERSION,
            },
        )
        path = tmp_path / "rep"
        rep.save(path)

        # config.json 不含 regime_series 字段
        with open(path / "config.json", "r", encoding="utf-8") as f:
            cfg_json = json.load(f)
        assert "regime_series" not in cfg_json

        # 但 regime_input.parquet 存在；load 后回填
        assert (path / "regime_input.parquet").exists()
        rep2 = BacktestReport.load(path)
        assert rep2.config.regime_series is not None
        assert len(rep2.config.regime_series) == 20


# ---------------------------------------------------------------------------
# attach_deviation 注入（不真正调 deviation_attribution）
# ---------------------------------------------------------------------------

class TestAttachDeviationStub:
    """这里只测 attach_deviation 的 lazy import 路径；真实功能见 test_attribution"""

    def test_attach_deviation_method_exists(self):
        rep = _make_report()
        assert hasattr(rep, "attach_deviation")
        assert callable(rep.attach_deviation)
