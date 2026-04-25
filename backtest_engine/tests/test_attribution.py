"""
test_attribution.py — cost_decomposition / deviation_attribution / regime_breakdown /
compute_per_symbol_cost

覆盖 §11.7.11 测试设计 + per-symbol vs portfolio 加总一致性护栏（D11）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_model.backtest.performance import BacktestResult

from backtest_engine.attribution import (
    cost_decomposition, deviation_attribution, regime_breakdown,
    compute_per_symbol_cost,
)
from backtest_engine.config import BacktestConfig, RunMode
from backtest_engine.pnl import PnLTracker
from backtest_engine.rebalancer import ExecutionReport
from backtest_engine.report import BacktestReport, SCHEMA_VERSION


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _zero_exec_report(symbols, t):
    return ExecutionReport(
        timestamp=t,
        actual_delta=pd.Series(0.0, index=symbols),
        trade_values=pd.Series(0.0, index=symbols),
        fee_cost=0.0, spread_cost=0.0, impact_cost=0.0,
        filtered_symbols=[],
    )


def _make_tracker_with_costs(symbols, n=20, fee=0.0001, spread=0.00005,
                              impact=0.00008, funding_at=None,
                              funding_rate=0.0001) -> PnLTracker:
    """构造一个跑过 n bar 的 PnLTracker，带固定成本"""
    ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    tracker = PnLTracker(10000.0)
    for t in ts:
        if funding_at is not None and t == funding_at:
            tracker.apply_funding_settlement(
                t, pd.Series([0.5] * len(symbols), index=symbols),
                pd.Series([funding_rate] * len(symbols), index=symbols),
            )
        report = ExecutionReport(
            timestamp=t,
            actual_delta=pd.Series(0.0, index=symbols),
            trade_values=pd.Series(0.0, index=symbols),
            fee_cost=fee, spread_cost=spread, impact_cost=impact,
            filtered_symbols=[],
        )
        tracker.record(
            t, pd.Series([0.5] * len(symbols), index=symbols),
            pd.Series([100.0] * len(symbols), index=symbols), report,
        )
    return tracker


def _make_synthetic_result(n=100, mean_ret=0.0001, vol=0.001, seed=0):
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    rng = np.random.default_rng(seed)
    returns = pd.Series(rng.normal(mean_ret, vol, n), index=idx)
    equity = (1 + returns).cumprod() * 10000
    return BacktestResult(
        equity_curve=equity, returns=returns,
        turnover=pd.Series(np.abs(rng.normal(0, 0.01, n)), index=idx),
        weights_history=pd.DataFrame(
            rng.uniform(-0.5, 0.5, (n, 1)),
            index=idx, columns=["BTC/USDT"],
        ),
        gross_returns=returns,
        total_cost=0.001,
    )


# ---------------------------------------------------------------------------
# cost_decomposition
# ---------------------------------------------------------------------------

class TestCostDecomposition:

    def test_basic_keys_5_5_4(self):
        """absolute / annualized_bp 含 5 keys；share 含 4 keys（无 total）"""
        symbols = ["BTC/USDT"]
        tracker = _make_tracker_with_costs(symbols, n=20)
        out = cost_decomposition(tracker, 525960)
        assert set(out["absolute"].keys()) == {"fee", "spread", "impact", "funding", "total"}
        assert set(out["annualized_bp"].keys()) == {"fee", "spread", "impact", "funding", "total"}
        assert set(out["share"].keys()) == {"fee", "spread", "impact", "funding"}

    def test_absolute_sums_correctly(self):
        symbols = ["BTC/USDT"]
        tracker = _make_tracker_with_costs(symbols, n=20, fee=0.001, spread=0.0005, impact=0.0003)
        out = cost_decomposition(tracker, 525960)
        # 20 bars × 0.001 = 0.02
        assert np.isclose(out["absolute"]["fee"], 0.02, rtol=1e-12)
        assert np.isclose(out["absolute"]["spread"], 0.01, rtol=1e-12)
        assert np.isclose(out["absolute"]["impact"], 0.006, rtol=1e-12)

    def test_annualized_bp_formula(self):
        symbols = ["BTC/USDT"]
        n = 20
        fee = 0.0001
        tracker = _make_tracker_with_costs(symbols, n=n, fee=fee, spread=0, impact=0)
        out = cost_decomposition(tracker, 525960)
        # 年化 bp = mean × ppy × 1e4 = 0.0001 × 525960 × 1e4
        expected = fee * 525960 * 1e4
        assert np.isclose(out["annualized_bp"]["fee"], expected, rtol=1e-12)

    def test_share_zero_total_returns_nan(self):
        """total = 0 时 share 全部 NaN（不抛除零）"""
        symbols = ["BTC/USDT"]
        tracker = _make_tracker_with_costs(symbols, n=10, fee=0, spread=0, impact=0)
        out = cost_decomposition(tracker, 525960)
        for k in ("fee", "spread", "impact", "funding"):
            assert np.isnan(out["share"][k])

    def test_funding_negative_supported(self):
        symbols = ["BTC/USDT"]
        # short 持仓 + positive funding → funding 收款（负值）
        tracker = PnLTracker(10000.0)
        ts = pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC")

        # bar 0: 建仓 short
        tracker.record(
            ts[0], pd.Series([-0.5], index=symbols),
            pd.Series([100.0], index=symbols), _zero_exec_report(symbols, ts[0]),
        )

        # bar 1: funding event
        tracker.apply_funding_settlement(
            ts[1], pd.Series([-0.5], index=symbols),
            pd.Series([0.001], index=symbols),
        )
        tracker.record(
            ts[1], pd.Series([-0.5], index=symbols),
            pd.Series([100.0], index=symbols), _zero_exec_report(symbols, ts[1]),
        )

        out = cost_decomposition(tracker, 525960)
        # funding_rate_total = -0.5 × 0.001 = -5e-4
        assert out["absolute"]["funding"] < 0


# ---------------------------------------------------------------------------
# deviation_attribution
# ---------------------------------------------------------------------------

def _make_dummy_report(funding_total=-0.001, optimize_n=1) -> BacktestReport:
    base = _make_synthetic_result(seed=0)
    cfg = BacktestConfig(
        strategy_name="t",
        symbols=["BTC/USDT"],
        start=pd.Timestamp("2024-01-01", tz="UTC"),
        end=pd.Timestamp("2024-02-01", tz="UTC"),
        run_mode=RunMode.EVENT_DRIVEN_DYNAMIC_COST,
        constraints=__import__("alpha_model").core.types.PortfolioConstraints(),
        optimize_every_n_bars=optimize_n,
    )
    funding_settlements = {
        "n_events": 3,
        "total_rate": funding_total,
        "mean_rate_per_event": funding_total / 3,
        "first_event": pd.Timestamp("2024-01-01 08:00", tz="UTC"),
        "last_event": pd.Timestamp("2024-01-01 16:00", tz="UTC"),
    }
    return BacktestReport(
        base=base, config=cfg,
        cost_breakdown={"absolute": {}, "annualized_bp": {}, "share": {}},
        deviation=None, regime_stats=None,
        funding_settlements=funding_settlements,
        bankruptcy_timestamp=None,
        run_metadata={
            "run_mode": cfg.run_mode.value, "cost_mode": "full_cost",
            "execution_mode": "market", "start": cfg.start, "end": cfg.end,
            "n_bars": 100, "n_bars_planned": 100,
            "walltime_seconds": 1.0, "schema_version": SCHEMA_VERSION,
        },
    )


class TestDeviationAttribution:

    def test_p1_mode_basic_rows(self):
        """ablation_results=None：funding 一阶近似量化，其他 not_quantified"""
        ed_report = _make_dummy_report()
        vec_result = _make_synthetic_result(seed=1)
        df = deviation_attribution(ed_report, vec_result)

        assert "bias_source" in df.columns
        # 默认行（不含 optimize_every_n_bars，因为 N=1）
        bias_sources = list(df["bias_source"])
        assert "funding 事件" in bias_sources
        assert "min_trade_value 过滤" in bias_sources
        assert "强平不建模" in bias_sources
        assert "总差（实测）" in bias_sources

    def test_total_row_quantified(self):
        ed_report = _make_dummy_report()
        vec_result = _make_synthetic_result(seed=1)
        df = deviation_attribution(ed_report, vec_result)
        total_row = df.loc[df["bias_source"] == "总差（实测）"].iloc[0]
        assert total_row["quantified"] == True  # noqa: E712
        assert not np.isnan(total_row["delta_terminal_return"])
        assert not np.isnan(total_row["delta_sharpe"])

    def test_funding_first_order_approx(self):
        """funding 一阶（vec-ed 约定，Step 3 修订）：delta = +total_funding_rate

        funding_total = -0.002（净收款）→ ed 增 → vec - ed 减 → delta = +(-0.002) = -0.002
        """
        ed_report = _make_dummy_report(funding_total=-0.002)
        vec_result = _make_synthetic_result(seed=1)
        df = deviation_attribution(ed_report, vec_result)
        funding_row = df.loc[df["bias_source"] == "funding 事件"].iloc[0]
        # vec-ed 约定下 funding_delta = +funding_total（v6 sign flip）
        assert np.isclose(funding_row["delta_terminal_return"], -0.002, rtol=1e-12)
        assert funding_row["method"] == "first_order_approx"

    def test_skip_optimize_n_when_eq_1(self):
        ed_report = _make_dummy_report(optimize_n=1)
        vec_result = _make_synthetic_result(seed=1)
        df = deviation_attribution(ed_report, vec_result)
        # 不应包含 optimize_every_n_bars 行
        assert not any(
            "optimize_every_n_bars" in str(s) for s in df["bias_source"]
        )

    def test_include_optimize_n_when_gt_1(self):
        ed_report = _make_dummy_report(optimize_n=5)
        vec_result = _make_synthetic_result(seed=1)
        df = deviation_attribution(ed_report, vec_result)
        opt_rows = df[df["bias_source"].str.contains("optimize_every_n_bars")]
        assert len(opt_rows) == 1
        assert "5" in opt_rows.iloc[0]["bias_source"]

    def test_p2_mode_with_ablation(self):
        """ablation_results 含 funding + min_trade_value：含残差行（≥ 2 项 quantified）"""
        ed_report = _make_dummy_report()
        vec_result = _make_synthetic_result(seed=1)
        ablation_funding = _make_synthetic_result(seed=2)
        ablation_min_trade = _make_synthetic_result(seed=3)

        df = deviation_attribution(
            ed_report, vec_result,
            ablation_results={
                "funding": ablation_funding,
                "min_trade_value": ablation_min_trade,
            },
        )
        funding_row = df.loc[df["bias_source"] == "funding 事件"].iloc[0]
        assert funding_row["method"] == "ablation"
        min_trade_row = df.loc[df["bias_source"] == "min_trade_value 过滤"].iloc[0]
        assert min_trade_row["method"] == "ablation"
        # ≥ 2 quantified（funding + min_trade_value），ablation 模式 → 残差行存在
        residual = df[df["bias_source"] == "未归因残差"]
        assert len(residual) == 1


# ---------------------------------------------------------------------------
# regime_breakdown
# ---------------------------------------------------------------------------

class TestRegimeBreakdown:

    def test_basic_breakdown(self):
        n = 200
        result = _make_synthetic_result(n=n)
        idx = result.returns.index
        # 前 100 bull, 后 100 bear
        regime = pd.Series(["bull"] * 100 + ["bear"] * 100, index=idx)

        cost_series = {
            "fee":    pd.Series(0.0001, index=idx),
            "spread": pd.Series(0.00005, index=idx),
            "impact": pd.Series(0.00003, index=idx),
        }

        df = regime_breakdown(
            result, regime, periods_per_year=525960,
            cost_series=cost_series, min_bars_per_regime=30,
        )
        assert set(df.columns) == {
            "n_bars", "total_return", "sharpe", "max_drawdown",
            "turnover_mean", "cost_rate_bp",
        }
        assert "bull" in df.index
        assert "bear" in df.index
        assert df.loc["bull", "n_bars"] == 100
        assert df.loc["bear", "n_bars"] == 100

    def test_skips_small_regimes(self):
        n = 100
        result = _make_synthetic_result(n=n)
        idx = result.returns.index
        # 仅 5 bar 是 sideways（小于 min=30）
        regime = pd.Series(
            ["bull"] * 50 + ["sideways"] * 5 + ["bear"] * 45, index=idx,
        )
        cost_series = {
            "fee":    pd.Series(0.0001, index=idx),
            "spread": pd.Series(0.0, index=idx),
            "impact": pd.Series(0.0, index=idx),
        }

        df = regime_breakdown(
            result, regime, periods_per_year=525960,
            cost_series=cost_series, min_bars_per_regime=30,
        )
        # sideways 应被跳过
        assert "sideways" not in df.index

    def test_ffill_alignment(self):
        n = 100
        result = _make_synthetic_result(n=n)
        idx = result.returns.index
        # regime 标签稀疏：仅 idx[0] 和 idx[50] 有值
        regime = pd.Series(
            ["bull", "bear"], index=[idx[0], idx[50]],
        )
        cost_series = {
            "fee":    pd.Series(0.0, index=idx),
            "spread": pd.Series(0.0, index=idx),
            "impact": pd.Series(0.0, index=idx),
        }
        df = regime_breakdown(
            result, regime, periods_per_year=525960,
            cost_series=cost_series, min_bars_per_regime=10,
        )
        # ffill 后：[0:50] = bull, [50:100] = bear
        assert df.loc["bull", "n_bars"] == 50
        assert df.loc["bear", "n_bars"] == 50

    def test_missing_cost_series_keys_raises(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="1min", tz="UTC")
        result = _make_synthetic_result(n=10)
        regime = pd.Series(["bull"] * 10, index=idx)
        with pytest.raises(KeyError, match="缺 keys"):
            regime_breakdown(
                result, regime, periods_per_year=525960,
                cost_series={"fee": pd.Series(0, index=idx)},  # 缺 spread/impact
            )

    def test_cost_rate_bp_excludes_funding(self):
        """cost_rate_bp 不含 funding（§11.7.5 规范）"""
        n = 100
        result = _make_synthetic_result(n=n)
        idx = result.returns.index
        regime = pd.Series(["bull"] * n, index=idx)
        cost_series = {
            "fee":    pd.Series(0.0001, index=idx),
            "spread": pd.Series(0.0001, index=idx),
            "impact": pd.Series(0.0001, index=idx),
        }
        df = regime_breakdown(
            result, regime, periods_per_year=525960,
            cost_series=cost_series, min_bars_per_regime=10,
        )
        # 三 series 各 0.0001，sum mean = 0.0003 → 0.0003 × 525960 × 1e4 ≈ 1.578e6
        expected = 0.0003 * 525960 * 1e4
        assert np.isclose(df.loc["bull", "cost_rate_bp"], expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# compute_per_symbol_cost — D11 关键护栏
# ---------------------------------------------------------------------------

class TestComputePerSymbolCost:

    def test_per_symbol_sums_to_portfolio(self):
        """关键护栏（§11.7.11 D11）：per-symbol 加总 ≡ portfolio 总"""
        symbols = ["BTC/USDT", "ETH/USDT"]
        n = 20
        ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        rng = np.random.default_rng(0)

        # 构造 weights / panels / V history
        weights = pd.DataFrame(
            rng.uniform(-0.3, 0.3, (n, 2)), index=ts, columns=symbols,
        )
        spread_panel = pd.DataFrame(
            np.full((n, 2), 0.0005), index=ts, columns=symbols,
        )
        adv_panel = pd.DataFrame(
            np.full((n, 2), 1e9), index=ts, columns=symbols,
        )
        vol_panel = pd.DataFrame(
            np.full((n, 2), 0.03), index=ts, columns=symbols,
        )
        V_history = pd.Series([10000.0] * n, index=ts)

        # per-symbol 重算
        per_sym = compute_per_symbol_cost(
            weights_history=weights,
            spread_panel=spread_panel, adv_panel=adv_panel, vol_panel=vol_panel,
            fee_rate=0.0004, impact_coeff=0.1,
            portfolio_value_history=V_history,
        )

        # 用 Rebalancer 模拟一遍以拿到 portfolio 总
        from backtest_engine.rebalancer import Rebalancer
        from backtest_engine.config import ExecutionMode, CostMode
        from execution_optimizer import MarketContext

        r = Rebalancer(
            execution_mode=ExecutionMode.MARKET, cost_mode=CostMode.FULL_COST,
            min_trade_value=0.001, fee_rate=0.0004, impact_coeff=0.1,
        )
        portfolio_fee, portfolio_spread, portfolio_impact = 0.0, 0.0, 0.0
        # Step 4 修订：从 i=0 开始（含首 bar），current_w 初始为 0
        # 与 compute_per_symbol_cost 用 shift(1, fill_value=0.0) 一致
        for i in range(n):
            cw_for_step = (
                pd.Series(0.0, index=symbols) if i == 0 else weights.iloc[i-1]
            )
            ctx = MarketContext(
                timestamp=ts[i], symbols=symbols,
                spread=pd.Series([0.0005, 0.0005], index=symbols),
                volatility=pd.Series([0.03, 0.03], index=symbols),
                adv=pd.Series([1e9, 1e9], index=symbols),
                portfolio_value=10000.0, funding_rate=None,
            )
            _, report = r.execute(
                cw_for_step, weights.iloc[i], ctx, pd.Series(),
            )
            portfolio_fee += report.fee_cost
            portfolio_spread += report.spread_cost
            portfolio_impact += report.impact_cost

        # per-symbol sum vs portfolio total
        assert np.isclose(per_sym["fee"].sum().sum(), portfolio_fee, rtol=1e-12)
        assert np.isclose(per_sym["spread"].sum().sum(), portfolio_spread, rtol=1e-12)
        assert np.isclose(per_sym["impact"].sum().sum(), portfolio_impact, rtol=1e-10)

    def test_uses_dynamic_v(self):
        """impact 用 portfolio_value_history.shift(1)（不用 initial_V）"""
        symbols = ["BTC/USDT"]
        n = 5
        ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        # bar 0: w=0, bar 1: w=0.5（Δw=0.5，V_prev=V[0]）, bar 2: w=0.7（Δw=0.2，V_prev=V[1]）
        weights = pd.DataFrame({"BTC/USDT": [0.0, 0.5, 0.7, 0.7, 0.7]}, index=ts)
        # V 大幅变化：A 平稳 10000；B bar 1 跳到 100000
        V_a = pd.Series([10000.0] * n, index=ts)
        V_b = pd.Series([10000.0, 100000.0, 100000.0, 100000.0, 100000.0], index=ts)

        spread_panel = pd.DataFrame({"BTC/USDT": [0.0] * n}, index=ts)
        adv_panel = pd.DataFrame({"BTC/USDT": [1e9] * n}, index=ts)
        vol_panel = pd.DataFrame({"BTC/USDT": [0.03] * n}, index=ts)

        per_a = compute_per_symbol_cost(
            weights, spread_panel, adv_panel, vol_panel,
            fee_rate=0.0, impact_coeff=0.1, portfolio_value_history=V_a,
        )
        per_b = compute_per_symbol_cost(
            weights, spread_panel, adv_panel, vol_panel,
            fee_rate=0.0, impact_coeff=0.1, portfolio_value_history=V_b,
        )
        # bar 2: V_prev = V[1]；A=10000, B=100000 → impact 与 √V 成正比 → ratio=√10
        ratio = per_b["impact"].iloc[2, 0] / per_a["impact"].iloc[2, 0]
        assert np.isclose(ratio, np.sqrt(10), rtol=1e-6)


# ---------------------------------------------------------------------------
# Step 3 (A2/Z1/Z7/Z9): vec-ed P2 残差测试
# ---------------------------------------------------------------------------

def _make_pair_with_funding_offset(
    funding_total: float, vec_seed: int = 1, n: int = 100,
) -> tuple[BacktestResult, BacktestReport, BacktestResult]:
    """Z1 helper（含 Z7 物理正确 + Z9 returns/equity 自洽 + O7 措辞精度）

    保证（物理符号正确，Z7）：
      sign(ed_term - vec_term) = sign(-funding_total)
      （funding_total > 0 扣款 → ed < vec；funding_total < 0 收款 → ed > vec）

    保证（returns / equity_curve 严格自洽，Z9）：
      ed.equity_curve = (1 + ed.returns).cumprod() × V_initial   严格相等

    数值精度（O7）:
      ed_term ≈ vec_term - funding_total（首阶 magnitude）
      二阶 compounding 误差 ~funding_total × vec_term ~ 1e-4
      （非严格 = -funding_total，但 residual=0 by construction symmetry，
        不依赖 ed_term 具体值）
    """
    vec = _make_synthetic_result(n=n, seed=vec_seed)

    # Z9: 让 ed.returns 末值偏移 -funding_total，cumprod 自然给出对的 equity_curve
    ed_returns = vec.returns.copy()
    ed_returns.iloc[-1] = ed_returns.iloc[-1] - funding_total   # Z7: 物理正确符号
    ed_equity = (1 + ed_returns).cumprod() * vec.equity_curve.iloc[0]

    ed_base = BacktestResult(
        equity_curve=ed_equity,
        returns=ed_returns,                    # Z9: 与 equity_curve 自洽
        turnover=vec.turnover.copy(),
        weights_history=vec.weights_history.copy(),
        gross_returns=vec.gross_returns.copy() if vec.gross_returns is not None else None,
        total_cost=0.0,
    )
    cfg = BacktestConfig(
        strategy_name="t",
        symbols=["BTC/USDT"],
        start=pd.Timestamp("2024-01-01", tz="UTC"),
        end=pd.Timestamp("2024-02-01", tz="UTC"),
        run_mode=RunMode.EVENT_DRIVEN_DYNAMIC_COST,
        constraints=__import__("alpha_model").core.types.PortfolioConstraints(),
    )
    ed_report = BacktestReport(
        base=ed_base, config=cfg,
        cost_breakdown={"absolute": {}, "annualized_bp": {}, "share": {}},
        deviation=None, regime_stats=None,
        funding_settlements={
            "n_events": 1, "total_rate": funding_total,
            "mean_rate_per_event": funding_total,
            "first_event": vec.equity_curve.index[0],
            "last_event": vec.equity_curve.index[-1],
        },
        bankruptcy_timestamp=None,
        run_metadata={
            "run_mode": cfg.run_mode.value, "cost_mode": "full_cost",
            "execution_mode": "market", "start": cfg.start, "end": cfg.end,
            "n_bars": n, "n_bars_planned": n,
            "walltime_seconds": 1.0, "schema_version": SCHEMA_VERSION,
        },
    )

    # abl_funding: 关掉 funding → equity_curve 等于 vec
    abl_funding = BacktestResult(
        equity_curve=vec.equity_curve.copy(),
        returns=vec.returns.copy(),
        turnover=vec.turnover.copy(),
        weights_history=vec.weights_history.copy(),
        gross_returns=vec.gross_returns.copy() if vec.gross_returns is not None else None,
        total_cost=0.0,
    )
    return vec, ed_report, abl_funding


class TestDeviationP2ResidualZero:
    """Z1 + Z7: P2 path 单一 funding 偏差 + 哨兵零变化 ablation：残差 ≈ 0 (rtol=1e-12)"""

    def test_residual_zero_net_received(self):
        """Case 1: funding_total = -0.002（净收款，ed > vec）"""
        vec, ed_report, abl_funding = _make_pair_with_funding_offset(
            funding_total=-0.002, vec_seed=1, n=100,
        )
        # 物理验证（Z7）：净收款下 ed > vec
        ed_term = ed_report.base.equity_curve.iloc[-1] / ed_report.base.equity_curve.iloc[0] - 1
        vec_term = vec.equity_curve.iloc[-1] / vec.equity_curve.iloc[0] - 1
        assert ed_term > vec_term, "Z7: 净收款下 ed_term 应 > vec_term"

        df = deviation_attribution(
            ed_report, vec,
            ablation_results={
                "funding": abl_funding,
                "min_trade_value": ed_report.base,    # 哨兵零变化（Option A）
            },
        )
        residual_row = df[df["bias_source"] == "未归因残差"]
        assert len(residual_row) == 1, "残差行未生成"
        assert abs(residual_row["delta_terminal_return"].iloc[0]) < 1e-12

    def test_residual_zero_funding_charge(self):
        """Case 2: funding_total = +0.001（扣款，ed < vec）"""
        vec, ed_report, abl_funding = _make_pair_with_funding_offset(
            funding_total=+0.001, vec_seed=2, n=100,
        )
        # 物理验证（Z7）：扣款下 ed < vec
        ed_term = ed_report.base.equity_curve.iloc[-1] / ed_report.base.equity_curve.iloc[0] - 1
        vec_term = vec.equity_curve.iloc[-1] / vec.equity_curve.iloc[0] - 1
        assert ed_term < vec_term, "Z7: 扣款下 ed_term 应 < vec_term"

        df = deviation_attribution(
            ed_report, vec,
            ablation_results={
                "funding": abl_funding,
                "min_trade_value": ed_report.base,
            },
        )
        residual_row = df[df["bias_source"] == "未归因残差"]
        assert len(residual_row) == 1
        assert abs(residual_row["delta_terminal_return"].iloc[0]) < 1e-12

    def test_helper_equity_curve_self_consistent(self):
        """Z9: helper 构造的 ed.equity_curve == (1+ed.returns).cumprod() × V_initial"""
        vec, ed_report, _ = _make_pair_with_funding_offset(
            funding_total=-0.002, vec_seed=1, n=100,
        )
        ed_base = ed_report.base
        V_initial = vec.equity_curve.iloc[0]
        reconstructed = (1 + ed_base.returns).cumprod() * V_initial
        np.testing.assert_allclose(
            ed_base.equity_curve.values, reconstructed.values, rtol=1e-12,
        )


# ---------------------------------------------------------------------------
# Step 4 (A3/M2): per-symbol 严格 invariant — 含 funding + 第 1 bar 非零调仓 + V 大幅波动
# ---------------------------------------------------------------------------

class TestPerSymbolStrictInvariant:
    """A3 + M2 关键护栏：per-symbol 加总 == portfolio total，rtol=1e-12

    Step 4 修订：
      - weights_history.shift(1, fill_value=0.0) 包含首 bar 调仓
      - 用 v_at_bar_open_history（PnLTracker 实例字段）替代 portfolio_value_history.shift(1)
        消除"V_prev 与 Rebalancer 看到的 V 不一致"的 1-bar 偏移误差
    """

    def test_per_symbol_sums_to_portfolio_strict(self):
        """含 funding + 第 1 bar 非零调仓 + V 波动：rtol=1e-12 严格成立"""
        symbols = ["BTC/USDT", "ETH/USDT"]
        n = 5
        ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        rng = np.random.default_rng(42)

        # 首 bar 非零调仓 [0.4, -0.4]；后续小幅调整
        weights_arr = rng.normal(0, 0.05, (n, 2))
        weights_arr[0] = [0.4, -0.4]
        weights = pd.DataFrame(weights_arr, index=ts, columns=symbols)

        prices = pd.DataFrame(
            100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, 2)), axis=0)),
            index=ts, columns=symbols,
        )

        # 跑 PnLTracker：含一个 funding event 让 V 跳变
        from backtest_engine.rebalancer import Rebalancer
        from backtest_engine.config import ExecutionMode, CostMode
        from execution_optimizer import MarketContext

        spread = 0.0005
        sigma = 0.03
        adv = 1e9
        coeff = 0.1
        fee_rate = 0.0004
        V0 = 10000.0

        tracker = PnLTracker(V0)
        r = Rebalancer(
            execution_mode=ExecutionMode.MARKET, cost_mode=CostMode.FULL_COST,
            min_trade_value=0.0001, fee_rate=fee_rate, impact_coeff=coeff,
        )
        cw = pd.Series([0.0, 0.0], index=symbols)
        for i, t in enumerate(ts):
            # bar 2 funding event（让 V 大幅缩水）
            if i == 2:
                tracker.apply_funding_settlement(
                    t, cw, pd.Series([0.05, 0.05], index=symbols),  # 5% × 2
                )
            ctx = MarketContext(
                timestamp=t, symbols=symbols,
                spread=pd.Series([spread] * 2, index=symbols),
                volatility=pd.Series([sigma] * 2, index=symbols),
                adv=pd.Series([adv] * 2, index=symbols),
                portfolio_value=tracker.portfolio_value,
                funding_rate=None,
            )
            tw = weights.iloc[i]
            actual_w, exec_report = r.execute(cw, tw, ctx, prices.iloc[i])
            tracker.record(t, actual_w, prices.iloc[i], exec_report)
            cw = actual_w

        # portfolio total（来自 PnLTracker）
        cost_decomp = cost_decomposition(tracker, 525960)

        # per-symbol total（来自 compute_per_symbol_cost，用 v_at_bar_open_history）
        spread_panel = pd.DataFrame([[spread] * 2] * n, index=ts, columns=symbols)
        adv_panel = pd.DataFrame([[adv] * 2] * n, index=ts, columns=symbols)
        vol_panel = pd.DataFrame([[sigma] * 2] * n, index=ts, columns=symbols)

        # 用实际跑出的 weights_history 而非原始 weights（min_trade_value 过滤后可能略有不同）
        wh = pd.DataFrame(tracker._weights_history).T.sort_index()

        per_sym = compute_per_symbol_cost(
            weights_history=wh,
            spread_panel=spread_panel, adv_panel=adv_panel, vol_panel=vol_panel,
            fee_rate=fee_rate, impact_coeff=coeff,
            v_at_bar_open_history=tracker.v_at_bar_open_history,   # ★ Step 4
        )

        # 严格 invariant: per-symbol total == portfolio total
        for k in ("fee", "spread", "impact"):
            np.testing.assert_allclose(
                per_sym[k].sum().sum(), cost_decomp["absolute"][k], rtol=1e-12,
                err_msg=f"per-symbol {k} sum != portfolio total",
            )
