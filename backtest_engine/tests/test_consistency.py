"""
test_consistency.py — 跨模块/跨模式 数值一致性护栏（rtol=1e-12）

reviewer 第 8 轮（设计者视角）补足：单测必须严格抓住"未来重构破坏 invariant"的回归。
本文件集中所有跨模式 / 跨实现的精度护栏，目的是在 impact 公式 / cost_mode /
per_symbol 重算等关键不变量被破坏时立即失败。

包含：

  TestZeroFrictionEquivalence:
      关闭 fee/spread/impact/funding 后，FIXED_GAMMA 与 VECTORIZED 的 returns
      与 equity 应在 rtol=1e-12 下精确相等（剥离首 bar 因 vec dropna 的索引偏移）。

  TestPerSymbolCostSumsToPortfolioViaEngine:
      engine.run() 路径产出 report 后，compute_per_symbol_cost 加总三分量应
      逐项等于 cost_decomposition().absolute（rtol=1e-12）。

  TestImpactFormulaFourPathsConsistency:
      impact 公式 (2/3)·coeff·σ·√(V/ADV)·|Δw|^1.5 在四处实现：
        1. alpha_model.backtest.vectorized.estimate_market_impact   (panel)
        2. execution_optimizer.cost.build_cost_expression           (cvxpy)
        3. backtest_engine.rebalancer.Rebalancer._execute_market    (np scalar)
        4. backtest_engine.attribution.compute_per_symbol_cost      (panel)
      给定同一组输入，四条路径应数值精确相等（rtol=1e-12）。
"""
from __future__ import annotations

from dataclasses import replace

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from alpha_model.backtest.vectorized import estimate_market_impact
from alpha_model.core.types import PortfolioConstraints
from alpha_model.store.signal_store import SignalStore

from execution_optimizer import ExecutionOptimizer, MarketContext
from execution_optimizer.cost import build_cost_expression

from backtest_engine.attribution import (
    cost_decomposition, compute_per_symbol_cost,
)
from backtest_engine.config import (
    BacktestConfig, RunMode, ExecutionMode, CostMode,
)
from backtest_engine.engine import EventDrivenBacktester
from backtest_engine.pnl import PnLTracker
from backtest_engine.rebalancer import Rebalancer

from .conftest import (
    FakeDataReader,
    _make_synthetic_ohlcv,
    _make_synthetic_orderbook,
    _make_synthetic_funding,
    _make_synthetic_weights,
)


# =============================================================================
# helpers / fixtures（本文件局部）
# =============================================================================

SHORT_VOL_MIN_PERIODS = 200


@pytest.fixture(autouse=True)
def patch_context_builder(monkeypatch):
    """让 engine 内部 MarketContextBuilder 用 200 vol_min_periods 适配短时段合成数据"""
    from backtest_engine import context as context_mod

    original_init = context_mod.MarketContextBuilder.__init__

    def patched_init(self, reader, config, **kwargs):
        kwargs.setdefault("vol_min_periods", SHORT_VOL_MIN_PERIODS)
        original_init(self, reader, config, **kwargs)

    monkeypatch.setattr(context_mod.MarketContextBuilder, "__init__", patched_init)


@pytest.fixture
def short_period(synthetic_period):
    """短评估区间：1 天 = 1441 bar"""
    _, _, end = synthetic_period
    return end - pd.Timedelta(days=1), end


@pytest.fixture
def no_funding_reader(synthetic_symbols, synthetic_period):
    """fake_reader 但 funding_by_symbol 为空 → funding_rates_panel 全空，
    事件循环 (a) 阶段永远不触发 funding，与 VECTORIZED 同口径"""
    earliest, _start, end = synthetic_period
    return FakeDataReader(
        ohlcv_by_symbol=_make_synthetic_ohlcv(synthetic_symbols, earliest, end),
        orderbook_by_symbol=_make_synthetic_orderbook(synthetic_symbols, earliest, end),
        funding_by_symbol={},   # 全空
    )


@pytest.fixture
def zero_first_bar_signal_store(
    tmp_path_factory, synthetic_symbols, synthetic_period,
):
    """SignalStore 但 weights.iloc[0] = 0：让事件驱动首 bar Δw=0，
    与 vectorized weights.diff() 首行 NaN→0 行为一致（首 bar 三分量成本均为 0）"""
    from alpha_model.core.types import ModelMeta, TrainConfig

    _earliest, start, end = synthetic_period
    base_dir = tmp_path_factory.mktemp("zero_first_bar_signal_store")
    store = SignalStore(base_dir=base_dir)

    idx = pd.date_range(start, end, freq="1min", tz="UTC")
    weights = _make_synthetic_weights(synthetic_symbols, idx, seed=0)
    # 关键：首 bar 强制 0（与 vectorized 首行 NaN→0 对齐）
    weights.iloc[0] = 0.0
    signals = pd.DataFrame(0.0, index=idx, columns=synthetic_symbols)

    meta = ModelMeta(
        name="zero_first_bar",
        factor_names=["dummy"],
        target_horizon=10,
        train_config=TrainConfig(),
        constraints=PortfolioConstraints(),
    )
    store.save(
        strategy_name="zero_first_bar",
        weights=weights, signals=signals, meta=meta, performance={},
    )
    return store


# =============================================================================
# A. Zero-friction equivalence: FIXED_GAMMA + MATCH_VECTORIZED + 零摩擦 ≈ VECTORIZED
# =============================================================================

class TestZeroFrictionEquivalence:
    """
    关闭所有摩擦源（fee_rate=0, impact_coeff=0, MATCH_VECTORIZED, 无 funding）后，
    FIXED_GAMMA 应与 VECTORIZED 在 rtol=1e-12 数值精确相等。

    护栏价值：
      - 后续修改 impact / fee / spread 公式时，若 vectorized.py 与 Rebalancer 不同步，
        zero-friction 路径不再保留无变化语义 → 此处立即失败。
      - 修复了 v1 旧测试 rtol=0.05 + 仅看 fee 分量的弱化。
    """

    @pytest.fixture
    def common_cfg(self, synthetic_symbols, short_period):
        start, end = short_period
        return BacktestConfig(
            strategy_name="zero_first_bar",
            symbols=synthetic_symbols,
            start=start, end=end,
            run_mode=RunMode.VECTORIZED,
            fee_rate=0.0,
            impact_coeff=0.0,
            cost_mode=CostMode.MATCH_VECTORIZED,
        )

    def test_returns_strict_equality(
        self, no_funding_reader, zero_first_bar_signal_store, common_cfg,
    ):
        """returns 序列在 rtol=1e-12 下逐 bar 相等（剥离 vec 首 bar 的 dropna）"""
        rep_vec = EventDrivenBacktester().run(
            common_cfg,
            reader=no_funding_reader, signal_store=zero_first_bar_signal_store,
        )
        cfg_fg = replace(
            common_cfg,
            run_mode=RunMode.EVENT_DRIVEN_FIXED_GAMMA,
            min_trade_value=0.0001,   # 实际 = 0，不过滤
        )
        rep_fg = EventDrivenBacktester().run(
            cfg_fg,
            reader=no_funding_reader, signal_store=zero_first_bar_signal_store,
        )

        # vec.returns 首行 NaN（来自 shift），event-driven returns 首行 = 0（prev_w=None）
        # 取 vec 已 dropna 的 equity_curve.index 做共同对齐基准
        common_idx = rep_vec.base.equity_curve.index
        ed_returns = rep_fg.base.returns.reindex(common_idx)
        vec_returns = rep_vec.base.returns.reindex(common_idx)

        # 严格 rtol=1e-12：零摩擦零 funding 下 ed_returns ≡ vec_returns
        np.testing.assert_allclose(
            ed_returns.values, vec_returns.values, rtol=1e-12, atol=1e-15,
            err_msg="zero-friction 下 FIXED_GAMMA returns ≠ VECTORIZED returns",
        )

    def test_equity_strict_equality(
        self, no_funding_reader, zero_first_bar_signal_store, common_cfg,
    ):
        """equity_curve 在 rtol=1e-12 下严格等价（归一化后比较）

        ed.equity 起点 = initial_V，归一化后 = (1+r).cumprod()
        vec.equity = cumulative_returns(net.dropna()) = (1+r).cumprod() - 1
        → ed.equity / V_init - 1 == vec.equity（严格等式）
        """
        rep_vec = EventDrivenBacktester().run(
            common_cfg,
            reader=no_funding_reader, signal_store=zero_first_bar_signal_store,
        )
        cfg_fg = replace(
            common_cfg,
            run_mode=RunMode.EVENT_DRIVEN_FIXED_GAMMA,
            min_trade_value=0.0001,
        )
        rep_fg = EventDrivenBacktester().run(
            cfg_fg,
            reader=no_funding_reader, signal_store=zero_first_bar_signal_store,
        )

        V0 = common_cfg.initial_portfolio_value
        ed_normalized = rep_fg.base.equity_curve / V0 - 1.0
        common_idx = rep_vec.base.equity_curve.index

        np.testing.assert_allclose(
            ed_normalized.reindex(common_idx).values,
            rep_vec.base.equity_curve.reindex(common_idx).values,
            rtol=1e-12, atol=1e-15,
            err_msg="zero-friction 下归一化 equity_curve 不严格相等",
        )

    def test_cost_breakdown_components_zero(
        self, no_funding_reader, zero_first_bar_signal_store, common_cfg,
    ):
        """零摩擦下两边的 fee/spread/impact 都应严格 = 0（rtol=1e-12 是 abs check）"""
        rep_vec = EventDrivenBacktester().run(
            common_cfg,
            reader=no_funding_reader, signal_store=zero_first_bar_signal_store,
        )
        cfg_fg = replace(
            common_cfg,
            run_mode=RunMode.EVENT_DRIVEN_FIXED_GAMMA,
            min_trade_value=0.0001,
        )
        rep_fg = EventDrivenBacktester().run(
            cfg_fg,
            reader=no_funding_reader, signal_store=zero_first_bar_signal_store,
        )

        for k in ("fee", "spread", "impact"):
            assert abs(rep_vec.cost_breakdown["absolute"][k]) < 1e-12, (
                f"VECTORIZED {k} 应为 0，实际 {rep_vec.cost_breakdown['absolute'][k]}"
            )
            assert abs(rep_fg.cost_breakdown["absolute"][k]) < 1e-12, (
                f"FIXED_GAMMA {k} 应为 0，实际 {rep_fg.cost_breakdown['absolute'][k]}"
            )


# =============================================================================
# B. Per-symbol 加总 == portfolio total（engine.run() 路径，rtol=1e-12）
# =============================================================================

class TestPerSymbolCostSumsToPortfolioViaEngine:
    """
    通过 engine.run() 完整路径产出 report 后，调用 compute_per_symbol_cost 重算
    per-symbol cost，加总三分量应严格等于 cost_decomposition()["absolute"]。

    与 test_attribution.TestPerSymbolStrictInvariant 的区别：
      - 该测试用合成 Rebalancer + PnLTracker（白盒）
      - 本测试用 engine.run() 全路径（黑盒）；护栏所有 panel 对齐 / impact_coeff 透传 /
        v_at_bar_open_history 正确性等链路。
    """

    @pytest.fixture
    def fixed_gamma_cfg(self, synthetic_symbols, short_period):
        start, end = short_period
        return BacktestConfig(
            strategy_name="synthetic_test",
            symbols=synthetic_symbols,
            start=start, end=end,
            run_mode=RunMode.EVENT_DRIVEN_FIXED_GAMMA,
        )

    def test_per_symbol_sums_match_portfolio_via_engine(
        self, fake_reader, synthetic_signal_store, fixed_gamma_cfg, monkeypatch,
    ):
        """三分量逐项严格 rtol=1e-12 等于 portfolio total"""
        # 捕获 engine 内部 PnLTracker（用 v_at_bar_open_history 精确重算）
        captured: list[PnLTracker] = []
        original_init = PnLTracker.__init__

        def capturing_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            captured.append(self)

        monkeypatch.setattr(PnLTracker, "__init__", capturing_init)

        rep = EventDrivenBacktester().run(
            fixed_gamma_cfg,
            reader=fake_reader, signal_store=synthetic_signal_store,
        )
        tracker = captured[-1]

        # 重新构造 panels（与 engine 同一 reader / config → 同一 vol/adv/spread 面板）
        from backtest_engine.context import MarketContextBuilder
        builder = MarketContextBuilder(
            fake_reader, fixed_gamma_cfg, vol_min_periods=SHORT_VOL_MIN_PERIODS,
        )
        panels = builder.build_panels()

        wh = rep.base.weights_history
        per_sym = compute_per_symbol_cost(
            weights_history=wh,
            spread_panel=panels["spread_panel"],
            adv_panel=panels["adv_panel"],
            vol_panel=panels["vol_panel"],
            fee_rate=fixed_gamma_cfg.fee_rate,
            impact_coeff=fixed_gamma_cfg.impact_coeff,
            v_at_bar_open_history=tracker.v_at_bar_open_history,
        )

        cost_abs = rep.cost_breakdown["absolute"]
        for k in ("fee", "spread", "impact"):
            np.testing.assert_allclose(
                per_sym[k].sum().sum(), cost_abs[k], rtol=1e-12,
                err_msg=(
                    f"engine 路径 per-symbol {k} 加总 ≠ portfolio total "
                    f"({per_sym[k].sum().sum()} vs {cost_abs[k]})"
                ),
            )


# =============================================================================
# C. impact 公式四处实现 一致性（rtol=1e-12）
# =============================================================================

class TestImpactFormulaFourPathsConsistency:
    """
    impact 公式 (2/3) × coeff × σ × √(V/ADV) × |Δw|^1.5 在四个实现中的数值一致性。

    给定相同输入：
      coeff=0.1, σ=0.03, V=10000, ADV=1e6, Δw=0.1
      expected = (2/3) × 0.1 × 0.03 × √(10000/1e6) × 0.1^1.5
              ≈ 6.32456e-6
    """

    SYM = "BTC/USDT"
    DELTA_W = 0.1
    SIGMA = 0.03
    ADV = 1e6
    V = 10000.0
    COEFF = 0.1

    @property
    def expected(self) -> float:
        return (
            (2.0 / 3.0) * self.COEFF * self.SIGMA
            * np.sqrt(self.V / self.ADV)
            * (self.DELTA_W ** 1.5)
        )

    def test_path1_panel_estimate_market_impact(self):
        """alpha_model.backtest.vectorized.estimate_market_impact (panel)"""
        idx = pd.DatetimeIndex([pd.Timestamp("2024-01-01", tz="UTC")])
        delta_weights = pd.DataFrame([[self.DELTA_W]], index=idx, columns=[self.SYM])
        adv_panel = pd.DataFrame([[self.ADV]], index=idx, columns=[self.SYM])
        vol_panel = pd.DataFrame([[self.SIGMA]], index=idx, columns=[self.SYM])

        result = estimate_market_impact(
            delta_weights, adv_panel, vol_panel,
            portfolio_value=self.V, impact_coeff=self.COEFF,
        )
        np.testing.assert_allclose(
            result.iloc[0, 0], self.expected, rtol=1e-12,
            err_msg="path1 estimate_market_impact ≠ expected",
        )

    def test_path2_cvxpy_build_cost_expression(self):
        """execution_optimizer.cost.build_cost_expression (cvxpy expression)

        与 path1 严格相等：fee_rate=0, spread=0 → expr.value = impact 部分
        """
        symbols = [self.SYM]
        ctx = MarketContext(
            timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
            symbols=symbols,
            spread=pd.Series([0.0], index=symbols),
            volatility=pd.Series([self.SIGMA], index=symbols),
            adv=pd.Series([self.ADV], index=symbols),
            portfolio_value=self.V,
            funding_rate=None,
        )
        delta_w = cp.Variable(1)
        delta_w.value = np.array([self.DELTA_W])

        expr = build_cost_expression(
            delta_w, ctx, impact_coeff=self.COEFF, fee_rate=0.0,
        )
        # fee=0 + spread=0 → expr.value 即 impact
        np.testing.assert_allclose(
            float(expr.value), self.expected, rtol=1e-12,
            err_msg="path2 build_cost_expression ≠ expected",
        )

    def test_path3_rebalancer_execute(self):
        """backtest_engine.rebalancer.Rebalancer._execute_market (np 标量)"""
        symbols = [self.SYM]
        rebalancer = Rebalancer(
            execution_mode=ExecutionMode.MARKET,
            cost_mode=CostMode.FULL_COST,
            min_trade_value=0.0001,
            fee_rate=0.0, impact_coeff=self.COEFF,
        )
        ctx = MarketContext(
            timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
            symbols=symbols,
            spread=pd.Series([0.0], index=symbols),
            volatility=pd.Series([self.SIGMA], index=symbols),
            adv=pd.Series([self.ADV], index=symbols),
            portfolio_value=self.V,
            funding_rate=None,
        )
        current_w = pd.Series([0.0], index=symbols)
        target_w = pd.Series([self.DELTA_W], index=symbols)

        _actual_w, exec_report = rebalancer.execute(
            current_w, target_w, ctx,
            price_at_t=pd.Series([100.0], index=symbols),
        )
        np.testing.assert_allclose(
            exec_report.impact_cost, self.expected, rtol=1e-12,
            err_msg="path3 Rebalancer.impact_cost ≠ expected",
        )

    def test_path4_compute_per_symbol_cost(self):
        """backtest_engine.attribution.compute_per_symbol_cost (panel)"""
        idx = pd.DatetimeIndex([pd.Timestamp("2024-01-01", tz="UTC")])
        # weights_history: 单 bar [Δw]；shift(1, fill_value=0) → 首 bar Δw=DELTA_W
        weights_history = pd.DataFrame([[self.DELTA_W]], index=idx, columns=[self.SYM])
        spread_panel = pd.DataFrame([[0.0]], index=idx, columns=[self.SYM])
        adv_panel = pd.DataFrame([[self.ADV]], index=idx, columns=[self.SYM])
        vol_panel = pd.DataFrame([[self.SIGMA]], index=idx, columns=[self.SYM])
        v_at_bar_open = pd.Series([self.V], index=idx)

        result = compute_per_symbol_cost(
            weights_history=weights_history,
            spread_panel=spread_panel,
            adv_panel=adv_panel,
            vol_panel=vol_panel,
            fee_rate=0.0, impact_coeff=self.COEFF,
            v_at_bar_open_history=v_at_bar_open,
        )
        np.testing.assert_allclose(
            result["impact"].iloc[0, 0], self.expected, rtol=1e-12,
            err_msg="path4 compute_per_symbol_cost.impact ≠ expected",
        )

    def test_all_four_paths_mutually_equal(self):
        """护栏汇总：四条路径两两严格相等（rtol=1e-12），仿黑盒回归测试"""
        # path 1
        idx = pd.DatetimeIndex([pd.Timestamp("2024-01-01", tz="UTC")])
        symbols = [self.SYM]
        p1 = estimate_market_impact(
            pd.DataFrame([[self.DELTA_W]], index=idx, columns=symbols),
            pd.DataFrame([[self.ADV]], index=idx, columns=symbols),
            pd.DataFrame([[self.SIGMA]], index=idx, columns=symbols),
            portfolio_value=self.V, impact_coeff=self.COEFF,
        ).iloc[0, 0]

        # path 2
        ctx = MarketContext(
            timestamp=idx[0], symbols=symbols,
            spread=pd.Series([0.0], index=symbols),
            volatility=pd.Series([self.SIGMA], index=symbols),
            adv=pd.Series([self.ADV], index=symbols),
            portfolio_value=self.V, funding_rate=None,
        )
        dv = cp.Variable(1)
        dv.value = np.array([self.DELTA_W])
        p2 = float(build_cost_expression(
            dv, ctx, impact_coeff=self.COEFF, fee_rate=0.0,
        ).value)

        # path 3
        r = Rebalancer(
            execution_mode=ExecutionMode.MARKET, cost_mode=CostMode.FULL_COST,
            min_trade_value=0.0001, fee_rate=0.0, impact_coeff=self.COEFF,
        )
        _, er = r.execute(
            pd.Series([0.0], index=symbols),
            pd.Series([self.DELTA_W], index=symbols),
            ctx, price_at_t=pd.Series([100.0], index=symbols),
        )
        p3 = er.impact_cost

        # path 4
        p4 = compute_per_symbol_cost(
            weights_history=pd.DataFrame([[self.DELTA_W]], index=idx, columns=symbols),
            spread_panel=pd.DataFrame([[0.0]], index=idx, columns=symbols),
            adv_panel=pd.DataFrame([[self.ADV]], index=idx, columns=symbols),
            vol_panel=pd.DataFrame([[self.SIGMA]], index=idx, columns=symbols),
            fee_rate=0.0, impact_coeff=self.COEFF,
            v_at_bar_open_history=pd.Series([self.V], index=idx),
        )["impact"].iloc[0, 0]

        # 两两 rtol=1e-12
        np.testing.assert_allclose([p1, p2, p3], p4, rtol=1e-12,
            err_msg=f"impact 四条路径不一致：p1={p1}, p2={p2}, p3={p3}, p4={p4}")
