"""
EventDrivenBacktester — Phase 3 顶层协调器

详见 docs/phase3_design.md §11.6。

职责：把 §11.1–§11.5 / §11.7–§11.8 各子模块串成完整回测流程，对外暴露唯一入口 run(config)。

  1. _validate_environment：8 项前置校验（含 v3 修订的 #6/#7/#8 tz/index/freq 校验）
  2. _build_dependencies:  按 BacktestConfig 实例化所有子模块
  3. 按 run_mode 分流：
        VECTORIZED              → _run_vectorized (薄包装 vectorized_backtest)
        EVENT_DRIVEN_*          → _run_event_driven (主循环 + funding + 破产终止)
  4. cost_mode 由消费方各自持有（v1 修订）：engine 不 mutate context

设计要点：
  - 无状态 engine（选择 A）
  - 异常 fail-fast（选择 G）：BacktestConfig / SignalStore / DataReader 错误直接抛
  - 破产分两个时机（选择 D v3 修订）：funding 后早退 + record 后破产终止
  - optimize_every_n_bars 加速（选择 E）：仅 DYNAMIC_COST 跳过 weights_source
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data_infra.data.reader import DataReader
from alpha_model.store.signal_store import SignalStore
from alpha_model.backtest.performance import BacktestResult
from alpha_model.backtest.vectorized import vectorized_backtest
from execution_optimizer import ExecutionOptimizer

from backtest_engine import attribution
from backtest_engine.config import (
    BacktestConfig, RunMode, ExecutionMode, CostMode,
)
from backtest_engine.context import MarketContextBuilder, _to_pd_freq


def _normalize_freq(s: str | None) -> str | None:
    """归一化 pd freq 字符串：'min' / '1min' / '1T' 都 → 'min' (M3)

    pandas 2.2+ 下 pd.infer_freq() 返回 'min'，但 _to_pd_freq("1m") 返回 '1min'，
    字面比较会误报不一致。to_offset(...).freqstr 双侧归一化解决此问题。
    """
    if s is None:
        return None
    try:
        from pandas.tseries.frequencies import to_offset
        return to_offset(s).freqstr
    except (ValueError, TypeError):
        return None
from backtest_engine.pnl import PnLTracker
from backtest_engine.rebalancer import Rebalancer
from backtest_engine.report import BacktestReport, SCHEMA_VERSION
from backtest_engine.weights_source import (
    WeightsSource, PrecomputedWeights, OnlineOptimizer,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 内部依赖容器（§11.6.2 选择 B）
# ---------------------------------------------------------------------------

@dataclass
class _BacktestDependencies:
    """私有依赖容器；engine 装配后传给 _run_* 方法"""
    config: BacktestConfig
    reader: DataReader
    context_builder: MarketContextBuilder
    bar_timestamps: pd.DatetimeIndex
    price_panel: pd.DataFrame
    funding_rates_panel: pd.DataFrame   # 原始 8h 费率面板（区别于 PnLTracker.funding_events）

    # 事件驱动模式独有
    weights_source: WeightsSource | None = None
    rebalancer: Rebalancer | None = None
    pnl_tracker: PnLTracker | None = None
    optimizer: ExecutionOptimizer | None = None


# ---------------------------------------------------------------------------
# EventDrivenBacktester
# ---------------------------------------------------------------------------

class EventDrivenBacktester:
    """Phase 3 顶层。无状态，多次 run 等价于新建实例。"""

    def __init__(self) -> None:
        pass  # 无构造参数

    def run(
        self,
        config: BacktestConfig,
        *,
        progress: bool = False,
        # 测试 / 替换：允许注入合成 reader（默认用真 DataReader）
        reader: DataReader | None = None,
        signal_store: SignalStore | None = None,
    ) -> BacktestReport:
        """
        顶层入口

        Args:
            config:       BacktestConfig
            progress:     True → 用 tqdm 显示进度条；False → logger 进度（每 10000 bar）
            reader:       注入点（测试用）；None 则用 DataReader()
            signal_store: 注入点（测试用）；None 则用 SignalStore()
        """
        t0 = time.monotonic()

        # 默认实例化（注入点优先）
        if reader is None:
            reader = DataReader()
        if signal_store is None:
            signal_store = SignalStore()

        self._validate_environment(config, reader=reader, signal_store=signal_store)
        deps = self._build_dependencies(config, reader=reader, signal_store=signal_store)

        if config.run_mode == RunMode.VECTORIZED:
            return self._run_vectorized(config, deps, signal_store=signal_store, t0=t0)
        return self._run_event_driven(config, deps, progress=progress, t0=t0)

    # ------------------------------------------------------------------
    # _validate_environment（§11.6.9，含 v3 修订）
    # ------------------------------------------------------------------

    def _validate_environment(
        self, config: BacktestConfig, *, reader: DataReader, signal_store: SignalStore,
    ) -> None:
        """前置校验：尽早暴露环境问题，不让错误流到事件循环"""
        # 1. SignalStore 中 strategy 存在
        if not signal_store.exists(config.strategy_name):
            raise FileNotFoundError(
                f"SignalStore 中没有策略 '{config.strategy_name}'；"
                f"请先用 alpha_model 训练并保存"
            )

        # 2. SignalStore 中 weights/signals 的 columns ⊇ symbols
        if config.run_mode == RunMode.EVENT_DRIVEN_DYNAMIC_COST:
            panel = signal_store.load_signals(config.strategy_name)
        else:
            panel = signal_store.load_weights(config.strategy_name)

        missing = set(config.symbols) - set(panel.columns)
        if missing:
            raise KeyError(
                f"SignalStore 缺 symbols：{missing}；"
                f"strategy 覆盖范围与 config.symbols 不匹配"
            )

        # 3. SignalStore 时段覆盖
        if not (panel.index.min() <= config.start and panel.index.max() >= config.end):
            raise ValueError(
                f"SignalStore 时段不足：strategy {panel.index.min()} ~ {panel.index.max()}, "
                f"config 要求 {config.start} ~ {config.end}"
            )

        # 6. 时区一致性（v3 修订）：bar_timestamps 必为 tz-aware UTC
        pd_freq = _to_pd_freq(config.bar_freq)
        bar_ts = pd.date_range(config.start, config.end, freq=pd_freq, tz="UTC")
        if bar_ts.tz is None:
            raise ValueError(
                "bar_timestamps 必须 tz-aware (UTC)；config.start/end 应为 tz-aware Timestamp"
            )

        # 7. price_panel.index ⊇ bar_timestamps（v3 修订）
        # 简化校验：只要 reader 能读到任一 symbol 的数据即可
        # Step 6 / B1 / M3：同时做 #8 freq 推断校验（用 to_offset 归一化避免 'min' vs '1min' 误报）
        from pandas.tseries.frequencies import to_offset
        expected_freq = _normalize_freq(pd_freq)
        for sym in config.symbols:
            df = reader.get_ohlcv(sym, config.bar_freq, start=config.start, end=config.end)
            if df is None or len(df) == 0:
                raise ValueError(
                    f"DataReader 没有 {sym} 在 [{config.start}, {config.end}] 的数据"
                )
            ts_col = pd.to_datetime(df["timestamp"]) if "timestamp" in df.columns else df.index
            data_ts = pd.DatetimeIndex(ts_col)
            if data_ts.tz is None:
                # 若 reader 返回 naive，将其本地化（合成 fixture 已经 tz-aware）
                data_ts = data_ts.tz_localize("UTC")
            missing_bars = bar_ts.difference(data_ts)
            if len(missing_bars) > 0:
                raise ValueError(
                    f"{sym} price 缺 {len(missing_bars)} 个 bar（前 5: "
                    f"{missing_bars[:5].tolist()}）；compute_backtest_result 时会含 NaN"
                )

            # 8. price_panel.index.freq 与 config.bar_freq 归一化后一致
            # （reviewer M3：'min' 与 '1min' 在 pandas 2.x 下用 to_offset 归一化才能正确比较）
            sample = data_ts[:100] if len(data_ts) > 100 else data_ts
            if len(sample) >= 3:    # infer_freq 至少需 3 个点
                inferred = _normalize_freq(pd.infer_freq(sample))
                if inferred is not None and expected_freq is not None and inferred != expected_freq:
                    raise ValueError(
                        f"{sym} price 推断频率 {inferred} ≠ config.bar_freq "
                        f"{expected_freq}（{config.bar_freq}）；"
                        f"impact 公式的 bars_per_day 会算错"
                    )

        # 5. funding_rate 数据覆盖（仅 DYNAMIC_COST 严格要求）
        if config.run_mode == RunMode.EVENT_DRIVEN_DYNAMIC_COST:
            fr_seen = False
            for sym in config.symbols:
                fr = reader.get_funding_rate(sym, start=config.start, end=config.end)
                if fr is not None and len(fr) > 0:
                    fr_seen = True
                    # tz 一致性（v3 修订 #6）
                    fr_ts_col = pd.to_datetime(fr["timestamp"])
                    fr_ts = pd.DatetimeIndex(fr_ts_col)
                    if fr_ts.tz is None:
                        fr_ts = fr_ts.tz_localize("UTC")
                    if fr_ts.tz != bar_ts.tz:
                        raise ValueError(
                            f"funding_rates_panel.index.tz ({fr_ts.tz}) "
                            f"与 bar_timestamps.tz ({bar_ts.tz}) 不一致；"
                            f"funding 事件会全部漏扣"
                        )
            if not fr_seen:
                raise ValueError("DYNAMIC_COST 模式需要 funding_rate 数据")

    # ------------------------------------------------------------------
    # _build_dependencies（§11.6.4）
    # ------------------------------------------------------------------

    def _build_dependencies(
        self, config: BacktestConfig, *, reader: DataReader, signal_store: SignalStore,
    ) -> _BacktestDependencies:
        """按 config 装配所有子模块"""
        # MarketContextBuilder（含 21 天热身期预加载）
        context_builder = MarketContextBuilder(reader, config)

        # 时段切片
        pd_freq = _to_pd_freq(config.bar_freq)
        bar_timestamps = pd.date_range(
            config.start, config.end, freq=pd_freq, tz="UTC",
        )

        # price_panel：用 context_builder 已加载的（避免重复读 reader）
        price_panel = context_builder._price_panel.reindex(bar_timestamps)
        # 仅取 config.symbols 列（context 可能含历史预热数据）
        price_panel = price_panel[list(config.symbols)]

        # funding_rates_panel: 真实 8h 结算面板（每 sym 一个 series 拼成 panel）
        funding_dict = {}
        for sym in config.symbols:
            try:
                df = reader.get_funding_rate(sym, start=config.start, end=config.end)
            except Exception:
                df = None
            if df is None or len(df) == 0:
                continue
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
            funding_dict[sym] = df["funding_rate"]
        funding_rates_panel = (
            pd.DataFrame(funding_dict) if funding_dict else
            pd.DataFrame(columns=config.symbols)
        )

        deps = _BacktestDependencies(
            config=config, reader=reader, context_builder=context_builder,
            bar_timestamps=bar_timestamps,
            price_panel=price_panel,
            funding_rates_panel=funding_rates_panel,
        )

        if config.run_mode == RunMode.VECTORIZED:
            return deps  # VECTORIZED 不需后续装配

        # ── 事件驱动独有 ──
        deps.pnl_tracker = PnLTracker(initial_portfolio_value=config.initial_portfolio_value)
        deps.rebalancer = Rebalancer(
            execution_mode=config.execution_mode,
            cost_mode=config.cost_mode,
            min_trade_value=config.min_trade_value,
            fee_rate=config.fee_rate,
            impact_coeff=config.impact_coeff,
        )

        if config.run_mode == RunMode.EVENT_DRIVEN_FIXED_GAMMA:
            weights_panel = signal_store.load_weights(config.strategy_name)
            sliced = weights_panel.loc[config.start:config.end]
            deps.weights_source = PrecomputedWeights(sliced)
        else:  # DYNAMIC_COST
            deps.optimizer = ExecutionOptimizer(
                constraints=config.constraints,
                impact_coeff=config.impact_coeff,
                fee_rate=config.fee_rate,
                max_participation=config.max_participation,
                periods_per_year=config.periods_per_year,
            )
            signals_panel = signal_store.load_signals(config.strategy_name)
            sliced = signals_panel.loc[config.start:config.end]
            deps.weights_source = OnlineOptimizer(
                deps.optimizer, sliced, cost_mode=config.cost_mode,
            )

        return deps

    # ------------------------------------------------------------------
    # _run_vectorized（§11.6.7）
    # ------------------------------------------------------------------

    def _run_vectorized(
        self,
        config: BacktestConfig,
        deps: _BacktestDependencies,
        *,
        signal_store: SignalStore,
        t0: float,
    ) -> BacktestReport:
        """VECTORIZED：薄包装 vectorized_backtest + 简化 attribution"""
        panels = deps.context_builder.build_panels()

        # cost_mode → spread_panel 切换（§10.3 行为映射）
        spread_arg = (
            panels["spread_panel"] if config.cost_mode == CostMode.FULL_COST else None
        )

        weights = signal_store.load_weights(config.strategy_name).loc[
            config.start:config.end
        ]
        # 对齐到 bar_timestamps
        weights = weights.reindex(deps.bar_timestamps)[list(config.symbols)]

        base_result = vectorized_backtest(
            weights=weights,
            price_panel=panels["price_panel"],
            fee_rate=config.fee_rate,
            impact_coeff=config.impact_coeff,
            adv_panel=panels["adv_panel"],
            spread_panel=spread_arg,
            vol_panel=panels["vol_panel"],     # ★ Step 1 (C5): 跨模式 σ 同源
            portfolio_value=config.initial_portfolio_value,
            periods_per_year=config.periods_per_year,
        )

        cost_breakdown_vec, cost_series_vec = self._vectorized_cost_breakdown(
            weights, panels, config,
        )

        regime_stats = (
            attribution.regime_breakdown(
                base_result, config.regime_series, config.periods_per_year,
                cost_series=cost_series_vec,
            )
            if config.regime_series is not None else None
        )

        run_metadata = {
            "run_mode": config.run_mode.value,
            "cost_mode": config.cost_mode.value,
            "execution_mode": config.execution_mode.value,
            "start": config.start, "end": config.end,
            "n_bars": int(len(weights)),
            "n_bars_planned": int(len(weights)),
            "walltime_seconds": time.monotonic() - t0,
            "schema_version": SCHEMA_VERSION,
        }

        return BacktestReport(
            base=base_result,
            config=config,
            cost_breakdown=cost_breakdown_vec,
            deviation=None,                  # VECTORIZED 自身就是 baseline
            regime_stats=regime_stats,
            funding_settlements=None,        # VECTORIZED 不处理 funding
            bankruptcy_timestamp=None,       # VECTORIZED 不检测破产
            run_metadata=run_metadata,
            context_panels=None,
        )

    def _vectorized_cost_breakdown(
        self,
        weights: pd.DataFrame,
        panels: dict,
        config: BacktestConfig,
    ) -> tuple[dict, dict]:
        """
        重算 fee/spread/impact 三分量构造 cost_breakdown + 返回 cost_series 给 regime_breakdown

        公式与 Rebalancer / cost.py / vectorized.py 一致（§11.4 选择 D）。
        """
        delta_w = weights.diff().fillna(0.0)
        abs_dw = delta_w.abs()

        # fee
        fee_per_bar = config.fee_rate * abs_dw.sum(axis=1)

        # spread（FULL_COST 才有；MATCH_VECTORIZED 下 spread_panel 仍提供，但传给 vectorized 是 None）
        if config.cost_mode == CostMode.FULL_COST:
            spread_aligned = panels["spread_panel"].reindex(weights.index)[list(weights.columns)]
            spread_per_bar = (
                spread_aligned / 2.0 * abs_dw
            ).sum(axis=1).fillna(0.0)
        else:
            spread_per_bar = pd.Series(0.0, index=weights.index)

        # impact（vectorized 用面板算）
        adv_aligned = panels["adv_panel"].reindex(weights.index)[list(weights.columns)]
        vol_aligned = panels["vol_panel"].reindex(weights.index)[list(weights.columns)]
        adv_safe = adv_aligned.where(adv_aligned >= 1.0, 1.0)
        if isinstance(config.impact_coeff, pd.Series):
            coeff_arr = config.impact_coeff.reindex(weights.columns).values.astype(float)
        else:
            coeff_arr = float(config.impact_coeff)
        sqrt_ratio = np.sqrt(config.initial_portfolio_value / adv_safe.values)
        impact_per_sym = (
            (2.0 / 3.0) * coeff_arr * vol_aligned.values * sqrt_ratio
            * np.power(abs_dw.values, 1.5)
        )
        impact_per_bar = pd.DataFrame(
            impact_per_sym, index=weights.index, columns=weights.columns,
        ).sum(axis=1).fillna(0.0)

        total_per_bar = fee_per_bar + spread_per_bar + impact_per_bar
        ppy = config.periods_per_year

        # cost_breakdown（§12.5 keys）—— funding 全 0
        absolute = {
            "fee": float(fee_per_bar.sum()),
            "spread": float(spread_per_bar.sum()),
            "impact": float(impact_per_bar.sum()),
            "funding": 0.0,
            "total": float(total_per_bar.sum()),
        }
        annualized_bp = {
            "fee":     float(fee_per_bar.mean()    * ppy * 1e4),
            "spread":  float(spread_per_bar.mean() * ppy * 1e4),
            "impact":  float(impact_per_bar.mean() * ppy * 1e4),
            "funding": 0.0,
            "total":   float(total_per_bar.mean()  * ppy * 1e4),
        }
        denom = abs(absolute["total"])
        if denom == 0.0:
            share = {k: float("nan") for k in ("fee", "spread", "impact", "funding")}
        else:
            share = {k: absolute[k] / denom for k in ("fee", "spread", "impact", "funding")}

        cost_breakdown = {
            "absolute": absolute,
            "annualized_bp": annualized_bp,
            "share": share,
        }

        cost_series = {
            "fee": fee_per_bar,
            "spread": spread_per_bar,
            "impact": impact_per_bar,
        }
        return cost_breakdown, cost_series

    # ------------------------------------------------------------------
    # _run_event_driven（§11.6.5）
    # ------------------------------------------------------------------

    def _run_event_driven(
        self,
        config: BacktestConfig,
        deps: _BacktestDependencies,
        *,
        progress: bool,
        t0: float,
    ) -> BacktestReport:
        """事件循环：funding (a) → 早退 (a') → context → 决策 → 执行 → record → 终止判断"""
        symbols = list(config.symbols)
        current_w = pd.Series(0.0, index=symbols)
        last_target_w: pd.Series | None = None

        if progress:
            try:
                from tqdm import tqdm  # type: ignore
                iter_obj = tqdm(deps.bar_timestamps)
            except ImportError:
                raise ImportError(
                    "progress=True 要求 tqdm；pip install tqdm 或 progress=False"
                )
        else:
            iter_obj = deps.bar_timestamps

        # Step 5 / A4: 独立 n_recorded 计数器（替代 i+1）；记录 record() 实际调用次数。
        # (a') 早退路径上 record() 未调用，n_recorded 不递增 → 与 weights_history 行数一致。
        n_recorded = 0
        for i, t in enumerate(iter_obj):
            # (a) Funding 事件检测与扣款
            if t in deps.funding_rates_panel.index:
                row = deps.funding_rates_panel.loc[t]
                # 同 t 多 row 的极端情况（重复结算时间戳）：取首条
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                deps.pnl_tracker.apply_funding_settlement(t, current_w, row)

            # (a') funding 触发真实破产（V≤0 finite）→ 显式 break
            # 注：v1 自然路径下 current_w=0 → funding_total=0 → 不会触发；仅 v2 启用
            if deps.pnl_tracker.is_bankrupt:
                logger.warning(
                    "funding 触发破产：t=%s, V=%s，本 bar 不再决策/执行",
                    t, deps.pnl_tracker.portfolio_value,
                )
                break

            # (b) 构造市场状态（保持 raw；cost_mode 由 Rebalancer/OnlineOptimizer 各自处理）
            context = deps.context_builder.build(
                t, current_w, deps.pnl_tracker.portfolio_value,
            )

            # (c) 决策（按 optimize_every_n_bars 控制频率，仅 DYNAMIC_COST 生效）
            is_dynamic = (config.run_mode == RunMode.EVENT_DRIVEN_DYNAMIC_COST)
            if (not is_dynamic) or (i % config.optimize_every_n_bars == 0):
                target_w = deps.weights_source.get_target_weights(
                    t, current_w, context, deps.price_panel.loc[:t],
                )
                last_target_w = target_w
            else:
                # v1 universe 固定 → reindex 是 no-op；v2 动态 universe 的兜底
                target_w = last_target_w.reindex(context.symbols, fill_value=0.0)

            # (d) 执行模拟
            actual_w, exec_report = deps.rebalancer.execute(
                current_w, target_w, context, deps.price_panel.loc[t],
            )

            # (e) 记录
            deps.pnl_tracker.record(
                t, actual_w, deps.price_panel.loc[t], exec_report,
            )
            n_recorded += 1
            current_w = actual_w

            # (f) 破产终止
            if deps.pnl_tracker.is_bankrupt:
                logger.warning(
                    "破产终止：t=%s, V=%.2f，跳过剩余 %d bar",
                    t, deps.pnl_tracker.portfolio_value,
                    len(deps.bar_timestamps) - i - 1,
                )
                break

            # (g) 进度日志
            if not progress and (i + 1) % 10000 == 0:
                logger.info(
                    "已处理 %d/%d bar, V=%.2f",
                    i + 1, len(deps.bar_timestamps), deps.pnl_tracker.portfolio_value,
                )

        # Step 5 / M5: n_recorded == 0 时的两种场景：
        #   - bar_timestamps 为空（config.start/end 在数据范围外）→ 抛 RuntimeError
        #   - 第 1 bar funding 即破产（v1 不可达，仅防御性兜底）→ 返回 sentinel report
        if n_recorded == 0:
            if len(deps.bar_timestamps) == 0:
                raise RuntimeError(
                    "bar_timestamps 为空；config.start/end 可能在数据范围外"
                )
            logger.warning(
                "首 bar t=%s 无任何 record（funding 即破产 (a') 早退）；返回 sentinel BacktestReport",
                deps.bar_timestamps[0],
            )
            return self._build_sentinel_report(config, deps, t0)

        base_result = deps.pnl_tracker.compute_backtest_result(
            deps.price_panel, config.periods_per_year,
        )
        cost_breakdown = attribution.cost_decomposition(
            deps.pnl_tracker, config.periods_per_year,
        )
        regime_stats = (
            attribution.regime_breakdown(
                base_result, config.regime_series, config.periods_per_year,
                cost_series={
                    "fee":    deps.pnl_tracker.fee_series,
                    "spread": deps.pnl_tracker.spread_series,
                    "impact": deps.pnl_tracker.impact_series,
                },
            )
            if config.regime_series is not None else None
        )

        # funding_settlements 统计（§11.8.3.1）
        funding_evt = deps.pnl_tracker.funding_events
        if len(funding_evt) > 0:
            funding_settlements = {
                "n_events": int(len(funding_evt)),
                "total_rate": float(funding_evt.sum()),
                "mean_rate_per_event": float(funding_evt.mean()),
                "first_event": funding_evt.index[0],
                "last_event": funding_evt.index[-1],
            }
        else:
            funding_settlements = None

        run_metadata = {
            "run_mode": config.run_mode.value,
            "cost_mode": config.cost_mode.value,
            "execution_mode": config.execution_mode.value,
            "start": config.start, "end": config.end,
            "n_bars": n_recorded,                  # Step 5 / A4: 用 n_recorded 替代 i+1，
                                                     # (a') 早退时不计入未 record 的 bar
            "n_bars_planned": int(len(deps.bar_timestamps)),
            "walltime_seconds": time.monotonic() - t0,
            "schema_version": SCHEMA_VERSION,
        }

        return BacktestReport(
            base=base_result,
            config=config,
            cost_breakdown=cost_breakdown,
            deviation=None,                              # 用户 post-hoc attach_deviation 填
            regime_stats=regime_stats,
            funding_settlements=funding_settlements,
            bankruptcy_timestamp=deps.pnl_tracker.bankruptcy_timestamp,
            run_metadata=run_metadata,
            context_panels=None,
        )

    def _build_sentinel_report(
        self, config: BacktestConfig, deps: _BacktestDependencies, t0: float,
    ) -> BacktestReport:
        """
        首 bar funding 即破产时返回的退化 report (Step 5 / M5 / Z2 / Z3)。

        Z2 docstring: 在 v1 当前实现下，事件循环 current_w 初始化为 pd.Series(0.0, index=symbols)。
        第 1 bar (a) 时 funding_rate_total = (0_vec × rates).sum() = 0，V 不变，is_bankrupt
        不会触发。所以 v1 路径下此 sentinel 实际不可达，仅作防御性兜底；v2 引入"非零初始
        权重"或"持仓快照恢复"时启用。

        O3 docstring: cost_breakdown.annualized_bp 全 0 是"年化无意义"占位（n_bars=0 时
        mean 数学上无定义）。用户应看 cost_breakdown.absolute['funding'] 字段获取真实
        funding 总扣款；annualized_bp 不应作为投资分析依据。
        """
        t0_bar = deps.bar_timestamps[0]
        V = deps.pnl_tracker.portfolio_value
        sentinel_idx = pd.DatetimeIndex([t0_bar])

        base = BacktestResult(
            equity_curve=pd.Series([V], index=sentinel_idx),
            returns=pd.Series([0.0], index=sentinel_idx),
            turnover=pd.Series([0.0], index=sentinel_idx),
            weights_history=pd.DataFrame(0.0, index=sentinel_idx, columns=config.symbols),
            gross_returns=pd.Series([0.0], index=sentinel_idx),
            total_cost=0.0,
        )

        funding_evt = deps.pnl_tracker.funding_events
        funding_total_rate = float(funding_evt.sum()) if len(funding_evt) > 0 else 0.0
        funding_settlements = (
            {"n_events": int(len(funding_evt)),
             "total_rate": funding_total_rate,
             "mean_rate_per_event": float(funding_evt.mean()),
             "first_event": funding_evt.index[0], "last_event": funding_evt.index[-1]}
            if len(funding_evt) > 0 else None
        )

        # Z3: 显式定义 cost_breakdown 各字段（不再用 ...）
        absolute = {
            "fee": 0.0, "spread": 0.0, "impact": 0.0,
            "funding": funding_total_rate,
            "total": funding_total_rate,
        }
        # n_recorded=0 时 mean 数学无定义；约定置 0（O3 docstring 说明）
        annualized_bp = dict.fromkeys(["fee", "spread", "impact", "funding", "total"], 0.0)
        # share: total 非零时 funding 占 ±100%；total=0 时全 NaN（与 cost_decomposition 一致）
        denom = abs(funding_total_rate)
        if denom == 0.0:
            share = {k: float("nan") for k in ("fee", "spread", "impact", "funding")}
        else:
            share = {
                "fee": 0.0, "spread": 0.0, "impact": 0.0,
                "funding": funding_total_rate / denom,    # ±1.0
            }
        cost_breakdown = {
            "absolute": absolute,
            "annualized_bp": annualized_bp,
            "share": share,
        }

        return BacktestReport(
            base=base, config=config,
            cost_breakdown=cost_breakdown,
            deviation=None, regime_stats=None,
            funding_settlements=funding_settlements,
            bankruptcy_timestamp=t0_bar,
            run_metadata={
                "run_mode": config.run_mode.value,
                "cost_mode": config.cost_mode.value,
                "execution_mode": config.execution_mode.value,
                "start": config.start, "end": config.end,
                "n_bars": 0,
                "n_bars_planned": int(len(deps.bar_timestamps)),
                "walltime_seconds": time.monotonic() - t0,
                "schema_version": SCHEMA_VERSION,
            },
            context_panels=None,
        )
