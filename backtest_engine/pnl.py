"""
PnLTracker — P&L 状态管理中心

详见 docs/phase3_design.md §11.5。

唯一持有 portfolio_value 的模块；事件循环中"时序敏感状态"的集中地。

核心设计选择（§11.5.2）:
  A. 两阶段存储（E.2 混合模式落地）：循环内时序记录 + 循环后向量化产出
  B. 精确 V 更新（方案 Z）：V_t = V_{t-1} × (1 + gross − funding − cost)
     funding 实时扣 V（保持事件语义），record 用 V_before_funding 做基准
  C. funding 记录扣款率（无量纲，符号约定：正=扣款，负=收款）
  D. ExecutionReport 不全量留存，仅保留 fee/spread/impact 紧凑 series
  E. gross_returns 循环后向量化重算（与循环内的临时计算一致）
  F. 破产后 equity_curve 截断到 [start, t*]
  G. 不持有 price_panel 引用（compute_backtest_result 时通过参数传入）

破产 / 数值异常分流（§11.5.5 v3 修订）:
  - V ≤ 0（finite，真实策略亏光）→ is_bankrupt 标志 + bankruptcy_timestamp 字段，
    engine break；equity_curve 截断保留诊断价值
  - V 是 NaN / Inf（数据/公式 bug）→ 抛 NumericalError，立即终止事件循环
"""
from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from alpha_model.backtest.performance import BacktestResult

from backtest_engine.rebalancer import ExecutionReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 模块级异常
# ---------------------------------------------------------------------------

class NumericalError(RuntimeError):
    """
    V 变为 NaN / Inf。表示数据 / 公式 bug，不是策略问题。

    必须 raise 而非标志位记录——这是 fail-fast：
      - 静默继续会让所有后续 bar 的 V/equity 都是 NaN，用户毫无察觉
      - 抛异常立即中止事件循环，要求用户修数据 / 公式后重跑

    与"V≤0 真实破产"区分（后者用 is_bankrupt 标志 + break 表达）
    """


# ---------------------------------------------------------------------------
# PnLTracker
# ---------------------------------------------------------------------------

class PnLTracker:
    """
    P&L 状态管理中心。事件循环每 bar 调用 record；funding 事件时调 apply_funding_settlement；
    循环结束后调 compute_backtest_result 产出 BacktestResult。
    """

    def __init__(self, initial_portfolio_value: float) -> None:
        if initial_portfolio_value <= 0:
            raise ValueError(
                f"initial_portfolio_value 必须 > 0，收到 {initial_portfolio_value}"
            )

        self.initial_V = float(initial_portfolio_value)

        # 运行时状态
        self._V: float = self.initial_V
        self._v_before_funding: float = self.initial_V
        # 显式标志（v2 修订）：替代 "t in _funding_events" 隐式判断
        self._funding_applied_at_t: pd.Timestamp | None = None

        self._prev_actual_w: pd.Series | None = None
        self._prev_price: pd.Series | None = None

        self._is_bankrupt: bool = False
        self._bankruptcy_timestamp: pd.Timestamp | None = None

        # 时间序列（dict[t, value]，循环后向量化）
        self._weights_history: dict[pd.Timestamp, pd.Series] = {}
        self._fee_series: dict[pd.Timestamp, float] = {}
        self._spread_series: dict[pd.Timestamp, float] = {}
        self._impact_series: dict[pd.Timestamp, float] = {}
        self._portfolio_value_history: dict[pd.Timestamp, float] = {}
        # Step 4 / M2：record() 入口时的 V（即 Rebalancer 用于 impact 公式 √(V/ADV) 的 V）
        # 用于 compute_per_symbol_cost 精确重算 per-symbol impact，与运行时严格一致。
        # （funding 应用后、record 修改 V 之前的 V，等价于"本 bar Rebalancer 看到的 context.V"）
        self._v_at_bar_open: dict[pd.Timestamp, float] = {}

        # 离散事件 / 统计
        self._funding_events: dict[pd.Timestamp, float] = {}
        self._filter_statistics: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # 循环内接口
    # ------------------------------------------------------------------

    def apply_funding_settlement(
        self,
        t: pd.Timestamp,
        current_weights: pd.Series,
        funding_rates: pd.Series,
    ) -> None:
        """
        Funding 事件：按真实 8h 时间戳触发，立即扣 V

        Args:
            t:                结算时间戳
            current_weights:  本 bar 决策前的实际持仓（即上一步 actual_w）
            funding_rates:    真实 8h 费率（不是摊销值）
        """
        # 选择 B：扣款前先存基准
        # （正常情况下 record 末尾已重置 _v_before_funding == _V，但显式覆盖保证 invariant）
        self._v_before_funding = self._V

        # N1-warning（v3）：检测 NaN 时显式警告，避免静默兜底掩盖上游数据 gap
        if funding_rates.isna().any():
            nan_syms = list(funding_rates.index[funding_rates.isna()])
            logger.warning(
                "apply_funding_settlement: funding_rates 含 NaN（symbols: %s），已视为 0；"
                "可能是上游数据 gap，请检查 DataReader.get_funding_rate 输出",
                nan_syms,
            )

        # 对齐 weights 与 rates 到相同 index
        # N1（v3）：显式 fillna(0) 视为"该 symbol 无 funding 数据 = 该 symbol 无 funding 事件"，
        # 不依赖 pandas 默认 skipna 行为；sum(skipna=False) 让任何意外 NaN（aligned_w 残留）暴露
        aligned_w = current_weights.reindex(funding_rates.index).fillna(0.0).astype(float)
        aligned_rates = funding_rates.fillna(0.0).astype(float)

        funding_rate_total = float((aligned_w * aligned_rates).sum(skipna=False))
        self._V *= (1.0 - funding_rate_total)
        self._funding_events[t] = funding_rate_total
        self._funding_applied_at_t = t

        self._check_bankruptcy(t)

    def record(
        self,
        t: pd.Timestamp,
        actual_weights: pd.Series,
        price_at_t: pd.Series,
        exec_report: ExecutionReport,
    ) -> None:
        """每 bar 末尾调用：算 gross + cost + funding，更新 V，记录所有时序"""
        # ── Step 0 (M2 / Step 4 v6): 记录 record() 入口时的 V ──
        # 此时 self._V 已被 funding 应用过（如果有），但还未被本 bar gross/cost 改动
        # → 与 Rebalancer/Optimizer 在 (b)-(d) 期间看到的 context.V 一致
        # 用于 compute_per_symbol_cost 精确重算 per-symbol impact
        self._v_at_bar_open[t] = self._V

        # ── Step 1: 算本步 gross ──
        if self._prev_actual_w is None:
            gross_rate = 0.0
        else:
            # 对齐到 actual_weights.index（current period 的 symbols）
            common_syms = self._prev_actual_w.index
            prev_price = self._prev_price.reindex(common_syms)
            curr_price = price_at_t.reindex(common_syms)
            returns = curr_price / prev_price - 1.0
            # A1（v3）：sum(skipna=False) 让 price NaN → returns NaN → gross NaN → V NaN
            # → _check_bankruptcy 抛 NumericalError；防"NaN 被默认 skipna 静默吃为 0" silent error
            gross_rate = float((self._prev_actual_w * returns).sum(skipna=False))

        # ── Step 2: 本步 cost rate ──
        cost_rate = (
            exec_report.fee_cost + exec_report.spread_cost + exec_report.impact_cost
        )

        # ── Step 3: 本步 funding rate（若本 bar 有事件）──
        # v2 修订：用显式标志替代 "t in dict" 判断
        has_funding_this_bar = (self._funding_applied_at_t == t)
        funding_rate = self._funding_events.get(t, 0.0) if has_funding_this_bar else 0.0

        # ── Step 4: 精确更新 V（选择 B 方案 Z）──
        base_V = self._v_before_funding if has_funding_this_bar else self._V
        self._V = base_V * (1.0 + gross_rate - funding_rate - cost_rate)
        self._v_before_funding = self._V       # 重置：下次 funding 前保持 == _V
        self._funding_applied_at_t = None      # 清零显式标志

        # ── Step 5: 破产检测 ──
        self._check_bankruptcy(t)

        # ── Step 6: 记录（选择 D：紧凑 series）──
        self._weights_history[t] = actual_weights.copy()
        self._fee_series[t] = float(exec_report.fee_cost)
        self._spread_series[t] = float(exec_report.spread_cost)
        self._impact_series[t] = float(exec_report.impact_cost)
        self._portfolio_value_history[t] = self._V
        for sym in exec_report.filtered_symbols:
            self._filter_statistics[sym] += 1

        # ── Step 7: 更新 prev ──
        self._prev_actual_w = actual_weights.copy()
        self._prev_price = price_at_t.copy()

    # ------------------------------------------------------------------
    # 破产 / 数值异常检测（§11.5.5 v3 修订）
    # ------------------------------------------------------------------

    def _check_bankruptcy(self, t: pd.Timestamp) -> None:
        """
        优先检测数值异常（NaN/Inf）→ raise NumericalError；
        然后检测真实破产（V ≤ 0 且 finite）→ 标志 + 字段 + warning（不抛）
        """
        if not np.isfinite(self._V):
            raise NumericalError(
                f"PnLTracker: V 变为 {self._V} at t={t}。"
                f"可能原因：context.spread / funding_rate / adv 含 NaN，或 impact 公式发散。"
                f"检查上游数据完整性 / context_builder 热身期是否充分。"
            )
        if self._V <= 0 and not self._is_bankrupt:
            self._is_bankrupt = True
            self._bankruptcy_timestamp = t
            logger.warning(
                "PnLTracker: 破产事件，t=%s, V=%.2f", t, self._V,
            )

    # ------------------------------------------------------------------
    # 运行时查询 properties
    # ------------------------------------------------------------------

    @property
    def portfolio_value(self) -> float:
        return self._V

    @property
    def is_bankrupt(self) -> bool:
        return self._is_bankrupt

    @property
    def bankruptcy_timestamp(self) -> pd.Timestamp | None:
        return self._bankruptcy_timestamp

    # ------------------------------------------------------------------
    # 循环后产出（compute_backtest_result）
    # ------------------------------------------------------------------

    def compute_backtest_result(
        self,
        price_panel: pd.DataFrame,
        periods_per_year: float,
    ) -> BacktestResult:
        """
        循环结束后向量化产出 BacktestResult（与 Phase 2b 类型 100% 字段兼容）

        Args:
            price_panel:      timestamp × symbol close 价（含 weights_history 的全部 t）
            periods_per_year: 年化系数（用于 BacktestResult.summary 的默认值）
        """
        del periods_per_year  # BacktestResult.summary 自己接 periods_per_year，这里不用

        if len(self._weights_history) == 0:
            raise RuntimeError("PnLTracker 未收到任何 record，无法产出 BacktestResult")

        weights_df = pd.DataFrame(self._weights_history).T.sort_index()
        symbols = weights_df.columns
        idx = weights_df.index

        # 向量化算 gross
        # 对齐 price_panel 到 weights_history 的 index + columns
        prices = price_panel.reindex(idx)[list(symbols)]
        # A1（v3）：returns 不全 fillna(0)；首 bar 的 NaN（pct_change 首行）单独 = 0；
        # 其他 NaN（来自 price 数据 gap）保留 → sum(skipna=False) 让 NaN 暴露
        returns = prices.pct_change()
        if len(returns) > 0:
            returns.iloc[0] = 0.0
        # shift(1, fill_value=0.0) 让首行用 0（精确语义，不靠 fillna）
        shifted_w = weights_df.shift(1, fill_value=0.0)
        gross_series = (shifted_w * returns).sum(axis=1, skipna=False)

        # 成本 + funding
        fee_s = pd.Series(self._fee_series).reindex(idx, fill_value=0.0)
        spread_s = pd.Series(self._spread_series).reindex(idx, fill_value=0.0)
        impact_s = pd.Series(self._impact_series).reindex(idx, fill_value=0.0)
        cost_series = fee_s + spread_s + impact_s

        if len(self._funding_events) > 0:
            funding_full = pd.Series(self._funding_events).reindex(idx, fill_value=0.0)
        else:
            funding_full = pd.Series(0.0, index=idx)

        # 净收益（funding 独立项）
        net_series = gross_series - cost_series - funding_full

        # 净值曲线（基础 = initial_V，1.0 起始）
        # 选择 F：破产则截断到 bankruptcy_timestamp（已经包含在 idx 中）
        equity_normalized = (1.0 + net_series).cumprod()
        equity_curve = equity_normalized * self.initial_V

        # 换手率
        turnover = weights_df.diff().abs().sum(axis=1)

        return BacktestResult(
            equity_curve=equity_curve,
            returns=net_series,
            turnover=turnover,
            weights_history=weights_df,
            gross_returns=gross_series,
            total_cost=float(cost_series.sum() + funding_full.sum()),
        )

    # ------------------------------------------------------------------
    # attribution 读取接口（§11.5.12）
    # ------------------------------------------------------------------

    @property
    def fee_series(self) -> pd.Series:
        """每 bar 手续费率"""
        return pd.Series(self._fee_series, dtype=float).sort_index()

    @property
    def spread_series(self) -> pd.Series:
        """每 bar 价差成本率"""
        return pd.Series(self._spread_series, dtype=float).sort_index()

    @property
    def impact_series(self) -> pd.Series:
        """每 bar 市场冲击率"""
        return pd.Series(self._impact_series, dtype=float).sort_index()

    @property
    def funding_events(self) -> pd.Series:
        """每个 funding 事件的扣款率（index=结算 t；非每 bar）"""
        return pd.Series(self._funding_events, dtype=float).sort_index()

    @property
    def filter_statistics(self) -> dict[str, int]:
        """symbol → 被 min_trade_value 过滤的次数"""
        return dict(self._filter_statistics)

    @property
    def portfolio_value_history(self) -> pd.Series:
        """每 bar 末尾 V"""
        return pd.Series(self._portfolio_value_history, dtype=float).sort_index()

    @property
    def v_at_bar_open_history(self) -> pd.Series:
        """每 bar 开始时（funding 应用后、record 改动前）的 V，
        即 Rebalancer/Optimizer 用于 impact 公式 √(V/ADV) 的 V。

        Step 4 / M2：让 compute_per_symbol_cost 用此 series 精确重算 per-symbol impact，
        与运行时 portfolio total 严格一致（rtol=1e-12）。
        """
        return pd.Series(self._v_at_bar_open, dtype=float).sort_index()
