"""
Rebalancer — 执行模拟层

详见 docs/phase3_design.md §11.4。

职责：
  - 拿到决策层产出的 target_w，按"执行现实"产出 actual_w
  - 最小下单量过滤（选择 A）
  - 计算本步三分量成本（fee + spread + impact）—— 选择 C 用 actual_delta
  - 产出 ExecutionReport 供 PnLTracker 消费

不做：
  - 不做交易决策（不判断 target_w 合不合理、不重新优化）
  - 不做破产检查（归 PnLTracker）

cost_mode（v1 修订，§11.4.4）:
  MATCH_VECTORIZED 下 spread_cost = 0.0（与 vectorized_backtest(spread_panel=None) 对齐）
  FULL_COST 下计算 spread/2 × |Δw|

成交价方向性（选择 B）：
  买入吃 ask（成交价 = close × (1 + spread/2)），卖出吃 bid。
  对总 P&L 的扣除收敛到 spread_cost = Σ(spread_i/2 × |Δw_i|)。

impact 公式（选择 D，与 cost.py / vectorized.py 一致）：
  (2/3) × coeff × σ × √(V/ADV) × |Δw|^1.5
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from execution_optimizer import MarketContext

from backtest_engine.config import ExecutionMode, CostMode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ExecutionReport (§11.4.0' canonical schema)
# ---------------------------------------------------------------------------

@dataclass
class ExecutionReport:
    """
    执行结果。Rebalancer.execute 返回，PnLTracker.record 消费。

    Schema 字段集冻结（§11.4.0'，§12.1）；任何字段增删视为 schema breaking change。
    """
    timestamp: pd.Timestamp
    """与事件循环 t 一致；非 None"""

    actual_delta: pd.Series
    """index=symbols（严格 == Rebalancer 声明 symbols），actual_w − current_w，dtype=float64"""

    trade_values: pd.Series
    """index=symbols, |actual_delta| × portfolio_value，元素必 ≥ 0"""

    fee_cost: float
    """收益率空间，≥ 0（不是 USDT）"""

    spread_cost: float
    """收益率空间，≥ 0；MATCH_VECTORIZED 模式恒 0"""

    impact_cost: float
    """收益率空间，≥ 0（funding 不在此处）"""

    filtered_symbols: list[str]
    """被 min_trade_value 过滤的 symbol（list，可空）"""


# ---------------------------------------------------------------------------
# Rebalancer
# ---------------------------------------------------------------------------

class Rebalancer:
    """
    v1 只实现 MARKET（单类 + 私有方法分派，选择 E）

    v2 扩展路径（§11.4.10）：升级为基类，新增 LimitRebalancer / TWAPRebalancer。
    """

    def __init__(
        self,
        execution_mode: ExecutionMode,
        cost_mode: CostMode,
        min_trade_value: float,
        fee_rate: float,
        impact_coeff: float | pd.Series,
    ) -> None:
        """
        Args:
            execution_mode:   v1 仅 MARKET
            cost_mode:        FULL_COST / MATCH_VECTORIZED（决定是否计 spread_cost）
            min_trade_value:  USDT 阈值；调仓金额低于此则放弃
            fee_rate:         taker 手续费率
            impact_coeff:     sqrt-model 校准系数；float 或 pd.Series(index=symbols)
        """
        if execution_mode != ExecutionMode.MARKET:
            raise NotImplementedError(
                f"v1 仅支持 ExecutionMode.MARKET，收到 {execution_mode}"
            )
        if min_trade_value <= 0:
            raise ValueError(f"min_trade_value 必须 > 0，收到 {min_trade_value}")
        if fee_rate < 0:
            raise ValueError(f"fee_rate 必须 >= 0，收到 {fee_rate}")

        self.execution_mode = execution_mode
        self._cost_mode = cost_mode
        self._min_trade_value = min_trade_value
        self._fee_rate = fee_rate
        self._impact_coeff = impact_coeff

        # Z16 dedup 集合：事件循环每 bar 调用 _execute_market，若 ADV 持续 NaN 会 log spam；
        # 此 set 让"首次出现"的 NaN symbol 才 warning。
        self._adv_nan_warned: set[str] = set()

    def execute(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        context: MarketContext,
        price_at_t: pd.Series,  # 预留 v2 LIMIT/TWAP；v1 不用
    ) -> tuple[pd.Series, ExecutionReport]:
        """单 bar 执行模拟"""
        del price_at_t  # v1 MARKET 不消费

        if self.execution_mode == ExecutionMode.MARKET:
            return self._execute_market(current_weights, target_weights, context)
        raise NotImplementedError(f"{self.execution_mode} 将在 v2 支持")

    def _execute_market(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        context: MarketContext,
    ) -> tuple[pd.Series, ExecutionReport]:
        """
        MARKET 撮合的伪代码（§11.4.3）:
          1. 最小下单量过滤（选择 A）
          2. 计算 actual_delta 和 trade_values
          3. 三分量成本（基于 actual_delta，选择 C + D）
          4. 封装 ExecutionReport
        """
        symbols = list(context.symbols)
        V = context.portfolio_value

        # 对齐到 context.symbols（current/target 缺失视为 0）
        cw = current_weights.reindex(symbols).fillna(0.0).astype(float)
        tw = target_weights.reindex(symbols).fillna(0.0).astype(float)

        # ── Step 1: 最小下单量过滤（选择 A，§11.4.2 A）──
        delta_target = tw - cw
        trade_value_target = delta_target.abs() * V

        # 过滤条件：调仓金额 < min_trade_value 时 actual = current（放弃这个 symbol 的调仓）
        below_min = trade_value_target < self._min_trade_value
        actual_w = tw.copy()
        actual_w[below_min] = cw[below_min]
        filtered_symbols = list(actual_w.index[below_min])

        # ── Step 2: actual_delta（本步真实换手）──
        actual_delta = actual_w - cw
        abs_delta = actual_delta.abs()
        trade_values = abs_delta * V

        # ── Step 3: 三分量成本（基于 actual_delta，选择 C + D）──
        spread_arr = context.spread.reindex(symbols).values.astype(float)
        sigma_arr = context.volatility.reindex(symbols).values.astype(float)
        adv_arr = context.adv.reindex(symbols).values.astype(float)
        delta_arr = actual_delta.values.astype(float)
        abs_delta_arr = abs_delta.values.astype(float)

        # 逐标的 impact_coeff
        if isinstance(self._impact_coeff, pd.Series):
            coeff_arr = self._impact_coeff.reindex(symbols).values.astype(float)
        else:
            coeff_arr = np.full(len(symbols), float(self._impact_coeff))

        # ① 手续费：fee_rate × Σ|Δw|
        fee_cost = float(self._fee_rate * abs_delta_arr.sum())

        # ② spread（cost_mode 决定）
        if self._cost_mode == CostMode.MATCH_VECTORIZED:
            spread_cost = 0.0
        else:
            spread_cost = float(np.sum(spread_arr / 2.0 * abs_delta_arr))

        # ③ impact: (2/3) × Σ(coeff × σ × √(V/ADV) × |Δw|^1.5)
        # ADV 兜底（Z4/Z12 跨四处统一 + Z16 dedup；修 pre-existing np.maximum(NaN,1)=NaN bug）
        from alpha_model.backtest.adv_helpers import safe_adv_array
        adv_safe = safe_adv_array(
            adv_arr, symbols,
            context="Rebalancer._execute_market",
            warned_set=self._adv_nan_warned,
        )
        sqrt_ratio = np.sqrt(V / adv_safe)
        impact_per_sym = (
            (2.0 / 3.0) * coeff_arr * sigma_arr * sqrt_ratio * np.power(abs_delta_arr, 1.5)
        )
        impact_cost = float(np.sum(impact_per_sym))

        # ── Step 4: 封装 ExecutionReport ──
        report = ExecutionReport(
            timestamp=context.timestamp,
            actual_delta=pd.Series(delta_arr, index=symbols, dtype=float),
            trade_values=pd.Series(trade_values.values, index=symbols, dtype=float),
            fee_cost=fee_cost,
            spread_cost=spread_cost,
            impact_cost=impact_cost,
            filtered_symbols=filtered_symbols,
        )

        return actual_w, report
