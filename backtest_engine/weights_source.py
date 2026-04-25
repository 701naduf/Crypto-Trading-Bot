"""
WeightsSource Protocol + 两个实现

详见 docs/phase3_design.md §11.3。

把 FIXED_GAMMA / DYNAMIC_COST 在事件循环中的差异（target_w 怎么来）收敛到一个 Protocol。

实现：
  - PrecomputedWeights：FIXED_GAMMA 模式，忠实重放 Phase 2b 决策路径
  - OnlineOptimizer:    DYNAMIC_COST 模式，每步实时调 ExecutionOptimizer.optimize_step

cost_mode 由消费方各自持有（v1 修订，§11.6.2 选择 C）：
  - PrecomputedWeights 不感知 cost_mode（不调 optimizer）
  - OnlineOptimizer 在 MATCH_VECTORIZED 下把 context.spread 清零后再喂 optimizer
"""
from __future__ import annotations

from dataclasses import replace
from typing import Protocol, runtime_checkable

import pandas as pd

from execution_optimizer import ExecutionOptimizer, MarketContext

from backtest_engine.config import CostMode


# ---------------------------------------------------------------------------
# Protocol（§11.3.3）
# ---------------------------------------------------------------------------

@runtime_checkable
class WeightsSource(Protocol):
    """
    决策层抽象：事件引擎对"target_w 来源"中立。

    实现必须接受统一签名（见 §11.3.3）；简单实现容忍多余参数是架构一致性的代价。
    返回值：
      - pd.Series, index = context.symbols（顺序可不同但集合必须一致）
      - dtype = float64
      - 不允许 NaN / inf（实现内部应 fallback）
    """

    def get_target_weights(
        self,
        t: pd.Timestamp,
        current_weights: pd.Series,
        context: MarketContext,
        price_history: pd.DataFrame,
    ) -> pd.Series: ...


# ---------------------------------------------------------------------------
# 实现 1：PrecomputedWeights（FIXED_GAMMA）
# ---------------------------------------------------------------------------

class PrecomputedWeights:
    """
    FIXED_GAMMA 模式：忠实重放 Phase 2b 决策

    给定预计算 weights 序列，按 t 查面板返回 target_w。**忽略** current_w / context /
    price_history（§11.3.2 如实重放语义）：要"基于当前持仓重新求最优"应改用 DYNAMIC_COST。
    """

    def __init__(self, weights_panel: pd.DataFrame) -> None:
        """
        Args:
            weights_panel: timestamp × symbol 面板。**engine 已切片到 [config.start, end]**。

        Schema 约束（§11.3.4）:
            - 非空
            - index = pd.DatetimeIndex (unique, tz-aware)
            - columns = list[str] (flat 单层 index，不允许 MultiIndex)
            - dtype = float64
        """
        self._validate_panel(weights_panel)
        self._panel = weights_panel

    @staticmethod
    def _validate_panel(panel: pd.DataFrame) -> None:
        if panel is None or len(panel) == 0:
            raise ValueError("weights_panel 不能为空")
        if not isinstance(panel.index, pd.DatetimeIndex):
            raise ValueError(
                f"weights_panel.index 必须是 DatetimeIndex，收到 {type(panel.index).__name__}"
            )
        # Step 13 / C3: tz-aware 校验（防 naive index 与 tz-aware bar_timestamps 静默漏匹配）
        if panel.index.tz is None:
            raise ValueError(
                "weights_panel.index 必须 tz-aware（推荐 tz='UTC'）；"
                "naive index 与 tz-aware bar_timestamps 比较会静默漏匹配 → KeyError or 漏 bar"
            )
        if not panel.index.is_unique:
            raise ValueError("weights_panel.index 必须 unique（不允许重复 timestamp）")
        if isinstance(panel.columns, pd.MultiIndex):
            raise ValueError("weights_panel.columns 不允许 MultiIndex（必须 flat 单层）")

    def get_target_weights(
        self,
        t: pd.Timestamp,
        current_weights: pd.Series,
        context: MarketContext,
        price_history: pd.DataFrame,
    ) -> pd.Series:
        """忠实重放：只用 t；忽略其他参数（§11.3.2）"""
        del current_weights, price_history  # 显式声明不用

        if t not in self._panel.index:
            raise KeyError(
                f"weights_panel 中找不到 timestamp={t}（严格语义，不 ffill）"
            )

        symbols = list(context.symbols)
        missing = set(symbols) - set(self._panel.columns)
        if missing:
            raise KeyError(
                f"weights_panel.columns 不包含全部 symbols：缺 {missing}"
            )

        # 缺失的 symbol（其 weights 为 NaN）填 0.0，预计算阶段缺失视为"未进入组合"
        target = self._panel.loc[t].reindex(symbols).fillna(0.0).astype(float)
        return target


# ---------------------------------------------------------------------------
# 实现 2：OnlineOptimizer（DYNAMIC_COST）
# ---------------------------------------------------------------------------

class OnlineOptimizer:
    """
    DYNAMIC_COST 模式：每步实时求解 ExecutionOptimizer.optimize_step

    cost_mode（v1 修订，§11.3.5）:
      MATCH_VECTORIZED 下把 context.spread 清零后再喂 optimizer，与 vectorized 对齐。
      FULL_COST 下直接透传 raw context。
    """

    def __init__(
        self,
        optimizer: ExecutionOptimizer,
        signals_panel: pd.DataFrame,
        cost_mode: CostMode,
    ) -> None:
        """
        Args:
            optimizer:     已构造的 Phase 2c 实例（engine 负责构造，此处只负责调用）
            signals_panel: timestamp × symbol 信号面板，已被 engine 切片到 [start, end]
            cost_mode:     CostMode.FULL_COST / MATCH_VECTORIZED
        """
        if signals_panel is None or len(signals_panel) == 0:
            raise ValueError("signals_panel 不能为空")
        if not isinstance(signals_panel.index, pd.DatetimeIndex):
            raise ValueError(
                f"signals_panel.index 必须是 DatetimeIndex，收到 {type(signals_panel.index).__name__}"
            )
        if not isinstance(optimizer, ExecutionOptimizer):
            raise TypeError(
                f"optimizer 必须是 ExecutionOptimizer 实例，收到 {type(optimizer).__name__}"
            )

        self._optimizer = optimizer
        self._signals = signals_panel
        self._cost_mode = cost_mode

        # Z16 dedup 集合：事件循环每 bar 调 optimize_step → cost.py，若 ADV 持续 NaN
        # 会 log spam；此 set 让"首次出现"的 NaN symbol 才 warning。
        self._adv_nan_warned: set[str] = set()

    def get_target_weights(
        self,
        t: pd.Timestamp,
        current_weights: pd.Series,
        context: MarketContext,
        price_history: pd.DataFrame,
    ) -> pd.Series:
        """每步调 optimize_step；MATCH_VECTORIZED 下先清零 spread"""
        if t not in self._signals.index:
            raise KeyError(
                f"signals_panel 中找不到 timestamp={t}（严格语义，不 ffill）"
            )

        symbols = list(context.symbols)
        signal_t = self._signals.loc[t].reindex(symbols)

        if self._cost_mode == CostMode.MATCH_VECTORIZED:
            # 清零 spread；保留其他字段（vol/adv/funding 仍由 optimizer 使用）
            ctx_for_opt = replace(
                context, spread=pd.Series(0.0, index=symbols, dtype=float),
            )
        else:
            ctx_for_opt = context

        return self._optimizer.optimize_step(
            signals_t=signal_t,
            current_weights=current_weights,
            market_context=ctx_for_opt,
            price_history=price_history,
            adv_nan_warned=self._adv_nan_warned,
        )
