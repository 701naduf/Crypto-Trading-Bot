"""
backtest_engine 顶层配置 + 三个枚举

BacktestConfig 是 Phase 3 对外的"配置契约":
  - 参数收集: 把用户所有选择汇总到一个 dataclass
  - 参数校验: 非法组合在 __post_init__ 直接报错
  - 参数透传: 被 EventDrivenBacktester 拆解后分发给各子模块

设计要点（详见 docs/phase3_design.md §11.1）:
  - dataclass(kw_only=True): 字段 20+，强制关键字传参
  - 致命错误（影响结果正确性）→ ValueError
  - 未实现（v1 限制）→ NotImplementedError
  - 静默冗余（可能误传但无害）→ logger.warning
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import pandas as pd

from alpha_model.core.types import PortfolioConstraints
from alpha_model.config import (
    MINUTES_PER_YEAR,
    DEFAULT_FEE_RATE,
    DEFAULT_IMPACT_COEFF,
    DEFAULT_PORTFOLIO_VALUE,
)
from execution_optimizer.config import DEFAULT_MAX_PARTICIPATION

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# bar_freq translator（B4: 从 context.py 挪到 config.py，去下划线公开）
# ---------------------------------------------------------------------------

def to_pd_freq(bar_freq: str) -> str:
    """
    把项目语义 bar_freq（"1m"/"5m"/"1h"）翻译为 pandas date_range freq 字符串。

    pandas >= 2.2 起 "1m" 表示 month-end（不是 minute），必须用 "1min"。
    被 context / engine 三处共享调用。
    """
    mapping = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h"}
    return mapping.get(bar_freq, bar_freq)


# ---------------------------------------------------------------------------
# 三个枚举
# ---------------------------------------------------------------------------

class RunMode(Enum):
    """决策层模式（三种精度递增的回测路径）"""

    VECTORIZED = "vectorized"
    """快速筛选：调 alpha_model.backtest.vectorized_backtest，无事件循环"""

    EVENT_DRIVEN_FIXED_GAMMA = "event_driven_fixed_gamma"
    """事件循环 + 重放预计算 weights：评估已训练策略的执行摩擦"""

    EVENT_DRIVEN_DYNAMIC_COST = "event_driven_dynamic_cost"
    """事件循环 + 每步实时求解 ExecutionOptimizer：成本进入决策层的真实回测"""


class ExecutionMode(Enum):
    """执行层模式（v1 仅 MARKET）"""

    MARKET = "market"
    """全部 taker 单：min_trade_value 过滤 + spread/2 slippage"""

    LIMIT = "limit"
    """v2: 挂单 + 超时 fallback"""

    TWAP = "twap"
    """v2: 时间加权切片"""


class CostMode(Enum):
    """成本模型切换"""

    FULL_COST = "full_cost"
    """完整三分量（fee + spread + impact），生产回测用"""

    MATCH_VECTORIZED = "match_vectorized"
    """与 vectorized 对齐（spread_cost=0）。仅用于可比性校验"""


# ---------------------------------------------------------------------------
# BacktestConfig
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class BacktestConfig:
    """
    Phase 3 顶层回测配置

    字段分组（详见 docs/phase3_design.md §11.1.2）:
      A. 标识与范围: strategy_name / symbols / start / end / bar_freq
      B. 运行模式:   run_mode / execution_mode / cost_mode
      C. 资金管理:   initial_portfolio_value
      D. 优化参数:   constraints / impact_coeff / fee_rate / max_participation /
                    periods_per_year / optimize_every_n_bars
      E. 执行参数:   min_trade_value
      F. 可选分析:   regime_series
      G. 语义约束:   time_convention（v1 仅 bar_close，E.1 决议）
    """

    # ── A. 标识与范围 ──
    strategy_name: str
    symbols: list[str]
    start: pd.Timestamp
    end: pd.Timestamp
    bar_freq: str = "1m"

    # ── B. 运行模式 ──
    run_mode: RunMode
    execution_mode: ExecutionMode = ExecutionMode.MARKET
    cost_mode: CostMode = CostMode.FULL_COST

    # ── C. 资金管理 ──
    initial_portfolio_value: float = DEFAULT_PORTFOLIO_VALUE

    # ── D. 组合优化参数 ──
    constraints: PortfolioConstraints | None = None
    impact_coeff: float | pd.Series = DEFAULT_IMPACT_COEFF
    fee_rate: float = DEFAULT_FEE_RATE
    max_participation: float = DEFAULT_MAX_PARTICIPATION
    periods_per_year: float = MINUTES_PER_YEAR
    optimize_every_n_bars: int = 1

    # ── E. 执行参数 ──
    min_trade_value: float = 5.0

    # ── F. 可选分析 ──
    regime_series: pd.Series | None = None

    # ── G. 语义约束（E.1）──
    time_convention: Literal["bar_close"] = "bar_close"

    def __post_init__(self) -> None:
        """
        校验清单（§11.1.3，含 v3 修订的 #15 / #16 tz 校验）

        致命错误（ValueError）— 错配会导致结果错
        未实现（NotImplementedError）— v1 范围外
        静默冗余（logger.warning）— 用户可能误传但无害
        """
        # 1. DYNAMIC_COST 必须有 constraints
        if self.run_mode == RunMode.EVENT_DRIVEN_DYNAMIC_COST and self.constraints is None:
            raise ValueError(
                "DYNAMIC_COST 模式必须传入 constraints（PortfolioConstraints）；"
                "ExecutionOptimizer 依赖此字段构造目标函数与约束"
            )

        # 2. 非 DYNAMIC_COST 传 constraints 是冗余
        if self.run_mode != RunMode.EVENT_DRIVEN_DYNAMIC_COST and self.constraints is not None:
            logger.warning(
                "%s 模式不使用 constraints；该参数将被忽略", self.run_mode.name,
            )

        # 3. 时段顺序（在 tz 校验后比较，避免 naive vs tz-aware TypeError）
        # 提前 tz 校验（原 §11.1.3 #15，提到 #3 之前）：
        # 防止 pd.Timestamp 比较抛 TypeError 而非业务化的 ValueError
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError(
                "start / end 必须为 tz-aware Timestamp（推荐 tz='UTC'）；"
                "naive Timestamp 与 tz-aware funding_rates_panel.index 比较"
                "会静默漏匹配导致 funding 全部不触发"
            )
        if self.start >= self.end:
            raise ValueError(
                f"start ({self.start}) 必须严格小于 end ({self.end})"
            )

        # 4. 非空 symbols
        if len(self.symbols) == 0:
            raise ValueError("symbols 不能为空列表")

        # 5. v1 仅支持 1m bar
        if self.bar_freq != "1m":
            raise NotImplementedError(
                f"v1 仅支持 bar_freq='1m'，收到 '{self.bar_freq}'"
            )

        # 6. v1 仅支持 MARKET 执行
        if self.execution_mode != ExecutionMode.MARKET:
            raise NotImplementedError(
                f"v1 仅支持 ExecutionMode.MARKET，收到 {self.execution_mode}"
            )

        # 7. 初始资金
        if self.initial_portfolio_value <= 0:
            raise ValueError(
                f"initial_portfolio_value 必须 > 0，收到 {self.initial_portfolio_value}"
            )

        # 8. periods_per_year
        if self.periods_per_year <= 0:
            raise ValueError(
                f"periods_per_year 必须 > 0，收到 {self.periods_per_year}"
            )

        # 9. min_trade_value
        if self.min_trade_value <= 0:
            raise ValueError(
                f"min_trade_value 必须 > 0，收到 {self.min_trade_value}"
            )

        # 10. v1 仅支持 bar_close 时间语义
        if self.time_convention != "bar_close":
            raise NotImplementedError(
                f"v1 仅支持 time_convention='bar_close'，收到 '{self.time_convention}'"
            )

        # 11. VECTORIZED 与 execution_mode 关系
        if self.run_mode == RunMode.VECTORIZED and self.execution_mode != ExecutionMode.MARKET:
            logger.warning(
                "VECTORIZED 模式不使用 execution_mode；该字段将被忽略"
            )

        # 12. VECTORIZED + MATCH_VECTORIZED 是 no-op
        if self.run_mode == RunMode.VECTORIZED and self.cost_mode == CostMode.MATCH_VECTORIZED:
            logger.warning(
                "VECTORIZED 模式下 cost_mode=MATCH_VECTORIZED 是 no-op"
                "（VECTORIZED 自身就是 baseline）"
            )

        # 13. optimize_every_n_bars 范围
        if self.optimize_every_n_bars < 1:
            raise ValueError(
                f"optimize_every_n_bars 必须 >= 1，收到 {self.optimize_every_n_bars}"
            )

        # 14. optimize_every_n_bars 仅对 DYNAMIC_COST 有意义
        if self.run_mode != RunMode.EVENT_DRIVEN_DYNAMIC_COST and self.optimize_every_n_bars != 1:
            logger.warning(
                "%s 模式下 optimize_every_n_bars 被忽略（仅 DYNAMIC_COST 生效）",
                self.run_mode.name,
            )

        # 15. start / end 的 tz-aware 校验已前置到 #3 之前（避免 TypeError）

        # 16. regime_series（若提供）必须为 tz-aware DatetimeIndex
        if self.regime_series is not None:
            idx = self.regime_series.index
            if not isinstance(idx, pd.DatetimeIndex):
                raise ValueError(
                    "regime_series.index 必须为 pd.DatetimeIndex"
                )
            if idx.tz is None:
                raise ValueError(
                    "regime_series.index 必须为 tz-aware（推荐 tz='UTC'）；"
                    "regime_breakdown 内 reindex(returns.index) 在 tz 不一致时会 TypeError"
                )
