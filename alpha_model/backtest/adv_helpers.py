"""
ADV 兜底 + NaN 处理共享 helper（Z12 跨四处统一）

公开模块（无下划线前缀，Z14）：被以下四处导入：
  - alpha_model.backtest.vectorized.estimate_market_impact
  - execution_optimizer.cost.build_cost_expression
  - backtest_engine.rebalancer.Rebalancer._execute_market
  - backtest_engine.attribution.compute_per_symbol_cost

设计选择（Z12 Option B，与 N1 funding NaN 处理风格一致）:
  ADV NaN → silent 替换为 1.0 + logger.warning 暴露上游 gap
  ADV < 1.0 → 强制按 ADV=1.0 计算 impact（floor）

不抛 NumericalError 的理由：
  ADV NaN 多源于上游数据 gap（DataReader 输出空 close × volume），
  让用户 7h 回测中断不友好。warning 仍显式暴露，事后审查 log 可发现。

Z16 dedup 机制：
  cost.py / Rebalancer 在事件循环每 bar 调用，若某 symbol ADV 持续 NaN，
  warning 每 bar 触发一次 → 1M+ 行 log spam。
  通过 warned_set kwarg 让调用方持有 dedup 集合，仅首次出现的 NaN symbol 触发 warning。
  集合归宿：Rebalancer / OnlineOptimizer 实例字段（避免 module-level 跨 run 脏状态）。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def safe_adv_panel(
    adv_panel: pd.DataFrame,
    *,
    context: str = "",
) -> pd.DataFrame:
    """
    DataFrame 路径 ADV 兜底（vectorized + compute_per_symbol_cost 用）

    单次回测调用，无 dedup 需求（Z16 不适用）。

    Args:
        adv_panel: timestamp × symbol，USDT 单位
        context:   warning 消息前缀（标识调用方）

    Returns:
        adv_safe: NaN → 1.0；ADV < 1.0 → 1.0；其他不变
    """
    if adv_panel.isna().any().any():
        nan_syms = list(adv_panel.columns[adv_panel.isna().any()])
        logger.warning(
            "%s: adv_panel 含 NaN（symbols: %s），已视为 ADV=1.0；"
            "可能是上游数据 gap，请检查 DataReader.get_ohlcv 输出",
            context or "safe_adv_panel", nan_syms,
        )
    # where 处理 < 1.0；fillna 双重防御 NaN（pandas 比较 NaN >= 1.0 为 False，
    # where 已替换 NaN 为 1.0；fillna(1.0) 实为 no-op，保留作未来 pandas 行为变化兜底）
    return adv_panel.where(adv_panel >= 1.0, 1.0).fillna(1.0)


def safe_adv_array(
    adv_arr: np.ndarray,
    symbols: list[str],
    *,
    context: str = "",
    warned_set: set[str] | None = None,
) -> np.ndarray:
    """
    np.array 路径 ADV 兜底（cost.py + Rebalancer 用）

    修 pre-existing bug: np.maximum(NaN, 1.0) 返回 NaN（让 NaN 传播到下游 sum 静默 swallow）

    Args:
        adv_arr:    1D numpy array，同 symbols 顺序
        symbols:    symbol 列表（必须 len(adv_arr) == len(symbols)，Z15 校验）
        context:    warning 消息前缀
        warned_set: Z16 dedup 集合（调用方持有）。
                    None 表示无 dedup（每次都 warning，适用于一次性调用）；
                    传入 set 时仅首次出现的 NaN symbol 触发 warning，
                    避免事件循环每 bar 调用导致 log spam。

    Returns:
        adv_safe: NaN → 1.0；ADV < 1.0 → 1.0

    Raises:
        ValueError: Z15 — len(adv_arr) ≠ len(symbols) 时（防御 programmer error）
    """
    # Z15 防御性校验
    if len(adv_arr) != len(symbols):
        raise ValueError(
            f"safe_adv_array: len(adv_arr)={len(adv_arr)} ≠ len(symbols)={len(symbols)}"
        )

    nan_mask = np.isnan(adv_arr)
    if nan_mask.any():
        nan_syms = [symbols[i] for i in range(len(adv_arr)) if nan_mask[i]]

        # Z16 dedup
        if warned_set is not None:
            new_syms = [s for s in nan_syms if s not in warned_set]
            warned_set.update(nan_syms)
            if new_syms:
                logger.warning(
                    "%s: adv 数组含 NaN（首次出现 symbols: %s），已视为 ADV=1.0；"
                    "可能是上游数据 gap，后续相同 symbol NaN 不再重复 warning",
                    context or "safe_adv_array", new_syms,
                )
        else:
            logger.warning(
                "%s: adv 数组含 NaN（symbols: %s），已视为 ADV=1.0",
                context or "safe_adv_array", nan_syms,
            )

        # 先把 NaN 替换为 1.0（修 np.maximum(NaN, 1.0) = NaN 的 pre-existing bug），
        # 再 max(adv, 1.0) floor
        adv_arr = np.where(nan_mask, 1.0, adv_arr)

    return np.maximum(adv_arr, 1.0)
