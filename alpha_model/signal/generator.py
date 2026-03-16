"""
信号生成

将模型预测值（alpha score）转化为标准化信号。

默认不截断极端值。如有需要，提供可选 clip_sigma 参数。

设计理由：默认不截断
    模型输出的极端预测值可能正是信号强烈的证明。
    截断极端值会削弱策略的表达能力。
    真正的风控由 portfolio 层的约束（max_weight、leverage_cap）来实现，
    而不是在信号层削弱信息。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_signal(
    predictions: pd.DataFrame,
    method: str = "cross_sectional_zscore",
    clip_sigma: float | None = None,
) -> pd.DataFrame:
    """
    预测值 → 标准化信号

    Args:
        predictions: 模型预测值面板 (timestamp × symbol)
        method:      截面标准化方式
                     "cross_sectional_zscore" — 截面 z-score
                     "cross_sectional_rank"   — 截面百分位排名 [0, 1]
        clip_sigma:  截断阈值（标准差倍数）。默认 None 不截断。
                     如果指定，将 |z| > clip_sigma 的值截断到 ±clip_sigma。
                     仅在 method="cross_sectional_zscore" 时生效。

    Returns:
        信号面板 (timestamp × symbol)，值为标准化后的信号强度

    Raises:
        ValueError: 不支持的 method
    """
    if predictions.empty:
        return predictions.copy()

    if method == "cross_sectional_zscore":
        # 每行独立做 z-score
        row_mean = predictions.mean(axis=1)
        row_std = predictions.std(axis=1)
        # std == 0 → NaN（不返回 0）
        row_std = row_std.replace(0, np.nan)
        signal = predictions.sub(row_mean, axis=0).div(row_std, axis=0)

        # 可选截断
        if clip_sigma is not None:
            signal = signal.clip(lower=-clip_sigma, upper=clip_sigma)

    elif method == "cross_sectional_rank":
        # 每行独立做百分位排名
        signal = predictions.rank(axis=1, pct=True)

    else:
        raise ValueError(
            f"不支持的 method: {method}，"
            f"可选 'cross_sectional_zscore' 或 'cross_sectional_rank'"
        )

    return signal
