"""
协方差矩阵估计

样本协方差矩阵在 T/N 比较小时（T=样本数，N=资产数）
估计误差极大，需要 shrinkage 修正。

方法:
    Ledoit-Wolf Shrinkage:
        Σ_shrunk = δ × F + (1-δ) × S
        F = 结构化目标（对角矩阵，只保留方差）
        S = 样本协方差矩阵
        δ = 最优收缩强度（数据驱动，自动计算）

    直接使用 sklearn.covariance.LedoitWolf。

    为什么不用样本协方差？
        5 标的场景下 N=5 很小，但即使如此，样本协方差的估计误差
        在优化中会被放大（优化器倾向于利用估计误差）。
        Ledoit-Wolf 是业界标准的稳健估计方法。

依赖: sklearn.covariance.LedoitWolf, numpy
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)


def estimate_covariance(
    returns_panel: pd.DataFrame,
    lookback: int = 60,
    method: str = "ledoit_wolf",
    min_periods: int = 20,
) -> np.ndarray:
    """
    估计协方差矩阵

    使用 returns_panel 的最后 lookback 行数据估计协方差。

    Args:
        returns_panel: 收益率面板 (timestamp × symbol)
        lookback:      回望窗口长度
        method:        估计方法:
                       "ledoit_wolf" — Ledoit-Wolf shrinkage（推荐）
                       "sample"      — 样本协方差（仅用于对比）
                       "exponential" — 指数加权协方差
        min_periods:   最少需要的有效期数

    Returns:
        np.ndarray: N×N 协方差矩阵

    Raises:
        ValueError: 有效数据不足
    """
    # 取最后 lookback 行
    data = returns_panel.tail(lookback).dropna()

    if len(data) < min_periods:
        raise ValueError(
            f"有效数据 {len(data)} 行不足 min_periods={min_periods}"
        )

    if method == "ledoit_wolf":
        lw = LedoitWolf()
        lw.fit(data.values)
        return lw.covariance_

    elif method == "sample":
        return data.cov().values

    elif method == "exponential":
        # 指数加权协方差，span = lookback
        ewm_cov = data.ewm(span=lookback, min_periods=min_periods).cov()
        last_ts = data.index[-1]
        return ewm_cov.loc[last_ts].values

    else:
        raise ValueError(
            f"不支持的 method: {method}，可选 'ledoit_wolf', 'sample', 'exponential'"
        )


def rolling_covariance(
    returns_panel: pd.DataFrame,
    lookback: int = 60,
    method: str = "ledoit_wolf",
    min_periods: int = 20,
) -> dict[pd.Timestamp, np.ndarray]:
    """
    滚动协方差矩阵序列

    对每个时间点，使用前 lookback 行数据估计协方差矩阵。
    [P5] 直接对窗口数据调用估计器，避免经过 estimate_covariance 的 tail() 冗余层。

    Args:
        returns_panel: 收益率面板 (timestamp × symbol)
        lookback:      回望窗口长度
        method:        估计方法 ("ledoit_wolf" 或 "sample")
        min_periods:   最少需要的有效期数

    Returns:
        {timestamp: cov_matrix} 字典
    """
    result = {}
    timestamps = returns_panel.index

    for i in range(lookback, len(timestamps)):
        ts = timestamps[i]
        window_data = returns_panel.iloc[i - lookback:i].dropna()

        if len(window_data) < min_periods:
            continue

        if len(window_data) < lookback * 0.5:
            logger.debug(
                "时刻 %s: dropna 后仅剩 %d/%d 行 (%.0f%%)，协方差估计质量可能下降",
                ts, len(window_data), lookback,
                100 * len(window_data) / lookback,
            )

        try:
            if method == "ledoit_wolf":
                lw = LedoitWolf()
                lw.fit(window_data.values)
                result[ts] = lw.covariance_
            elif method == "sample":
                result[ts] = window_data.cov().values
            else:
                raise ValueError(
                    f"rolling_covariance 不支持 method: {method}，"
                    f"可选 'ledoit_wolf', 'sample'"
                )
        except Exception as e:
            logger.debug("时刻 %s 协方差估计失败: %s", ts, e)

    return result
