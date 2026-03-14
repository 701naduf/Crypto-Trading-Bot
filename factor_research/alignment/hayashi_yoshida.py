"""
Hayashi-Yoshida 协方差/相关性估计模块

解决非同步交易数据下的协方差估计问题。

背景:
    传统协方差计算要求两个时间序列在相同时刻有观测。
    但 tick 数据天然是非同步的——BTC 和 ETH 的成交时刻不重合。
    如果用传统方法（先同步采样再计算），会导致 Epps 效应:
        采样频率越高，估计的相关性越低（假象）。

Hayashi-Yoshida 估计量:
    不要求时间对齐，而是利用所有时间区间有重叠的观测对。

    BTC: |--Δp₁--|---Δp₂---|
    ETH:    |---Δq₁---|--Δq₂--|

    只要 BTC 的 Δpᵢ 和 ETH 的 Δqⱼ 在时间上有重叠，
    就把 Δpᵢ × Δqⱼ 计入协方差估计。

理论优势:
    - 渐近无偏且一致
    - 完全避免 Epps 效应
    - 使用全量 tick 数据，信息无损失

参考文献:
    Hayashi, T., & Yoshida, N. (2005).
    "On covariance estimation of non-synchronously observed diffusion processes."
    Bernoulli, 11(2), 359-379.

依赖: numpy, pandas
"""

import numpy as np
import pandas as pd


def hy_covariance(
    series_x: pd.Series,
    series_y: pd.Series,
) -> float:
    """
    Hayashi-Yoshida 协方差估计

    计算两个非同步时间序列的协方差。

    Args:
        series_x: 第一个价格/对数价格序列 (index 为不规则 DatetimeIndex)
        series_y: 第二个价格/对数价格序列

    Returns:
        float: 协方差估计值

    Examples:
        >>> btc_prices = pd.Series(
        ...     [100, 101, 102, 101.5],
        ...     index=pd.to_datetime(["10:00:00", "10:00:01", "10:00:03", "10:00:05"])
        ... )
        >>> eth_prices = pd.Series(
        ...     [50, 50.5, 50.3, 50.8],
        ...     index=pd.to_datetime(["10:00:00.5", "10:00:02", "10:00:04", "10:00:06"])
        ... )
        >>> cov = hy_covariance(btc_prices, eth_prices)
    """
    if len(series_x) < 2 or len(series_y) < 2:
        return np.nan

    # 转换为 numpy 进行高效计算
    times_x = series_x.index.values.astype(np.int64)
    vals_x = series_x.values.astype(np.float64)
    times_y = series_y.index.values.astype(np.int64)
    vals_y = series_y.values.astype(np.float64)

    # 计算 X 和 Y 的增量
    # X 的第 i 个区间: [times_x[i], times_x[i+1]]，增量 = vals_x[i+1] - vals_x[i]
    n_x = len(times_x) - 1
    n_y = len(times_y) - 1

    delta_x = np.diff(vals_x)
    delta_y = np.diff(vals_y)
    start_x = times_x[:-1]
    end_x = times_x[1:]
    start_y = times_y[:-1]
    end_y = times_y[1:]

    # HY 协方差 = Σ Δxᵢ * Δyⱼ，对所有重叠的区间对 (i, j)
    # 两个区间重叠的条件: start_x[i] < end_y[j] 且 start_y[j] < end_x[i]
    cov = 0.0
    j_start = 0

    for i in range(n_x):
        # 优化: 利用有序性，跳过不可能重叠的 j
        while j_start < n_y and end_y[j_start] <= start_x[i]:
            j_start += 1

        for j in range(j_start, n_y):
            if start_y[j] >= end_x[i]:
                break  # 后续的 j 区间都不会与 i 重叠

            # 区间重叠
            cov += delta_x[i] * delta_y[j]

    return float(cov)


def hy_correlation(
    series_x: pd.Series,
    series_y: pd.Series,
) -> float:
    """
    Hayashi-Yoshida 相关性估计

    基于 HY 协方差，归一化为相关系数。

    公式:
        corr = cov_xy / sqrt(var_x * var_y)

    其中 var_x 和 var_y 使用已实现方差（即自身的 HY 协方差）。

    Args:
        series_x: 第一个价格序列
        series_y: 第二个价格序列

    Returns:
        float: 相关系数，范围 [-1, 1]
    """
    cov_xy = hy_covariance(series_x, series_y)
    var_x = hy_covariance(series_x, series_x)  # 已实现方差
    var_y = hy_covariance(series_y, series_y)

    if np.isnan(cov_xy) or np.isnan(var_x) or np.isnan(var_y):
        return np.nan
    if var_x <= 0 or var_y <= 0:
        return np.nan

    corr = cov_xy / np.sqrt(var_x * var_y)
    # 数值精度可能导致略微超过 [-1, 1]
    return float(np.clip(corr, -1.0, 1.0))


def hy_covariance_matrix(
    series_dict: dict[str, pd.Series],
) -> pd.DataFrame:
    """
    多标的 Hayashi-Yoshida 协方差矩阵

    Args:
        series_dict: {symbol: 价格 Series} 字典

    Returns:
        pd.DataFrame: 协方差矩阵 (symbol × symbol)
    """
    symbols = sorted(series_dict.keys())
    n = len(symbols)

    cov_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            cov = hy_covariance(series_dict[symbols[i]], series_dict[symbols[j]])
            cov_matrix[i, j] = cov
            cov_matrix[j, i] = cov

    return pd.DataFrame(cov_matrix, index=symbols, columns=symbols)


def hy_correlation_matrix(
    series_dict: dict[str, pd.Series],
) -> pd.DataFrame:
    """
    多标的 Hayashi-Yoshida 相关性矩阵

    Args:
        series_dict: {symbol: 价格 Series} 字典

    Returns:
        pd.DataFrame: 相关性矩阵 (symbol × symbol)
    """
    cov = hy_covariance_matrix(series_dict)
    variances = np.diag(cov.values)

    # 处理零方差
    std = np.sqrt(np.maximum(variances, 0))
    std[std == 0] = np.nan

    corr = cov.values / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)

    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)
