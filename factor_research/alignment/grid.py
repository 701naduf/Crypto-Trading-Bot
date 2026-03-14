"""
网格对齐模块（Grid Alignment + Forward Fill）

将多个不规则时间序列（各标的的因子值）对齐到统一的规则网格上。
这是本项目默认的对齐方案。

工作原理:
    1. 构建一个规则时间网格（如每 1 秒一个点）
    2. 将每个标的的因子值 reindex 到网格上
    3. 用 forward fill 填充网格中的空值
       （使用该标的最近一次的因子值）

Forward Fill 的合理性:
    在因子更新之前，上一个因子值仍然是"最新的已知信息"。
    这与实际交易中使用的信息集一致（你只知道最近一次的因子值）。

误差分析:
    对齐误差 ≤ 网格间距。对于 1 秒网格和分钟级策略，
    误差被因子计算的 lookback 窗口（通常几十秒到几分钟）吸收。

依赖: pandas
"""

import pandas as pd


def grid_align(
    series_dict: dict[str, pd.Series],
    freq: str = "1s",
    start: pd.Timestamp = None,
    end: pd.Timestamp = None,
    fill_method: str = "ffill",
    max_gap: int = None,
) -> pd.DataFrame:
    """
    将多个不规则时间序列对齐到规则网格

    Args:
        series_dict: {symbol: pd.Series} 字典
                     每个 Series 的 index 为 DatetimeIndex
        freq:        网格频率，如 "1s", "1min", "100ms"
        start:       网格起始时间，默认取所有序列的最早时间
        end:         网格结束时间，默认取所有序列的最晚时间
        fill_method: 填充方法，"ffill" (默认) 或 "bfill"
        max_gap:     最大填充间隔（网格点数），超过此间隔的填充变为 NaN。
                     防止长时间没有更新的标的用过时数据填充。
                     None 表示不限制。

    Returns:
        pd.DataFrame: 对齐后的面板
            index:   DatetimeIndex (规则网格)
            columns: symbols
            values:  对齐后的因子值

    Examples:
        >>> btc_factor = pd.Series(
        ...     [0.1, 0.2, 0.15],
        ...     index=pd.to_datetime(["2024-01-01 10:00:00.100",
        ...                           "2024-01-01 10:00:00.500",
        ...                           "2024-01-01 10:00:01.200"])
        ... )
        >>> eth_factor = pd.Series(
        ...     [0.3, -0.1],
        ...     index=pd.to_datetime(["2024-01-01 10:00:00.300",
        ...                           "2024-01-01 10:00:00.900"])
        ... )
        >>> panel = grid_align(
        ...     {"BTC/USDT": btc_factor, "ETH/USDT": eth_factor},
        ...     freq="1s"
        ... )
    """
    if not series_dict:
        return pd.DataFrame()

    # 确定网格范围
    if start is None:
        start = min(s.index.min() for s in series_dict.values() if len(s) > 0)
    if end is None:
        end = max(s.index.max() for s in series_dict.values() if len(s) > 0)

    # 构建规则网格
    grid_index = pd.date_range(start=start, end=end, freq=freq)

    if len(grid_index) == 0:
        return pd.DataFrame()

    # 对每个标的做 reindex + fill
    result = {}
    for symbol, series in series_dict.items():
        if series.empty:
            result[symbol] = pd.Series(dtype=float, index=grid_index)
            continue

        # reindex 到网格: 先合并原始 index 和网格 index，做 ffill，再选出网格点
        combined = series.reindex(series.index.union(grid_index))

        if fill_method == "ffill":
            combined = combined.ffill()
        elif fill_method == "bfill":
            combined = combined.bfill()

        aligned = combined.reindex(grid_index)

        # 如果设置了最大填充间隔，将过长的填充变为 NaN
        if max_gap is not None:
            aligned = _apply_max_gap(aligned, series.index, grid_index, max_gap)

        result[symbol] = aligned

    return pd.DataFrame(result, index=grid_index)


def _apply_max_gap(
    aligned: pd.Series,
    original_index: pd.DatetimeIndex,
    grid_index: pd.DatetimeIndex,
    max_gap: int,
) -> pd.Series:
    """
    将超过 max_gap 个网格点没有原始数据的填充值设为 NaN

    原理:
        对于网格上的每个点，计算距离最近一个原始数据点的距离（网格点数）。
        如果距离超过 max_gap，则认为该填充已过时，设为 NaN。
    """
    # 标记原始数据出现的位置
    has_data = pd.Series(False, index=grid_index)
    for ts in original_index:
        # 找到最近的网格点
        idx = grid_index.searchsorted(ts)
        if idx < len(grid_index):
            has_data.iloc[idx] = True

    # 计算每个网格点距最近原始数据的距离
    gap_count = 0
    for i in range(len(grid_index)):
        if has_data.iloc[i]:
            gap_count = 0
        else:
            gap_count += 1

        if gap_count > max_gap:
            aligned.iloc[i] = float("nan")

    return aligned
