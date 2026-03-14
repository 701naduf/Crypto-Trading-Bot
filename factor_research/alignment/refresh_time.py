"""
刷新时间采样模块（Refresh Time Sampling）

比网格对齐更严格的对齐方案。
只在"所有标的都有新数据"的时刻采样。

定义:
    刷新时刻 = 自上一个刷新时刻以来，所有标的都至少发生过一笔新观测的时刻。

    BTC ticks:  --|---|-|-------|--|--->
    ETH ticks:  ----|--|---|----|---|--->
    SOL ticks:  --|----|---|--|----|---->
                          ↑       ↑
                       刷新时刻  刷新时刻
                  （三者都有了新观测）

优势:
    保证每个标的的数据都是"新鲜的"（自上次采样后确实有更新）。

代价:
    刷新频率取决于最不活跃的标的。
    如果 DOGE 10 秒才成交一笔，刷新频率不会高于 0.1Hz。

适用场景:
    需要严格截面对齐的因子（如截面排名因子）。
    大多数场景下网格对齐已足够。

依赖: pandas
"""

import pandas as pd


def refresh_time_align(
    series_dict: dict[str, pd.Series],
    min_freshness: str = None,
) -> pd.DataFrame:
    """
    刷新时间采样对齐

    找到所有标的都有新数据的时刻，在这些时刻采样。
    每个标的取其最近一笔观测。

    Args:
        series_dict:   {symbol: pd.Series} 字典
                       每个 Series 的 index 为不规则 DatetimeIndex
        min_freshness: 可选，最小新鲜度要求。
                       如 "1s" 表示每个标的的数据至少更新到最近 1 秒以内。
                       None 表示只要求自上次刷新后有更新即可。

    Returns:
        pd.DataFrame: 对齐后的面板
            index:   DatetimeIndex (刷新时刻)
            columns: symbols
            values:  每个标的在刷新时刻的最新值

    Examples:
        >>> btc = pd.Series([1, 2, 3], index=pd.to_datetime(
        ...     ["10:00:00.1", "10:00:00.5", "10:00:01.0"]))
        >>> eth = pd.Series([4, 5], index=pd.to_datetime(
        ...     ["10:00:00.3", "10:00:00.8"]))
        >>> panel = refresh_time_align({"BTC": btc, "ETH": eth})
        # 刷新时刻: 10:00:00.3（BTC@10:00:00.1已更新，ETH首次出现）
        #          10:00:00.8（BTC@10:00:00.5已更新，ETH@10:00:00.8已更新）
    """
    if not series_dict or len(series_dict) < 2:
        if len(series_dict) == 1:
            sym, s = next(iter(series_dict.items()))
            return pd.DataFrame({sym: s})
        return pd.DataFrame()

    symbols = list(series_dict.keys())

    # 合并所有时间点并排序
    all_times = set()
    for s in series_dict.values():
        all_times.update(s.index.tolist())
    all_times = sorted(all_times)

    if not all_times:
        return pd.DataFrame()

    # 跟踪每个标的的最新更新时间和值
    latest_update_time = {sym: None for sym in symbols}
    latest_value = {sym: float("nan") for sym in symbols}
    last_refresh_update = {sym: None for sym in symbols}

    refresh_times = []
    refresh_values = {sym: [] for sym in symbols}

    for t in all_times:
        # 更新在时刻 t 有新数据的标的
        for sym in symbols:
            if t in series_dict[sym].index:
                latest_update_time[sym] = t
                latest_value[sym] = series_dict[sym].loc[t]

        # 检查是否所有标的都自上次刷新后有了新数据
        all_fresh = True
        for sym in symbols:
            if latest_update_time[sym] is None:
                all_fresh = False
                break
            if last_refresh_update[sym] is not None:
                if latest_update_time[sym] <= last_refresh_update[sym]:
                    all_fresh = False
                    break

        # 可选: 检查最小新鲜度
        if all_fresh and min_freshness is not None:
            freshness_td = pd.Timedelta(min_freshness)
            for sym in symbols:
                if t - latest_update_time[sym] > freshness_td:
                    all_fresh = False
                    break

        if all_fresh:
            refresh_times.append(t)
            for sym in symbols:
                refresh_values[sym].append(latest_value[sym])
                last_refresh_update[sym] = latest_update_time[sym]

    if not refresh_times:
        return pd.DataFrame()

    return pd.DataFrame(
        refresh_values,
        index=pd.DatetimeIndex(refresh_times),
    )
