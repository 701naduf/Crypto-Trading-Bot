"""
多频率因子对齐

将不同频率的因子面板对齐到统一的时间网格。

不同因子可能具有不同的时间频率（如 10 秒 orderbook、5 分钟 kline）。
在组合建模前需要对齐到统一时间网格。

复用: factor_research.alignment.grid.grid_align
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _infer_freq(panel: pd.DataFrame) -> pd.Timedelta | None:
    """
    推断面板的时间频率

    通过计算相邻时间戳的中位数差值来推断频率。

    Args:
        panel: 因子面板 (timestamp × symbol)

    Returns:
        推断出的频率（Timedelta），如果无法推断则返回 None
    """
    if len(panel) < 2:
        return None
    diffs = pd.Series(panel.index).diff().dropna()
    if diffs.empty:
        return None
    return diffs.median()


def align_factor_panels(
    factor_panels: dict[str, pd.DataFrame],
    target_freq: str | None = None,
    fill_method: str = "ffill",
    max_gap: int | None = None,
) -> dict[str, pd.DataFrame]:
    """
    将多个因子面板对齐到统一频率

    自动推断:
        如果 target_freq=None，取所有因子中最低频率（最大间隔）作为目标频率。
        高频因子向低频重采样，使用最后一个有效值（ffill 语义）。

    Args:
        factor_panels: {factor_name: panel} 字典，每个 panel 为 (timestamp × symbol)
        target_freq:   目标频率字符串（如 "1min", "5min"），None 则自动推断
        fill_method:   填充方法: "ffill" 或 None（不填充，保留 NaN）
        max_gap:       最大允许前向填充行数，None 则不限制

    Returns:
        对齐后的 {factor_name: panel}，所有面板具有相同的时间索引

    Raises:
        ValueError: 无因子面板输入或无法推断频率时
    """
    if not factor_panels:
        raise ValueError("factor_panels 不能为空")

    # --- 推断目标频率 ---
    if target_freq is None:
        # 取所有因子中最低频率（最大 timedelta）
        max_td = pd.Timedelta(0)
        for name, panel in factor_panels.items():
            td = _infer_freq(panel)
            if td is not None and td > max_td:
                max_td = td
        if max_td == pd.Timedelta(0):
            raise ValueError("无法自动推断目标频率，请手动指定 target_freq")
        target_freq_td = max_td
        logger.info("自动推断目标频率: %s", target_freq_td)
    else:
        target_freq_td = pd.tseries.frequencies.to_offset(target_freq)

    # --- 构建统一时间网格 ---
    # 取所有面板的时间范围交集
    all_starts = []
    all_ends = []
    for panel in factor_panels.values():
        if len(panel) > 0:
            all_starts.append(panel.index.min())
            all_ends.append(panel.index.max())

    if not all_starts:
        raise ValueError("所有因子面板均为空")

    grid_start = max(all_starts)
    grid_end = min(all_ends)

    if grid_start >= grid_end:
        raise ValueError(
            f"因子面板时间范围无交集: 最晚起点 {grid_start} >= 最早终点 {grid_end}"
        )

    grid_index = pd.date_range(
        start=grid_start, end=grid_end, freq=target_freq_td
    )
    logger.info(
        "对齐网格: %s ~ %s, %d 行, 频率 %s",
        grid_start, grid_end, len(grid_index), target_freq_td,
    )

    # --- 对齐每个面板 ---
    aligned = {}
    for name, panel in factor_panels.items():
        # 重新索引到统一网格
        reindexed = panel.reindex(grid_index, method=None)

        # 前向填充
        if fill_method == "ffill":
            if max_gap is not None:
                reindexed = reindexed.ffill(limit=max_gap)
            else:
                reindexed = reindexed.ffill()
        elif fill_method is not None:
            raise ValueError(f"不支持的 fill_method: {fill_method}，可选 'ffill' 或 None")

        aligned[name] = reindexed
        n_nan = reindexed.isna().sum().sum()
        if n_nan > 0:
            logger.debug(
                "因子 '%s' 对齐后有 %d 个 NaN（%.1f%%）",
                name, n_nan,
                100 * n_nan / (reindexed.shape[0] * reindexed.shape[1]),
            )

    return aligned
