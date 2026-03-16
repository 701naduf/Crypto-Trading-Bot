"""
辅助工具函数

提供跨模块的通用辅助功能。
"""

from __future__ import annotations

import pandas as pd


def load_price_panel(
    symbols: list[str],
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    从 data_infra 加载价格面板 (timestamp × symbol)

    读取 1m kline 的 close 价格，按 symbol 对齐。

    Args:
        symbols: 标的列表
        start:   起始时间（str 或 Timestamp）
        end:     结束时间（str 或 Timestamp）

    Returns:
        价格面板 (timestamp × symbol)，close 价格

    Raises:
        ValueError: 无法加载任何标的的价格数据
    """
    from data_infra.data.reader import DataReader

    # [R3] DataReader.get_ohlcv 期望 datetime 类型，
    # 需要将 str 转为 pd.Timestamp
    if isinstance(start, str):
        start = pd.Timestamp(start)
    if isinstance(end, str):
        end = pd.Timestamp(end)

    reader = DataReader()
    panels = {}
    for symbol in symbols:
        df = reader.get_ohlcv(symbol, "1m", start=start, end=end)
        if df is not None and len(df) > 0:
            panels[symbol] = df["close"]

    if not panels:
        raise ValueError(f"无法加载任何标的的价格数据: {symbols}")

    return pd.DataFrame(panels)
