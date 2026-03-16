"""
信号平滑

通过 EMA 衰减等方式平滑信号序列，降低高频噪声和换手率。
"""

from __future__ import annotations

import pandas as pd


def ema_smooth(
    signal: pd.DataFrame,
    halflife: int = 5,
) -> pd.DataFrame:
    """
    指数移动平均平滑

    使用 pandas ewm，halflife 控制衰减速度。
    halflife 越小，平滑越弱（响应更快）；越大，平滑越强（信号越滞后）。

    Args:
        signal:   信号面板 (timestamp × symbol)
        halflife: 半衰期（bar 数），必须 >= 1

    Returns:
        平滑后的信号面板

    Raises:
        ValueError: halflife < 1
    """
    if halflife < 1:
        raise ValueError(f"halflife 必须 >= 1, 收到 {halflife}")

    return signal.ewm(halflife=halflife, min_periods=1).mean()
