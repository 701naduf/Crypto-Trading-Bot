"""
数据聚合模块

提供两种聚合能力:

1. tick → OHLCV: 将逐笔成交数据聚合为任意时间周期的 K线
   适用于亚分钟周期（如 10s, 30s），这些周期无法从 1m K线降采样得到

2. 低周期 K线 → 高周期 K线: 将 1m K线降采样为 5m, 15m, 1h 等
   适用于分钟及以上周期

聚合规则:
    - open:   周期内第一笔的 open（K线降采样）或 price（tick 聚合）
    - high:   周期内最高价
    - low:    周期内最低价
    - close:  周期内最后一笔的 close（K线降采样）或 price（tick 聚合）
    - volume: 周期内所有 volume 之和

依赖: utils.time_utils
"""

import pandas as pd

from data_infra.utils.logger import get_logger
from data_infra.utils.time_utils import align_to_timeframe, timeframe_to_seconds

logger = get_logger(__name__)


def aggregate_ticks_to_ohlcv(
    ticks: pd.DataFrame, timeframe: str
) -> pd.DataFrame:
    """
    将逐笔成交聚合为指定周期的 OHLCV K线

    每个时间窗口内的所有成交被聚合为一根 K线:
        - open:   窗口内第一笔成交价
        - high:   窗口内最高成交价
        - low:    窗口内最低成交价
        - close:  窗口内最后一笔成交价
        - volume: 窗口内总成交量

    无成交的时间窗口不生成 K线（跳过）。

    Args:
        ticks:     DataFrame，必须包含 [timestamp, price, amount] 列
                   - timestamp: UTC datetime
                   - price: 成交价格
                   - amount: 成交数量
        timeframe: 目标周期，如 "10s", "30s", "1m", "5m"

    Returns:
        OHLCV DataFrame [timestamp, open, high, low, close, volume]
        - timestamp: 每根 K线 的开盘时间（窗口起始时间）
        - 按 timestamp 升序排列

    Example:
        >>> ticks = pd.DataFrame({
        ...     "timestamp": [...],
        ...     "price": [42000, 42100, 41900, 42050],
        ...     "amount": [0.1, 0.2, 0.15, 0.3],
        ... })
        >>> ohlcv = aggregate_ticks_to_ohlcv(ticks, "10s")
    """
    if ticks.empty:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

    # 确保按时间排序
    ticks = ticks.sort_values("timestamp").copy()

    # 计算每笔成交所属的时间窗口起始时间
    # 例如 10:00:03 属于 10s 周期 → 10:00:00
    period_seconds = timeframe_to_seconds(timeframe)
    period_str = f"{period_seconds}s"

    # 使用 pandas 的 Grouper 按时间窗口分组
    ticks = ticks.set_index("timestamp")

    # resample: 按固定时间间隔分组
    # origin="epoch": 以 Unix 纪元为对齐基准，确保窗口边界一致
    ohlcv = ticks.resample(period_str, origin="epoch").agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("amount", "sum"),
    )

    # 去掉无成交的窗口（NaN）
    ohlcv = ohlcv.dropna(subset=["open"])

    # 将 index 转为 timestamp 列
    ohlcv = ohlcv.reset_index()
    ohlcv = ohlcv.rename(columns={"index": "timestamp"})

    # 如果 reset_index 后列名是原始的 index 名
    if "timestamp" not in ohlcv.columns and ticks.index.name in ohlcv.columns:
        ohlcv = ohlcv.rename(columns={ticks.index.name: "timestamp"})

    logger.debug(
        f"Tick → {timeframe} OHLCV: {len(ticks)} 笔成交 → {len(ohlcv)} 根K线"
    )

    return ohlcv


def resample_ohlcv(
    df: pd.DataFrame, source_tf: str, target_tf: str
) -> pd.DataFrame:
    """
    将低周期 K线 降采样为高周期 K线

    例如: 1m → 5m, 1m → 1h, 5m → 15m

    降采样规则:
        - open:   窗口内第一根 K线 的 open
        - high:   窗口内所有 K线 的 high 的最大值
        - low:    窗口内所有 K线 的 low 的最小值
        - close:  窗口内最后一根 K线 的 close
        - volume: 窗口内所有 K线 的 volume 之和

    Args:
        df:        源 OHLCV DataFrame [timestamp, open, high, low, close, volume]
        source_tf: 源周期，如 "1m"
        target_tf: 目标周期，如 "5m", "1h"
                   必须是 source_tf 的整数倍

    Returns:
        降采样后的 OHLCV DataFrame

    Raises:
        ValueError: 目标周期不是源周期的整数倍
    """
    if df.empty:
        return df.copy()

    source_seconds = timeframe_to_seconds(source_tf)
    target_seconds = timeframe_to_seconds(target_tf)

    # 校验: 目标周期必须是源周期的整数倍
    if target_seconds % source_seconds != 0:
        raise ValueError(
            f"目标周期 {target_tf} ({target_seconds}s) "
            f"不是源周期 {source_tf} ({source_seconds}s) 的整数倍"
        )

    if target_seconds == source_seconds:
        return df.copy()

    # 设置 timestamp 为 index
    df = df.sort_values("timestamp").copy()
    df = df.set_index("timestamp")

    # resample 到目标周期
    period_str = f"{target_seconds}s"

    resampled = df.resample(period_str, origin="epoch").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })

    # 去掉不完整的窗口（NaN）
    resampled = resampled.dropna(subset=["open"])

    resampled = resampled.reset_index()

    ratio = target_seconds // source_seconds
    logger.debug(
        f"{source_tf} → {target_tf} ({ratio}:1): "
        f"{len(df)} 根 → {len(resampled)} 根"
    )

    return resampled
