"""
时间处理工具

交易所 API 使用毫秒时间戳（如 1705305600000），
本地处理使用 Python datetime 对象。
本模块提供两者之间的转换，以及 K线时间对齐等工具函数。

重要约定:
    - 所有时间统一使用 UTC 时区，避免时区混乱
    - 存储和传输时使用毫秒时间戳（int）
    - 计算和展示时使用 datetime 对象（带 UTC 时区信息）

依赖: 无
"""

from datetime import datetime, timezone


def ms_to_datetime(ms: int) -> datetime:
    """
    将毫秒时间戳转换为 UTC datetime 对象

    Args:
        ms: 毫秒时间戳，如 1705305600000

    Returns:
        带 UTC 时区信息的 datetime 对象

    Example:
        >>> ms_to_datetime(1705305600000)
        datetime(2024, 1, 15, 8, 0, 0, tzinfo=timezone.utc)
    """
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def datetime_to_ms(dt: datetime) -> int:
    """
    将 datetime 对象转换为毫秒时间戳

    如果传入的 datetime 没有时区信息（naive），会被视为 UTC。

    Args:
        dt: datetime 对象

    Returns:
        毫秒时间戳（整数）

    Example:
        >>> dt = datetime(2024, 1, 15, 8, 0, 0, tzinfo=timezone.utc)
        >>> datetime_to_ms(dt)
        1705305600000
    """
    # 如果是 naive datetime（无时区信息），假定为 UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return int(dt.timestamp() * 1000)


def align_to_timeframe(dt: datetime, timeframe: str) -> datetime:
    """
    将时间对齐（向下取整）到指定周期的起始时刻

    用于将任意时间点对齐到 K线 的起始时间。
    例如 10:23:45 对齐到 1h 周期 → 10:00:00

    支持的周期格式:
        秒级: "10s", "30s"
        分钟级: "1m", "5m", "15m", "30m"
        小时级: "1h", "4h"
        天级: "1d"

    Args:
        dt: 待对齐的时间
        timeframe: 周期字符串

    Returns:
        对齐后的时间（同一时区）

    Examples:
        >>> dt = datetime(2024, 1, 15, 10, 23, 45, tzinfo=timezone.utc)
        >>> align_to_timeframe(dt, "1h")
        datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        >>> align_to_timeframe(dt, "5m")
        datetime(2024, 1, 15, 10, 20, 0, tzinfo=timezone.utc)

        >>> align_to_timeframe(dt, "30s")
        datetime(2024, 1, 15, 10, 23, 30, tzinfo=timezone.utc)
    """
    seconds = timeframe_to_seconds(timeframe)

    # 将 datetime 转换为 UNIX 秒数，然后向下取整到周期边界
    ts = dt.timestamp()
    aligned_ts = (int(ts) // seconds) * seconds

    return datetime.fromtimestamp(aligned_ts, tz=dt.tzinfo or timezone.utc)


def timeframe_to_seconds(timeframe: str) -> int:
    """
    将周期字符串转换为秒数

    Args:
        timeframe: 周期字符串，如 "1m", "1h", "1d"
                   支持的单位: s(秒), m(分钟), h(小时), d(天)

    Returns:
        对应的秒数

    Raises:
        ValueError: 不支持的周期格式

    Examples:
        >>> timeframe_to_seconds("10s")
        10
        >>> timeframe_to_seconds("1m")
        60
        >>> timeframe_to_seconds("1h")
        3600
        >>> timeframe_to_seconds("1d")
        86400
    """
    # 将周期字符串拆分为 数值 + 单位
    # 例: "15m" → 数值=15, 单位="m"
    unit = timeframe[-1]
    value = int(timeframe[:-1])

    multipliers = {
        "s": 1,
        "m": 60,
        "h": 3600,
        "d": 86400,
    }

    if unit not in multipliers:
        raise ValueError(
            f"不支持的周期格式: '{timeframe}'。"
            f"支持的单位: {list(multipliers.keys())}"
        )

    return value * multipliers[unit]


def is_standard_timeframe(timeframe: str) -> bool:
    """
    判断是否为标准 K线 周期（可从 1m 降采样得到）

    标准周期: 1m 本身，以及 1m 的整数倍周期。
    非标准周期（如 10s, 30s）需要从逐笔成交数据聚合。

    Args:
        timeframe: 周期字符串

    Returns:
        True 表示标准周期（存储在 SQLite 中或可降采样得到）
        False 表示需要从 tick 数据聚合

    Examples:
        >>> is_standard_timeframe("1m")
        True
        >>> is_standard_timeframe("5m")
        True
        >>> is_standard_timeframe("10s")
        False
    """
    # 标准周期列表: 1m 及其整数倍
    standard = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"}
    return timeframe in standard
