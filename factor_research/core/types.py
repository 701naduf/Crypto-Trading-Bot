"""
因子模块的核心类型定义

本文件定义了因子研究框架中所有共享的类型和数据结构。
这些类型贯穿整个因子管道——从因子声明、数据请求到计算结果，
是框架各模块之间的"通用语言"。

核心类型:
    FactorType:   因子类型枚举（时序 / 截面 / 跨标的）
    DataType:     数据类型枚举（对应 DataReader 的各接口）
    DataRequest:  因子声明自己需要的数据（类型 + 参数）
    FactorMeta:   因子元数据（名称、分类、参数等）

因子面板格式约定:
    所有因子的输出统一为 pd.DataFrame 面板格式:
        index:   DatetimeIndex (UTC, 名为 "timestamp")
        columns: 各 symbol（如 "BTC/USDT", "ETH/USDT"）
        values:  float64 因子值

    示例:
        timestamp            | BTC/USDT | ETH/USDT | SOL/USDT
        2024-01-15 10:00:00 |  0.35    |  -0.12   |  0.08
        2024-01-15 10:01:00 |  0.28    |   0.05   |  -0.15

依赖: 无（最底层模块）
被依赖: core.base, core.registry, core.engine, store, evaluation
"""

from dataclasses import dataclass, field
from enum import Enum


class FactorType(Enum):
    """
    因子类型枚举

    三种类型对应不同的数据输入方式和引擎调度策略:

    TIME_SERIES:
        时序因子 — 单标的输入 → 单标的输出。
        引擎对每个 symbol 分别调用 compute_single()，最后自动合并为面板。
        例: BTC 的 5min 已实现波动率。

    CROSS_SECTIONAL:
        截面因子 — 全标的输入 → 全标的输出（通常经过排名或标准化）。
        引擎将所有标的的数据一次性传入 compute()。
        例: 5 个币种的动量排名。

    CROSS_ASSET:
        跨标的因子 — 指定标的（通常是 BTC）的数据 → 其他标的的因子值。
        引擎只读取 input_symbols 的数据，输出覆盖 output_symbols。
        例: BTC 涨跌 → ETH 的领先信号。
    """
    TIME_SERIES = "time_series"
    CROSS_SECTIONAL = "cross_sectional"
    CROSS_ASSET = "cross_asset"


class DataType(Enum):
    """
    数据类型枚举

    每个枚举值对应 DataReader 的一个读取接口。
    因子通过 DataRequest 声明需要哪种数据类型，
    引擎据此调用 DataReader 的对应方法。

    映射关系:
        OHLCV             → reader.get_ohlcv(symbol, timeframe, start, end)
        TICK              → reader.get_ticks(symbol, start, end)
        ORDERBOOK         → reader.get_orderbook(symbol, start, end, levels)
        FUNDING_RATE      → reader.get_funding_rate(symbol, start, end)
        OPEN_INTEREST     → reader.get_open_interest(symbol, start, end)
        LONG_SHORT_RATIO  → reader.get_long_short_ratio(symbol, start, end)
        TAKER_BUY_SELL    → reader.get_taker_buy_sell(symbol, start, end)
    """
    OHLCV = "ohlcv"
    TICK = "tick"
    ORDERBOOK = "orderbook"
    FUNDING_RATE = "funding_rate"
    OPEN_INTEREST = "open_interest"
    LONG_SHORT_RATIO = "long_short_ratio"
    TAKER_BUY_SELL = "taker_buy_sell"


@dataclass
class DataRequest:
    """
    因子数据需求声明

    因子通过 FactorMeta.data_requirements 中包含的 DataRequest 列表
    告诉引擎"我需要什么数据"。引擎据此从 DataReader 读取并组织数据。

    Attributes:
        data_type:        数据类型（必填）
        timeframe:        K线周期，仅 OHLCV 类型需要（如 "1m", "5m", "10s"）
        orderbook_levels: 订单簿档位数，仅 ORDERBOOK 类型需要（如 5, 10）
        lookback_bars:    需要的历史回溯长度（bar 数），引擎会据此扩展请求的时间范围。
                          例如因子需要 60 根 K线的滚动窗口，则设为 60。
        symbols:          指定标的列表。None 表示使用全局配置的全部标的 (settings.SYMBOLS)。
                          跨标的因子可能只需要特定标的的数据。

    Examples:
        # 需要 1m K线，回溯 60 根
        DataRequest(DataType.OHLCV, timeframe="1m", lookback_bars=60)

        # 需要 10 档订单簿
        DataRequest(DataType.ORDERBOOK, orderbook_levels=10)

        # 需要逐笔成交（无特殊参数）
        DataRequest(DataType.TICK)

        # 只需要 BTC 的资金费率
        DataRequest(DataType.FUNDING_RATE, symbols=["BTC/USDT"])
    """
    data_type: DataType
    timeframe: str = None
    orderbook_levels: int = None
    lookback_bars: int = 0
    symbols: list[str] = None


@dataclass
class FactorMeta:
    """
    因子元数据

    描述一个因子的基本信息，用于注册、检索、报告生成和因子存储。
    每个因子类必须通过 meta() 方法返回一个 FactorMeta 实例。

    Attributes:
        name:              唯一标识名，如 "orderbook_imbalance_10s"。
                           用作注册 key 和存储目录名，不可重复。
        display_name:      展示名称，如 "订单簿不平衡度 (10s)"。用于报告和日志。
        factor_type:       因子类型（时序 / 截面 / 跨标的）
        category:          因子分类，如 "microstructure", "momentum", "volatility",
                           "orderflow", "cross_asset"。用于按类检索。
        description:       一句话描述因子的计算逻辑和经济含义。
        data_requirements: 需要的数据列表。引擎根据此列表准备输入数据。
        output_freq:       输出频率字符串，如 "1s", "1m", "5m"。
                           用于因子值层面的网格对齐。
        params:            超参数字典，如 {"window": 60, "decay": 0.94}。
                           记录因子的可调参数，便于参数搜索和报告。
        family:            因子族名称，如 "multi_scale_returns"。
                           同族因子共享计算逻辑、仅参数不同。
                           空字符串表示独立因子（不属于任何族）。
        author:            作者标识。
        version:           版本号，因子逻辑变更时递增。

    Examples:
        FactorMeta(
            name="orderbook_imbalance_10s",
            display_name="订单簿不平衡度 (10s)",
            factor_type=FactorType.TIME_SERIES,
            category="microstructure",
            description="(bid_qty_sum - ask_qty_sum) / total_qty，衡量订单簿买卖力量对比",
            data_requirements=[
                DataRequest(DataType.ORDERBOOK, orderbook_levels=10),
            ],
            output_freq="1s",
            params={"levels": 10, "resample_freq": "1s"},
        )
    """
    name: str
    display_name: str
    factor_type: FactorType
    category: str
    description: str
    data_requirements: list[DataRequest]
    output_freq: str
    params: dict = field(default_factory=dict)
    family: str = ""
    author: str = ""
    version: str = "1.0"
