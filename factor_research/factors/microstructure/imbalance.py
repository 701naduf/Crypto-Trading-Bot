"""
订单簿不平衡度因子

经济含义:
    衡量订单簿买卖两侧的力量对比。
    当买方挂单量显著大于卖方时，imbalance > 0，暗示短期上行压力；反之亦然。

    这是最基础也是最重要的微观结构因子之一。
    在 HFT 研究中，订单簿不平衡度被广泛用于预测短期价格走向。

计算逻辑:
    imbalance = (bid_qty_sum - ask_qty_sum) / (bid_qty_sum + ask_qty_sum)

    其中 bid_qty_sum 和 ask_qty_sum 分别是买卖各档位的挂单量之和。
    使用所有可用档位（默认 10 档），以充分捕捉订单簿深度信息。

输出:
    范围 [-1, 1]，0 表示买卖平衡
    > 0: 买方力量占优
    < 0: 卖方力量占优

因子类型: 时序因子（各标的独立计算）
数据需求: 10 档订单簿快照
输出频率: 1s（从 100ms 快照聚合为 1 秒均值）

依赖: core.base, core.types
"""

import pandas as pd

from factor_research.core.base import TimeSeriesFactor
from factor_research.core.registry import register_factor
from factor_research.core.types import (
    DataRequest,
    DataType,
    FactorMeta,
    FactorType,
)


@register_factor
class OrderbookImbalance(TimeSeriesFactor):
    """
    订单簿不平衡度因子

    使用所有档位的挂单量计算买卖力量对比。
    输出 1 秒频率的不平衡度序列。

    参数:
        levels:        订单簿档位数（默认 10）
        resample_freq: 降采样频率（默认 "1s"）
    """

    def __init__(self, levels: int = 10, resample_freq: str = "1s"):
        self.levels = levels
        self.resample_freq = resample_freq

    def meta(self) -> FactorMeta:
        return FactorMeta(
            name="orderbook_imbalance",
            display_name="订单簿不平衡度",
            factor_type=FactorType.TIME_SERIES,
            category="microstructure",
            description="(bid_qty_sum - ask_qty_sum) / total_qty，衡量订单簿买卖力量对比",
            data_requirements=[
                DataRequest(DataType.ORDERBOOK, orderbook_levels=self.levels),
            ],
            output_freq=self.resample_freq,
            params={"levels": self.levels, "resample_freq": self.resample_freq},
            author="system",
            version="1.0",
        )

    def compute_single(self, symbol: str, data: dict) -> pd.Series:
        """
        计算单个标的的订单簿不平衡度

        Args:
            symbol: 交易对名称
            data:   {DataType.ORDERBOOK: pd.DataFrame}

        Returns:
            pd.Series: 1 秒频率的不平衡度序列
        """
        ob = data[DataType.ORDERBOOK]

        if ob.empty:
            return pd.Series(dtype=float)

        # 确定可用的档位数
        bid_cols = [f"bid_qty_{i}" for i in range(self.levels) if f"bid_qty_{i}" in ob.columns]
        ask_cols = [f"ask_qty_{i}" for i in range(self.levels) if f"ask_qty_{i}" in ob.columns]

        if not bid_cols or not ask_cols:
            return pd.Series(dtype=float)

        # 买卖各档位挂单量之和
        bid_qty = ob[bid_cols].sum(axis=1)
        ask_qty = ob[ask_cols].sum(axis=1)

        # 不平衡度: [-1, 1]
        total = bid_qty + ask_qty
        # 防止除零
        total = total.replace(0, float("nan"))
        imbalance = (bid_qty - ask_qty) / total

        # 设置时间索引
        if "timestamp" in ob.columns:
            imbalance.index = pd.to_datetime(ob["timestamp"], utc=True)
        elif ob.index.name == "timestamp" or hasattr(ob.index, "tz"):
            imbalance.index = ob.index

        # 降采样到指定频率（取均值），减少数据量
        if len(imbalance) > 0 and hasattr(imbalance.index, 'floor'):
            imbalance = imbalance.resample(self.resample_freq).mean()

        imbalance.name = self.meta().name
        return imbalance.dropna()
