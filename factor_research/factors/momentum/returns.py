"""
多时间尺度收益率因子

经济含义:
    过去 N 根 K线的累计收益率。
    这是最基础的动量/反转因子。

    - 短期收益率（5-10 bar）往往有反转效应（均值回复）
    - 中期收益率（30-60 bar）可能有动量效应（趋势延续）
    - 具体效应取决于市场状态和标的特性

    在 crypto 市场中，短期反转通常比中期动量更显著。
    但这需要通过因子评价来验证。

计算逻辑:
    returns_n = close_t / close_{t-n} - 1

    使用简单收益而非对数收益，因为:
    1. 简单收益在截面上可加（组合收益 = 加权和）
    2. 在实际价格变动范围内，两者差异极小

因子类型: 时序因子
数据需求: 1m OHLCV
输出频率: 1m

本文件通过 @register_factor_family 装饰器和 _param_grid 属性，
自动注册多个回溯窗口的因子变体，无需手动创建子类。

依赖: core.base, core.types
"""

import pandas as pd

from factor_research.core.base import TimeSeriesFactor
from factor_research.core.registry import register_factor_family
from factor_research.core.types import (
    DataRequest,
    DataType,
    FactorMeta,
    FactorType,
)


@register_factor_family
class MultiScaleReturns(TimeSeriesFactor):
    """
    多尺度收益率因子（参数化）

    通过 _param_grid 指定所有需要注册的 lookback 值，
    装饰器自动展开为多个因子实例。

    参数:
        lookback: 回溯窗口（K线数量）。
                  _param_grid 中列出的每个值都会注册为独立因子。
    """

    # 作者人工指定参数网格
    _param_grid = {"lookback": [5, 10, 30, 60]}

    def __init__(self, lookback: int = 5):
        self.lookback = lookback

    def meta(self) -> FactorMeta:
        n = self.lookback
        return FactorMeta(
            name=f"returns_{n}m",
            display_name=f"收益率 ({n}m)",
            factor_type=FactorType.TIME_SERIES,
            category="momentum",
            description=f"过去 {n} 根 1m K线的累计收益率",
            data_requirements=[
                DataRequest(DataType.OHLCV, timeframe="1m", lookback_bars=n),
            ],
            output_freq="1m",
            params={"lookback": n},
            family="multi_scale_returns",
            author="system",
            version="1.0",
        )

    def compute_single(self, symbol: str, data: dict) -> pd.Series:
        """
        计算单个标的的收益率

        Args:
            symbol: 交易对名称
            data:   {DataType.OHLCV: pd.DataFrame}

        Returns:
            pd.Series: 1m 频率的收益率序列
        """
        ohlcv = data[DataType.OHLCV]

        if ohlcv.empty or len(ohlcv) < self.lookback + 1:
            return pd.Series(dtype=float)

        close = ohlcv["close"]

        # N 根 K线的累计收益率
        returns = close / close.shift(self.lookback) - 1

        # 设置时间索引
        if "timestamp" in ohlcv.columns:
            returns.index = pd.to_datetime(ohlcv["timestamp"], utc=True)
        elif ohlcv.index.name == "timestamp":
            returns.index = ohlcv.index

        returns.name = self.meta().name
        return returns.dropna()
