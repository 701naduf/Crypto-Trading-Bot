"""
因子基类

所有因子必须继承此基类并实现 compute() 方法。
框架通过基类提供统一的接口，引擎和评价系统依赖此接口工作。

三种因子类型及其引擎行为:

1. TimeSeriesFactor (时序因子):
   - 子类只需实现 compute_single(symbol, data) -> pd.Series
   - 引擎自动对每个 symbol 分别调用，合并为面板
   - compute_single 的 data 参数: {DataType: pd.DataFrame}
     只包含该标的在 meta().data_requirements 中声明的数据
   - 例: BTC 的 5min 波动率

2. CrossSectionalFactor (截面因子):
   - 子类实现 compute(data) -> pd.DataFrame
   - 引擎传入全部标的数据，一次调用
   - compute 的 data 参数: {DataType: {symbol: pd.DataFrame}}
     外层 key 为 DataType，内层 key 为 symbol
   - 例: 5 个币种的动量排名

3. CrossAssetFactor (跨标的因子):
   - 子类实现 compute(data) -> pd.DataFrame
   - 需额外声明 input_symbols 和 output_symbols 属性
   - data 格式同截面因子，但只包含 input_symbols 的数据
   - 输出面板的列只包含 output_symbols
   - 例: BTC 涨跌 → ETH 的领先信号

依赖: core.types
被依赖: core.registry, core.engine, factors/
"""

from abc import ABC, abstractmethod

import pandas as pd

from .types import FactorMeta, FactorType, DataType


class Factor(ABC):
    """
    因子基类

    所有因子的公共接口。框架中的引擎、注册表、评价系统
    都通过此接口与具体因子交互。

    子类必须实现:
        meta()    — 返回因子元数据
        compute() — 计算因子值（或由子类基类提供默认实现）
    """

    @abstractmethod
    def meta(self) -> FactorMeta:
        """
        返回因子元数据

        元数据描述了因子的基本信息（名称、类型、数据需求等），
        引擎据此决定如何准备数据和调度计算。

        Returns:
            FactorMeta: 因子元数据实例
        """
        ...

    @abstractmethod
    def compute(self, data: dict) -> pd.DataFrame:
        """
        计算因子值

        Args:
            data: 引擎准备好的数据字典。
                  具体格式取决于因子类型，详见各子类说明。

        Returns:
            pd.DataFrame: 因子面板
                index:   DatetimeIndex (UTC)
                columns: symbol 列表（如 ["BTC/USDT", "ETH/USDT"]）
                values:  float64 因子值
        """
        ...


class TimeSeriesFactor(Factor):
    """
    时序因子基类

    最常用的因子类型。每个标的的因子值只依赖该标的自身的历史数据，
    标的之间互相独立。

    引擎行为:
        对每个 symbol 分别调用 compute_single()，然后自动合并为面板。
        子类只需实现 compute_single()，无需关心多标的合并逻辑。

    compute_single() 的 data 参数格式:
        {
            DataType.OHLCV: pd.DataFrame,       # 该标的的 OHLCV
            DataType.TICK: pd.DataFrame,         # 该标的的逐笔成交
            DataType.ORDERBOOK: pd.DataFrame,    # 该标的的订单簿
            ...
        }
        只包含 meta().data_requirements 中声明的数据类型。

    Examples:
        @register_factor
        class RealizedVolatility(TimeSeriesFactor):
            def meta(self) -> FactorMeta:
                return FactorMeta(
                    name="realized_vol_5m",
                    display_name="已实现波动率 (5m)",
                    factor_type=FactorType.TIME_SERIES,
                    category="volatility",
                    data_requirements=[DataRequest(DataType.OHLCV, timeframe="1m")],
                    output_freq="5m",
                    params={"window": 5},
                )

            def compute_single(self, symbol, data):
                ohlcv = data[DataType.OHLCV]
                returns = ohlcv["close"].pct_change()
                return returns.rolling(5).std()
    """

    def compute(self, data: dict) -> pd.DataFrame:
        """
        引擎调用此方法。遍历每个 symbol，调用 compute_single()，合并为面板。

        子类无需重写此方法。

        Args:
            data: {symbol: {DataType: pd.DataFrame}} 格式的数据字典。
                  外层 key 为 symbol，内层 key 为 DataType。
                  这个结构由引擎在调用前组装好。

        Returns:
            pd.DataFrame: 因子面板 (timestamp × symbol)
        """
        panels = {}
        for symbol, symbol_data in data.items():
            series = self.compute_single(symbol, symbol_data)
            panels[symbol] = series
        return pd.DataFrame(panels)

    @abstractmethod
    def compute_single(self, symbol: str, data: dict) -> pd.Series:
        """
        计算单个标的的因子值

        这是时序因子子类唯一需要实现的计算方法。

        Args:
            symbol: 交易对名称（如 "BTC/USDT"）
            data:   该标的的数据字典 {DataType: pd.DataFrame}
                    只包含 meta().data_requirements 中声明的数据类型

        Returns:
            pd.Series: index 为 DatetimeIndex (UTC)，values 为 float64 因子值。
                       引擎会自动将多个标的的 Series 合并为面板。
        """
        ...


class CrossSectionalFactor(Factor):
    """
    截面因子基类

    需要同时看到所有标的数据才能计算的因子，
    典型场景是截面排名或标准化。

    引擎行为:
        将全部标的的数据传入 compute()，一次调用得到完整面板。
        子类直接实现 compute()。

    compute() 的 data 参数格式:
        {
            DataType.OHLCV: {
                "BTC/USDT": pd.DataFrame,
                "ETH/USDT": pd.DataFrame,
                ...
            },
            DataType.TICK: {
                "BTC/USDT": pd.DataFrame,
                ...
            },
            ...
        }
        外层 key 为 DataType，内层 key 为 symbol。

    注意:
        在本项目 5 标的场景下，截面分析的统计意义有限。
        框架完整支持截面功能是为了扩展性和展示需要。
        实际研究以时序因子为主。

    Examples:
        @register_factor
        class MomentumRank(CrossSectionalFactor):
            def meta(self) -> FactorMeta:
                return FactorMeta(
                    name="momentum_rank_60m",
                    factor_type=FactorType.CROSS_SECTIONAL,
                    category="momentum",
                    ...
                )

            def compute(self, data):
                ohlcv_dict = data[DataType.OHLCV]
                # 计算每个标的 60 根 K线的累计收益
                returns = {}
                for sym, df in ohlcv_dict.items():
                    returns[sym] = df["close"].pct_change(60)
                panel = pd.DataFrame(returns)
                # 截面排名: 每个时刻对所有标的排名
                return panel.rank(axis=1, pct=True)
    """
    pass  # 直接使用 Factor.compute()


class CrossAssetFactor(Factor):
    """
    跨标的因子基类

    一种特殊类型的因子：输入数据来自特定标的（通常是 BTC），
    输出因子值对应其他标的。捕捉标的之间的领先-滞后和溢出效应。

    在 crypto 市场中，BTC 的主导地位使得跨标的效应尤为显著:
        - 领先-滞后: BTC 短期收益 → 预测 ETH/SOL 等的后续走势
        - 波动率传染: BTC 波动率突增 → 其他币种波动率跟随
        - 订单簿压力溢出: BTC 订单簿极度不平衡 → 其他币种跟随反应

    引擎行为:
        根据 input_symbols 读取数据，传入 compute()。
        输出面板的列只包含 output_symbols。

    compute() 的 data 参数格式:
        与 CrossSectionalFactor 相同，但只包含 input_symbols 的数据。

    子类需额外声明:
        input_symbols:  需要哪些标的的数据作为输入
        output_symbols: 输出因子值对应哪些标的

    Examples:
        @register_factor
        class BTCLeadLag(CrossAssetFactor):
            @property
            def input_symbols(self):
                return ["BTC/USDT"]

            @property
            def output_symbols(self):
                return ["ETH/USDT", "SOL/USDT", "BNB/USDT", "DOGE/USDT"]

            def compute(self, data):
                btc_ohlcv = data[DataType.OHLCV]["BTC/USDT"]
                btc_ret = btc_ohlcv["close"].pct_change(5)
                # BTC 过去 5 根 K线收益 → 作为其他币种的领先信号
                panel = pd.DataFrame(
                    {sym: btc_ret for sym in self.output_symbols},
                    index=btc_ohlcv["timestamp"],
                )
                return panel
    """

    @property
    @abstractmethod
    def input_symbols(self) -> list[str]:
        """
        需要哪些标的的数据作为输入

        引擎只读取这些标的的数据传入 compute()。
        通常是 ["BTC/USDT"]。

        Returns:
            标的名称列表
        """
        ...

    @property
    @abstractmethod
    def output_symbols(self) -> list[str]:
        """
        输出因子值对应哪些标的

        输出面板的列只包含这些标的。
        通常是除 BTC 外的其他标的。

        Returns:
            标的名称列表
        """
        ...
