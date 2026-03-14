"""
因子批量计算引擎

负责协调因子计算的完整流程:
    1. 根据因子的 DataRequest 从 DataReader 读取原始数据
    2. 根据因子类型组织数据格式
    3. 调用因子的 compute() 方法
    4. 将计算结果写入 FactorStore

数据准备策略（由因子类型决定）:
    - TimeSeriesFactor:
        数据组织: {symbol: {DataType: DataFrame}}
        引擎对每个 symbol 分别准备数据，传给 compute()
        compute() 内部自动遍历 symbols 调用 compute_single()

    - CrossSectionalFactor:
        数据组织: {DataType: {symbol: DataFrame}}
        引擎准备所有 symbol 的数据，整体传给 compute()

    - CrossAssetFactor:
        数据组织: 同截面因子，但只包含 input_symbols 的数据
        输出面板只包含 output_symbols

引擎同时支持:
    - 单因子计算（notebook 场景）: engine.compute_factor("my_factor")
    - 因子实例计算（探索场景）: engine.compute_factor_instance(factor)
    - 因子族计算（共享数据）:   engine.compute_family("multi_scale_returns")
    - 批量计算（管道场景）:     engine.compute_all()

依赖: core.base, core.registry, store.factor_store,
      data_infra.data.reader.DataReader
"""

from datetime import datetime

import pandas as pd

from data_infra.data.reader import DataReader
from data_infra.utils.logger import get_logger

from ..config import DEFAULT_SYMBOLS

from .base import CrossAssetFactor, CrossSectionalFactor, Factor, TimeSeriesFactor
from .registry import FactorRegistry, get_default_registry
from .types import DataType, FactorType
from ..store.factor_store import FactorStore

logger = get_logger(__name__)

# DataType → DataReader 方法名的映射
_DATA_TYPE_TO_READER_METHOD = {
    DataType.OHLCV: "get_ohlcv",
    DataType.TICK: "get_ticks",
    DataType.ORDERBOOK: "get_orderbook",
    DataType.FUNDING_RATE: "get_funding_rate",
    DataType.OPEN_INTEREST: "get_open_interest",
    DataType.LONG_SHORT_RATIO: "get_long_short_ratio",
    DataType.TAKER_BUY_SELL: "get_taker_buy_sell",
}


class FactorEngine:
    """
    因子计算引擎

    协调数据读取 → 因子计算 → 结果存储的完整流程。
    """

    def __init__(
        self,
        reader: DataReader = None,
        store: FactorStore = None,
        registry: FactorRegistry = None,
    ):
        """
        初始化引擎

        Args:
            reader:   DataReader 实例（默认新建）
            store:    FactorStore 实例（默认新建）
            registry: FactorRegistry 实例（默认使用全局注册表）
        """
        self._reader = reader if reader is not None else DataReader()
        self._store = store if store is not None else FactorStore()
        self._registry = registry if registry is not None else get_default_registry()

        logger.debug("FactorEngine 已初始化")

    def compute_factor(
        self,
        factor_name: str,
        symbols: list[str] = None,
        start: datetime = None,
        end: datetime = None,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        计算单个因子

        完整流程: 获取因子实例 → 准备数据 → 计算 → 可选保存。

        Args:
            factor_name: 因子名称（必须已通过 @register_factor 注册）
            symbols:     标的列表（默认从 settings.SYMBOLS 读取）
            start/end:   时间范围
            save:        是否存入 FactorStore

        Returns:
            pd.DataFrame: 因子面板 (index=timestamp, columns=symbols)

        Raises:
            KeyError: 如果因子未注册
        """
        factor = self._registry.get(factor_name)
        return self._compute_factor_impl(factor, symbols, start, end, save)

    def compute_factor_instance(
        self,
        factor: Factor,
        symbols: list[str] = None,
        start: datetime = None,
        end: datetime = None,
        save: bool = False,
    ) -> pd.DataFrame:
        """
        计算任意因子实例（无需注册）

        用于 notebook 中快速探索未注册的参数组合。
        与 compute_factor() 逻辑相同，但接受实例而非名称。

        Args:
            factor: 已实例化的因子对象（如 MultiScaleReturns(lookback=45)）
            symbols: 标的列表
            start/end: 时间范围
            save:   默认 False（探索性计算通常不保存）

        Returns:
            pd.DataFrame: 因子面板
        """
        return self._compute_factor_impl(factor, symbols, start, end, save)

    def prepare_data(
        self,
        factor: Factor,
        symbols: list[str] = None,
        start: datetime = None,
        end: datetime = None,
    ) -> dict:
        """
        暴露数据准备接口（原 _prepare_data 的公开版本）

        用于 FamilyAnalyzer 等需要复用数据的场景:
            data = engine.prepare_data(MultiScaleReturns(), symbols=SYMBOLS)
            family = FamilyAnalyzer(MultiScaleReturns, data=data, price_panel=prices)

        因子族内所有变体通常需要相同的数据类型，
        因此只需用默认参数实例准备一次数据，即可在 sweep 中复用。

        Args:
            factor:  因子实例（用于确定数据需求）
            symbols: 标的列表
            start/end: 时间范围

        Returns:
            dict: 按因子类型组织的数据字典
        """
        if symbols is None:
            symbols = DEFAULT_SYMBOLS
        return self._prepare_data(factor, symbols, start, end)

    def compute_family(
        self,
        family_name: str,
        symbols: list[str] = None,
        start: datetime = None,
        end: datetime = None,
        save: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        计算指定因子族的所有变体（共享数据准备）

        核心优化: 同族因子的 data_requirements 相同或兼容，
        数据只准备一次，所有变体复用。

        Args:
            family_name: 族名
            symbols:     标的列表
            start/end:   时间范围
            save:        是否存入 FactorStore

        Returns:
            {factor_name: 因子面板} 字典

        Raises:
            KeyError: 如果族名不存在
        """
        metas = self._registry.list_family(family_name)
        if not metas:
            raise KeyError(f"因子族 '{family_name}' 不存在或无已注册变体")

        if symbols is None:
            symbols = DEFAULT_SYMBOLS

        # 用第一个变体的 data_requirements 准备数据（同族共享）
        first_factor = self._registry.get(metas[0].name)
        data = self._prepare_data(first_factor, symbols, start, end)

        logger.info(
            f"开始计算因子族: {family_name} | "
            f"{len(metas)} 个变体 | 数据准备 1 次"
        )

        results = {}
        for meta in metas:
            try:
                factor = self._registry.get(meta.name)
                panel = factor.compute(data)
                if panel.index.name is None:
                    panel.index.name = "timestamp"
                if save and not panel.empty:
                    self._store.save(meta.name, panel, meta)
                results[meta.name] = panel
            except Exception as e:
                logger.error(f"因子 {meta.name} 计算失败: {e}")

        logger.info(f"因子族计算完成: {len(results)}/{len(metas)} 成功")
        return results

    def compute_all(
        self,
        symbols: list[str] = None,
        start: datetime = None,
        end: datetime = None,
        categories: list[str] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        批量计算所有已注册因子（或指定分类）

        自动按 family 分组，族内共享数据准备，消除重复 I/O。
        独立因子（family=""）逐个计算。

        Args:
            symbols:    标的列表
            start/end:  时间范围
            categories: 只计算指定分类的因子，None 则计算全部

        Returns:
            {factor_name: 因子面板} 字典
        """
        # 获取要计算的因子列表
        if categories:
            metas = []
            for cat in categories:
                metas.extend(self._registry.list_by_category(cat))
        else:
            metas = self._registry.list_all()

        if not metas:
            logger.warning("没有找到要计算的因子")
            return {}

        if symbols is None:
            symbols = DEFAULT_SYMBOLS

        logger.info(f"批量计算 {len(metas)} 个因子...")

        # 按 family 分组，共享数据准备
        family_groups: dict[str, list] = {}   # {family_name: [meta, ...]}
        standalone: list = []                  # family 为空的独立因子
        for meta in metas:
            if meta.family:
                family_groups.setdefault(meta.family, []).append(meta)
            else:
                standalone.append(meta)

        results = {}

        # 族内共享数据
        for family_name, family_metas in family_groups.items():
            first_factor = self._registry.get(family_metas[0].name)
            data = self._prepare_data(first_factor, symbols, start, end)
            for meta in family_metas:
                try:
                    factor = self._registry.get(meta.name)
                    panel = factor.compute(data)
                    if panel.index.name is None:
                        panel.index.name = "timestamp"
                    if not panel.empty:
                        self._store.save(meta.name, panel, meta)
                    results[meta.name] = panel
                except Exception as e:
                    logger.error(f"因子 {meta.name} 计算失败: {e}")

        # 独立因子逐个处理
        for meta in standalone:
            try:
                panel = self.compute_factor(
                    meta.name, symbols=symbols, start=start, end=end, save=True
                )
                results[meta.name] = panel
            except Exception as e:
                logger.error(f"因子 {meta.name} 计算失败: {e}")

        logger.info(f"批量计算完成: {len(results)}/{len(metas)} 成功")
        return results

    # ------------------------------------------------------------------
    # 私有方法
    # ------------------------------------------------------------------

    def _compute_factor_impl(
        self,
        factor: Factor,
        symbols: list[str],
        start: datetime,
        end: datetime,
        save: bool,
    ) -> pd.DataFrame:
        """
        因子计算的内部实现（compute_factor 和 compute_factor_instance 共用）

        Args:
            factor:  因子实例
            symbols: 标的列表
            start/end: 时间范围
            save:    是否存入 FactorStore
        """
        meta = factor.meta()

        if symbols is None:
            symbols = DEFAULT_SYMBOLS

        logger.info(
            f"开始计算因子: {meta.name} | "
            f"类型: {meta.factor_type.value} | "
            f"标的: {len(symbols)} 个 | "
            f"时间: {start} ~ {end}"
        )

        # 1. 准备数据
        data = self._prepare_data(factor, symbols, start, end)

        # 2. 计算
        panel = factor.compute(data)

        # 3. 确保 index 名称
        if panel.index.name is None:
            panel.index.name = "timestamp"

        logger.info(
            f"因子计算完成: {meta.name} | "
            f"{panel.shape[0]} 行 × {panel.shape[1]} 列"
        )

        # 4. 可选保存
        if save and not panel.empty:
            self._store.save(meta.name, panel, meta)

        return panel

    def _prepare_data(
        self,
        factor: Factor,
        symbols: list[str],
        start: datetime,
        end: datetime,
    ) -> dict:
        """
        根据因子的 DataRequest 从 DataReader 读取并组织数据

        不同因子类型的数据组织方式:

        TimeSeriesFactor:
            返回 {symbol: {DataType: DataFrame}}
            这样 compute() 可以遍历 symbols 调用 compute_single()

        CrossSectionalFactor:
            返回 {DataType: {symbol: DataFrame}}
            所有 symbol 的数据按 DataType 分组

        CrossAssetFactor:
            同截面因子，但只读取 input_symbols 的数据
        """
        meta = factor.meta()

        if isinstance(factor, TimeSeriesFactor):
            return self._prepare_timeseries_data(meta, symbols, start, end)
        elif isinstance(factor, CrossAssetFactor):
            return self._prepare_cross_asset_data(factor, meta, start, end)
        elif isinstance(factor, CrossSectionalFactor):
            return self._prepare_cross_sectional_data(meta, symbols, start, end)
        else:
            # 默认按截面处理
            return self._prepare_cross_sectional_data(meta, symbols, start, end)

    def _prepare_timeseries_data(
        self, meta, symbols, start, end
    ) -> dict:
        """
        为时序因子准备数据

        返回: {symbol: {DataType: DataFrame}}
        """
        result = {}
        for symbol in symbols:
            symbol_data = {}
            for req in meta.data_requirements:
                df = self._read_data(req, symbol, start, end)
                symbol_data[req.data_type] = df
            result[symbol] = symbol_data
        return result

    def _prepare_cross_sectional_data(
        self, meta, symbols, start, end
    ) -> dict:
        """
        为截面因子准备数据

        返回: {DataType: {symbol: DataFrame}}
        """
        result = {}
        for req in meta.data_requirements:
            symbol_data = {}
            for symbol in symbols:
                df = self._read_data(req, symbol, start, end)
                symbol_data[symbol] = df
            result[req.data_type] = symbol_data
        return result

    def _prepare_cross_asset_data(
        self, factor: CrossAssetFactor, meta, start, end
    ) -> dict:
        """
        为跨标的因子准备数据

        只读取 input_symbols 的数据。
        返回: {DataType: {symbol: DataFrame}}
        """
        input_symbols = factor.input_symbols
        return self._prepare_cross_sectional_data(meta, input_symbols, start, end)

    def _read_data(self, req, symbol, start, end) -> pd.DataFrame:
        """
        根据单个 DataRequest 从 DataReader 读取数据

        将 DataRequest 的参数映射到 DataReader 的方法调用。
        """
        method_name = _DATA_TYPE_TO_READER_METHOD[req.data_type]
        method = getattr(self._reader, method_name)

        kwargs = {}
        if start is not None:
            kwargs["start"] = start
        if end is not None:
            kwargs["end"] = end

        # OHLCV 需要 timeframe 参数
        if req.data_type == DataType.OHLCV:
            timeframe = req.timeframe or "1m"
            return method(symbol, timeframe, **kwargs)

        # ORDERBOOK 需要 levels 参数
        if req.data_type == DataType.ORDERBOOK and req.orderbook_levels:
            kwargs["levels"] = req.orderbook_levels

        return method(symbol, **kwargs)
