"""
统一数据读取模块

上层模块（因子、策略、回测、执行）获取数据的唯一接口。
封装了所有存储细节和数据转换逻辑，调用方只需指定「要什么数据」。

路由逻辑:
    get_ohlcv("BTC/USDT", "1m")   → SQLite 直接读取
    get_ohlcv("BTC/USDT", "5m")   → SQLite 读 1m + resample_ohlcv
    get_ohlcv("BTC/USDT", "1h")   → SQLite 读 1m + resample_ohlcv
    get_ohlcv("BTC/USDT", "10s")  → Parquet 读 tick + aggregate_ticks_to_ohlcv
    get_ticks("BTC/USDT")         → Parquet 直接读取
    get_orderbook("BTC/USDT")     → Parquet 直接读取
    get_funding_rate("BTC/USDT")  → SQLite 直接读取
    ...

亚分钟 OHLCV 生成策略（防 OOM）:
    大量 tick 数据一次性加载可能导致内存溢出。
    对于亚分钟周期（如 10s），按天分片处理:
        1. 确定日期范围
        2. 逐天读取 tick → 聚合为 OHLCV → 合并结果
    这样内存中同时只有一天的 tick 数据。

依赖: data.kline_store, data.tick_store, data.orderbook_store,
      data.market_store, data.aggregator
被依赖: 所有上层模块
"""

from datetime import datetime, timedelta, timezone

import pandas as pd

from data_infra.data.kline_store import KlineStore
from data_infra.data.tick_store import TickStore
from data_infra.data.orderbook_store import OrderbookStore
from data_infra.data.market_store import MarketStore
from data_infra.data.aggregator import aggregate_ticks_to_ohlcv, resample_ohlcv
from data_infra.utils.logger import get_logger
from data_infra.utils.time_utils import is_standard_timeframe

logger = get_logger(__name__)


class DataReader:
    """
    统一数据读取器

    上层模块的唯一数据入口。屏蔽存储细节，自动路由数据源，
    按需进行降采样或聚合。
    """

    def __init__(self):
        """初始化所有底层 Store 实例"""
        self._kline_store = KlineStore()
        self._tick_store = TickStore()
        self._orderbook_store = OrderbookStore()
        self._market_store = MarketStore()

        logger.debug("DataReader 已初始化")

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime = None,
        end: datetime = None,
    ) -> pd.DataFrame:
        """
        获取 OHLCV K线 数据（自动路由数据源 + 聚合）

        路由规则:
            1. "1m" → 直接从 SQLite 读取
            2. 标准周期（"5m", "15m", "1h" 等）→ 读 1m + 降采样
            3. 亚分钟周期（"10s", "30s"）→ 读 tick + 聚合（按天分片防OOM）

        Args:
            symbol:    交易对
            timeframe: K线周期
            start:     起始时间（包含）
            end:       结束时间（包含）

        Returns:
            OHLCV DataFrame [timestamp, open, high, low, close, volume]
        """
        # 路径1: 直接读取 1m
        if timeframe == "1m":
            return self._kline_store.read(symbol, "1m", start, end)

        # 路径2: 标准周期 → 从 1m 降采样
        if is_standard_timeframe(timeframe):
            df_1m = self._kline_store.read(symbol, "1m", start, end)
            if df_1m.empty:
                return df_1m
            return resample_ohlcv(df_1m, "1m", timeframe)

        # 路径3: 亚分钟周期 → 从 tick 聚合（按天分片防 OOM）
        return self._ohlcv_from_ticks(symbol, timeframe, start, end)

    def _ohlcv_from_ticks(
        self,
        symbol: str,
        timeframe: str,
        start: datetime = None,
        end: datetime = None,
    ) -> pd.DataFrame:
        """
        从 tick 数据聚合生成亚分钟 OHLCV（按天分片防 OOM）

        流程:
            1. 确定日期范围
            2. 逐天: 读取当天 tick → 聚合为 OHLCV
            3. 合并所有天的结果

        这样内存中同时只有一天的 tick 数据（约 50-100MB 压缩前），
        避免一次性加载几个月的 tick 导致 OOM。
        """
        # 确定日期范围
        if start is None:
            start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        if end is None:
            end = datetime.now(timezone.utc)

        all_frames = []
        current_date = start.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = end.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

        while current_date < end_date:
            next_date = current_date + timedelta(days=1)

            # 读取当天的 tick 数据
            day_ticks = self._tick_store.read(symbol, current_date, next_date)

            if not day_ticks.empty:
                # 聚合为 OHLCV
                day_ohlcv = aggregate_ticks_to_ohlcv(day_ticks, timeframe)
                if not day_ohlcv.empty:
                    all_frames.append(day_ohlcv)

            current_date = next_date

        if not all_frames:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        result = pd.concat(all_frames, ignore_index=True)

        # 精确过滤时间范围
        if start:
            result = result[result["timestamp"] >= pd.Timestamp(start, tz="UTC")]
        if end:
            result = result[result["timestamp"] <= pd.Timestamp(end, tz="UTC")]

        return result.sort_values("timestamp").reset_index(drop=True)

    def get_ticks(
        self,
        symbol: str,
        start: datetime = None,
        end: datetime = None,
    ) -> pd.DataFrame:
        """
        获取逐笔成交数据

        Args:
            symbol: 交易对
            start:  起始时间
            end:    结束时间

        Returns:
            DataFrame [trade_id, timestamp, price, amount, side]
        """
        return self._tick_store.read(symbol, start, end)

    def get_orderbook(
        self,
        symbol: str,
        start: datetime = None,
        end: datetime = None,
        levels: int = None,
    ) -> pd.DataFrame:
        """
        获取订单簿快照序列

        Args:
            symbol: 交易对
            start:  起始时间
            end:    结束时间
            levels: 只返回前 N 档，None 返回全部
                    例如 levels=5 只读买卖各 5 档，减少内存占用

        Returns:
            DataFrame，包含 timestamp + 各档位价格/数量列
        """
        return self._orderbook_store.read(symbol, start, end, levels)

    def get_funding_rate(
        self,
        symbol: str,
        start: datetime = None,
        end: datetime = None,
    ) -> pd.DataFrame:
        """获取资金费率"""
        return self._market_store.read("funding_rates", symbol, start, end)

    def get_open_interest(
        self,
        symbol: str,
        start: datetime = None,
        end: datetime = None,
    ) -> pd.DataFrame:
        """获取持仓量"""
        return self._market_store.read("open_interest", symbol, start, end)

    def get_long_short_ratio(
        self,
        symbol: str,
        start: datetime = None,
        end: datetime = None,
    ) -> pd.DataFrame:
        """获取多空持仓比"""
        return self._market_store.read("long_short_ratio", symbol, start, end)

    def get_taker_buy_sell(
        self,
        symbol: str,
        start: datetime = None,
        end: datetime = None,
    ) -> pd.DataFrame:
        """获取主动买卖量"""
        return self._market_store.read("taker_buy_sell", symbol, start, end)
