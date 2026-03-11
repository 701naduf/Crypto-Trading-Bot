"""
统一数据写入模块

所有采集脚本写入数据的唯一入口。
写入前调用 validator 校验，不合格数据被过滤并记录日志。

设计理念:
    采集脚本不直接操作 Store，而是通过 DataWriter 间接写入。
    DataWriter 负责:
        1. 写入前校验（调用 validator）
        2. 路由到正确的 Store
        3. 提供断点续传所需的查询接口（最新时间、最新 trade_id）

    这样采集脚本只需关注「拉取数据」，不需要关心校验和存储细节。

依赖: data.kline_store, data.tick_store, data.orderbook_store,
      data.market_store, data.validator
"""

from datetime import datetime

from data.kline_store import KlineStore
from data.tick_store import TickStore
from data.orderbook_store import OrderbookStore
from data.market_store import MarketStore
from data import validator
from utils.logger import get_logger

logger = get_logger(__name__)


class DataWriter:
    """
    统一数据写入器

    聚合四种 Store，提供统一的写入接口和断点查询能力。
    每个采集脚本创建一个 DataWriter 实例即可。
    """

    def __init__(self):
        """
        初始化所有底层 Store 实例

        各 Store 会自动创建数据库/目录/表结构。
        """
        self._kline_store = KlineStore()
        self._tick_store = TickStore()
        self._orderbook_store = OrderbookStore()
        self._market_store = MarketStore()

        logger.debug("DataWriter 已初始化")

    # ===================================================================
    # K线写入
    # ===================================================================

    def write_ohlcv(self, df, symbol: str, timeframe: str) -> int:
        """
        校验并写入 K线 数据

        Args:
            df:        OHLCV DataFrame
            symbol:    交易对
            timeframe: K线周期

        Returns:
            新增行数
        """
        if df.empty:
            return 0

        # 写入前校验
        valid_df, invalid_df = validator.validate_ohlcv(df)

        if valid_df.empty:
            logger.warning(f"{symbol} {timeframe}: 全部 K线 未通过校验")
            return 0

        return self._kline_store.write(valid_df, symbol, timeframe)

    # ===================================================================
    # 逐笔成交写入
    # ===================================================================

    def write_ticks(self, df, symbol: str) -> int:
        """
        校验并写入逐笔成交数据

        Args:
            df:     Tick DataFrame
            symbol: 交易对

        Returns:
            新增记录数
        """
        if df.empty:
            return 0

        valid_df, invalid_df = validator.validate_ticks(df)

        if valid_df.empty:
            logger.warning(f"{symbol}: 全部 tick 未通过校验")
            return 0

        return self._tick_store.write(valid_df, symbol)

    # ===================================================================
    # 订单簿写入
    # ===================================================================

    def append_orderbook(self, symbol: str, snapshot: dict):
        """
        追加订单簿快照到缓冲

        Args:
            symbol:   交易对
            snapshot: 订单簿快照 dict
        """
        # 校验
        from config import settings
        if not validator.validate_orderbook(snapshot, settings.ORDERBOOK_DEPTH):
            return

        self._orderbook_store.append(symbol, snapshot)

    def flush_orderbook(self, symbol: str = None):
        """
        刷盘订单簿缓冲

        Args:
            symbol: 指定币对，None 刷盘所有
        """
        self._orderbook_store.flush(symbol)

    def flush_and_close_orderbook(self):
        """退出前刷盘所有订单簿缓冲"""
        self._orderbook_store.flush_and_close()

    # ===================================================================
    # 合约市场数据写入
    # ===================================================================

    def write_funding_rate(self, df, symbol: str) -> int:
        """写入资金费率"""
        if df.empty:
            return 0

        valid_df, _ = validator.validate_market_data(df, "funding_rate")
        if valid_df.empty:
            return 0

        return self._market_store.write_funding_rate(valid_df, symbol)

    def write_open_interest(self, symbol: str, data: dict) -> int:
        """校验并写入持仓量快照"""
        if not validator.validate_open_interest(data):
            logger.warning(f"{symbol}: 持仓量数据未通过校验，跳过写入")
            return 0

        return self._market_store.write_open_interest(symbol, data)

    def write_long_short_ratio(self, df, symbol: str) -> int:
        """写入多空持仓比"""
        if df.empty:
            return 0

        valid_df, _ = validator.validate_market_data(df, "long_short_ratio")
        if valid_df.empty:
            return 0

        return self._market_store.write_long_short_ratio(valid_df, symbol)

    def write_taker_buy_sell(self, df, symbol: str) -> int:
        """写入主动买卖量"""
        if df.empty:
            return 0

        valid_df, _ = validator.validate_market_data(df, "taker_buy_sell")
        if valid_df.empty:
            return 0

        return self._market_store.write_taker_buy_sell(valid_df, symbol)

    # ===================================================================
    # 断点续传查询接口
    # ===================================================================

    def get_latest_kline_time(
        self, symbol: str, timeframe: str
    ) -> datetime | None:
        """获取最新 K线 时间，用于增量采集"""
        return self._kline_store.get_latest_timestamp(symbol, timeframe)

    def get_latest_trade_id(self, symbol: str) -> int | None:
        """获取最新 trade_id，用于 tick 增量采集"""
        return self._tick_store.get_latest_trade_id(symbol)

    def get_latest_market_time(
        self, table: str, symbol: str
    ) -> datetime | None:
        """获取指定市场数据表的最新时间"""
        return self._market_store.get_latest_timestamp(table, symbol)
