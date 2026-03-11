"""
逐笔成交数据拉取模块（异步）

核心机制 —— 基于 trade_id 的追赶模式:
    Binance 每笔成交都有一个全局递增的 trade_id。
    通过 fromId 参数从上次停止的位置继续拉取。
    满额（1000条）则立即继续，不满额说明已追上实时。

    这种机制保证零遗漏:
    - 不依赖时间戳（时间可能有精度问题）
    - 不依赖轮询间隔（高峰期不会遗漏）
    - trade_id 是递增的，即使有断号也不影响

关于 trade_id 连续性:
    Binance 的 trade_id 全局递增但不保证连续（可能存在断号）。
    断号是交易所内部行为，不代表数据丢失。
    我们的追赶逻辑基于 fromId（从此 ID 之后获取），不要求连续。

为什么用异步:
    tick 采集需要持续高频请求，异步可以在等待 API 响应时处理其他任务。
    虽然当前是串行处理各币对，但异步框架便于未来并行采集。

数据流: 交易所 API → ccxt.async → DataFrame
依赖: config.settings, utils.logger, utils.retry
被依赖: scripts/collect_ticks.py
"""

from datetime import datetime, timezone

import ccxt.async_support as ccxt_async
import pandas as pd

from config import settings
from utils.logger import get_logger
from utils.retry import async_retry_on_failure
from utils.time_utils import datetime_to_ms

logger = get_logger(__name__)


class TickFetcher:
    """
    逐笔成交数据拉取器（异步），基于 trade_id 追赶模式

    使用 ccxt 的异步版本，需要在 asyncio 事件循环中运行。
    使用完毕后必须调用 close() 释放连接资源。
    """

    def __init__(self, exchange_id: str = None):
        """
        初始化异步拉取器

        Args:
            exchange_id: 交易所标识，默认 settings.EXCHANGE_ID
        """
        exchange_id = exchange_id or settings.EXCHANGE_ID

        exchange_class = getattr(ccxt_async, exchange_id)

        config = {
            "apiKey": settings.API_KEY,
            "secret": settings.API_SECRET,
            "timeout": settings.REQUEST_TIMEOUT * 1000,
            "enableRateLimit": True,
            "options": {
                "fetchCurrencies": False,  # 禁用 SAPI 货币请求，避免不可达端点
            },
        }

        # ccxt 异步版基于 aiohttp，代理配置使用 aiohttp_proxy 字段
        if settings.PROXY_HOST and settings.PROXY_PORT:
            proxy_url = f"http://{settings.PROXY_HOST}:{settings.PROXY_PORT}"
            config["aiohttp_proxy"] = proxy_url

        self.exchange = exchange_class(config)
        logger.info(f"Tick 拉取器已初始化 (异步): {exchange_id}")

    @async_retry_on_failure()
    async def fetch_trades(
        self, symbol: str, from_id: int = None, limit: int = None
    ) -> pd.DataFrame:
        """
        从指定 trade_id 之后拉取一批成交数据

        Args:
            symbol:  交易对，如 "BTC/USDT"
            from_id: 起始 trade_id（不包含），拉取此 ID 之后的数据
                     None 表示从最新开始
            limit:   最大返回条数，默认 settings.TICK_FETCH_LIMIT

        Returns:
            DataFrame [trade_id, timestamp, price, amount, side]
            按 trade_id 升序排列

        ccxt fetch_trades 返回格式:
            [
                {
                    "id": "123456",        # trade_id（字符串）
                    "timestamp": 1705305600000,
                    "datetime": "2024-01-15T08:00:00.000Z",
                    "symbol": "BTC/USDT",
                    "side": "buy",
                    "price": 42000.5,
                    "amount": 0.001,
                    ...
                },
                ...
            ]
        """
        if limit is None:
            limit = settings.TICK_FETCH_LIMIT

        # ccxt 的 params 参数用于传递交易所特有的 API 参数
        # Binance 的 fromId 参数: 从此 ID 之后开始获取
        params = {}
        if from_id is not None:
            params["fromId"] = from_id

        raw = await self.exchange.fetch_trades(
            symbol, limit=limit, params=params
        )

        if not raw:
            return pd.DataFrame(
                columns=["trade_id", "timestamp", "price", "amount", "side"]
            )

        # 转为标准化 DataFrame
        records = []
        for trade in raw:
            records.append({
                "trade_id": int(trade["id"]),
                "timestamp": pd.Timestamp(trade["timestamp"], unit="ms", tz="UTC"),
                "price": float(trade["price"]),
                "amount": float(trade["amount"]),
                "side": trade["side"],  # "buy" 或 "sell"
            })

        df = pd.DataFrame(records)
        df = df.sort_values("trade_id").reset_index(drop=True)

        logger.debug(
            f"{symbol}: 拉取 {len(df)} 笔成交, "
            f"trade_id {df['trade_id'].iloc[0]} ~ {df['trade_id'].iloc[-1]}"
        )

        return df

    async def fetch_until_latest(
        self, symbol: str, from_id: int = None
    ) -> pd.DataFrame:
        """
        持续拉取直到追上最新数据

        从 from_id 开始，反复调用 fetch_trades 直到返回不满额。
        满额（limit 条）说明还有更多数据，立即继续拉取。
        不满额说明已追上实时。

        Args:
            symbol:  交易对
            from_id: 起始 trade_id

        Returns:
            本次追赶期间拉取的所有成交数据（合并后的 DataFrame）

        注意:
            如果从很早的 trade_id 开始追赶，可能需要很长时间。
            建议在外层分批调用并定期写入存储，而不是等全部追完再写。
        """
        all_frames = []
        current_id = from_id
        total = 0
        limit = settings.TICK_FETCH_LIMIT

        while True:
            df = await self.fetch_trades(symbol, from_id=current_id, limit=limit)

            if df.empty:
                break

            all_frames.append(df)
            total += len(df)

            # 更新 from_id 为本批最大 trade_id
            current_id = int(df["trade_id"].iloc[-1])

            # 不满额说明已追上实时
            if len(df) < limit:
                break

        if not all_frames:
            return pd.DataFrame(
                columns=["trade_id", "timestamp", "price", "amount", "side"]
            )

        result = pd.concat(all_frames, ignore_index=True)
        result = result.drop_duplicates(subset=["trade_id"]).reset_index(drop=True)

        logger.info(
            f"{symbol}: 追赶完成, 共 {len(result)} 笔, "
            f"trade_id {result['trade_id'].iloc[0]} ~ {result['trade_id'].iloc[-1]}"
        )

        return result

    async def resolve_cold_start(self, symbol: str) -> int | None:
        """
        冷启动时确定起始 trade_id

        首次采集某个币对时，本地没有历史 trade_id，需要确定从哪里开始。

        根据 settings.TICK_COLD_START_MODE:
            "latest":    获取最新一批 trades，取最小 trade_id。
                         放弃历史数据，从当前时刻开始采集。
                         适合快速启动。

            "from_date": 通过 startTime 参数获取指定日期的第一批 trades。
                         从指定日期开始追赶，需要较长时间赶上实时。
                         适合需要历史数据的场景。

        Args:
            symbol: 交易对

        Returns:
            起始 trade_id，如果无法确定则返回 None
        """
        mode = settings.TICK_COLD_START_MODE

        if mode == "latest":
            # 获取最新一批 trades（不带 fromId 参数）
            raw = await self.exchange.fetch_trades(symbol, limit=10)
            if not raw:
                logger.warning(f"{symbol}: 冷启动获取最新 trades 为空")
                return None

            # 取最小的 trade_id 作为起点
            first_id = min(int(t["id"]) for t in raw)
            logger.info(
                f"{symbol}: 冷启动 (latest 模式), 起始 trade_id = {first_id}"
            )
            return first_id

        elif mode == "from_date":
            # 通过 startTime 获取指定日期的第一批 trades
            start_date = datetime.strptime(
                settings.TICK_COLD_START_DATE, "%Y-%m-%d"
            ).replace(tzinfo=timezone.utc)

            start_ms = datetime_to_ms(start_date)

            raw = await self.exchange.fetch_trades(
                symbol, limit=10, params={"startTime": start_ms}
            )
            if not raw:
                logger.warning(
                    f"{symbol}: 冷启动 from_date={settings.TICK_COLD_START_DATE} 无数据"
                )
                return None

            first_id = min(int(t["id"]) for t in raw)
            logger.info(
                f"{symbol}: 冷启动 (from_date 模式), "
                f"日期={settings.TICK_COLD_START_DATE}, "
                f"起始 trade_id = {first_id}"
            )
            return first_id

        else:
            raise ValueError(f"不支持的冷启动模式: {mode}")

    async def close(self):
        """
        关闭异步交易所连接

        释放底层的 aiohttp Session。
        必须在使用完毕后调用，否则会有资源泄漏警告。
        """
        await self.exchange.close()
        logger.debug("Tick 拉取器已关闭")
