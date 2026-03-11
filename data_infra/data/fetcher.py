"""
K线数据拉取模块（同步）

通过 ccxt 统一接口从交易所 REST API 获取 OHLCV (K线) 数据。
返回标准化的 pandas DataFrame，不负责存储（存储由 DataWriter 处理）。

ccxt 的 fetch_ohlcv 接口:
    exchange.fetch_ohlcv(symbol, timeframe, since, limit)
    - symbol:    "BTC/USDT"
    - timeframe: "1m", "5m", "1h" 等
    - since:     起始时间（毫秒时间戳），None 表示最新
    - limit:     最大返回条数，Binance 上限 1000

    返回格式: [[timestamp_ms, open, high, low, close, volume], ...]

数据流: 交易所 API → ccxt → DataFrame
依赖: config.settings, utils.logger, utils.retry, utils.time_utils
被依赖: scripts/collect_klines.py, scripts/backfill.py
"""

from datetime import datetime

import ccxt
import pandas as pd

from data_infra.config import settings
from data_infra.utils.logger import get_logger
from data_infra.utils.retry import retry_on_failure
from data_infra.utils.time_utils import datetime_to_ms, timeframe_to_seconds

logger = get_logger(__name__)


class KlineFetcher:
    """
    K线数据拉取器

    封装 ccxt 交易所实例的创建和 K线 数据请求逻辑。
    支持单次拉取和批量历史拉取（自动分页）。
    """

    def __init__(self, exchange_id: str = None):
        """
        初始化拉取器，创建 ccxt 交易所实例

        Args:
            exchange_id: 交易所标识，默认使用 settings.EXCHANGE_ID
                         支持 "binance", "okx" 等 ccxt 支持的交易所

        交易所实例配置:
            - API Key / Secret: 从 settings 读取
            - 超时: settings.REQUEST_TIMEOUT
            - 代理: settings.PROXY_HOST / PROXY_PORT（如配置）
        """
        exchange_id = exchange_id or settings.EXCHANGE_ID

        # 动态创建 ccxt 交易所实例
        # getattr(ccxt, "binance") 等价于 ccxt.binance
        exchange_class = getattr(ccxt, exchange_id)

        config = {
            "apiKey": settings.API_KEY,
            "secret": settings.API_SECRET,
            "timeout": settings.REQUEST_TIMEOUT * 1000,  # ccxt 用毫秒
            "enableRateLimit": True,  # ccxt 内置的请求频率限制
            "options": {
                "fetchCurrencies": False,  # 禁用 SAPI 货币请求，避免不可达端点
            },
        }

        # 配置代理（本地开发环境可能需要）
        if settings.PROXY_HOST and settings.PROXY_PORT:
            proxy_url = f"http://{settings.PROXY_HOST}:{settings.PROXY_PORT}"
            config["proxies"] = {
                "http": proxy_url,
                "https": proxy_url,
            }

        self.exchange = exchange_class(config)
        logger.info(f"K线拉取器已初始化: {exchange_id}")

    @retry_on_failure()
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        since: int = None,
        limit: int = None,
    ) -> pd.DataFrame:
        """
        拉取单个币对的 K线 数据（单次请求）

        Args:
            symbol:    交易对，如 "BTC/USDT"
            timeframe: K线周期，如 "1m"
            since:     起始时间（毫秒时间戳），None 表示最新
            limit:     最大返回条数，默认 settings.KLINE_LIMIT

        Returns:
            DataFrame，列: [timestamp, open, high, low, close, volume]
            - timestamp: UTC datetime
            - OHLCV: float64
            空 DataFrame 表示无数据

        Raises:
            ccxt.BadRequest: 参数错误（不重试）
            ccxt.NetworkError: 网络问题（自动重试）
        """
        if limit is None:
            limit = settings.KLINE_LIMIT

        # 调用 ccxt 接口
        # 返回: [[timestamp_ms, open, high, low, close, volume], ...]
        raw = self.exchange.fetch_ohlcv(
            symbol, timeframe, since=since, limit=limit
        )

        if not raw:
            logger.debug(f"{symbol} {timeframe}: 返回空数据")
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        # 转为 DataFrame
        df = pd.DataFrame(
            raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

        # 时间戳: 毫秒 → UTC datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        logger.debug(
            f"{symbol} {timeframe}: 拉取 {len(df)} 根K线, "
            f"范围 {df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]}"
        )

        return df

    def fetch_ohlcv_batch(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        批量拉取历史 K线（自动分页）

        从 start 到 end 之间的所有 K线，自动处理分页。
        每次请求 KLINE_LIMIT 根，直到覆盖整个时间范围。

        Args:
            symbol:    交易对
            timeframe: K线周期
            start:     起始时间 (UTC datetime)
            end:       结束时间 (UTC datetime)

        Returns:
            DataFrame，按时间升序排列，已去重

        Example:
            >>> fetcher = KlineFetcher()
            >>> df = fetcher.fetch_ohlcv_batch(
            ...     "BTC/USDT", "1m",
            ...     datetime(2024, 1, 1, tzinfo=timezone.utc),
            ...     datetime(2024, 1, 2, tzinfo=timezone.utc),
            ... )
        """
        all_frames = []

        # 将 start 转为毫秒时间戳
        since = datetime_to_ms(start)
        end_ms = datetime_to_ms(end)

        # 每根 K线 的时间跨度（毫秒）
        period_ms = timeframe_to_seconds(timeframe) * 1000

        total_fetched = 0

        while since < end_ms:
            df = self.fetch_ohlcv(symbol, timeframe, since=since)

            if df.empty:
                # 无更多数据
                break

            all_frames.append(df)
            total_fetched += len(df)

            # 更新 since 为最后一根 K线 的下一个周期起始时间
            last_ts = datetime_to_ms(df["timestamp"].iloc[-1])
            since = last_ts + period_ms

            # 如果返回不满额，说明已到最新数据
            if len(df) < settings.KLINE_LIMIT:
                break

        if not all_frames:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        # 合并所有分页结果
        result = pd.concat(all_frames, ignore_index=True)

        # 去重（分页边界可能有重叠）+ 过滤时间范围
        result = result.drop_duplicates(subset=["timestamp"], keep="last")
        end_ts = pd.Timestamp(end).tz_localize("UTC") if end.tzinfo is None else pd.Timestamp(end)
        result = result[result["timestamp"] <= end_ts]
        result = result.sort_values("timestamp").reset_index(drop=True)

        logger.info(
            f"{symbol} {timeframe}: 批量拉取完成, "
            f"共 {len(result)} 根K线, "
            f"范围 {result['timestamp'].iloc[0]} ~ {result['timestamp'].iloc[-1]}"
        )

        return result
