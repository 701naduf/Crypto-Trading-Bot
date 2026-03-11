"""
合约市场数据拉取模块（同步）

通过 Binance Futures REST API 获取以下市场数据:
    - 资金费率 (Funding Rate):    /fapi/v1/fundingRate
    - 持仓量 (Open Interest):     /fapi/v1/openInterest
    - 多空持仓比:                  /futures/data/topLongShortPositionRatio
    - 主动买卖量:                  /futures/data/takeBuySellVol

这些数据量极小（每日每币对几百条），使用同步 REST 轮询即可。
部分接口（资金费率）支持历史查询，可用于回填。

注意: 这些是合约 (Futures) 接口，需要交易所实例开启 futures 模式。
ccxt 中通过 options["defaultType"] = "future" 设置。

Binance 符号转换:
    现货: "BTC/USDT"
    合约: "BTC/USDT:USDT" (ccxt 的 USDT 永续合约格式)

数据流: Futures REST API → ccxt / requests → DataFrame
依赖: config.settings, utils.logger, utils.retry
被依赖: scripts/collect_market.py, scripts/backfill.py
"""

from datetime import datetime, timezone

import ccxt
import pandas as pd

from config import settings
from utils.logger import get_logger
from utils.retry import retry_on_failure

logger = get_logger(__name__)


def _to_futures_symbol(symbol: str) -> str:
    """
    将现货交易对转为 ccxt 合约格式

    Args:
        symbol: "BTC/USDT"

    Returns:
        "BTC/USDT:USDT" —— ccxt 的 USDT 永续合约标识
    """
    if ":USDT" not in symbol:
        return f"{symbol}:USDT"
    return symbol


def _to_binance_symbol(symbol: str) -> str:
    """
    将标准交易对转为 Binance API 所需格式

    用于直接调用 Binance 私有 API（ccxt 不直接支持的接口）。

    Args:
        symbol: "BTC/USDT"

    Returns:
        "BTCUSDT"
    """
    return symbol.replace("/", "")


class MarketFetcher:
    """
    合约市场数据拉取器

    封装资金费率、持仓量、多空持仓比、主动买卖量的获取逻辑。
    内部使用两种方式:
        1. ccxt 封装好的接口（资金费率、持仓量）
        2. 直接调用 Binance API（多空持仓比、主动买卖量，ccxt 未封装）
    """

    def __init__(self, exchange_id: str = None):
        """
        初始化拉取器，创建 ccxt 合约交易所实例

        Args:
            exchange_id: 交易所标识，默认 settings.EXCHANGE_ID
        """
        exchange_id = exchange_id or settings.EXCHANGE_ID

        exchange_class = getattr(ccxt, exchange_id)

        config = {
            "apiKey": settings.API_KEY,
            "secret": settings.API_SECRET,
            "timeout": settings.REQUEST_TIMEOUT * 1000,
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",  # 使用合约 API
                "fetchCurrencies": False,  # 禁用 SAPI 货币请求，避免不可达端点
            },
        }

        if settings.PROXY_HOST and settings.PROXY_PORT:
            proxy_url = f"http://{settings.PROXY_HOST}:{settings.PROXY_PORT}"
            config["proxies"] = {
                "http": proxy_url,
                "https": proxy_url,
            }

        self.exchange = exchange_class(config)
        logger.info(f"合约市场数据拉取器已初始化: {exchange_id}")

    @retry_on_failure()
    def fetch_funding_rate(
        self, symbol: str, since: int = None, limit: int = None
    ) -> pd.DataFrame:
        """
        拉取资金费率历史

        Binance 永续合约每 8 小时结算一次资金费率。
        支持历史查询（通过 since 参数）。

        Args:
            symbol: 交易对，如 "BTC/USDT"
            since:  起始时间（毫秒时间戳）
            limit:  最大返回条数

        Returns:
            DataFrame [timestamp, symbol, funding_rate]
            - timestamp: UTC datetime
            - symbol: 交易对
            - funding_rate: float（如 0.0001 表示万分之一）
        """
        futures_symbol = _to_futures_symbol(symbol)

        # ccxt 的 fetchFundingRateHistory 接口
        raw = self.exchange.fetch_funding_rate_history(
            futures_symbol, since=since, limit=limit
        )

        if not raw:
            return pd.DataFrame(columns=["timestamp", "symbol", "funding_rate"])

        records = []
        for item in raw:
            records.append({
                "timestamp": pd.Timestamp(item["timestamp"], unit="ms", tz="UTC"),
                "symbol": symbol,
                "funding_rate": float(item["fundingRate"]),
            })

        df = pd.DataFrame(records)
        logger.debug(f"{symbol}: 拉取 {len(df)} 条资金费率")
        return df

    @retry_on_failure()
    def fetch_open_interest(self, symbol: str) -> dict:
        """
        拉取当前持仓量

        持仓量 (Open Interest) 是市场上所有未平仓合约的总量。
        这是一个实时快照，不支持历史查询。

        Args:
            symbol: 交易对

        Returns:
            {
                "timestamp": datetime,          # 当前时间 (UTC)
                "open_interest": float,         # 持仓量（合约张数）
                "open_interest_value": float,   # 持仓价值（USDT）
            }
        """
        futures_symbol = _to_futures_symbol(symbol)
        binance_symbol = _to_binance_symbol(symbol)

        # 使用 Binance 的 fapiPublicGetOpenInterest 隐式方法
        # ccxt v4 只保留 camelCase 命名，分区为 fapiPublic
        response = self.exchange.fapiPublicGetOpenInterest({
            "symbol": binance_symbol,
        })

        oi = float(response.get("openInterest", 0))

        # 获取当前价格，计算持仓价值
        ticker = self.exchange.fetch_ticker(futures_symbol)
        price = ticker.get("last", 0) or 0
        oi_value = oi * price

        result = {
            "timestamp": datetime.now(timezone.utc),
            "symbol": symbol,
            "open_interest": oi,
            "open_interest_value": oi_value,
        }

        logger.debug(
            f"{symbol}: OI = {oi:.2f}, 价值 = {oi_value:.0f} USDT"
        )
        return result

    @retry_on_failure()
    def fetch_long_short_ratio(
        self, symbol: str, period: str = "5m", limit: int = 30
    ) -> pd.DataFrame:
        """
        拉取多空持仓比（大户持仓比）

        Binance API: /futures/data/topLongShortPositionRatio
        统计持仓量前 20% 的大户的多空比。

        Args:
            symbol: 交易对
            period: 统计周期，如 "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"
            limit:  返回条数，最大 500

        Returns:
            DataFrame [timestamp, symbol, long_ratio, short_ratio, long_short_ratio]
        """
        binance_symbol = _to_binance_symbol(symbol)

        # ccxt v4 隐式方法：fapiData 分区 + camelCase 命名
        # 对应 Binance API: /futures/data/topLongShortPositionRatio
        response = self.exchange.fapiDataGetTopLongShortPositionRatio({
            "symbol": binance_symbol,
            "period": period,
            "limit": limit,
        })

        if not response:
            return pd.DataFrame(
                columns=["timestamp", "symbol", "long_ratio", "short_ratio", "long_short_ratio"]
            )

        records = []
        for item in response:
            records.append({
                "timestamp": pd.Timestamp(int(item["timestamp"]), unit="ms", tz="UTC"),
                "symbol": symbol,
                "long_ratio": float(item["longAccount"]),
                "short_ratio": float(item["shortAccount"]),
                "long_short_ratio": float(item["longShortRatio"]),
            })

        df = pd.DataFrame(records)
        logger.debug(f"{symbol}: 拉取 {len(df)} 条多空持仓比")
        return df

    @retry_on_failure()
    def fetch_taker_buy_sell_volume(
        self, symbol: str, period: str = "5m", limit: int = 30
    ) -> pd.DataFrame:
        """
        拉取主动买卖量

        Binance API: /futures/data/takerlongshortRatio
        统计主动买入和主动卖出的成交量比。

        Args:
            symbol: 交易对
            period: 统计周期
            limit:  返回条数

        Returns:
            DataFrame [timestamp, symbol, buy_vol, sell_vol, buy_sell_ratio]
        """
        binance_symbol = _to_binance_symbol(symbol)

        # ccxt v4 隐式方法：fapiData 分区 + camelCase 命名
        # 对应 Binance API: /futures/data/takerlongshortRatio
        response = self.exchange.fapiDataGetTakerlongshortRatio({
            "symbol": binance_symbol,
            "period": period,
            "limit": limit,
        })

        if not response:
            return pd.DataFrame(
                columns=["timestamp", "symbol", "buy_vol", "sell_vol", "buy_sell_ratio"]
            )

        records = []
        for item in response:
            records.append({
                "timestamp": pd.Timestamp(int(item["timestamp"]), unit="ms", tz="UTC"),
                "symbol": symbol,
                "buy_vol": float(item.get("buyVol", 0)),
                "sell_vol": float(item.get("sellVol", 0)),
                "buy_sell_ratio": float(item.get("buySellRatio", 0)),
            })

        df = pd.DataFrame(records)
        logger.debug(f"{symbol}: 拉取 {len(df)} 条主动买卖量")
        return df
