"""
重试与容错工具

提供装饰器和工具函数，为 API 请求添加自动重试、指数退避、
异常分类处理能力。所有 fetcher 模块共用此逻辑，避免重复实现。

异常分类与处理策略:
    - 网络超时 / 连接断开 (NetworkError)     → 指数退避重试
    - API 限频 (RateLimitExceeded, 429)      → 暂停 RATE_LIMIT_PAUSE 秒后重试
    - 客户端错误 (BadRequest, 400)            → 不重试，立即抛出
    - 交易所维护 (ExchangeNotAvailable, 503)  → 长间隔重试（60秒）
    - 未知异常                                → 记录日志，指数退避重试

用法:
    from utils.retry import retry_on_failure, async_retry_on_failure

    # 同步函数（如 K线采集）
    @retry_on_failure()
    def fetch_data():
        ...

    # 异步函数（如 tick 采集）
    @async_retry_on_failure()
    async def fetch_trades():
        ...

    # 自定义重试次数
    @retry_on_failure(max_retries=10)
    def important_fetch():
        ...

依赖: config.settings (MAX_RETRIES, RETRY_BASE_DELAY, RATE_LIMIT_PAUSE)
"""

import asyncio
import functools
import time

import ccxt

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def classify_exception(e: Exception) -> str:
    """
    对异常进行分类，返回处理策略标识

    ccxt 的异常体系:
        ccxt.BaseError
        ├── ccxt.NetworkError          ← 网络层错误（超时、断连等）
        │   ├── ccxt.RequestTimeout
        │   └── ccxt.ExchangeNotAvailable  ← 交易所维护 (503)
        ├── ccxt.ExchangeError         ← 交易所返回的业务错误
        │   ├── ccxt.RateLimitExceeded ← 限频 (429)
        │   ├── ccxt.BadRequest        ← 参数错误 (400)
        │   └── ...
        └── ...

    Returns:
        "rate_limit"       - 需要长时间等待后重试
        "network"          - 网络问题，指数退避重试
        "maintenance"      - 交易所维护，长间隔重试
        "client_error"     - 客户端错误，不重试
        "unknown"          - 未知异常，指数退避重试
    """
    # 限频 (429) —— 需要特殊处理，等待较长时间
    # 注意: RateLimitExceeded 是 ExchangeError 的子类，
    #       要在 ExchangeError 之前判断
    if isinstance(e, ccxt.RateLimitExceeded):
        return "rate_limit"

    # 交易所维护 (503) —— ExchangeNotAvailable 是 NetworkError 的子类，
    # 要在 NetworkError 之前判断
    if isinstance(e, ccxt.ExchangeNotAvailable):
        return "maintenance"

    # 网络层错误（超时、连接断开等）
    if isinstance(e, ccxt.NetworkError):
        return "network"

    # 客户端错误（参数错误等），不应重试
    if isinstance(e, (ccxt.BadRequest, ccxt.AuthenticationError,
                      ccxt.PermissionDenied, ccxt.BadSymbol)):
        return "client_error"

    # 其他交易所错误，尝试重试
    if isinstance(e, ccxt.ExchangeError):
        return "unknown"

    # 非 ccxt 异常（如 Python 内置异常），视为未知
    return "unknown"


def get_wait_time(category: str, attempt: int) -> float:
    """
    根据异常分类和重试次数，计算等待时间

    Args:
        category: 异常分类（来自 classify_exception）
        attempt:  当前重试次数（从 0 开始）

    Returns:
        等待秒数

    等待策略:
        - rate_limit:   固定等待 RATE_LIMIT_PAUSE 秒（默认 30 秒）
        - maintenance:  固定等待 60 秒
        - network:      指数退避 RETRY_BASE_DELAY × 2^attempt
        - unknown:      指数退避 RETRY_BASE_DELAY × 2^attempt
    """
    if category == "rate_limit":
        return settings.RATE_LIMIT_PAUSE

    if category == "maintenance":
        return 60.0

    # 指数退避: 1, 2, 4, 8, 16, ...（上限 60 秒）
    wait = settings.RETRY_BASE_DELAY * (2 ** attempt)
    return min(wait, 60.0)


def retry_on_failure(max_retries: int = None):
    """
    同步函数重试装饰器

    为被装饰的同步函数添加自动重试能力。
    遇到可重试异常时，按照异常分类采取不同的等待策略。

    Args:
        max_retries: 最大重试次数，默认使用 settings.MAX_RETRIES

    Example:
        @retry_on_failure()
        def fetch_klines(symbol):
            return exchange.fetch_ohlcv(symbol, "1m")

        # 自定义重试次数
        @retry_on_failure(max_retries=10)
        def fetch_important_data():
            ...
    """
    if max_retries is None:
        max_retries = settings.MAX_RETRIES

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e
                    category = classify_exception(e)

                    # 客户端错误不重试，直接抛出
                    if category == "client_error":
                        logger.error(
                            f"{func.__name__} 客户端错误，不重试: "
                            f"{type(e).__name__}: {e}"
                        )
                        raise

                    # 已达最大重试次数
                    if attempt >= max_retries:
                        logger.error(
                            f"{func.__name__} 重试 {max_retries} 次后仍失败: "
                            f"{type(e).__name__}: {e}"
                        )
                        raise

                    # 计算等待时间
                    wait = get_wait_time(category, attempt)

                    logger.warning(
                        f"{func.__name__} 第 {attempt + 1}/{max_retries} 次重试 | "
                        f"异常分类: {category} | "
                        f"等待 {wait:.1f}s | "
                        f"{type(e).__name__}: {e}"
                    )

                    time.sleep(wait)

            # 理论上不会走到这里，但作为安全措施
            raise last_exception  # type: ignore

        return wrapper
    return decorator


def async_retry_on_failure(max_retries: int = None):
    """
    异步函数重试装饰器

    与 retry_on_failure 逻辑完全相同，但使用 asyncio.sleep 替代 time.sleep，
    避免阻塞事件循环。

    Args:
        max_retries: 最大重试次数，默认使用 settings.MAX_RETRIES

    Example:
        @async_retry_on_failure()
        async def fetch_trades(symbol, from_id):
            return await exchange.fetch_trades(symbol, params={"fromId": from_id})
    """
    if max_retries is None:
        max_retries = settings.MAX_RETRIES

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)

                except Exception as e:
                    last_exception = e
                    category = classify_exception(e)

                    # 客户端错误不重试
                    if category == "client_error":
                        logger.error(
                            f"{func.__name__} 客户端错误，不重试: "
                            f"{type(e).__name__}: {e}"
                        )
                        raise

                    # 已达最大重试次数
                    if attempt >= max_retries:
                        logger.error(
                            f"{func.__name__} 重试 {max_retries} 次后仍失败: "
                            f"{type(e).__name__}: {e}"
                        )
                        raise

                    wait = get_wait_time(category, attempt)

                    logger.warning(
                        f"{func.__name__} 第 {attempt + 1}/{max_retries} 次重试 | "
                        f"异常分类: {category} | "
                        f"等待 {wait:.1f}s | "
                        f"{type(e).__name__}: {e}"
                    )

                    await asyncio.sleep(wait)

            raise last_exception  # type: ignore

        return wrapper
    return decorator
