"""
历史数据回填脚本

用于补充采集启动前的历史数据。
支持 K线、逐笔成交、资金费率的回填（订单簿无法回填）。

用法:
    # 回填 K线
    python -m data_infra.scripts.backfill --type kline --start 2024-01-01 --end 2024-12-31
    python -m data_infra.scripts.backfill --type kline --start 2024-01-01 --symbols BTC/USDT

    # 回填 tick（从指定日期开始追赶）
    python -m data_infra.scripts.backfill --type tick --start 2024-06-01

    # 回填 tick（从指定 trade_id 开始）
    python -m data_infra.scripts.backfill --type tick --symbol BTC/USDT --from-id 123456789

    # 回填资金费率
    python -m data_infra.scripts.backfill --type funding_rate --start 2024-01-01

注意:
    - 订单簿 (orderbook) 无法回填，Binance 不提供历史订单簿 API
    - tick 回填可能需要很长时间（取决于历史跨度）
    - K线回填受 API 频率限制，建议分批操作
"""

import argparse
import asyncio
import sys
from datetime import datetime, timezone

from data_infra.config import settings
from data_infra.data.fetcher import KlineFetcher
from data_infra.data.tick_fetcher import TickFetcher
from data_infra.data.market_fetcher import MarketFetcher
from data_infra.data.writer import DataWriter
from data_infra.utils.logger import get_logger
from data_infra.utils.time_utils import datetime_to_ms

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="历史数据回填")
    parser.add_argument(
        "--type", required=True,
        choices=["kline", "tick", "funding_rate"],
        help="回填数据类型"
    )
    parser.add_argument(
        "--start", default=None,
        help="起始日期，格式 YYYY-MM-DD"
    )
    parser.add_argument(
        "--end", default=None,
        help="结束日期，格式 YYYY-MM-DD（默认到当前）"
    )
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="指定交易对（默认全部）"
    )
    parser.add_argument(
        "--from-id", type=int, default=None,
        help="(tick 回填) 指定起始 trade_id"
    )
    return parser.parse_args()


def backfill_klines(symbols, start, end):
    """回填 K线历史数据"""
    fetcher = KlineFetcher()
    writer = DataWriter()

    for symbol in symbols:
        logger.info(f"回填 K线: {symbol}, {start} ~ {end}")
        try:
            df = fetcher.fetch_ohlcv_batch(symbol, "1m", start, end)
            count = writer.write_ohlcv(df, symbol, "1m")
            logger.info(f"{symbol}: 回填 {count} 根 1m K线")
        except Exception as e:
            logger.error(f"{symbol} K线回填失败: {e}", exc_info=True)


async def backfill_ticks(symbols, start=None, from_id=None):
    """回填逐笔成交数据"""
    fetcher = TickFetcher()
    writer = DataWriter()

    try:
        for symbol in symbols:
            # 确定起始 trade_id
            start_id = from_id
            if start_id is None and start is not None:
                # 通过 startTime 找到指定日期的第一个 trade_id
                start_ms = datetime_to_ms(start)
                raw = await fetcher.exchange.fetch_trades(
                    symbol, limit=10, params={"startTime": start_ms}
                )
                if raw:
                    start_id = min(int(t["id"]) for t in raw)
                    logger.info(
                        f"{symbol}: 从日期 {start} 定位到 trade_id={start_id}"
                    )
                else:
                    logger.warning(f"{symbol}: 指定日期无数据，跳过")
                    continue

            if start_id is None:
                logger.warning(f"{symbol}: 无法确定起始 trade_id，跳过")
                continue

            logger.info(f"回填 Tick: {symbol}, from_id={start_id}")

            # 逐批拉取并立即写入，避免大量追赶中途崩溃丢失进度
            # 直接使用 fetch_trades（单批），不使用 fetch_until_latest（全量合并）
            current_id = start_id
            total = 0
            limit = settings.TICK_FETCH_LIMIT

            while True:
                df = await fetcher.fetch_trades(
                    symbol, from_id=current_id, limit=limit
                )
                if df.empty:
                    break

                count = writer.write_ticks(df, symbol)
                total += count
                current_id = int(df["trade_id"].iloc[-1])

                logger.info(
                    f"{symbol}: 已回填 {total} 笔, latest_id={current_id}"
                )

                # 不满额说明已追上最新数据
                if len(df) < limit:
                    break

            logger.info(f"{symbol}: Tick 回填完成, 共 {total} 笔")

    finally:
        await fetcher.close()


def backfill_funding_rate(symbols, start, end):
    """回填资金费率"""
    fetcher = MarketFetcher()
    writer = DataWriter()

    for symbol in symbols:
        logger.info(f"回填资金费率: {symbol}, {start} ~ {end}")
        try:
            since = datetime_to_ms(start) if start else None
            df = fetcher.fetch_funding_rate(symbol, since=since, limit=1000)

            # 过滤时间范围
            if end and not df.empty:
                df = df[df["timestamp"] <= end]

            count = writer.write_funding_rate(df, symbol)
            logger.info(f"{symbol}: 回填 {count} 条资金费率")
        except Exception as e:
            logger.error(f"{symbol} 资金费率回填失败: {e}", exc_info=True)


def main():
    args = parse_args()
    symbols = args.symbols or settings.SYMBOLS

    # 解析日期
    start = None
    end = None
    if args.start:
        start = datetime.strptime(args.start, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
    if args.end:
        # 将 --end 日期解析为当天结束（23:59:59 UTC），符合用户直觉
        # 用户输入 --end 2024-12-01 意为"回填到 12 月 1 日结束"
        end = datetime.strptime(args.end, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=timezone.utc
        )
    else:
        end = datetime.now(timezone.utc)

    if args.type == "kline":
        if start is None:
            logger.error("K线回填需要 --start 参数")
            sys.exit(1)
        backfill_klines(symbols, start, end)

    elif args.type == "tick":
        asyncio.run(backfill_ticks(symbols, start, args.from_id))

    elif args.type == "funding_rate":
        backfill_funding_rate(symbols, start, end)

    logger.info("回填完成")


if __name__ == "__main__":
    main()
