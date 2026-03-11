"""
合约市场数据持续采集脚本（独立进程）

采集资金费率、持仓量、多空持仓比、主动买卖量。
所有数据通过 REST API 轮询，每 5 分钟一次。

容错:
    - 单项数据失败不影响其他项
    - 重启后自动从断点继续

监控:
    - 集成 Heartbeat
    - 状态文件 logs/collect_market.status.json

用法:
    python -m data_infra.scripts.collect_market
    python -m data_infra.scripts.collect_market --symbols BTC/USDT
    python -m data_infra.scripts.collect_market --once
"""

import argparse
import signal
import sys
import time

from data_infra.config import settings
from data_infra.data.market_fetcher import MarketFetcher
from data_infra.data.writer import DataWriter
from data_infra.utils.heartbeat import Heartbeat
from data_infra.utils.logger import get_logger
from data_infra.utils.time_utils import datetime_to_ms

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="合约市场数据持续采集")
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="指定采集的交易对"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="只运行一轮后退出"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    symbols = args.symbols or settings.SYMBOLS

    logger.info(f"合约市场数据采集启动: {symbols}")

    fetcher = MarketFetcher()
    writer = DataWriter()
    heartbeat = Heartbeat("collect_market")

    running = True

    def signal_handler(signum, frame):
        nonlocal running
        logger.info(f"收到退出信号 ({signum})，准备停止...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    heartbeat.start()

    while running:
        for symbol in symbols:
            if not running:
                break

            records_count = 0

            # === 1. 资金费率 ===
            try:
                last_fr_time = writer.get_latest_market_time(
                    "funding_rates", symbol
                )
                since = None
                if last_fr_time is not None:
                    since = datetime_to_ms(last_fr_time) + 1

                fr = fetcher.fetch_funding_rate(symbol, since=since)
                count = writer.write_funding_rate(fr, symbol)
                records_count += count
                if count > 0:
                    logger.info(f"{symbol}: 新增 {count} 条资金费率")

            except Exception as e:
                heartbeat.report_error(e)
                logger.error(f"{symbol} 资金费率采集失败: {e}")

            # === 2. 持仓量 ===
            try:
                oi = fetcher.fetch_open_interest(symbol)
                count = writer.write_open_interest(symbol, oi)
                records_count += count

            except Exception as e:
                heartbeat.report_error(e)
                logger.error(f"{symbol} 持仓量采集失败: {e}")

            # === 3. 多空持仓比 ===
            try:
                lsr = fetcher.fetch_long_short_ratio(symbol)
                count = writer.write_long_short_ratio(lsr, symbol)
                records_count += count
                if count > 0:
                    logger.info(f"{symbol}: 新增 {count} 条多空持仓比")

            except Exception as e:
                heartbeat.report_error(e)
                logger.error(f"{symbol} 多空持仓比采集失败: {e}")

            # === 4. 主动买卖量 ===
            try:
                tbs = fetcher.fetch_taker_buy_sell_volume(symbol)
                count = writer.write_taker_buy_sell(tbs, symbol)
                records_count += count
                if count > 0:
                    logger.info(f"{symbol}: 新增 {count} 条主动买卖量")

            except Exception as e:
                heartbeat.report_error(e)
                logger.error(f"{symbol} 主动买卖量采集失败: {e}")

            heartbeat.update(symbol, records=records_count)

        heartbeat.tick()

        if args.once:
            logger.info("单轮模式，采集完成")
            break

        # 等待下一轮
        logger.debug(f"等待 {settings.MARKET_COLLECT_INTERVAL}s...")
        for _ in range(settings.MARKET_COLLECT_INTERVAL):
            if not running:
                break
            time.sleep(1)

    heartbeat.stop()
    logger.info("合约市场数据采集已停止")


if __name__ == "__main__":
    main()
