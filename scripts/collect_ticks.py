"""
逐笔成交持续采集脚本（独立进程，异步）

基于 trade_id 追赶模式，零遗漏地采集逐笔成交数据。
满额（1000条）立即继续拉取，不满额说明已追上实时。

容错:
    - 断点续传: 重启后从本地最新 trade_id 继续
    - 冷启动: 首次采集通过 resolve_cold_start 确定起点
    - 单币对失败不影响其他

监控:
    - 集成 Heartbeat，区分 catching_up / idle 状态
    - 状态文件 logs/collect_ticks.status.json

用法:
    python -m scripts.collect_ticks
    python -m scripts.collect_ticks --symbols BTC/USDT ETH/USDT
"""

import argparse
import asyncio
import signal

from config import settings
from data.tick_fetcher import TickFetcher
from data.writer import DataWriter
from utils.heartbeat import Heartbeat
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="逐笔成交持续采集")
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="指定采集的交易对"
    )
    return parser.parse_args()


async def run(symbols: list[str]):
    """异步主循环"""
    fetcher = TickFetcher()
    writer = DataWriter()
    heartbeat = Heartbeat("collect_ticks")

    running = True

    def signal_handler(signum, frame):
        nonlocal running
        logger.info(f"收到退出信号 ({signum})，准备停止...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    heartbeat.start()

    try:
        while running:
            heartbeat.set_status("catching_up")

            for symbol in symbols:
                if not running:
                    break

                try:
                    # 获取断点: 本地最新 trade_id
                    last_id = writer.get_latest_trade_id(symbol)

                    # 冷启动: 首次采集确定起点
                    if last_id is None:
                        last_id = await fetcher.resolve_cold_start(symbol)
                        if last_id is None:
                            logger.warning(f"{symbol}: 冷启动失败，跳过")
                            continue

                    # 追赶到最新
                    df = await fetcher.fetch_until_latest(symbol, from_id=last_id)

                    if not df.empty:
                        count = writer.write_ticks(df, symbol)
                        latest_id = int(df["trade_id"].iloc[-1])
                        heartbeat.update(
                            symbol, records=count, latest_id=latest_id
                        )

                        if count > 0:
                            logger.info(
                                f"{symbol}: 新增 {count} 笔成交, "
                                f"latest_id={latest_id}"
                            )

                except Exception as e:
                    heartbeat.report_error(e)
                    logger.error(
                        f"{symbol} tick 采集失败: {e}", exc_info=True
                    )

            # 一轮结束，切换到 idle 状态
            heartbeat.set_status("idle")
            heartbeat.tick()

            # 等待下一轮
            for _ in range(int(settings.TICK_IDLE_INTERVAL)):
                if not running:
                    break
                await asyncio.sleep(1)

    finally:
        await fetcher.close()
        heartbeat.stop()
        logger.info("Tick 采集已停止")


def main():
    args = parse_args()
    symbols = args.symbols or settings.SYMBOLS

    logger.info(f"Tick 采集启动: {symbols}")
    asyncio.run(run(symbols))


if __name__ == "__main__":
    main()
