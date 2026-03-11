"""
订单簿快照持续采集脚本（独立进程，异步 WebSocket）

通过 WebSocket 接收 100ms 粒度的 10 档订单簿推送。
无法回填历史，只能从启动时刻开始积累。

容错:
    - WebSocket 断线自动重连（指数退避）
    - 重连期间数据丢失不可恢复，但会记录日志

监控:
    - 集成 Heartbeat，报告接收快照数和缓冲区状态
    - 状态文件 logs/collect_orderbook.status.json

退出:
    - 收到 SIGTERM/SIGINT 时先刷盘再退出
    - 防止内存缓冲区中的数据丢失

用法:
    python -m scripts.collect_orderbook
    python -m scripts.collect_orderbook --symbols BTC/USDT ETH/USDT
"""

import argparse
import asyncio
import signal

from config import settings
from data.orderbook_fetcher import OrderbookFetcher
from data.writer import DataWriter
from utils.heartbeat import Heartbeat
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="订单簿快照持续采集")
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="指定采集的交易对"
    )
    return parser.parse_args()


async def run(symbols: list[str]):
    """异步主循环"""
    writer = DataWriter()
    heartbeat = Heartbeat("collect_orderbook")
    fetcher = OrderbookFetcher(symbols=symbols)

    # 快照计数器（用于定期刷盘）
    snapshot_count = 0
    flush_interval = 10000  # 每 10000 个快照强制刷盘一次

    def on_snapshot(symbol: str, data: dict):
        """
        每收到一个订单簿快照时的回调

        将快照追加到 DataWriter 的内存缓冲中。
        OrderbookStore 会在缓冲满时自动刷盘。
        """
        nonlocal snapshot_count
        writer.append_orderbook(symbol, data)
        heartbeat.update(symbol, records=1)
        snapshot_count += 1

        # 定期输出心跳（WebSocket 是持续接收，没有自然的循环边界）
        if snapshot_count % flush_interval == 0:
            heartbeat.tick()

    # 注册退出处理: 先刷盘再退出
    def shutdown_handler(signum, frame):
        logger.info(f"收到退出信号 ({signum})，刷盘中...")
        writer.flush_and_close_orderbook()
        heartbeat.stop()
        logger.info("订单簿采集已安全退出")
        # 让 asyncio 的 stop 生效
        raise SystemExit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    heartbeat.start()
    heartbeat.set_status("running")

    try:
        await fetcher.start(on_snapshot)
    except SystemExit:
        pass
    except Exception as e:
        logger.error(f"订单簿采集异常退出: {e}", exc_info=True)
        heartbeat.report_error(e)
    finally:
        writer.flush_and_close_orderbook()
        await fetcher.stop()
        heartbeat.stop()


def main():
    args = parse_args()
    symbols = args.symbols or settings.SYMBOLS

    logger.info(
        f"订单簿采集启动: {symbols}, "
        f"深度={settings.ORDERBOOK_DEPTH}, "
        f"频率={settings.ORDERBOOK_UPDATE_SPEED}"
    )
    asyncio.run(run(symbols))


if __name__ == "__main__":
    main()
