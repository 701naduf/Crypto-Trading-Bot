"""
K线数据持续采集脚本（独立进程）

采集 1m K线，其他周期（5m, 15m, 1h 等）由 DataReader 按需降采样生成。
每轮遍历所有配置的交易对，增量拉取新 K线。

容错:
    - 单币对失败不影响其他币对（try/except 隔离）
    - 重启后自动从断点继续（通过查询最新 K线 时间）
    - API 错误由 retry 装饰器处理（指数退避重试）

监控:
    - 集成 Heartbeat，定期输出采集统计
    - 状态文件 logs/collect_klines.status.json

用法:
    python -m data_infra.scripts.collect_klines                    # 持续采集所有币对
    python -m data_infra.scripts.collect_klines --symbols BTC/USDT # 只采集指定币对
    python -m data_infra.scripts.collect_klines --once             # 只运行一轮
"""

import argparse
import signal
import sys
import time

from data_infra.config import settings
from data_infra.data.fetcher import KlineFetcher
from data_infra.data.writer import DataWriter
from data_infra.utils.heartbeat import Heartbeat
from data_infra.utils.logger import get_logger
from data_infra.utils.time_utils import datetime_to_ms

logger = get_logger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="K线数据持续采集")
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="指定采集的交易对，如 BTC/USDT ETH/USDT"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="只运行一轮后退出（用于测试）"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    symbols = args.symbols or settings.SYMBOLS
    timeframe = settings.KLINE_COLLECT_TIMEFRAME  # "1m"

    logger.info(f"K线采集启动: {symbols}, 周期={timeframe}")

    # 初始化组件
    fetcher = KlineFetcher()
    writer = DataWriter()
    heartbeat = Heartbeat("collect_klines")

    # 优雅退出: 收到 SIGTERM/SIGINT 时设置 running=False
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

            try:
                # 查询断点: 最新 K线 时间
                latest_time = writer.get_latest_kline_time(symbol, timeframe)

                # 计算 since 参数
                # 如果有历史数据，从最新时间的下一根 K线 开始
                # 如果是首次采集，不传 since（获取最新 1000 根）
                since = None
                if latest_time is not None:
                    since = datetime_to_ms(latest_time) + 60000  # +1 分钟

                # 拉取数据
                df = fetcher.fetch_ohlcv(
                    symbol, timeframe, since=since
                )

                # 写入（DataWriter 内部会校验）
                count = writer.write_ohlcv(df, symbol, timeframe)

                heartbeat.update(symbol, records=count)

                if count > 0:
                    logger.info(f"{symbol}: 新增 {count} 根 {timeframe} K线")

            except Exception as e:
                heartbeat.report_error(e)
                logger.error(f"{symbol} K线采集失败: {e}", exc_info=True)

        heartbeat.tick()

        if args.once:
            logger.info("单轮模式，采集完成")
            break

        # 等待下一轮
        logger.debug(f"等待 {settings.KLINE_COLLECT_INTERVAL}s...")
        for _ in range(settings.KLINE_COLLECT_INTERVAL):
            if not running:
                break
            time.sleep(1)

    heartbeat.stop()
    logger.info("K线采集已停止")


if __name__ == "__main__":
    main()
