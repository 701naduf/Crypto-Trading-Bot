"""
加密货币量化交易系统 —— 主入口

提供各采集脚本和工具的启动说明。
各采集脚本作为独立进程运行，不在此处统一调度。

用法:
    python main.py              # 显示用法说明
    python main.py status       # 查看各采集器运行状态
"""

import sys

from config import settings


def print_usage():
    """打印各模块的启动命令和说明"""
    symbols = ", ".join(settings.SYMBOLS)

    print("=" * 60)
    print("  加密货币量化交易系统 - 数据基建 (Phase 1)")
    print("=" * 60)
    print(f"\n交易对: {symbols}")
    print(f"交易所: {settings.EXCHANGE_ID}")

    print("\n--- 数据采集 ---")
    print("  K线采集:     python -m scripts.collect_klines")
    print("  逐笔成交:   python -m scripts.collect_ticks")
    print("  订单簿:     python -m scripts.collect_orderbook")
    print("  合约数据:   python -m scripts.collect_market")

    print("\n--- 工具 ---")
    print("  运行状态:   python -m scripts.status")
    print("  数据巡检:   python -m scripts.check_data")
    print("  数据巡检+修复: python -m scripts.check_data --fix")

    print("\n--- 历史回填 ---")
    print("  K线回填:    python -m scripts.backfill --type kline --start 2024-01-01")
    print("  Tick回填:   python -m scripts.backfill --type tick --start 2024-06-01")
    print("  资金费率:   python -m scripts.backfill --type funding_rate --start 2024-01-01")

    print("\n--- 测试 ---")
    print("  运行测试:   python -m pytest tests/ -v")
    print()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        # 委托给 status 脚本
        from scripts.status import main as status_main
        status_main()
    else:
        print_usage()


if __name__ == "__main__":
    main()
