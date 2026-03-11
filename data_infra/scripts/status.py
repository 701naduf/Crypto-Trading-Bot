"""
采集状态总览脚本

读取所有采集脚本的 .status.json 文件，输出简洁的状态仪表板。

输出示例:
    $ python -m data_infra.scripts.status

    采集状态总览 (2024-01-15 12:30:00 UTC)
    ┌────────────────────┬──────────┬──────────┬────────────────┬────────┐
    │ 脚本                │ 状态     │ 运行时间  │ 最近心跳        │ 错误数 │
    ├────────────────────┼──────────┼──────────┼────────────────┼────────┤
    │ collect_klines     │ running  │ 3d 2h    │ 12秒前          │ 0      │
    │ collect_ticks      │ idle     │ 3d 2h    │ 8秒前           │ 2      │
    │ collect_orderbook  │ running  │ 1d 5h    │ 3秒前           │ 0      │
    │ collect_market     │ running  │ 3d 2h    │ 45秒前          │ 0      │
    └────────────────────┴──────────┴──────────┴────────────────┴────────┘

用法:
    python -m data_infra.scripts.status
"""

import json
import os
from datetime import datetime, timezone

from data_infra.config import settings

# 所有采集脚本名称
SCRIPTS = [
    "collect_klines",
    "collect_ticks",
    "collect_orderbook",
    "collect_market",
]


def format_uptime(seconds: int) -> str:
    """将秒数格式化为易读字符串"""
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    remaining_m = minutes % 60
    if hours < 24:
        return f"{hours}h {remaining_m}m"
    days = hours // 24
    remaining_h = hours % 24
    return f"{days}d {remaining_h}h"


def format_time_ago(iso_str: str) -> str:
    """将 ISO 时间字符串转为 '多少秒前' 格式"""
    try:
        ts = datetime.fromisoformat(iso_str)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        delta = (datetime.now(timezone.utc) - ts).total_seconds()

        if delta < 60:
            return f"{int(delta)}秒前"
        elif delta < 3600:
            return f"{int(delta // 60)}分钟前"
        elif delta < 86400:
            return f"{int(delta // 3600)}小时前"
        else:
            return f"{int(delta // 86400)}天前"
    except (ValueError, TypeError):
        return "未知"


def main():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n采集状态总览 ({now})")
    print("=" * 75)

    # 表头
    header = f"{'脚本':<22} {'状态':<12} {'运行时间':<10} {'最近心跳':<14} {'错误数':<6}"
    print(header)
    print("-" * 75)

    for script in SCRIPTS:
        status_file = os.path.join(settings.LOG_DIR, f"{script}.status.json")

        if not os.path.exists(status_file):
            print(f"{script:<22} {'未启动':<12} {'-':<10} {'-':<14} {'-':<6}")
            continue

        try:
            with open(status_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            status = data.get("status", "unknown")
            uptime = format_uptime(data.get("uptime_seconds", 0))
            heartbeat = format_time_ago(data.get("last_heartbeat", ""))
            errors = data.get("errors_count", 0)

            # 状态颜色标记（终端支持时）
            status_display = status

            print(
                f"{script:<22} {status_display:<12} {uptime:<10} "
                f"{heartbeat:<14} {errors:<6}"
            )

            # 如果有错误，显示最近错误
            last_error = data.get("last_error")
            if last_error:
                print(f"  └─ 最近错误: {last_error[:60]}")

        except (json.JSONDecodeError, OSError) as e:
            print(f"{script:<22} {'读取失败':<12} {'-':<10} {'-':<14} {'-':<6}")

    print("=" * 75)
    print(f"状态文件目录: {settings.LOG_DIR}/")
    print()


if __name__ == "__main__":
    main()
