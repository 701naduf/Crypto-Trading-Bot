"""
运行状态监控模块

为每个长期运行的采集脚本提供运行状态可观测能力:

1. 心跳日志: 每隔 HEARTBEAT_INTERVAL 秒输出一条摘要日志，格式如:
   [HEARTBEAT] collect_ticks | 运行 2h 13m | 本轮: BTC +3,200笔 ETH +1,800笔 | 总计: 1,234,567笔 | 错误: 0

2. 状态文件: 每隔 STATUS_FILE_UPDATE_INTERVAL 秒写入 JSON 状态文件，供外部查看:
   logs/collect_ticks.status.json:
   {
       "script": "collect_ticks",
       "status": "running",
       "started_at": "2024-01-15T10:00:00Z",
       "last_heartbeat": "2024-01-15T12:13:00Z",
       "uptime_seconds": 7980,
       "stats": {
           "BTC/USDT": {"records_this_round": 3200, "total_records": 654321},
           "ETH/USDT": {"records_this_round": 1800, "total_records": 580246}
       },
       "errors_count": 0,
       "last_error": null
   }

用法:
    heartbeat = Heartbeat("collect_ticks")
    heartbeat.start()                           # 开始计时
    heartbeat.update("BTC/USDT", records=3200)  # 更新统计
    heartbeat.report_error(e)                   # 记录错误
    heartbeat.tick()                             # 每轮循环结束时调用

依赖: config.settings (HEARTBEAT_INTERVAL, STATUS_FILE_UPDATE_INTERVAL, LOG_DIR)
"""

import json
import os
import time
from datetime import datetime, timezone

from data_infra.config import settings
from data_infra.utils.logger import get_logger


class Heartbeat:
    """
    运行状态监控器

    每个采集脚本创建一个实例，自动管理心跳日志和状态文件。
    心跳日志输出到各模块自己的 logger，状态文件写入 logs/ 目录。
    """

    def __init__(self, script_name: str):
        """
        初始化监控器

        Args:
            script_name: 脚本名称，如 "collect_klines"、"collect_ticks"
                         用于日志前缀和状态文件名
        """
        self.script_name = script_name
        self.logger = get_logger(f"heartbeat.{script_name}")

        # === 时间记录 ===
        self._started_at = None          # 脚本启动时间 (UTC datetime)
        self._last_heartbeat_time = 0.0  # 上次输出心跳日志的时间 (monotonic)
        self._last_status_time = 0.0     # 上次更新状态文件的时间 (monotonic)

        # === 运行状态 ===
        # running:     正常运行中
        # catching_up: 正在追赶历史数据（tick 采集时）
        # idle:        已追上实时，等待下一轮
        # error:       遇到错误
        # stopped:     已停止
        self._status = "initialized"

        # === 采集统计 ===
        # 按币对记录: { "BTC/USDT": {"records_this_round": 0, "total_records": 0, ...} }
        self._stats: dict[str, dict] = {}

        # === 错误记录 ===
        self._errors_count = 0
        self._last_error = None          # 最近一次错误的描述字符串

        # === 状态文件路径 ===
        # 例: logs/collect_ticks.status.json
        os.makedirs(settings.LOG_DIR, exist_ok=True)
        self._status_file = os.path.join(
            settings.LOG_DIR, f"{script_name}.status.json"
        )

    def start(self):
        """
        标记脚本开始运行

        记录启动时间，初始化心跳计时器，输出启动日志。
        应在脚本的主循环开始前调用一次。
        """
        self._started_at = datetime.now(timezone.utc)
        self._status = "running"
        now = time.monotonic()
        self._last_heartbeat_time = now
        self._last_status_time = now

        self.logger.info(
            f"[START] {self.script_name} 启动 | "
            f"心跳间隔: {settings.HEARTBEAT_INTERVAL}s | "
            f"状态文件: {self._status_file}"
        )

        # 立即写入一次状态文件，标记已启动
        self._write_status_file()

    def update(self, symbol: str, records: int = 0, **kwargs):
        """
        更新指定币对的采集统计

        每次成功写入数据后调用，累加记录数。

        Args:
            symbol:  交易对，如 "BTC/USDT"
            records: 本次新增的记录数
            **kwargs: 额外的统计字段，如 latest_id=123456789
                      会被合并到该币对的统计字典中
        """
        if symbol not in self._stats:
            self._stats[symbol] = {
                "records_this_round": 0,
                "total_records": 0,
            }

        stat = self._stats[symbol]
        stat["records_this_round"] += records
        stat["total_records"] += records

        # 合并额外字段
        for key, value in kwargs.items():
            stat[key] = value

    def set_status(self, status: str):
        """
        设置当前运行状态

        Args:
            status: 状态字符串
                    "running"     - 正常运行中
                    "catching_up" - 正在追赶历史数据
                    "idle"        - 已追上实时，等待下一轮
                    "error"       - 遇到错误
        """
        self._status = status

    def report_error(self, error: Exception):
        """
        记录错误信息

        Args:
            error: 异常实例

        不影响重试逻辑（重试由 retry.py 处理），仅用于心跳统计。
        """
        self._errors_count += 1
        self._last_error = f"{type(error).__name__}: {error}"
        self.logger.error(
            f"[ERROR] {self.script_name} | "
            f"累计错误: {self._errors_count} | "
            f"{self._last_error}"
        )

    def tick(self):
        """
        每轮循环结束时调用

        自动检查是否到达心跳间隔或状态文件更新间隔:
        - 到达 HEARTBEAT_INTERVAL → 输出心跳日志，重置本轮统计
        - 到达 STATUS_FILE_UPDATE_INTERVAL → 更新状态文件
        """
        now = time.monotonic()

        # 检查是否需要输出心跳日志
        if now - self._last_heartbeat_time >= settings.HEARTBEAT_INTERVAL:
            self._emit_heartbeat_log()
            self._last_heartbeat_time = now

        # 检查是否需要更新状态文件
        if now - self._last_status_time >= settings.STATUS_FILE_UPDATE_INTERVAL:
            self._write_status_file()
            self._last_status_time = now

    def stop(self):
        """
        标记脚本停止，更新状态文件

        应在脚本退出时（如收到 SIGTERM/SIGINT）调用。
        """
        self._status = "stopped"
        self._write_status_file()
        self.logger.info(
            f"[STOP] {self.script_name} 停止 | "
            f"运行时长: {self._format_uptime()} | "
            f"累计错误: {self._errors_count}"
        )

    # ===================================================================
    # 内部方法
    # ===================================================================

    def _emit_heartbeat_log(self):
        """
        输出心跳日志

        格式:
            [HEARTBEAT] collect_ticks | 运行 2h 13m | 本轮: BTC +3,200笔 ETH +1,800笔 | 总计: 1,234,567笔 | 错误: 0
        """
        # 构建本轮摘要: "BTC +3,200笔 ETH +1,800笔"
        round_parts = []
        total_all = 0
        for symbol, stat in self._stats.items():
            round_count = stat["records_this_round"]
            if round_count > 0:
                # 简化币对名: "BTC/USDT" → "BTC"
                short_name = symbol.split("/")[0]
                round_parts.append(f"{short_name} +{round_count:,}笔")
            total_all += stat["total_records"]

        round_summary = " ".join(round_parts) if round_parts else "无新数据"

        self.logger.info(
            f"[HEARTBEAT] {self.script_name} | "
            f"运行 {self._format_uptime()} | "
            f"状态: {self._status} | "
            f"本轮: {round_summary} | "
            f"总计: {total_all:,}笔 | "
            f"错误: {self._errors_count}"
        )

        # 重置本轮统计（总计保留）
        for stat in self._stats.values():
            stat["records_this_round"] = 0

    def _write_status_file(self):
        """
        将当前状态写入 JSON 文件

        使用原子写入（先写 .tmp 再重命名），防止读取方读到不完整的文件。
        """
        now_utc = datetime.now(timezone.utc)
        uptime = (now_utc - self._started_at).total_seconds() if self._started_at else 0

        status_data = {
            "script": self.script_name,
            "status": self._status,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "last_heartbeat": now_utc.isoformat(),
            "uptime_seconds": int(uptime),
            "stats": self._stats,
            "errors_count": self._errors_count,
            "last_error": self._last_error,
        }

        # 原子写入: 先写临时文件，再重命名
        tmp_path = self._status_file + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(status_data, f, indent=2, ensure_ascii=False)
            # os.replace 是原子操作（同一文件系统内）
            os.replace(tmp_path, self._status_file)
        except OSError as e:
            self.logger.warning(f"状态文件写入失败: {e}")

    def _format_uptime(self) -> str:
        """
        将运行时长格式化为易读字符串

        Examples:
            "3m"       — 3 分钟
            "2h 13m"   — 2 小时 13 分钟
            "1d 5h"    — 1 天 5 小时
        """
        if self._started_at is None:
            return "未启动"

        elapsed = (datetime.now(timezone.utc) - self._started_at).total_seconds()
        elapsed = int(elapsed)

        if elapsed < 60:
            return f"{elapsed}s"

        minutes = elapsed // 60
        if minutes < 60:
            return f"{minutes}m"

        hours = minutes // 60
        remaining_minutes = minutes % 60
        if hours < 24:
            return f"{hours}h {remaining_minutes}m"

        days = hours // 24
        remaining_hours = hours % 24
        return f"{days}d {remaining_hours}h"
