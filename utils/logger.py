"""
统一日志模块

提供标准化的日志输出格式，同时输出到控制台和文件。
日志文件按日期轮转，保留最近 30 天。

日志格式:
    2024-01-15 10:30:45.123 | INFO     | data.fetcher | BTC/USDT: 拉取 1000 根 K线

用法:
    from utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("启动K线采集")
    logger.warning("API 返回空数据")
    logger.error("连接超时", exc_info=True)

日志级别说明:
    - DEBUG:   调试信息（如每次 API 请求的参数和返回行数）
    - INFO:    常规运行信息（如每轮采集的汇总）
    - WARNING: 异常但可恢复的情况（如重试、数据校验失败）
    - ERROR:   错误（如多次重试后仍失败）
    - CRITICAL: 致命错误（如数据库损坏）

依赖: config.settings (LOG_DIR)
"""

import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler

from config import settings

# === 模块级单例: 所有 logger 共享同一个文件 handler ===
# 原设计中，每个 get_logger("不同名字") 都创建独立的 TimedRotatingFileHandler，
# 全部指向同一个 app.log。当多个 handler 同时尝试轮转时：
#   - Windows: os.rename 无法操作被其他 handler 占用的文件 → PermissionError
#   - Linux: 第一个 handler 轮转后，其余 handler 仍持有旧文件描述符 → 日志写入错误文件
# 改为单例后，只有一个 handler 操作文件，彻底消除竞争。
_shared_file_handler: logging.Handler | None = None


def _get_shared_file_handler() -> logging.Handler:
    """
    获取共享的文件 handler（懒加载单例）

    首次调用时创建 TimedRotatingFileHandler，后续调用直接返回同一个实例。
    Windows 兼容: 自定义 rotator，在轮转前先删除已存在的目标文件，
    避免 os.rename 在 Windows 上因目标已存在而报 PermissionError。
    （Linux 上 os.rename 本身支持原子覆盖，该逻辑不会产生副作用。）
    """
    global _shared_file_handler
    if _shared_file_handler is not None:
        return _shared_file_handler

    os.makedirs(settings.LOG_DIR, exist_ok=True)
    log_file = os.path.join(settings.LOG_DIR, "app.log")

    handler = TimedRotatingFileHandler(
        filename=log_file,
        when="midnight",     # 每天午夜轮转
        interval=1,          # 每 1 天
        backupCount=30,      # 保留 30 天
        encoding="utf-8",    # 支持中文日志
    )

    # Windows 兼容: 自定义 rotator
    # 默认 rotator 使用 os.rename()，在 Windows 上如果目标文件已存在会失败。
    # 场景: 进程在同一天内重启，上次运行已成功轮转生成了 app.log.2026-03-11，
    # 新进程再次尝试轮转到同名目标文件时，os.rename 会报错。
    # 解决: 先删除已存在的目标文件，再执行 rename。
    if sys.platform == "win32":
        def _windows_rotator(source: str, dest: str) -> None:
            """Windows 下安全的日志轮转: 先删除目标再重命名"""
            if os.path.exists(dest):
                os.remove(dest)
            os.rename(source, dest)
        handler.rotator = _windows_rotator

    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)

    _shared_file_handler = handler
    return _shared_file_handler


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的 logger 实例

    每个模块应使用 get_logger(__name__) 获取自己的 logger。
    所有 logger 共享同一个文件 handler（输出到同一个日志文件），
    但通过 name 参数区分来源。

    Args:
        name: logger 名称，通常传入 __name__
              例: "data.fetcher", "scripts.collect_klines"

    Returns:
        配置好的 Logger 实例
        - 控制台输出: INFO 及以上级别
        - 文件输出: DEBUG 及以上级别（包含更详细的调试信息）
    """
    logger = logging.getLogger(name)

    # 避免重复添加 handler（模块被多次 import 时）
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # === 日志格式 ===
    # 统一格式: 时间 | 级别 | 模块名 | 消息
    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # === 控制台 Handler ===
    # 每个 logger 独立创建控制台 handler，只输出 INFO 及以上级别
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # === 文件 Handler ===
    # 所有 logger 共享同一个 handler 实例，避免多个 handler 同时轮转同一文件
    logger.addHandler(_get_shared_file_handler())

    return logger
