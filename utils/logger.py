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
from logging.handlers import TimedRotatingFileHandler

from config import settings


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
    # 只输出 INFO 及以上级别，避免 DEBUG 信息刷屏
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # === 文件 Handler ===
    # 按天轮转，保留最近 30 天的日志文件
    # 文件命名: app.log, app.log.2024-01-14, app.log.2024-01-13, ...
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    log_file = os.path.join(settings.LOG_DIR, "app.log")

    file_handler = TimedRotatingFileHandler(
        filename=log_file,
        when="midnight",     # 每天午夜轮转
        interval=1,          # 每 1 天
        backupCount=30,      # 保留 30 天
        encoding="utf-8",    # 支持中文日志
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
