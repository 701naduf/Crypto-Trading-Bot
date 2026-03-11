"""
config 模块测试

测试配置加载和基本配置项的正确性。
不需要网络连接，纯本地测试。
"""

import os
from pathlib import Path

import pytest


class TestSettings:
    """settings.py 基本配置测试"""

    def test_import(self):
        """能正常导入 settings 模块"""
        from data_infra.config import settings
        assert settings is not None

    def test_exchange_id(self):
        """交易所 ID 是有效值"""
        from data_infra.config import settings
        assert settings.EXCHANGE_ID in ("binance", "okx")

    def test_symbols_not_empty(self):
        """交易对列表不为空"""
        from data_infra.config import settings
        assert len(settings.SYMBOLS) > 0

    def test_symbols_format(self):
        """交易对格式正确: BASE/QUOTE"""
        from data_infra.config import settings
        for s in settings.SYMBOLS:
            assert "/" in s, f"交易对格式错误: {s}"
            base, quote = s.split("/")
            assert len(base) > 0 and len(quote) > 0

    def test_kline_config(self):
        """K线配置项合理"""
        from data_infra.config import settings
        assert settings.KLINE_COLLECT_TIMEFRAME == "1m"
        assert settings.KLINE_LIMIT > 0
        assert settings.KLINE_COLLECT_INTERVAL > 0

    def test_tick_config(self):
        """Tick 配置项合理"""
        from data_infra.config import settings
        assert settings.TICK_FETCH_LIMIT > 0
        assert settings.TICK_IDLE_INTERVAL > 0
        assert settings.TICK_COLD_START_MODE in ("latest", "from_date")

    def test_orderbook_config(self):
        """订单簿配置项合理"""
        from data_infra.config import settings
        assert settings.ORDERBOOK_DEPTH in (5, 10, 20)
        assert settings.ORDERBOOK_UPDATE_SPEED in ("100ms", "1000ms")

    def test_paths_exist_or_creatable(self):
        """存储路径的父目录存在"""
        from data_infra.config import settings
        # DB_DIR 应该可以创建
        parent = Path(settings.DB_DIR).parent
        assert parent.exists(), f"DB_DIR 父目录不存在: {parent}"

    def test_retry_config(self):
        """重试配置合理"""
        from data_infra.config import settings
        assert settings.MAX_RETRIES >= 1
        assert settings.RETRY_BASE_DELAY > 0
        assert settings.RATE_LIMIT_PAUSE > 0
        assert settings.REQUEST_TIMEOUT > 0

    def test_heartbeat_config(self):
        """心跳配置合理"""
        from data_infra.config import settings
        assert settings.HEARTBEAT_INTERVAL > 0
        assert settings.STATUS_FILE_UPDATE_INTERVAL > 0
