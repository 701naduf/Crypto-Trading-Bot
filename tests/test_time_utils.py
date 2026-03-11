"""
time_utils 模块测试

纯本地测试，不需要网络连接。
"""

from datetime import datetime, timezone

import pytest

from utils.time_utils import (
    align_to_timeframe,
    datetime_to_ms,
    is_standard_timeframe,
    ms_to_datetime,
    timeframe_to_seconds,
)


class TestMsToDatetime:
    def test_basic(self):
        dt = ms_to_datetime(1705305600000)
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15
        assert dt.hour == 8
        assert dt.tzinfo == timezone.utc

    def test_zero(self):
        dt = ms_to_datetime(0)
        assert dt == datetime(1970, 1, 1, tzinfo=timezone.utc)

    def test_roundtrip(self):
        """ms → datetime → ms 往返转换"""
        original = 1705305600000
        dt = ms_to_datetime(original)
        result = datetime_to_ms(dt)
        assert result == original


class TestDatetimeToMs:
    def test_basic(self):
        dt = datetime(2024, 1, 15, 8, 0, 0, tzinfo=timezone.utc)
        assert datetime_to_ms(dt) == 1705305600000

    def test_naive_treated_as_utc(self):
        """没有时区信息的 datetime 应被视为 UTC"""
        dt_naive = datetime(2024, 1, 15, 8, 0, 0)
        dt_utc = datetime(2024, 1, 15, 8, 0, 0, tzinfo=timezone.utc)
        assert datetime_to_ms(dt_naive) == datetime_to_ms(dt_utc)


class TestTimeframeToSeconds:
    def test_seconds(self):
        assert timeframe_to_seconds("10s") == 10
        assert timeframe_to_seconds("30s") == 30

    def test_minutes(self):
        assert timeframe_to_seconds("1m") == 60
        assert timeframe_to_seconds("5m") == 300
        assert timeframe_to_seconds("15m") == 900

    def test_hours(self):
        assert timeframe_to_seconds("1h") == 3600
        assert timeframe_to_seconds("4h") == 14400

    def test_days(self):
        assert timeframe_to_seconds("1d") == 86400

    def test_invalid_unit(self):
        with pytest.raises(ValueError):
            timeframe_to_seconds("1w")


class TestAlignToTimeframe:
    def test_1h(self):
        dt = datetime(2024, 1, 15, 10, 23, 45, tzinfo=timezone.utc)
        aligned = align_to_timeframe(dt, "1h")
        assert aligned == datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

    def test_5m(self):
        dt = datetime(2024, 1, 15, 10, 23, 45, tzinfo=timezone.utc)
        aligned = align_to_timeframe(dt, "5m")
        assert aligned == datetime(2024, 1, 15, 10, 20, 0, tzinfo=timezone.utc)

    def test_30s(self):
        dt = datetime(2024, 1, 15, 10, 23, 45, tzinfo=timezone.utc)
        aligned = align_to_timeframe(dt, "30s")
        assert aligned == datetime(2024, 1, 15, 10, 23, 30, tzinfo=timezone.utc)

    def test_already_aligned(self):
        """已对齐的时间不应变化"""
        dt = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        assert align_to_timeframe(dt, "1h") == dt


class TestIsStandardTimeframe:
    def test_standard(self):
        for tf in ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"]:
            assert is_standard_timeframe(tf), f"{tf} 应该是标准周期"

    def test_non_standard(self):
        for tf in ["10s", "30s", "2m", "7m"]:
            assert not is_standard_timeframe(tf), f"{tf} 不应该是标准周期"
