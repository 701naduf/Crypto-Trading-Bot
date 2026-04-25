"""
test_adv_helpers.py — Step 0b（Z12/Z14/Z15/Z16）

覆盖 alpha_model.backtest.adv_helpers 单元测试 + 跨模块一致性护栏。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from alpha_model.backtest.adv_helpers import safe_adv_panel, safe_adv_array


# ---------------------------------------------------------------------------
# safe_adv_panel
# ---------------------------------------------------------------------------

class TestSafeAdvPanel:

    def test_handles_nan_with_warning(self, caplog):
        idx = pd.date_range("2024-01-01", periods=5, freq="1min", tz="UTC")
        adv = pd.DataFrame({
            "BTC/USDT": [1e9, 1e9, np.nan, 1e9, 1e9],
            "ETH/USDT": [1e9] * 5,
        }, index=idx)
        with caplog.at_level(logging.WARNING):
            out = safe_adv_panel(adv, context="test")
        # NaN → 1.0
        assert not out.isna().any().any()
        assert out.iloc[2, 0] == 1.0
        # 非 NaN 不变
        assert out.iloc[0, 0] == 1e9
        # warning 触发
        assert any("含 NaN" in r.message for r in caplog.records)

    def test_handles_sub_one_floor(self):
        """ADV ∈ (0, 1) → floor 到 1.0；无 warning"""
        idx = pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC")
        adv = pd.DataFrame({"BTC/USDT": [0.5, 1e9, 0.0]}, index=idx)
        out = safe_adv_panel(adv, context="test")
        assert out.iloc[0, 0] == 1.0   # 0.5 → 1.0
        assert out.iloc[1, 0] == 1e9   # 1e9 不变
        assert out.iloc[2, 0] == 1.0   # 0 → 1.0

    def test_no_nan_no_warning(self, caplog):
        idx = pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC")
        adv = pd.DataFrame({"BTC/USDT": [1e9, 1e9, 1e9]}, index=idx)
        with caplog.at_level(logging.WARNING):
            safe_adv_panel(adv, context="test")
        # 不应有 warning
        warning_count = sum(1 for r in caplog.records if "含 NaN" in r.message)
        assert warning_count == 0


# ---------------------------------------------------------------------------
# safe_adv_array
# ---------------------------------------------------------------------------

class TestSafeAdvArray:

    def test_handles_nan(self, caplog):
        """NaN → 1.0 + warning（修 pre-existing np.maximum(NaN,1) bug）"""
        adv = np.array([np.nan, 1e9, 0.5])
        with caplog.at_level(logging.WARNING):
            out = safe_adv_array(adv, ["BTC", "ETH", "SOL"], context="test")
        assert out[0] == 1.0   # NaN → 1.0（修 bug）
        assert out[1] == 1e9
        assert out[2] == 1.0   # 0.5 → 1.0
        # warning 触发
        assert any("含 NaN" in r.message for r in caplog.records)

    def test_pre_existing_bug_fix(self):
        """单独验证修 np.maximum(NaN, 1.0) = NaN 这个 pre-existing bug"""
        # 直接验证 numpy 行为：np.maximum(NaN, 1.0) 返回 NaN
        assert np.isnan(np.maximum(np.nan, 1.0))
        # safe_adv_array 修后应返回 1.0
        out = safe_adv_array(np.array([np.nan]), ["BTC"], context="test")
        assert out[0] == 1.0
        assert not np.isnan(out[0])

    def test_len_mismatch_raises(self):
        """Z15: len(adv_arr) ≠ len(symbols) → ValueError"""
        with pytest.raises(ValueError, match="len.*"):
            safe_adv_array(np.array([1.0, 2.0]), ["BTC"], context="test")

    def test_dedup_same_symbol_only_warns_once(self, caplog):
        """Z16: 传入 warned_set，相同 symbol NaN 仅首次 warning"""
        warned: set[str] = set()
        adv = np.array([np.nan, 1e9])
        with caplog.at_level(logging.WARNING):
            safe_adv_array(adv, ["BTC", "ETH"], context="ctx1", warned_set=warned)
            safe_adv_array(adv, ["BTC", "ETH"], context="ctx2", warned_set=warned)
            safe_adv_array(adv, ["BTC", "ETH"], context="ctx3", warned_set=warned)
        # 仅首次触发 warning
        warning_msgs = [r.message for r in caplog.records if "含 NaN" in r.message]
        assert len(warning_msgs) == 1
        assert "BTC" in warned

    def test_dedup_new_symbol_still_warns(self, caplog):
        """Z16: 第二次出现新 symbol NaN 时仍 warning"""
        warned: set[str] = set()
        with caplog.at_level(logging.WARNING):
            # 第一次：BTC NaN
            safe_adv_array(np.array([np.nan, 1e9]), ["BTC", "ETH"], warned_set=warned)
            # 第二次：ETH NaN（新 symbol）
            safe_adv_array(np.array([1e9, np.nan]), ["BTC", "ETH"], warned_set=warned)
        warning_msgs = [r.message for r in caplog.records if "含 NaN" in r.message]
        assert len(warning_msgs) == 2   # 两次都 warning（不同 symbol 首次出现）

    def test_no_dedup_warns_each_call(self, caplog):
        """warned_set=None 时每次都 warning（适用于一次性调用）"""
        with caplog.at_level(logging.WARNING):
            safe_adv_array(np.array([np.nan]), ["BTC"], warned_set=None)
            safe_adv_array(np.array([np.nan]), ["BTC"], warned_set=None)
        warning_msgs = [r.message for r in caplog.records if "含 NaN" in r.message]
        assert len(warning_msgs) == 2   # 每次都 warning
