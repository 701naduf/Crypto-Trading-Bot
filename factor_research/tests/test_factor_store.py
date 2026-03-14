"""store/factor_store.py 的单元测试"""

import numpy as np
import pandas as pd
import pytest

from factor_research.core.types import (
    DataRequest,
    DataType,
    FactorMeta,
    FactorType,
)
from factor_research.store.factor_store import FactorStore


@pytest.fixture
def store(tmp_path):
    """使用临时目录的 FactorStore"""
    return FactorStore(base_dir=str(tmp_path / "factors"))


@pytest.fixture
def sample_panel():
    """生成测试用因子面板"""
    index = pd.date_range("2024-01-01", periods=100, freq="1min", tz="UTC")
    np.random.seed(42)
    return pd.DataFrame(
        {
            "BTC/USDT": np.random.randn(100),
            "ETH/USDT": np.random.randn(100),
        },
        index=index,
    )


@pytest.fixture
def sample_meta():
    """生成测试用 FactorMeta"""
    return FactorMeta(
        name="test_factor",
        display_name="测试因子",
        factor_type=FactorType.TIME_SERIES,
        category="test",
        description="测试用",
        data_requirements=[
            DataRequest(DataType.OHLCV, timeframe="1m", lookback_bars=60),
        ],
        output_freq="1m",
        params={"window": 60},
        author="test",
        version="1.0",
    )


class TestFactorStore:

    def test_save_and_load(self, store, sample_panel, sample_meta):
        store.save("test_factor", sample_panel, sample_meta)

        loaded = store.load("test_factor")
        assert isinstance(loaded, pd.DataFrame)
        assert loaded.shape == sample_panel.shape
        assert list(loaded.columns) == list(sample_panel.columns)
        # 值近似相等（浮点精度）
        np.testing.assert_allclose(loaded.values, sample_panel.values, atol=1e-10)
        assert list(loaded.index) == list(sample_panel.index)

    def test_save_empty_raises(self, store, sample_meta):
        empty = pd.DataFrame()
        with pytest.raises(ValueError, match="面板为空"):
            store.save("empty", empty, sample_meta)

    def test_load_nonexistent_raises(self, store):
        with pytest.raises(FileNotFoundError):
            store.load("nonexistent")

    def test_load_meta(self, store, sample_panel, sample_meta):
        store.save("test_factor", sample_panel, sample_meta)

        loaded_meta = store.load_meta("test_factor")
        assert loaded_meta.name == sample_meta.name
        assert loaded_meta.factor_type == sample_meta.factor_type
        assert loaded_meta.category == sample_meta.category
        assert loaded_meta.params == sample_meta.params
        assert len(loaded_meta.data_requirements) == 1
        assert loaded_meta.data_requirements[0].data_type == DataType.OHLCV

    def test_load_meta_nonexistent_raises(self, store):
        with pytest.raises(FileNotFoundError):
            store.load_meta("nonexistent")

    def test_list_factors(self, store, sample_panel, sample_meta):
        assert store.list_factors() == []

        store.save("factor_a", sample_panel, sample_meta)
        store.save("factor_b", sample_panel, sample_meta)

        factors = store.list_factors()
        assert sorted(factors) == ["factor_a", "factor_b"]

    def test_exists(self, store, sample_panel, sample_meta):
        assert not store.exists("test")
        store.save("test", sample_panel, sample_meta)
        assert store.exists("test")

    def test_delete(self, store, sample_panel, sample_meta):
        store.save("test", sample_panel, sample_meta)
        assert store.exists("test")

        store.delete("test")
        assert not store.exists("test")

    def test_delete_nonexistent_raises(self, store):
        with pytest.raises(FileNotFoundError):
            store.delete("nonexistent")

    def test_overwrite(self, store, sample_panel, sample_meta):
        """覆盖已有因子"""
        store.save("test", sample_panel, sample_meta)

        new_panel = sample_panel * 2
        store.save("test", new_panel, sample_meta)

        loaded = store.load("test")
        np.testing.assert_allclose(loaded.values, new_panel.values, atol=1e-10)
