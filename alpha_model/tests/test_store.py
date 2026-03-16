"""
store/ 模块的单元测试

测试:
    - signal_store.py: 信号持久化
    - model_store.py: 模型持久化
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path

from alpha_model.core.types import (
    ModelMeta,
    TrainConfig,
    PortfolioConstraints,
)
from alpha_model.store.signal_store import SignalStore
from alpha_model.store.model_store import ModelStore


class _DummyModel:
    """用于测试的模型"""

    def __init__(self, value=42.0):
        self.value = value

    def fit(self, X, y, **kwargs):
        self.value = y.mean()

    def predict(self, X):
        return np.full(len(X), self.value)

    def save_model(self, path):
        path = Path(path)
        with open(path / "dummy.txt", "w") as f:
            f.write(str(self.value))

    def load_model(self, path):
        path = Path(path)
        with open(path / "dummy.txt", "r") as f:
            self.value = float(f.read())


def _make_meta() -> ModelMeta:
    return ModelMeta(
        name="test_strategy",
        factor_names=["f1", "f2"],
        target_horizon=10,
        train_config=TrainConfig(),
        constraints=PortfolioConstraints(),
    )


def _make_weights(n=50) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {"BTC/USDT": np.random.randn(n) * 0.1, "ETH/USDT": np.random.randn(n) * 0.1},
        index=idx,
    )


# ---------------------------------------------------------------------------
# SignalStore
# ---------------------------------------------------------------------------

class TestSignalStore:
    """信号持久化"""

    @pytest.fixture
    def store(self, tmp_path):
        return SignalStore(base_dir=tmp_path / "signals")

    def test_save_and_load_weights(self, store):
        """保存并加载权重"""
        weights = _make_weights()
        store.save("test", weights=weights)
        loaded = store.load_weights("test")
        pd.testing.assert_frame_equal(weights, loaded, check_freq=False)

    def test_save_and_load_signals(self, store):
        """保存并加载信号"""
        weights = _make_weights()
        signals = _make_weights()
        store.save("test", weights=weights, signals=signals)
        loaded = store.load_signals("test")
        pd.testing.assert_frame_equal(signals, loaded, check_freq=False)

    def test_save_and_load_meta(self, store):
        """保存并加载元数据"""
        weights = _make_weights()
        meta = _make_meta()
        store.save("test", weights=weights, meta=meta)
        loaded = store.load_meta("test")
        assert loaded.name == meta.name
        assert loaded.factor_names == meta.factor_names

    def test_save_and_load_performance(self, store):
        """保存并加载绩效"""
        weights = _make_weights()
        perf = {"sharpe_ratio": 1.5, "max_drawdown": -0.1}
        store.save("test", weights=weights, performance=perf)
        loaded = store.load_performance("test")
        assert loaded["sharpe_ratio"] == 1.5

    def test_list_strategies(self, store):
        """列出策略"""
        store.save("strat_a", weights=_make_weights())
        store.save("strat_b", weights=_make_weights())
        strategies = store.list_strategies()
        assert "strat_a" in strategies
        assert "strat_b" in strategies

    def test_exists(self, store):
        """检查策略是否存在"""
        store.save("test", weights=_make_weights())
        assert store.exists("test")
        assert not store.exists("nonexistent")

    def test_delete(self, store):
        """删除策略"""
        store.save("test", weights=_make_weights())
        assert store.exists("test")
        store.delete("test")
        assert not store.exists("test")

    def test_overwrite(self, store):
        """覆盖已有策略"""
        w1 = _make_weights(n=10)
        w2 = _make_weights(n=20)
        store.save("test", weights=w1)
        store.save("test", weights=w2)
        loaded = store.load_weights("test")
        assert len(loaded) == 20

    def test_load_nonexistent_raises(self, store):
        """加载不存在的策略应报错"""
        with pytest.raises(FileNotFoundError):
            store.load_weights("nonexistent")


# ---------------------------------------------------------------------------
# ModelStore
# ---------------------------------------------------------------------------

class TestModelStore:
    """模型持久化"""

    @pytest.fixture
    def store(self, tmp_path):
        return ModelStore(base_dir=tmp_path / "models")

    def test_save_and_load(self, store):
        """保存并加载模型"""
        model = _DummyModel(value=3.14)
        meta = _make_meta()
        store.save("test_model", model, meta)

        loaded_model, loaded_meta = store.load(
            "test_model",
            model_factory=lambda: _DummyModel(),
        )
        assert loaded_model.value == 3.14
        assert loaded_meta.name == "test_strategy"

    def test_save_with_importance(self, store):
        """保存并加载因子重要性"""
        model = _DummyModel()
        meta = _make_meta()
        importance = {"f1": 0.8, "f2": 0.2}
        store.save("test", model, meta, importance=importance)

        loaded = store.load_importance("test")
        assert loaded["f1"] == 0.8
        assert loaded["f2"] == 0.2

    def test_list_models(self, store):
        """列出模型"""
        model = _DummyModel()
        meta = _make_meta()
        store.save("model_a", model, meta)
        store.save("model_b", model, meta)
        models = store.list_models()
        assert "model_a" in models
        assert "model_b" in models

    def test_exists(self, store):
        model = _DummyModel()
        meta = _make_meta()
        store.save("test", model, meta)
        assert store.exists("test")
        assert not store.exists("nonexistent")

    def test_delete(self, store):
        model = _DummyModel()
        meta = _make_meta()
        store.save("test", model, meta)
        store.delete("test")
        assert not store.exists("test")

    def test_load_nonexistent_raises(self, store):
        with pytest.raises(FileNotFoundError):
            store.load("nonexistent", lambda: _DummyModel())

    def test_roundtrip_predict(self, store):
        """保存 → 加载 → predict 结果一致"""
        model = _DummyModel(value=99.0)
        meta = _make_meta()
        store.save("test", model, meta)

        loaded_model, _ = store.load("test", lambda: _DummyModel())
        X = pd.DataFrame({"a": [1, 2, 3]})
        preds = loaded_model.predict(X)
        np.testing.assert_array_equal(preds, [99.0, 99.0, 99.0])
