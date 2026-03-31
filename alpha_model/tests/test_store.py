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
from sklearn.linear_model import Ridge
from alpha_model.core.pipeline import AlphaPipeline
from alpha_model.config import DEFAULT_SYMBOLS
from alpha_model.store.signal_store import SignalStore
from alpha_model.store.model_store import ModelStore
from factor_research.store.factor_store import FactorStore
from factor_research.core.types import FactorMeta, FactorType


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


# T1–T4 用到的 helper，与 _make_weights 等价（保持语义清晰）
make_dummy_panel = _make_weights


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
# TestRawPredictions（T1–T4）
# ---------------------------------------------------------------------------

class TestRawPredictions:
    """SignalStore raw_predictions 新功能测试（T1–T4）"""

    @pytest.fixture
    def store(self, tmp_path):
        return SignalStore(base_dir=tmp_path / "signals")

    def test_save_load_raw_predictions_roundtrip(self, store):
        """T1: save() 传入 raw_predictions → load_raw_predictions() 可正确还原"""
        raw = make_dummy_panel()
        store.save("s1", weights=make_dummy_panel(), raw_predictions=raw)
        loaded = store.load_raw_predictions("s1")
        pd.testing.assert_frame_equal(loaded, raw, check_freq=False)

    def test_has_raw_predictions(self, store):
        """T2: 不传 raw_predictions 时为 False；传入后为 True"""
        store.save("no_raw", weights=make_dummy_panel())
        assert store.has_raw_predictions("no_raw") is False

        store.save("with_raw", weights=make_dummy_panel(),
                   raw_predictions=make_dummy_panel())
        assert store.has_raw_predictions("with_raw") is True

    def test_has_signals(self, store):
        """T3: 不传 signals 时为 False；传入后为 True"""
        store.save("no_sig", weights=make_dummy_panel())
        assert store.has_signals("no_sig") is False

        store.save("with_sig", weights=make_dummy_panel(),
                   signals=make_dummy_panel())
        assert store.has_signals("with_sig") is True

    def test_load_raw_predictions_missing_raises(self, store):
        """T4: 未存储 raw_predictions 时，load_raw_predictions 应抛出 FileNotFoundError"""
        store.save("s1", weights=make_dummy_panel())
        with pytest.raises(FileNotFoundError, match="原始预测文件不存在"):
            store.load_raw_predictions("s1")


# ---------------------------------------------------------------------------
# T5：端到端 pipeline raw_predictions 存储测试
# ---------------------------------------------------------------------------

def test_pipeline_save_includes_raw_predictions(tmp_path, monkeypatch):
    """
    T5: 端到端——pipeline.run() + pipeline.save() 之后，
    SignalStore 中应存在 raw_predictions，
    且 shape 与 WalkForwardResult.predictions 一致。
    """
    symbols = ["BTC/USDT", "ETH/USDT"]
    n = 500
    rng = np.random.RandomState(0)
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")

    # 构建合成价格面板
    returns = rng.randn(n, len(symbols)) * 0.001
    price_panel = pd.DataFrame(
        100 * np.exp(np.cumsum(returns, axis=0)),
        index=idx, columns=symbols,
    )

    # 构建合成因子并写入临时 FactorStore
    factor_dir = tmp_path / "factors"
    monkeypatch.setattr("factor_research.config.FACTOR_STORE_DIR", str(factor_dir))
    monkeypatch.setattr(
        "factor_research.store.factor_store.FACTOR_STORE_DIR", str(factor_dir)
    )
    factor_store = FactorStore()
    factor_names = []
    for i in range(3):
        name = f"t5_factor_{i}"
        panel = pd.DataFrame(
            rng.randn(n, len(symbols)) * 0.01,
            index=idx, columns=symbols,
        )
        meta = FactorMeta(
            name=name, display_name=name,
            factor_type=FactorType.TIME_SERIES, category="test",
            description="t5 test factor",
            data_requirements=[],
            output_freq="1min",
        )
        factor_store.save(name, panel, meta)
        factor_names.append(name)

    # patch alpha_model 存储路径
    # 必须同时 patch config 模块和 store 模块的局部绑定（与 FactorStore 同理）[R1]
    monkeypatch.setattr("alpha_model.config.SIGNAL_STORE_DIR", tmp_path / "signals")
    monkeypatch.setattr("alpha_model.store.signal_store.SIGNAL_STORE_DIR", tmp_path / "signals")
    monkeypatch.setattr("alpha_model.config.MODEL_STORE_DIR", tmp_path / "models")
    monkeypatch.setattr("alpha_model.store.model_store.MODEL_STORE_DIR", tmp_path / "models")

    # 构建并运行 pipeline
    pipeline = AlphaPipeline(
        model=Ridge(alpha=1.0),
        train_config=TrainConfig(
            train_periods=100, test_periods=50,
            target_horizon=5, purge_periods=10,
        ),
        constraints=PortfolioConstraints(vol_target=None),
        factor_names=factor_names,
    )
    pipeline.run(price_panel, symbols=symbols)
    pipeline.save("test_strategy")

    # 验证 raw_predictions 已存入 SignalStore
    # 使用无参构造（已 patch SIGNAL_STORE_DIR），与 pipeline.save() 写入路径一致
    store = SignalStore()
    assert store.has_raw_predictions("test_strategy"), "raw_predictions 应已存储"

    raw = store.load_raw_predictions("test_strategy")
    assert isinstance(raw, pd.DataFrame)
    assert set(raw.columns) == set(symbols)
    assert len(raw) > 0

    # 验证与 pipeline 内部 predictions 一致
    # check_names=False：_load_parquet 将 index.name 设为 None（既有行为），
    # 而 predictions.index.name 为 'timestamp'，两者数值完全一致
    pd.testing.assert_frame_equal(
        raw, pipeline._wf_result.predictions, check_freq=False, check_names=False
    )


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
