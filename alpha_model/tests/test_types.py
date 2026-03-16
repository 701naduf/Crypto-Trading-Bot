"""
core/types.py 的单元测试

测试:
    - AlphaModel Protocol 协议检查
    - TrainConfig 验证逻辑
    - PortfolioConstraints 验证逻辑
    - ModelMeta 序列化/反序列化
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from alpha_model.core.types import (
    AlphaModel,
    TrainMode,
    WalkForwardMode,
    TrainConfig,
    PortfolioConstraints,
    ModelMeta,
)


# ---------------------------------------------------------------------------
# AlphaModel Protocol
# ---------------------------------------------------------------------------

class _DummyModel:
    """最小满足 AlphaModel 协议的模型"""
    def fit(self, X, y, **kwargs):
        pass

    def predict(self, X):
        return np.zeros(len(X))


class _IncompleteModel:
    """不满足 AlphaModel 协议的对象（没有 predict）"""
    def fit(self, X, y):
        pass


class TestAlphaModelProtocol:
    """AlphaModel 协议检查"""

    def test_dummy_model_satisfies_protocol(self):
        """最小模型满足协议"""
        model = _DummyModel()
        assert isinstance(model, AlphaModel)

    def test_sklearn_ridge_satisfies_protocol(self):
        """sklearn Ridge 天然满足协议"""
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
        assert isinstance(model, AlphaModel)

    def test_incomplete_model_fails_protocol(self):
        """缺少 predict 的对象不满足协议"""
        model = _IncompleteModel()
        assert not isinstance(model, AlphaModel)

    def test_dummy_model_fit_predict(self):
        """最小模型可以正常 fit/predict"""
        model = _DummyModel()
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([0.1, 0.2, 0.3])
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 3


# ---------------------------------------------------------------------------
# TrainConfig
# ---------------------------------------------------------------------------

class TestTrainConfig:
    """TrainConfig 验证"""

    def test_default_config(self):
        """默认配置有效"""
        config = TrainConfig()
        assert config.train_mode == TrainMode.POOLED
        assert config.wf_mode == WalkForwardMode.EXPANDING
        assert config.target_horizon == 10
        assert config.purge_periods >= config.target_horizon

    def test_valid_config(self):
        """自定义有效配置"""
        config = TrainConfig(
            train_mode=TrainMode.PER_SYMBOL,
            wf_mode=WalkForwardMode.ROLLING,
            target_horizon=5,
            train_periods=3000,
            test_periods=500,
            purge_periods=60,
        )
        assert config.train_mode == TrainMode.PER_SYMBOL
        assert config.purge_periods == 60

    def test_invalid_target_horizon(self):
        """target_horizon < 1 应报错"""
        with pytest.raises(ValueError, match="target_horizon"):
            TrainConfig(target_horizon=0)

    def test_invalid_train_periods(self):
        """train_periods < 1 应报错"""
        with pytest.raises(ValueError, match="train_periods"):
            TrainConfig(train_periods=0)

    def test_invalid_test_periods(self):
        """test_periods < 1 应报错"""
        with pytest.raises(ValueError, match="test_periods"):
            TrainConfig(test_periods=0)

    def test_purge_less_than_horizon(self):
        """purge_periods < target_horizon 应报错"""
        with pytest.raises(ValueError, match="purge_periods"):
            TrainConfig(target_horizon=10, purge_periods=5)

    def test_purge_equals_horizon_ok(self):
        """purge_periods == target_horizon 应通过"""
        config = TrainConfig(target_horizon=10, purge_periods=10)
        assert config.purge_periods == 10


# ---------------------------------------------------------------------------
# PortfolioConstraints
# ---------------------------------------------------------------------------

class TestPortfolioConstraints:
    """PortfolioConstraints 验证"""

    def test_default_constraints(self):
        """默认约束有效"""
        c = PortfolioConstraints()
        assert c.max_weight == 0.4
        assert c.dollar_neutral is True
        assert c.vol_target is None

    def test_valid_constraints(self):
        """自定义有效约束"""
        c = PortfolioConstraints(
            max_weight=0.5,
            dollar_neutral=False,
            beta_neutral=True,
            vol_target=0.15,
            leverage_cap=3.0,
        )
        assert c.max_weight == 0.5
        assert c.vol_target == 0.15

    def test_invalid_max_weight_zero(self):
        """max_weight <= 0 应报错"""
        with pytest.raises(ValueError, match="max_weight"):
            PortfolioConstraints(max_weight=0)

    def test_invalid_max_weight_over_one(self):
        """max_weight > 1 应报错"""
        with pytest.raises(ValueError, match="max_weight"):
            PortfolioConstraints(max_weight=1.5)

    def test_invalid_leverage_cap(self):
        """leverage_cap <= 0 应报错"""
        with pytest.raises(ValueError, match="leverage_cap"):
            PortfolioConstraints(leverage_cap=0)

    def test_invalid_vol_target(self):
        """vol_target <= 0 应报错（非 None 时）"""
        with pytest.raises(ValueError, match="vol_target"):
            PortfolioConstraints(vol_target=-0.1)

    def test_invalid_risk_aversion(self):
        """risk_aversion < 0 应报错"""
        with pytest.raises(ValueError, match="risk_aversion"):
            PortfolioConstraints(risk_aversion=-1)

    def test_invalid_turnover_penalty(self):
        """turnover_penalty < 0 应报错"""
        with pytest.raises(ValueError, match="turnover_penalty"):
            PortfolioConstraints(turnover_penalty=-0.5)


# ---------------------------------------------------------------------------
# ModelMeta
# ---------------------------------------------------------------------------

class TestModelMeta:
    """ModelMeta 序列化/反序列化"""

    def _make_meta(self) -> ModelMeta:
        return ModelMeta(
            name="test_strategy",
            factor_names=["factor_a", "factor_b"],
            target_horizon=10,
            train_config=TrainConfig(),
            constraints=PortfolioConstraints(),
            description="测试策略",
        )

    def test_created_at_auto_filled(self):
        """created_at 应自动填充"""
        meta = self._make_meta()
        assert meta.created_at is not None
        assert isinstance(meta.created_at, datetime)

    def test_to_dict(self):
        """序列化为字典"""
        meta = self._make_meta()
        d = meta.to_dict()
        assert d["name"] == "test_strategy"
        assert d["factor_names"] == ["factor_a", "factor_b"]
        assert d["target_horizon"] == 10
        assert "train_config" in d
        assert "constraints" in d

    def test_roundtrip(self):
        """序列化 → 反序列化往返一致"""
        meta = self._make_meta()
        d = meta.to_dict()
        restored = ModelMeta.from_dict(d)

        assert restored.name == meta.name
        assert restored.factor_names == meta.factor_names
        assert restored.target_horizon == meta.target_horizon
        assert restored.train_config.train_mode == meta.train_config.train_mode
        assert restored.train_config.wf_mode == meta.train_config.wf_mode
        assert restored.constraints.max_weight == meta.constraints.max_weight
        assert restored.constraints.dollar_neutral == meta.constraints.dollar_neutral
        assert restored.version == meta.version
        assert restored.description == meta.description

    def test_from_dict_with_defaults(self):
        """反序列化时缺失字段使用默认值"""
        meta = self._make_meta()
        d = meta.to_dict()
        # 删除可选字段
        del d["version"]
        del d["description"]
        restored = ModelMeta.from_dict(d)
        assert restored.version == "1.0"
        assert restored.description == ""
