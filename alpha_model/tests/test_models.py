"""
models/ 参考实现的单元测试

测试:
    - linear_models.py: SklearnModelWrapper
    - tree_models.py: LGBMModelWrapper (如已安装)
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from alpha_model.core.types import AlphaModel
from alpha_model.models.linear_models import SklearnModelWrapper


def _make_data(n=200, n_features=5, seed=42):
    """生成训练数据"""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        rng.randn(n, n_features),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.Series(X.sum(axis=1) + rng.randn(n) * 0.1)
    return X, y


# ---------------------------------------------------------------------------
# SklearnModelWrapper
# ---------------------------------------------------------------------------

class TestSklearnModelWrapper:
    """sklearn 模型封装"""

    def test_satisfies_protocol(self):
        """应满足 AlphaModel 协议"""
        from sklearn.linear_model import Ridge
        model = SklearnModelWrapper(Ridge())
        assert isinstance(model, AlphaModel)

    def test_fit_predict(self):
        """基本训练和预测"""
        from sklearn.linear_model import Ridge
        model = SklearnModelWrapper(Ridge(alpha=1.0))
        X, y = _make_data()
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        # 预测值应有一定相关性
        corr = np.corrcoef(preds, y)[0, 1]
        assert corr > 0.5

    def test_feature_importance(self):
        """因子重要性"""
        from sklearn.linear_model import Ridge
        model = SklearnModelWrapper(Ridge(alpha=1.0))
        X, y = _make_data()
        model.fit(X, y)
        importance = model.get_feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) == X.shape[1]

    def test_save_load_roundtrip(self, tmp_path):
        """保存 → 加载往返一致"""
        from sklearn.linear_model import Ridge
        model = SklearnModelWrapper(Ridge(alpha=1.0))
        X, y = _make_data()
        model.fit(X, y)

        preds_before = model.predict(X)
        model.save_model(tmp_path)

        model2 = SklearnModelWrapper(Ridge())
        model2.load_model(tmp_path)
        preds_after = model2.predict(X)

        np.testing.assert_allclose(preds_before, preds_after)

    def test_get_params(self):
        """获取模型参数"""
        from sklearn.linear_model import Ridge
        model = SklearnModelWrapper(Ridge(alpha=2.0))
        params = model.get_params()
        assert params["alpha"] == 2.0

    def test_lasso_wrapper(self):
        """Lasso 封装"""
        from sklearn.linear_model import Lasso
        model = SklearnModelWrapper(Lasso(alpha=0.01))
        X, y = _make_data()
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)

    def test_elasticnet_wrapper(self):
        """ElasticNet 封装"""
        from sklearn.linear_model import ElasticNet
        model = SklearnModelWrapper(ElasticNet(alpha=0.1, l1_ratio=0.5))
        X, y = _make_data()
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)


class TestSklearnNativeProtocol:
    """sklearn 原生模型直接满足协议（不需要封装）"""

    def test_ridge_native(self):
        """Ridge 原生直接使用"""
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
        assert isinstance(model, AlphaModel)
        X, y = _make_data()
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)


# ---------------------------------------------------------------------------
# LGBMModelWrapper (可选)
# ---------------------------------------------------------------------------

class TestLGBMModelWrapper:
    """LightGBM 封装（如果已安装）"""

    @pytest.fixture(autouse=True)
    def check_lgbm(self):
        try:
            import lightgbm
        except ImportError:
            pytest.skip("lightgbm 未安装")

    def test_fit_predict(self):
        from alpha_model.models.tree_models import LGBMModelWrapper
        model = LGBMModelWrapper(n_estimators=10, num_leaves=8)
        X, y = _make_data()
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)

    def test_feature_importance(self):
        from alpha_model.models.tree_models import LGBMModelWrapper
        model = LGBMModelWrapper(n_estimators=10, num_leaves=8)
        X, y = _make_data()
        model.fit(X, y)
        importance = model.get_feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) == X.shape[1]

    def test_save_load(self, tmp_path):
        from alpha_model.models.tree_models import LGBMModelWrapper
        model = LGBMModelWrapper(n_estimators=10, num_leaves=8)
        X, y = _make_data()
        model.fit(X, y)

        preds_before = model.predict(X)
        model.save_model(tmp_path)

        model2 = LGBMModelWrapper()
        model2.load_model(tmp_path)
        preds_after = model2.predict(X)

        np.testing.assert_allclose(preds_before, preds_after)


# ---------------------------------------------------------------------------
# XGBModelWrapper (可选) [T10]
# ---------------------------------------------------------------------------

class TestXGBModelWrapper:
    """XGBoost 封装（如果已安装）"""

    @pytest.fixture(autouse=True)
    def check_xgb(self):
        try:
            import xgboost
        except ImportError:
            pytest.skip("xgboost 未安装")

    def test_fit_predict(self):
        from alpha_model.models.tree_models import XGBModelWrapper
        model = XGBModelWrapper(n_estimators=10, max_depth=3)
        X, y = _make_data()
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)

    def test_feature_importance(self):
        from alpha_model.models.tree_models import XGBModelWrapper
        model = XGBModelWrapper(n_estimators=10, max_depth=3)
        X, y = _make_data()
        model.fit(X, y)
        importance = model.get_feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) == X.shape[1]

    def test_save_load(self, tmp_path):
        from alpha_model.models.tree_models import XGBModelWrapper
        model = XGBModelWrapper(n_estimators=10, max_depth=3)
        X, y = _make_data()
        model.fit(X, y)

        preds_before = model.predict(X)
        model.save_model(tmp_path)

        model2 = XGBModelWrapper()
        model2.load_model(tmp_path)
        preds_after = model2.predict(X)
        np.testing.assert_allclose(preds_before, preds_after, atol=1e-5)


# ---------------------------------------------------------------------------
# TorchModelBase (可选) [T10/R6]
# ---------------------------------------------------------------------------

class TestTorchModelBase:
    """PyTorch 基础封装（如果已安装）"""

    @pytest.fixture(autouse=True)
    def check_torch(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch 未安装")

    def _make_mlp(self):
        """构造简单 MLP 子类"""
        from alpha_model.models.torch_base import TorchModelBase

        class SimpleMLP(TorchModelBase):
            def build_network(self, n_features):
                import torch.nn as nn
                return nn.Sequential(
                    nn.Linear(n_features, 16),
                    nn.ReLU(),
                    nn.Linear(16, 1),
                )
        return SimpleMLP(
            val_ratio=0.2, patience=5, max_epochs=20,
            batch_size=32, lr=1e-3,
        )

    def test_fit_predict(self):
        model = self._make_mlp()
        X, y = _make_data(n=200)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert np.all(np.isfinite(preds))

    def test_save_load_roundtrip(self, tmp_path):
        model = self._make_mlp()
        X, y = _make_data(n=200)
        model.fit(X, y)

        preds_before = model.predict(X)
        model.save_model(tmp_path)

        model2 = self._make_mlp()
        model2.load_model(tmp_path)
        preds_after = model2.predict(X)
        np.testing.assert_allclose(preds_before, preds_after, atol=1e-5)

    def test_early_stopping_activates(self):
        """[R6] early stopping 应在 patience 轮无改善后提前停止"""
        model = self._make_mlp()
        model.max_epochs = 1000
        model.patience = 3
        X, y = _make_data(n=200)
        model.fit(X, y)
        # 显式验证 actual_epochs_ < max_epochs
        assert model.actual_epochs_ < model.max_epochs
        preds = model.predict(X)
        assert len(preds) == len(X)
