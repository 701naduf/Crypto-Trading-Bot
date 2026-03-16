"""
树模型参考实现

封装 LightGBM / XGBoost，使其满足 AlphaModel 协议。

注意:
    lightgbm 和 xgboost 是可选依赖。
    如果未安装，import 此模块不会报错，但实例化时会抛出 ImportError。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class LGBMModelWrapper:
    """
    LightGBM 封装

    用法:
        model = LGBMModelWrapper(
            objective="regression",
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=100,
        )
    """

    def __init__(self, **params):
        """
        Args:
            **params: LightGBM 训练参数
        """
        self.params = params
        self.model = None
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """训练 LightGBM 模型"""
        import lightgbm as lgb

        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()

        # 分离 n_estimators（lgb.train 使用 num_boost_round）
        params = dict(self.params)
        n_estimators = params.pop("n_estimators", 100)
        if "objective" not in params:
            params["objective"] = "regression"
        # 抑制输出
        params.setdefault("verbose", -1)

        dtrain = lgb.Dataset(X, label=y)
        self.model = lgb.train(params, dtrain, num_boost_round=n_estimators)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """生成预测值"""
        if self.model is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()")
        return self.model.predict(X)

    def save_model(self, path: Path) -> None:
        """保存模型"""
        if self.model is None:
            raise RuntimeError("模型尚未训练")
        path = Path(path)
        self.model.save_model(str(path / "model.txt"))

    def load_model(self, path: Path) -> None:
        """加载模型"""
        import lightgbm as lgb
        path = Path(path)
        self.model = lgb.Booster(model_file=str(path / "model.txt"))

    def get_feature_importance(self) -> dict[str, float]:
        """返回因子重要性（gain）"""
        if self.model is not None:
            importance = self.model.feature_importance(importance_type="gain")
            names = self.model.feature_name()
            return dict(zip(names, importance.tolist()))
        return {}

    def get_params(self) -> dict:
        """返回模型参数"""
        return dict(self.params)

    def __repr__(self) -> str:
        return f"LGBMModelWrapper({self.params})"


class XGBModelWrapper:
    """
    XGBoost 封装

    用法:
        model = XGBModelWrapper(
            objective="reg:squarederror",
            max_depth=6,
            learning_rate=0.05,
            n_estimators=100,
        )
    """

    def __init__(self, **params):
        self.params = params
        self.model = None
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """训练 XGBoost 模型"""
        import xgboost as xgb

        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()

        params = dict(self.params)
        if "objective" not in params:
            params["objective"] = "reg:squarederror"

        self.model = xgb.XGBRegressor(**params)
        self.model.fit(X, y, verbose=False)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """生成预测值"""
        if self.model is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()")
        return self.model.predict(X)

    def save_model(self, path: Path) -> None:
        """保存模型"""
        if self.model is None:
            raise RuntimeError("模型尚未训练")
        path = Path(path)
        self.model.save_model(str(path / "model.json"))

    def load_model(self, path: Path) -> None:
        """加载模型"""
        import xgboost as xgb
        path = Path(path)
        self.model = xgb.XGBRegressor()
        self.model.load_model(str(path / "model.json"))

    def get_feature_importance(self) -> dict[str, float]:
        """返回因子重要性"""
        if self.model is not None and hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
            if len(importance) == len(self.feature_names_):
                return dict(zip(self.feature_names_, importance.tolist()))
        return {}

    def get_params(self) -> dict:
        return dict(self.params)

    def __repr__(self) -> str:
        return f"XGBModelWrapper({self.params})"
