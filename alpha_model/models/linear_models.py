"""
线性模型参考实现

封装 sklearn 的线性模型，使其满足 AlphaModel 协议的可选方法。
sklearn 原生模型已有 fit/predict，这里补充 save_model/load_model/get_feature_importance。

用法:
    model = SklearnModelWrapper(Ridge(alpha=1.0))
    model = SklearnModelWrapper(Lasso(alpha=0.01))
    model = SklearnModelWrapper(ElasticNet(alpha=0.1, l1_ratio=0.5))

注意:
    sklearn 原生模型可以直接使用（天然满足 AlphaModel 协议的 fit/predict），
    此封装只是补充了 save_model/load_model/get_feature_importance 等可选方法。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class SklearnModelWrapper:
    """
    sklearn 模型通用封装

    将任何 sklearn 估计器封装为完整的 AlphaModel 协议实现。

    Attributes:
        estimator:      sklearn 估计器实例
        feature_names_: 训练时的特征名列表（fit 后设置）
    """

    def __init__(self, estimator):
        """
        Args:
            estimator: sklearn 估计器实例（如 Ridge(), Lasso()）
        """
        self.estimator = estimator
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """训练模型"""
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
        self.estimator.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """生成预测值"""
        return self.estimator.predict(X)

    def save_model(self, path: Path) -> None:
        """使用 joblib 保存模型"""
        import joblib
        path = Path(path)
        joblib.dump(self.estimator, path / "model.joblib")

    def load_model(self, path: Path) -> None:
        """使用 joblib 加载模型"""
        import joblib
        path = Path(path)
        self.estimator = joblib.load(path / "model.joblib")

    def get_feature_importance(self) -> dict[str, float]:
        """
        返回因子重要性

        线性模型的重要性 = |coef_|（系数绝对值）
        """
        if hasattr(self.estimator, "coef_"):
            coef = np.abs(self.estimator.coef_).flatten()
            if len(coef) == len(self.feature_names_):
                return dict(zip(self.feature_names_, coef.tolist()))
        return {}

    def get_params(self) -> dict:
        """返回模型参数"""
        return self.estimator.get_params()

    def __repr__(self) -> str:
        return f"SklearnModelWrapper({self.estimator!r})"
