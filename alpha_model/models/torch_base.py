"""
PyTorch 模型基础封装

用户继承此类，只需实现 build_network()。
fit/predict/val拆分/early stopping/GPU 全部自动处理。

注意: 这是参考实现。用户完全可以不用此基类，
      只要自己的类实现了 fit/predict 即可。

依赖: torch（可选）
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TorchModelBase:
    """
    PyTorch 模型基础封装

    用法:
        class MyMLP(TorchModelBase):
            def build_network(self, n_features):
                import torch.nn as nn
                return nn.Sequential(
                    nn.Linear(n_features, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                )

        model = MyMLP(val_ratio=0.2, patience=10, max_epochs=100)
    """

    def __init__(
        self,
        val_ratio: float = 0.2,
        patience: int = 10,
        max_epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str = "auto",
    ):
        """
        Args:
            val_ratio:  验证集比例（从训练集尾部按时间顺序切分）
            patience:   early stopping 容忍轮次
            max_epochs: 最大训练轮次
            batch_size: 批大小
            lr:         学习率
            device:     设备（"auto", "cpu", "cuda"）
        """
        self.val_ratio = val_ratio
        self.patience = patience
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device_str = device
        self.net = None
        self.n_features_ = 0
        self.feature_names_: list[str] = []

    def build_network(self, n_features: int):
        """
        用户实现：返回一个 nn.Module

        Args:
            n_features: 输入特征数量

        Returns:
            nn.Module 实例
        """
        raise NotImplementedError("子类必须实现 build_network()")

    def _get_device(self):
        """确定设备"""
        import torch
        if self.device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device_str)

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """
        完整训练流程:
        1. 拆分 train/val（按时间顺序，尾部 val_ratio 为验证集）
        2. DataFrame → Tensor → DataLoader
        3. 训练循环（epochs, batch, gradient）
        4. 验证集 early stopping
        5. 加载 best checkpoint
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        # 删除 NaN 行
        valid = ~(np.isnan(X_arr).any(axis=1) | np.isnan(y_arr.flatten()))
        X_arr = X_arr[valid]
        y_arr = y_arr[valid]

        self.n_features_ = X_arr.shape[1]
        device = self._get_device()

        # 构建网络
        self.net = self.build_network(self.n_features_).to(device)

        # 拆分 train/val（按时间顺序）
        n = len(X_arr)
        val_size = max(int(n * self.val_ratio), 1)
        X_train_t = torch.tensor(X_arr[:-val_size], device=device)
        y_train_t = torch.tensor(y_arr[:-val_size], device=device)
        X_val_t = torch.tensor(X_arr[-val_size:], device=device)
        y_val_t = torch.tensor(y_arr[-val_size:], device=device)

        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=self.batch_size, shuffle=True,
        )

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(self.max_epochs):
            # 训练
            self.net.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pred = self.net(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()

            # 验证
            self.net.eval()
            with torch.no_grad():
                val_pred = self.net(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    logger.debug(
                        "Early stopping at epoch %d, best val_loss=%.6f",
                        epoch, best_val_loss,
                    )
                    break

        # [R6] 记录实际训练 epoch 数（用于验证 early stopping）
        self.actual_epochs_ = epoch + 1

        # 加载 best checkpoint
        if best_state is not None:
            self.net.load_state_dict(best_state)
            self.net.to(device)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """eval mode + no_grad + Tensor → numpy"""
        import torch

        if self.net is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()")

        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        device = self._get_device()
        self.net.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_arr, device=device)
            pred = self.net(X_t).cpu().numpy().flatten()
        return pred

    def save_model(self, path: Path) -> None:
        """保存 state_dict + 网络配置"""
        import torch

        path = Path(path)
        if self.net is None:
            raise RuntimeError("模型尚未训练")

        torch.save(self.net.state_dict(), path / "model.pt")
        # 保存配置（用于重建网络）
        config = {
            "n_features": self.n_features_,
            "val_ratio": self.val_ratio,
            "patience": self.patience,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f)

    def load_model(self, path: Path) -> None:
        """加载 state_dict"""
        import torch

        path = Path(path)

        # 加载配置
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
            self.n_features_ = config["n_features"]

        # 重建网络并加载权重
        device = self._get_device()
        self.net = self.build_network(self.n_features_).to(device)
        self.net.load_state_dict(
            torch.load(path / "model.pt", map_location=device)
        )

    def get_feature_importance(self) -> dict[str, float]:
        """PyTorch 模型默认不提供因子重要性"""
        return {}

    def get_params(self) -> dict:
        return {
            "val_ratio": self.val_ratio,
            "patience": self.patience,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "device": self.device_str,
        }
