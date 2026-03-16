"""
训练好的模型持久化

存储模型对象、元数据和因子重要性。
用途: 实盘推理（加载模型做预测）、模型比对、超参调优、集成模型。

路径: db/models/{model_name}/
    ├── model/                 # 模型文件目录（由 AlphaModel.save_model 写入）
    │   ├── model.joblib       # sklearn 模型（参考实现的默认格式）
    │   └── ...                # 或 model.pt（PyTorch），或其他格式
    ├── meta.json              # ModelMeta + TrainConfig 序列化
    └── importance.json        # 因子重要性排名（可选）

设计要点:
    ModelStore 不假设模型的序列化格式。
    它调用 model.save_model(path) 和 model.load_model(path)，
    由模型自行决定如何保存/加载。
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Callable

from alpha_model.config import MODEL_STORE_DIR
from alpha_model.core.types import AlphaModel, ModelMeta

logger = logging.getLogger(__name__)


class ModelStore:
    """模型持久化"""

    def __init__(self, base_dir: str | Path | None = None):
        """
        Args:
            base_dir: 存储根目录，默认 db/models/
        """
        self.base_dir = Path(base_dir) if base_dir else MODEL_STORE_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _model_dir(self, model_name: str) -> Path:
        return self.base_dir / model_name

    def save(
        self,
        model_name: str,
        model: AlphaModel,
        meta: ModelMeta,
        importance: dict[str, float] | None = None,
    ) -> None:
        """
        保存模型

        1. 创建 model/ 子目录
        2. 调用 model.save_model(model_dir)
        3. 保存 meta.json 和 importance.json

        Args:
            model_name:  模型名称
            model:       满足 AlphaModel 协议的模型对象
            meta:        模型元数据
            importance:  因子重要性字典（可选）
        """
        final_dir = self._model_dir(model_name)
        tmp_dir = final_dir.with_suffix(".tmp")

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 保存模型对象
            model_subdir = tmp_dir / "model"
            model_subdir.mkdir(exist_ok=True)
            if hasattr(model, "save_model"):
                model.save_model(model_subdir)
            else:
                logger.warning(
                    "模型未实现 save_model 方法，跳过模型文件保存"
                )

            # 保存元数据
            with open(tmp_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta.to_dict(), f, indent=2, ensure_ascii=False)

            # 保存因子重要性
            if importance is not None:
                # 转为 float 以确保 JSON 序列化
                importance_safe = {
                    k: float(v) for k, v in importance.items()
                }
                with open(tmp_dir / "importance.json", "w", encoding="utf-8") as f:
                    json.dump(importance_safe, f, indent=2, ensure_ascii=False)

            # 原子替换
            if final_dir.exists():
                shutil.rmtree(final_dir)
            os.rename(str(tmp_dir), str(final_dir))

            logger.info("模型 '%s' 已保存到 %s", model_name, final_dir)

        except Exception:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            raise

    def load(
        self,
        model_name: str,
        model_factory: Callable[[], AlphaModel],
    ) -> tuple[AlphaModel, ModelMeta]:
        """
        加载模型

        Args:
            model_name:    模型名称
            model_factory: 返回空模型实例的工厂函数
                           例如 lambda: Ridge(alpha=1.0)

        Returns:
            (model, meta) 元组

        Raises:
            FileNotFoundError: 模型不存在
        """
        model_dir = self._model_dir(model_name)
        if not model_dir.exists():
            raise FileNotFoundError(f"模型 '{model_name}' 不存在")

        # 创建模型实例
        model = model_factory()

        # 加载模型权重
        model_subdir = model_dir / "model"
        if model_subdir.exists() and hasattr(model, "load_model"):
            model.load_model(model_subdir)

        # 加载元数据
        meta_path = model_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"模型 '{model_name}' 缺少 meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = ModelMeta.from_dict(json.load(f))

        return model, meta

    def load_importance(self, model_name: str) -> dict[str, float]:
        """加载因子重要性"""
        path = self._model_dir(model_name) / "importance.json"
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_models(self) -> list[str]:
        """列出所有已存储的模型名"""
        if not self.base_dir.exists():
            return []
        return sorted([
            d.name for d in self.base_dir.iterdir()
            if d.is_dir() and not d.name.endswith(".tmp")
        ])

    def exists(self, model_name: str) -> bool:
        """检查模型是否存在"""
        return self._model_dir(model_name).exists()

    def delete(self, model_name: str) -> None:
        """删除模型数据"""
        d = self._model_dir(model_name)
        if d.exists():
            shutil.rmtree(d)
            logger.info("模型 '%s' 已删除", model_name)
