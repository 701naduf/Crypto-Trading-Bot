"""
策略输出持久化

存储策略的目标权重、原始信号、元数据和绩效摘要。
这是 Phase 2b → Phase 3 的唯一输出接口。

路径: db/signals/{strategy_name}/
    ├── weights.parquet        # 目标权重面板 (timestamp × symbol)
    ├── signals.parquet        # 原始信号面板（权重之前，用于调试）
    ├── meta.json              # ModelMeta 序列化
    └── performance.json       # BacktestResult.summary() 结果

写入策略: 原子写入（先写 .tmp 再 rename），与 FactorStore 一致。
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

import pandas as pd

from alpha_model.config import SIGNAL_STORE_DIR
from alpha_model.core.types import ModelMeta

logger = logging.getLogger(__name__)


class SignalStore:
    """策略输出持久化"""

    def __init__(self, base_dir: str | Path | None = None):
        """
        Args:
            base_dir: 存储根目录，默认 db/signals/
        """
        self.base_dir = Path(base_dir) if base_dir else SIGNAL_STORE_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _strategy_dir(self, strategy_name: str) -> Path:
        return self.base_dir / strategy_name

    def save(
        self,
        strategy_name: str,
        weights: pd.DataFrame,
        signals: pd.DataFrame | None = None,
        meta: ModelMeta | None = None,
        performance: dict | None = None,
    ) -> None:
        """
        保存策略输出（原子写入）

        Args:
            strategy_name: 策略名称
            weights:       目标权重面板 (timestamp × symbol)
            signals:       原始信号面板（可选）
            meta:          策略元数据（可选）
            performance:   绩效摘要字典（可选）
        """
        final_dir = self._strategy_dir(strategy_name)
        tmp_dir = final_dir.with_suffix(".tmp")

        # 清理可能残留的 tmp
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 保存权重（必须）
            self._save_parquet(weights, tmp_dir / "weights.parquet")

            # 保存信号（可选）
            if signals is not None:
                self._save_parquet(signals, tmp_dir / "signals.parquet")

            # 保存元数据（可选）
            if meta is not None:
                with open(tmp_dir / "meta.json", "w", encoding="utf-8") as f:
                    json.dump(meta.to_dict(), f, indent=2, ensure_ascii=False)

            # 保存绩效（可选）
            if performance is not None:
                # 将不可 JSON 序列化的值转为安全类型
                # float(nan)/float(inf) 会导致 json.dump 抛出 ValueError
                def _json_safe(v):
                    if isinstance(v, (int, float)):
                        import math
                        if math.isnan(v) or math.isinf(v):
                            return str(v)  # "nan", "inf", "-inf"
                        return float(v)
                    return str(v)

                perf_safe = {k: _json_safe(v) for k, v in performance.items()}
                with open(tmp_dir / "performance.json", "w", encoding="utf-8") as f:
                    json.dump(perf_safe, f, indent=2, ensure_ascii=False)

            # 原子替换
            if final_dir.exists():
                shutil.rmtree(final_dir)
            os.rename(str(tmp_dir), str(final_dir))

            logger.info("策略 '%s' 已保存到 %s", strategy_name, final_dir)

        except Exception:
            # 清理失败的 tmp
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            raise

    def load_weights(self, strategy_name: str) -> pd.DataFrame:
        """加载目标权重面板"""
        path = self._strategy_dir(strategy_name) / "weights.parquet"
        if not path.exists():
            raise FileNotFoundError(f"策略 '{strategy_name}' 的权重文件不存在")
        return self._load_parquet(path)

    def load_signals(self, strategy_name: str) -> pd.DataFrame:
        """加载原始信号面板"""
        path = self._strategy_dir(strategy_name) / "signals.parquet"
        if not path.exists():
            raise FileNotFoundError(f"策略 '{strategy_name}' 的信号文件不存在")
        return self._load_parquet(path)

    def load_meta(self, strategy_name: str) -> ModelMeta:
        """加载策略元数据"""
        path = self._strategy_dir(strategy_name) / "meta.json"
        if not path.exists():
            raise FileNotFoundError(f"策略 '{strategy_name}' 的元数据文件不存在")
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return ModelMeta.from_dict(d)

    def load_performance(self, strategy_name: str) -> dict:
        """加载绩效摘要"""
        path = self._strategy_dir(strategy_name) / "performance.json"
        if not path.exists():
            raise FileNotFoundError(f"策略 '{strategy_name}' 的绩效文件不存在")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_strategies(self) -> list[str]:
        """列出所有已存储的策略名"""
        if not self.base_dir.exists():
            return []
        return sorted([
            d.name for d in self.base_dir.iterdir()
            if d.is_dir() and not d.name.endswith(".tmp")
        ])

    def exists(self, strategy_name: str) -> bool:
        """检查策略是否存在"""
        return self._strategy_dir(strategy_name).exists()

    def delete(self, strategy_name: str) -> None:
        """删除策略数据"""
        d = self._strategy_dir(strategy_name)
        if d.exists():
            shutil.rmtree(d)
            logger.info("策略 '%s' 已删除", strategy_name)

    @staticmethod
    def _save_parquet(df: pd.DataFrame, path: Path) -> None:
        """保存 DataFrame 为 Parquet（保留 timestamp 列以兼容 FactorStore 风格）"""
        save_df = df.copy()
        save_df.insert(0, "timestamp", save_df.index)
        save_df.to_parquet(path, index=False)

    @staticmethod
    def _load_parquet(path: Path) -> pd.DataFrame:
        """加载 Parquet 并还原为 DatetimeIndex"""
        df = pd.read_parquet(path)
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
            df.index = pd.to_datetime(df.index, utc=True)
            df.index.name = None  # 与原始面板保持一致
        return df
