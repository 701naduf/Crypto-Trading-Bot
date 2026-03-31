"""
策略输出持久化

存储策略的目标权重、标准化信号、模型原始预测、元数据和绩效摘要。
这是 Phase 2b → Phase 3 / Phase 4 / execution_optimizer 的唯一持久化接口。

存储路径: db/signals/{strategy_name}/
    ├── weights.parquet          # 目标权重面板 (timestamp × symbol)，必须
    ├── signals.parquet          # 标准化信号面板 (post-generator, pre-cvxpy)，可选
    ├── raw_predictions.parquet  # 模型原始预测面板 (post-predict, pre-generator)，可选
    ├── meta.json                # ModelMeta 序列化
    └── performance.json         # BacktestResult.summary() 结果

写入策略: 原子写入（先写 .tmp 目录，全部完成后 rename），与 FactorStore 一致。

数据层定义:
    raw_predictions: model.predict() 的 OOS 原始输出，未经任何后处理
    signals:         经 SignalGenerator 标准化（z-score / rank 等）的截面信号
    weights:         经 PortfolioConstructor (cvxpy + γ) 优化后的目标权重

消费方与适用数据层:
    Phase 3 向量化回测       → weights（主要路径）
    Phase 3 事件驱动回测     → weights 或 execution_optimizer 输出权重
    execution_optimizer      → signals 或 raw_predictions（多资产 / 单资产场景）
    单资产择时策略           → raw_predictions（直接按得分阈值触发，无需组合优化）
    Phase 4 实盘执行         → execution_optimizer 输出权重
    模型监控 / 审计          → raw_predictions + signals（诊断用）
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
        raw_predictions: pd.DataFrame | None = None,
        meta: ModelMeta | None = None,
        performance: dict | None = None,
    ) -> None:
        """
        保存策略输出（原子写入）

        修改前后行为对比：
            修改前：save(name, weights, signals, meta, performance)  → 存 3 层
            修改后：save(name, weights, signals, raw_predictions, meta, performance) → 存 4 层
            不传 raw_predictions 时，行为与修改前完全一致

        Args:
            strategy_name:    策略名称（对应 db/signals/{strategy_name}/）
            weights:          目标权重面板 (timestamp × symbol)，必须提供
            signals:          标准化信号面板 (post-generator, pre-cvxpy)，可选
            raw_predictions:  模型原始预测面板 (post-predict, pre-generator)，可选
            meta:             策略元数据，可选
            performance:      绩效摘要字典，可选
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

            # 保存模型原始预测（可选）
            if raw_predictions is not None:
                self._save_parquet(raw_predictions, tmp_dir / "raw_predictions.parquet")

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
        """加载标准化信号面板 (post-generator, pre-cvxpy)"""
        path = self._strategy_dir(strategy_name) / "signals.parquet"
        if not path.exists():
            raise FileNotFoundError(f"策略 '{strategy_name}' 的信号文件不存在")
        return self._load_parquet(path)

    def load_raw_predictions(self, strategy_name: str) -> pd.DataFrame:
        """
        加载模型原始预测面板 (post-predict, pre-generator)

        与 load_signals() 的区别：
            load_signals()         → 标准化后的截面信号，适合组合优化输入
            load_raw_predictions() → 模型原始得分，适合单资产择时 / 模型诊断

        Args:
            strategy_name: 策略名称

        Returns:
            DataFrame，index=DatetimeIndex(UTC)，columns=symbols

        Raises:
            FileNotFoundError: 若保存时未传入 raw_predictions 参数
        """
        path = self._strategy_dir(strategy_name) / "raw_predictions.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"策略 '{strategy_name}' 的原始预测文件不存在。"
                f"请确认 AlphaPipeline.save() 时已正确传入 raw_predictions 参数。"
            )
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

    def has_signals(self, strategy_name: str) -> bool:
        """
        检查该策略是否存储了标准化信号文件。

        save() 中不传 signals 参数时，此方法返回 False。
        """
        return (self._strategy_dir(strategy_name) / "signals.parquet").exists()

    def has_raw_predictions(self, strategy_name: str) -> bool:
        """
        检查该策略是否存储了模型原始预测文件。

        save() 中不传 raw_predictions 参数时，此方法返回 False。
        """
        return (self._strategy_dir(strategy_name) / "raw_predictions.parquet").exists()

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
