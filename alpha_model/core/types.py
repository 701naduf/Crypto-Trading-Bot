"""
核心协议与类型定义

包含:
    AlphaModel Protocol  — 框架对模型的唯一要求
    TrainMode            — 训练粒度枚举
    WalkForwardMode      — Walk-Forward 切分模式枚举
    TrainConfig          — 训练配置
    PortfolioConstraints — 组合约束配置
    ModelMeta            — 模型/策略元数据

设计理由:
    使用 Protocol 而非 ABC，是因为 sklearn 原生模型已经有 fit/predict 方法，
    无需继承即可直接使用。用户也可以完全从零手写一个类，只要实现了协议方法。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# AlphaModel Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class AlphaModel(Protocol):
    """
    模型协议 — 框架对模型的唯一要求

    任何实现了 fit/predict 的对象都可以接入 Walk-Forward 训练框架，
    包括 sklearn 原生模型、LightGBM、PyTorch、用户手写脚本等。

    必须实现:
        fit(X, y, **kwargs)  — 训练模型
        predict(X)           — 生成预测值

    可选实现（用于持久化和分析）:
        save_model(path)     — 保存模型到指定目录
        load_model(path)     — 从指定目录加载模型
        get_feature_importance()  — 返回因子重要性
        get_params()         — 返回模型参数

    示例:
        # 直接用 sklearn 原生
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)  # 天然满足 AlphaModel 协议

        # 手写模型
        class MyModel:
            def fit(self, X, y, **kwargs): ...
            def predict(self, X): ...

        # PyTorch 封装
        class MyLSTM:
            def fit(self, X, y, **kwargs):
                # 内部自行处理: val拆分, DataLoader, epoch循环, early stopping, GPU
                ...
            def predict(self, X):
                # 内部自行处理: eval mode, no_grad, Tensor→numpy
                ...
            def save_model(self, path):
                torch.save(self.net.state_dict(), path / "model.pt")
            def load_model(self, path):
                self.net.load_state_dict(torch.load(path / "model.pt"))
    """

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """
        训练模型

        Args:
            X: 特征矩阵
                Pooled 模式: index=MultiIndex(timestamp, symbol), columns=factor_names
                Per-Symbol 模式: index=timestamp, columns=factor_names
            y: 目标变量（forward return）
                index 同 X
            **kwargs: 额外参数（框架保留，目前不传递）

        注意:
            - 如果模型需要验证集（如 DL 的 early stopping），应在此方法内部
              自行从 X/y 中拆分，框架不负责验证集划分
            - 框架保证传入的 X/y 中无未来信息（purge gap 已处理）
        """
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray | pd.Series:
        """
        生成预测值

        Args:
            X: 特征矩阵（格式同 fit 的 X）

        Returns:
            预测值数组，长度 = len(X)
            可以是 np.ndarray 或 pd.Series
        """
        ...


# ---------------------------------------------------------------------------
# 枚举类型
# ---------------------------------------------------------------------------

class TrainMode(Enum):
    """训练粒度"""
    POOLED = "pooled"           # 所有标的堆叠为一个训练集，训练一个模型
    PER_SYMBOL = "per_symbol"   # 每个标的独立训练一个模型


class WalkForwardMode(Enum):
    """Walk-Forward 切分模式"""
    EXPANDING = "expanding"     # 训练窗口起点固定，终点随 fold 前进（数据越来越多）
    ROLLING = "rolling"         # 训练窗口固定大小，整体向前滑动


# ---------------------------------------------------------------------------
# 配置数据类
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """
    训练配置

    控制 Walk-Forward 训练的行为：训练粒度、切分模式、窗口大小。

    Attributes:
        train_mode:     训练粒度（Pooled 或 Per-Symbol）
        wf_mode:        Walk-Forward 切分模式（Expanding 或 Rolling）
        target_horizon: 预测目标的前瞻窗口（bar 数）。
                        用于计算 forward return 作为训练标签。
                        purge_periods 必须 ≥ 此值。
        train_periods:  训练窗口长度（bar 数）。
                        Expanding 模式下为初始训练窗口，之后逐渐增大。
                        Rolling 模式下为固定窗口大小。
        test_periods:   测试窗口长度（bar 数）。
                        每个 fold 的样本外预测区间。
        purge_periods:  隔离期（bar 数），必须 ≥ target_horizon。
                        防止训练集末尾的 forward return 标签与测试集重叠。
                        实际 embargo = max(purge_periods, max_factor_lookback)，
                        由 splitter 在运行时确定。
    """
    train_mode: TrainMode = TrainMode.POOLED
    wf_mode: WalkForwardMode = WalkForwardMode.EXPANDING
    target_horizon: int = 10
    train_periods: int = 5000
    test_periods: int = 1000
    purge_periods: int = 60

    def __post_init__(self):
        if self.target_horizon < 1:
            raise ValueError(
                f"target_horizon 必须 >= 1, 收到 {self.target_horizon}"
            )
        if self.train_periods < 1:
            raise ValueError(
                f"train_periods 必须 >= 1, 收到 {self.train_periods}"
            )
        if self.test_periods < 1:
            raise ValueError(
                f"test_periods 必须 >= 1, 收到 {self.test_periods}"
            )
        if self.purge_periods < self.target_horizon:
            raise ValueError(
                f"purge_periods ({self.purge_periods}) 必须 >= "
                f"target_horizon ({self.target_horizon})"
            )


@dataclass
class PortfolioConstraints:
    """
    组合约束配置

    三层约束（可选组合启用）：
    1. 仓位上限：|w_i| ≤ max_weight，防止单标的过度集中
    2. Dollar-neutral：Σ(w+) = Σ|w-|，多空等金额
    3. Beta-neutral：Σ(w_i × beta_i) = 0，对市场方向中性

    波动率目标：
        动态缩放总仓位，使组合预期年化波动率 = vol_target

    Attributes:
        max_weight:     单标的最大绝对权重（如 0.4 = 最多 40%）
        dollar_neutral: 是否启用 dollar-neutral 约束
        beta_neutral:   是否启用 beta-neutral 约束
        beta_lookback:  滚动 beta 估计窗口（天数）
        vol_target:     年化波动率目标（如 0.15 = 15%），None 则不启用
        vol_lookback:   波动率估计窗口（天数）
        leverage_cap:   最大杠杆倍数（Σ|w_i| 的上限）
        risk_aversion:  风险厌恶系数 λ（控制收益-风险权衡）
        turnover_penalty: 换手率惩罚系数 γ（抑制频繁交易）
    """
    max_weight: float = 0.4
    dollar_neutral: bool = True
    beta_neutral: bool = False
    beta_lookback: int = 60
    vol_target: float | None = None
    vol_lookback: int = 60
    leverage_cap: float = 2.0
    risk_aversion: float = 1.0
    turnover_penalty: float = 0.01

    def __post_init__(self):
        if self.max_weight <= 0 or self.max_weight > 1.0:
            raise ValueError(
                f"max_weight 必须在 (0, 1.0] 范围内, 收到 {self.max_weight}"
            )
        if self.leverage_cap <= 0:
            raise ValueError(
                f"leverage_cap 必须 > 0, 收到 {self.leverage_cap}"
            )
        if self.vol_target is not None and self.vol_target <= 0:
            raise ValueError(
                f"vol_target 必须 > 0 或 None, 收到 {self.vol_target}"
            )
        if self.risk_aversion < 0:
            raise ValueError(
                f"risk_aversion 必须 >= 0, 收到 {self.risk_aversion}"
            )
        if self.turnover_penalty < 0:
            raise ValueError(
                f"turnover_penalty 必须 >= 0, 收到 {self.turnover_penalty}"
            )


@dataclass
class ModelMeta:
    """
    模型/策略元数据

    记录策略的完整配置，确保可复现性。

    Attributes:
        name:           策略名称（如 "ridge_momentum_v1"）
        factor_names:   使用的因子列表
        target_horizon: 预测目标前瞻窗口
        train_config:   训练配置
        constraints:    组合约束配置
        created_at:     创建时间
        version:        版本号
        description:    策略描述
    """
    name: str
    factor_names: list[str]
    target_horizon: int
    train_config: TrainConfig
    constraints: PortfolioConstraints
    created_at: datetime = None
    version: str = "1.0"
    description: str = ""

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(tz=None)

    def to_dict(self) -> dict:
        """序列化为可 JSON 化的字典"""
        return {
            "name": self.name,
            "factor_names": self.factor_names,
            "target_horizon": self.target_horizon,
            "train_config": {
                "train_mode": self.train_config.train_mode.value,
                "wf_mode": self.train_config.wf_mode.value,
                "target_horizon": self.train_config.target_horizon,
                "train_periods": self.train_config.train_periods,
                "test_periods": self.train_config.test_periods,
                "purge_periods": self.train_config.purge_periods,
            },
            "constraints": {
                "max_weight": self.constraints.max_weight,
                "dollar_neutral": self.constraints.dollar_neutral,
                "beta_neutral": self.constraints.beta_neutral,
                "beta_lookback": self.constraints.beta_lookback,
                "vol_target": self.constraints.vol_target,
                "vol_lookback": self.constraints.vol_lookback,
                "leverage_cap": self.constraints.leverage_cap,
                "risk_aversion": self.constraints.risk_aversion,
                "turnover_penalty": self.constraints.turnover_penalty,
            },
            "created_at": self.created_at.isoformat(),
            "version": self.version,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModelMeta":
        """从字典反序列化"""
        tc = d["train_config"]
        train_config = TrainConfig(
            train_mode=TrainMode(tc["train_mode"]),
            wf_mode=WalkForwardMode(tc["wf_mode"]),
            target_horizon=tc["target_horizon"],
            train_periods=tc["train_periods"],
            test_periods=tc["test_periods"],
            purge_periods=tc["purge_periods"],
        )
        pc = d["constraints"]
        constraints = PortfolioConstraints(
            max_weight=pc["max_weight"],
            dollar_neutral=pc["dollar_neutral"],
            beta_neutral=pc["beta_neutral"],
            beta_lookback=pc["beta_lookback"],
            vol_target=pc.get("vol_target"),
            vol_lookback=pc["vol_lookback"],
            leverage_cap=pc["leverage_cap"],
            risk_aversion=pc.get("risk_aversion", 1.0),
            turnover_penalty=pc.get("turnover_penalty", 0.01),
        )
        return cls(
            name=d["name"],
            factor_names=d["factor_names"],
            target_horizon=d["target_horizon"],
            train_config=train_config,
            constraints=constraints,
            created_at=datetime.fromisoformat(d["created_at"]),
            version=d.get("version", "1.0"),
            description=d.get("description", ""),
        )