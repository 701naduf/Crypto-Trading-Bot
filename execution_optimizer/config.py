"""
execution_optimizer 配置类型

MarketContext 是对外的唯一数据接口，
由调用方（Phase 3 或 Phase 4）在每个时间步构造并传入。
"""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass
class MarketContext:
    """
    单个时间步的市场状态快照

    所有字段均为当前时刻的点估计值（非预测值），
    调用方负责保证不包含未来信息。
    """
    timestamp: pd.Timestamp
    symbols: list[str]

    spread: pd.Series
    """买卖价差，index=symbols，比率形式 (ask-bid)/mid。例：0.0002 = 2bps"""

    volatility: pd.Series
    """近期已实现波动率，index=symbols，日化标准差（非年化），20 日窗口"""

    adv: pd.Series
    """20 日平均日成交量，index=symbols，USD"""

    portfolio_value: float
    """当前组合总市值，USD"""

    funding_rate: pd.Series | None = None
    """
    各标的当前资金费率，index=symbols，单位：每 bar 费率。
    正值 = 多头付费（持多有成本），负值 = 多头收费（持多有收益）。

    调用方须按 bar 频率归一化：
        Phase 3 (1m bar): funding_rate_per_bar = rate_per_8h / 480
        Phase 3 (5m bar): funding_rate_per_bar = rate_per_8h / 96

    None 表示不考虑资金费率（现货策略或忽略时）。
    """


DEFAULT_IMPACT_COEFF: float = 0.1
"""sqrt-model 校准系数，经验范围 0.05~0.3"""

DEFAULT_FEE_RATE: float = 0.0004
"""Taker 手续费率 0.04%"""

DEFAULT_MAX_PARTICIPATION: float = 0.05
"""单步最大 ADV 参与率（5%），防止在低流动性时段过度冲击市场"""
