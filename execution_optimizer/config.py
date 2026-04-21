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
    """
    近期已实现波动率，index=symbols，**日化**标准差（非年化），20 日窗口。

    推荐实现（1m bar 下）:
        returns_1m = price_1m.pct_change()
        vol_daily  = returns_1m.rolling(20*1440, min_periods=5000).std() * np.sqrt(1440)

    说明:
        - σ 是日化标准差，与 Almgren-Chriss impact 公式约定一致
          （文献里 σ 单位是 1/√day，impact 公式 σ√(q/ADV) 天然为日级）
        - 与 ExecutionOptimizer 内部协方差矩阵 Σ（基于 1m returns 的 1m 级方差）
          是不同对象：σ 喂给 impact 公式，Σ 喂给 quad_form(w, Σ)，不要混淆
        - min_periods=5000（约 3.5 天）是"样本量足以估计分钟级波动"的经验阈值；
          低于此值时 rolling std 估计误差过大，宁可返回 NaN 由上游处理
    """

    adv: pd.Series
    """
    20 日平均日成交量（USD），index=symbols。

    推荐实现（1m bar 下，NaN-safe）:
        notional_1m = close_1m * volume_1m
        window      = 20 * 1440
        # 用实际有效 bar 数除，防止维护窗口/停盘期低估 ADV
        rolling_sum = notional_1m.rolling(window, min_periods=5000).sum()
        rolling_cnt = notional_1m.rolling(window, min_periods=5000).count()
        adv         = rolling_sum / rolling_cnt * 1440   # 平均每分钟成交额 × 1440 分钟/日

    说明:
        - ADV 单位是 USD，喂入 impact 公式时与 V（portfolio_value, USD）相除得无量纲 √(V/ADV)
        - 不要用 rolling(window).sum()/20 的简化写法——若窗口内有 NaN，sum() 会当作 0 处理，
          导致 ADV 被系统性低估
    """

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
