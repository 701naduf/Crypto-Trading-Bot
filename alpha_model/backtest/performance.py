"""
绩效指标汇总

复用 Phase 2a:
    factor_research.evaluation.metrics.sharpe_ratio
    factor_research.evaluation.metrics.max_drawdown
    factor_research.evaluation.metrics.annualize_return
    factor_research.evaluation.metrics.annualize_volatility
    factor_research.evaluation.metrics.cumulative_returns

新增策略级指标:
    Calmar ratio, Sortino ratio, 平均换手率, 最长回撤持续期, 胜率
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from factor_research.evaluation.metrics import (
    cumulative_returns,
    annualize_return,
    annualize_volatility,
    sharpe_ratio,
    max_drawdown,
)

from alpha_model.config import MINUTES_PER_YEAR


# ---------------------------------------------------------------------------
# 新增绩效指标
# ---------------------------------------------------------------------------

def sortino_ratio(
    returns: pd.Series,
    periods_per_year: float = MINUTES_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Sortino ratio（下行波动率版本）

    Sortino = annualized_return / annualized_downside_deviation

    只考虑负收益的波动率，不惩罚正的波动。

    Args:
        returns:          逐期收益率序列
        periods_per_year: 年化系数
        risk_free_rate:   无风险利率（年化）

    Returns:
        Sortino ratio
    """
    excess = returns - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return np.inf if returns.mean() > 0 else np.nan
    downside_std = downside.std() * np.sqrt(periods_per_year)
    ann_ret = annualize_return(returns, periods_per_year)
    return (ann_ret - risk_free_rate) / downside_std


def calmar_ratio(
    returns: pd.Series,
    periods_per_year: float = MINUTES_PER_YEAR,
) -> float:
    """
    Calmar ratio = 年化收益 / |最大回撤|

    Args:
        returns:          逐期收益率序列
        periods_per_year: 年化系数

    Returns:
        Calmar ratio
    """
    ann_ret = annualize_return(returns, periods_per_year)
    cum_ret = cumulative_returns(returns)
    mdd = max_drawdown(cum_ret)  # 负数
    if mdd == 0:
        return np.inf if ann_ret > 0 else np.nan
    return ann_ret / abs(mdd)


def max_drawdown_duration(returns: pd.Series) -> int:
    """
    最长回撤持续期（bar 数）

    从净值新高到恢复（或未恢复到序列末尾）的最长持续时间。

    Args:
        returns: 逐期收益率序列

    Returns:
        最长回撤持续期（bar 数）
    """
    cum = cumulative_returns(returns)
    running_max = cum.cummax()
    in_drawdown = cum < running_max

    if not in_drawdown.any():
        return 0

    # 计算每段连续回撤的长度
    drawdown_groups = (~in_drawdown).cumsum()
    # 只看回撤中的组
    dd_lengths = in_drawdown.groupby(drawdown_groups).sum()
    return int(dd_lengths.max())


# ---------------------------------------------------------------------------
# BacktestResult 数据类
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """
    回测结果

    包含策略的完整绩效数据，提供 summary() 方法生成摘要。

    Attributes:
        equity_curve:    净值曲线 (1.0 起始)
        returns:         逐期收益率
        turnover:        逐期换手率
        weights_history: 权重历史 (timestamp × symbol)
        gross_returns:   毛收益（交易成本之前），可选
        total_cost:      总交易成本，可选
    """
    equity_curve: pd.Series         # 净值曲线 (1.0 起始)
    returns: pd.Series              # 逐期净收益率
    turnover: pd.Series             # 逐期换手率
    weights_history: pd.DataFrame   # 权重历史 (timestamp × symbol)
    gross_returns: pd.Series | None = None   # 毛收益
    total_cost: float = 0.0                  # 总交易成本

    def summary(self) -> dict:
        """
        生成绩效摘要

        Returns:
            绩效指标字典
        """
        returns = self.returns.dropna()

        if len(returns) == 0:
            return {"error": "无有效收益率数据"}

        cum = cumulative_returns(returns)

        return {
            "annual_return": annualize_return(returns, MINUTES_PER_YEAR),
            "annual_volatility": annualize_volatility(returns, MINUTES_PER_YEAR),
            "sharpe_ratio": sharpe_ratio(returns, MINUTES_PER_YEAR),
            "sortino_ratio": sortino_ratio(returns, MINUTES_PER_YEAR),
            "calmar_ratio": calmar_ratio(returns, MINUTES_PER_YEAR),
            "max_drawdown": max_drawdown(cum),
            "max_drawdown_duration": max_drawdown_duration(returns),
            "avg_turnover": self.turnover.mean() if len(self.turnover) > 0 else 0.0,
            "total_cost": self.total_cost,
            "win_rate": (returns > 0).mean(),
            "n_periods": len(returns),
            "total_return": cum.iloc[-1] - 1.0 if len(cum) > 0 else 0.0,
        }
