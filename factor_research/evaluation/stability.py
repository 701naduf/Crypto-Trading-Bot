"""
稳定性分析模块

评价体系的第二层 API——因子稳定性分析。

一个好因子不仅整体 IC 高，还需要在不同市场环境下都保持稳定。
如果一个因子只在特定 regime 下有效，使用价值会大打折扣。

分析维度:
    1. 分 regime IC: 按趋势/震荡、高波/低波分别计算 IC
    2. 月度 IC 分解: 按月份分别计算 IC，检查是否有系统性失效月份
    3. 滚动 IC: IC 的时间演变趋势
    4. IC 最大回撤: IC 累计曲线的最大回撤

依赖: evaluation.metrics, evaluation.ic
"""

import numpy as np
import pandas as pd

from ..config import DEFAULT_ROLLING_WINDOW
from .ic import ic_series, ic_summary
from .metrics import max_drawdown


def stability_analysis(
    factor_panel: pd.DataFrame,
    price_panel: pd.DataFrame,
    horizon: int = 1,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
) -> dict:
    """
    因子稳定性分析（第二层 API 入口）

    Args:
        factor_panel:  因子面板
        price_panel:   价格面板
        horizon:       前瞻窗口（bar 数）
        rolling_window: 滚动 IC 的窗口大小（bar 数），默认 60

    Returns:
        dict: {
            "regime_ic":       分 regime 的 IC 统计,
            "monthly_ic":      月度 IC 分解,
            "rolling_ic":      滚动 IC 序列,
            "ic_max_drawdown": IC 累计曲线的最大回撤,
        }
    """
    ic_ts = ic_series(factor_panel, price_panel, horizon=horizon)

    return {
        "regime_ic": _regime_ic(factor_panel, price_panel, horizon),
        "monthly_ic": _monthly_ic(ic_ts),
        "rolling_ic": _rolling_ic(ic_ts, window=rolling_window),
        "ic_max_drawdown": _ic_max_drawdown(ic_ts),
    }


def _regime_ic(
    factor_panel: pd.DataFrame,
    price_panel: pd.DataFrame,
    horizon: int,
) -> dict:
    """
    分 regime 计算 IC

    Regime 划分:
        - 趋势/震荡: 基于收益率的滚动均值
        - 高波/低波: 基于收益率的滚动标准差

    使用中位数作为分界线（简单但稳健）。
    """
    # 计算整体 IC 序列
    ic_ts = ic_series(factor_panel, price_panel, horizon=horizon)

    if ic_ts.dropna().empty:
        return {"trend": {}, "vol": {}}

    # 用第一个 symbol 的收益率作为市场代理
    # （在 crypto 中 BTC 通常是最好的市场代理）
    first_col = price_panel.columns[0]
    market_ret = price_panel[first_col].pct_change()

    # 对齐到 IC 的 index
    common_idx = ic_ts.index.intersection(market_ret.index)
    ic_aligned = ic_ts.loc[common_idx]
    ret_aligned = market_ret.loc[common_idx]

    # --- 趋势/震荡 regime: 基于滚动均值 ---
    rolling_mean = ret_aligned.rolling(60, min_periods=10).mean()
    trend_threshold = rolling_mean.median()

    trend_up = ic_aligned[rolling_mean > trend_threshold].dropna()
    trend_down = ic_aligned[rolling_mean <= trend_threshold].dropna()

    # --- 高波/低波 regime: 基于滚动标准差 ---
    rolling_vol = ret_aligned.rolling(60, min_periods=10).std()
    vol_threshold = rolling_vol.median()

    high_vol = ic_aligned[rolling_vol > vol_threshold].dropna()
    low_vol = ic_aligned[rolling_vol <= vol_threshold].dropna()

    return {
        "trend": {
            "uptrend": ic_summary(trend_up),
            "downtrend": ic_summary(trend_down),
        },
        "vol": {
            "high_vol": ic_summary(high_vol),
            "low_vol": ic_summary(low_vol),
        },
    }


def _monthly_ic(ic_ts: pd.Series) -> pd.DataFrame:
    """
    月度 IC 分解

    按月份分组计算 IC 均值和标准差，
    检查是否有系统性失效的月份。

    Returns:
        pd.DataFrame: index=year-month, columns=[ic_mean, ic_std, n_obs]
    """
    valid = ic_ts.dropna()
    if valid.empty:
        return pd.DataFrame(columns=["ic_mean", "ic_std", "n_obs"])

    monthly = valid.groupby(valid.index.to_period("M"))
    result = pd.DataFrame({
        "ic_mean": monthly.mean(),
        "ic_std": monthly.std(),
        "n_obs": monthly.count(),
    })
    return result


def _rolling_ic(ic_ts: pd.Series, window: int = 60) -> pd.Series:
    """
    滚动 IC

    计算 IC 的滚动均值，观察 IC 的时间演变趋势。
    如果 IC 持续下降，说明因子可能在衰减。

    Args:
        ic_ts:  IC 时间序列
        window: 滚动窗口大小

    Returns:
        pd.Series: 滚动 IC 均值
    """
    return ic_ts.rolling(window, min_periods=max(1, window // 4)).mean()


def _ic_max_drawdown(ic_ts: pd.Series) -> float:
    """
    IC 最大回撤

    将 IC 序列视为"收益"，计算其累计曲线的最大回撤。
    衡量 IC 最差的连续时期有多严重。

    Returns:
        float: IC 最大回撤（正数）
    """
    valid = ic_ts.dropna()
    if len(valid) < 2:
        return np.nan

    # 用 IC 的累计和来模拟"IC 曲线"
    cum_ic = valid.cumsum()
    running_max = cum_ic.cummax()
    drawdown = running_max - cum_ic

    if running_max.max() == 0:
        return 0.0

    # 归一化回撤
    return float(drawdown.max())
