"""
尾部特征分析模块

评价体系的第二层 API——尾部分析。

核心思想:
    因子值极端时（|factor| > threshold），信号的质量如何？
    极端信号往往是最有价值的交易机会，但也可能是噪声。

分析维度:
    - 条件 IC: 极端信号时的 IC（比整体 IC 更高则说明尾部信号更有价值）
    - 尾部命中率: 极端信号方向正确的比例
    - 尾部期望收益: 极端信号的平均收益
    - 最大逆向偏移 (MAE): 进场后的最大浮亏（衡量止损风险）
    - 极端信号频率: 多久出一次极端信号

依赖: evaluation.metrics
"""

import numpy as np
import pandas as pd

from ..config import DEFAULT_TAIL_THRESHOLD, MIN_IC_OBSERVATIONS, MIN_REGRESSION_OBSERVATIONS
from .metrics import compute_forward_returns_panel, spearman_ic


def tail_analysis(
    factor_panel: pd.DataFrame,
    price_panel: pd.DataFrame,
    threshold: float = DEFAULT_TAIL_THRESHOLD,
    horizon: int = 1,
) -> dict:
    """
    尾部特征分析（第二层 API 入口）

    Args:
        factor_panel: 因子面板 (index=timestamp, columns=symbols)
        price_panel:  价格面板，格式同上
        threshold:    尾部阈值分位数（0-1），默认 0.9
                      表示取因子值绝对值最大的 top 10% 和 bottom 10%
        horizon:      前瞻窗口（bar 数）

    Returns:
        dict: {
            "conditional_ic":      极端信号时的 IC,
            "tail_hit_rate":       极端信号方向正确的比例,
            "tail_expected_return": 极端信号的平均收益,
            "tail_frequency":      极端信号的频率（占比）,
            "long_tail_return":    因子极正时的平均收益,
            "short_tail_return":   因子极负时的平均收益,
            "mae":                 平均最大逆向偏移（进场后最大浮亏，通常 ≤ 0）,
            "n_tail_observations": 尾部观测数,
        }
    """
    forward_ret = compute_forward_returns_panel(price_panel, horizon)

    # 对齐
    common_idx = factor_panel.index.intersection(forward_ret.index)
    common_cols = factor_panel.columns.intersection(forward_ret.columns)
    f = factor_panel.loc[common_idx, common_cols]
    r = forward_ret.loc[common_idx, common_cols]

    # 展平为一维（所有 timestamp × symbol 的观测）
    f_flat = f.stack().dropna()
    r_flat = r.stack().dropna()

    # 对齐展平后的 index
    common_flat_idx = f_flat.index.intersection(r_flat.index)
    f_flat = f_flat.loc[common_flat_idx]
    r_flat = r_flat.loc[common_flat_idx]

    if len(f_flat) < MIN_REGRESSION_OBSERVATIONS:
        return _empty_tail_result()

    # --- 确定尾部阈值 ---
    abs_factor = f_flat.abs()
    upper_threshold = abs_factor.quantile(threshold)

    # 尾部观测: |factor| 超过分位数阈值
    tail_mask = abs_factor >= upper_threshold
    n_tail = tail_mask.sum()

    if n_tail < MIN_IC_OBSERVATIONS:
        return _empty_tail_result()

    f_tail = f_flat[tail_mask]
    r_tail = r_flat[tail_mask]

    # --- 条件 IC ---
    conditional_ic = spearman_ic(f_tail, r_tail)

    # --- 尾部命中率: sign(factor) == sign(return) 的比例 ---
    correct = (np.sign(f_tail) == np.sign(r_tail))
    hit_rate = correct.mean()

    # --- 尾部期望收益（按因子方向调整: factor>0 取 return, factor<0 取 -return）---
    direction_adjusted_return = r_tail * np.sign(f_tail)
    tail_expected_return = direction_adjusted_return.mean()

    # --- 极端信号频率 ---
    tail_frequency = n_tail / len(f_flat)

    # --- 分方向分析 ---
    long_mask = f_tail > 0
    short_mask = f_tail < 0

    long_tail_return = r_tail[long_mask].mean() if long_mask.any() else np.nan
    short_tail_return = r_tail[short_mask].mean() if short_mask.any() else np.nan

    # --- MAE: 最大逆向偏移 ---
    mae = _compute_mae(price_panel, f_tail, horizon)

    return {
        "conditional_ic": float(conditional_ic),
        "tail_hit_rate": float(hit_rate),
        "tail_expected_return": float(tail_expected_return),
        "tail_frequency": float(tail_frequency),
        "long_tail_return": float(long_tail_return),
        "short_tail_return": float(short_tail_return),
        "mae": float(mae),
        "n_tail_observations": int(n_tail),
    }


def _compute_mae(
    price_panel: pd.DataFrame,
    f_tail: pd.Series,
    horizon: int,
) -> float:
    """
    计算尾部信号的平均最大逆向偏移 (Maximum Adverse Excursion)

    对每个尾部信号时刻 t、标的 s:
        1. 取 price[t : t+horizon] 的逐 bar 累计收益
        2. 根据因子方向调整符号（做多取原始, 做空取反）
        3. MAE_t = min(direction_adjusted_cumulative_return)

    整体 MAE = mean(MAE_t)

    Args:
        price_panel: 价格面板 (index=timestamp, columns=symbols)
        f_tail:      尾部信号的因子值 (MultiIndex: timestamp × symbol)
        horizon:     前瞻窗口（bar 数）

    Returns:
        float: 平均 MAE（通常为负值，表示平均最大浮亏）
    """
    mae_values = []
    price_idx = price_panel.index

    for (ts, sym), factor_val in f_tail.items():
        if sym not in price_panel.columns:
            continue

        # 找到 ts 在价格面板中的位置
        try:
            loc = price_idx.get_loc(ts)
        except KeyError:
            continue

        # 取 horizon+1 个价格点（含进场价）
        end_loc = min(loc + horizon + 1, len(price_idx))
        if end_loc - loc < 2:
            continue

        prices = price_panel.iloc[loc:end_loc][sym].values
        entry_price = prices[0]

        if entry_price == 0 or np.isnan(entry_price):
            continue

        # 逐 bar 累计收益（从进场点开始）
        cum_returns = (prices[1:] - entry_price) / entry_price

        # 根据因子方向调整：做空信号取反
        direction = np.sign(factor_val)
        if direction == 0:
            continue
        adjusted_cum_returns = cum_returns * direction

        # MAE = 路径上的最小值（最大浮亏）
        mae_values.append(np.nanmin(adjusted_cum_returns))

    if not mae_values:
        return np.nan

    return float(np.mean(mae_values))


def _empty_tail_result() -> dict:
    """空数据时返回的默认结果"""
    return {
        "conditional_ic": np.nan,
        "tail_hit_rate": np.nan,
        "tail_expected_return": np.nan,
        "tail_frequency": np.nan,
        "long_tail_return": np.nan,
        "short_tail_return": np.nan,
        "mae": np.nan,
        "n_tail_observations": 0,
    }
