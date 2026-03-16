"""
分层回测模块（Quantile Backtest）

评价体系的第二层 API——截面分层分析。

核心思想:
    1. 在每个时刻 t，按因子值将所有标的排序后分为 N 组
    2. 计算每组在 t+h 的平均收益
    3. 分析组间收益的单调性和多空价差

如果因子有效:
    - 因子值最高组和最低组的收益应有显著差异
    - 各组收益应呈现单调递增或递减趋势
    - 多空组合（做多顶组、做空底组）应有正收益

注意:
    本项目 5 标的场景下，分 5 组意味着每组 1 个标的。
    统计意义有限，但能直观看到因子的排序能力。
    框架完整支持更多标的的场景。

依赖: evaluation.metrics
"""

import numpy as np
import pandas as pd

from ..config import DEFAULT_N_GROUPS
from .metrics import (
    annualize_return,
    annualize_volatility,
    compute_forward_returns_panel,
    cumulative_returns,
    sharpe_ratio,
)


def quantile_backtest(
    factor_panel: pd.DataFrame,
    price_panel: pd.DataFrame,
    n_groups: int = DEFAULT_N_GROUPS,
    horizon: int = 1,
    grouping: str = "equal_freq",
) -> dict:
    """
    分层回测（第二层 API 入口）

    Args:
        factor_panel: 因子面板 (index=timestamp, columns=symbols)
        price_panel:  价格面板（收盘价），格式同上
        n_groups:     分层组数，默认 5
        horizon:      前瞻窗口（bar 数）
        grouping:     分组方式:
                      - "equal_freq": 等频分组（按排名均分，每组样本量相同）
                        适用于大多数场景，确保每组有足够样本进行统计。
                      - "equal_width": 等距分组（按因子值区间均分）
                        反映因子值的真实分布特征，能暴露因子值集中区域。
                        当因子值分布严重偏斜时，某些组可能样本极少。

    Returns:
        dict: {
            "group_returns":     各组平均年化收益,
            "group_sharpe":      各组夏普比率,
            "long_short_return": 多空年化收益,
            "long_short_sharpe": 多空夏普比率,
            "monotonicity":      单调性 (Spearman corr of group rank vs return),
            "cumulative_by_group": 各组累计收益曲线 DataFrame,
            "long_short_cumulative": 多空累计收益曲线 Series,
        }
    """
    if grouping not in ("equal_freq", "equal_width"):
        raise ValueError(
            f"grouping 必须为 'equal_freq' 或 'equal_width'，收到 '{grouping}'"
        )
    forward_ret = compute_forward_returns_panel(price_panel, horizon)

    # 对齐
    common_idx = factor_panel.index.intersection(forward_ret.index)
    common_cols = factor_panel.columns.intersection(forward_ret.columns)
    f = factor_panel.loc[common_idx, common_cols]
    r = forward_ret.loc[common_idx, common_cols]

    n_symbols = len(common_cols)
    if n_symbols < n_groups:
        n_groups = n_symbols

    # --- 逐期分组并计算组平均收益 ---
    group_returns = {g: [] for g in range(1, n_groups + 1)}

    for ts in common_idx:
        factor_row = f.loc[ts].dropna()
        return_row = r.loc[ts].dropna()

        valid_symbols = factor_row.index.intersection(return_row.index)
        if len(valid_symbols) < n_groups:
            continue

        fv = factor_row[valid_symbols]
        rv = return_row[valid_symbols]

        # 分组: equal_freq 按排名均分，equal_width 按因子值区间均分
        if grouping == "equal_freq":
            # 等频分组: 按排名均分，每组样本量相同
            ranked = fv.rank(method="first")
            group_labels = pd.cut(ranked, bins=n_groups, labels=False) + 1
        else:
            # 等距分组: 按因子值区间均分，反映真实分布
            group_labels = pd.cut(fv, bins=n_groups, labels=False) + 1

        for g in range(1, n_groups + 1):
            mask = group_labels == g
            if mask.any():
                group_returns[g].append(rv[mask].mean())
            else:
                group_returns[g].append(np.nan)

    # --- 构造分组收益 DataFrame ---
    group_ret_df = pd.DataFrame(group_returns, index=common_idx[:len(group_returns[1])])
    group_ret_df = group_ret_df.dropna(how="all")

    if group_ret_df.empty:
        return _empty_quantile_result(n_groups)

    # --- 各组统计 ---
    group_ann_return = {}
    group_ann_sharpe = {}
    for g in range(1, n_groups + 1):
        series = group_ret_df[g].dropna()
        group_ann_return[g] = annualize_return(series)
        group_ann_sharpe[g] = sharpe_ratio(series)

    # --- 多空组合: 做多顶组(最高因子值) - 做空底组(最低因子值) ---
    ls_returns = group_ret_df[n_groups] - group_ret_df[1]
    ls_returns = ls_returns.dropna()

    ls_ann_return = annualize_return(ls_returns) if len(ls_returns) > 0 else np.nan
    ls_sharpe = sharpe_ratio(ls_returns) if len(ls_returns) > 0 else np.nan

    # --- 单调性: 组序号与组平均收益的 Spearman 相关 ---
    group_mean_returns = pd.Series(
        {g: group_ret_df[g].mean() for g in range(1, n_groups + 1)}
    )
    from scipy.stats import spearmanr
    if len(group_mean_returns.dropna()) >= 3:
        mono_corr, _ = spearmanr(
            range(1, n_groups + 1), group_mean_returns.values
        )
    else:
        mono_corr = np.nan

    # --- 累计收益曲线 ---
    cum_by_group = group_ret_df.apply(cumulative_returns)
    ls_cumulative = cumulative_returns(ls_returns)

    return {
        "group_returns": group_ann_return,
        "group_sharpe": group_ann_sharpe,
        "long_short_return": float(ls_ann_return),
        "long_short_sharpe": float(ls_sharpe),
        "monotonicity": float(mono_corr),
        "cumulative_by_group": cum_by_group,
        "long_short_cumulative": ls_cumulative,
    }


def _empty_quantile_result(n_groups: int) -> dict:
    """空数据时返回的默认结果"""
    return {
        "group_returns": {g: np.nan for g in range(1, n_groups + 1)},
        "group_sharpe": {g: np.nan for g in range(1, n_groups + 1)},
        "long_short_return": np.nan,
        "long_short_sharpe": np.nan,
        "monotonicity": np.nan,
        "cumulative_by_group": pd.DataFrame(),
        "long_short_cumulative": pd.Series(dtype=float),
    }
