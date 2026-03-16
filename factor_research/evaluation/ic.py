"""
IC / IR / IC 衰减分析模块

评价体系的第二层 API——IC 分析。

IC（Information Coefficient）是衡量因子预测力的核心指标。
本模块提供以下分析:
    1. 逐期 IC 序列: 每个时刻计算因子截面值与前瞻收益的 Spearman 相关
    2. IC 统计摘要: 均值、标准差、IC_IR、胜率
    3. IC 衰减曲线: IC 在不同前瞻窗口下的变化趋势
    4. IC 自相关: IC 时序的稳定性

IC 的计算方式:
    对于每个时刻 t:
        IC_t = Spearman_corr(factor_t[all_symbols], forward_return_{t+h}[all_symbols])
    即截面上所有标的的因子值与未来收益的秩相关。

    注意: 5 标的场景下截面 IC 的统计意义有限，
    但框架设计完整支持以确保扩展性。

依赖: evaluation.metrics
"""

import numpy as np
import pandas as pd
from scipy import stats

from data_infra.utils.logger import get_logger

from ..config import DEFAULT_HORIZONS, MIN_IC_OBSERVATIONS
from .metrics import (
    compute_forward_returns_panel,
    spearman_ic,
)

logger = get_logger(__name__)


def ic_series(
    factor_panel: pd.DataFrame,
    price_panel: pd.DataFrame,
    horizon: int = 1,
) -> pd.Series:
    """
    计算逐期 IC 序列

    对面板的每一行（每个时刻），计算截面 Spearman IC。

    Args:
        factor_panel: 因子面板 (index=timestamp, columns=symbols)
        price_panel:  价格面板（收盘价），格式同上
        horizon:      前瞻窗口（bar 数）

    Returns:
        pd.Series: IC 时间序列 (index=timestamp, values=IC)
    """
    forward_ret = compute_forward_returns_panel(price_panel, horizon)

    # 对齐 index 和 columns
    common_idx = factor_panel.index.intersection(forward_ret.index)
    common_cols = factor_panel.columns.intersection(forward_ret.columns)
    f = factor_panel.loc[common_idx, common_cols]
    r = forward_ret.loc[common_idx, common_cols]

    n_cols = len(common_cols)

    # 诊断日志: 数据对齐情况
    f_nan_ratio = f.isna().sum().sum() / max(f.size, 1)
    r_nan_ratio = r.isna().sum().sum() / max(r.size, 1)
    logger.debug(
        f"IC(h={horizon}): {len(common_idx)} 期 × {n_cols} 标的 | "
        f"因子 NaN 比例={f_nan_ratio:.2%}, 收益 NaN 比例={r_nan_ratio:.2%}"
    )

    if n_cols < MIN_IC_OBSERVATIONS:
        # 标的数不足，所有 IC 为 NaN
        logger.warning(
            f"IC(h={horizon}): 有效标的数 {n_cols} < 最小要求 {MIN_IC_OBSERVATIONS}，"
            f"返回全 NaN"
        )
        return pd.Series(np.nan, index=common_idx, name=f"IC_h{horizon}")

    # --- 向量化 Spearman: 逐行 rank → 逐行 Pearson ---
    # 预计算 rank 矩阵 (axis=1: 截面排名)
    # 对 NaN 的处理: rank 会跳过 NaN
    f_rank = f.rank(axis=1)
    r_rank = r.rank(axis=1)

    # 逐行有效观测数
    f_valid = f.notna()
    r_valid = r.notna()
    both_valid = f_valid & r_valid
    n_valid = both_valid.sum(axis=1)

    # 将无效位置设为 NaN，然后计算行均值和标准差
    f_rank_masked = f_rank.where(both_valid)
    r_rank_masked = r_rank.where(both_valid)

    f_mean = f_rank_masked.mean(axis=1)
    r_mean = r_rank_masked.mean(axis=1)

    f_centered = f_rank_masked.sub(f_mean, axis=0)
    r_centered = r_rank_masked.sub(r_mean, axis=0)

    # Pearson on ranks = Spearman
    cov = (f_centered * r_centered).sum(axis=1)
    f_std = (f_centered ** 2).sum(axis=1).pow(0.5)
    r_std = (r_centered ** 2).sum(axis=1).pow(0.5)

    denom = f_std * r_std
    ic_values = cov / denom

    # 有效观测不足的行设为 NaN
    ic_values = ic_values.where(n_valid >= MIN_IC_OBSERVATIONS, np.nan)
    # 零标准差（常数行）设为 NaN
    ic_values = ic_values.where(denom > 0, np.nan)

    ic_values.name = f"IC_h{horizon}"

    # 诊断日志: IC 计算结果
    valid_count = ic_values.notna().sum()
    total_count = len(ic_values)
    logger.debug(
        f"IC(h={horizon}): 有效 IC 数 = {valid_count}/{total_count} "
        f"({valid_count/max(total_count,1):.1%})"
    )

    return ic_values


def ic_summary(ic_ts: pd.Series) -> dict:
    """
    IC 序列的统计摘要

    Args:
        ic_ts: IC 时间序列（ic_series 的输出）

    Returns:
        dict: {
            "ic_mean":    IC 均值（平均预测力）,
            "ic_std":     IC 标准差（预测力波动）,
            "ic_ir":      IC_IR = ic_mean / ic_std（信息比率）,
            "ic_win_rate": IC > 0 的比例（IC 胜率）,
            "ic_skew":    IC 分布偏度,
            "ic_autocorr": IC 自相关（lag=1，衡量 IC 时序稳定性）,
            "n_observations": 有效观测数,
        }
    """
    valid = ic_ts.dropna()
    n = len(valid)

    if n == 0:
        return {
            "ic_mean": np.nan, "ic_std": np.nan, "ic_ir": np.nan,
            "ic_win_rate": np.nan, "ic_skew": np.nan,
            "ic_autocorr": np.nan, "n_observations": 0,
        }

    ic_mean = valid.mean()
    ic_std = valid.std()
    ic_ir = ic_mean / ic_std if ic_std > 0 else np.nan

    return {
        "ic_mean": float(ic_mean),
        "ic_std": float(ic_std),
        "ic_ir": float(ic_ir),
        "ic_win_rate": float((valid > 0).mean()),
        "ic_skew": float(valid.skew()),
        "ic_autocorr": float(valid.autocorr(lag=1)) if n > 1 else np.nan,
        "n_observations": n,
    }


def ic_decay(
    factor_panel: pd.DataFrame,
    price_panel: pd.DataFrame,
    horizons: list[int] = None,
) -> pd.DataFrame:
    """
    IC 衰减曲线

    计算因子在不同前瞻窗口下的 IC 均值和 IC_IR，
    揭示因子的预测力随时间衰减的速度。

    快速衰减: 因子捕捉的是短期效应（适合高频策略）
    缓慢衰减: 因子信号持续时间长（适合低频策略）

    Args:
        factor_panel: 因子面板
        price_panel:  价格面板
        horizons:     前瞻窗口列表，默认 [1, 5, 10, 30, 60]

    Returns:
        pd.DataFrame: index=horizon, columns=[ic_mean, ic_std, ic_ir, ic_win_rate]
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    results = []
    for h in horizons:
        ic_ts = ic_series(factor_panel, price_panel, horizon=h)
        summary = ic_summary(ic_ts)
        summary["horizon"] = h
        results.append(summary)

    df = pd.DataFrame(results).set_index("horizon")
    return df


def ic_analysis(
    factor_panel: pd.DataFrame,
    price_panel: pd.DataFrame,
    horizons: list[int] = None,
) -> dict:
    """
    完整 IC 分析（第二层 API 入口）

    返回所有 IC 相关指标: 逐期 IC 序列、汇总统计、衰减曲线。

    Args:
        factor_panel: 因子面板
        price_panel:  价格面板
        horizons:     前瞻窗口列表

    Returns:
        dict: {
            "ic_series":  {horizon: pd.Series},  逐期 IC 序列
            "ic_summary": {horizon: dict},        各窗口的 IC 统计摘要
            "ic_decay":   pd.DataFrame,           IC 衰减曲线
        }
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    series_dict = {}
    summary_dict = {}

    for h in horizons:
        ic_ts = ic_series(factor_panel, price_panel, horizon=h)
        series_dict[h] = ic_ts
        summary_dict[h] = ic_summary(ic_ts)

    decay_df = pd.DataFrame(
        [{"horizon": h, **summary_dict[h]} for h in horizons]
    ).set_index("horizon")

    return {
        "ic_series": series_dict,
        "ic_summary": summary_dict,
        "ic_decay": decay_df,
    }
