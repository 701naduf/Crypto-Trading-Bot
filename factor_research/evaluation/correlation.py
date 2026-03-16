"""
多因子相关性与冗余检测模块

评价体系的第二层 API——多因子诊断。

当因子库积累到一定数量后，需要检查因子之间的冗余性:
    - 高相关因子提供重复信息，模型中同时使用会降低效率
    - 新因子的边际贡献（增量 IC）可能很小
    - 多重共线性 (VIF) 高的因子组合不适合线性模型

分析维度:
    1. 因子相关性矩阵: 所有因子对之间的 Spearman 相关
    2. 增量 IC: 新因子加入后 IC 的变化
    3. VIF: 方差膨胀因子，衡量多重共线性

依赖: evaluation.metrics, evaluation.ic
"""

import numpy as np
import pandas as pd
from scipy import stats

from ..config import MIN_REGRESSION_OBSERVATIONS


def correlation_analysis(
    factor_panels: dict[str, pd.DataFrame],
) -> dict:
    """
    多因子相关性分析（第二层 API 入口）

    Args:
        factor_panels: {因子名: 因子面板} 字典
                       每个面板格式: index=timestamp, columns=symbols

    Returns:
        dict: {
            "correlation_matrix": 因子相关性矩阵 DataFrame,
            "vif":                各因子的 VIF 值 dict,
        }
    """
    if len(factor_panels) < 2:
        return {
            "correlation_matrix": pd.DataFrame(),
            "vif": {},
        }

    # 将所有因子面板展平为一维，合并成一个 DataFrame
    flat_factors = {}
    for name, panel in factor_panels.items():
        flat_factors[name] = panel.stack()

    factor_df = pd.DataFrame(flat_factors).dropna()

    if len(factor_df) < MIN_REGRESSION_OBSERVATIONS:
        return {
            "correlation_matrix": pd.DataFrame(),
            "vif": {},
        }

    # 1. 因子相关性矩阵 (Spearman)
    corr_matrix = factor_df.corr(method="spearman")

    # 2. VIF (方差膨胀因子)
    vif_values = _compute_vif(factor_df)

    return {
        "correlation_matrix": corr_matrix,
        "vif": vif_values,
    }


def incremental_ic(
    new_factor: pd.DataFrame,
    existing_factors: dict[str, pd.DataFrame],
    price_panel: pd.DataFrame,
    horizon: int = 1,
) -> dict:
    """
    增量 IC 分析

    衡量新因子相对于已有因子组合的边际贡献。

    方法:
        1. 计算新因子的原始 IC
        2. 将新因子对已有因子做截面回归，取残差
        3. 计算残差的 IC → 即新因子的增量 IC
        4. 增量 IC / 原始 IC → 信息保留率

    Args:
        new_factor:       新因子面板
        existing_factors: {名称: 面板} 已有因子字典
        price_panel:      价格面板
        horizon:          前瞻窗口

    Returns:
        dict: {
            "raw_ic":        原始 IC 均值,
            "incremental_ic": 增量 IC 均值（去除已有因子解释后的剩余预测力）,
            "info_retention": 信息保留率 = incremental_ic / raw_ic,
        }
    """
    from .ic import ic_series, ic_summary

    # 原始 IC
    raw_ic_ts = ic_series(new_factor, price_panel, horizon=horizon)
    raw_summary = ic_summary(raw_ic_ts)
    raw_ic_mean = raw_summary["ic_mean"]

    if not existing_factors or np.isnan(raw_ic_mean):
        return {
            "raw_ic": raw_ic_mean,
            "incremental_ic": raw_ic_mean,
            "info_retention": 1.0,
        }

    # 构造残差因子: 新因子对已有因子做截面回归取残差
    residual_panel = _compute_residual_factor(new_factor, existing_factors)

    if residual_panel.empty:
        return {
            "raw_ic": raw_ic_mean,
            "incremental_ic": np.nan,
            "info_retention": np.nan,
        }

    # 增量 IC
    incr_ic_ts = ic_series(residual_panel, price_panel, horizon=horizon)
    incr_summary = ic_summary(incr_ic_ts)
    incr_ic_mean = incr_summary["ic_mean"]

    # 信息保留率
    retention = incr_ic_mean / raw_ic_mean if raw_ic_mean != 0 else np.nan

    return {
        "raw_ic": float(raw_ic_mean),
        "incremental_ic": float(incr_ic_mean),
        "info_retention": float(retention) if not np.isnan(retention) else np.nan,
    }


def _compute_vif(factor_df: pd.DataFrame) -> dict:
    """
    计算各因子的方差膨胀因子 (VIF)

    VIF_i = 1 / (1 - R²_i)
    其中 R²_i 是将因子 i 对其余因子做回归的 R²。

    VIF 经验阈值:
        VIF < 5:  可接受
        5 ≤ VIF < 10: 有共线性嫌疑
        VIF ≥ 10: 严重共线性

    数值稳定性保障:
        - 回归前检查设计矩阵的条件数，过大时发出警告
        - 使用 lstsq（SVD 分解）而非直接矩阵求逆
        - 对 R² 做 clip 处理，防止数值误差导致的 >1 情况

    Returns:
        dict: {因子名: VIF 值}
    """
    from numpy.linalg import LinAlgError

    from data_infra.utils.logger import get_logger
    logger = get_logger(__name__)

    vif = {}
    columns = factor_df.columns.tolist()

    for i, col in enumerate(columns):
        y = factor_df[col].values
        X = factor_df.drop(columns=col).values

        if X.shape[1] == 0:
            vif[col] = 1.0
            continue

        try:
            # 添加截距
            X_with_const = np.column_stack([np.ones(len(X)), X])

            # 条件数检查: 过大说明设计矩阵近奇异
            cond = np.linalg.cond(X_with_const)
            if cond > 1e12:
                logger.warning(
                    f"VIF 计算: 因子 '{col}' 的设计矩阵条件数 = {cond:.2e}，"
                    f"结果可能不稳定"
                )

            # OLS (SVD 分解，数值稳定)
            beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
            y_pred = X_with_const @ beta
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)

            if ss_tot == 0:
                # 因子为常数，无方差 → VIF 无定义
                vif[col] = np.nan
                continue

            r_squared = 1 - ss_res / ss_tot
            # clip 防止数值误差导致 R² 略超 1 或略小于 0
            r_squared = np.clip(r_squared, 0.0, 1.0)

            vif[col] = 1 / (1 - r_squared) if r_squared < 1.0 else float("inf")
        except (LinAlgError, ValueError) as e:
            logger.warning(f"VIF 计算失败: 因子 '{col}', 原因: {e}")
            vif[col] = np.nan

    return vif


def _compute_residual_factor(
    new_factor: pd.DataFrame,
    existing_factors: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    计算新因子相对于已有因子的残差

    在每个时刻，做截面回归: new_factor ~ existing_factors，取残差。
    残差代表新因子中不能被已有因子解释的部分。

    Returns:
        pd.DataFrame: 残差因子面板
    """
    # 对齐所有面板的 index 和 columns
    all_panels = list(existing_factors.values()) + [new_factor]
    common_idx = all_panels[0].index
    common_cols = all_panels[0].columns
    for p in all_panels[1:]:
        common_idx = common_idx.intersection(p.index)
        common_cols = common_cols.intersection(p.columns)

    if len(common_idx) < MIN_REGRESSION_OBSERVATIONS or len(common_cols) < 2:
        return pd.DataFrame()

    new_aligned = new_factor.loc[common_idx, common_cols]

    # 展平后做回归
    new_flat = new_aligned.stack()
    existing_flat = pd.DataFrame({
        name: panel.loc[common_idx, common_cols].stack()
        for name, panel in existing_factors.items()
    })

    combined = pd.concat([new_flat.rename("target"), existing_flat], axis=1).dropna()

    if len(combined) < MIN_REGRESSION_OBSERVATIONS:
        return pd.DataFrame()

    y = combined["target"].values
    X = combined.drop(columns="target").values
    X_with_const = np.column_stack([np.ones(len(X)), X])

    try:
        beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
        residuals = y - X_with_const @ beta
    except Exception:
        return pd.DataFrame()

    # 重建为面板
    residual_series = pd.Series(residuals, index=combined.index)
    return residual_series.unstack()
