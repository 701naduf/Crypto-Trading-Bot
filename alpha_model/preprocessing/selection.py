"""
因子筛选模块

从 FactorStore 中的已入库因子中，筛选出适合组合建模的因子子集。

支持三种模式:
  A) threshold 模式: 按阈值逐步过滤（IC → VIF → 增量IC）
  B) top_k 模式: 按综合评分排序，取前 k 个
  C) 族级筛选: 每个族先选最优变体，再跨族筛选

筛选步骤是可选的，可以跳过（如树模型不需要筛选）。

复用 Phase 2a:
    factor_research.evaluation.ic.ic_analysis
    factor_research.evaluation.correlation.correlation_analysis, incremental_ic
    factor_research.evaluation.nonlinear.nonlinear_analysis  (互信息)
    factor_research.evaluation.quantile.quantile_backtest    (分层单调性)
    factor_research.evaluation.family_analyzer.FamilyAnalyzer (族级筛选)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from factor_research.evaluation.ic import ic_analysis
from factor_research.evaluation.correlation import correlation_analysis, incremental_ic
from factor_research.evaluation.nonlinear import nonlinear_analysis
from factor_research.evaluation.quantile import quantile_backtest
from factor_research.store.factor_store import FactorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 单因子评分
# ---------------------------------------------------------------------------

def _score_factor(
    factor_panel: pd.DataFrame,
    price_panel: pd.DataFrame,
    horizon: int,
    metric: str,
) -> float:
    """
    计算单个因子的评分

    Args:
        factor_panel: 因子面板 (timestamp × symbol)
        price_panel:  价格面板 (timestamp × symbol)
        horizon:      预测期
        metric:       评分指标 ("ic", "mi", "monotonicity")

    Returns:
        评分值（越高越好，取绝对值，因为负 IC 也有预测力）
    """
    if metric == "ic":
        # 使用 IC 的绝对值均值
        result = ic_analysis(factor_panel, price_panel, horizons=[horizon])
        ic_summary = result["ic_summary"][horizon]
        return abs(ic_summary["ic_mean"])

    elif metric == "mi":
        # 使用互信息
        result = nonlinear_analysis(
            factor_panel, price_panel, horizon=horizon,
        )
        return result["mutual_info"]

    elif metric == "monotonicity":
        # 使用分层收益单调性
        result = quantile_backtest(
            factor_panel, price_panel, horizon=horizon,
        )
        return abs(result["monotonicity"])

    else:
        raise ValueError(f"不支持的 metric: {metric}，可选 'ic', 'mi', 'monotonicity'")


# ---------------------------------------------------------------------------
# threshold 模式
# ---------------------------------------------------------------------------

def _threshold_select(
    factor_panels: dict[str, pd.DataFrame],
    price_panel: pd.DataFrame,
    horizon: int,
    metric: str,
    min_ic: float,
    max_vif: float,
    min_incremental_ic: float,
    min_factors: int,
) -> dict[str, pd.DataFrame]:
    """
    threshold 模式因子筛选

    三步过滤:
    1. 按 metric 评分过滤低质量因子
    2. VIF 过滤共线性因子
    3. 贪心增量 IC 筛选

    如果结果 < min_factors，保留评分最高的 min_factors 个。
    """
    # --- Step 1: 按 metric 评分过滤 ---
    scores = {}
    for name, panel in factor_panels.items():
        try:
            scores[name] = _score_factor(panel, price_panel, horizon, metric)
        except Exception as e:
            logger.warning("因子 '%s' 评分失败（已跳过）: %s", name, e)
            # 不赋 0.0 — 评分失败的因子不应通过筛选（不降级原则）

    # 按评分阈值过滤
    if metric == "ic":
        threshold = min_ic
    elif metric == "mi":
        # MI 没有自然阈值，使用中位数作为基准
        threshold = np.median(list(scores.values())) if scores else 0.0
    elif metric == "monotonicity":
        threshold = 0.5  # 单调性至少 > 0.5
    else:
        threshold = 0.0

    passed = {
        name: factor_panels[name]
        for name, score in scores.items()
        if score >= threshold
    }
    logger.info(
        "Step 1 (%s >= %.4f): %d/%d 因子通过",
        metric, threshold, len(passed), len(factor_panels),
    )

    if len(passed) <= min_factors:
        # 因子太少，保留评分最高的 min_factors 个
        if len(passed) < min_factors:
            sorted_names = sorted(scores, key=scores.get, reverse=True)
            passed = {
                name: factor_panels[name]
                for name in sorted_names[:min_factors]
                if name in factor_panels
            }
            logger.info("因子不足，放宽至评分 top %d", len(passed))
        return passed

    # --- Step 2: VIF 过滤 ---
    try:
        corr_result = correlation_analysis(passed)
        vif_dict = corr_result["vif"]
        # 逐步移除 VIF 最高的因子
        current = dict(passed)
        while True:
            # 重新计算 VIF
            if len(current) <= min_factors:
                break
            corr_result = correlation_analysis(current)
            vif_dict = corr_result["vif"]
            max_vif_name = max(vif_dict, key=vif_dict.get)
            if vif_dict[max_vif_name] <= max_vif:
                break
            logger.debug(
                "移除 VIF 最高因子 '%s' (VIF=%.2f)",
                max_vif_name, vif_dict[max_vif_name],
            )
            del current[max_vif_name]
        passed = current
    except Exception as e:
        logger.warning("VIF 过滤失败: %s，跳过此步", e)

    logger.info("Step 2 (VIF <= %.1f): %d 因子保留", max_vif, len(passed))

    if len(passed) <= min_factors:
        return passed

    # --- Step 3: 贪心增量 IC 筛选 ---
    # 按评分排序，从最高分开始贪心加入
    sorted_names = sorted(
        passed.keys(),
        key=lambda n: scores.get(n, 0), reverse=True,
    )
    selected_names = [sorted_names[0]]
    selected_panels = {sorted_names[0]: passed[sorted_names[0]]}

    for name in sorted_names[1:]:
        try:
            inc_result = incremental_ic(
                passed[name], selected_panels, price_panel, horizon=horizon,
            )
            if inc_result["incremental_ic"] >= min_incremental_ic:
                selected_names.append(name)
                selected_panels[name] = passed[name]
            else:
                logger.debug(
                    "因子 '%s' 增量 IC 不足 (%.4f < %.4f)，跳过",
                    name, inc_result["incremental_ic"], min_incremental_ic,
                )
        except Exception as e:
            logger.warning("因子 '%s' 增量 IC 计算失败: %s", name, e)

    logger.info("Step 3 (增量IC >= %.4f): %d 因子最终保留", min_incremental_ic, len(selected_panels))

    # 确保不少于 min_factors
    if len(selected_panels) < min_factors:
        for name in sorted_names:
            if name not in selected_panels:
                selected_panels[name] = passed[name]
                if len(selected_panels) >= min_factors:
                    break
        logger.info("增量 IC 筛选后不足，补足至 %d 个", len(selected_panels))

    return selected_panels


# ---------------------------------------------------------------------------
# top_k 模式
# ---------------------------------------------------------------------------

def _topk_select(
    factor_panels: dict[str, pd.DataFrame],
    price_panel: pd.DataFrame,
    horizon: int,
    top_k: int,
    score_weights: dict[str, float] | None,
) -> dict[str, pd.DataFrame]:
    """
    top_k 模式因子筛选

    对每个因子计算多维评分，加权综合后取前 top_k 个。
    """
    if score_weights is None:
        score_weights = {"ic": 0.5, "mi": 0.3, "monotonicity": 0.2}

    # 计算各维度评分
    all_scores = {}
    for name, panel in factor_panels.items():
        factor_scores = {}
        for metric in score_weights:
            try:
                factor_scores[metric] = _score_factor(
                    panel, price_panel, horizon, metric,
                )
            except Exception as e:
                logger.warning("因子 '%s' %s 评分失败: %s", name, metric, e)
                factor_scores[metric] = float("nan")
        all_scores[name] = factor_scores

    # 各维度内排名归一化（使不同指标的量纲可比较）
    score_df = pd.DataFrame(all_scores).T  # rows=factors, columns=metrics
    for col in score_df.columns:
        col_max = score_df[col].max()
        col_min = score_df[col].min()
        if col_max > col_min:
            score_df[col] = (score_df[col] - col_min) / (col_max - col_min)
        else:
            score_df[col] = 0.0

    # 加权综合评分
    composite = pd.Series(0.0, index=score_df.index)
    for metric, weight in score_weights.items():
        if metric in score_df.columns:
            composite += weight * score_df[metric]

    # 排序取前 top_k
    top_names = composite.nlargest(min(top_k, len(composite))).index.tolist()

    logger.info("top_k 模式: 从 %d 因子中选出 %d 个", len(factor_panels), len(top_names))
    for name in top_names:
        logger.debug("  %s: composite=%.4f", name, composite[name])

    return {name: factor_panels[name] for name in top_names}


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------

def select_factors(
    factor_panels: dict[str, pd.DataFrame],
    price_panel: pd.DataFrame,
    horizon: int = 10,
    mode: str = "threshold",
    # --- threshold 模式参数 ---
    metric: str = "ic",
    min_ic: float = 0.02,
    max_vif: float = 10.0,
    min_incremental_ic: float = 0.005,
    min_factors: int = 3,
    # --- top_k 模式参数 ---
    top_k: int | None = None,
    score_weights: dict[str, float] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    筛选因子

    threshold 模式:
        1. 对每个因子按 metric 计算评分，过滤低于阈值的因子
        2. 对剩余因子计算 VIF，逐步移除 VIF > max_vif 的共线性因子
        3. 贪心增量 IC 筛选：逐个加入因子，保留增量贡献足够大的
        4. 如果结果 < min_factors，放宽阈值重跑

    top_k 模式:
        1. 对每个因子计算多维评分（IC、MI、单调性等）
        2. 按 score_weights 加权综合评分
        3. 排序取前 top_k 个

    Args:
        factor_panels: {factor_name: panel (timestamp × symbol)}
        price_panel:   价格面板 (timestamp × symbol)
        horizon:       预测期（bar 数）
        mode:          筛选模式 ("threshold" 或 "top_k")
        metric:        threshold 模式的筛选指标
        min_ic:        IC 阈值
        max_vif:       VIF 阈值
        min_incremental_ic: 增量 IC 阈值
        min_factors:   threshold 模式最少保留数量
        top_k:         top_k 模式选出的因子数量
        score_weights: 各指标权重

    Returns:
        筛选后的 {factor_name: panel} 字典
    """
    if not factor_panels:
        raise ValueError("factor_panels 不能为空")

    logger.info("开始因子筛选: mode=%s, %d 个候选因子", mode, len(factor_panels))

    if mode == "threshold":
        return _threshold_select(
            factor_panels, price_panel, horizon,
            metric, min_ic, max_vif, min_incremental_ic, min_factors,
        )
    elif mode == "top_k":
        if top_k is None:
            top_k = max(3, len(factor_panels) // 3)
        return _topk_select(
            factor_panels, price_panel, horizon, top_k, score_weights,
        )
    else:
        raise ValueError(f"不支持的 mode: {mode}，可选 'threshold' 或 'top_k'")


def select_from_families(
    family_names: list[str],
    price_panel: pd.DataFrame,
    horizon: int = 10,
    family_select_metric: str = "ic_mean",
    store: FactorStore | None = None,
    **kwargs: Any,
) -> dict[str, pd.DataFrame]:
    """
    族级筛选

    步骤:
    1. 对每个族，加载全部变体
    2. 按 family_select_metric 排序选出最优变体
    3. 将各族最优变体收集为 {factor_name: panel}
    4. 跨族执行 select_factors() 进一步过滤

    Args:
        family_names:          因子族名列表
        price_panel:           价格面板 (timestamp × symbol)
        horizon:               预测期
        family_select_metric:  族内选优的排序指标
        store:                 FactorStore 实例，None 则使用默认
        **kwargs:              传递给 select_factors 的参数

    Returns:
        筛选后的 {factor_name: panel} 字典
    """
    if store is None:
        store = FactorStore()

    best_variants = {}

    for family_name in family_names:
        # 加载族内所有变体
        family_panels = store.load_family(family_name)
        if not family_panels:
            logger.warning("因子族 '%s' 无变体，跳过", family_name)
            continue

        # 计算每个变体的 IC 评分用于排序
        variant_scores = {}
        for variant_name, panel in family_panels.items():
            try:
                result = ic_analysis(panel, price_panel, horizons=[horizon])
                summary = result["ic_summary"][horizon]
                if family_select_metric in summary:
                    variant_scores[variant_name] = abs(summary[family_select_metric])
                else:
                    variant_scores[variant_name] = abs(summary.get("ic_mean", 0.0))
            except Exception as e:
                logger.warning("变体 '%s' 评分失败: %s", variant_name, e)
                variant_scores[variant_name] = 0.0

        # 选出最优变体
        if variant_scores:
            best_name = max(variant_scores, key=variant_scores.get)
            best_variants[best_name] = family_panels[best_name]
            logger.info(
                "族 '%s': 从 %d 变体中选出 '%s' (%s=%.4f)",
                family_name, len(family_panels), best_name,
                family_select_metric, variant_scores[best_name],
            )

    if not best_variants:
        raise ValueError("所有因子族均无有效变体")

    # 跨族筛选
    return select_factors(
        best_variants, price_panel, horizon=horizon, **kwargs,
    )
