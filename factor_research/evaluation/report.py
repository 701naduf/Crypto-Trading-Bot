"""
评价报告生成模块

将各评价维度的结果汇总为结构化报告，
并提供可视化图表生成功能。

报告包含:
    - 文本摘要: IC/IR、分层收益、换手率等关键指标
    - 可视化图表: IC 衰减曲线、分层收益柱状图、因子特征曲线等

图表使用 matplotlib 生成，可选使用 seaborn 美化。
报告对象可以直接在 notebook 中显示，也可导出为图片。

依赖: matplotlib, seaborn (可选)
"""

import numpy as np
import pandas as pd

from data_infra.utils.logger import get_logger

logger = get_logger(__name__)


def format_report_text(report: dict, factor_name: str = "") -> str:
    """
    将评价报告格式化为可读的文本摘要

    Args:
        report: full_report() 返回的完整报告字典
        factor_name: 因子名称（用于标题）

    Returns:
        str: 格式化的文本报告
    """
    lines = []
    lines.append(f"{'=' * 60}")
    lines.append(f"因子评价报告: {factor_name}")
    lines.append(f"{'=' * 60}")

    # --- IC 分析 ---
    ic_report = report.get("ic", {})
    ic_decay = ic_report.get("ic_decay")
    if ic_decay is not None and not ic_decay.empty:
        lines.append("\n--- IC / IR 分析 ---")
        for h in ic_decay.index:
            row = ic_decay.loc[h]
            lines.append(
                f"  h={h:>3d}:  IC={row.get('ic_mean', np.nan):+.4f}  "
                f"IR={row.get('ic_ir', np.nan):+.4f}  "
                f"胜率={row.get('ic_win_rate', np.nan):.1%}"
            )

    # --- 分层回测 ---
    quantile_report = report.get("quantile", {})
    group_returns = quantile_report.get("group_returns", {})
    if group_returns:
        lines.append("\n--- 分层回测 ---")
        for g, ret in sorted(group_returns.items()):
            lines.append(f"  组 {g}: 年化收益 = {ret:+.2%}")
        ls_ret = quantile_report.get("long_short_return", np.nan)
        ls_sharpe = quantile_report.get("long_short_sharpe", np.nan)
        mono = quantile_report.get("monotonicity", np.nan)
        lines.append(f"  多空收益 = {ls_ret:+.2%}  Sharpe = {ls_sharpe:.2f}  单调性 = {mono:.2f}")

    # --- 尾部分析 ---
    tail_report = report.get("tail", {})
    if tail_report.get("n_tail_observations", 0) > 0:
        lines.append("\n--- 尾部分析 ---")
        lines.append(f"  条件IC = {tail_report.get('conditional_ic', np.nan):+.4f}")
        lines.append(f"  命中率 = {tail_report.get('tail_hit_rate', np.nan):.1%}")
        lines.append(f"  尾部期望收益 = {tail_report.get('tail_expected_return', np.nan):+.6f}")
        mae = tail_report.get("mae", np.nan)
        if not np.isnan(mae):
            lines.append(f"  最大逆向偏移 (MAE) = {mae:+.6f}")
        lines.append(f"  极端信号频率 = {tail_report.get('tail_frequency', np.nan):.1%}")

    # --- 换手率 ---
    turnover_report = report.get("turnover", {})
    autocorr = turnover_report.get("autocorrelation", np.nan)
    flip = turnover_report.get("signal_flip_rate", np.nan)
    if not np.isnan(autocorr):
        lines.append("\n--- 换手率分析 ---")
        lines.append(f"  因子自相关 = {autocorr:.4f}")
        lines.append(f"  信号翻转率 = {flip:.1%}")

    # --- 非线性分析 ---
    nonlinear_report = report.get("nonlinear", {})
    mi = nonlinear_report.get("mutual_info", np.nan)
    if not np.isnan(mi):
        lines.append("\n--- 非线性分析 ---")
        lines.append(f"  互信息 = {mi:.4f}")
        cond_ic = nonlinear_report.get("conditional_ic", {})
        lines.append(
            f"  条件IC: 低={cond_ic.get('low_ic', np.nan):+.4f}  "
            f"中={cond_ic.get('mid_ic', np.nan):+.4f}  "
            f"高={cond_ic.get('high_ic', np.nan):+.4f}"
        )

    # --- 稳定性 ---
    stability_report = report.get("stability", {})
    ic_mdd = stability_report.get("ic_max_drawdown", np.nan)
    if not np.isnan(ic_mdd):
        lines.append("\n--- 稳定性分析 ---")
        lines.append(f"  IC 最大回撤 = {ic_mdd:.4f}")

    lines.append(f"\n{'=' * 60}")
    return "\n".join(lines)


def plot_report(report: dict, factor_name: str = "") -> dict:
    """
    生成评价报告的可视化图表

    返回 matplotlib Figure 对象字典，方便在 notebook 中显示或导出。

    Args:
        report: full_report() 返回的完整报告字典
        factor_name: 因子名称（用于图标标题）

    Returns:
        dict: {图表名称: matplotlib.figure.Figure}
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib 未安装，跳过图表生成")
        return {}

    figures = {}

    # --- 1. IC 衰减曲线 ---
    ic_report = report.get("ic", {})
    ic_decay = ic_report.get("ic_decay")
    if ic_decay is not None and not ic_decay.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(
            [str(h) for h in ic_decay.index],
            ic_decay["ic_mean"],
            color=["#2196F3" if v > 0 else "#F44336" for v in ic_decay["ic_mean"]],
            alpha=0.8,
        )
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_xlabel("前瞻窗口 (bars)")
        ax.set_ylabel("IC 均值")
        ax.set_title(f"{factor_name} - IC 衰减曲线")
        fig.tight_layout()
        figures["ic_decay"] = fig

    # --- 2. 分层收益柱状图 ---
    quantile_report = report.get("quantile", {})
    group_returns = quantile_report.get("group_returns", {})
    if group_returns:
        fig, ax = plt.subplots(figsize=(8, 5))
        groups = sorted(group_returns.keys())
        returns = [group_returns[g] for g in groups]
        colors = ["#4CAF50" if r > 0 else "#F44336" for r in returns]
        ax.bar([f"组{g}" for g in groups], returns, color=colors, alpha=0.8)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_ylabel("年化收益率")
        ax.set_title(f"{factor_name} - 分层回测")
        fig.tight_layout()
        figures["quantile_returns"] = fig

    # --- 3. 分层累计收益曲线 ---
    cum_by_group = quantile_report.get("cumulative_by_group")
    if cum_by_group is not None and not cum_by_group.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        for col in cum_by_group.columns:
            ax.plot(cum_by_group.index, cum_by_group[col], label=f"组{col}", alpha=0.8)
        ax.legend()
        ax.set_ylabel("累计收益")
        ax.set_title(f"{factor_name} - 各组累计收益")
        fig.tight_layout()
        figures["quantile_cumulative"] = fig

    # --- 4. IC 时间序列 ---
    ic_series_dict = ic_report.get("ic_series", {})
    if 1 in ic_series_dict:
        ic_ts = ic_series_dict[1]
        if not ic_ts.dropna().empty:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.bar(range(len(ic_ts)), ic_ts.values, width=1.0, alpha=0.5, color="#2196F3")
            # 滚动均值
            rolling = ic_ts.rolling(60, min_periods=1).mean()
            ax.plot(range(len(rolling)), rolling.values, color="red", linewidth=1.5, label="60-bar MA")
            ax.axhline(y=0, color="black", linewidth=0.5)
            ax.legend()
            ax.set_ylabel("IC")
            ax.set_title(f"{factor_name} - IC 时间序列 (h=1)")
            fig.tight_layout()
            figures["ic_timeseries"] = fig

    # --- 5. 因子特征曲线 ---
    nonlinear_report = report.get("nonlinear", {})
    profile = nonlinear_report.get("factor_profile")
    if profile is not None and not profile.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(profile["factor_median"], profile["return_mean"], "o-", color="#2196F3")
        ax.fill_between(
            profile["factor_median"],
            profile["return_mean"] - profile["return_std"],
            profile["return_mean"] + profile["return_std"],
            alpha=0.2, color="#2196F3",
        )
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_xlabel("因子值 (分箱中位数)")
        ax.set_ylabel("平均收益")
        ax.set_title(f"{factor_name} - 因子特征曲线")
        fig.tight_layout()
        figures["factor_profile"] = fig

    plt.close("all")
    return figures
