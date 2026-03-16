"""
标准化工具箱

提供多种标准化工具函数。这些是独立的工具函数，不在管道中自动执行。
用户/模型可根据需求自行调用。

所有函数严格遵守无未来信息约束：
- 时序标准化使用 expanding/rolling window，只看当前及之前
- 截面标准化在每个时刻独立计算，不涉及时序方向

设计理由：标准化不硬编码
    - 树模型（LightGBM/XGBoost）对单调变换不敏感，不需要标准化
    - 线性模型需要标准化，但 expanding/rolling/截面 z-score 各有优劣
    - 标准化方式是"研究决策"而非"工程决策"，不应由框架替用户做选择

expanding vs rolling 的权衡:
    - expanding: 锚定长期基准，信号含义 = "相对历史全局有多异常"。
                 早期样本少时估计不稳定。
    - rolling:   锚定近期基准，信号含义 = "相对近期有多反常"。
                 适合 regime shift 频繁的 crypto 市场，
                 但在新 regime 完全进入窗口后信号会消失。

复用 Phase 2a:
    factor_research.evaluation.metrics.cross_sectional_rank
    factor_research.evaluation.metrics.cross_sectional_zscore
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_model.core.types import TrainMode


# ---------------------------------------------------------------------------
# 时序标准化（逐列独立，无截面交互）
# ---------------------------------------------------------------------------

def expanding_zscore(
    panel: pd.DataFrame,
    min_periods: int = 252,
) -> pd.DataFrame:
    """
    Expanding window z-score（逐行累积）

    z_t = (x_t - mean(x_1..t)) / std(x_1..t)

    前 min_periods 行因样本不足输出 NaN。

    Args:
        panel:       因子面板 (timestamp × symbol)
        min_periods: 最少样本数，之前的行输出 NaN

    Returns:
        标准化后的面板，与输入同形状

    适用: 时序因子。度量 "这在历史上有多罕见"。
    注意: 早期（样本少）时均值/标准差估计不稳定。
    """
    expanding = panel.expanding(min_periods=min_periods)
    mean = expanding.mean()
    std = expanding.std()
    # std == 0 时返回 NaN（零波动率无法标准化，不返回 0）
    return (panel - mean) / std.replace(0, np.nan)


def rolling_zscore(
    panel: pd.DataFrame,
    window: int = 1000,
) -> pd.DataFrame:
    """
    Rolling window z-score

    z_t = (x_t - mean(x_{t-window+1}..t)) / std(x_{t-window+1}..t)

    前 window-1 行输出 NaN。

    Args:
        panel:  因子面板 (timestamp × symbol)
        window: 滚动窗口大小

    Returns:
        标准化后的面板

    适用: 时序因子，regime shift 频繁的市场。
          度量 "这在最近有多反常"。
    """
    rolling = panel.rolling(window=window, min_periods=window)
    mean = rolling.mean()
    std = rolling.std()
    return (panel - mean) / std.replace(0, np.nan)


# ---------------------------------------------------------------------------
# 截面标准化（每行独立，无时序交互）
# ---------------------------------------------------------------------------

def cross_sectional_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    """
    截面 z-score（每行独立）

    z_{i,t} = (x_{i,t} - mean_i(x_{·,t})) / std_i(x_{·,t})

    完全无未来信息，也无早期不稳定问题。

    Args:
        panel: 因子面板 (timestamp × symbol)

    Returns:
        截面标准化后的面板

    适用: 截面因子。
    """
    row_mean = panel.mean(axis=1)
    row_std = panel.std(axis=1)
    # std == 0 时返回 NaN
    row_std = row_std.replace(0, np.nan)
    return panel.sub(row_mean, axis=0).div(row_std, axis=0)


def cross_sectional_rank(panel: pd.DataFrame) -> pd.DataFrame:
    """
    截面百分位排名（每行独立）

    rank_{i,t} = percentile_rank of x_{i,t} among all symbols at time t

    值域 [0, 1]，完全无分布假设，最鲁棒。

    Args:
        panel: 因子面板 (timestamp × symbol)

    Returns:
        截面排名面板，值域 [0, 1]

    适用: 通用。
    """
    return panel.rank(axis=1, pct=True)


# ---------------------------------------------------------------------------
# 去极值
# ---------------------------------------------------------------------------

def winsorize(
    panel: pd.DataFrame,
    sigma: float = 3.0,
    method: str = "expanding",
    window: int = 1000,
    min_periods: int = 60,
) -> pd.DataFrame:
    """
    去极值（Winsorize）

    将超过 ±sigma 个标准差的值截断到边界值。
    标准差的计算方式由 method 参数决定。

    Args:
        panel:       因子面板 (timestamp × symbol)
        sigma:       截断阈值（标准差倍数）
        method:      标准差计算方式:
                     "expanding" — expanding window（逐列时序）
                     "rolling"   — rolling window（逐列时序）
                     "cross_sectional" — 截面（每行独立）
        window:      rolling 窗口大小（method="rolling" 时使用）
        min_periods: 最少样本数（method="expanding" 时使用）

    Returns:
        去极值后的面板

    适用: 原始因子值的数据清洗（如流动性缺失导致的 spike）。
    注意: 不应用于模型预测值的截断（见信号生成部分）。
    """
    if method == "expanding":
        exp = panel.expanding(min_periods=min_periods)
        mean = exp.mean()
        std = exp.std()
    elif method == "rolling":
        roll = panel.rolling(window=window, min_periods=min(window, min_periods))
        mean = roll.mean()
        std = roll.std()
    elif method == "cross_sectional":
        mean = panel.mean(axis=1)
        std = panel.std(axis=1)
        # 转为与 panel 同形状以便后续运算
        mean = pd.DataFrame(
            np.tile(mean.values[:, None], (1, panel.shape[1])),
            index=panel.index, columns=panel.columns,
        )
        std = pd.DataFrame(
            np.tile(std.values[:, None], (1, panel.shape[1])),
            index=panel.index, columns=panel.columns,
        )
    else:
        raise ValueError(
            f"不支持的 method: {method}，可选 'expanding', 'rolling', 'cross_sectional'"
        )

    upper = mean + sigma * std
    lower = mean - sigma * std
    return panel.clip(lower=lower, upper=upper)


# ---------------------------------------------------------------------------
# 特征矩阵构建
# ---------------------------------------------------------------------------

def build_feature_matrix(
    factor_panels: dict[str, pd.DataFrame],
    symbols: list[str],
    train_mode: TrainMode,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """
    将多个因子面板合并为模型输入的特征矩阵

    注意：此函数只做合并和 reshape，不做标准化。
    标准化应由用户在调用此函数之前或之后自行处理。

    Pooled 模式:
        将所有 symbol 堆叠，返回
        index = MultiIndex(timestamp, symbol)
        columns = factor_names

    Per-Symbol 模式:
        返回 {symbol: DataFrame(index=timestamp, columns=factor_names)}

    Args:
        factor_panels: {factor_name: panel (timestamp × symbol)}
        symbols:       要包含的标的列表
        train_mode:    训练模式

    Returns:
        Pooled: pd.DataFrame with MultiIndex(timestamp, symbol)
        Per-Symbol: dict[str, pd.DataFrame]

    Raises:
        ValueError: 无因子面板或无有效标的时
    """
    if not factor_panels:
        raise ValueError("factor_panels 不能为空")
    if not symbols:
        raise ValueError("symbols 不能为空")

    # 取所有面板共同的时间索引
    common_index = None
    for panel in factor_panels.values():
        if common_index is None:
            common_index = panel.index
        else:
            common_index = common_index.intersection(panel.index)

    if common_index is None or len(common_index) == 0:
        raise ValueError("因子面板无共同时间索引")

    if train_mode == TrainMode.POOLED:
        # 构建 Pooled 格式: MultiIndex(timestamp, symbol), columns=factor_names
        pieces = []
        for symbol in symbols:
            symbol_data = {}
            for factor_name, panel in factor_panels.items():
                if symbol in panel.columns:
                    symbol_data[factor_name] = panel.loc[common_index, symbol]
            if symbol_data:
                df = pd.DataFrame(symbol_data, index=common_index)
                df["symbol"] = symbol
                pieces.append(df)

        if not pieces:
            raise ValueError(f"没有找到 symbols {symbols} 中的有效数据")

        combined = pd.concat(pieces, axis=0)
        combined = combined.set_index("symbol", append=True)
        combined.index.names = ["timestamp", "symbol"]
        # 按 timestamp 排序（保持时间顺序），然后 symbol
        combined = combined.sort_index(level=["timestamp", "symbol"])
        return combined

    elif train_mode == TrainMode.PER_SYMBOL:
        # 构建 Per-Symbol 格式: {symbol: DataFrame(index=timestamp, columns=factor_names)}
        result = {}
        for symbol in symbols:
            symbol_data = {}
            for factor_name, panel in factor_panels.items():
                if symbol in panel.columns:
                    symbol_data[factor_name] = panel.loc[common_index, symbol]
            if symbol_data:
                result[symbol] = pd.DataFrame(symbol_data, index=common_index)

        if not result:
            raise ValueError(f"没有找到 symbols {symbols} 中的有效数据")

        return result

    else:
        raise ValueError(f"不支持的 train_mode: {train_mode}")


# ---------------------------------------------------------------------------
# Pooled 目标变量构建 [P3]
# ---------------------------------------------------------------------------

def build_pooled_target(
    X: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    symbols: list[str],
) -> pd.Series:
    """
    为 Pooled 模式构建目标变量

    将 fwd_returns (timestamp × symbol) 重塑为
    与 X 同样的 MultiIndex(timestamp, symbol) 的 Series。

    Args:
        X:            Pooled 特征矩阵, index=MultiIndex(timestamp, symbol)
        fwd_returns:  forward return 面板 (timestamp × symbol)
        symbols:      标的列表

    Returns:
        pd.Series, index=MultiIndex(timestamp, symbol), 与 X 对齐
    """
    pieces = []
    for symbol in symbols:
        if symbol not in fwd_returns.columns:
            continue
        s = fwd_returns[symbol].copy()
        idx = pd.MultiIndex.from_arrays(
            [s.index, [symbol] * len(s)],
            names=["timestamp", "symbol"],
        )
        pieces.append(pd.Series(s.values, index=idx, name="target"))

    if not pieces:
        raise ValueError("无法构建目标变量")

    y_full = pd.concat(pieces).sort_index()
    return y_full.reindex(X.index)
