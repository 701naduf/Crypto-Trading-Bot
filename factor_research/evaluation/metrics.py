"""
底层计算原语（纯函数）

评价体系的第三层 API——最底层。
所有函数都是无状态的纯函数，不依赖任何类或全局状态。
上层的 ic.py、quantile.py 等模块组合调用这些原语。

函数分类:
    1. 相关性度量: spearman_ic, pearson_ic
    2. 前瞻收益: compute_forward_returns
    3. 排名工具: rank_normalize, cross_sectional_rank
    4. 非线性度量: mutual_information
    5. 收益计算: cumulative_returns, annualize_return, annualize_volatility

设计原则:
    - 输入/输出都是 numpy array 或 pandas Series/DataFrame
    - 不做日志记录、不抛业务异常（只有 ValueError）
    - 不依赖 factor_research 的其他模块
    - 可直接在 notebook 中独立使用

依赖: numpy, pandas, scipy
被依赖: evaluation 下的所有上层模块
"""

import numpy as np
import pandas as pd
from scipy import stats

from ..config import MIN_IC_OBSERVATIONS


# =========================================================================
# 1. 相关性度量
# =========================================================================

def spearman_ic(factor: pd.Series, returns: pd.Series) -> float:
    """
    计算 Spearman 秩相关系数（Information Coefficient）

    Spearman IC 是因子评价中最核心的指标:
        IC = corr(rank(factor_t), rank(forward_return_{t+h}))

    使用秩相关而非 Pearson:
        - 对异常值稳健（不受极端因子值影响）
        - 只要求单调关系，不要求线性关系
        - 量化领域的行业标准

    Args:
        factor:  因子值 Series（已与收益对齐，无需额外处理）
        returns: 前瞻收益 Series（同 index）

    Returns:
        float: Spearman 相关系数，范围 [-1, 1]
               返回 NaN 如果有效观测不足
    """
    # 去掉两边都有值的观测
    mask = factor.notna() & returns.notna()
    f = factor[mask]
    r = returns[mask]

    if len(f) < MIN_IC_OBSERVATIONS:
        return np.nan

    corr, _ = stats.spearmanr(f, r)
    return float(corr)


def pearson_ic(factor: pd.Series, returns: pd.Series) -> float:
    """
    计算 Pearson 线性相关系数

    与 Spearman IC 的区别:
        - 衡量线性关系强度（而非单调关系）
        - 对异常值敏感
        - 很少单独用于因子评价，但在特定分析中有用

    Args:
        factor:  因子值 Series
        returns: 前瞻收益 Series

    Returns:
        float: Pearson 相关系数，范围 [-1, 1]
    """
    mask = factor.notna() & returns.notna()
    f = factor[mask]
    r = returns[mask]

    if len(f) < MIN_IC_OBSERVATIONS:
        return np.nan

    corr, _ = stats.pearsonr(f, r)
    return float(corr)


# =========================================================================
# 2. 前瞻收益计算
# =========================================================================

def compute_forward_returns(
    prices: pd.Series,
    horizon: int,
) -> pd.Series:
    """
    计算前瞻收益（simple forward return）

    公式: forward_return_t = price_{t+h} / price_t - 1

    这是因子 IC 分析的目标变量。使用简单收益而非对数收益，
    因为简单收益在截面上可加（组合收益 = 成分收益的加权和）。

    Args:
        prices:  价格 Series (index 为 DatetimeIndex)
        horizon: 前瞻窗口（bar 数）

    Returns:
        pd.Series: 前瞻收益，最后 horizon 个值为 NaN（未来未知）

    Examples:
        >>> prices = pd.Series([100, 102, 101, 105, 103])
        >>> compute_forward_returns(prices, 2)
        # [0.01, 0.0294, 0.0198, NaN, NaN]
    """
    return prices.shift(-horizon) / prices - 1


def compute_forward_returns_panel(
    price_panel: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """
    计算面板格式的前瞻收益

    对价格面板的每一列（每个 symbol）独立计算前瞻收益。

    Args:
        price_panel: 价格面板 (index=timestamp, columns=symbols)
        horizon:     前瞻窗口（bar 数）

    Returns:
        pd.DataFrame: 前瞻收益面板，格式同输入
    """
    return price_panel.apply(lambda col: compute_forward_returns(col, horizon))


# =========================================================================
# 3. 排名与标准化工具
# =========================================================================

def rank_normalize(series: pd.Series) -> pd.Series:
    """
    百分位排名标准化

    将因子值转换为 [0, 1] 区间的百分位排名。
    在截面分析中常用于消除因子量纲的影响。

    Args:
        series: 因子值 Series

    Returns:
        pd.Series: 百分位排名，范围 [0, 1]
    """
    return series.rank(pct=True)


def cross_sectional_rank(panel: pd.DataFrame) -> pd.DataFrame:
    """
    截面百分位排名

    对面板的每一行（每个时刻）进行截面排名。
    这是截面因子分析的标准预处理步骤。

    Args:
        panel: 因子面板 (index=timestamp, columns=symbols)

    Returns:
        pd.DataFrame: 截面排名面板，每行的值在 [0, 1] 区间
    """
    return panel.rank(axis=1, pct=True)


def zscore_normalize(series: pd.Series) -> pd.Series:
    """
    Z-score 标准化

    公式: z = (x - mean) / std

    Args:
        series: 因子值 Series

    Returns:
        pd.Series: 标准化后的值
    """
    mean = series.mean()
    std = series.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - mean) / std


def cross_sectional_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    """
    截面 Z-score 标准化

    对面板的每一行（每个时刻）进行截面 Z-score 标准化。

    Args:
        panel: 因子面板 (index=timestamp, columns=symbols)

    Returns:
        pd.DataFrame: 标准化面板
    """
    row_mean = panel.mean(axis=1)
    row_std = panel.std(axis=1)
    row_std = row_std.replace(0, np.nan)
    return panel.sub(row_mean, axis=0).div(row_std, axis=0)


# =========================================================================
# 4. 非线性度量
# =========================================================================

def mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    n_neighbors: int = 5,
) -> float:
    """
    互信息（Mutual Information）估计

    使用 k-近邻方法（KSG 估计量）估计两个连续变量的互信息。
    互信息能捕捉任意统计依赖关系（不仅限于线性或单调），
    是 IC（Spearman 相关）的重要补充。

    MI = 0 表示统计独立，MI > 0 表示存在某种依赖关系。
    MI 没有上界，其绝对值难以直观解读，通常用于比较不同因子。

    Args:
        x: 第一个变量（1D array）
        y: 第二个变量（1D array）
        n_neighbors: KNN 邻居数，默认 5

    Returns:
        float: 互信息估计值（非负）
    """
    from sklearn.feature_selection import mutual_info_regression

    # 去掉 NaN
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask].reshape(-1, 1)
    y_clean = y[mask]

    if len(x_clean) < n_neighbors + 1:
        return np.nan

    mi = mutual_info_regression(
        x_clean, y_clean, n_neighbors=n_neighbors, random_state=42
    )
    return float(mi[0])


# =========================================================================
# 5. 收益计算工具
# =========================================================================

def cumulative_returns(returns: pd.Series) -> pd.Series:
    """
    累计收益曲线

    公式: cum_ret_t = (1 + r_1) * (1 + r_2) * ... * (1 + r_t) - 1

    Args:
        returns: 简单收益 Series

    Returns:
        pd.Series: 累计收益曲线（从 0 开始）
    """
    return (1 + returns).cumprod() - 1


def annualize_return(
    returns: pd.Series,
    periods_per_year: float = 525960,
) -> float:
    """
    年化收益率

    默认假设收益是分钟频率 (525960 = 365.25 * 24 * 60)，
    crypto 市场 7×24 运行。

    Args:
        returns:         简单收益 Series
        periods_per_year: 每年的 bar 数

    Returns:
        float: 年化收益率
    """
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    if n_periods == 0:
        return 0.0
    return (1 + total_return) ** (periods_per_year / n_periods) - 1


def annualize_volatility(
    returns: pd.Series,
    periods_per_year: float = 525960,
) -> float:
    """
    年化波动率

    Args:
        returns:         简单收益 Series
        periods_per_year: 每年的 bar 数

    Returns:
        float: 年化波动率
    """
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(
    returns: pd.Series,
    periods_per_year: float = 525960,
    risk_free_rate: float = 0.0,
) -> float:
    """
    夏普比率

    Args:
        returns:         简单收益 Series
        periods_per_year: 每年的 bar 数
        risk_free_rate:  年化无风险利率

    Returns:
        float: 夏普比率
    """
    ann_ret = annualize_return(returns, periods_per_year)
    ann_vol = annualize_volatility(returns, periods_per_year)
    if np.isnan(ann_vol):
        return np.nan
    excess = ann_ret - risk_free_rate
    # 波动率极小（浮点噪声级别）时视为零波动率
    # 阈值 1e-12 远小于任何有意义的年化波动率（最小也在 0.01 量级）
    if ann_vol < 1e-12:
        if np.isinf(excess) and excess > 0:
            return np.inf
        elif np.isinf(excess) and excess < 0:
            return -np.inf
        elif excess > 1e-12:
            return np.inf
        elif excess < -1e-12:
            return -np.inf
        else:
            return np.nan
    return excess / ann_vol


def max_drawdown(cumulative: pd.Series) -> float:
    """
    最大回撤

    基于净值（wealth = 1 + cumulative_return）计算。
    返回负数，符合行业惯例: MDD = -0.15 表示从峰值回撤 15%。

    公式:
        drawdown_t = (wealth_t - running_max_t) / running_max_t
        MDD = min(drawdown_t)  （负数或零）

    Args:
        cumulative: 累计收益曲线（从 0 开始，即 cumulative_returns 的输出）

    Returns:
        float: 最大回撤（非正数）。
               -0.15 表示 15% 回撤，0.0 表示无回撤。
    """
    wealth = 1 + cumulative
    running_max = wealth.cummax()
    drawdown = (wealth - running_max) / running_max  # 非正数序列
    return float(drawdown.min())
