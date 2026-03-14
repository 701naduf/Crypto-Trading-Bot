"""
因子族参数敏感性分析器

评价体系的补充组件——参数空间探索。

FamilyAnalyzer 与 FactorAnalyzer 的关系:
    - FactorAnalyzer: 单因子 × 多 horizon × 多维度 → 完整深度报告
    - FamilyAnalyzer:  多参数 × 多 horizon × 少量核心指标 → 全景扫描

两者互补: FamilyAnalyzer 做广度探索（定位最优参数区域），
FactorAnalyzer 做深度评价（对选中参数做完整 6 维度分析）。

工作流:
    1. sweep()           — 遍历参数网格，计算轻量指标 → 扫描表
    2. plot_sensitivity() — 参数敏感性折线图
    3. plot_heatmap()     — 双参数热力图
    4. select()           — 按阈值自动筛选候选参数
    5. robustness()       — 参数稳健性评分
    6. detail()           — 对选中参数调用 FactorAnalyzer 做完整评价

依赖: evaluation.ic, evaluation.turnover, evaluation.quantile, evaluation.analyzer
"""

import itertools

import numpy as np
import pandas as pd

from data_infra.utils.logger import get_logger

from ..config import DEFAULT_HORIZONS
from .ic import ic_series, ic_summary
from .turnover import turnover_analysis
from .quantile import quantile_backtest
from .analyzer import FactorAnalyzer

logger = get_logger(__name__)


class FamilyAnalyzer:
    """
    因子族参数敏感性分析器

    对参数化因子族进行参数空间扫描，通过轻量指标和可视化
    帮助研究者快速定位最优参数区域，再对候选因子做详细评价。
    """

    def __init__(
        self,
        factor_class: type,
        data: dict,
        price_panel: pd.DataFrame,
        param_grid: dict = None,
        horizons: list[int] = None,
    ):
        """
        初始化 FamilyAnalyzer

        Args:
            factor_class: 参数化因子类（必须有参数化 __init__）
            data:         原始数据字典，格式与引擎 _prepare_data 输出一致。
                          TimeSeriesFactor: {symbol: {DataType: DataFrame}}
                          CrossSectionalFactor: {DataType: {symbol: DataFrame}}
                          传入后在 sweep 中复用，避免重复读取。
            price_panel:  价格面板 (timestamp × symbol)，用于 IC 评价
            param_grid:   参数网格 {参数名: [值列表]}。
                          默认 None → 使用 factor_class._param_grid。
                          可手动覆盖以做更精细的探索。
            horizons:     评价用的前瞻窗口列表，默认 DEFAULT_HORIZONS

        Raises:
            ValueError: 如果因子类没有 _param_grid 且未传入 param_grid
        """
        self._factor_class = factor_class
        self._data = data
        self._price_panel = price_panel

        # 确定参数网格
        if param_grid is not None:
            self._param_grid = param_grid
        elif hasattr(factor_class, "_param_grid") and factor_class._param_grid:
            self._param_grid = factor_class._param_grid
        else:
            raise ValueError(
                f"{factor_class.__name__} 没有 _param_grid 属性，"
                "且未传入 param_grid 参数"
            )

        self._horizons = horizons or DEFAULT_HORIZONS
        self._param_keys = list(self._param_grid.keys())

        # sweep 结果缓存
        self._sweep_df: pd.DataFrame = None

    def sweep(self, metrics: list[str] = None) -> pd.DataFrame:
        """
        遍历参数网格 × 前瞻窗口，计算轻量指标

        默认指标:
            - ic_mean:          IC 均值（因子预测力）
            - ic_ir:            IC 信息比率（预测力 / 波动）
            - ic_positive_pct:  IC 为正的比例（胜率）
            - turnover_autocorr: 因子自相关（换手成本代理）
            - monotonicity:     分层单调性（截面排序能力）

        Args:
            metrics: 指定要计算的指标列表，默认全部

        Returns:
            pd.DataFrame，每行一个 (参数组合, horizon)。
            列 = 参数名列表 + ["horizon"] + 指标名列表。
        """
        all_metrics = [
            "ic_mean", "ic_ir", "ic_positive_pct",
            "turnover_autocorr", "monotonicity",
        ]
        if metrics is not None:
            all_metrics = [m for m in all_metrics if m in metrics]

        # 笛卡尔积展开参数
        value_lists = [self._param_grid[k] for k in self._param_keys]
        combos = list(itertools.product(*value_lists))

        results = []

        for combo in combos:
            params = dict(zip(self._param_keys, combo))

            # 实例化因子并计算面板
            try:
                factor = self._factor_class(**params)
                panel = factor.compute(self._data)
            except Exception as e:
                logger.warning(f"参数组合 {params} 计算失败: {e}")
                continue

            if panel.empty:
                logger.warning(f"参数组合 {params} 输出为空面板")
                continue

            # 换手率分析（与 horizon 无关，只算一次）
            turnover = turnover_analysis(panel)
            turnover_autocorr = turnover.get("autocorrelation", np.nan)

            # 对每个 horizon 计算 IC 相关指标
            for h in self._horizons:
                row = {**params, "horizon": h}

                # IC 指标
                if any(m in all_metrics for m in ["ic_mean", "ic_ir", "ic_positive_pct"]):
                    ic_ts = ic_series(panel, self._price_panel, horizon=h)
                    ic_stats = ic_summary(ic_ts)
                    if "ic_mean" in all_metrics:
                        row["ic_mean"] = ic_stats["ic_mean"]
                    if "ic_ir" in all_metrics:
                        row["ic_ir"] = ic_stats["ic_ir"]
                    if "ic_positive_pct" in all_metrics:
                        row["ic_positive_pct"] = ic_stats["ic_win_rate"]

                # 换手自相关
                if "turnover_autocorr" in all_metrics:
                    row["turnover_autocorr"] = turnover_autocorr

                # 单调性
                if "monotonicity" in all_metrics:
                    try:
                        qb = quantile_backtest(panel, self._price_panel, horizon=h)
                        row["monotonicity"] = qb.get("monotonicity", np.nan)
                    except Exception:
                        row["monotonicity"] = np.nan

                results.append(row)

        self._sweep_df = pd.DataFrame(results)
        logger.info(
            f"参数扫描完成: {len(combos)} 组参数 × {len(self._horizons)} 个 horizon "
            f"→ {len(results)} 行结果"
        )
        return self._sweep_df

    def plot_sensitivity(
        self,
        metric: str = "ic_mean",
        param: str = None,
        horizons: list[int] = None,
    ):
        """
        参数敏感性折线图

        单参数因子: X 轴 = 参数值，每条线 = 一个 horizon
        多参数因子: 需指定 param，其他参数取第一个值

        Args:
            metric:   要绘制的指标名（如 "ic_mean", "ic_ir"）
            param:    X 轴参数名，单参数因子可省略
            horizons: 要显示的 horizon 列表，默认全部

        Returns:
            matplotlib.figure.Figure

        Raises:
            ValueError: 如果尚未运行 sweep()
        """
        import matplotlib.pyplot as plt

        if self._sweep_df is None:
            raise ValueError("请先运行 sweep() 生成扫描数据")

        df = self._sweep_df.copy()

        # 确定 X 轴参数
        if param is None:
            if len(self._param_keys) == 1:
                param = self._param_keys[0]
            else:
                param = self._param_keys[0]
                # 多参数: 固定其他参数为第一个值
                for k in self._param_keys:
                    if k != param:
                        first_val = self._param_grid[k][0]
                        df = df[df[k] == first_val]

        if horizons is not None:
            df = df[df["horizon"].isin(horizons)]

        fig, ax = plt.subplots(figsize=(10, 6))
        for h, group in df.groupby("horizon"):
            group_sorted = group.sort_values(param)
            ax.plot(
                group_sorted[param].astype(str),
                group_sorted[metric],
                marker="o",
                label=f"h={h}",
            )

        ax.set_xlabel(param)
        ax.set_ylabel(metric)
        ax.set_title(f"{self._factor_class.__name__}: {metric} vs {param}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_heatmap(
        self,
        metric: str = "ic_ir",
        horizon: int = 1,
        x_param: str = None,
        y_param: str = None,
    ):
        """
        双参数热力图

        X 轴 = 第一个参数，Y 轴 = 第二个参数，颜色 = 指标值。
        只有因子有 2 个参数时可用；1 个参数时退化为柱状图。

        Args:
            metric:  指标名
            horizon: 只显示指定 horizon 的数据
            x_param: X 轴参数名，默认第一个参数
            y_param: Y 轴参数名，默认第二个参数

        Returns:
            matplotlib.figure.Figure

        Raises:
            ValueError: 如果尚未运行 sweep() 或参数不足
        """
        import matplotlib.pyplot as plt

        if self._sweep_df is None:
            raise ValueError("请先运行 sweep() 生成扫描数据")

        df = self._sweep_df[self._sweep_df["horizon"] == horizon].copy()

        if len(self._param_keys) < 2:
            # 单参数: 退化为柱状图
            param = self._param_keys[0]
            df_sorted = df.sort_values(param)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(df_sorted[param].astype(str), df_sorted[metric])
            ax.set_xlabel(param)
            ax.set_ylabel(metric)
            ax.set_title(f"{self._factor_class.__name__}: {metric} (h={horizon})")
            fig.tight_layout()
            return fig

        x_param = x_param or self._param_keys[0]
        y_param = y_param or self._param_keys[1]

        # 构建透视表
        pivot = df.pivot_table(
            values=metric, index=y_param, columns=x_param, aggfunc="first"
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(v) for v in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(v) for v in pivot.index])
        ax.set_xlabel(x_param)
        ax.set_ylabel(y_param)
        ax.set_title(f"{self._factor_class.__name__}: {metric} (h={horizon})")
        fig.colorbar(im, ax=ax)

        # 标注数值
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}",
                            ha="center", va="center", fontsize=8)

        fig.tight_layout()
        return fig

    def select(
        self,
        min_ic_mean: float = None,
        min_ic_ir: float = None,
        max_turnover_autocorr: float = None,
        min_monotonicity: float = None,
        horizon: int = None,
        top_n: int = None,
        sort_by: str = "ic_ir",
    ) -> pd.DataFrame:
        """
        从扫描结果中筛选符合条件的参数组合

        Args:
            min_ic_mean:           IC 均值下限
            min_ic_ir:             IC_IR 下限
            max_turnover_autocorr: 换手自相关上限（越高换手越低）
            min_monotonicity:      单调性下限
            horizon:               只看指定 horizon，None 则对所有 horizon 取均值
            top_n:                 返回排名前 N 的组合
            sort_by:               排序指标

        Returns:
            pd.DataFrame: 筛选后的扫描表子集，按 sort_by 降序

        Raises:
            ValueError: 如果尚未运行 sweep()
        """
        if self._sweep_df is None:
            raise ValueError("请先运行 sweep() 生成扫描数据")

        df = self._sweep_df.copy()

        # 按 horizon 筛选或聚合
        if horizon is not None:
            df = df[df["horizon"] == horizon]
        else:
            # 对所有 horizon 取均值（按参数组合分组）
            metric_cols = [c for c in df.columns
                           if c not in self._param_keys and c != "horizon"]
            df = df.groupby(self._param_keys, as_index=False)[metric_cols].mean()

        # 条件筛选
        if min_ic_mean is not None and "ic_mean" in df.columns:
            df = df[df["ic_mean"] >= min_ic_mean]
        if min_ic_ir is not None and "ic_ir" in df.columns:
            df = df[df["ic_ir"] >= min_ic_ir]
        if max_turnover_autocorr is not None and "turnover_autocorr" in df.columns:
            df = df[df["turnover_autocorr"] <= max_turnover_autocorr]
        if min_monotonicity is not None and "monotonicity" in df.columns:
            df = df[df["monotonicity"] >= min_monotonicity]

        # 排序
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=False)

        # Top N
        if top_n is not None:
            df = df.head(top_n)

        return df.reset_index(drop=True)

    def robustness(self, metric: str = "ic_ir", horizon: int = 1) -> pd.DataFrame:
        """
        参数稳健性评估

        对每个参数组合，计算其邻域内指标的变化幅度。
        变化小 = 稳健（平坦区域），变化大 = 脆弱（尖峰）。

        评分方法:
            对于数值参数: 相邻参数值之间指标变化的绝对值取均值
            对于非数值参数: 固定其他参数后，该参数不同取值间指标的标准差

        Args:
            metric:  指标名
            horizon: 只看指定 horizon

        Returns:
            pd.DataFrame: 参数组合 + robustness_score 列（越小越稳健）

        Raises:
            ValueError: 如果尚未运行 sweep()
        """
        if self._sweep_df is None:
            raise ValueError("请先运行 sweep() 生成扫描数据")

        df = self._sweep_df[self._sweep_df["horizon"] == horizon].copy()

        if df.empty or metric not in df.columns:
            return pd.DataFrame()

        # 单参数: 相邻差分的绝对值
        if len(self._param_keys) == 1:
            param = self._param_keys[0]
            df_sorted = df.sort_values(param).reset_index(drop=True)
            diffs = df_sorted[metric].diff().abs()

            # 每个点的稳健性 = 与前后邻居的平均绝对差
            scores = []
            for i in range(len(df_sorted)):
                neighbors = []
                if i > 0:
                    neighbors.append(diffs.iloc[i])
                if i < len(diffs) - 1:
                    neighbors.append(diffs.iloc[i + 1])
                scores.append(np.nanmean(neighbors) if neighbors else np.nan)

            result = df_sorted[self._param_keys + [metric]].copy()
            result["robustness_score"] = scores
            return result

        # 多参数: 在每个维度上计算标准差，取均值
        scores = []
        for _, row in df.iterrows():
            dim_scores = []
            for pk in self._param_keys:
                # 固定其他参数，变化当前参数
                mask = pd.Series(True, index=df.index)
                for ok in self._param_keys:
                    if ok != pk:
                        mask = mask & (df[ok] == row[ok])
                subset = df[mask][metric]
                if len(subset) > 1:
                    dim_scores.append(subset.std())
            scores.append(np.nanmean(dim_scores) if dim_scores else np.nan)

        result = df[self._param_keys + [metric]].copy()
        result["robustness_score"] = scores
        return result

    def detail(self, **params) -> dict:
        """
        对指定参数组合生成完整评价报告

        创建因子实例 → 计算面板 → 调用 FactorAnalyzer.full_report()

        Args:
            **params: 因子参数（如 lookback=10）

        Returns:
            dict: FactorAnalyzer.full_report() 的返回值
        """
        factor = self._factor_class(**params)
        panel = factor.compute(self._data)

        if panel.empty:
            raise ValueError(f"参数组合 {params} 的因子面板为空")

        analyzer = FactorAnalyzer(panel, self._price_panel)
        return analyzer.full_report(horizons=self._horizons)
