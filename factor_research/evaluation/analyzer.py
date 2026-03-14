"""
因子综合分析器

评价体系的第一层 API——一键出报告。

整合所有第二层评价模块，提供一键生成完整因子评价报告的功能。
适用于 notebook 中快速评估和批量管道中自动评价。

用法:
    from factor_research.evaluation.analyzer import FactorAnalyzer

    analyzer = FactorAnalyzer(
        factor_panel,              # 因子面板 (timestamp × symbol)
        prices,                    # 价格面板 (timestamp × symbol)
    )

    # 完整报告
    report = analyzer.full_report(horizons=[1, 5, 10, 30, 60])
    print(analyzer.summary_text())

    # 也可以单独调用某个维度
    ic_report = analyzer.ic_analysis(horizons=[1, 5, 10, 30, 60])
    quantile_report = analyzer.quantile_backtest(n_groups=5)

依赖: evaluation 下的所有子模块
"""

import pandas as pd

from data_infra.utils.logger import get_logger

from ..config import DEFAULT_HORIZONS, DEFAULT_N_GROUPS, DEFAULT_TAIL_THRESHOLD
from .ic import ic_analysis
from .nonlinear import nonlinear_analysis
from .quantile import quantile_backtest
from .report import format_report_text, plot_report
from .stability import stability_analysis
from .tail import tail_analysis
from .turnover import turnover_analysis

logger = get_logger(__name__)


class FactorAnalyzer:
    """
    因子综合分析器

    封装因子面板和价格面板，提供统一的评价入口。
    所有评价方法都返回结构化字典，便于进一步处理。
    """

    def __init__(
        self,
        factor_panel: pd.DataFrame,
        prices: pd.DataFrame,
        symbols: list[str] = None,
    ):
        """
        初始化分析器

        Args:
            factor_panel: 因子面板，index=DatetimeIndex (UTC), columns=symbols
            prices:       价格面板（收盘价），格式同上
            symbols:      分析的标的范围，默认使用面板中全部列

        Raises:
            ValueError: 如果面板为空或没有共同标的
        """
        if factor_panel.empty:
            raise ValueError("因子面板为空")
        if prices.empty:
            raise ValueError("价格面板为空")

        # 确定分析的标的范围
        if symbols is not None:
            common_cols = [s for s in symbols
                          if s in factor_panel.columns and s in prices.columns]
        else:
            common_cols = list(
                factor_panel.columns.intersection(prices.columns)
            )

        if not common_cols:
            raise ValueError(
                f"因子面板和价格面板没有共同标的。"
                f"因子列: {list(factor_panel.columns)}，"
                f"价格列: {list(prices.columns)}"
            )

        self._factor = factor_panel[common_cols]
        self._prices = prices[common_cols]
        self._symbols = common_cols
        self._report_cache = None

        logger.debug(
            f"FactorAnalyzer 已初始化: {len(common_cols)} 标的, "
            f"{len(factor_panel)} 行因子, {len(prices)} 行价格"
        )

    def full_report(
        self,
        horizons: list[int] = None,
        n_groups: int = DEFAULT_N_GROUPS,
        tail_threshold: float = DEFAULT_TAIL_THRESHOLD,
    ) -> dict:
        """
        生成完整评价报告

        一键调用所有评价维度，返回汇总结果字典。

        Args:
            horizons:       前瞻窗口列表（单位: bar），默认 [1, 5, 10, 30, 60]
            n_groups:       分层组数，默认 5
            tail_threshold: 尾部阈值分位数，默认 0.9

        Returns:
            dict: {
                "ic":        IC/IR/衰减分析结果,
                "quantile":  分层回测结果,
                "tail":      尾部特征分析结果,
                "stability": 稳定性分析结果,
                "nonlinear": 非线性分析结果,
                "turnover":  换手率分析结果,
            }
        """
        if horizons is None:
            horizons = DEFAULT_HORIZONS

        logger.info(f"开始因子评价: {len(self._symbols)} 标的, 前瞻窗口 {horizons}")

        report = {
            "ic": self.ic_analysis(horizons=horizons),
            "quantile": self.quantile_backtest(n_groups=n_groups, horizon=horizons[0]),
            "tail": self.tail_analysis(threshold=tail_threshold, horizon=horizons[0]),
            "stability": self.stability_analysis(horizon=horizons[0]),
            "nonlinear": self.nonlinear_analysis(horizon=horizons[0]),
            "turnover": self.turnover_analysis(),
        }

        self._report_cache = report
        logger.info("因子评价完成")
        return report

    def ic_analysis(self, horizons: list[int] = None) -> dict:
        """IC/IR/衰减分析"""
        return ic_analysis(self._factor, self._prices, horizons=horizons)

    def quantile_backtest(self, n_groups: int = DEFAULT_N_GROUPS, horizon: int = 1) -> dict:
        """分层回测"""
        return quantile_backtest(
            self._factor, self._prices, n_groups=n_groups, horizon=horizon
        )

    def tail_analysis(self, threshold: float = DEFAULT_TAIL_THRESHOLD, horizon: int = 1) -> dict:
        """尾部特征分析"""
        return tail_analysis(
            self._factor, self._prices, threshold=threshold, horizon=horizon
        )

    def stability_analysis(self, horizon: int = 1) -> dict:
        """稳定性分析"""
        return stability_analysis(self._factor, self._prices, horizon=horizon)

    def nonlinear_analysis(self, horizon: int = 1) -> dict:
        """非线性分析"""
        return nonlinear_analysis(self._factor, self._prices, horizon=horizon)

    def turnover_analysis(self) -> dict:
        """换手率分析"""
        return turnover_analysis(self._factor)

    def summary_text(self, factor_name: str = "") -> str:
        """
        生成文本摘要

        如果还没有运行 full_report()，会先运行。

        Args:
            factor_name: 因子名称

        Returns:
            str: 格式化的文本报告
        """
        if self._report_cache is None:
            self.full_report()
        return format_report_text(self._report_cache, factor_name)

    def plot(self, factor_name: str = "") -> dict:
        """
        生成可视化图表

        Args:
            factor_name: 因子名称

        Returns:
            dict: {图表名: matplotlib.figure.Figure}
        """
        if self._report_cache is None:
            self.full_report()
        return plot_report(self._report_cache, factor_name)
