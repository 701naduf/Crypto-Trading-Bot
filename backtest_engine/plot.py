"""
plot.py — 8 张标准量化回测图

详见 docs/phase3_design.md §11.9。

纯函数模块，每个图一个函数；BacktestReport.plot() 是 plot_all 的薄包装。

API 设计（§11.9.3）:
  - 每个 plot 函数签名一致：plot_xxx(report, *, ax=None, **kwargs) → Figure
  - ax=None：创建新 fig + ax；返回 fig
  - ax 传入：在 ax 上画；返回 ax.figure（用于 plot_all 拼装）

边界处理（§11.9.5）:
  - regime_stats / deviation 为 None 时调用单图函数 → ValueError
  - bankruptcy_timestamp != None：equity / drawdown 在 t* 处加红色竖线
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    from backtest_engine.report import BacktestReport


# Step 14 / C4: 配置 matplotlib CJK 字体 fallback，避免 deviation_attribution 等
# 含中文标签的图渲染成方框（DejaVu Sans 缺 CJK glyph 时 matplotlib 默认 missing → 方框 + warning）
def _configure_cjk_font_fallback() -> None:
    try:
        import matplotlib
        # 用首选 CJK 字体覆盖默认；Windows / macOS / Linux 常见安装字体均覆盖
        matplotlib.rcParams['font.sans-serif'] = [
            'Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'PingFang SC', 'DejaVu Sans',
        ]
        matplotlib.rcParams['axes.unicode_minus'] = False
    except ImportError:
        pass


_configure_cjk_font_fallback()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 内部 helpers
# ---------------------------------------------------------------------------

def _ensure_ax(ax, figsize=(8, 5)):
    """若 ax=None，创建新 fig+ax；否则用 ax.figure"""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def _mark_bankruptcy(ax, report: "BacktestReport") -> None:
    """破产时间戳红色竖线"""
    if report.bankruptcy_timestamp is not None:
        ax.axvline(
            report.bankruptcy_timestamp, color="red", linestyle="--",
            linewidth=1.0, label="bankruptcy", alpha=0.8,
        )


# ---------------------------------------------------------------------------
# 8 张图
# ---------------------------------------------------------------------------

def plot_equity_curve(
    report: "BacktestReport", *,
    ax: "matplotlib.axes.Axes | None" = None, **kwargs,
) -> "matplotlib.figure.Figure":
    """净值曲线 + 高水位线"""
    fig, ax = _ensure_ax(ax)
    eq = report.base.equity_curve
    if eq.isna().all():
        ax.text(0.5, 0.5, "Equity curve all NaN", ha="center", transform=ax.transAxes)
        ax.axis("off")
        return fig

    ax.plot(eq.index, eq.values, color="C0", linewidth=1.0, label="Equity")
    high_water = eq.cummax()
    ax.plot(eq.index, high_water.values, color="C1", linestyle="--",
            linewidth=0.8, alpha=0.6, label="High water")
    _mark_bankruptcy(ax, report)
    ax.set_title("Equity Curve")
    ax.set_ylabel("Portfolio Value")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return fig


def plot_drawdown(
    report: "BacktestReport", *,
    ax: "matplotlib.axes.Axes | None" = None, **kwargs,
) -> "matplotlib.figure.Figure":
    """回撤曲线（填充式 area）"""
    fig, ax = _ensure_ax(ax)
    eq = report.base.equity_curve
    if eq.isna().all():
        ax.text(0.5, 0.5, "Equity curve all NaN", ha="center", transform=ax.transAxes)
        ax.axis("off")
        return fig

    running_max = eq.cummax()
    drawdown = eq / running_max - 1.0
    ax.fill_between(drawdown.index, drawdown.values, 0, color="C3", alpha=0.4)
    ax.plot(drawdown.index, drawdown.values, color="C3", linewidth=0.8)
    _mark_bankruptcy(ax, report)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    return fig


def plot_returns_distribution(
    report: "BacktestReport", *,
    ax: "matplotlib.axes.Axes | None" = None,
    bins: int = 50, **kwargs,
) -> "matplotlib.figure.Figure":
    """收益直方图 + 正态拟合 + VaR/CVaR 标记"""
    fig, ax = _ensure_ax(ax)
    rets = report.base.returns.dropna()
    if len(rets) == 0:
        ax.text(0.5, 0.5, "No returns", ha="center", transform=ax.transAxes)
        ax.axis("off")
        return fig

    ax.hist(rets.values, bins=bins, density=True, color="C0", alpha=0.6,
            label="Empirical")

    # 正态拟合
    mu, sigma = float(rets.mean()), float(rets.std(ddof=0))
    if sigma > 0:
        x = np.linspace(rets.min(), rets.max(), 200)
        pdf = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        ax.plot(x, pdf, color="C1", linewidth=1.0, label=f"N(μ={mu:.4f},σ={sigma:.4f})")

    # VaR / CVaR 5%
    var_5 = float(rets.quantile(0.05))
    cvar_5 = float(rets[rets <= var_5].mean()) if (rets <= var_5).any() else var_5
    ax.axvline(var_5, color="C2", linestyle="--", linewidth=0.8, label=f"VaR 5%={var_5:.4f}")
    ax.axvline(cvar_5, color="C3", linestyle="--", linewidth=0.8, label=f"CVaR 5%={cvar_5:.4f}")

    ax.set_title("Returns Distribution")
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig


def plot_cost_breakdown(
    report: "BacktestReport", *,
    ax: "matplotlib.axes.Axes | None" = None, **kwargs,
) -> "matplotlib.figure.Figure":
    """柱状图（fee/spread/impact/funding bp）"""
    fig, ax = _ensure_ax(ax)
    bp = report.cost_breakdown.get("annualized_bp", {})
    components = ["fee", "spread", "impact", "funding"]
    values = [float(bp.get(k, 0.0)) for k in components]
    colors = ["C0", "C1", "C2", "C3"]
    ax.bar(components, values, color=colors)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.1f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Cost Breakdown (annualized bp)")
    ax.set_ylabel("bp")
    ax.grid(True, alpha=0.3, axis="y")
    return fig


def plot_weights_history(
    report: "BacktestReport", *,
    ax: "matplotlib.axes.Axes | None" = None,
    cmap: str = "RdBu_r", **kwargs,
) -> "matplotlib.figure.Figure":
    """heatmap (timestamp × symbol)"""
    fig, ax = _ensure_ax(ax, figsize=(10, 5))
    wh = report.base.weights_history
    if wh is None or wh.empty:
        ax.text(0.5, 0.5, "No weights", ha="center", transform=ax.transAxes)
        ax.axis("off")
        return fig

    # 转置：行=symbol, 列=timestamp
    arr = wh.T.values
    vmax = max(abs(arr.min()), abs(arr.max()), 1e-8)
    im = ax.imshow(
        arr, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
        extent=[0, len(wh), -0.5, len(wh.columns) - 0.5],
        origin="lower",
    )
    fig.colorbar(im, ax=ax, label="Weight")

    # symbol 标签
    cols = list(wh.columns)
    if len(cols) <= 30:
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(cols)
    else:
        ax.set_yticks([])
    ax.set_title("Weights History (timestamp × symbol)")
    ax.set_xlabel("bar index")
    return fig


def plot_regime_stats(
    report: "BacktestReport", *,
    ax: "matplotlib.axes.Axes | None" = None, **kwargs,
) -> "matplotlib.figure.Figure":
    """分组柱状图（regime × 6 指标）"""
    if report.regime_stats is None:
        raise ValueError("BacktestReport.regime_stats is None；无 regime 数据可绘")

    fig, ax = _ensure_ax(ax, figsize=(10, 5))
    df = report.regime_stats
    metrics = ["total_return", "sharpe", "max_drawdown", "turnover_mean", "cost_rate_bp"]
    available = [m for m in metrics if m in df.columns]
    if not available:
        ax.text(0.5, 0.5, "No metrics in regime_stats", ha="center", transform=ax.transAxes)
        ax.axis("off")
        return fig

    n_metrics = len(available)
    n_regimes = len(df)
    width = 0.8 / n_metrics

    for i, m in enumerate(available):
        offsets = np.arange(n_regimes) + i * width - 0.4 + width / 2
        ax.bar(offsets, df[m].values, width=width, label=m)

    ax.set_xticks(range(n_regimes))
    ax.set_xticklabels([str(x) for x in df.index])
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Regime Stats")
    ax.grid(True, alpha=0.3, axis="y")
    return fig


def plot_rolling_sharpe(
    report: "BacktestReport", *,
    ax: "matplotlib.axes.Axes | None" = None,
    window: int | None = None, **kwargs,
) -> "matplotlib.figure.Figure":
    """滚动 Sharpe（默认窗口 = 21 日 / bar_freq）"""
    fig, ax = _ensure_ax(ax)
    rets = report.base.returns.dropna()
    if len(rets) < 30:
        ax.text(0.5, 0.5, "Not enough data for rolling Sharpe", ha="center",
                transform=ax.transAxes)
        ax.axis("off")
        return fig

    ppy = report.config.periods_per_year
    bars_per_day = ppy / 365.25
    if window is None:
        window = max(int(round(21 * bars_per_day)), 30)
    window = min(window, len(rets) // 2)

    rolling_mean = rets.rolling(window).mean()
    rolling_std = rets.rolling(window).std(ddof=0)
    rolling_sharpe = rolling_mean / rolling_std.replace(0, np.nan) * np.sqrt(ppy)

    ax.plot(rolling_sharpe.index, rolling_sharpe.values, color="C0", linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.5)
    _mark_bankruptcy(ax, report)
    ax.set_title(f"Rolling Sharpe (window={window} bars)")
    ax.set_ylabel("Sharpe")
    ax.grid(True, alpha=0.3)
    return fig


def plot_deviation_attribution(
    report: "BacktestReport", *,
    ax: "matplotlib.axes.Axes | None" = None, **kwargs,
) -> "matplotlib.figure.Figure":
    """瀑布图：各偏差源贡献 + 总差"""
    if report.deviation is None:
        raise ValueError("BacktestReport.deviation is None；先调 attach_deviation")

    fig, ax = _ensure_ax(ax, figsize=(10, 5))
    df = report.deviation

    sources = list(df["bias_source"])
    deltas = df["delta_terminal_return"].values
    quantified = df["quantified"].values

    colors = []
    for q, d in zip(quantified, deltas):
        if not q or pd.isna(d):
            colors.append("lightgray")
        elif d >= 0:
            colors.append("C2")
        else:
            colors.append("C3")

    ax.barh(range(len(sources)), [0.0 if pd.isna(d) else d for d in deltas],
            color=colors, alpha=0.8)
    ax.set_yticks(range(len(sources)))
    ax.set_yticklabels(sources, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Δ Terminal Return")
    ax.set_title("Deviation Attribution (vs vectorized baseline)")
    ax.grid(True, alpha=0.3, axis="x")
    return fig


# ---------------------------------------------------------------------------
# plot_all（§11.9.4）
# ---------------------------------------------------------------------------

def plot_all(
    report: "BacktestReport", *,
    figsize: tuple[float, float] = (16, 10),
) -> "matplotlib.figure.Figure":
    """
    一键全图（2 × 4 subplot）

    布局:
      Row 0: equity / drawdown / returns_dist / cost_bp
      Row 1: rolling_sharpe / weights_heatmap (跨 2 列) / regime 或 deviation
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.3)

    # Row 0
    ax_eq      = fig.add_subplot(gs[0, 0])
    ax_dd      = fig.add_subplot(gs[0, 1])
    ax_ret     = fig.add_subplot(gs[0, 2])
    ax_cost    = fig.add_subplot(gs[0, 3])

    # Row 1
    ax_sharpe  = fig.add_subplot(gs[1, 0])
    ax_weights = fig.add_subplot(gs[1, 1:3])
    ax_extra   = fig.add_subplot(gs[1, 3])

    plot_equity_curve(report, ax=ax_eq)
    plot_drawdown(report, ax=ax_dd)
    plot_returns_distribution(report, ax=ax_ret)
    plot_cost_breakdown(report, ax=ax_cost)

    plot_rolling_sharpe(report, ax=ax_sharpe)
    plot_weights_history(report, ax=ax_weights)

    if report.regime_stats is not None:
        plot_regime_stats(report, ax=ax_extra)
    elif report.deviation is not None:
        plot_deviation_attribution(report, ax=ax_extra)
    else:
        ax_extra.text(0.5, 0.5, "No regime / deviation data",
                      ha="center", transform=ax_extra.transAxes)
        ax_extra.axis("off")

    title = (
        f"{report.config.strategy_name} | "
        f"{report.run_metadata.get('run_mode', '?')}"
    )
    fig.suptitle(title, fontsize=12, y=0.995)
    return fig
