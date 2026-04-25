"""
reporting.py — Markdown 单文件报告生成

详见 docs/phase3_design.md §11.10。

把 BacktestReport 渲染为 Markdown + 配套 figures/ 子目录：
  output_dir/
    report.md
    figures/*.png
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from backtest_engine.report import BacktestReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 公开入口
# ---------------------------------------------------------------------------

def to_markdown(
    report: "BacktestReport",
    output_dir: str | Path,
    *,
    figure_dpi: int = 100,
    overwrite: bool = False,
) -> Path:
    """
    生成 Markdown 报告 + 配套 figures/

    Args:
        report:      BacktestReport
        output_dir:  输出目录（不存在则创建；存在且 overwrite=False → FileExistsError）
        figure_dpi:  图片 dpi
        overwrite:   True 则删除旧 figures/ 后重新生成

    Returns:
        生成的 report.md 路径
    """
    # headless backend（防 CI/CD 无 X server）
    import matplotlib
    matplotlib.use("Agg", force=False)

    # Step 14 / C4: 配置 CJK 字体 fallback（防中文 glyph missing 渲染成方框）
    matplotlib.rcParams['font.sans-serif'] = [
        'Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'PingFang SC', 'DejaVu Sans',
    ]
    matplotlib.rcParams['axes.unicode_minus'] = False

    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"

    if output_dir.exists():
        if not overwrite:
            # 仅在 figures/ 存在或 report.md 存在时才视为已有报告
            if (output_dir / "report.md").exists() or figures_dir.exists():
                raise FileExistsError(
                    f"output_dir 已含 report.md 或 figures/：{output_dir}（"
                    f"显式传 overwrite=True）"
                )
        else:
            if figures_dir.exists():
                import shutil
                shutil.rmtree(figures_dir)

    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. 生成图片
    fig_paths = _save_all_figures(report, figures_dir, dpi=figure_dpi)

    # 2. 渲染 markdown
    md_text = _render_markdown(report, fig_paths)

    md_path = output_dir / "report.md"
    md_path.write_text(md_text, encoding="utf-8")
    logger.info("Markdown 报告已生成: %s", md_path)
    return md_path


# ---------------------------------------------------------------------------
# 内部：图片保存
# ---------------------------------------------------------------------------

def _save_all_figures(
    report: "BacktestReport", figures_dir: Path, dpi: int,
) -> dict[str, Path]:
    """按 report 字段决定哪些图跑；保存到 figures_dir，返回 {name: relative_path}"""
    import matplotlib.pyplot as plt
    from backtest_engine import plot

    fig_paths: dict[str, Path] = {}
    plot_jobs = [
        ("equity_curve", plot.plot_equity_curve, True),
        ("drawdown", plot.plot_drawdown, True),
        ("returns_distribution", plot.plot_returns_distribution, True),
        ("cost_breakdown", plot.plot_cost_breakdown, True),
        ("weights_history", plot.plot_weights_history, True),
        ("rolling_sharpe", plot.plot_rolling_sharpe, True),
        ("regime_stats", plot.plot_regime_stats, report.regime_stats is not None),
        ("deviation_attribution", plot.plot_deviation_attribution,
         report.deviation is not None),
    ]
    for name, fn, should_run in plot_jobs:
        if not should_run:
            continue
        try:
            fig = fn(report)
            png_path = figures_dir / f"{name}.png"
            fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            fig_paths[name] = png_path
        except Exception as e:
            logger.warning("plot %s 失败：%s（跳过）", name, e)

    return fig_paths


# ---------------------------------------------------------------------------
# 内部：markdown 渲染
# ---------------------------------------------------------------------------

def _render_markdown(report: "BacktestReport", fig_paths: dict[str, Path]) -> str:
    """拼字符串生成完整 markdown"""
    s = report.summary()
    cfg = report.config
    meta = report.run_metadata

    lines: list[str] = []

    # ── Header ──
    lines.append(f"# Backtest Report: {cfg.strategy_name}")
    lines.append("")
    lines.append(
        f"**Run mode**: {meta.get('run_mode', '?')} | "
        f"**Period**: {cfg.start} → {cfg.end} | "
        f"**Walltime**: {meta.get('walltime_seconds', 0):.1f}s"
    )
    lines.append("")

    # 破产 warning
    if report.bankruptcy_timestamp is not None:
        lines.append(
            f"> ⚠️ **Bankruptcy at {report.bankruptcy_timestamp}, only first "
            f"{meta.get('n_bars', '?')} bars completed**"
        )
        lines.append("")

    # ── Performance Summary ──
    lines.append("## Performance Summary")
    lines.append("")
    lines.append(_render_summary_table(s))
    lines.append("")
    if "equity_curve" in fig_paths:
        lines.append(f"![Equity Curve](figures/equity_curve.png)")
    lines.append("")
    if "drawdown" in fig_paths:
        lines.append(f"![Drawdown](figures/drawdown.png)")
    lines.append("")

    # ── Cost Breakdown ──
    lines.append("## Cost Breakdown")
    lines.append("")
    lines.append(_render_cost_breakdown_table(report.cost_breakdown))
    lines.append("")
    is_vec = meta.get("run_mode") == "vectorized"
    if not is_vec:
        lines.append(
            "⚠️ Funding 是合约持仓成本，不是执行摩擦。负值表示净收款。"
        )
        lines.append("")
    if "cost_breakdown" in fig_paths:
        lines.append(f"![Cost Breakdown](figures/cost_breakdown.png)")
    lines.append("")

    # ── Returns Distribution ──
    lines.append("## Returns Distribution")
    lines.append("")
    if "returns_distribution" in fig_paths:
        lines.append(f"![Returns Distribution](figures/returns_distribution.png)")
    lines.append("")

    # ── Position Evolution ──
    lines.append("## Position Evolution")
    lines.append("")
    if "weights_history" in fig_paths:
        lines.append(f"![Weights History](figures/weights_history.png)")
    lines.append("")

    # ── Rolling Sharpe ──
    lines.append("## Rolling Sharpe")
    lines.append("")
    if "rolling_sharpe" in fig_paths:
        lines.append(f"![Rolling Sharpe](figures/rolling_sharpe.png)")
    lines.append("")

    # ── Regime Analysis（可选）──
    if report.regime_stats is not None:
        lines.append("## Regime Analysis")
        lines.append("")
        lines.append(_render_regime_table(report.regime_stats))
        lines.append("")
        if "regime_stats" in fig_paths:
            lines.append(f"![Regime Stats](figures/regime_stats.png)")
        lines.append("")

    # ── Deviation Attribution（可选）──
    if report.deviation is not None:
        lines.append("## Deviation Attribution")
        lines.append("")
        lines.append(_render_deviation_table(report.deviation))
        lines.append("")
        if "deviation_attribution" in fig_paths:
            lines.append(f"![Deviation Attribution](figures/deviation_attribution.png)")
        lines.append("")

    # ── Configuration ──
    lines.append("## Configuration")
    lines.append("")
    lines.append("```json")
    from backtest_engine.report import _serialize_config
    lines.append(json.dumps(_serialize_config(cfg), indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")

    # ── Run Metadata ──
    lines.append("## Run Metadata")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    for k, v in meta.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 表格渲染辅助
# ---------------------------------------------------------------------------

def _render_summary_table(summary: dict) -> str:
    """生成 markdown 表格（Performance Summary）"""
    rows = [
        ("Total return", _fmt_pct(summary.get("total_return"))),
        ("Annual return", _fmt_pct(summary.get("annual_return"))),
        ("Sharpe ratio", _fmt_num(summary.get("sharpe_ratio"))),
        ("Sortino ratio", _fmt_num(summary.get("sortino_ratio"))),
        ("Calmar ratio", _fmt_num(summary.get("calmar_ratio"))),
        ("Max drawdown", _fmt_pct(summary.get("max_drawdown"))),
        ("Max drawdown duration", f"{summary.get('max_drawdown_duration', 0)} bars"),
        ("Avg turnover", _fmt_num(summary.get("avg_turnover"))),
        ("Win rate", _fmt_pct(summary.get("win_rate"))),
        (
            "Bankruptcy",
            "Yes" if summary.get("bankruptcy_flag") else "No",
        ),
    ]
    out = ["| Metric | Value |", "|--------|-------|"]
    for k, v in rows:
        out.append(f"| {k} | {v} |")
    return "\n".join(out)


def _render_cost_breakdown_table(cb: dict) -> str:
    bp = cb.get("annualized_bp", {})
    sh = cb.get("share", {})
    rows = [
        ("Fee", bp.get("fee", 0.0), sh.get("fee", 0.0)),
        ("Spread", bp.get("spread", 0.0), sh.get("spread", 0.0)),
        ("Impact", bp.get("impact", 0.0), sh.get("impact", 0.0)),
        ("Funding", bp.get("funding", 0.0), sh.get("funding", 0.0)),
    ]
    out = ["| Component | Annualized bp | Share |", "|-----------|--------------|-------|"]
    for k, v, sv in rows:
        out.append(f"| {k} | {v:.1f} | {_fmt_pct(sv)} |")
    out.append(f"| **Total** | **{bp.get('total', 0.0):.1f}** | 100% |")
    return "\n".join(out)


def _render_regime_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(no regime data)_"
    cols = ["n_bars", "total_return", "sharpe", "max_drawdown", "turnover_mean", "cost_rate_bp"]
    available = [c for c in cols if c in df.columns]
    header = "| Regime | " + " | ".join(available) + " |"
    sep = "|" + "|".join(["---"] * (len(available) + 1)) + "|"
    out = [header, sep]
    for regime, row in df.iterrows():
        cells = [str(regime)]
        for c in available:
            v = row[c]
            if c in ("total_return", "max_drawdown"):
                cells.append(_fmt_pct(v))
            else:
                cells.append(_fmt_num(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _render_deviation_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(no deviation data)_"
    out = [
        "| Bias source | Δ Terminal return | Δ Sharpe | Quantified | Method | Note |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in df.iterrows():
        out.append(
            f"| {row['bias_source']} "
            f"| {_fmt_pct(row.get('delta_terminal_return'))} "
            f"| {_fmt_num(row.get('delta_sharpe'))} "
            f"| {row['quantified']} "
            f"| {row['method']} "
            f"| {row.get('note', '')} |"
        )
    return "\n".join(out)


def _fmt_pct(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v:+.2%}"


def _fmt_num(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v:.4f}"
