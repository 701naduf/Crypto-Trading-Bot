"""
BacktestReport — Phase 3 对外的统一交付对象

详见 docs/phase3_design.md §11.8。

字段：
  base                 — BacktestResult（Phase 2b 类型）
  config               — BacktestConfig 快照（reproducibility 必需）
  cost_breakdown       — §11.7 cost_decomposition 输出
  deviation            — §11.7 deviation_attribution 输出（engine 不填，用户 attach_deviation 填）
  regime_stats         — §11.7 regime_breakdown 输出
  funding_settlements  — dict 统计（n_events / total_rate / mean / first / last）
  bankruptcy_timestamp — None = 未破产
  run_metadata         — schema_version / run_mode / cost_mode / walltime / 等
  context_panels       — B3 跨机器分发支持（默认 None；save 时 user 显式传 context_builder）

持久化路线：parquet + JSON（不用 pickle，§11.8.6）—— v3 修订原子化写入（tmp_dir + os.replace）。
plot / to_markdown 是薄包装到 §11.9 / §11.10（lazy import 防潜在循环）。
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from alpha_model.backtest.performance import BacktestResult
from alpha_model.core.types import PortfolioConstraints

from backtest_engine.config import (
    BacktestConfig, RunMode, ExecutionMode, CostMode,
)

if TYPE_CHECKING:
    from backtest_engine.context import MarketContextBuilder

logger = logging.getLogger(__name__)


SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# BacktestReport dataclass
# ---------------------------------------------------------------------------

@dataclass
class BacktestReport:
    """统一交付对象（§11.8.3 字段规范）"""

    base: BacktestResult
    config: BacktestConfig
    cost_breakdown: dict[str, dict[str, float]]
    deviation: pd.DataFrame | None
    regime_stats: pd.DataFrame | None
    funding_settlements: dict | None
    bankruptcy_timestamp: pd.Timestamp | None
    run_metadata: dict
    context_panels: dict[str, pd.DataFrame] | None = None

    # ------------------------------------------------------------------
    # 摘要 / 字典化
    # ------------------------------------------------------------------

    def summary(self, *, periods_per_year: float | None = None) -> dict:
        """
        增强版 summary（§11.8.4）：base 12 项 + cost bp 5 项 + 状态 2 项 + 偏差 1 项可选。
        """
        ppy = periods_per_year if periods_per_year is not None else self.config.periods_per_year
        base_summary = self.base.summary(periods_per_year=ppy)

        bp = self.cost_breakdown.get("annualized_bp", {})
        out = dict(base_summary)
        out.update({
            "fee_bp":        float(bp.get("fee", 0.0)),
            "spread_bp":     float(bp.get("spread", 0.0)),
            "impact_bp":     float(bp.get("impact", 0.0)),
            "funding_bp":    float(bp.get("funding", 0.0)),
            "cost_total_bp": float(bp.get("total", 0.0)),
            "bankruptcy_flag":      self.bankruptcy_timestamp is not None,
            "bankruptcy_timestamp": self.bankruptcy_timestamp,
        })

        if self.deviation is not None:
            try:
                row = self.deviation.loc[self.deviation["bias_source"] == "总差（实测）"]
                if len(row) > 0:
                    out["total_deviation_terminal"] = float(
                        row["delta_terminal_return"].iloc[0]
                    )
            except (KeyError, ValueError):
                pass

        return out

    def to_dict(self) -> dict:
        """全量字典化，供 MongoDB / JSON pipeline 灌库"""
        return {
            "schema_version": SCHEMA_VERSION,
            "summary": _json_safe(self.summary()),
            "config": _serialize_config(self.config),
            "base": {
                "equity_curve": _series_to_dict(self.base.equity_curve),
                "returns": _series_to_dict(self.base.returns),
                "turnover": _series_to_dict(self.base.turnover),
                "weights_history": _df_to_records(self.base.weights_history),
                "gross_returns": (
                    _series_to_dict(self.base.gross_returns)
                    if self.base.gross_returns is not None else None
                ),
                "total_cost": float(self.base.total_cost),
            },
            "cost_breakdown": _json_safe(self.cost_breakdown),
            "deviation": (
                _df_to_records(self.deviation) if self.deviation is not None else None
            ),
            "regime_stats": (
                _df_to_records(self.regime_stats) if self.regime_stats is not None else None
            ),
            "funding_settlements": _json_safe(self.funding_settlements),
            "bankruptcy_timestamp": (
                self.bankruptcy_timestamp.isoformat()
                if self.bankruptcy_timestamp is not None else None
            ),
            "run_metadata": _json_safe(self.run_metadata),
        }

    def __repr__(self) -> str:
        """人类可读摘要（≤ 25 行；不递归 dump DataFrame/Series）"""
        s = self.summary()
        meta = self.run_metadata
        run_mode_str = meta.get("run_mode", "?")
        is_vec = run_mode_str == RunMode.VECTORIZED.value

        period_pct = (
            100.0 * meta.get("n_bars", 0) / max(meta.get("n_bars_planned", 1), 1)
        )
        walltime_min = meta.get("walltime_seconds", 0.0) / 60.0
        bankruptcy_str = "No" if self.bankruptcy_timestamp is None else (
            f"YES at {self.bankruptcy_timestamp}"
        )

        lines = [
            f"BacktestReport(strategy={self.config.strategy_name!r}, run_mode={run_mode_str})",
            f"  Period:    {self.config.start} → {self.config.end} "
            f"({meta.get('n_bars', '?')} bars, {period_pct:.0f}% completed)",
            f"  Walltime:  {walltime_min:.1f} min",
            "",
            "  Performance:",
            f"    Total return:   {s.get('total_return', float('nan')):+.2%}",
            f"    Annual return:  {s.get('annual_return', float('nan')):+.2%}",
            f"    Sharpe:         {s.get('sharpe_ratio', float('nan')):.2f}",
            f"    Max drawdown:   {s.get('max_drawdown', float('nan')):.2%}",
            f"    Win rate:       {s.get('win_rate', float('nan')):.1%}",
            "",
            "  Cost (annualized bp):",
            f"    Fee:      {s.get('fee_bp', 0.0):>7.1f}",
            f"    Spread:   {s.get('spread_bp', 0.0):>7.1f}",
            f"    Impact:   {s.get('impact_bp', 0.0):>7.1f}",
        ]
        if not is_vec:
            funding_bp = s.get('funding_bp', 0.0)
            sign = "(net received)" if funding_bp < 0 else ""
            lines.append(f"    Funding:  {funding_bp:>7.1f}  {sign}")
        lines.append(f"    Total:    {s.get('cost_total_bp', 0.0):>7.1f}")
        lines.append("")

        if not is_vec and self.funding_settlements is not None:
            fs = self.funding_settlements
            lines.append(
                f"  Funding events: {fs['n_events']} "
                f"(mean rate {fs['mean_rate_per_event']:.4%})"
            )
        if not is_vec:
            lines.append(f"  Bankruptcy:     {bankruptcy_str}")
        if self.regime_stats is not None:
            try:
                regimes = list(self.regime_stats.index)
                lines.append(f"  Regime stats:   {len(regimes)} regimes ({' / '.join(map(str, regimes))})")
            except Exception:
                lines.append(f"  Regime stats:   (available)")
        if not is_vec and self.deviation is not None:
            tot = s.get("total_deviation_terminal")
            if tot is not None:
                lines.append(f"  Deviation:      total {tot:+.2%} (vs vectorized baseline)")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 偏差归因（B1：用户 post-hoc 调）
    # ------------------------------------------------------------------

    def attach_deviation(
        self,
        vectorized_result: BacktestResult,
        *,
        ablation_results: dict[str, BacktestResult] | None = None,
    ) -> None:
        """
        计算并附加偏差归因（原地修改 self.deviation）。

        典型用法:
            report_ed = engine.run(config_event_driven)
            report_vec = engine.run(config_vectorized)
            report_ed.attach_deviation(report_vec.base)
        """
        from backtest_engine.attribution import deviation_attribution  # lazy
        self.deviation = deviation_attribution(self, vectorized_result, ablation_results)

    # ------------------------------------------------------------------
    # 可视化 / 报告（薄包装）
    # ------------------------------------------------------------------

    def plot(self, *, figsize: tuple[float, float] = (16, 10)):
        """一键全图。委托给 plot.py。"""
        from backtest_engine.plot import plot_all  # lazy
        return plot_all(self, figsize=figsize)

    def to_markdown(
        self,
        output_dir: str | Path,
        *,
        figure_dpi: int = 100,
        overwrite: bool = False,
    ) -> Path:
        """生成 Markdown 报告。委托给 reporting.py。"""
        from backtest_engine.reporting import to_markdown  # lazy
        return to_markdown(
            self, output_dir=output_dir,
            figure_dpi=figure_dpi, overwrite=overwrite,
        )

    # ------------------------------------------------------------------
    # 持久化（§11.8.6 v3 修订：原子化写入）
    # ------------------------------------------------------------------

    def save(
        self,
        report_dir: str | Path,
        *,
        context_builder: "MarketContextBuilder | None" = None,
        overwrite: bool = False,
    ) -> None:
        """
        持久化：parquet + JSON 路线（§11.8.6）

        v3 修订关键：所有写入先到 tmp_dir，全部成功后 os.replace 原子化重命名。

        Args:
            report_dir:       输出目录（不存在则创建）
            context_builder:  非 None 时调 build_panels() 嵌入 context/ 子目录，
                              使跨机器加载后仍能调 compute_per_symbol_cost
            overwrite:        True 时覆盖已存在目录，默认 False
        """
        report_dir = Path(report_dir)
        if report_dir.exists() and not overwrite:
            raise FileExistsError(
                f"report_dir 已存在 {report_dir}（不覆盖；显式传 overwrite=True）"
            )

        tmp_dir = report_dir.parent / f"{report_dir.name}.tmp_{uuid.uuid4().hex[:8]}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=False)

        try:
            # 1. base/*.parquet
            base_dir = tmp_dir / "base"
            base_dir.mkdir()
            self.base.equity_curve.to_frame("value").to_parquet(base_dir / "equity_curve.parquet")
            self.base.returns.to_frame("value").to_parquet(base_dir / "returns.parquet")
            self.base.turnover.to_frame("value").to_parquet(base_dir / "turnover.parquet")
            self.base.weights_history.to_parquet(base_dir / "weights_history.parquet")
            if self.base.gross_returns is not None:
                self.base.gross_returns.to_frame("value").to_parquet(
                    base_dir / "gross_returns.parquet"
                )
            with open(base_dir / "scalars.json", "w", encoding="utf-8") as f:
                json.dump({"total_cost": float(self.base.total_cost)}, f, indent=2)

            # 2. deviation / regime_stats parquet（仅非 None）
            if self.deviation is not None:
                self.deviation.to_parquet(tmp_dir / "deviation.parquet")
            if self.regime_stats is not None:
                self.regime_stats.to_parquet(tmp_dir / "regime_stats.parquet")

            # 3. metadata.json
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "run_metadata": _json_safe(self.run_metadata),
                "bankruptcy_timestamp": (
                    self.bankruptcy_timestamp.isoformat()
                    if self.bankruptcy_timestamp is not None else None
                ),
                "funding_settlements": _json_safe(self.funding_settlements),
                "cost_breakdown": _json_safe(self.cost_breakdown),
            }
            with open(tmp_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            # 4. config.json（regime_series 单独存 parquet）
            config_dict = _serialize_config(self.config)
            with open(tmp_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            if self.config.regime_series is not None:
                self.config.regime_series.to_frame("regime").to_parquet(
                    tmp_dir / "regime_input.parquet"
                )

            # 5. context_panels（B3 嵌入）
            if context_builder is not None:
                ctx_dir = tmp_dir / "context"
                ctx_dir.mkdir()
                panels = context_builder.build_panels()
                # 仅取 §12.6 三 panel（不存 price_panel）
                for key in ("spread_panel", "adv_panel", "vol_panel"):
                    panels[key].to_parquet(ctx_dir / f"{key}.parquet")

            # 6. 原子化提交
            if report_dir.exists():
                shutil.rmtree(report_dir)
            os.replace(str(tmp_dir), str(report_dir))

        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        logger.info("BacktestReport 已保存到 %s", report_dir)

    @classmethod
    def load(cls, report_dir: str | Path) -> "BacktestReport":
        """反持久化（schema_version 兼容性校验）"""
        report_dir = Path(report_dir)
        if not report_dir.exists():
            raise FileNotFoundError(f"report_dir 不存在: {report_dir}")

        # 1. metadata
        meta_path = report_dir / "metadata.json"
        if not meta_path.exists():
            raise ValueError(f"metadata.json 不存在；旧版或损坏的 report: {report_dir}")
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        sv = metadata.get("schema_version")
        if sv is None:
            raise ValueError("schema_version 缺失；旧版或损坏的 report")
        if sv.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
            raise ValueError(
                f"schema_version 不兼容主版本：当前 {SCHEMA_VERSION}，文件 {sv}；"
                f"需要迁移工具"
            )

        # 2. config
        with open(report_dir / "config.json", "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        config = _deserialize_config(config_dict)

        # 2.5 regime_input（若有，回填 config.regime_series）
        regime_path = report_dir / "regime_input.parquet"
        if regime_path.exists():
            df = pd.read_parquet(regime_path)
            # to_frame("regime") 后单列；还原 Series 含原 index
            config = dataclasses.replace(config, regime_series=df["regime"])

        # 3. base
        base_dir = report_dir / "base"
        # Step 12 / C2: with 块替代裸 open（防 Windows 文件锁定 + GC 依赖）
        with open(base_dir / "scalars.json", "r", encoding="utf-8") as f:
            scalars = json.load(f)
        base = BacktestResult(
            equity_curve=_load_value_series(base_dir / "equity_curve.parquet"),
            returns=_load_value_series(base_dir / "returns.parquet"),
            turnover=_load_value_series(base_dir / "turnover.parquet"),
            weights_history=pd.read_parquet(base_dir / "weights_history.parquet"),
            gross_returns=(
                _load_value_series(base_dir / "gross_returns.parquet")
                if (base_dir / "gross_returns.parquet").exists() else None
            ),
            total_cost=float(scalars["total_cost"]),
        )

        # 4. deviation / regime_stats
        deviation = None
        if (report_dir / "deviation.parquet").exists():
            deviation = pd.read_parquet(report_dir / "deviation.parquet")
        regime_stats = None
        if (report_dir / "regime_stats.parquet").exists():
            regime_stats = pd.read_parquet(report_dir / "regime_stats.parquet")

        # 5. context_panels（自动检测）
        context_panels = None
        ctx_dir = report_dir / "context"
        if ctx_dir.exists():
            context_panels = {
                "spread_panel": pd.read_parquet(ctx_dir / "spread_panel.parquet"),
                "adv_panel":    pd.read_parquet(ctx_dir / "adv_panel.parquet"),
                "vol_panel":    pd.read_parquet(ctx_dir / "vol_panel.parquet"),
            }

        # 6. metadata 字段还原
        bts = metadata.get("bankruptcy_timestamp")
        bankruptcy_ts = pd.Timestamp(bts) if bts else None

        funding_settlements = metadata.get("funding_settlements")
        if funding_settlements is not None:
            for k in ("first_event", "last_event"):
                if funding_settlements.get(k):
                    funding_settlements[k] = pd.Timestamp(funding_settlements[k])

        run_metadata = metadata.get("run_metadata", {})
        for k in ("start", "end"):
            if k in run_metadata and isinstance(run_metadata[k], str):
                run_metadata[k] = pd.Timestamp(run_metadata[k])

        return cls(
            base=base,
            config=config,
            cost_breakdown=metadata.get("cost_breakdown", {}),
            deviation=deviation,
            regime_stats=regime_stats,
            funding_settlements=funding_settlements,
            bankruptcy_timestamp=bankruptcy_ts,
            run_metadata=run_metadata,
            context_panels=context_panels,
        )


# ---------------------------------------------------------------------------
# 序列化辅助（§11.8.7）
# ---------------------------------------------------------------------------

def _serialize_config(config: BacktestConfig) -> dict:
    """BacktestConfig → JSON-safe dict"""
    constraints_dict = None
    if config.constraints is not None:
        constraints_dict = dataclasses.asdict(config.constraints)
        # Callable 字段护栏（§11.8.7）
        for k, v in constraints_dict.items():
            if callable(v):
                raise NotImplementedError(
                    f"PortfolioConstraints.{k} 含 callable 字段，v1 不支持序列化。"
                    f"请改用静态参数；如必须 callable，等待 v2 自定义序列化器支持。"
                )

    impact_coeff_serialized = (
        {"_type": "pd.Series", "data": config.impact_coeff.to_dict()}
        if isinstance(config.impact_coeff, pd.Series)
        else float(config.impact_coeff)
    )

    return {
        "strategy_name": config.strategy_name,
        "symbols": list(config.symbols),
        "start": config.start.isoformat(),
        "end": config.end.isoformat(),
        "bar_freq": config.bar_freq,
        "run_mode": config.run_mode.value,
        "execution_mode": config.execution_mode.value,
        "cost_mode": config.cost_mode.value,
        "initial_portfolio_value": float(config.initial_portfolio_value),
        "constraints": constraints_dict,
        "impact_coeff": impact_coeff_serialized,
        "fee_rate": float(config.fee_rate),
        "max_participation": float(config.max_participation),
        "periods_per_year": float(config.periods_per_year),
        "optimize_every_n_bars": int(config.optimize_every_n_bars),
        "min_trade_value": float(config.min_trade_value),
        "time_convention": config.time_convention,
        # regime_series 不在此处（独立 parquet）
    }


def _deserialize_config(d: dict) -> BacktestConfig:
    """JSON dict → BacktestConfig"""
    constraints = None
    if d.get("constraints") is not None:
        constraints = PortfolioConstraints(**d["constraints"])

    ic = d["impact_coeff"]
    if isinstance(ic, dict) and ic.get("_type") == "pd.Series":
        impact_coeff = pd.Series(ic["data"])
    else:
        impact_coeff = float(ic)

    return BacktestConfig(
        strategy_name=d["strategy_name"],
        symbols=list(d["symbols"]),
        start=pd.Timestamp(d["start"]),
        end=pd.Timestamp(d["end"]),
        bar_freq=d.get("bar_freq", "1m"),
        run_mode=RunMode(d["run_mode"]),
        execution_mode=ExecutionMode(d["execution_mode"]),
        cost_mode=CostMode(d["cost_mode"]),
        initial_portfolio_value=float(d["initial_portfolio_value"]),
        constraints=constraints,
        impact_coeff=impact_coeff,
        fee_rate=float(d["fee_rate"]),
        max_participation=float(d["max_participation"]),
        periods_per_year=float(d["periods_per_year"]),
        optimize_every_n_bars=int(d["optimize_every_n_bars"]),
        min_trade_value=float(d["min_trade_value"]),
        time_convention=d.get("time_convention", "bar_close"),
    )


# ---------------------------------------------------------------------------
# JSON / parquet 辅助
# ---------------------------------------------------------------------------

def _json_safe(obj):
    """递归把 NaN/Inf/Timestamp 等转 JSON-safe；保留嵌套结构"""
    if obj is None:
        return None
    if isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int,)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, pd.Series):
        return _series_to_dict(obj)
    if isinstance(obj, pd.DataFrame):
        return _df_to_records(obj)
    return str(obj)


def _series_to_dict(s: pd.Series) -> dict[str, float]:
    out = {}
    for k, v in s.items():
        ks = k.isoformat() if isinstance(k, pd.Timestamp) else str(k)
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            out[ks] = None
        else:
            out[ks] = float(v) if not isinstance(v, str) else v
    return out


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    return _json_safe(df.reset_index().to_dict(orient="records"))


def _load_value_series(path: Path) -> pd.Series:
    """读 to_frame('value') 写入的单列 parquet 还原 Series（保留 DatetimeIndex；name 还原为 None）"""
    df = pd.read_parquet(path)
    s = df["value"] if "value" in df.columns else df.iloc[:, 0]
    s.name = None
    return s
