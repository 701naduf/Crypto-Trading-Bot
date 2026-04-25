"""
attribution.py — Phase 3 高价值产出模块

详见 docs/phase3_design.md §11.7。

纯函数模块（无 class、无状态、无副作用）：

  cost_decomposition(pnl_tracker, periods_per_year)
      → 回答"我的钱去哪了？"（fee/spread/impact/funding 各占多少 + 年化 bp + 占比）

  deviation_attribution(event_driven_report, vectorized_result, ablation_results=None)
      → 回答"我的事件驱动结果与理想执行差多少？"（偏差拆解表）

  regime_breakdown(base_result, regime_series, periods_per_year, cost_series, ...)
      → 回答"在不同市场环境下表现如何？"

  compute_per_symbol_cost(weights_history, spread_panel, adv_panel, vol_panel,
                          fee_rate, impact_coeff, portfolio_value_history)
      → 按需重算 per-symbol cost（D3 A+ 折中，不膨胀 PnLTracker）

设计要点：
  - attribution 不内部触发回测（避免循环依赖；ablation 由用户 / 上层组织）
  - funding 一阶近似默认量化（解析可近似，给数字胜过空白）
  - per-symbol 与 portfolio total 必须一致（公式护栏测试）
  - VECTORIZED 模式 deviation = None（自身就是 baseline）
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from alpha_model.backtest.performance import BacktestResult

from backtest_engine.pnl import PnLTracker

if TYPE_CHECKING:
    from backtest_engine.report import BacktestReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# §11.7.3 cost_decomposition
# ---------------------------------------------------------------------------

def cost_decomposition(
    pnl_tracker: PnLTracker,
    periods_per_year: float,
) -> dict[str, dict[str, float]]:
    """
    portfolio-level fee/spread/impact/funding 分解 + 年化 bp + 占比

    Returns:
        {
            "absolute": {"fee", "spread", "impact", "funding", "total"},      # 5 keys
            "annualized_bp": 同上,                                              # 5 keys
            "share": {"fee", "spread", "impact", "funding"},                  # 4 keys (无 total)
        }
    """
    fee_s = pnl_tracker.fee_series
    spread_s = pnl_tracker.spread_series
    impact_s = pnl_tracker.impact_series

    if len(fee_s) == 0:
        # zero-bar 回测：返回全 0
        zeros = {"fee": 0.0, "spread": 0.0, "impact": 0.0, "funding": 0.0, "total": 0.0}
        return {
            "absolute": dict(zeros),
            "annualized_bp": dict(zeros),
            "share": {k: float("nan") for k in ("fee", "spread", "impact", "funding")},
        }

    # funding 对齐到 fee_s 的 index（funding_events 是稀疏；fill 0）
    funding_evt = pnl_tracker.funding_events
    if len(funding_evt) > 0:
        funding_aligned = funding_evt.reindex(fee_s.index, fill_value=0.0)
    else:
        funding_aligned = pd.Series(0.0, index=fee_s.index)

    total_s = fee_s + spread_s + impact_s + funding_aligned

    absolute = {
        "fee": float(fee_s.sum()),
        "spread": float(spread_s.sum()),
        "impact": float(impact_s.sum()),
        "funding": float(funding_aligned.sum()),
        "total": float(total_s.sum()),
    }

    # 年化 bp：用 mean × periods_per_year × 1e4（数学等价 sum × ppy / N，但 mean 路径数值更稳）
    annualized_bp = {
        "fee":     float(fee_s.mean()      * periods_per_year * 1e4),
        "spread":  float(spread_s.mean()   * periods_per_year * 1e4),
        "impact":  float(impact_s.mean()   * periods_per_year * 1e4),
        "funding": float(funding_aligned.mean() * periods_per_year * 1e4),
        "total":   float(total_s.mean()    * periods_per_year * 1e4),
    }

    # share：占 |total| 比例。total=0 时全部 NaN（不抛除零）
    denom = abs(absolute["total"])
    if denom == 0.0:
        share = {k: float("nan") for k in ("fee", "spread", "impact", "funding")}
    else:
        share = {
            k: absolute[k] / denom
            for k in ("fee", "spread", "impact", "funding")
        }

    return {
        "absolute": absolute,
        "annualized_bp": annualized_bp,
        "share": share,
    }


# ---------------------------------------------------------------------------
# §11.7.4 deviation_attribution
# ---------------------------------------------------------------------------

# 偏差源标识（与 §11.7.4 表格列一致）
_BIAS_FUNDING = "funding 事件"
_BIAS_MIN_TRADE = "min_trade_value 过滤"
_BIAS_DYNAMIC_V = "动态 V 更新（impact 偏差）"
_BIAS_OPTIMIZE_N = "optimize_every_n_bars (N={})"
_BIAS_LIQUIDATION = "强平不建模"
_BIAS_HALFLIFE = "信号半衰期假设"
_BIAS_TOTAL = "总差（实测）"
_BIAS_RESIDUAL = "未归因残差"


def deviation_attribution(
    event_driven_report: "BacktestReport",
    vectorized_result: BacktestResult,
    ablation_results: dict[str, BacktestResult] | None = None,
) -> pd.DataFrame:
    """
    偏差拆解表（默认 P1 模式 + ablation 可选 P2 模式）

    Args:
        event_driven_report: 事件驱动回测产出（含 base / config / funding_settlements）
        vectorized_result:   vectorized_backtest 产出（用户先跑，传入做 baseline）
        ablation_results:    可选 dict，key ∈ {"min_trade_value", "funding", "dynamic_v",
                             "optimize_every_n_bars"}；value = 对应 ablation 跑出的 BacktestResult

    Returns:
        DataFrame[bias_source, delta_terminal_return, delta_sharpe, quantified, method, note]
    """
    base = event_driven_report.base
    cfg = event_driven_report.config

    if len(base.equity_curve) == 0 or len(vectorized_result.equity_curve) == 0:
        raise ValueError("无数据可归因（event-driven 或 vectorized 结果为空）")

    # 总差（实测）
    ed_term = float(base.equity_curve.iloc[-1] / base.equity_curve.iloc[0] - 1.0)
    vec_term = float(
        vectorized_result.equity_curve.iloc[-1] / vectorized_result.equity_curve.iloc[0] - 1.0
    )
    total_delta_return = vec_term - ed_term

    ed_sharpe = base.summary(periods_per_year=cfg.periods_per_year)["sharpe_ratio"]
    vec_sharpe = vectorized_result.summary(periods_per_year=cfg.periods_per_year)["sharpe_ratio"]
    total_delta_sharpe = vec_sharpe - ed_sharpe

    rows: list[dict] = []
    ablations = ablation_results or {}

    # ── min_trade_value 过滤 ──
    if "min_trade_value" in ablations:
        delta = _ablation_delta(ablations["min_trade_value"], base)
        rows.append({
            "bias_source": _BIAS_MIN_TRADE,
            "delta_terminal_return": delta["terminal"],
            "delta_sharpe": delta["sharpe"],
            "quantified": True,
            "method": "ablation",
            "note": "ablation 跑：config.min_trade_value=0",
        })
    else:
        rows.append({
            "bias_source": _BIAS_MIN_TRADE,
            "delta_terminal_return": np.nan,
            "delta_sharpe": np.nan,
            "quantified": False,
            "method": "not_quantified",
            "note": "跑 ablation：'config.min_trade_value=0'",
        })

    # ── funding（默认 first-order 近似量化）──
    if "funding" in ablations:
        delta = _ablation_delta(ablations["funding"], base)
        rows.append({
            "bias_source": _BIAS_FUNDING,
            "delta_terminal_return": delta["terminal"],
            "delta_sharpe": delta["sharpe"],
            "quantified": True,
            "method": "ablation",
            "note": "ablation 跑：屏蔽 funding 事件",
        })
    else:
        # first-order 近似：从 funding_settlements.total_rate
        settlement = event_driven_report.funding_settlements
        total_funding_rate = (
            float(settlement["total_rate"]) if settlement else 0.0
        )
        # 去掉 funding 总扣款 → terminal return 增加 -total_funding_rate
        funding_delta = -total_funding_rate
        rows.append({
            "bias_source": _BIAS_FUNDING,
            "delta_terminal_return": funding_delta,
            "delta_sharpe": np.nan,  # 一阶近似不能直接给 Sharpe
            "quantified": True,
            "method": "first_order_approx",
            "note": "一阶近似；精确值需 ablation",
        })

    # ── 动态 V 更新（impact 偏差）──
    if "dynamic_v" in ablations:
        delta = _ablation_delta(ablations["dynamic_v"], base)
        rows.append({
            "bias_source": _BIAS_DYNAMIC_V,
            "delta_terminal_return": delta["terminal"],
            "delta_sharpe": delta["sharpe"],
            "quantified": True,
            "method": "ablation",
            "note": "ablation：使用 static_v PnLTracker（v1 未实现）",
        })
    else:
        rows.append({
            "bias_source": _BIAS_DYNAMIC_V,
            "delta_terminal_return": np.nan,
            "delta_sharpe": np.nan,
            "quantified": False,
            "method": "not_quantified",
            "note": "跑 ablation：'使用 static_v PnLTracker'（v1 未实现）",
        })

    # ── optimize_every_n_bars 加速（仅 N>1 出现）──
    if cfg.optimize_every_n_bars > 1:
        bias_name = _BIAS_OPTIMIZE_N.format(cfg.optimize_every_n_bars)
        if "optimize_every_n_bars" in ablations:
            delta = _ablation_delta(ablations["optimize_every_n_bars"], base)
            rows.append({
                "bias_source": bias_name,
                "delta_terminal_return": delta["terminal"],
                "delta_sharpe": delta["sharpe"],
                "quantified": True,
                "method": "ablation",
                "note": "ablation：N=1（精确语义）",
            })
        else:
            rows.append({
                "bias_source": bias_name,
                "delta_terminal_return": np.nan,
                "delta_sharpe": np.nan,
                "quantified": False,
                "method": "not_quantified",
                "note": "跑 ablation：'optimize_every_n_bars=1'",
            })

    # ── 强平不建模（永远 quantified=False）──
    rows.append({
        "bias_source": _BIAS_LIQUIDATION,
        "delta_terminal_return": np.nan,
        "delta_sharpe": np.nan,
        "quantified": False,
        "method": "n/a",
        "note": "v1 已知乐观偏差（§11.4.9 / §11.5.10）",
    })

    # ── 信号半衰期假设（永远 quantified=False）──
    rows.append({
        "bias_source": _BIAS_HALFLIFE,
        "delta_terminal_return": np.nan,
        "delta_sharpe": np.nan,
        "quantified": False,
        "method": "n/a",
        "note": "v1 已知偏差源（phase3_debug.md §12.8）",
    })

    # ── 总差（必有，quantified=True）──
    rows.append({
        "bias_source": _BIAS_TOTAL,
        "delta_terminal_return": total_delta_return,
        "delta_sharpe": total_delta_sharpe,
        "quantified": True,
        "method": "direct",
        "note": "vec_terminal − event_driven_terminal",
    })

    # ── 未归因残差（仅 ablation 模式且 ≥ 2 项 quantified）──
    quantified_deltas = [
        r["delta_terminal_return"] for r in rows
        if r["quantified"] and r["bias_source"] not in (_BIAS_TOTAL, _BIAS_RESIDUAL)
    ]
    if len(ablations) > 0 and len(quantified_deltas) >= 2:
        residual = total_delta_return - sum(quantified_deltas)
        rows.append({
            "bias_source": _BIAS_RESIDUAL,
            "delta_terminal_return": residual,
            "delta_sharpe": np.nan,
            "quantified": True,
            "method": "residual",
            "note": "总差 − Σ 量化项；残差大说明偏差源相互耦合显著",
        })

    return pd.DataFrame(rows)


def _ablation_delta(
    ablation_result: BacktestResult, base: BacktestResult,
) -> dict[str, float]:
    """计算 ablation 与 base 的 terminal/sharpe 差"""
    abl_term = float(
        ablation_result.equity_curve.iloc[-1] / ablation_result.equity_curve.iloc[0] - 1.0
    )
    base_term = float(
        base.equity_curve.iloc[-1] / base.equity_curve.iloc[0] - 1.0
    )
    abl_sharpe = ablation_result.summary()["sharpe_ratio"]
    base_sharpe = base.summary()["sharpe_ratio"]
    return {
        "terminal": abl_term - base_term,
        "sharpe": abl_sharpe - base_sharpe,
    }


# ---------------------------------------------------------------------------
# §11.7.5 regime_breakdown
# ---------------------------------------------------------------------------

def regime_breakdown(
    base_result: BacktestResult,
    regime_series: pd.Series,
    periods_per_year: float,
    cost_series: dict[str, pd.Series],
    *,
    min_bars_per_regime: int = 30,
) -> pd.DataFrame:
    """
    按 regime 分段统计 6 项精选指标

    Args:
        base_result:    BacktestResult
        regime_series:  index = timestamp（与 base_result.returns.index 部分重叠即可），
                        value = regime 标签（任意 hashable）
        periods_per_year: 年化系数
        cost_series:    {"fee", "spread", "impact"}: pd.Series（per-bar 总和）
                        来源：PnLTracker 的 fee_series / spread_series / impact_series properties
                        VECTORIZED 模式由 _vectorized_cost_breakdown 重算后传入
        min_bars_per_regime: 跳过 n_bars < 此值的 regime（默认 30）

    Returns:
        DataFrame[regime, n_bars, total_return, sharpe, max_drawdown, turnover_mean, cost_rate_bp]
        index = regime 标签
    """
    # 校验 cost_series 必须含三 keys（§12.5'，**3 项无 funding**）
    missing = {"fee", "spread", "impact"} - set(cost_series.keys())
    if missing:
        raise KeyError(f"cost_series 缺 keys: {missing}")

    # 对齐：ffill regime 标签到 base_result.returns.index（无 max_gap，§11.7.5 注）
    aligned = regime_series.reindex(base_result.returns.index, method="ffill")
    aligned = aligned.dropna()

    if len(aligned) == 0:
        logger.warning("regime_breakdown: regime_series ffill 后全 NaN，返回空 DataFrame")
        return pd.DataFrame(columns=[
            "n_bars", "total_return", "sharpe", "max_drawdown",
            "turnover_mean", "cost_rate_bp",
        ])

    # 按 regime 分组
    rows: list[dict] = []
    skipped: list = []

    fee_s = cost_series["fee"]
    spread_s = cost_series["spread"]
    impact_s = cost_series["impact"]

    for label, idx in aligned.groupby(aligned).groups.items():
        if len(idx) < min_bars_per_regime:
            skipped.append(label)
            continue

        sub_returns = base_result.returns.loc[idx]
        sub_turnover = base_result.turnover.loc[idx]
        sub_fee = fee_s.reindex(idx).fillna(0.0)
        sub_spread = spread_s.reindex(idx).fillna(0.0)
        sub_impact = impact_s.reindex(idx).fillna(0.0)

        cum = (1.0 + sub_returns).cumprod()
        total_return = float(cum.iloc[-1] - 1.0) if len(cum) > 0 else 0.0

        # Sharpe（年化）
        if sub_returns.std(ddof=0) > 0:
            sharpe = float(
                sub_returns.mean() / sub_returns.std(ddof=0) * np.sqrt(periods_per_year)
            )
        else:
            sharpe = float("nan")

        # Max drawdown
        running_max = cum.cummax()
        drawdown = cum / running_max - 1.0
        max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

        # Cost rate bp（不含 funding）
        cost_rate_bp = float(
            (sub_fee.mean() + sub_spread.mean() + sub_impact.mean())
            * periods_per_year * 1e4
        )

        rows.append({
            "regime": label,
            "n_bars": int(len(idx)),
            "total_return": total_return,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "turnover_mean": float(sub_turnover.mean()),
            "cost_rate_bp": cost_rate_bp,
        })

    if skipped:
        logger.warning(
            "regime_breakdown: 跳过 %d 个 regime（n_bars < %d）：%s",
            len(skipped), min_bars_per_regime, skipped,
        )

    if len(rows) == 0:
        return pd.DataFrame(columns=[
            "n_bars", "total_return", "sharpe", "max_drawdown",
            "turnover_mean", "cost_rate_bp",
        ])

    df = pd.DataFrame(rows).set_index("regime")
    return df


# ---------------------------------------------------------------------------
# §11.7.6 compute_per_symbol_cost
# ---------------------------------------------------------------------------

def compute_per_symbol_cost(
    weights_history: pd.DataFrame,
    spread_panel: pd.DataFrame,
    adv_panel: pd.DataFrame,
    vol_panel: pd.DataFrame,
    fee_rate: float,
    impact_coeff: float | pd.Series,
    portfolio_value_history: pd.Series,
) -> dict[str, pd.DataFrame]:
    """
    按需重算 per-symbol cost（D3 A+ 折中方案）

    公式必须与 Rebalancer / cost.py / vectorized.py 一致（§11.4 选择 D），
    否则 per-symbol 加总 ≠ portfolio 总。

    Args:
        weights_history:         timestamp × symbol 权重历史
        spread_panel:            timestamp × symbol，比率 (ask-bid)/mid
        adv_panel:               timestamp × symbol，USDT
        vol_panel:               timestamp × symbol，日化 σ
        fee_rate:                taker 手续费率
        impact_coeff:            float 或 pd.Series(index=symbols)
        portfolio_value_history: 每 bar 末尾 V（用 shift(1) 取上一步 V 算 impact）

    Returns:
        {"fee": DataFrame, "spread": DataFrame, "impact": DataFrame}
    """
    delta_w = weights_history.diff()
    abs_dw = delta_w.abs()
    symbols = list(weights_history.columns)
    idx = delta_w.index

    # 对齐面板到 weights_history 索引
    spread_aligned = spread_panel.reindex(idx)[symbols]
    adv_aligned = adv_panel.reindex(idx)[symbols]
    vol_aligned = vol_panel.reindex(idx)[symbols]

    # 1. fee per symbol
    fee = fee_rate * abs_dw

    # 2. spread per symbol = spread/2 × |Δw|
    spread = spread_aligned / 2.0 * abs_dw

    # 3. impact per symbol = (2/3) × coeff × σ × √(V_prev/ADV) × |Δw|^1.5
    V_prev = portfolio_value_history.shift(1).reindex(idx)
    # ADV 兜底（Z4/Z12 跨四处统一 + NaN warning）
    from alpha_model.backtest.adv_helpers import safe_adv_panel
    adv_safe = safe_adv_panel(adv_aligned, context="compute_per_symbol_cost")
    sqrt_ratio = np.sqrt(V_prev.values[:, None] / adv_safe.values)

    if isinstance(impact_coeff, pd.Series):
        coeff_arr = impact_coeff.reindex(symbols).values.astype(float)
    else:
        coeff_arr = float(impact_coeff)

    impact_arr = (
        (2.0 / 3.0) * coeff_arr * vol_aligned.values * sqrt_ratio
        * np.power(abs_dw.values, 1.5)
    )
    impact = pd.DataFrame(impact_arr, index=idx, columns=symbols)

    # 第一行的 delta_w 是 NaN（diff 首行）；其余对应 V_prev=NaN 也 NaN — fillna(0)
    return {
        "fee":    fee.fillna(0.0),
        "spread": spread.fillna(0.0),
        "impact": impact.fillna(0.0),
    }
