"""
ExecutionOptimizer：动态成本感知的组合优化器

Phase 2b PortfolioConstructor（固定 γ）的事件驱动替代方案。
唯一公开方法 optimize_step()，每个时间步调用一次。

目标函数（全部无量纲）:
    minimize:  w'Σw
               - λ × α'w
               + cost_expression(Δw, MarketContext)
               + funding_rate'w
    subject to:
        build_constraints(w, PortfolioConstraints)
        |Δw_i| × V ≤ participation × ADV_i

与 PortfolioConstructor 互斥使用（γ=0，成本完全由动态模型提供）。
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
import cvxpy as cp

from alpha_model.core.types import PortfolioConstraints
from alpha_model.portfolio.constraints import build_constraints
from alpha_model.portfolio.covariance import estimate_covariance
from alpha_model.portfolio.beta import rolling_beta
from alpha_model.config import MINUTES_PER_YEAR

from execution_optimizer.config import (
    MarketContext, DEFAULT_IMPACT_COEFF, DEFAULT_FEE_RATE,
    DEFAULT_MAX_PARTICIPATION,
)
from execution_optimizer.cost import build_cost_expression

logger = logging.getLogger(__name__)


class ExecutionOptimizer:
    """
    动态成本感知的组合优化器

    用法:
        optimizer = ExecutionOptimizer(constraints)
        for bar in event_loop:
            target_w = optimizer.optimize_step(
                signals_t, current_w, context, prices[:t]
            )
    """

    def __init__(
        self,
        constraints: PortfolioConstraints,
        impact_coeff: float | pd.Series = DEFAULT_IMPACT_COEFF,
        fee_rate: float = DEFAULT_FEE_RATE,
        max_participation: float | None = DEFAULT_MAX_PARTICIPATION,
        periods_per_year: float = MINUTES_PER_YEAR,
    ):
        """
        Args:
            constraints:       组合约束（与 Phase 2b 相同类型）。
                               turnover_penalty 不被使用（γ=0），
                               risk_aversion (λ) 仍有效。
            impact_coeff:      sqrt-model 校准系数。
                               float → 全标的共享；
                               pd.Series(index=symbols) → 逐标的独立校准。
                               建议对流动性差异大的标的使用 Series 形式。
            fee_rate:          taker 手续费率
            max_participation: 单步最大 ADV 参与率。
                               0.05 = 单步交易量不超过 ADV 的 5%。
                               None = 不限制（不推荐用于实盘）。
            periods_per_year:  年化系数，用于 vol_target 缩放中把 1 期方差
                               (`w'Σw`) 转为年化波动率。默认 `MINUTES_PER_YEAR`
                               (525960)，对应 price_history 是 1m bar。
                               其他频率：5m→105192, 15m→35064, 1h→8766。
                               改变此值必须保证 price_history 的频率与之匹配
                               （协方差矩阵 Σ 来自 price_history.pct_change()，
                               单期方差的时间尺度 = price_history 的 bar 尺度）。
        """
        self.constraints = constraints
        self.impact_coeff = impact_coeff
        self.fee_rate = fee_rate
        self.max_participation = max_participation
        self.periods_per_year = periods_per_year

    def optimize_step(
        self,
        signals_t: pd.Series,
        current_weights: pd.Series,
        market_context: MarketContext,
        price_history: pd.DataFrame,
        adv_nan_warned: set[str] | None = None,
    ) -> pd.Series:
        """
        单步优化：给定当前信号和市场状态，返回最优目标权重

        Args:
            signals_t:       当前时刻截面信号, index=symbols。
                             可以是 load_signals() 或 load_raw_predictions() 的输出
            current_weights: 当前实际持仓权重, index=symbols。
                             须为上一步执行后的真实持仓（非上一步的 target）
            market_context:  当前时刻市场状态（调用方构造并注入）
            price_history:   截至当前时刻（含）的价格面板,
                             index=DatetimeIndex, columns ⊇ symbols。
                             用于协方差估计和 beta 估计。
                             调用方须保证无未来信息（严格 ≤ t）
            adv_nan_warned:  Z16 dedup 集合（OnlineOptimizer 持有）。
                             事件循环每 bar 调用 optimize_step，若 ADV 持续 NaN
                             会 log spam；此 set 让"首次出现"的 NaN symbol 才 warning。
                             None 表示无 dedup（每次都 warning，适用于一次性调用）。

        Returns:
            目标权重 pd.Series, index=symbols。
            求解失败时返回 current_weights（维持当前持仓）
        """
        symbols = market_context.symbols
        n = len(symbols)

        alpha  = signals_t.reindex(symbols).fillna(0.0).values.astype(float)
        w_prev = current_weights.reindex(symbols).fillna(0.0).values.astype(float)

        # ── 协方差估计（Ledoit-Wolf）──
        returns_panel = price_history[symbols].pct_change().dropna()
        lookback = min(self.constraints.vol_lookback, len(returns_panel))

        try:
            cov = estimate_covariance(
                returns_panel, lookback=lookback, method="ledoit_wolf",
            )
        except Exception as e:
            logger.warning("协方差估计失败，维持当前持仓: %s", e)
            return current_weights.reindex(symbols).fillna(0.0)

        # PSD 保证（与 Phase 2b PortfolioConstructor._solve_single_period 一致）
        min_eig = np.linalg.eigvalsh(cov).min()
        if min_eig < 0:
            cov += (-min_eig + 1e-8) * np.eye(n)

        # ── cvxpy 建模 ──
        w = cp.Variable(n)
        delta_w = w - w_prev

        cost_expr = build_cost_expression(
            delta_w, market_context,
            impact_coeff=self.impact_coeff, fee_rate=self.fee_rate,
            adv_nan_warned=adv_nan_warned,
        )

        # 目标函数
        lam = self.constraints.risk_aversion
        obj_expr = cp.quad_form(w, cov) - lam * (alpha @ w) + cost_expr

        # 资金费率项：持多仓且费率为正时增加成本
        if market_context.funding_rate is not None:
            fr = market_context.funding_rate.reindex(symbols).fillna(0.0).values.astype(float)
            obj_expr += fr @ w

        objective = cp.Minimize(obj_expr)

        # ── 约束（复用 Phase 2b 约束构建器）──
        beta_vec = None
        if self.constraints.beta_neutral:
            beta_panel = rolling_beta(
                returns_panel, lookback=self.constraints.beta_lookback,
            )
            if len(beta_panel) > 0:
                last_beta = beta_panel.iloc[-1].reindex(symbols).values
                if not np.any(np.isnan(last_beta)):
                    beta_vec = last_beta

        cvx_constraints = build_constraints(w, self.constraints, beta=beta_vec)

        # ADV 参与率约束: |Δw_i| × V ≤ participation × ADV_i
        if self.max_participation is not None:
            adv_arr = market_context.adv.reindex(symbols).values.astype(float)
            V = market_context.portfolio_value
            # 转换为权重空间: |Δw_i| ≤ participation × ADV_i / V
            participation_limit = self.max_participation * adv_arr / np.maximum(V, 1.0)
            cvx_constraints.append(cp.abs(delta_w) <= participation_limit)

        # ── 求解 ──
        prob = cp.Problem(objective, cvx_constraints)
        try:
            prob.solve(warm_start=True)
        except Exception as e:
            logger.warning("cvxpy 求解异常，维持当前持仓: %s", e)
            return current_weights.reindex(symbols).fillna(0.0)

        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            logger.warning("cvxpy 状态=%s，维持当前持仓", prob.status)
            return current_weights.reindex(symbols).fillna(0.0)

        w_opt = w.value

        # ── 单步 Vol Targeting（事后缩放）──
        # Phase 2b 使用 apply_vol_target（面板级函数，依赖 shift(1)，不适用于单步）。
        # 此处使用协方差矩阵直接估计组合波动率，无需历史组合收益序列。
        if self.constraints.vol_target is not None:
            port_var = w_opt @ cov @ w_opt                        # 1 期方差（bar 尺度）
            port_vol = np.sqrt(port_var * self.periods_per_year)  # 年化波动率

            if port_vol > 0:
                scale = self.constraints.vol_target / port_vol
                # 缩放后仍须满足杠杆上限
                proposed_lev = np.sum(np.abs(w_opt * scale))
                if proposed_lev > self.constraints.leverage_cap:
                    scale *= self.constraints.leverage_cap / proposed_lev
                w_opt = w_opt * scale

        return pd.Series(w_opt, index=symbols)
