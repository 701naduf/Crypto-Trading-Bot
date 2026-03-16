"""
信号 → 目标权重（cvxpy 凸优化）

核心思想: Mean-Variance 优化
    将所有约束建模为凸优化问题，cvxpy 联合求解。
    所有约束同时满足，不存在顺序规则的冲突问题。

处理流程:
1. 估计协方差矩阵（Ledoit-Wolf shrinkage，防止估计误差放大）
2. 构建优化问题:
   - 目标: 最小化 风险 - λ×收益 + γ×换手率
   - 约束: 仓位上限、dollar/beta-neutral、杠杆上限
3. 求解并返回最优权重
4. 波动率目标: 事后缩放使预期 vol ≈ target

QP 问题:
    minimize    w' Σ w - λ α' w + γ ||w - w_prev||₁
    subject to  |w_i| ≤ max_weight
                1' w = 0                      (dollar-neutral, 可选)
                β' w = 0                      (beta-neutral, 可选)
                ||w||₁ ≤ leverage_cap

依赖: portfolio.covariance, portfolio.beta, portfolio.constraints, cvxpy
"""

from __future__ import annotations

import logging

import cvxpy as cp
import numpy as np
import pandas as pd

from alpha_model.core.types import PortfolioConstraints
from alpha_model.portfolio.beta import rolling_beta
from alpha_model.portfolio.constraints import build_constraints
from alpha_model.portfolio.covariance import estimate_covariance
from alpha_model.portfolio.risk_budget import apply_vol_target

logger = logging.getLogger(__name__)


class PortfolioConstructor:
    """
    凸优化组合构建器

    用法:
        constraints = PortfolioConstraints(dollar_neutral=True, max_weight=0.4)
        constructor = PortfolioConstructor(constraints)
        weights = constructor.construct(signal, price_panel)
    """

    def __init__(self, constraints: PortfolioConstraints):
        self.constraints = constraints

    def construct(
        self,
        signal: pd.DataFrame,
        price_panel: pd.DataFrame,
        prev_weights: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        信号 → 目标权重（逐期求解凸优化问题）

        Args:
            signal:       信号面板 (timestamp × symbol)
            price_panel:  价格面板 (timestamp × symbol)，用于协方差估计和 beta
            prev_weights: 初始持仓权重面板 (timestamp × symbol)。
                          仅用于第一期的换手率惩罚基准。
                          如果提供，取 signal 起始时间之前（不含）最近的一行作为初始持仓。
                          None 则从零仓位开始。后续各期由循环自动追踪。

        Returns:
            目标权重面板 (timestamp × symbol)
        """
        symbols = signal.columns.tolist()
        n_symbols = len(symbols)

        # 计算收益率面板（用于协方差和 beta）
        returns_panel = price_panel[symbols].pct_change().dropna()

        # 滚动 beta（如需要）
        beta_panel = None
        if self.constraints.beta_neutral:
            beta_panel = rolling_beta(
                returns_panel,
                lookback=self.constraints.beta_lookback,
            )

        # 逐期求解
        weights_dict = {}

        # [P6+R1] 确定初始 prev_w
        # prev_weights 仅用于初始化第一期的换手率惩罚基准，
        # 后续各期由循环内 prev_w = w_opt 自然追踪
        prev_w = np.zeros(n_symbols)
        if prev_weights is not None:
            first_ts = signal.index[0]
            # 取 signal 起始时间之前（严格小于 first_ts）最近的一行
            prior = prev_weights.loc[prev_weights.index < first_ts]
            if len(prior) > 0:
                prev_w = prior.iloc[-1].reindex(symbols).fillna(0).values

        for ts in signal.index:
            alpha = signal.loc[ts].reindex(symbols).values

            # 如果信号全为 NaN，跳过（权重为 0）
            if np.all(np.isnan(alpha)):
                weights_dict[ts] = np.zeros(n_symbols)
                continue

            # 用 0 填充 NaN（无信号的标的不参与）
            alpha = np.nan_to_num(alpha, nan=0.0)

            # 估计协方差矩阵
            # 使用当前时刻之前的数据（不含当前行，避免前瞻）
            returns_before = returns_panel.loc[:ts].iloc[:-1] if ts in returns_panel.index else returns_panel.loc[:ts]
            cov_lookback = min(
                self.constraints.vol_lookback, len(returns_before),
            )

            try:
                cov_matrix = estimate_covariance(
                    returns_before[symbols],
                    lookback=cov_lookback,
                    method="ledoit_wolf",
                )
            except Exception as e:
                logger.warning("时刻 %s 协方差估计失败: %s, 使用上一期权重", ts, e)
                weights_dict[ts] = prev_w.copy()
                continue

            # 获取 beta 向量
            beta_vec = None
            if beta_panel is not None and ts in beta_panel.index:
                beta_vec = beta_panel.loc[ts].reindex(symbols).values
                if np.any(np.isnan(beta_vec)):
                    beta_vec = None  # beta 不完整时退化为不使用 beta-neutral

            # 求解单期优化
            try:
                w_opt = self._solve_single_period(
                    alpha=alpha,
                    cov_matrix=cov_matrix,
                    prev_w=prev_w,
                    beta=beta_vec,
                )
            except Exception as e:
                logger.warning("时刻 %s 优化求解失败: %s, 使用上一期权重", ts, e)
                w_opt = prev_w.copy()

            weights_dict[ts] = w_opt
            prev_w = w_opt

        # 构建权重面板
        weights_df = pd.DataFrame(
            weights_dict, index=symbols,
        ).T
        weights_df.index.name = signal.index.name

        # 波动率目标缩放（事后）
        if self.constraints.vol_target is not None:
            weights_df = apply_vol_target(
                weights_df, price_panel,
                vol_target=self.constraints.vol_target,
                lookback=self.constraints.vol_lookback,
                leverage_cap=self.constraints.leverage_cap,
            )

        return weights_df

    def _solve_single_period(
        self,
        alpha: np.ndarray,
        cov_matrix: np.ndarray,
        prev_w: np.ndarray,
        beta: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        单期凸优化求解

        cvxpy 建模:
            w = cp.Variable(n)
            risk = cp.quad_form(w, cov_matrix)
            ret = alpha @ w
            turnover = cp.norm(w - prev_w, 1)
            objective = cp.Minimize(risk - lambda_ * ret + gamma * turnover)

        Returns:
            np.ndarray: 最优权重向量
        """
        n = len(alpha)

        # 确保协方差矩阵是正半定的（加微小正则项）
        cov_matrix = cov_matrix.copy()
        min_eig = np.linalg.eigvalsh(cov_matrix).min()
        if min_eig < 0:
            cov_matrix += (-min_eig + 1e-8) * np.eye(n)

        w = cp.Variable(n)

        # 目标函数
        risk = cp.quad_form(w, cov_matrix)
        ret = alpha @ w
        turnover = cp.norm(w - prev_w, 1)

        lambda_ = self.constraints.risk_aversion
        gamma = self.constraints.turnover_penalty

        objective = cp.Minimize(risk - lambda_ * ret + gamma * turnover)

        # 约束
        constraints = build_constraints(w, self.constraints, beta=beta)

        # 求解
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)

        if prob.status in ("infeasible", "unbounded"):
            logger.warning(
                "优化问题状态: %s, 退化为等权", prob.status,
            )
            # 退化为等权（满足 dollar-neutral 则为 0）
            if self.constraints.dollar_neutral:
                return np.zeros(n)
            else:
                return np.ones(n) * self.constraints.leverage_cap / n

        if w.value is None:
            raise RuntimeError(f"cvxpy 求解失败, status={prob.status}")

        return w.value
