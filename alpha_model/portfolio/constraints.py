"""
组合约束（cvxpy 约束生成器）

所有约束被建模为 cvxpy 约束表达式，传入优化器联合求解。
不再有顺序应用的问题——所有约束同时满足。

约束分层（可单独启用或组合）：

第一层 — 仓位上限:
    |w_i| ≤ max_weight
    cvxpy: cp.abs(w) <= max_weight

第二层 — Dollar-neutral:
    Σw_i = 0  (多空完全对冲)
    cvxpy: cp.sum(w) == 0

第三层 — Beta-neutral:
    β'w = 0  (组合对市场的 beta 暴露 = 0)
    cvxpy: beta @ w == 0

第四层 — 杠杆上限:
    Σ|w_i| ≤ leverage_cap
    cvxpy: cp.norm(w, 1) <= leverage_cap

依赖: cvxpy
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import cvxpy as cp

from alpha_model.core.types import PortfolioConstraints


def build_constraints(
    w: "cp.Variable",
    config: PortfolioConstraints,
    beta: np.ndarray | None = None,
) -> list:
    """
    根据配置生成 cvxpy 约束列表

    Args:
        w:      cvxpy 权重变量（一维，长度 = N_symbols）
        config: 约束配置
        beta:   各标的 beta 向量（长度 = N_symbols），beta-neutral 需要

    Returns:
        cvxpy 约束列表
    """
    import cvxpy as cp

    constraints = []

    # 第一层: 仓位上限 |w_i| ≤ max_weight
    constraints.append(cp.abs(w) <= config.max_weight)

    # 第二层: Dollar-neutral Σw_i = 0
    if config.dollar_neutral:
        constraints.append(cp.sum(w) == 0)

    # 第三层: Beta-neutral β'w = 0
    if config.beta_neutral:
        if beta is None:
            raise ValueError(
                "启用 beta_neutral 约束但未提供 beta 向量"
            )
        constraints.append(beta @ w == 0)

    # 第四层: 杠杆上限 Σ|w_i| ≤ leverage_cap
    constraints.append(cp.norm(w, 1) <= config.leverage_cap)

    return constraints
