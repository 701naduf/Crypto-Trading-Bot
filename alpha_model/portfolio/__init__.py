"""
组合构建（cvxpy 凸优化）

包含:
    constructor.py   信号 → 目标权重（cvxpy QP 求解主入口）
    covariance.py    协方差矩阵估计（Ledoit-Wolf shrinkage）
    constraints.py   约束生成器（cvxpy 约束表达式）
    risk_budget.py   波动率目标（vol targeting）
    beta.py          滚动 beta 估计
"""