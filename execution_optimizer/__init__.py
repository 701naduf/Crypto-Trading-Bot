"""
execution_optimizer — 动态成本感知的组合优化器

Phase 2b PortfolioConstructor 的事件驱动替代方案。
唯一公开方法：ExecutionOptimizer.optimize_step()
"""
from execution_optimizer.config import MarketContext
from execution_optimizer.optimizer import ExecutionOptimizer

__all__ = ["ExecutionOptimizer", "MarketContext"]
