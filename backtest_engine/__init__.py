"""
backtest_engine — Phase 3 事件驱动回测引擎

顶层公开接口:
    EventDrivenBacktester  — 入口（无状态，复用同实例多次 run）
    BacktestConfig         — 配置契约（kw_only dataclass）
    BacktestReport         — 交付对象（组合持有 BacktestResult）
    RunMode / ExecutionMode / CostMode  — 三个运行模式枚举

详见 docs/phase3_design.md。
"""
from backtest_engine.config import (
    BacktestConfig, RunMode, ExecutionMode, CostMode,
)
from backtest_engine.engine import EventDrivenBacktester
from backtest_engine.report import BacktestReport

__all__ = [
    "BacktestConfig",
    "BacktestReport",
    "EventDrivenBacktester",
    "RunMode",
    "ExecutionMode",
    "CostMode",
]
