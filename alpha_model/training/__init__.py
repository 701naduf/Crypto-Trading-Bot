"""
训练框架

包含:
    splitter.py       时序切分器（expanding/rolling + embargo period）
    walk_forward.py   Walk-Forward 引擎（调用 AlphaModel 协议）
    trainer.py        训练调度器（Pooled/Per-Symbol 模式）
"""