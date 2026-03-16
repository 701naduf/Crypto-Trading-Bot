"""
Alpha Model 模块 (Phase 2b)

将因子面板转化为可交易的目标持仓权重。

核心理念：框架做管道，模型做黑盒。
框架负责数据流编排（预处理→训练→信号→组合→回测），
模型实现完全外接，通过 AlphaModel 协议交互。

模块结构:
    core/           核心协议与管道
    preprocessing/  因子预处理（筛选、对齐、标准化工具箱）
    training/       训练框架（时序切分、Walk-Forward）
    signal/         信号生成（截面标准化、平滑）
    portfolio/      组合构建（cvxpy 凸优化）
    backtest/       向量化回测（快速验证）
    store/          持久化（SignalStore、ModelStore）
    models/         参考模型实现（示例代码，非核心架构）

上游接口:
    factor_research.store.factor_store.FactorStore  — 因子面板加载
    data_infra.data.reader.DataReader               — 价格数据加载

下游接口:
    store.signal_store.SignalStore  — 策略输出（权重面板），Phase 3 唯一输入
"""