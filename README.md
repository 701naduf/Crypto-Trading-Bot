# Crypto Trading Bot — 量化加密货币交易框架

量化加密货币交易机器人，聚焦 Binance 交易所。模块化架构，各阶段独立开发。

## 项目结构

```
Crypto-Trading-Bot/
├── data_infra/                 # 第一阶段：数据基础设施
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py         #   全局配置（交易所、币对、路径、容错等）
│   ├── utils/
│   │   ├── logger.py           #   统一日志（控制台 + 文件，按日轮转）
│   │   ├── time_utils.py       #   时间工具（ms ↔ datetime, 时间对齐）
│   │   ├── retry.py            #   重试装饰器（异常分类 + 指数退避）
│   │   └── heartbeat.py        #   运行状态监控（心跳日志 + 状态文件）
│   ├── data/
│   │   ├── fetcher.py          #   K线拉取（同步 REST）
│   │   ├── tick_fetcher.py     #   逐笔成交拉取（异步, trade_id 追赶）
│   │   ├── orderbook_fetcher.py#   订单簿采集（异步 WebSocket）
│   │   ├── market_fetcher.py   #   合约市场数据拉取（同步 REST）
│   │   ├── kline_store.py      #   SQLite K线存储（WAL 模式）
│   │   ├── tick_store.py       #   Parquet 逐笔成交存储（按天分区, 原子写入）
│   │   ├── orderbook_store.py  #   Parquet 订单簿存储（缓冲 + 刷盘）
│   │   ├── market_store.py     #   SQLite 合约市场数据存储
│   │   ├── aggregator.py       #   数据聚合（tick→OHLCV, K线降采样）
│   │   ├── validator.py        #   数据质量校验
│   │   ├── writer.py           #   统一写入入口（校验→存储）
│   │   └── reader.py           #   统一读取入口（自动路由 + 聚合）
│   ├── scripts/
│   │   ├── collect_klines.py   #   K线采集（独立进程, 7×24）
│   │   ├── collect_ticks.py    #   逐笔成交采集（独立进程, 7×24）
│   │   ├── collect_orderbook.py#   订单簿采集（独立进程, WebSocket, 7×24）
│   │   ├── collect_market.py   #   合约市场数据采集（独立进程, 7×24）
│   │   ├── backfill.py         #   历史数据回填
│   │   ├── check_data.py       #   数据质量巡检
│   │   └── status.py           #   查看所有采集脚本运行状态
│   └── tests/                  #   单元测试（125 项）
│
├── factor_research/            # 第二阶段(a)：因子研究框架
│   ├── config.py               #   集中配置（对 data_infra 的唯一引用入口）
│   ├── core/                   #   核心层
│   │   ├── types.py            #     类型定义（FactorType, DataType, FactorMeta 等）
│   │   ├── base.py             #     因子基类（Factor, TimeSeriesFactor, CrossSectional, CrossAsset）
│   │   ├── registry.py         #     因子注册表 + @register_factor / @register_factor_family 装饰器
│   │   └── engine.py           #     因子计算引擎（数据读取→计算→存储）
│   ├── evaluation/             #   因子评价体系（三层 API）
│   │   ├── metrics.py          #     底层计算原语（IC, 收益率, 排名, 夏普等）
│   │   ├── ic.py               #     IC/IR/衰减分析
│   │   ├── quantile.py         #     分层回测（分组收益、单调性）
│   │   ├── tail.py             #     尾部特征分析（含MAE）
│   │   ├── stability.py        #     稳定性分析（分regime/月度/滚动IC）
│   │   ├── nonlinear.py        #     非线性分析（互信息、分bin收益）
│   │   ├── turnover.py         #     换手率分析（自相关、信号翻转）
│   │   ├── correlation.py      #     因子相关性 + VIF + 增量IC
│   │   ├── report.py           #     文本报告 + matplotlib 可视化
│   │   ├── analyzer.py         #     FactorAnalyzer 一键报告入口
│   │   └── family_analyzer.py  #     因子族参数分析器（FamilyAnalyzer）
│   ├── alignment/              #   异步数据对齐
│   │   ├── grid.py             #     网格对齐（等间距 + ffill + max_gap）
│   │   ├── refresh_time.py     #     刷新时间采样
│   │   └── hayashi_yoshida.py  #     Hayashi-Yoshida 异步协方差估计
│   ├── store/                  #   因子存储
│   │   ├── factor_store.py     #     FactorStore（Parquet + JSON，原子写入）
│   │   └── catalog.py          #     FactorCatalog（因子目录扫描与检索）
│   ├── factors/                #   具体因子实现
│   │   ├── microstructure/
│   │   │   └── imbalance.py    #     订单簿不平衡度因子
│   │   ├── momentum/
│   │   │   └── returns.py      #     多尺度收益率因子族（@register_factor_family, lookback=[5,10,30,60]）
│   │   ├── volatility/         #     （预留）波动率因子
│   │   ├── orderflow/          #     （预留）订单流因子
│   │   └── cross_asset/        #     （预留）跨标的因子
│   └── tests/                  #   单元测试（200 项）
│       ├── test_types.py
│       ├── test_base.py
│       ├── test_registry.py
│       ├── test_engine.py
│       ├── test_factor_store.py
│       ├── test_catalog.py
│       ├── test_factors.py
│       ├── test_evaluation.py
│       ├── test_alignment.py
│       └── test_family_analyzer.py
│
├── alpha_model/                # 第二阶段(b)：Alpha 模型框架
│   ├── config.py               #   集中配置（存储路径、默认参数、年化常数）
│   ├── core/                   #   核心层
│   │   ├── types.py            #     AlphaModel Protocol、TrainConfig、PortfolioConstraints、ModelMeta
│   │   └── pipeline.py         #     AlphaPipeline 一键管道（预处理→训练→信号→组合→回测）
│   ├── preprocessing/          #   预处理
│   │   ├── alignment.py        #     多频率因子对齐（自动推断 + 统一网格）
│   │   ├── transform.py        #     标准化工具箱（expanding/rolling/截面 z-score、去极值）+ 特征矩阵构建
│   │   └── selection.py        #     因子筛选（threshold / top_k / 族级筛选）
│   ├── training/               #   训练框架
│   │   ├── splitter.py         #     时序切分器（Expanding / Rolling + Embargo）
│   │   ├── walk_forward.py     #     Walk-Forward 引擎（Pooled / Per-Symbol）
│   │   └── trainer.py          #     训练调度器（一站式接口）
│   ├── signal/                 #   信号生成
│   │   ├── generator.py        #     预测值 → 标准化信号（截面 z-score / rank）
│   │   └── smoother.py         #     信号平滑（EMA）
│   ├── portfolio/              #   组合构建
│   │   ├── constructor.py      #     凸优化组合构建器（Mean-Variance + cvxpy）
│   │   ├── constraints.py      #     cvxpy 约束生成器（仓位/dollar/beta-neutral/杠杆）
│   │   ├── covariance.py       #     协方差矩阵估计（Ledoit-Wolf / 样本 / 指数加权）
│   │   ├── beta.py             #     滚动 Beta 估计（OLS）
│   │   └── risk_budget.py      #     波动率目标（动态杠杆缩放）
│   ├── backtest/               #   向量化回测
│   │   ├── vectorized.py       #     向量化 P&L（手续费 + 市场冲击）
│   │   └── performance.py      #     绩效指标汇总（BacktestResult）
│   ├── store/                  #   持久化
│   │   ├── signal_store.py     #     策略输出存储（权重 + 信号 + 原始预测 + 元数据 + 绩效）
│   │   └── model_store.py      #     模型存储（模型文件 + 元数据 + 因子重要性）
│   ├── models/                 #   参考模型实现（示例，非核心架构）
│   │   ├── linear_models.py    #     SklearnModelWrapper（Ridge/Lasso/ElasticNet）
│   │   ├── tree_models.py      #     LGBMModelWrapper / XGBModelWrapper
│   │   └── torch_base.py       #     TorchModelBase（PyTorch 基础封装）
│   ├── utils.py                #   辅助工具（load_price_panel 等）
│   └── tests/                  #   单元测试（191 项）
│       ├── test_types.py
│       ├── test_preprocessing.py
│       ├── test_training.py
│       ├── test_signal.py
│       ├── test_portfolio.py
│       ├── test_backtest.py
│       ├── test_store.py
│       ├── test_models.py
│       └── test_pipeline.py
│
├── execution_optimizer/        # 第二阶段(c)：执行优化器
│   ├── __init__.py             #   公开导出 ExecutionOptimizer, MarketContext
│   ├── config.py               #   MarketContext 数据类 + 默认常量
│   ├── cost.py                 #   动态成本表达式（commission + spread + 1.5次幂 impact）
│   ├── optimizer.py            #   ExecutionOptimizer.optimize_step() 单步优化
│   └── tests/                  #   单元测试（36 项）
│       ├── test_cost.py        #     成本函数数值/单调/零值
│       └── test_optimizer.py   #     可解性/成本压制/ADV/资金费率/VolTarget/Fallback/Beta
│
├── backtest_engine/            # 第三阶段：事件驱动回测引擎
│   ├── __init__.py             #   公开导出 EventDrivenBacktester / BacktestConfig / BacktestReport / 三模式枚举
│   ├── config.py               #   BacktestConfig (kw_only) + RunMode/ExecutionMode/CostMode 枚举
│   ├── context.py              #   MarketContextBuilder（DataReader → MarketContext + build_panels）
│   ├── weights_source.py       #   WeightsSource Protocol + PrecomputedWeights + OnlineOptimizer
│   ├── rebalancer.py           #   Rebalancer 执行模拟层（v1 仅 MARKET, cost_mode 自持）
│   ├── pnl.py                  #   PnLTracker P&L 状态中心 + NumericalError fail-fast
│   ├── engine.py               #   EventDrivenBacktester 顶层协调（按 RunMode 分流）
│   ├── attribution.py          #   cost 分解 + 偏差归因 + regime 分段（纯函数）
│   ├── report.py               #   BacktestReport 容器 + parquet/JSON 原子持久化
│   ├── plot.py                 #   8 张标准量化回测图（matplotlib）
│   ├── reporting.py            #   to_markdown 单文件报告生成
│   └── tests/                  #   单元测试（219 项）
│       ├── test_config.py      #     16 项 __post_init__ 校验
│       ├── test_context.py     #     批量/逐步双模式 + warmup 校验
│       ├── test_weights_source.py #  Precomputed / Online + cost_mode 自持
│       ├── test_rebalancer.py  #     执行模拟 + ExecutionReport schema
│       ├── test_pnl.py         #     破产双通道（V≤0 标志 / NaN-Inf 抛异常）
│       ├── test_engine.py      #     8 项 _validate_environment + (a') (f) 早退 + 4 项稳健性链路
│       ├── test_attribution.py #     cost_decomposition / deviation / regime / per_symbol
│       ├── test_report.py      #     summary / to_dict / save_load 原子化
│       ├── test_plot.py        #     8 张图返回类型 + headless
│       ├── test_reporting.py   #     markdown 渲染 + 跨 section 缺失
│       └── test_consistency.py #   ★ 跨模块护栏：4 处 impact 公式一致 + 零摩擦严格等价 (rtol=1e-12)
│
├── db/                         # 共享数据存储（各模块通过此目录通信）
│   ├── factors/                #   因子存储目录（FactorStore 的数据根）
│   ├── signals/                #   策略输出目录（SignalStore 的数据根）
│   └── models/                 #   模型存储目录（ModelStore 的数据根）
├── logs/                       # 日志 + 状态文件
├── docs/                       # 设计文档
├── study/                      # 学习笔记
├── notebooks/                  # Jupyter 研究笔记
├── main.py                     # 主入口
├── requirements.txt
└── .env                        # API 密钥（.gitignore）
```

---

## 第一阶段：数据基础设施 (data_infra)

### 采集的数据类型

| 数据类型 | 采集方式 | 频率 | 存储 | 脚本 |
|---------|---------|------|------|------|
| 1m K线 | REST 轮询 | 60s | SQLite | `collect_klines.py` |
| 逐笔成交 | REST trade_id追赶 | 持续 | Parquet/天 | `collect_ticks.py` |
| 10档订单簿 | WebSocket推送 | 100ms | Parquet/天 | `collect_orderbook.py` |
| 合约市场数据 | REST 轮询 | 5min | SQLite | `collect_market.py` |

合约市场数据包含：资金费率、持仓量(OI)、多空持仓比、主动买卖量。

### 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 创建 .env 文件（配置 API 密钥）
BINANCE_API_KEY=your_api_key_here
BINANCE_SECRET=your_secret_here

# 3. 运行数据基础设施测试
python -m pytest data_infra/tests/ -v    # 预期 125 项全通过

# 4. 启动采集脚本（每个独立进程）
python -m data_infra.scripts.collect_klines

# 逐笔成交采集（持续追赶模式）
python -m data_infra.scripts.collect_ticks

# 订单簿采集（WebSocket 100ms推送）
python -m data_infra.scripts.collect_orderbook

# 合约市场数据采集（每5分钟轮询）
python -m data_infra.scripts.collect_market

# 查看状态 / 数据巡检
python -m data_infra.scripts.status
```

### 6. 数据质量巡检

```bash
python -m data_infra.scripts.check_data
```

---

## 第二阶段(a)：因子研究框架 (factor_research)

### 概述

因子研究框架提供从"因子定义 → 计算 → 存储 → 评价"的完整管道。
核心设计原则：

- **统一面板格式**：所有因子输出为 `pd.DataFrame (timestamp × symbol)`
- **三种因子类型**：时序、截面、跨标的，覆盖不同信号来源
- **三层评价 API**：从一键报告到底层原语，灵活性递增
- **模块解耦**：factor_research 仅通过 DataReader 和 FactorStore 与外部通信

### 三种因子类型

| 类型 | 基类 | 数据输入 | 典型用例 |
|------|------|---------|---------|
| 时序因子 | `TimeSeriesFactor` | 单标的数据 → 单标的因子 | BTC 订单簿不平衡度 |
| 截面因子 | `CrossSectionalFactor` | 全标的数据 → 截面排名 | 5 币种动量排名 |
| 跨标的因子 | `CrossAssetFactor` | 指定标的 → 其他标的因子 | BTC 领先-ETH 滞后 |

### 编写新因子

#### 时序因子（最常用）

```python
from factor_research.core.base import TimeSeriesFactor
from factor_research.core.registry import register_factor
from factor_research.core.types import *

@register_factor
class MyFactor(TimeSeriesFactor):
    def meta(self) -> FactorMeta:
        return FactorMeta(
            name="my_factor",
            display_name="我的因子",
            factor_type=FactorType.TIME_SERIES,
            category="momentum",
            description="过去 10 根 K线的累计收益率",
            data_requirements=[
                DataRequest(DataType.OHLCV, timeframe="1m", lookback_bars=10),
            ],
            output_freq="1m",
            params={"lookback": 10},
        )

    def compute_single(self, symbol, data):
        ohlcv = data[DataType.OHLCV]
        close = ohlcv["close"]
        returns = close / close.shift(10) - 1
        returns.index = pd.to_datetime(ohlcv["timestamp"], utc=True)
        return returns.dropna()
```

#### 参数化因子族（@register_factor_family）

通过 `_param_grid` 类属性声明参数网格，装饰器自动展开为多个因子实例：

```python
from factor_research.core.registry import register_factor_family

@register_factor_family
class MultiScaleReturns(TimeSeriesFactor):
    _param_grid = {"lookback": [5, 10, 30, 60]}  # 作者人工指定

    def __init__(self, lookback: int = 5):
        self.lookback = lookback

    def meta(self):
        return FactorMeta(
            name=f"returns_{self.lookback}m",
            family="multi_scale_returns",  # 族名
            params={"lookback": self.lookback},
            ...
        )

    def compute_single(self, symbol, data):
        close = data[DataType.OHLCV]["close"]
        return (close / close.shift(self.lookback) - 1).dropna()
# → 自动注册 4 个因子: returns_5m, returns_10m, returns_30m, returns_60m
```

参数可以是任意类型（数字、字符串等），多参数自动做笛卡尔积展开。

### 因子计算引擎

```python
from factor_research.core.engine import FactorEngine

engine = FactorEngine()

# 单因子计算（自动保存到 FactorStore）
panel = engine.compute_factor("returns_5m", symbols=["BTC/USDT", "ETH/USDT"])

# 批量计算所有已注册因子
results = engine.compute_all()

# 按分类计算
results = engine.compute_all(categories=["momentum", "microstructure"])
```

### 因子族参数扫描 (FamilyAnalyzer)

对参数化因子族进行参数空间扫描，快速定位最优参数区域：

```python
from factor_research.evaluation.family_analyzer import FamilyAnalyzer

# 1. 参数扫描
family = FamilyAnalyzer(
    factor_class=MultiScaleReturns,
    data=prepared_data,        # 引擎准备好的数据（复用，避免重复 I/O）
    price_panel=prices,
)
sweep_df = family.sweep()      # 遍历参数网格 × 前瞻窗口 → 轻量指标表

# 2. 可视化
family.plot_sensitivity(metric="ic_ir")    # 参数敏感性折线图
family.plot_heatmap(metric="ic_ir")        # 双参数热力图

# 3. 自动筛选
candidates = family.select(min_ic_ir=0.3, top_n=3)

# 4. 钻取详细报告（调用 FactorAnalyzer.full_report()）
report = family.detail(lookback=10)
```

### 因子存储 (FactorStore)

因子通过 FactorStore 持久化，存储在 `db/factors/{factor_name}/` 下：
- `output.parquet`：因子面板数据
- `meta.json`：因子元数据

```python
from factor_research.store.factor_store import FactorStore

store = FactorStore()
store.save("my_factor", panel, meta)           # 保存
loaded = store.load("my_factor")               # 加载
store.list_factors()                           # 列出已有因子
store.exists("my_factor")                      # 检查是否存在
store.delete("my_factor")                      # 删除
```

### 因子评价体系

三层 API 设计，灵活性递增：

#### 第一层：FactorAnalyzer（一键报告）

```python
from factor_research.evaluation.analyzer import FactorAnalyzer

analyzer = FactorAnalyzer(factor_panel, price_panel)

# 一键完整报告
report = analyzer.full_report(horizons=[1, 5, 10, 30, 60])

# 文本摘要
print(analyzer.summary_text(factor_name="my_factor"))

# 可视化图表
figs = analyzer.plot(factor_name="my_factor")
```

#### 第二层：独立分析函数

| 函数 | 模块 | 功能 |
|------|------|------|
| `ic_analysis()` | `evaluation.ic` | IC/IR 均值、标准差、胜率、衰减曲线 |
| `quantile_backtest()` | `evaluation.quantile` | 分层组收益、多空收益、单调性检验 |
| `tail_analysis()` | `evaluation.tail` | 尾部条件 IC、尾部命中率、尾部频率 |
| `stability_analysis()` | `evaluation.stability` | 分 regime IC、月度 IC、滚动 IC |
| `nonlinear_analysis()` | `evaluation.nonlinear` | 互信息、因子 profile、条件 IC |
| `turnover_analysis()` | `evaluation.turnover` | 自相关、排名变化率、信号翻转率 |
| `correlation_analysis()` | `evaluation.correlation` | 因子相关矩阵、VIF、增量 IC |

```python
from factor_research.evaluation.ic import ic_analysis
from factor_research.evaluation.quantile import quantile_backtest

ic_result = ic_analysis(factor_panel, price_panel, horizons=[1, 5, 10])
qt_result = quantile_backtest(factor_panel, price_panel, n_groups=5, horizon=1)
```

#### 第三层：底层计算原语

```python
from factor_research.evaluation.metrics import (
    spearman_ic,             # Spearman 秩相关
    pearson_ic,              # Pearson 相关
    compute_forward_returns, # 单标的前瞻收益
    rank_normalize,          # 秩归一化
    cross_sectional_rank,    # 截面排名
    zscore_normalize,        # Z-score 标准化
    sharpe_ratio,            # 夏普比
    max_drawdown,            # 最大回撤
    mutual_information,      # 互信息
)
```

### 数据对齐 (alignment)

处理异步到达的高频数据（tick、订单簿），将不规则时间序列对齐为同步面板：

| 方法 | 模块 | 适用场景 |
|------|------|---------|
| 网格对齐 | `alignment.grid` | 将多标的数据对齐到等间距网格（ffill + max_gap） |
| 刷新时间采样 | `alignment.refresh_time` | 在所有标的都刷新时取样 |
| Hayashi-Yoshida | `alignment.hayashi_yoshida` | 异步数据的协方差/相关性估计（无信息损失） |

```python
from factor_research.alignment.grid import grid_align
from factor_research.alignment.hayashi_yoshida import hy_correlation_matrix

# 将 tick 级数据对齐到 1 秒网格（max_gap 支持 int 或 pd.Timedelta）
panel = grid_align({"BTC": btc_series, "ETH": eth_series}, freq="1s", max_gap=pd.Timedelta("5s"))

# 计算异步相关矩阵
corr = hy_correlation_matrix({"BTC": btc_prices, "ETH": eth_prices})
```

### 已实现的示例因子

| 因子/族 | 分类 | 数据源 | 说明 |
|---------|------|--------|------|
| `orderbook_imbalance` | microstructure | 10档订单簿 | 买卖挂单量不平衡度 [-1, 1]（独立因子） |
| `multi_scale_returns` 族 | momentum | 1m OHLCV | 多尺度收益率因子族（`_param_grid = {"lookback": [5, 10, 30, 60]}`），自动注册 returns_5m / returns_10m / returns_30m / returns_60m |

### 因子评价指标参考基准

因子评价的核心目标是回答一个问题：**这个因子是否包含对未来收益的预测信息？** 以下是各指标的经验阈值和判读方法。

#### IC / IC_IR 经验阈值

| 指标 | 阈值 | 判读 |
|------|------|------|
| \|IC\| | < 0.02 | 无实际预测能力，可能是噪声 |
| \|IC\| | 0.02 ~ 0.05 | 有一定信息量，可继续研究 |
| \|IC\| | > 0.05 | 较强信号，crypto 中已属优秀 |
| IC_IR (IC均值/IC标准差) | < 0.5 | 信号不稳定，实盘价值有限 |
| IC_IR | 0.5 ~ 1.0 | 可用，建议结合其他因子使用 |
| IC_IR | > 1.0 | 优秀，信号稳定，可作为核心因子 |
| IC 胜率 | > 50% | 多数截面上因子方向一致 |

> **注意**: 以上阈值基于 crypto 1m 频率、5 标的的场景。传统股票市场（3000+ 标的）的 IC 通常更低但更稳定。crypto 标的少，IC 数值会偏高但方差也大，IC_IR 比 IC 绝对值更有参考意义。

#### 分层回测单调性判读

分层回测将因子值按截面排名分为 N 组（默认 5 组），分别计算每组的前瞻收益。

| 现象 | 判读 |
|------|------|
| Q1 到 Q5 收益单调递增/递减 | 因子预测方向一致，理想状态 |
| 多空收益（Q5 - Q1）显著非零 | 因子有多空收益来源 |
| 中间组收益非单调 | 因子在中间区域预测力弱，仅尾部有效 |
| 单调性但 Q1 和 Q5 差距极小 | IC 可能由噪声驱动，实际不可交易 |

#### 指标使用优先级

实际因子研究中，建议按以下优先级逐步深入：

1. **IC / IC_IR** — 最核心，快速筛选（`ic_analysis()`）
2. **分层回测** — 验证 IC 的经济含义是否成立（`quantile_backtest()`）
3. **稳定性分析** — 确认信号不是特定时段的偶然现象（`stability_analysis()`）
4. **换手率分析** — 评估交易成本可行性（`turnover_analysis()`）
5. **尾部分析** — 极端信号的风险收益特征（`tail_analysis()`）
6. **非线性分析** — IC 遗漏的非线性信息（`nonlinear_analysis()`）
7. **多因子相关性** — 入库前检查冗余（`correlation_analysis()` + `incremental_ic()`）

### API Quick Reference

所有公开接口按使用场景组织。

#### 因子计算

| 函数/方法 | 模块 | 说明 |
|-----------|------|------|
| `FactorEngine.compute_factor(name, symbols, save)` | `core.engine` | 计算单个因子，返回面板 DataFrame |
| `FactorEngine.compute_factor_instance(factor, symbols, save)` | `core.engine` | 计算因子实例（无需注册，适合 notebook 探索） |
| `FactorEngine.compute_family(family_name, symbols, save)` | `core.engine` | 计算因子族所有变体（共享数据准备，减少 I/O） |
| `FactorEngine.compute_all(categories, symbols, save)` | `core.engine` | 批量计算所有因子（自动按族分组共享数据） |
| `FactorEngine.prepare_data(factor, symbols)` | `core.engine` | 暴露数据准备接口（供 FamilyAnalyzer 复用） |

#### 因子评价 — 第一层（一键报告）

| 函数/方法 | 模块 | 说明 |
|-----------|------|------|
| `FactorAnalyzer(factor_panel, prices)` | `evaluation.analyzer` | 初始化分析器 |
| `.full_report(horizons)` | `evaluation.analyzer` | 全维度综合报告，返回嵌套 dict |
| `.summary_text(factor_name)` | `evaluation.analyzer` | 文本摘要（适合 notebook 打印） |
| `.plot(factor_name)` | `evaluation.analyzer` | 生成 matplotlib 可视化图表集 |

#### 因子评价 — 第二层（独立分析函数）

| 函数 | 模块 | 输入 | 输出关键字段 |
|------|------|------|-------------|
| `ic_analysis(factor_panel, price_panel, horizons)` | `evaluation.ic` | 因子面板 + 价格面板 | ic_mean, ic_std, ic_ir, win_rate, ic_decay |
| `quantile_backtest(factor_panel, price_panel, n_groups, horizon, grouping)` | `evaluation.quantile` | 因子面板 + 价格面板 | group_returns, long_short, monotonicity |
| `tail_analysis(factor_panel, price_panel, threshold, horizon)` | `evaluation.tail` | 因子面板 + 价格面板 | tail_ic, tail_hit_rate, tail_freq, mae |
| `stability_analysis(factor_panel, price_panel, horizon)` | `evaluation.stability` | 因子面板 + 价格面板 | regime_ic, monthly_ic, rolling_ic, ic_max_drawdown |
| `nonlinear_analysis(factor_panel, price_panel, horizon)` | `evaluation.nonlinear` | 因子面板 + 价格面板 | mutual_info, factor_profile, conditional_ic |
| `turnover_analysis(factor_panel)` | `evaluation.turnover` | 因子面板 | autocorrelation, rank_change_rate, signal_flip_rate |
| `correlation_analysis(factor_panels)` | `evaluation.correlation` | {名称: 面板} dict | correlation_matrix, vif |
| `incremental_ic(new_factor, existing_factors, price_panel, horizon)` | `evaluation.correlation` | 新因子 + 已有因子 + 价格 | raw_ic, incremental_ic, info_retention |

#### 因子评价 — 第三层（底层原语）

| 函数 | 模块 | 说明 |
|------|------|------|
| `spearman_ic(factor, returns)` | `evaluation.metrics` | Spearman 秩相关（标准 IC） |
| `pearson_ic(factor, returns)` | `evaluation.metrics` | Pearson 线性相关 |
| `compute_forward_returns(prices, horizon)` | `evaluation.metrics` | 单标的前瞻收益 |
| `compute_forward_returns_panel(price_panel, horizon)` | `evaluation.metrics` | 面板前瞻收益 |
| `rank_normalize(series)` | `evaluation.metrics` | 百分位排名 [0, 1] |
| `cross_sectional_rank(panel)` | `evaluation.metrics` | 截面百分位排名 |
| `zscore_normalize(series)` | `evaluation.metrics` | Z-score 标准化 |
| `cross_sectional_zscore(panel)` | `evaluation.metrics` | 截面 Z-score 标准化 |
| `mutual_information(x, y, n_neighbors)` | `evaluation.metrics` | KSG 互信息估计 |
| `cumulative_returns(returns)` | `evaluation.metrics` | 累计收益曲线 |
| `annualize_return(returns, periods_per_year)` | `evaluation.metrics` | 年化收益率 |
| `annualize_volatility(returns, periods_per_year)` | `evaluation.metrics` | 年化波动率 |
| `sharpe_ratio(returns, periods_per_year, risk_free_rate)` | `evaluation.metrics` | 夏普比率（零波动率返回 ±inf） |
| `max_drawdown(cumulative)` | `evaluation.metrics` | 最大回撤（返回负数，如 -0.15 表示 15% 回撤） |

#### 因子存储

| 函数/方法 | 模块 | 说明 |
|-----------|------|------|
| `FactorStore(root_dir)` | `store.factor_store` | 初始化，默认路径 `db/factors/` |
| `.save(name, panel, meta)` | `store.factor_store` | 原子写入（Parquet + JSON） |
| `.load(name)` | `store.factor_store` | 加载因子面板 |
| `.exists(name)` | `store.factor_store` | 检查因子是否已入库 |
| `.delete(name)` | `store.factor_store` | 删除因子及元数据 |
| `.list_factors()` | `store.factor_store` | 列出所有已入库因子名称 |

#### 因子目录

| 函数/方法 | 模块 | 说明 |
|-----------|------|------|
| `FactorCatalog(store)` | `store.catalog` | 初始化并扫描元数据 |
| `.summary()` | `store.catalog` | 返回所有因子摘要 DataFrame |
| `.search(category, factor_type, family)` | `store.catalog` | 按分类/类型/族名筛选，返回 FactorMeta 列表 |
| `.get_meta(name)` | `store.catalog` | 获取单个因子元数据 |
| `.list_families()` | `store.catalog` | 列出所有已入库的因子族名称 |
| `.family_summary(family_name)` | `store.catalog` | 族内变体的参数对比摘要表 |
| `name in catalog` | `store.catalog` | 检查因子是否已入库 |
| `len(catalog)` | `store.catalog` | 已入库因子数量 |

### 因子开发完整工作流

从"因子想法"到"因子入库"的标准化 7 步流程：

```
步骤 1: Notebook 探索
    ┌─────────────────────────────────────────────────┐
    │  在 notebook 中用原始数据探索因子逻辑            │
    │  使用合成数据或 DataReader 读取真实数据           │
    │  快速验证：因子值分布、与收益的散点图            │
    └──────────────────────┬──────────────────────────┘
                           ▼
步骤 2: 代码化（继承基类 + @register_factor）
    ┌─────────────────────────────────────────────────┐
    │  选择基类: TimeSeriesFactor / CrossSectional     │
    │            / CrossAssetFactor                    │
    │  实现 meta() 和 compute_single()/compute()       │
    │  用 @register_factor 装饰器注册                  │
    └──────────────────────┬──────────────────────────┘
                           ▼
步骤 3: 单元测试
    ┌─────────────────────────────────────────────────┐
    │  测试 meta() 返回值正确                          │
    │  测试 compute_single() 的已知答案                │
    │  测试边界情况：空数据、数据不足、NaN 输入        │
    └──────────────────────┬──────────────────────────┘
                           ▼
步骤 4: FactorEngine 计算
    ┌─────────────────────────────────────────────────┐
    │  engine = FactorEngine()                         │
    │  panel = engine.compute_factor("my_factor",      │
    │      symbols=["BTC/USDT", "ETH/USDT"])           │
    │  # 自动从 DataReader 读取数据 → 计算 → 返回面板 │
    └──────────────────────┬──────────────────────────┘
                           ▼
步骤 5: FactorAnalyzer 评价
    ┌─────────────────────────────────────────────────┐
    │  analyzer = FactorAnalyzer(panel, prices)         │
    │  report = analyzer.full_report()                  │
    │  print(analyzer.summary_text("my_factor"))        │
    │  重点关注: IC_IR > 0.5? 分层单调? 换手可控?      │
    └──────────────────────┬──────────────────────────┘
                           ▼
步骤 6: 冗余检测（多因子场景）
    ┌─────────────────────────────────────────────────┐
    │  result = incremental_ic(new_panel,               │
    │      existing_factors, prices)                    │
    │  若 info_retention < 0.3 → 与已有因子高度冗余    │
    └──────────────────────┬──────────────────────────┘
                           ▼
步骤 7: FactorStore 入库
    ┌─────────────────────────────────────────────────┐
    │  store = FactorStore()                            │
    │  store.save("my_factor", panel, meta)             │
    │  # 或在 compute_factor() 时 save=True 自动入库   │
    │  # 入库后 Phase 2b（模型/策略）可通过             │
    │  #   FactorStore.load() 直接读取                  │
    └─────────────────────────────────────────────────┘
```

### 模块间依赖关系图

```
┌─────────────────────────────────────────────────────────────┐
│                    factor_research 内部依赖                   │
│                                                             │
│  factors/                                                   │
│  ├── momentum/returns.py ──────┐                            │
│  └── microstructure/imbalance.py──┤                          │
│                                   ▼                         │
│  core/                         core/registry.py             │
│  ├── types.py ◄───────────── core/base.py                   │
│  │     ▲                        ▲                           │
│  │     │                        │                           │
│  │  core/engine.py ─────────────┘                           │
│  │     │                                                    │
│  │     │  ┌─── evaluation/ ──────────────────────────┐      │
│  │     │  │  metrics.py  ◄── ic.py                   │      │
│  │     │  │      ▲       ◄── quantile.py             │      │
│  │     │  │      │       ◄── tail.py                 │      │
│  │     │  │      │       ◄── stability.py            │      │
│  │     │  │      │       ◄── nonlinear.py            │      │
│  │     │  │      │       ◄── turnover.py             │      │
│  │     │  │      └────── ◄── correlation.py          │      │
│  │     │  │                                          │      │
│  │     │  │  analyzer.py ──► 调用以上所有模块         │      │
│  │     │  │  report.py   ──► 格式化输出              │      │
│  │     │  └──────────────────────────────────────────┘      │
│  │     │                                                    │
│  │     ▼                                                    │
│  │  store/                                                  │
│  │  ├── factor_store.py                                     │
│  │  └── catalog.py ──► factor_store.py                      │
│  │                                                          │
│  │  alignment/                                              │
│  │  ├── grid.py                                             │
│  │  ├── refresh_time.py                                     │
│  │  └── hayashi_yoshida.py                                  │
│  │                                                          │
│  │  config.py ◄── 被以上大多数模块引用                       │
│  └──────────────────────────────────────────────────────────┘
│                                                             │
├─────────────── 接口边界 ────────────────────────────────────┤
│                                                             │
│  ┌─ data_infra（上游）─────────────────────────────────┐    │
│  │  config/settings.py ──► factor_research/config.py    │    │
│  │  data/reader.py     ──► core/engine.py               │    │
│  │  utils/logger.py    ──► 各模块日志                   │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─ Phase 2b（下游，尚未实现）─────────────────────────┐    │
│  │  仅通过 FactorStore.load() 读取因子数据              │    │
│  │  不直接依赖 factor_research 的任何其他模块            │    │
│  └──────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 错误处理与边界情况

#### 空数据行为约定

| 场景 | 行为 | 原因 |
|------|------|------|
| 因子面板为空 DataFrame | 各评价函数返回包含 NaN 的默认结构 | 不崩溃，允许管道继续运行 |
| 价格面板为空 | 同上，返回空/NaN 结果 | 同上 |
| 单标的数据为空 | `compute_single()` 返回空 Series，引擎跳过该标的 | 部分标的缺数据不应阻断整体计算 |
| FactorStore 目录为空 | `list_factors()` 返回空列表，`FactorCatalog.summary()` 返回空 DataFrame | 安全初始状态 |

#### 最小样本数约束

| 常量 | 值 | 影响的函数 | 行为 |
|------|-----|-----------|------|
| `MIN_IC_OBSERVATIONS` | 3 | `spearman_ic()`, `pearson_ic()` | 截面有效标的数 < 3 时返回 NaN |
| `MIN_REGRESSION_OBSERVATIONS` | 10 | `correlation_analysis()`, `incremental_ic()` | 展平后样本数 < 10 时返回空结果 |

> **为什么是这些值？**
> - `MIN_IC_OBSERVATIONS = 3`：Spearman 相关至少需要 3 个点才有意义。当前 5 标的场景下，去掉 NaN 后可能只剩 3~4 个有效值，阈值不宜过高。
> - `MIN_REGRESSION_OBSERVATIONS = 10`：回归分析的自由度要求。VIF 计算需要 OLS 回归，10 个样本是最低可接受标准。

#### NaN / Inf 处理策略

| 阶段 | 处理方式 |
|------|---------|
| 因子计算 (`compute_single`) | 因子实现者负责 `dropna()`，返回的 Series 中不应包含 NaN |
| 面板对齐 (`grid_align`) | 使用 `ffill` 填充，超过 `max_gap` 的间隔保持 NaN |
| IC 计算 (`spearman_ic`) | 自动跳过因子值或收益任一为 NaN 的观测 |
| 分层回测 (`quantile_backtest`) | NaN 标的不参与分组 |
| 底层原语 (`mutual_information`) | 显式过滤 NaN 和 Inf 后计算 |
| 存储 (`FactorStore.save`) | 保留 NaN（不做隐式填充），使用者需自行清洗 |
| Z-score 标准化 | 标准差为 0 时返回全零 Series（避免除零） |

### 配置说明

`factor_research/config.py` 集中管理因子模块的所有可调参数，是 factor_research 引用 data_infra 配置的**唯一入口**。

```python
# ── 因子存储路径 ──
FACTOR_STORE_DIR = settings.DB_DIR / "factors"   # 因子持久化目录

# ── 引擎默认参数 ──
DEFAULT_SYMBOLS = settings.SYMBOLS               # 默认计算的标的列表

# ── 因子评价默认参数 ──
DEFAULT_HORIZONS = [1, 5, 10, 30, 60]            # IC 分析的前瞻 horizon（bar 数）
DEFAULT_N_GROUPS = 5                             # 分层回测的分组数
DEFAULT_TAIL_THRESHOLD = 0.9                     # 尾部分析的百分位阈值
DEFAULT_ROLLING_WINDOW = 60                      # 滚动 IC 的窗口长度

# ── 最小样本数约束 ──
MIN_IC_OBSERVATIONS = 3                          # IC 计算最少截面观测数
MIN_REGRESSION_OBSERVATIONS = 10                 # 回归分析最少样本数
```

#### 自定义方式

**方式 1: 直接修改 `config.py`（全局生效）**

```python
# factor_research/config.py
DEFAULT_HORIZONS = [1, 5, 10, 30, 60, 120]  # 增加 2 小时 horizon
DEFAULT_N_GROUPS = 10                        # 10 分组（需要足够多标的）
```

**方式 2: 调用时传参覆盖（局部生效，推荐）**

```python
# 不修改配置文件，在调用时显式指定参数
ic_result = ic_analysis(factor_panel, price_panel, horizons=[1, 5, 10])
qt_result = quantile_backtest(factor_panel, price_panel, n_groups=3, horizon=5)
report = analyzer.full_report(horizons=[1, 10, 60])
```

> **建议**: 优先使用方式 2（调用时传参），保持 `config.py` 中的默认值作为"合理基线"。只有当确认需要全局修改时才改 `config.py`，修改后务必运行全量回归测试。

---

## 第二阶段(b)：Alpha 模型框架 (alpha_model)

### 概述

Alpha 模型框架将因子面板转化为可交易的目标持仓权重。核心理念：**框架做管道，模型做黑盒**。

- **AlphaModel Protocol**：框架对模型的唯一要求是实现 `fit/predict`，sklearn/LightGBM/PyTorch/手写脚本均可直接接入
- **Walk-Forward 防过拟合**：严格时序切分 + Embargo 隔离期，防止标签泄漏和因子 lookback 泄漏
- **cvxpy 凸优化组合构建**：所有约束（仓位上限、dollar/beta-neutral、杠杆上限）联合求解
- **标准化是工具箱，不是管道步骤**：树模型不需要标准化，线性模型自行选择 expanding/rolling/截面
- **模块间只通过 FactorStore / SignalStore 通信**：与上下游完全解耦

### AlphaModel 协议

框架对模型的唯一要求。使用 `Protocol`（而非 ABC），因为 sklearn 原生模型已有 `fit/predict`，无需继承即可使用。

```python
from alpha_model.core.types import AlphaModel

# 必须实现（框架调用）
model.fit(X, y, **kwargs)    # 训练
model.predict(X)              # 预测

# 可选实现（用于持久化和分析）
model.save_model(path)        # 保存到指定目录
model.load_model(path)        # 加载
model.get_feature_importance()  # 因子重要性
model.get_params()            # 模型参数
```

**直接使用 sklearn 原生模型（零封装）：**

```python
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)  # 天然满足 AlphaModel 协议
```

### 核心类型定义 (core/types)

| 类型 | 说明 |
|------|------|
| `AlphaModel` | Protocol — 模型协议（fit/predict + 可选 save_model/load_model/get_feature_importance/get_params） |
| `TrainMode` | 枚举 — `POOLED`（所有标的堆叠训练一个模型）/ `PER_SYMBOL`（每标的独立训练） |
| `WalkForwardMode` | 枚举 — `EXPANDING`（训练窗口逐步扩大）/ `ROLLING`（固定大小滑动） |
| `TrainConfig` | 数据类 — 训练配置（train_mode, wf_mode, target_horizon, train_periods, test_periods, purge_periods） |
| `PortfolioConstraints` | 数据类 — 组合约束（max_weight, dollar_neutral, beta_neutral, vol_target, leverage_cap, risk_aversion, turnover_penalty） |
| `ModelMeta` | 数据类 — 策略元数据，支持 `to_dict()` / `from_dict()` 序列化 |

### 预处理 (preprocessing)

#### 因子对齐 (alignment)

将不同频率的因子面板对齐到统一时间网格。

```python
from alpha_model.preprocessing.alignment import align_factor_panels

# 自动推断目标频率（取所有因子中最低频率）
aligned = align_factor_panels(factor_panels)

# 手动指定目标频率
aligned = align_factor_panels(factor_panels, target_freq="5min", max_gap=10)
```

#### 标准化工具箱 (transform)

标准化作为独立工具函数提供，**不在管道中自动执行**。用户/模型可根据需求自行调用。

| 函数 | 说明 | 适用场景 |
|------|------|---------|
| `expanding_zscore(panel, min_periods)` | Expanding window z-score | 时序因子，度量"历史上有多罕见" |
| `rolling_zscore(panel, window)` | Rolling window z-score | 时序因子，regime shift 频繁时 |
| `cross_sectional_zscore(panel)` | 截面 z-score（每行独立） | 截面因子 |
| `cross_sectional_rank(panel)` | 截面百分位排名 [0,1] | 通用，最鲁棒 |
| `winsorize(panel, sigma, method)` | 去极值（截断到 ±σ 标准差） | 原始因子数据清洗 |
| `build_feature_matrix(factor_panels, symbols, train_mode)` | 多因子面板 → 特征矩阵 | Pooled: MultiIndex(timestamp, symbol); Per-Symbol: dict |
| `build_pooled_target(X, fwd_returns, symbols)` | 构建 Pooled 模式目标变量 | 与 build_feature_matrix 配合 |

```python
from alpha_model.preprocessing.transform import (
    expanding_zscore, rolling_zscore, winsorize,
    build_feature_matrix,
)

# 标准化示例
panel_z = expanding_zscore(raw_panel, min_periods=252)
panel_w = winsorize(raw_panel, sigma=3.0, method="cross_sectional")

# 构建特征矩阵
X = build_feature_matrix(factor_panels, symbols, TrainMode.POOLED)
# → DataFrame with MultiIndex(timestamp, symbol), columns=factor_names
```

#### 因子筛选 (selection)

从已入库因子中筛选适合建模的子集。**可选步骤**（树模型不需要筛选）。

```python
from alpha_model.preprocessing.selection import select_factors, select_from_families

# threshold 模式: IC过滤 → VIF去共线性 → 贪心增量IC
selected = select_factors(
    factor_panels, price_panel, horizon=10,
    mode="threshold", min_ic=0.02, max_vif=10.0,
)

# top_k 模式: 多维评分（IC、MI、单调性）加权排序
selected = select_factors(
    factor_panels, price_panel,
    mode="top_k", top_k=5,
    score_weights={"ic": 0.5, "mi": 0.3, "monotonicity": 0.2},
)

# 族级筛选: 每族选最优变体 → 跨族再筛选
selected = select_from_families(
    ["multi_scale_returns"], price_panel,
    family_select_metric="ic_mean",
)
```

### 训练框架 (training)

#### 时序切分器 (splitter)

```python
from alpha_model.training.splitter import TimeSeriesSplitter
from alpha_model.core.types import WalkForwardMode

splitter = TimeSeriesSplitter(
    train_periods=3000,
    test_periods=1000,
    target_horizon=10,
    max_factor_lookback=60,
    mode=WalkForwardMode.EXPANDING,
)
folds = splitter.split(n_samples=10000)

for fold in folds:
    X_train = X[fold.train_start:fold.train_end]
    X_test = X[fold.test_start:fold.test_end]
```

**Embargo 机制**：`embargo_periods = max(target_horizon, max_factor_lookback)`，训练集和测试集之间的隔离期，同时防止 forward return 标签泄漏和因子 lookback 窗口泄漏。

#### Walk-Forward 引擎 (walk_forward)

```python
from alpha_model.training.walk_forward import WalkForwardEngine

engine = WalkForwardEngine(model, splitter, train_mode=TrainMode.POOLED)
result = engine.run(X, y, symbols)

# result.predictions:             样本外预测面板 (timestamp × symbol)
# result.fold_metrics:            每个 fold 的 IC 等指标
# result.feature_importance:      因子重要性（模型原生，各 fold 平均）
# result.permutation_importance:  置换重要性（模型无关，按需开启）

# 开启 Permutation Importance（模型无关的因子重要性）
engine = WalkForwardEngine(
    model, splitter, train_mode=TrainMode.POOLED,
    compute_permutation_importance=True,   # 开启
    n_permutations=5,                      # 每个特征打乱 5 次
    permutation_random_state=42,           # 可复现
)
result = engine.run(X, y, symbols)
# result.permutation_importance → DataFrame(feature, mean_importance)
# 原理: 打乱单列 → IC 下降幅度 = 该特征的贡献度
# 复杂度: O(n_features × n_permutations × n_folds) 次 predict
```

#### 训练调度器 (trainer)

一站式接口：从因子面板和价格面板出发，自动完成特征构建、目标变量计算、Walk-Forward 训练。

```python
from alpha_model.training.trainer import Trainer
from alpha_model.core.types import TrainConfig

trainer = Trainer(model=Ridge(), train_config=TrainConfig(target_horizon=10))
result = trainer.run(factor_panels, price_panel, symbols)
```

### 信号生成 (signal)

将模型预测值（alpha score）转化为标准化信号。**默认不截断极端值**——极端预测可能正是强信号，真正的风控由 portfolio 层约束实现。

```python
from alpha_model.signal.generator import generate_signal
from alpha_model.signal.smoother import ema_smooth

# 截面 z-score 标准化
signal = generate_signal(predictions, method="cross_sectional_zscore")

# 可选截断
signal = generate_signal(predictions, method="cross_sectional_zscore", clip_sigma=3.0)

# 截面百分位排名
signal = generate_signal(predictions, method="cross_sectional_rank")

# EMA 平滑（降低换手率）
signal_smooth = ema_smooth(signal, halflife=5)
```

### 组合构建 (portfolio)

#### Mean-Variance 凸优化 (constructor)

核心组件。使用 cvxpy 联合求解，所有约束同时满足。

```
QP 问题:
    minimize    w' Σ w - λ α' w + γ ||w - w_prev||₁
    subject to  |w_i| ≤ max_weight
                Σ w_i = 0                       (dollar-neutral, 可选)
                β' w = 0                        (beta-neutral, 可选)
                ||w||₁ ≤ leverage_cap
```

```python
from alpha_model.core.types import PortfolioConstraints
from alpha_model.portfolio.constructor import PortfolioConstructor

constraints = PortfolioConstraints(
    max_weight=0.4,
    dollar_neutral=True,
    beta_neutral=False,
    leverage_cap=2.0,
    risk_aversion=1.0,
    turnover_penalty=0.01,
    vol_target=0.15,        # 年化波动率目标 15%
)
constructor = PortfolioConstructor(constraints)
weights = constructor.construct(signal, price_panel)
```

#### 组合子模块

| 模块 | 功能 |
|------|------|
| `portfolio.covariance` | 协方差矩阵估计: Ledoit-Wolf shrinkage / 样本 / 指数加权。`estimate_covariance()` 和 `rolling_covariance()` |
| `portfolio.beta` | 滚动 Beta 估计: OLS 回归 `r_i = α + β × r_BTC + ε`。`rolling_beta()` |
| `portfolio.constraints` | cvxpy 约束生成器: 仓位上限、dollar-neutral、beta-neutral、杠杆上限。`build_constraints()` |
| `portfolio.risk_budget` | 波动率目标: 动态缩放仓位使年化 vol ≈ target，受 leverage_cap 约束。`apply_vol_target()` |

### 向量化回测 (backtest)

快速策略验证。**不替代** Phase 3 的事件驱动回测。

```python
from alpha_model.backtest.vectorized import vectorized_backtest

result = vectorized_backtest(
    weights, price_panel,
    fee_rate=0.0004,         # Binance taker 0.04%
    impact_coeff=0.1,        # 市场冲击系数
    adv_panel=adv_panel,     # 日均成交量（None 则退化为纯手续费）
)

print(result.summary())
# {
#     'annual_return': 0.12, 'annual_volatility': 0.15,
#     'sharpe_ratio': 0.80, 'sortino_ratio': 1.05, 'calmar_ratio': 0.95,
#     'max_drawdown': -0.12, 'max_drawdown_duration': 4320,
#     'avg_turnover': 0.15, 'total_cost': 0.023,
#     'win_rate': 0.52, 'total_return': 0.12,
# }
```

**交易成本模型（两层）：**

| 层 | 公式 | 说明 |
|---|------|------|
| 固定手续费 | `turnover × fee_rate` | 通常 0.04% taker |
| 市场冲击 (Square-root) | `impact_coeff × σ_i × √(|ΔV_i| / ADV_i)` | Almgren-Chriss 简化版 |

**绩效指标（BacktestResult.summary()）：**

| 指标 | 说明 |
|------|------|
| `annual_return` | 年化收益率 |
| `annual_volatility` | 年化波动率 |
| `sharpe_ratio` | 夏普比率 |
| `sortino_ratio` | Sortino ratio（下行波动率） |
| `calmar_ratio` | Calmar ratio（年化收益 / 最大回撤） |
| `max_drawdown` | 最大回撤（负数） |
| `max_drawdown_duration` | 最长回撤持续期（bar 数） |
| `avg_turnover` | 平均换手率 |
| `total_cost` | 总交易成本 |
| `win_rate` | 胜率 |
| `total_return` | 总收益率 |

### 持久化 (store)

#### SignalStore — 策略输出存储

Phase 2b → Phase 3 / Phase 4 / execution_optimizer 的唯一持久化接口。原子写入（先 `.tmp` 再 `rename`）。

三层数据：
- **weights** (post-cvxpy)：目标权重，主要供向量化回测消费
- **signals** (post-generator, pre-cvxpy)：标准化信号，供 execution_optimizer 多资产优化
- **raw_predictions** (post-predict, pre-generator)：模型原始预测，供单资产择时/模型诊断

```
db/signals/{strategy_name}/
├── weights.parquet          # 目标权重面板（必须）
├── signals.parquet          # 标准化信号面板（可选）
├── raw_predictions.parquet  # 模型原始预测面板（可选）
├── meta.json                # ModelMeta 序列化
└── performance.json         # 绩效摘要
```

```python
from alpha_model.store.signal_store import SignalStore

store = SignalStore()

# 保存（raw_predictions 由 AlphaPipeline.save() 自动传入）
store.save("strategy_v1", weights, signals=signal,
           raw_predictions=raw_preds, meta=meta, performance=result.summary())

# 加载
store.load_weights("strategy_v1")
store.load_signals("strategy_v1")
store.load_raw_predictions("strategy_v1")  # 模型原始预测
store.load_meta("strategy_v1")
store.load_performance("strategy_v1")

# 存在性检查
store.has_signals("strategy_v1")            # signals.parquet 是否存在
store.has_raw_predictions("strategy_v1")    # raw_predictions.parquet 是否存在
store.exists("strategy_v1")                 # 策略目录是否存在

store.list_strategies()
store.delete("strategy_v1")
```

#### ModelStore — 模型持久化

```
db/models/{model_name}/
├── model/                 # 模型文件（由 AlphaModel.save_model 写入）
│   └── model.joblib       # 或 model.pt、model.txt 等
├── meta.json              # ModelMeta + TrainConfig 序列化
└── importance.json        # 因子重要性排名
```

```python
from alpha_model.store.model_store import ModelStore

store = ModelStore()
store.save("ridge_v1", model, meta, importance=importance_dict)

# 加载需要提供 model_factory（重建空模型实例）
model, meta = store.load("ridge_v1", model_factory=lambda: SklearnModelWrapper(Ridge()))

store.load_importance("ridge_v1")
store.list_models()
store.exists("ridge_v1")
store.delete("ridge_v1")
```

### 参考模型实现 (models)

示例代码，展示如何封装不同框架的模型。**非核心架构**——用户完全可以不用这些封装，只要实现 `fit/predict`。

| 封装类 | 底层框架 | 特点 |
|--------|---------|------|
| `SklearnModelWrapper` | sklearn (Ridge/Lasso/ElasticNet 等) | 补充 save_model/load_model/get_feature_importance |
| `LGBMModelWrapper` | LightGBM | 自动 lgb.train，支持所有 LightGBM 参数 |
| `XGBModelWrapper` | XGBoost | XGBRegressor 封装 |
| `TorchModelBase` | PyTorch | 用户继承并实现 `build_network()`，自动处理 val 拆分/DataLoader/early stopping/GPU |

```python
from alpha_model.models.linear_models import SklearnModelWrapper
from alpha_model.models.tree_models import LGBMModelWrapper
from alpha_model.models.torch_base import TorchModelBase
from sklearn.linear_model import Ridge

# sklearn
model = SklearnModelWrapper(Ridge(alpha=1.0))

# LightGBM
model = LGBMModelWrapper(
    objective="regression", num_leaves=31,
    learning_rate=0.05, n_estimators=100,
)

# PyTorch（用户继承）
class MyMLP(TorchModelBase):
    def build_network(self, n_features):
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

model = MyMLP(val_ratio=0.2, patience=10, max_epochs=100)
```

### Notebook 逐步工作流

从"因子入库"到"策略回测"的标准化流程：

```
步骤 1: 从 FactorStore 加载因子
    ┌─────────────────────────────────────────────────┐
    │  store = FactorStore()                           │
    │  panels = {name: store.load(name) for name in    │
    │            ["returns_5m", "returns_10m", ...]}    │
    └──────────────────────┬──────────────────────────┘
                           ▼
步骤 2: (可选) 因子筛选
    ┌─────────────────────────────────────────────────┐
    │  selected = select_factors(panels, prices,       │
    │      mode="threshold", min_ic=0.02)              │
    │  # 或跳过此步（树模型不需要筛选）               │
    └──────────────────────┬──────────────────────────┘
                           ▼
步骤 3: 对齐 + 特征矩阵
    ┌─────────────────────────────────────────────────┐
    │  aligned = align_factor_panels(selected)         │
    │  X = build_feature_matrix(aligned, symbols,      │
    │      TrainMode.POOLED)                           │
    └──────────────────────┬──────────────────────────┘
                           ▼
步骤 4: Walk-Forward 训练
    ┌─────────────────────────────────────────────────┐
    │  model = SklearnModelWrapper(Ridge(alpha=1.0))   │
    │  trainer = Trainer(model, TrainConfig(...))       │
    │  result = trainer.run(aligned, prices, symbols)   │
    │  # 或直接用 splitter + WalkForwardEngine         │
    └──────────────────────┬──────────────────────────┘
                           ▼
步骤 5: 信号生成
    ┌─────────────────────────────────────────────────┐
    │  signal = generate_signal(result.predictions)    │
    │  signal = ema_smooth(signal, halflife=5)  # 可选 │
    └──────────────────────┬──────────────────────────┘
                           ▼
步骤 6: 组合构建
    ┌─────────────────────────────────────────────────┐
    │  constraints = PortfolioConstraints(...)          │
    │  constructor = PortfolioConstructor(constraints)  │
    │  weights = constructor.construct(signal, prices)  │
    └──────────────────────┬──────────────────────────┘
                           ▼
步骤 7: 向量化回测
    ┌─────────────────────────────────────────────────┐
    │  bt = vectorized_backtest(weights, prices)       │
    │  print(bt.summary())                             │
    └──────────────────────┬──────────────────────────┘
                           ▼
步骤 8: 持久化
    ┌─────────────────────────────────────────────────┐
    │  signal_store = SignalStore()                     │
    │  signal_store.save("my_strategy", weights, ...)  │
    │  model_store = ModelStore()                       │
    │  model_store.save("my_strategy", model, meta)    │
    └─────────────────────────────────────────────────┘
```

### AlphaPipeline 一键管道

将上述 8 步串联为一键调用：

```python
from alpha_model.core.pipeline import AlphaPipeline
from alpha_model.core.types import TrainConfig, PortfolioConstraints
from alpha_model.models.linear_models import SklearnModelWrapper
from sklearn.linear_model import Ridge

pipeline = AlphaPipeline(
    model=SklearnModelWrapper(Ridge(alpha=1.0)),
    train_config=TrainConfig(
        target_horizon=10,
        train_periods=5000,
        test_periods=1000,
        purge_periods=60,
    ),
    constraints=PortfolioConstraints(
        dollar_neutral=True,
        max_weight=0.4,
        vol_target=0.15,
    ),
    # 方式 A: 直接指定因子名
    factor_names=["returns_5m", "returns_10m", "orderbook_imbalance"],
    # 方式 B: 指定因子族（自动选最优变体）
    # factor_families=["multi_scale_returns"],
)

# 执行完整管道
result = pipeline.run(price_panel=prices)
print(result.summary())

# 保存策略
pipeline.save("ridge_momentum_v1")
```

### 模块间依赖关系图

```
┌─────────────────────────────────────────────────────────────┐
│                    alpha_model 内部依赖                       │
│                                                             │
│  core/                                                      │
│  ├── types.py ◄───────────── 被所有模块引用                  │
│  └── pipeline.py ──────────► 调用以下所有模块                 │
│                                                             │
│  preprocessing/                                             │
│  ├── alignment.py                                           │
│  ├── transform.py ─────────► core/types.py                  │
│  └── selection.py ─────────► factor_research.evaluation.*    │
│                                                             │
│  training/                                                  │
│  ├── splitter.py ──────────► core/types.py                  │
│  ├── walk_forward.py ──────► splitter.py, core/types.py     │
│  └── trainer.py ───────────► walk_forward.py, transform.py  │
│                                                             │
│  signal/                                                    │
│  ├── generator.py                                           │
│  └── smoother.py                                            │
│                                                             │
│  portfolio/                                                 │
│  ├── constructor.py ───────► covariance, beta, constraints, │
│  │                            risk_budget, cvxpy             │
│  ├── covariance.py ────────► sklearn.covariance.LedoitWolf  │
│  ├── beta.py                                                │
│  ├── constraints.py ───────► cvxpy                          │
│  └── risk_budget.py                                         │
│                                                             │
│  backtest/                                                  │
│  ├── vectorized.py ────────► performance.py                 │
│  └── performance.py ───────► factor_research.evaluation.*   │
│                                                             │
│  store/                                                     │
│  ├── signal_store.py                                        │
│  └── model_store.py                                         │
│                                                             │
│  models/ ──────────────────── 示例代码, 不被框架核心引用      │
│  config.py ◄─── 被以上大多数模块引用                         │
├─────────────── 接口边界 ────────────────────────────────────┤
│                                                             │
│  ┌─ factor_research（上游）─────────────────────────────┐   │
│  │  store.factor_store.FactorStore ──► pipeline.py       │   │
│  │  evaluation.metrics.*       ──► trainer, performance   │   │
│  │  evaluation.ic/correlation  ──► selection.py           │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─ data_infra（上游）──────────────────────────────────┐   │
│  │  data.reader.DataReader     ──► utils.load_price_panel │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─ Phase 3 / Phase 4 / execution_optimizer（下游）────┐   │
│  │  通过 SignalStore 读取 weights / signals /           │   │
│  │    raw_predictions（按场景选择数据层）                │   │
│  │  不直接依赖 alpha_model 的任何其他模块                 │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### API Quick Reference

#### 预处理

| 函数/方法 | 模块 | 说明 |
|-----------|------|------|
| `align_factor_panels(factor_panels, target_freq, fill_method, max_gap)` | `preprocessing.alignment` | 多频率因子对齐到统一网格 |
| `expanding_zscore(panel, min_periods)` | `preprocessing.transform` | Expanding window z-score |
| `rolling_zscore(panel, window)` | `preprocessing.transform` | Rolling window z-score |
| `cross_sectional_zscore(panel)` | `preprocessing.transform` | 截面 z-score |
| `cross_sectional_rank(panel)` | `preprocessing.transform` | 截面百分位排名 |
| `winsorize(panel, sigma, method, window, min_periods)` | `preprocessing.transform` | 去极值 |
| `build_feature_matrix(factor_panels, symbols, train_mode)` | `preprocessing.transform` | 因子面板 → 特征矩阵 |
| `build_pooled_target(X, fwd_returns, symbols)` | `preprocessing.transform` | 构建 Pooled 目标变量 |
| `select_factors(factor_panels, price_panel, horizon, mode, ...)` | `preprocessing.selection` | 因子筛选（threshold / top_k） |
| `select_from_families(family_names, price_panel, horizon, ...)` | `preprocessing.selection` | 族级筛选（每族选最优变体） |

#### 训练

| 函数/方法 | 模块 | 说明 |
|-----------|------|------|
| `TimeSeriesSplitter(train_periods, test_periods, target_horizon, max_factor_lookback, mode)` | `training.splitter` | 时序切分器 |
| `TimeSeriesSplitter.split(n_samples)` | `training.splitter` | 生成 Fold 列表 |
| `Fold.train_start/train_end/test_start/test_end` | `training.splitter` | 切分索引（含/不含） |
| `WalkForwardEngine(model, splitter, train_mode, compute_permutation_importance, n_permutations, permutation_random_state)` | `training.walk_forward` | Walk-Forward 引擎（后三个参数可选，控制置换重要性） |
| `WalkForwardEngine.run(X, y, symbols)` | `training.walk_forward` | 执行 Walk-Forward，返回 WalkForwardResult |
| `Trainer(model, train_config, max_factor_lookback)` | `training.trainer` | 训练调度器 |
| `Trainer.run(factor_panels, price_panel, symbols)` | `training.trainer` | 一站式训练（特征构建+目标变量+WF） |

#### 信号

| 函数/方法 | 模块 | 说明 |
|-----------|------|------|
| `generate_signal(predictions, method, clip_sigma)` | `signal.generator` | 预测值 → 标准化信号 |
| `ema_smooth(signal, halflife)` | `signal.smoother` | EMA 信号平滑 |

#### 组合

| 函数/方法 | 模块 | 说明 |
|-----------|------|------|
| `PortfolioConstructor(constraints)` | `portfolio.constructor` | 凸优化组合构建器 |
| `PortfolioConstructor.construct(signal, price_panel, prev_weights)` | `portfolio.constructor` | 信号 → 目标权重（逐期 QP 求解） |
| `estimate_covariance(returns_panel, lookback, method, min_periods)` | `portfolio.covariance` | 估计协方差矩阵 |
| `rolling_covariance(returns_panel, lookback, method, min_periods)` | `portfolio.covariance` | 滚动协方差矩阵序列 |
| `rolling_beta(returns_panel, market_symbol, lookback)` | `portfolio.beta` | 滚动 Beta 估计 |
| `build_constraints(w, config, beta)` | `portfolio.constraints` | 生成 cvxpy 约束列表 |
| `apply_vol_target(weights, price_panel, vol_target, lookback, leverage_cap)` | `portfolio.risk_budget` | 波动率目标动态缩放 |

#### 回测

| 函数/方法 | 模块 | 说明 |
|-----------|------|------|
| `vectorized_backtest(weights, price_panel, fee_rate, impact_coeff, adv_panel, portfolio_value)` | `backtest.vectorized` | 向量化回测 |
| `estimate_market_impact(delta_weights, adv_panel, volatility_panel, portfolio_value, impact_coeff)` | `backtest.vectorized` | Square-root 市场冲击估计 |
| `BacktestResult.summary()` | `backtest.performance` | 绩效摘要字典 |
| `sortino_ratio(returns, periods_per_year)` | `backtest.performance` | Sortino ratio |
| `calmar_ratio(returns, periods_per_year)` | `backtest.performance` | Calmar ratio |
| `max_drawdown_duration(returns)` | `backtest.performance` | 最长回撤持续期 |

#### 持久化

| 函数/方法 | 模块 | 说明 |
|-----------|------|------|
| `SignalStore(base_dir)` | `store.signal_store` | 策略输出存储，默认 `db/signals/` |
| `.save(strategy_name, weights, signals, raw_predictions, meta, performance)` | `store.signal_store` | 原子写入（signals、raw_predictions 均可选） |
| `.load_weights(strategy_name)` | `store.signal_store` | 加载权重面板 |
| `.load_signals(strategy_name)` | `store.signal_store` | 加载标准化信号面板 |
| `.load_raw_predictions(strategy_name)` | `store.signal_store` | 加载模型原始预测面板 |
| `.load_meta(strategy_name)` | `store.signal_store` | 加载策略元数据 |
| `.load_performance(strategy_name)` | `store.signal_store` | 加载绩效摘要 |
| `.has_signals(strategy_name)` | `store.signal_store` | 检查是否存储了信号文件 |
| `.has_raw_predictions(strategy_name)` | `store.signal_store` | 检查是否存储了原始预测文件 |
| `.list_strategies()` | `store.signal_store` | 列出所有策略 |
| `ModelStore(base_dir)` | `store.model_store` | 模型存储，默认 `db/models/` |
| `.save(model_name, model, meta, importance)` | `store.model_store` | 保存模型 + 元数据 |
| `.load(model_name, model_factory)` | `store.model_store` | 加载模型（需提供工厂函数） |
| `.load_importance(model_name)` | `store.model_store` | 加载因子重要性 |
| `.list_models()` | `store.model_store` | 列出所有模型 |

#### 管道

| 函数/方法 | 模块 | 说明 |
|-----------|------|------|
| `AlphaPipeline(model, train_config, constraints, factor_names, factor_families, ...)` | `core.pipeline` | 一键管道初始化 |
| `AlphaPipeline.run(price_panel, start, end, symbols)` | `core.pipeline` | 执行完整管道，返回 BacktestResult |
| `AlphaPipeline.save(strategy_name)` | `core.pipeline` | 保存到 SignalStore + ModelStore |

### 配置说明

`alpha_model/config.py` 集中管理存储路径和默认参数。

```python
# ── 存储路径 ──
SIGNAL_STORE_DIR = DB_DIR / "signals"    # 策略输出目录
MODEL_STORE_DIR = DB_DIR / "models"      # 模型存储目录

# ── 默认交易对 ──
DEFAULT_SYMBOLS = ["BTC/USDT", "DOGE/USDT", "SOL/USDT", "BNB/USDT", "ETH/USDT"]

# ── 回测默认参数 ──
DEFAULT_FEE_RATE = 0.0004               # Binance taker 手续费率（单边）
DEFAULT_IMPACT_COEFF = 0.1              # 市场冲击系数（square-root model）
DEFAULT_PORTFOLIO_VALUE = 10000.0       # 组合总资金（USDT）

# ── 年化常数 ──
MINUTES_PER_YEAR = 525_960              # 1m K线: 365.25 × 24 × 60
```

---

## 第二阶段(c)：执行优化器 (execution_optimizer)

### 概述

Phase 2b `PortfolioConstructor` 使用固定换手惩罚 γ‖Δw‖₁，适用于向量化回测。
Phase 2c `ExecutionOptimizer` 是其事件驱动替代方案，使用动态成本模型，
为事件驱动回测和实盘交易提供单步组合优化。两者互斥使用。

### 核心设计

**目标函数**（全部收益率空间，无量纲）：

```
minimize:  w'Σw - λ × α'w + cost(Δw, MarketContext) + funding_rate'w
```

**动态成本模型（3 分量）**：

| 分量 | 公式 | 说明 |
|------|------|------|
| 佣金 | `fee_rate × Σ\|Δw_i\|` | 线性，Binance taker 0.04% |
| 买卖价差 | `Σ(spread_i/2 × \|Δw_i\|)` | 线性，实时注入 |
| 市场冲击 | `Σ(eff_coeff_i × \|Δw_i\|^1.5)` | 超线性，SOCP 合规 |

**约束**：复用 Phase 2b 约束构建器 + ADV 参与率约束。

### 使用方式

```python
from execution_optimizer import ExecutionOptimizer, MarketContext
from alpha_model.core.types import PortfolioConstraints

constraints = PortfolioConstraints(max_weight=0.4, dollar_neutral=True)
optimizer = ExecutionOptimizer(constraints)

# 事件循环中每步调用
target_w = optimizer.optimize_step(signals_t, current_w, context, prices[:t])
```

### Phase 2c 测试覆盖

| 测试文件 | 测试项数 | 覆盖内容 |
|---------|---------|---------|
| test_cost.py | 14 | 成本分量数值正确性（手工比对）、成本单调性、零交易零成本、逐标的冲击系数 |
| test_optimizer.py | 22 | 基本可解性（无NaN/约束满足）、高成本压制交易、ADV参与率约束、资金费率影响、Vol Targeting（缩放+leverage二次检查）、Fallback路径（协方差失败/求解异常/状态非最优）、Beta-neutral路径 |

---

## 第三阶段：事件驱动回测引擎 (backtest_engine)

### 概述

Phase 3 是项目的**研究分支收尾模块**。它把 Phase 1-2(a/b/c) 串成完整的回测流程，
对外暴露唯一入口 `EventDrivenBacktester().run(config)`，按 `RunMode` 分流到三种精度
模式，统一交付 `BacktestReport`。

研究分支至此完成；实盘分支（Phase 4）独立开发。

### 三种 RunMode 一表看清

| RunMode | 决策层 | 执行层 | 速度 / 半年 1m bar | 适用场景 |
|---------|------|------|---------------|--------|
| `VECTORIZED` | Phase 2b 已有的 `vectorized_backtest`（薄包装）| 仅手续费近似 | 秒级 | 因子初筛、参数扫描 |
| `EVENT_DRIVEN_FIXED_GAMMA` | `PrecomputedWeights` 重放 SignalStore 权重面板 | Rebalancer 逐 bar 模拟 + spread/impact | 几分钟 | 评估已训练策略的执行摩擦敏感性 |
| `EVENT_DRIVEN_DYNAMIC_COST` | `OnlineOptimizer` 每 bar 调 Phase 2c 重优化 | 同上 | ~3.5 小时（N=1） / ~1.5 小时（N=5）| 模拟实盘决策路径 |

### 核心 API（极简）

```python
from backtest_engine import EventDrivenBacktester, BacktestConfig, RunMode, CostMode
from alpha_model.core.types import PortfolioConstraints
import pandas as pd

config = BacktestConfig(
    strategy_name="momentum_v1",            # 必须存在于 SignalStore
    symbols=["BTC/USDT", "ETH/USDT"],
    start=pd.Timestamp("2026-01-01", tz="UTC"),
    end=pd.Timestamp("2026-04-01", tz="UTC"),
    run_mode=RunMode.EVENT_DRIVEN_DYNAMIC_COST,
    constraints=PortfolioConstraints(max_weight=0.4, dollar_neutral=True),
    cost_mode=CostMode.FULL_COST,           # 或 MATCH_VECTORIZED 做可比性校验
    optimize_every_n_bars=5,                # 加速：每 5 bar 重优化一次
)

report = EventDrivenBacktester().run(config)
print(report)                               # 自定义 __repr__，~20 行摘要
print(report.summary())                     # 20 keys（12 base + 5 cost_bp + 状态 + 偏差）
report.plot()                               # 一键 8 张图
report.save("./reports/run_2026_04")        # parquet + JSON 原子化
report.to_markdown("./reports/run_2026_04_md")  # markdown + figures/
```

### 子模块职责（10 个源文件）

| 文件 | 职责 |
|------|------|
| `config.py` | `BacktestConfig` (kw_only) + 三个枚举；`__post_init__` 16 项校验前置暴露环境问题 |
| `context.py` | `MarketContextBuilder`：DataReader → MarketContext（逐步）+ build_panels（批量）|
| `weights_source.py` | `WeightsSource` Protocol + `PrecomputedWeights`/`OnlineOptimizer` 两实现；cost_mode 自持 |
| `rebalancer.py` | 执行模拟（v1 仅 MARKET）+ 三分量成本计算；cost_mode 自持，MATCH_VECTORIZED 跳 spread |
| `pnl.py` | P&L 状态中心；混合模式（循环内时序敏感 + 循环后向量化）；破产双通道 |
| `engine.py` | 顶层协调；`_validate_environment` 8 项校验 + (a') (f) 早退 |
| `attribution.py` | 纯函数：cost 分解 / 偏差归因 / regime 分段 / per_symbol 按需重算 |
| `report.py` | `BacktestReport` 容器 + summary/to_dict/save/load + plot/to_markdown 入口 |
| `plot.py` | 8 张标准图（equity/drawdown/returns dist/cost/weights heatmap/rolling sharpe/regime/deviation）|
| `reporting.py` | `to_markdown` Markdown 单文件报告（含图片嵌入）|

### CostMode 行为映射

| RunMode × CostMode | 行为 |
|---------------------|------|
| VECTORIZED + FULL_COST | `vectorized_backtest(spread_panel=真实)` |
| VECTORIZED + MATCH_VECTORIZED | `vectorized_backtest(spread_panel=None)` |
| FIXED_GAMMA + FULL_COST | Rebalancer 内部判断计 spread/2 slippage |
| FIXED_GAMMA + MATCH_VECTORIZED | Rebalancer 内部判断不计 spread |
| DYNAMIC_COST + FULL_COST | OnlineOptimizer 透传 raw context；Rebalancer 计 slippage |
| DYNAMIC_COST + MATCH_VECTORIZED | OnlineOptimizer 内部 `replace(context, spread=0)`；Rebalancer 不计 |

**MATCH_VECTORIZED 的唯一用途是可比性校验**——使三种 RunMode 的成本模型在底层对齐，便于偏差拆解。
生产回测应用 FULL_COST。

### 跨模块护栏：`test_consistency.py`（关键）

无策略可上线 → 无集成测试 → 单测必须严格抓住"未来重构破坏 invariant"的回归。
专门的 `test_consistency.py` 模块集中三大数值精度护栏（rtol=1e-12）：

| 护栏 | 验证 |
|------|------|
| **TestZeroFrictionEquivalence** | 关闭 fee/spread/impact/funding 后，FIXED_GAMMA 与 VECTORIZED 的 returns/equity 严格相等 |
| **TestPerSymbolCostSumsToPortfolioViaEngine** | engine 路径下 `compute_per_symbol_cost` 三分量加总 == `cost_decomposition` portfolio total |
| **TestImpactFormulaFourPathsConsistency** | impact 公式 `(2/3)·coeff·σ·√(V/ADV)·|Δw|^1.5` 在四处实现两两相等：cvxpy / vectorized panel / Rebalancer scalar / per-symbol 重算 |

任何一处 impact 公式漂移、任何一处 cost_mode 分支语义改变，护栏立即失败。

### Fail-fast 设计

数值异常**必须暴露**，不允许 NaN 静默传播：

| 触发条件 | 暴露方式 |
|---------|--------|
| `V ≤ 0`（finite，策略亏光）| 标志 `is_bankrupt=True` + `bankruptcy_timestamp`，engine break，equity_curve 截断 |
| `V` 是 NaN / Inf（数据/公式 bug）| `_check_bankruptcy` 立即抛 `NumericalError`，事件循环中止 |
| funding_rates_panel.index tz ≠ UTC | `_validate_environment` 抛 `ValueError`（防 funding 静默漏扣） |
| price_panel.index ⊉ bar_timestamps | 同上（防 weights_history 算 NaN gross） |
| price_panel 推断频率 ≠ config.bar_freq | 同上（防 impact 公式 bars_per_day 算错） |

### v1 已知边界

- **不建模强平**：合约保证金率破线时实盘会强平，v1 仅看 V≤0 → 已知乐观偏差，在偏差拆解表显式列出
- **仅 MARKET 执行模式**：LIMIT/TWAP 推 v2（需 tick 仿真 + 跨 bar pending state，复杂度跳一级）
- **universe 固定**：v1 不支持中途加/减 symbol（已在 §11.6.5 reindex 兜底为 v2 留位）
- **断点重连**：v1 不做（半年回测 ~3.5h，加速参数已大幅缓解；主流框架 zipline/backtrader/vnpy 均不内置）

### Phase 3 测试覆盖（219 项）

| 测试文件 | 测试项数 | 覆盖内容 |
|---------|---------|---------|
| test_config.py | 28 | 16 项 __post_init__ 校验（含 v3 修订 tz-aware、regime_series tz、optimize_every_n_bars） |
| test_context.py | 26 | MarketContext 字段完整性、批量/逐步双模式、warmup 校验、build_panels keys |
| test_weights_source.py | 18 | Precomputed schema 严格化、Online cost_mode 自持、fallback 透传 |
| test_rebalancer.py | 24 | 三分量成本数值、最小下单量过滤、cost_mode 分支、ExecutionReport canonical schema |
| test_pnl.py | 31 | 破产双通道（V≤0 标志 / NaN/Inf 异常）、apply_funding/record 时序、向量化重算与循环内一致性 |
| test_engine.py | 23 | 8 项 _validate_environment、(a')/(f) 双早退、optimize_every_n_bars、4 项稳健性链路（A4/G1/G4/H4） |
| test_attribution.py | 24 | cost_decomposition keys/bp/share/funding 一阶近似、regime ffill/min_bars、deviation P1/P2 双模式、per_symbol 加总恒等 |
| test_report.py | 23 | summary 20 keys、to_dict NaN 处理、save_load 原子化（tmp+rename）、context_panels 跨机器分发 |
| test_plot.py | 13 | 8 张图返回类型、ax 注入、headless 兼容 |
| test_reporting.py | 7 | markdown 渲染、figures 目录、可选 section 跳过、Bankruptcy warning |
| **test_consistency.py** | **9** | **★ 跨模块/跨模式 数值一致性护栏（rtol=1e-12）：4 路径 impact、零摩擦严格等价、per-symbol 加总** |

---

## 运行测试

```bash
# 第一阶段测试（125 项）
python -m pytest data_infra/tests/ -v

# 第二阶段(a) 测试（200 项）
python -m pytest factor_research/tests/ -v

# 第二阶段(b) 测试（191 项）
python -m pytest alpha_model/tests/ -v

# 第二阶段(c) 测试（36 项）
python -m pytest execution_optimizer/tests/ -v

# 第三阶段测试（219 项，含 9 项跨模块护栏）
python -m pytest backtest_engine/tests/ -v

# 全部测试（研究分支总计 771 项）
python -m pytest data_infra/tests/ factor_research/tests/ alpha_model/tests/ \
                 execution_optimizer/tests/ backtest_engine/tests/ -v
```

### Phase 2b 测试覆盖

| 测试文件 | 测试项数 | 覆盖内容 |
|---------|---------|---------|
| test_types.py | 23 | AlphaModel Protocol、TrainConfig/PortfolioConstraints 校验、ModelMeta 序列化/反序列化 |
| test_preprocessing.py | 36 | 因子对齐（自动推断/手动指定/边界）、标准化工具箱（expanding/rolling/截面/去极值）、特征矩阵构建（Pooled/Per-Symbol）、因子筛选 |
| test_training.py | 20 | 时序切分器（Expanding/Rolling/Embargo）、Walk-Forward（Pooled/Per-Symbol）、Permutation Importance（开关/信号vs噪声/Per-Symbol）、Trainer 一站式训练 |
| test_signal.py | 12 | 信号生成（z-score/rank/clip）、EMA 平滑、空数据/零方差边界 |
| test_portfolio.py | 22 | 凸优化组合构建、约束生成（dollar/beta-neutral/杠杆）、协方差估计（Ledoit-Wolf/样本/指数）、滚动 Beta、波动率目标 |
| test_backtest.py | 13 | 向量化回测、交易成本模型、绩效指标（Sharpe/Sortino/Calmar/回撤持续期）、BacktestResult.summary() |
| test_store.py | 21 | SignalStore CRUD/原子写入/列表/删除、raw_predictions 存取往返/has_signals/has_raw_predictions/缺失报错、Pipeline 端到端 raw_predictions 存储、ModelStore CRUD/工厂加载/因子重要性 |
| test_models.py | 17 | SklearnModelWrapper/LGBMModelWrapper/XGBModelWrapper/TorchModelBase 的 fit/predict/save/load/importance |
| test_pipeline.py | 2 | AlphaPipeline 端到端运行、save 持久化 |

### Phase 2a 测试覆盖

| 测试文件 | 测试项数 | 覆盖内容 |
|---------|---------|---------|
| test_types.py | 10+ | 枚举值、DataRequest 默认值、FactorMeta 序列化 |
| test_base.py | 8 | 三种因子类型的 compute()、基类抽象约束 |
| test_registry.py | 26 | 注册/获取/重复注册/列表/装饰器/参数化注册/register_factor_family/族操作 |
| test_engine.py | 20 | 三种因子类型端到端计算、save/批量/分类过滤/失败容错/compute_factor_instance/prepare_data/compute_family/族分组优化 |
| test_factor_store.py | 8 | 存储/加载/列表/删除/覆盖/元数据持久化 |
| test_catalog.py | 15 | 空目录/扫描/搜索/get_meta/损坏JSON处理/search(family)/list_families/family_summary |
| test_factors.py | 19 | MultiScaleReturns 参数化构造/family字段/注册/已知答案 + OrderbookImbalance 参数化构造/边界 |
| test_evaluation.py | 55+ | IC/分层回测/尾部(含MAE)/换手/非线性/相关性/稳定性/报告/Analyzer/plot |
| test_alignment.py | 12 | 网格对齐/刷新时间/Hayashi-Yoshida |
| test_family_analyzer.py | 19 | sweep/select/robustness/detail/plot_sensitivity/plot_heatmap/边界场景 |

---

## 关键设计决策

### 第一阶段
- **只采集 1m K线**：其他周期由 DataReader 从 1m 按需降采样，减少 API 消耗
- **trade_id 追赶模式**：逐笔成交基于 ID 而非时间拉取，零遗漏
- **SQLite WAL 模式**：K线和市场数据支持读写并发
- **Parquet 按天分区 + 原子写入**：tick/订单簿数据防崩溃损坏

### 第二阶段(a)
- **统一面板输出**：所有因子输出 DataFrame (timestamp × symbol)，评价体系无需适配
- **三层评价 API**：Analyzer 一键报告 → 独立分析函数 → 底层原语，兼顾便利与灵活
- **FactorStore 作为唯一接口**：factor_research 与后续 Phase 2b（模型/策略）仅通过 FactorStore 通信
- **@register_factor / @register_factor_family 装饰器**：因子定义即注册，参数化因子族自动展开
- **FamilyAnalyzer**：参数空间扫描 → 可视化 → 筛选 → 钻取，与 FactorAnalyzer 互补
- **IC 目标**：多 horizon 简单前瞻收益（h=1,5,10,30,60 bars）
- **数据对齐**：各标的独立计算，在因子输出层面对齐（grid + ffill）

### 第二阶段(b)
- **AlphaModel Protocol（非 ABC）**：使用 `Protocol` 而非抽象基类，sklearn 原生模型天然满足 `fit/predict`，无需继承即可使用
- **标准化是工具箱，不是管道步骤**：树模型不需要标准化，线性模型需要但方式各异，标准化是"研究决策"而非"工程决策"
- **信号层默认不截断**：极端预测可能是强信号，风控由 portfolio 层的约束（max_weight、leverage_cap）实现
- **cvxpy 凸优化组合构建**：所有约束（仓位上限、dollar/beta-neutral、杠杆上限）建模为 cvxpy 约束联合求解，不存在顺序冲突
- **Ledoit-Wolf 协方差收缩**：即使 5 标的场景，样本协方差的估计误差也会被优化器放大，Ledoit-Wolf 是业界标准稳健估计
- **Embargo = max(target_horizon, max_factor_lookback)**：同时防止 forward return 标签泄漏和因子 lookback 窗口间接使用测试集数据
- **每个 fold 深拷贝模型**：避免状态污染，确保各 fold 独立训练
- **两层交易成本模型**：固定手续费 + Square-root 市场冲击（Almgren-Chriss 简化版），对流动性差的标的更真实
- **SignalStore 作为唯一下游接口**：alpha_model 与 Phase 3（执行引擎）仅通过 SignalStore 的权重面板通信
- **models/ 是示例代码，不是核心架构**：用户完全可以不用任何封装类，只要实现 `fit/predict` 即可接入

### 第二阶段(c)
- **动态成本模型替代固定惩罚**：3 分量（commission + spread + 1.5 次幂 impact）替代 Phase 2b 的固定 γ‖Δw‖₁
- **收益率空间建模**：全部量在收益率空间表达，与目标函数其他项量纲一致
- **1.5 次幂 impact 保持 SOCP 合规**：cvxpy DCP 验证通过，ECOS 自动求解
- **资金费率纳入目标函数**：永续合约持仓成本/收益不可忽略
- **ADV 参与率约束**：`|Δw_i| × V ≤ participation × ADV_i`，防止单步冲击市场
- **约束零重复**：直接复用 `alpha_model.portfolio.constraints.build_constraints`
- **三路 Fallback**：协方差失败/求解异常/状态非最优 → 返回 current_weights，宁可不交易不可崩溃

### 第三阶段
- **三 RunMode 统一入口**：用户只切 `run_mode` 一个字段在 VECTORIZED / FIXED_GAMMA / DYNAMIC_COST 间切换；不学三套 API
- **WeightsSource Protocol 抽象**：把"决策从哪来"收敛到一个接口；为 v2 SingleAssetTiming 留扩展点
- **PnLTracker 混合模式**：循环内维护时序敏感状态（V / funding / weights / cost），循环后向量化算 equity 和 12 项绩效；最大化精度与速度
- **破产双通道**：`V≤0 finite` 用标志位 + break（保留截断 equity_curve 供分析）；`V=NaN/Inf` 抛 `NumericalError` fail-fast（防止数据 bug 让全部后续 bar 都是 NaN）
- **cost_mode 由消费方各自持有**：经过两轮人工审查从"engine mediator"重构为"Rebalancer/OnlineOptimizer 各自持有 cost_mode"，消除跨模块隐性约定，每个模块独立可测
- **funding 命名严格分离**：engine 侧 `funding_rates_panel: pd.DataFrame`（原始）vs PnLTracker 内部 `funding_events: pd.Series`（已结算事件聚合）；详见 docs/phase3_design.md §12.4
- **canonical schemas 单一来源**：12 个 dict-like / dataclass / DataFrame 的 keys 列表全部固化在 §12，跨模块契约只此一处定义
- **持久化 parquet + JSON 路线**：放弃 pickle（跨 Python/pandas 版本不稳）；`save` 用 `tmp + os.replace` 原子化
- **`schema_version` 演进控制**：`load` 时校验主版本兼容性，为 v2 演进留迁移空间
- **Fail-fast 不降级原则**：所有上游环境问题（tz mismatch / index 缺失 / freq 不符）在 `_validate_environment` 前置暴露；NaN 在 `_check_bankruptcy` 直接抛异常，绝不静默 fillna(0)
- **可比性校验为头等护栏**：`test_consistency.py` 集中三大 rtol=1e-12 护栏（impact 公式四路径一致 / 零摩擦严格等价 / per-symbol 加总恒等），无策略可上线场景下的核心质量保证

---

## 项目阶段路线图

```
研究分支（已完成）                    实盘分支（独立开发）
─────────────────────────             ─────────────────────
Phase 1  data_infra        ✅
   ↓                                   Phase 4  实盘交易系统  🚧
Phase 2a factor_research   ✅              （ccxt 实盘下单 + 风控守护
   ↓                                        + 监控告警 + Phase 2c 直接复用 +
Phase 2b alpha_model       ✅                 Phase 1 数据继续采集）
   ↓
Phase 2c execution_optimizer ✅
   ↓
Phase 3  backtest_engine   ✅
（研究分支至此完成 — 因子→策略→回测全链路打通）
```

**研究分支与实盘分支的并列关系**：两者都依赖 Phase 1-2(a/b/c)；Phase 3 与 Phase 4 共享 `ExecutionOptimizer.optimize_step()` 接口（同一动态成本优化在回测和实盘里行为一致）。

## 交易对

当前配置 5 个币对（可在 `data_infra/config/settings.py` 中修改）：

- BTC/USDT
- DOGE/USDT
- SOL/USDT
- BNB/USDT
- ETH/USDT

## 部署建议

- **云服务器**：香港/新加坡/日本，2核CPU / 2-4GB内存 / 100-200GB存储
- 直连 Binance，无需代理
- 每个采集脚本用 `nohup` 或 `systemd` 后台运行
- 定期 `rsync` 数据到本地用于研究
