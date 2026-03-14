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
│   └── tests/                  #   单元测试（198 项）
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
├── db/                         # 共享数据存储（各模块通过此目录通信）
│   └── factors/                #   因子存储目录（FactorStore 的数据根）
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

# 将 tick 级数据对齐到 1 秒网格
panel = grid_align({"BTC": btc_series, "ETH": eth_series}, freq="1s", max_gap=5)

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
| `quantile_backtest(factor_panel, price_panel, n_groups, horizon)` | `evaluation.quantile` | 因子面板 + 价格面板 | group_returns, long_short, monotonicity |
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
| `sharpe_ratio(returns, periods_per_year, risk_free_rate)` | `evaluation.metrics` | 夏普比率 |
| `max_drawdown(cumulative)` | `evaluation.metrics` | 最大回撤 |

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

## 运行测试

```bash
# 第一阶段测试（125 项）
python -m pytest data_infra/tests/ -v

# 第二阶段(a) 测试（198 项）
python -m pytest factor_research/tests/ -v

# 全部测试
python -m pytest data_infra/tests/ factor_research/tests/ -v  # 预期 323 项全通过
```

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
