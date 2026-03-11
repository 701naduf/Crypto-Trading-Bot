# Crypto Trading Bot — 第一阶段：数据基础设施

量化加密货币交易机器人，聚焦 Binance 交易所。本阶段构建完整的数据采集、存储、读取管道。

## 项目结构

```
Crypto-Trading-Bot/
├── config/
│   ├── __init__.py
│   └── settings.py              # 全局配置（交易所、币对、路径、容错等）
├── utils/
│   ├── logger.py                # 统一日志（控制台 + 文件，按日轮转）
│   ├── time_utils.py            # 时间工具（ms ↔ datetime, 时间对齐）
│   ├── retry.py                 # 重试装饰器（异常分类 + 指数退避）
│   └── heartbeat.py             # 运行状态监控（心跳日志 + 状态文件）
├── data/
│   ├── fetcher.py               # K线拉取（同步 REST）
│   ├── tick_fetcher.py          # 逐笔成交拉取（异步, trade_id 追赶）
│   ├── orderbook_fetcher.py     # 订单簿采集（异步 WebSocket）
│   ├── market_fetcher.py        # 合约市场数据拉取（同步 REST）
│   ├── kline_store.py           # SQLite K线存储（WAL 模式）
│   ├── tick_store.py            # Parquet 逐笔成交存储（按天分区, 原子写入）
│   ├── orderbook_store.py       # Parquet 订单簿存储（缓冲 + 刷盘）
│   ├── market_store.py          # SQLite 合约市场数据存储
│   ├── aggregator.py            # 数据聚合（tick→OHLCV, K线降采样）
│   ├── validator.py             # 数据质量校验
│   ├── writer.py                # 统一写入入口（校验→存储）
│   └── reader.py                # 统一读取入口（自动路由 + 聚合）
├── scripts/
│   ├── collect_klines.py        # K线采集（独立进程, 7×24）
│   ├── collect_ticks.py         # 逐笔成交采集（独立进程, 7×24）
│   ├── collect_orderbook.py     # 订单簿采集（独立进程, WebSocket, 7×24）
│   ├── collect_market.py        # 合约市场数据采集（独立进程, 7×24）
│   ├── backfill.py              # 历史数据回填
│   ├── check_data.py            # 数据质量巡检
│   └── status.py                # 查看所有采集脚本运行状态
├── tests/                       # 单元测试（125 项）
├── db/                          # 数据存储（.gitignore）
├── logs/                        # 日志 + 状态文件（.gitignore）
├── requirements.txt
└── .env                         # API 密钥（.gitignore）
```

## 采集的数据类型

| 数据类型 | 采集方式 | 频率 | 存储 | 脚本 |
|---------|---------|------|------|------|
| 1m K线 | REST 轮询 | 60s | SQLite | `collect_klines.py` |
| 逐笔成交 | REST trade_id追赶 | 持续 | Parquet/天 | `collect_ticks.py` |
| 10档订单簿 | WebSocket推送 | 100ms | Parquet/天 | `collect_orderbook.py` |
| 合约市场数据 | REST 轮询 | 5min | SQLite | `collect_market.py` |

合约市场数据包含：资金费率、持仓量(OI)、多空持仓比、主动买卖量。

## 快速开始

### 1. 环境准备

```bash
# Python >= 3.10
pip install -r requirements.txt
```

### 2. 配置 API 密钥

在项目根目录创建 `.env` 文件：

```env
BINANCE_API_KEY=your_api_key_here
BINANCE_SECRET=your_secret_here

# 本地开发需要代理时（云服务器不需要）
PROXY_HOST=127.0.0.1
PROXY_PORT=7890
```

### 3. 运行测试

```bash
python -m pytest tests/ -v
```

### 4. 启动采集

每个采集脚本是独立进程，可单独启停：

```bash
# K线采集（每60秒轮询一次）
python -m scripts.collect_klines

# 逐笔成交采集（持续追赶模式）
python -m scripts.collect_ticks

# 订单簿采集（WebSocket 100ms推送）
python -m scripts.collect_orderbook

# 合约市场数据采集（每5分钟轮询）
python -m scripts.collect_market
```

### 5. 查看采集状态

```bash
python -m scripts.status
```

### 6. 数据质量巡检

```bash
python -m scripts.check_data
```

---

## 测试指南

### 运行全部测试

```bash
python -m pytest tests/ -v
```

预期结果：125 项全部通过。

### 测试分类

#### 离线测试（不需要网络，日常开发使用）

```bash
# 配置模块
python -m pytest tests/test_config.py -v

# 时间工具
python -m pytest tests/test_time_utils.py -v

# 数据校验
python -m pytest tests/test_validator.py -v

# K线存储（使用临时数据库）
python -m pytest tests/test_kline_store.py -v

# Tick存储（使用临时目录）
python -m pytest tests/test_tick_store.py -v

# 订单簿存储
python -m pytest tests/test_orderbook_store.py -v

# 合约市场数据存储
python -m pytest tests/test_market_store.py -v

# 数据聚合
python -m pytest tests/test_aggregator.py -v

# DataReader路由
python -m pytest tests/test_reader.py -v
```

#### 在线功能测试（需要网络+API密钥，手动运行）

以下测试需要连接 Binance API，建议手动在终端中测试：

**测试 1: K线拉取**
```bash
python -c "
from data.fetcher import KlineFetcher
f = KlineFetcher()
df = f.fetch_ohlcv('BTC/USDT', '1m', limit=5)
print(df)
print(f'拉取 {len(df)} 根K线')
"
```

**测试 2: K线写入+读取完整流程**
```bash
python -c "
from data.fetcher import KlineFetcher
from data.writer import DataWriter
from data.reader import DataReader

fetcher = KlineFetcher()
writer = DataWriter()
reader = DataReader()

# 拉取
df = fetcher.fetch_ohlcv('BTC/USDT', '1m', limit=100)
print(f'拉取: {len(df)} 根')

# 写入
count = writer.write_ohlcv(df, 'BTC/USDT', '1m')
print(f'新增: {count} 根')

# 读取 1m
df_1m = reader.get_ohlcv('BTC/USDT', '1m')
print(f'1m: {len(df_1m)} 根')

# 自动降采样到 5m
df_5m = reader.get_ohlcv('BTC/USDT', '5m')
print(f'5m (降采样): {len(df_5m)} 根')
print(df_5m.tail())
"
```

**测试 3: K线单轮采集**
```bash
python -m scripts.collect_klines --once --symbols BTC/USDT
```

**测试 4: Tick 拉取**
```bash
python -c "
import asyncio
from data.tick_fetcher import TickFetcher

async def test():
    f = TickFetcher()
    try:
        df = await f.fetch_trades('BTC/USDT', limit=10)
        print(df)
        print(f'拉取 {len(df)} 笔成交')
    finally:
        await f.close()

asyncio.run(test())
"
```

**测试 5: 合约市场数据**
```bash
python -c "
from data.market_fetcher import MarketFetcher
f = MarketFetcher()

# 资金费率
fr = f.fetch_funding_rate('BTC/USDT', limit=3)
print('=== 资金费率 ===')
print(fr)

# 持仓量
oi = f.fetch_open_interest('BTC/USDT')
print('\n=== 持仓量 ===')
print(oi)

# 多空持仓比
lsr = f.fetch_long_short_ratio('BTC/USDT', limit=3)
print('\n=== 多空持仓比 ===')
print(lsr)

# 主动买卖量
tbs = f.fetch_taker_buy_sell_volume('BTC/USDT', limit=3)
print('\n=== 主动买卖量 ===')
print(tbs)
"
```

**测试 6: 合约市场数据单轮采集**
```bash
python -m scripts.collect_market --once --symbols BTC/USDT
```

**测试 7: 心跳与状态**
```bash
# 启动K线采集（后台）
python -m scripts.collect_klines &

# 等待一段时间后查看状态
python -m scripts.status
```

**测试 8: 历史回填**
```bash
# 回填 1 天的 K线历史
python -m scripts.backfill --type kline --start 2024-12-01 --end 2024-12-01 --symbols BTC/USDT
```

### 测试覆盖范围

| 模块 | 测试文件 | 测试项数 | 测试内容 |
|------|---------|---------|---------|
| config | test_config.py | 10 | 配置加载、格式校验 |
| time_utils | test_time_utils.py | 14 | 时间转换、对齐、周期判断 |
| validator | test_validator.py | 14 | OHLCV/Tick/订单簿/市场数据校验 |
| kline_store | test_kline_store.py | 7 | 写入/读取/幂等/范围查询 |
| tick_store | test_tick_store.py | 6 | 写入/去重/跨天/原子写入 |
| orderbook_store | test_orderbook_store.py | 5 | 缓冲/自动刷盘/读取/档位过滤 |
| market_store | test_market_store.py | 8 | 四种数据写入/读取/幂等 |
| aggregator | test_aggregator.py | 7 | tick聚合/K线降采样 |
| reader | test_reader.py | 6 | 路由/1m直读/5m降采样/1h降采样 |

---

## 关键设计决策

- **只采集 1m K线**：其他周期由 DataReader 从 1m 按需降采样，减少 API 消耗
- **trade_id 追赶模式**：逐笔成交基于 ID 而非时间拉取，零遗漏
- **SQLite WAL 模式**：K线和市场数据支持读写并发
- **Parquet 按天分区 + 原子写入**：tick/订单簿数据防崩溃损坏
- **采集模块完全独立**：每个脚本独立进程，任意一个停止不影响其他
- **采集与消费通过存储层解耦**：上层模块只依赖 DataReader

## 交易对

当前配置 5 个币对（可在 `config/settings.py` 中修改）：

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
