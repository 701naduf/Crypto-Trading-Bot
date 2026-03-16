# Phase 1 数据基建 — 代码学习指引

本文档是对照源码的系统学习路径，帮助你从零理解整个数据基建层的设计与实现。

建议按**自底向上**的顺序阅读：先理解基础工具，再看数据层，最后看采集脚本。
每个模块标注了**核心知识点**和**推荐的阅读重点行**，方便你在源码中定位。

---

## 目录

- [一、项目整体结构](#一项目整体结构)
- [二、建议学习路线](#二建议学习路线)
- [三、第 1 站：配置层 config/](#三第-1-站配置层-config)
- [四、第 2 站：工具层 utils/](#四第-2-站工具层-utils)
- [五、第 3 站：数据拉取层 Fetcher](#五第-3-站数据拉取层-fetcher)
- [六、第 4 站：数据存储层 Store](#六第-4-站数据存储层-store)
- [七、第 5 站：数据门面层 Writer / Reader / Validator](#七第-5-站数据门面层-writer--reader--validator)
- [八、第 6 站：数据处理层 Aggregator](#八第-6-站数据处理层-aggregator)
- [九、第 7 站：采集脚本 scripts/](#九第-7-站采集脚本-scripts)
- [十、第 8 站：测试 tests/](#十第-8-站测试-tests)
- [十一、核心设计模式总结](#十一核心设计模式总结)
- [十二、数据流全景图](#十二数据流全景图)
- [十三、延伸练习](#十三延伸练习)

---

## 一、项目整体结构

```
Crypto-Trading-Bot/
├── data_infra/                # 数据基础设施（Phase 1）
│   ├── config/                # 配置层（最底层，无依赖）
│   │   ├── __init__.py
│   │   └── settings.py        # 全局配置，所有模块的参数来源
│   │
│   ├── utils/                 # 工具层（依赖 config）
│   │   ├── logger.py          # 统一日志（控制台 + 按天轮转文件）
│   │   ├── time_utils.py      # 时间转换（ms ↔ datetime, 对齐, 周期转换）
│   │   ├── retry.py           # 重试装饰器（同步/异步, 异常分类, 指数退避）
│   │   └── heartbeat.py       # 运行状态监控（心跳日志 + 状态JSON文件）
│   │
│   ├── data/                  # 数据层核心
│   │   ├── fetcher.py         # K线拉取（同步 REST, ccxt）
│   │   ├── tick_fetcher.py    # 逐笔成交拉取（异步 REST, ccxt async）
│   │   ├── orderbook_fetcher.py # 订单簿拉取（异步 WebSocket）
│   │   ├── market_fetcher.py  # 合约市场数据拉取（同步 REST, 合约API）
│   │   │
│   │   ├── kline_store.py     # K线存储（SQLite + WAL）
│   │   ├── tick_store.py      # 逐笔成交存储（Parquet 按天分区）
│   │   ├── orderbook_store.py # 订单簿存储（Parquet + 内存缓冲刷盘）
│   │   ├── market_store.py    # 市场数据存储（SQLite 四张表）
│   │   │
│   │   ├── validator.py       # 数据校验（写入校验 + 完整性巡检）
│   │   ├── writer.py          # 统一写入入口（校验 → Store）
│   │   ├── reader.py          # 统一读取入口（路由 → Store/聚合）
│   │   └── aggregator.py      # 数据聚合（tick→OHLCV, 1m→5m/1h）
│   │
│   ├── scripts/               # 采集脚本（独立进程）
│   │   ├── collect_klines.py  # K线持续采集
│   │   ├── collect_ticks.py   # 逐笔成交持续采集（异步）
│   │   ├── collect_orderbook.py # 订单簿持续采集（WebSocket）
│   │   ├── collect_market.py  # 合约市场数据持续采集
│   │   ├── backfill.py        # 历史数据回填
│   │   ├── check_data.py      # 数据质量巡检 + 自动修复
│   │   └── status.py          # 查看各采集器运行状态
│   │
│   └── tests/                 # 单元测试（14个文件, 125个测试用例）
│
├── db/                        # 共享数据存储（各模块通过此目录通信）
├── logs/                      # 日志 + 状态文件
├── main.py                    # 入口（用法说明）
└── requirements.txt           # 依赖清单
```

**依赖方向（只能向下依赖，不能向上）：**

```
scripts/  →  data/  →  utils/  →  config/
tests/  →  data/  →  utils/  →  config/
```

---

## 二、建议学习路线

```
第1站 data_infra/config/settings.py      ← 理解所有配置项
  ↓
第2站 data_infra/utils/ (4个文件)         ← 理解基础工具
  ↓
第3站 data_infra/data/fetcher 系列 (4个文件) ← 理解数据从哪来
  ↓
第4站 data_infra/data/store 系列 (4个文件)   ← 理解数据存到哪去
  ↓
第5站 data_infra/data/writer + reader + validator ← 理解门面模式
  ↓
第6站 data_infra/data/aggregator.py      ← 理解数据聚合
  ↓
第7站 data_infra/scripts/ (7个文件)       ← 理解如何组装成完整采集流程
  ↓
第8站 data_infra/tests/ (14个文件)        ← 理解如何测试、学习mock技巧
```

每个站建议先看**模块顶部的文档字符串**，再看代码。
文档字符串是设计说明，代码是实现细节。

---

## 三、第 1 站：配置层 config/

### 文件：`data_infra/config/settings.py`

**核心知识点：**
- **环境变量 + .env 文件**：敏感信息（API Key）通过 `python-dotenv` 从 `.env` 文件读取，不硬编码在源码中
- **集中配置管理**：所有可调参数在一个文件中，其他模块通过 `from data_infra.config import settings` 引用
- **PROJECT_ROOT 的计算**：`Path(__file__).resolve().parent.parent.parent`——从 `data_infra/config/settings.py` 向上三级得到项目根目录

**阅读重点：**

| 行范围 | 内容 | 学什么 |
|--------|------|--------|
| 15-27 | 项目根路径和 .env 加载 | `pathlib.Path` 路径操作, `load_dotenv` |
| 34-46 | 交易所配置 | 为什么 API Key 不硬编码 |
| 53-59 | SYMBOLS 列表 | 全局币对配置的集中管理 |
| 64-76 | K线采集配置 | 为什么只采集 1m（注释讲了原因） |
| 82-99 | Tick 采集配置 | 追赶模式的参数含义, 冷启动策略 |
| 104-110 | 订单簿配置 | WebSocket 推送频率选择 |
| 137-158 | 容错配置 | 重试次数, 退避时间, 限频等待的取值依据 |

**思考题：**
1. 如果要新增一个交易对 `XRP/USDT`，需要改哪里？（答：只改 SYMBOLS 列表）
2. 为什么 KLINE_COLLECT_INTERVAL 设为 60 秒？

---

## 四、第 2 站：工具层 utils/

### 4.1 `data_infra/utils/logger.py` — 统一日志

**核心知识点：**
- **logging 模块三要素**：Logger, Handler, Formatter
- **双输出**：控制台（INFO级别）+ 文件（DEBUG级别，按天轮转保留30天）
- **防重复 handler**：`if logger.handlers: return logger`——模块被多次 import 时不重复添加

**阅读重点：**

| 行 | 学什么 |
|----|--------|
| 52-56 | `logging.getLogger(name)` 的单例模式 + 防重复添加 |
| 62-65 | Formatter 的格式字符串设计 |
| 80-89 | `TimedRotatingFileHandler` 按天轮转日志 |

### 4.2 `data_infra/utils/time_utils.py` — 时间处理

**核心知识点：**
- **毫秒时间戳 ↔ datetime** 的互转（交易所 API 用毫秒，Python 用 datetime）
- **时间对齐**：将任意时间"向下取整"到 K线 周期边界（如 10:23:45 → 10:20:00 for 5m）
- **周期字符串解析**：`"5m"` → 300 秒

**阅读重点：**

| 行 | 学什么 |
|----|--------|
| 36-57 | `datetime_to_ms`：naive datetime 的时区处理策略 |
| 60-97 | `align_to_timeframe`：UNIX 时间戳整除取整的巧妙用法 |
| 100-142 | `timeframe_to_seconds`：字符串解析的简洁实现 |

**动手练习：**
```python
from data_infra.utils.time_utils import *
from datetime import datetime, timezone

# 试试这些转换
dt = datetime(2024, 1, 15, 10, 23, 45, tzinfo=timezone.utc)
print(datetime_to_ms(dt))              # → 毫秒时间戳
print(ms_to_datetime(1705305600000))   # → datetime 对象
print(align_to_timeframe(dt, "5m"))    # → 10:20:00
print(align_to_timeframe(dt, "1h"))    # → 10:00:00
print(timeframe_to_seconds("4h"))      # → 14400
```

### 4.3 `data_infra/utils/retry.py` — 重试与容错

**核心知识点：**
- **装饰器模式**：`@retry_on_failure()` 给函数自动加上重试逻辑
- **异常分类**：ccxt 的异常体系树（NetworkError → 重试, BadRequest → 不重试, RateLimitExceeded → 长等待）
- **指数退避**：等待时间 = base_delay × 2^attempt，上限 60 秒
- **同步/异步双版本**：`time.sleep` vs `asyncio.sleep`

**阅读重点：**

| 行 | 学什么 |
|----|--------|
| 47-94 | `classify_exception`：ccxt 异常类层级及判断顺序（子类先判断） |
| 97-122 | `get_wait_time`：不同异常类别的等待策略 |
| 125-193 | `retry_on_failure`：**三层嵌套装饰器**的标准写法（工厂→装饰器→包装器） |
| 196-257 | `async_retry_on_failure`：async 版对比——只有 `await` 和 `asyncio.sleep` 的区别 |

**重点理解：装饰器嵌套结构**
```python
def retry_on_failure(max_retries=None):   # 第1层: 装饰器工厂（接受参数）
    def decorator(func):                   # 第2层: 真正的装饰器（接受函数）
        @functools.wraps(func)             # 保留原函数的元信息
        def wrapper(*args, **kwargs):      # 第3层: 包装函数（替换原函数）
            ...
        return wrapper
    return decorator
```

### 4.4 `data_infra/utils/heartbeat.py` — 运行状态监控

**核心知识点：**
- **长期运行进程的可观测性**：定期输出心跳日志 + 写入状态 JSON 文件
- **原子写入**：先写 `.tmp` 文件再 `os.replace`（同一文件系统上的原子操作）
- **`time.monotonic()` vs `time.time()`**：monotonic 不受系统时间调整影响，适合测量间隔

**阅读重点：**

| 行 | 学什么 |
|----|--------|
| 91-111 | `start()`：monotonic 计时器初始化 |
| 113-137 | `update()`：按币对维度的统计累加 + kwargs 灵活扩展 |
| 169-187 | `tick()`：基于时间间隔的双检查（心跳日志 + 状态文件） |
| 240-268 | `_write_status_file()`：原子写入 + JSON 状态快照 |

---

## 五、第 3 站：数据拉取层 Fetcher

四个 Fetcher 代表了三种不同的数据获取模式，是学习 API 交互的好材料。

### 5.1 `data_infra/data/fetcher.py` — K线拉取（同步 REST）

**核心知识点：**
- **ccxt 统一接口**：`exchange.fetch_ohlcv(symbol, timeframe, since, limit)` 对接任意交易所
- **动态创建交易所实例**：`getattr(ccxt, "binance")` 等价于 `ccxt.binance`
- **分页拉取**：`fetch_ohlcv_batch` 通过循环调用 + since 参数递推实现自动翻页
- **不满额退出**：返回 < limit 条说明已到数据尾部

**阅读重点：**

| 行 | 学什么 |
|----|--------|
| 55-76 | 交易所实例配置：API Key, 超时, 代理, 频率限制 |
| 80-134 | `fetch_ohlcv`：ccxt 返回的原始格式（嵌套列表）→ DataFrame 的转换 |
| 136-215 | `fetch_ohlcv_batch`：**分页循环的标准范式**——since 递推 + 不满额退出 + 去重合并 |

### 5.2 `data_infra/data/tick_fetcher.py` — 逐笔成交拉取（异步 REST）

**核心知识点：**
- **async/await 基础**：异步函数的定义与调用
- **ccxt.async_support**：ccxt 的异步版本，底层用 aiohttp
- **trade_id 追赶模式**：通过 `fromId` 参数从指定位置继续拉取，满额立即继续，不满额表示追上实时
- **冷启动策略**："latest"（放弃历史快速启动）vs "from_date"（从指定日期追赶）
- **资源释放**：异步交易所实例必须显式 `await close()`

**阅读重点：**

| 行 | 学什么 |
|----|--------|
| 78-146 | `fetch_trades`：async 函数 + ccxt params 传递交易所私有参数 |
| 148-203 | `fetch_until_latest`：**追赶模式的核心逻辑**——while True + 满额判断 |
| 205-268 | `resolve_cold_start`：两种冷启动模式的实现，`startTime` 参数的用法 |

**对比学习：** 对照 `fetcher.py`（同步）和 `tick_fetcher.py`（异步），观察差异：
- `def` vs `async def`
- `self.exchange.method()` vs `await self.exchange.method()`
- `ccxt.binance` vs `ccxt_async.binance`
- 同步版无需 `close()`，异步版必须

### 5.3 `data_infra/data/orderbook_fetcher.py` — 订单簿拉取（异步 WebSocket）

**核心知识点：**
- **WebSocket vs REST**：REST 是主动请求/被动响应，WebSocket 是建立连接后被动接收推送
- **Binance Combined Stream**：多个币对共用一个 WebSocket 连接
- **回调模式**：`on_snapshot` 回调函数——数据到达时自动触发处理
- **断线重连 + 指数退避**：`WS_RECONNECT_DELAY × 2^(n-1)`，上限 60 秒
- **stream 名称映射**：`"btcusdt@depth10@100ms"` ↔ `"BTC/USDT"`

**阅读重点：**

| 行 | 学什么 |
|----|--------|
| 60-91 | `__init__`：stream 名称映射表的构建 |
| 93-117 | `_make_stream_name` + `_build_ws_url`：WebSocket URL 的拼装规则 |
| 119-163 | `start`：**外层重连循环**——`while self._running + try/except + sleep` |
| 165-190 | `_connect_and_receive`：`websockets.connect` + `async for message` 接收消息 |
| 192-234 | `_process_message`：**JSON 解析 + 字符串→float 转换**（Binance 返回字符串格式的数字） |
| 236-244 | `_get_reconnect_delay`：指数退避公式 |

**对比三种拉取模式：**

| | K线 | Tick | 订单簿 |
|---|---|---|---|
| 协议 | REST | REST | WebSocket |
| 同步/异步 | 同步 | 异步 | 异步 |
| 驱动方式 | 轮询 | 追赶 | 推送 |
| 频率 | 60s/轮 | 满额立即继续 | 100ms |
| 历史回填 | 支持 | 支持 | 不支持 |

### 5.4 `data_infra/data/market_fetcher.py` — 合约市场数据拉取

**核心知识点：**
- **ccxt 合约模式**：`options["defaultType"] = "future"` 切换到合约 API
- **ccxt 隐式 API 方法**：通过 `exchange.fapiPublicGet*` / `exchange.fapiDataGet*` 调用 Binance 未封装的私有 API（ccxt v4 仅保留 camelCase 命名，且数据分析接口归入 `fapiData` 分区）
- **符号转换**：`"BTC/USDT"` → `"BTC/USDT:USDT"`（ccxt 合约格式）→ `"BTCUSDT"`（Binance API 格式）

**阅读重点：**

| 行 | 学什么 |
|----|--------|
| 37-64 | 两个辅助函数：现货→合约符号, 现货→Binance原生符号 |
| 88-96 | 交易所实例配置：`"defaultType": "future"` 的作用 |
| 109-148 | `fetch_funding_rate`：ccxt 封装好的接口 `fetch_funding_rate_history` |
| 152-194 | `fetch_open_interest`：ccxt v4 隐式方法 `fapiPublicGetOpenInterest`（fapiPublic 分区） |
| 197-241 | `fetch_long_short_ratio`：ccxt v4 隐式方法 `fapiDataGetTopLongShortPositionRatio`（fapiData 分区） |
| 243-289 | `fetch_taker_buy_sell_volume`：ccxt v4 隐式方法 `fapiDataGetTakerlongshortRatio`（fapiData 分区） |

---

## 六、第 4 站：数据存储层 Store

四个 Store 覆盖了两种存储引擎（SQLite + Parquet），以及不同的写入策略。

### 6.1 `data_infra/data/kline_store.py` — K线 SQLite 存储

**核心知识点：**
- **SQLite WAL 模式**：`PRAGMA journal_mode=WAL`——允许读写并发（采集写入不阻塞查询读取）
- **幂等写入**：`INSERT OR IGNORE`——主键冲突时静默跳过，重复写入不会产生脏数据
- **计算新增行数的技巧**：写入前后各 COUNT(*) 一次，差值即为新增

**阅读重点：**

| 行 | 学什么 |
|----|--------|
| 69-91 | `_init_db`：WAL 模式启用 + 建表语句 |
| 102-168 | `write`：iterrows 构造记录 + executemany 批量写入 + 计数技巧 |
| 170-210 | `read`：动态拼接 SQL WHERE 条件 + pd.read_sql_query |
| 212-240 | `get_latest_timestamp`：MAX(timestamp) 查询 + 字符串→datetime 解析 |

**动手练习：**
```python
from data_infra.data.kline_store import KlineStore
import pandas as pd, tempfile, os

# 用临时文件做实验
db_path = os.path.join(tempfile.mkdtemp(), "test.db")
store = KlineStore(db_path)

# 写入
df = pd.DataFrame({
    "timestamp": pd.date_range("2024-01-15 10:00", periods=5, freq="1min", tz="UTC"),
    "open": [100, 101, 102, 103, 104],
    "high": [110, 111, 112, 113, 114],
    "low": [90, 91, 92, 93, 94],
    "close": [101, 102, 103, 104, 105],
    "volume": [10, 20, 30, 40, 50],
})
print(store.write(df, "BTC/USDT", "1m"))     # → 5（新增5行）
print(store.write(df, "BTC/USDT", "1m"))     # → 0（全部重复，跳过）
print(store.count("BTC/USDT", "1m"))         # → 5
print(store.get_latest_timestamp("BTC/USDT", "1m"))
```

### 6.2 `data_infra/data/tick_store.py` — 逐笔成交 Parquet 存储

**核心知识点：**
- **Parquet 格式**：列式存储，适合分析型读取，Snappy 压缩
- **按天分区**：`db/ticks/BTC_USDT/2024-01-15.parquet`——便于按时间范围读取和管理
- **原子写入**：`.tmp` 文件 + `os.replace`——防止写入过程中崩溃导致数据损坏
- **PyArrow Schema**：定义统一的列类型，确保所有文件格式一致
- **trade_id 去重**：合并新旧数据时 `drop_duplicates(subset=["trade_id"])`

**阅读重点：**

| 行 | 学什么 |
|----|--------|
| 40-46 | `TICK_SCHEMA`：PyArrow schema 定义 |
| 72-87 | `_cleanup_tmp_files`：启动时清理崩溃残留——**防御性编程** |
| 118-156 | `write`：按天拆分 + 原子写入的完整流程 |
| 158-189 | `_write_day`：读旧→合并→去重→写tmp→rename 的**原子写入五步法** |
| 251-287 | `get_latest_trade_id`：找最新文件→读 trade_id 列→取 max |

**对比 SQLite vs Parquet：**

| | SQLite (kline, market) | Parquet (tick, orderbook) |
|---|---|---|
| 数据量 | 小（万级/天） | 大（百万级/天） |
| 写入方式 | INSERT OR IGNORE | 读旧→合并→重写 |
| 查询方式 | SQL | pandas read_parquet |
| 并发 | WAL 模式 | 文件级隔离 |
| 适用场景 | 结构化查询、小数据 | 大批量分析、列式读取 |

### 6.3 `data_infra/data/orderbook_store.py` — 订单簿存储（带内存缓冲）

**核心知识点：**
- **内存缓冲 + 批量刷盘**：每条数据先进内存缓冲，满 10000 条后一次性写入磁盘（减少 I/O）
- **嵌套→扁平展开**：`{"bids": [[price, qty], ...]}` → `bid_price_0, bid_qty_0, ...` 扁平列
- **为什么用扁平列**：Parquet 列式存储对扁平列压缩率最优；读取时可选择性加载特定档位
- **退出安全**：`flush_and_close()` 确保缓冲中的数据在进程退出前写入磁盘

**阅读重点：**

| 行 | 学什么 |
|----|--------|
| 44-74 | `_build_schema` / `_get_column_names`：动态构建 10 档 × 4 列的 schema |
| 88-97 | `__init__`：buffer_size 的内存估算（注释中有） |
| 137-181 | `append`：嵌套→扁平转换 + 缓冲满时自动触发 flush |
| 238-307 | `read`：按需加载档位（`levels` 参数）减少内存占用 |

### 6.4 `data_infra/data/market_store.py` — 市场数据 SQLite 存储

**核心知识点：**
- **四表共用一个数据库**：`market.db` 中四张表分别存储不同类型的市场数据
- **通用写入方法**：`_insert_many` 是一个通用的批量写入底层方法
- **SQL 注入防护**：`read` 方法中校验 table 是否在白名单中（因为表名无法用参数化查询）

**阅读重点：**

| 行 | 学什么 |
|----|--------|
| 80-131 | `_init_db`：一次性创建四张表 |
| 260-298 | `_insert_many`：通用的 INSERT OR IGNORE + 计数 |
| 325-333 | `read` 中的 SQL 注入防护：白名单校验 |

---

## 七、第 5 站：数据门面层 Writer / Reader / Validator

这三个模块是数据层的**门面（Facade）**，将底层复杂性封装起来。

### 7.1 `data_infra/data/writer.py` — 统一写入入口

**核心知识点：**
- **门面模式**：采集脚本不直接操作 Store，而是通过 DataWriter 间接写入
- **写入前校验**：每个 write 方法内部先调用 validator，不合格数据被过滤
- **断点续传查询**：DataWriter 同时提供 `get_latest_*` 查询接口，采集脚本用来确定增量起点

**阅读重点：**

| 行 | 学什么 |
|----|--------|
| 40-51 | `__init__`：一次性初始化所有四种 Store |
| 57-79 | `write_ohlcv`：**校验→过滤→写入** 的标准三步 |
| 111-137 | 订单簿写入的特殊性：append + flush 两步分离 |
| 188-203 | 断点续传查询接口 |

### 7.2 `data_infra/data/reader.py` — 统一读取入口

**核心知识点：**
- **路由模式**：`get_ohlcv("BTC/USDT", "5m")` 自动路由到 SQLite 读 1m + 降采样
- **三条读取路径**：1m 直读 / 标准周期降采样 / 亚分钟从 tick 聚合
- **按天分片防 OOM**：亚分钟周期的 tick 聚合按天处理，内存中同时只有一天的数据

**阅读重点：**

| 行 | 学什么 |
|----|--------|
| 61-97 | `get_ohlcv`：**三路分发**的路由逻辑 |
| 99-154 | `_ohlcv_from_ticks`：按天分片聚合的实现——内存友好 |

### 7.3 `data_infra/data/validator.py` — 数据校验

**核心知识点：**
- **校验哲学**：宁可漏放，不可误杀——只过滤明显不合理的数据
- **返回值设计**：返回 `(valid_df, invalid_df)` 二元组，让调用方决定如何处理
- **两类校验**：写入校验（实时）+ 完整性巡检（定期）

**阅读重点：**

| 行 | 学什么 |
|----|--------|
| 32-82 | `validate_ohlcv`：K线校验规则——OHLC 关系约束（high >= max(O,C)） |
| 123-170 | `validate_orderbook`：买卖盘不交叉检查 `bid_0 < ask_0` |
| 279-335 | `check_kline_continuity`：时间缺口检测——相邻 K线 时间差分析 |

---

## 八、第 6 站：数据处理层 Aggregator

### `data_infra/data/aggregator.py`

**核心知识点：**
- **tick → OHLCV**：按时间窗口分组聚合（resample）
- **低周期 → 高周期**：1m → 5m/1h 等降采样
- **pandas resample**：`origin="epoch"` 确保窗口对齐到 UNIX 纪元
- **整数倍校验**：目标周期必须是源周期的整数倍

**阅读重点：**

| 行 | 学什么 |
|----|--------|
| 30-106 | `aggregate_ticks_to_ohlcv`：set_index → resample → agg 的 pandas 标准流程 |
| 109-178 | `resample_ohlcv`：OHLCV 降采样的聚合规则——open 取 first, high 取 max, ... |
| 143-147 | 整数倍校验：`target_seconds % source_seconds != 0` 时报错 |

---

## 九、第 7 站：采集脚本 scripts/

采集脚本是整个系统的**组装点**——将 Fetcher、Writer、Heartbeat 组合成完整的采集流程。

### 9.1 通用结构模板

四个采集脚本都遵循相同的结构：

```python
def main():
    # 1. 解析命令行参数
    args = parse_args()
    symbols = args.symbols or settings.SYMBOLS

    # 2. 初始化组件
    fetcher = XxxFetcher()
    writer = DataWriter()
    heartbeat = Heartbeat("collect_xxx")

    # 3. 注册信号处理（优雅退出）
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 4. 启动心跳
    heartbeat.start()

    # 5. 主循环
    while running:
        for symbol in symbols:
            try:
                # a. 查询断点
                # b. 拉取数据
                # c. 写入（Writer 内部校验）
                # d. 更新心跳统计
            except Exception as e:
                heartbeat.report_error(e)

        heartbeat.tick()

        # 等待下一轮（可被信号中断的 sleep）
        for _ in range(interval):
            if not running: break
            time.sleep(1)

    # 6. 清理退出
    heartbeat.stop()
```

### 9.2 各脚本的差异

**`data_infra/scripts/collect_klines.py`**（最简单，建议第一个读）

| 行 | 学什么 |
|----|--------|
| 63-69 | 信号处理：`nonlocal running` 实现优雅退出 |
| 82-90 | 断点续传：查询最新 K线 时间 → 计算下一个 since |
| 117-120 | **可中断的 sleep**：`for _ in range(60): sleep(1)` 而不是 `sleep(60)` |

**`data_infra/scripts/collect_ticks.py`**（异步版）

| 行 | 学什么 |
|----|--------|
| 43-115 | `async def run`：异步主循环，`asyncio.run(run(symbols))` 启动 |
| 63 | `heartbeat.set_status("catching_up")`：区分追赶中 vs 空闲状态 |
| 74-78 | 冷启动处理：首次采集时调用 `resolve_cold_start` |

**`data_infra/scripts/collect_orderbook.py`**（回调模式）

| 行 | 学什么 |
|----|--------|
| 56-70 | `on_snapshot` 回调：WebSocket 每收到一个推送就触发 |
| 73-79 | 退出刷盘：`shutdown_handler` 中先 flush 再退出 |

**`data_infra/scripts/collect_market.py`**（多指标采集）

| 行 | 学什么 |
|----|--------|
| 78-131 | 单轮四个指标的采集：每个指标独立 try/except，互不影响 |

### 9.3 工具脚本

**`data_infra/scripts/backfill.py`** — 历史数据回填

| 行 | 学什么 |
|----|--------|
| 69-81 | K线批量回填：`fetch_ohlcv_batch` 自动分页 |
| 114-139 | Tick 逐批回填：直接调用 `fetch_trades`（单批）→ 立即写入 → 更新 current_id |

**`data_infra/scripts/check_data.py`** — 数据质量巡检

| 行 | 学什么 |
|----|--------|
| 60-107 | K线巡检 + `--fix` 自动回填缺口 |
| 121-166 | Tick 同步状态对比：获取交易所最新 trade_id 与本地比较 |
| 232-281 | 跨源 K线 对比：随机抽样与 API 比对 |

---

## 十、第 8 站：测试 tests/

### 10.1 测试文件与被测模块对照

| 测试文件 | 被测模块 | 关键测试技巧 |
|----------|----------|-------------|
| test_config.py | config/settings.py | 直接断言配置值 |
| test_time_utils.py | utils/time_utils.py | 纯函数测试，无需 mock |
| test_validator.py | data/validator.py | 构造边界数据验证校验规则 |
| test_kline_store.py | data/kline_store.py | `tempfile` 临时数据库 |
| test_tick_store.py | data/tick_store.py | `tempfile` 临时目录 |
| test_orderbook_store.py | data/orderbook_store.py | 缓冲→刷盘→读取完整流程 |
| test_market_store.py | data/market_store.py | 四张表的写入/读取/幂等 |
| test_aggregator.py | data/aggregator.py | 验证 OHLCV 聚合逻辑正确性 |
| test_reader.py | data/reader.py | 验证路由逻辑（1m/5m/10s） |
| **test_fetcher.py** | data/fetcher.py | **mock ccxt.binance** |
| **test_tick_fetcher.py** | data/tick_fetcher.py | **mock async ccxt + AsyncMock** |
| **test_orderbook_fetcher.py** | data/orderbook_fetcher.py | 测 _process_message 解析 |
| **test_market_fetcher.py** | data/market_fetcher.py | **mock 合约 API** |

### 10.2 核心 mock 技巧（重点学习）

**技巧 1：patch + MagicMock 替换外部依赖**

```python
# data_infra/tests/test_fetcher.py
@pytest.fixture
def fetcher():
    with patch("data_infra.data.fetcher.ccxt") as mock_ccxt:
        mock_exchange = MagicMock()
        mock_ccxt.binance.return_value = mock_exchange
        f = KlineFetcher()
        f.exchange = mock_exchange        # 替换真实交易所实例
        yield f
```

要点：
- `patch("data_infra.data.fetcher.ccxt")` 替换的是 **fetcher 模块中导入的 ccxt**，不是全局的
- `mock_exchange.fetch_ohlcv.return_value = [...]` 预设返回值
- `mock_exchange.fetch_ohlcv.side_effect = [page1, page2]` 预设多次调用的不同返回

**技巧 2：AsyncMock 测试异步函数**

```python
# data_infra/tests/test_tick_fetcher.py
mock_exchange = MagicMock()
mock_exchange.fetch_trades = AsyncMock()    # 异步方法用 AsyncMock
mock_exchange.close = AsyncMock()

# 运行异步测试
df = asyncio.run(fetcher.fetch_trades("BTC/USDT"))
```

**技巧 3：tempfile 隔离测试数据**

```python
# data_infra/tests/test_kline_store.py
@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test_kline.db")
    return KlineStore(db_path)
```

每个测试用例都在独立的临时目录中运行，互不干扰。

### 10.3 运行测试

```bash
# 运行全部测试
python -m pytest data_infra/tests/ -v

# 只运行某个文件
python -m pytest data_infra/tests/test_fetcher.py -v

# 只运行某个类/方法
python -m pytest data_infra/tests/test_fetcher.py::TestFetchOhlcv::test_returns_correct_columns -v

# 显示覆盖率
python -m pytest data_infra/tests/ --cov=data_infra.data --cov=data_infra.utils --cov-report=term-missing
```

---

## 十一、核心设计模式总结

### 模式 1：Fetcher-Store 分离

```
Fetcher（拉取）→ Writer（校验+路由）→ Store（存储）
     ↑ 不关心存储               ↑ 不关心拉取
```
好处：Fetcher 和 Store 可以独立测试和替换。

### 模式 2：Writer/Reader 门面

```
采集脚本 → DataWriter → validator → Store
上层模块 → DataReader → Store / Aggregator
```
调用方不需要知道数据存在 SQLite 还是 Parquet。

### 模式 3：trade_id 追赶

```
查询本地最新 trade_id → 从该 ID 之后拉取 → 满额继续/不满额等待
```
保证零遗漏，不依赖时间戳精度。

### 模式 4：原子写入

```
写 .tmp 文件 → os.replace → 正式文件
```
防止写入过程崩溃导致数据损坏。

### 模式 5：信号优雅退出

```python
running = True
def handler(sig, frame): running = False
signal.signal(signal.SIGINT, handler)

while running:
    ...
    for _ in range(60):    # 可中断的 sleep
        if not running: break
        time.sleep(1)
```

### 模式 6：幂等写入

```sql
INSERT OR IGNORE INTO klines ... VALUES ...
-- 主键冲突时跳过，不报错
```
重复采集同一数据不会产生脏数据。

---

## 十二、数据流全景图

```
                          Binance Exchange
                     ┌──────────┼──────────┐
                     │          │          │
              REST API    REST API    WebSocket
              (同步)      (异步)      (异步)
                     │          │          │
              ┌──────┴──┐ ┌────┴────┐ ┌───┴────────┐
              │ KlineFetcher │ TickFetcher │ OrderbookFetcher │  MarketFetcher
              └──────┬──┘ └────┬────┘ └───┬────────┘  └──────┬──┘
                     │          │          │                   │
                     ▼          ▼          ▼                   ▼
              ┌─────────────────────────────────────────────────────┐
              │                    DataWriter                       │
              │    validator.validate_xxx() → Store.write()         │
              └──────┬──────┬──────┬──────┬────────────────────────┘
                     │      │      │      │
              ┌──────┴──┐ ┌┴────┐ ┌┴────┐ ┌┴──────┐
              │KlineStore│ │TickStore│ │OBStore│ │MarketStore│
              │ SQLite   │ │Parquet │ │Parquet│ │ SQLite   │
              └──────┬──┘ └┬────┘ └┬────┘ └┬──────┘
                     │      │      │      │
              ┌──────┴──────┴──────┴──────┴────────────────────────┐
              │                    DataReader                       │
              │  路由 + 聚合 (aggregator.resample / aggregate)      │
              └────────────────────────────────────────────────────┘
                                   │
                                   ▼
                        上层模块（因子 / 策略 / 回测）
```

---

## 十三、延伸练习

完成以下练习，可以加深对代码的理解：

### 练习 1：配置变更（简单）
在 `data_infra/config/settings.py` 中添加一个新交易对 `XRP/USDT`。思考：你需要改几个文件？

### 练习 2：手动测试 Store（简单）
在 Python REPL 中手动创建一个临时 KlineStore，写入数据，读出来验证。

### 练习 3：阅读测试理解行为（中等）
通读 `data_infra/tests/test_tick_store.py`，仅通过测试用例的名字和断言，推断 TickStore 的行为规范。然后对照源码验证你的推断。

### 练习 4：追踪一次完整采集（中等）
从 `data_infra/scripts/collect_klines.py` 的 `main()` 开始，逐行追踪一次完整的 K线 采集流程：
`main → parse_args → KlineFetcher.fetch_ohlcv → DataWriter.write_ohlcv → validator.validate_ohlcv → KlineStore.write`

画出函数调用链，标注每个环节的输入和输出类型。

### 练习 5：写一个新的校验规则（中等）
在 `data_infra/data/validator.py` 中为 `validate_ohlcv` 添加一条新规则：volume 不应超过 close × 1,000,000（防止异常大量）。然后在 `data_infra/tests/test_validator.py` 中添加对应的测试。

### 练习 6：理解 mock 模式（进阶）
阅读 `data_infra/tests/test_tick_fetcher.py`，回答：
- 为什么 `fetcher.exchange.fetch_trades` 要用 `AsyncMock` 而不是 `MagicMock`？
- `side_effect = [batch1, batch2]` 是什么意思？
- 测试中的 `asyncio.run()` 起什么作用？

---

**祝学习顺利！建议搭配 IDE 的"跳转到定义"功能阅读，可以快速在模块间跳转。**

---
---

# Phase 2a 因子研究框架 — 代码学习指引

本文档是 `factor_research/` 模块的系统学习路径。
建议按**自底向上**的顺序阅读：先理解类型系统，再看计算原语，最后看一键报告。

---

## 目录

- [一、模块整体结构](#一模块整体结构)
- [二、建议学习路线](#二建议学习路线-1)
- [三、第 1 站：类型系统 core/types.py](#三第-1-站类型系统-coretypespy)
- [四、第 2 站：因子基类 core/base.py](#四第-2-站因子基类-corebasepy)
- [五、第 3 站：因子注册表 core/registry.py](#五第-3-站因子注册表-coreregistrypy)
- [六、第 4 站：因子存储 store/](#六第-4-站因子存储-store)
- [七、第 5 站：评价原语 evaluation/metrics.py](#七第-5-站评价原语-evaluationmetricspy)
- [八、第 6 站：评价分析模块 evaluation/](#八第-6-站评价分析模块-evaluation)
- [九、第 7 站：FactorAnalyzer 一键报告](#九第-7-站factoranalyzer-一键报告)
- [十、第 8 站：数据对齐 alignment/](#十第-8-站数据对齐-alignment)
- [十一、第 9 站：因子计算引擎 core/engine.py](#十一第-9-站因子计算引擎-coreenginepy)
- [十二、第 10 站：示例因子 factors/](#十二第-10-站示例因子-factors)
- [十三、第 11 站：测试 tests/](#十三第-11-站测试-tests)
- [十四、核心概念深入](#十四核心概念深入)
- [十五、设计模式总结](#十五设计模式总结)
- [十六、数据流全景图](#十六数据流全景图)
- [十七、延伸练习](#十七延伸练习)

---

## 一、模块整体结构

```
factor_research/
├── config.py                   # 集中配置（对 data_infra 的唯一引用入口）
├── core/                       # 核心层（类型、基类、引擎）
│   ├── types.py                #   FactorType, DataType, DataRequest, FactorMeta
│   ├── base.py                 #   Factor ABC, TimeSeriesFactor, CrossSectional, CrossAsset
│   ├── registry.py             #   FactorRegistry + @register_factor / @register_factor_family 装饰器
│   └── engine.py               #   FactorEngine（数据读取→计算→存储）
│
├── evaluation/                 # 因子评价体系（三层 API）
│   ├── metrics.py              #   第三层: 底层计算原语（纯函数）
│   ├── ic.py                   #   第二层: IC/IR/衰减分析
│   ├── quantile.py             #   第二层: 分层回测
│   ├── tail.py                 #   第二层: 尾部特征分析
│   ├── stability.py            #   第二层: 稳定性分析
│   ├── nonlinear.py            #   第二层: 非线性分析
│   ├── turnover.py             #   第二层: 换手率分析
│   ├── correlation.py          #   第二层: 因子相关性 + VIF
│   ├── report.py               #   文本报告 + matplotlib 可视化
│   ├── analyzer.py             #   第一层: FactorAnalyzer 一键报告
│   └── family_analyzer.py      #   因子族参数分析器（FamilyAnalyzer）
│
├── alignment/                  # 异步数据对齐
│   ├── grid.py                 #   网格对齐（等间距 + ffill）
│   ├── refresh_time.py         #   刷新时间采样
│   └── hayashi_yoshida.py      #   Hayashi-Yoshida 异步协方差
│
├── store/                      # 因子持久化存储
│   ├── factor_store.py         #   FactorStore（Parquet + JSON）
│   └── catalog.py              #   FactorCatalog（因子目录检索）
│
├── factors/                    # 具体因子实现
│   ├── microstructure/
│   │   └── imbalance.py        #   订单簿不平衡度
│   ├── momentum/
│   │   └── returns.py          #   多尺度收益率因子族（@register_factor_family）
│   ├── volatility/             #   （预留）
│   ├── orderflow/              #   （预留）
│   └── cross_asset/            #   （预留）
│
└── tests/                      # 单元测试（198 项）
```

**依赖方向（只能向下依赖，不能向上）：**

```
factors/   → core/ (base, registry, types)
engine.py  → core/ + store/ + factor_research/config
evaluation/analyzer → evaluation/* → evaluation/metrics
evaluation/family_analyzer → evaluation/* + core/base
alignment/ → numpy, pandas (独立于 core)
store/     → core/types + factor_research/config
config.py  → data_infra/config (唯一引用入口)
```

---

## 二、建议学习路线

```
第 1 站  core/types.py          ← 所有类型定义，理解"通用语言"
  ↓
第 2 站  core/base.py           ← 三种因子基类，理解继承设计
  ↓
第 3 站  core/registry.py       ← 注册表 + 装饰器，理解"定义即注册"
  ↓
第 4 站  store/factor_store.py  ← 因子持久化，理解存储接口
  ↓
第 5 站  evaluation/metrics.py  ← 底层原语（纯函数），理解因子评价基础
  ↓
第 6 站  evaluation/ic.py, quantile.py, etc. ← 第二层分析模块
  ↓
第 7 站  evaluation/analyzer.py ← 第一层一键报告
  ↓
第 8 站  alignment/             ← 异步对齐（高频数据处理核心）
  ↓
第 9 站  core/engine.py         ← 引擎（数据→计算→存储的编排）
  ↓
第10 站  factors/               ← 示例因子，学习如何编写新因子
  ↓
第11 站  tests/                 ← 测试，验证理解
```

---

## 三、第 1 站：类型系统 core/types.py

### 核心知识点

**Python 知识:**
- `Enum` 枚举类: 限制取值范围，防止字符串拼错
- `@dataclass`: 自动生成 `__init__`, `__repr__`, `__eq__`
- `field(default_factory=dict)`: dataclass 中可变默认值的正确写法

**量化知识:**
- 三种因子类型的区别（时序 vs 截面 vs 跨标的）
- 因子面板格式约定（timestamp × symbol）
- 数据需求声明模式（因子声明需要什么数据，引擎负责准备）

### 阅读重点

| 行范围 | 内容 | 学什么 |
|--------|------|--------|
| 1-27 | 模块文档字符串 | 面板格式约定和依赖关系 |
| 33-56 | `FactorType` | 三种因子类型的含义和引擎行为差异 |
| 59-82 | `DataType` | 7 种数据类型与 DataReader 方法的映射 |
| 85-119 | `DataRequest` | 因子如何声明数据需求（timeframe, lookback_bars, symbols） |
| 122-170 | `FactorMeta` | 因子元数据的完整字段说明 |

### 动手练习

```python
from factor_research.core.types import *

# 创建一个数据请求
req = DataRequest(DataType.OHLCV, timeframe="5m", lookback_bars=60)
print(req)               # DataRequest(data_type=<DataType.OHLCV: 'ohlcv'>, ...)
print(req.data_type)     # DataType.OHLCV
print(req.timeframe)     # '5m'

# 创建因子元数据
meta = FactorMeta(
    name="test",
    display_name="测试因子",
    factor_type=FactorType.TIME_SERIES,
    category="test",
    description="测试用",
    data_requirements=[req],
    output_freq="5m",
    params={"window": 60},
)
print(meta.name)         # 'test'
print(meta.params)       # {'window': 60}
```

### 思考题

1. 为什么 `DataRequest.symbols` 默认为 `None` 而不是空列表？（答: None 表示"使用全局配置"，空列表表示"不需要数据"，语义不同）
2. `FactorMeta.params` 为什么用 `field(default_factory=dict)` 而不是 `params: dict = {}`？（答: 可变默认值会在所有实例间共享，导致意外修改）

---

## 四、第 2 站：因子基类 core/base.py

### 核心知识点

**Python 知识:**
- `ABC` + `@abstractmethod`: 抽象基类，强制子类实现特定方法
- 模板方法模式: `TimeSeriesFactor.compute()` 提供固定流程，子类只需实现 `compute_single()`
- `@property` + `@abstractmethod`: 抽象属性

**量化知识:**
- 时序因子的"各标的独立计算"原则
- 截面因子需要同时看到所有标的数据
- 跨标的因子的 input/output 分离设计

### 阅读重点

| 行范围 | 内容 | 学什么 |
|--------|------|--------|
| 1-31 | 模块文档字符串 | 三种因子类型的数据输入格式差异 |
| 41-81 | `Factor` ABC | 基类的两个抽象方法: meta() 和 compute() |
| 84-161 | `TimeSeriesFactor` | **模板方法模式**——compute() 遍历 symbols 调用 compute_single() |
| 124-142 | `TimeSeriesFactor.compute()` | 合并逻辑: `{symbol: Series}` → `pd.DataFrame` |
| 163-215 | `CrossSectionalFactor` | 为什么是 `pass`——直接用 Factor 的 compute() |
| 218-290 | `CrossAssetFactor` | input_symbols / output_symbols 的抽象属性 |

### 重点理解: 模板方法模式

```python
# TimeSeriesFactor 的 compute() 是"模板方法"
# 它定义了固定流程: 遍历 → compute_single → 合并
# 子类只需实现 compute_single，无需关心多标的合并

class TimeSeriesFactor(Factor):
    def compute(self, data):           # 固定流程（模板）
        panels = {}
        for symbol, symbol_data in data.items():
            series = self.compute_single(symbol, symbol_data)  # 子类实现
            panels[symbol] = series
        return pd.DataFrame(panels)    # 自动合并

    @abstractmethod
    def compute_single(self, symbol, data):  # 子类实现
        ...
```

这个模式的好处:
- 子类只关心"单标的怎么算"，不需要处理多标的遍历和合并
- 框架保证所有时序因子的输出格式一致
- 引擎可以统一调度，不需要区分具体因子

### 思考题

1. 为什么 `CrossSectionalFactor` 不需要 `compute_single`？（答: 截面因子必须同时看到所有标的才能排名，无法逐标的计算）
2. 如果一个因子同时需要 BTC 和 ETH 的数据来计算 BTC 的因子值，它应该是哪种类型？（答: CrossSectionalFactor 或 CrossAssetFactor，不能是 TimeSeriesFactor）

---

## 五、第 3 站：因子注册表 core/registry.py

### 核心知识点

**Python 知识:**
- 类装饰器: `@register_factor` 装饰类而非函数
- 类变量作为全局状态: `_registry: dict = {}` 在类级别共享
- `__contains__`（in 操作符）和 `__len__`（len() 函数）的自定义

**设计模式:**
- 注册表模式（Registry Pattern）: 全局注册 → 按名查找
- "定义即注册": 装饰器在类定义时自动触发注册
- 单例 vs 全局字典: 使用类变量实现全局共享，比 Singleton 更简单

### 阅读重点

| 行范围 | 内容 | 学什么 |
|--------|------|--------|
| `FactorRegistry` 类 | 注册表核心实现 | `_registry` 类变量, register/get/list_all |
| `register()` | 注册逻辑 | 实例化因子类 → 获取 meta → 检查重名 → 存入字典 |
| `register_factor()` | 装饰器函数 | 类装饰器的写法——接受类，返回类 |

### 重点理解: 类装饰器

```python
# @register_factor 做了什么？

def register_factor(cls):
    """类装饰器: 在类定义时自动注册"""
    registry = FactorRegistry()
    registry.register(cls)     # 实例化 cls() 并注册
    return cls                 # 返回原类（不修改类本身）

# 使用:
@register_factor
class MyFactor(TimeSeriesFactor):
    ...

# 等价于:
class MyFactor(TimeSeriesFactor):
    ...
MyFactor = register_factor(MyFactor)

# 之后可以通过名字找到它:
registry = FactorRegistry()
factor = registry.get("my_factor_name")
```

### 思考题

1. 为什么 `register_factor` 返回 `cls` 本身？（答: 装饰器必须返回类，否则 `MyFactor` 变量会变成 None）
2. 如果两个不同的类注册了相同的 `name`，会发生什么？（答: 抛出 ValueError，防止名称冲突）
3. 为什么允许同一个类重复注册？（答: 模块被多次 import 时不应报错）

---

## 六、第 4 站：因子存储 store/

### 6.1 factor_store.py — FactorStore

**核心知识点:**

- Parquet + JSON 双文件存储: 面板数据用 Parquet（高效列式），元数据用 JSON（人类可读）
- 原子写入: `.tmp` → `os.replace`（继承 Phase 1 的设计）
- 枚举序列化: `FactorType.TIME_SERIES` → `"time_series"` → `FactorType("time_series")`
- FactorStore 是 Phase 2a（因子研究）与 Phase 2b（模型策略）之间的**唯一接口**

**阅读重点:**

| 行范围 | 内容 | 学什么 |
|--------|------|--------|
| 1-25 | 模块文档字符串 | 存储路径格式、原子写入策略 |
| 81-133 | `save()` | reset_index → to_parquet → os.replace 的完整流程 |
| 135-163 | `load()` | read_parquet → set_index → UTC 时间转换 |
| 246-265 | `_meta_to_dict()` | Enum → str、DataRequest → dict 的序列化 |
| 267-286 | `_dict_to_meta()` | str → Enum、dict → DataRequest 的反序列化 |

### 6.2 catalog.py — FactorCatalog

因子目录扫描与检索，类似"因子超市的目录"。扫描 FactorStore 目录下所有 meta.json，提供搜索和摘要功能。

### 动手练习

```python
from factor_research.store.factor_store import FactorStore
import pandas as pd, numpy as np, tempfile

# 使用临时目录
store = FactorStore(base_dir=tempfile.mkdtemp())

# 创建测试面板
from factor_research.core.types import *
index = pd.date_range("2024-01-01", periods=10, freq="1min", tz="UTC")
panel = pd.DataFrame({"BTC": np.random.randn(10), "ETH": np.random.randn(10)}, index=index)
meta = FactorMeta(
    name="test", display_name="Test", factor_type=FactorType.TIME_SERIES,
    category="test", description="test", data_requirements=[], output_freq="1m",
)

store.save("test", panel, meta)
print(store.list_factors())     # ['test']
print(store.exists("test"))     # True

loaded = store.load("test")
print(loaded.shape)             # (10, 2)
print(loaded.index.tz)          # UTC

loaded_meta = store.load_meta("test")
print(loaded_meta.factor_type)  # FactorType.TIME_SERIES
```

---

## 七、第 5 站：评价原语 evaluation/metrics.py

### 核心知识点

**量化知识（重点）:**

1. **Spearman IC (Information Coefficient)**
   - 因子评价最核心的指标
   - IC = Spearman_corr(因子值, 前瞻收益)
   - 用秩相关而非 Pearson: 对异常值稳健，只要求单调关系
   - |IC| > 0.03 即有价值（量化界共识）

2. **前瞻收益 (Forward Returns)**
   - `forward_return_t = price_{t+h} / price_t - 1`
   - 用简单收益而非对数收益（截面可加性）
   - h = 1,5,10,30,60 bars 多 horizon 评价

3. **排名归一化 (Rank Normalize)**
   - 因子值 → [0, 1] 百分位排名
   - 消除量纲差异，便于跨因子比较

4. **互信息 (Mutual Information)**
   - 衡量非线性相关性
   - MI > 0 说明因子包含收益的信息（不限于线性）

**Python 知识:**
- `scipy.stats.spearmanr`: Spearman 秩相关
- `sklearn.metrics.mutual_info_score`: 互信息
- 纯函数设计: 无状态、无副作用、输入输出都是 numpy/pandas

### 阅读重点

| 行范围 | 内容 | 学什么 |
|--------|------|--------|
| 34-70 | `spearman_ic()` | Spearman 相关的计算 + NaN 处理 + 最小样本数要求 |
| 72-100 | `pearson_ic()` | 与 Spearman 的对比 |
| 103-140 | `compute_forward_returns()` | 前瞻收益计算 + shift 的方向 |
| 142-170 | `compute_forward_returns_panel()` | 面板版: 逐列调用 |
| 172-195 | `rank_normalize()` | rank(pct=True) 的含义 |
| 197-220 | `cross_sectional_rank()` | rank(axis=1) 截面排名 |
| 250-280 | `mutual_information()` | sklearn 互信息 + 离散化处理 |
| 282-330 | `cumulative_returns()`, `sharpe_ratio()`, `max_drawdown()` | 组合绩效指标（注意: max_drawdown 返回负数，sharpe_ratio 零波动率返回 ±inf） |

### 动手练习

```python
from factor_research.evaluation.metrics import *
import pandas as pd, numpy as np

# Spearman IC
factor = pd.Series([1, 2, 3, 4, 5])
returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
print(f"IC = {spearman_ic(factor, returns):.4f}")   # 1.0 (完美正相关)

# 反转信号
returns_neg = pd.Series([0.05, 0.04, 0.03, 0.02, 0.01])
print(f"IC = {spearman_ic(factor, returns_neg):.4f}")  # -1.0

# 前瞻收益
prices = pd.Series([100.0, 102.0, 101.0, 105.0, 103.0])
fwd = compute_forward_returns(prices, 1)
print(fwd)  # [0.02, -0.0098, 0.0396, -0.019, NaN]

# 排名归一化
raw = pd.Series([10, 30, 20, 50, 40])
print(rank_normalize(raw))  # [0.2, 0.6, 0.4, 1.0, 0.8]

# 夏普比
daily_ret = pd.Series(np.random.randn(252) * 0.01 + 0.0005)
print(f"Sharpe = {sharpe_ratio(daily_ret):.2f}")
```

### 思考题

1. 为什么 `spearman_ic` 要求至少 3 个有效数据点？（答: Spearman 相关在 n<3 时无统计意义）
2. `compute_forward_returns` 最后一行为什么是 NaN？（答: 最后一个时刻没有 t+h 的价格数据）

---

## 八、第 6 站：评价分析模块 evaluation/

这是评价体系的第二层，每个模块封装一个分析维度。

### 8.1 ic.py — IC/IR/衰减分析

**核心概念:**

- **IC 序列**: 每个时刻计算截面 IC → 得到 IC 的时间序列
- **IC_IR**: IC_mean / IC_std，衡量 IC 的稳定性（类比夏普比）
- **IC 衰减**: IC 在不同 horizon 下的变化，揭示因子预测力的持续时间
- **IC 胜率**: IC > 0 的比例，50% 以上说明因子方向正确

| 行范围 | 内容 | 学什么 |
|--------|------|--------|
| 33-65 | `ic_series()` | 逐行截面 IC 计算——for 循环逐时刻 |
| 68-108 | `ic_summary()` | IC 统计汇总——均值/标准差/IR/胜率/偏度/自相关 |
| 111-144 | `ic_decay()` | 多 horizon 循环 → 衰减 DataFrame |
| 147-188 | `ic_analysis()` | 整合入口——返回 series + summary + decay |

### 8.2 quantile.py — 分层回测

**核心概念:**

- 将因子值从高到低排序，分成 N 组
- 每组计算平均收益，看是否单调递增/递减
- **多空收益**: 多头（最高组）减空头（最低组）的收益
- **单调性**: 组收益是否严格递增/递减（Spearman 相关度量）

### 8.3 tail.py — 尾部特征分析

**核心概念:**

- 因子极端值（>90% 分位）的预测力是否更强？
- **条件 IC**: 只在尾部样本上计算 IC
- **尾部命中率**: 极端因子值后收益方向正确的比例
- 好因子的尾部表现通常优于整体

### 8.4 stability.py — 稳定性分析

**核心概念:**

- **分 regime IC**: 趋势行情 vs 震荡行情、高波动 vs 低波动
- **月度 IC**: 按月分解，检查是否有系统性失效月份
- **滚动 IC**: 观察 IC 的时间演变趋势
- **IC 最大回撤**: IC 最差的连续时期有多严重

| 行范围 | 内容 | 学什么 |
|--------|------|--------|
| 58-111 | `_regime_ic()` | 滚动均值/标准差 → 中位数分界 → 分组 IC |
| 114-134 | `_monthly_ic()` | `groupby(to_period("M"))` 的用法 |
| 137-151 | `_rolling_ic()` | IC 的滚动均值 |
| 154-177 | `_ic_max_drawdown()` | IC 累计曲线 → cummax → drawdown |

### 8.5 nonlinear.py — 非线性分析

**核心概念:**

- **互信息 (MI)**: 捕捉线性和非线性关系
- **因子 Profile**: 因子值分 bin → 每 bin 的平均收益
- **条件 IC**: 低/中/高因子值区间分别计算 IC
- 如果条件 IC 差异大 → 因子有非线性效应

### 8.6 turnover.py — 换手率分析

**核心概念:**

- **自相关**: 因子值的 lag-1 自相关，高自相关 → 低换手
- **排名变化率**: 相邻时刻标的排名变化的平均幅度
- **信号翻转率**: 因子值正负翻转的频率
- 高换手 → 高交易成本 → 需要更强的因子信号才能盈利

### 8.7 correlation.py — 因子相关性

**核心概念:**

- **相关矩阵**: 多个因子之间的 Pearson 相关
- **VIF (方差膨胀因子)**: 检测多重共线性
- **增量 IC**: 控制已有因子后，新因子的边际贡献
- VIF > 5 → 高共线性，因子冗余

---

## 九、第 7 站：FactorAnalyzer 一键报告

### 核心知识点

- **Facade 模式**: 封装所有第二层模块，提供统一入口
- **三层 API 设计**: Analyzer → 独立函数 → 纯函数原语
- **缓存**: `_report_cache` 避免重复计算
- **文本报告**: format_report_text 将结构化数据转为可读文本

### 阅读重点

| 行范围 | 内容 | 学什么 |
|--------|------|--------|
| 51-97 | `__init__()` | 输入校验 + 共同标的对齐 |
| 99-141 | `full_report()` | 一键调用所有维度 |
| 143-169 | 各独立方法 | 委托给第二层模块 |
| 171-199 | `summary_text()`, `plot()` | 缓存 + 延迟计算 |

### 动手练习

```python
import numpy as np, pandas as pd
from factor_research.evaluation.analyzer import FactorAnalyzer

# 构造合成数据
np.random.seed(42)
n = 500
index = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")

# 因子值
factor_panel = pd.DataFrame({
    "BTC/USDT": np.random.randn(n),
    "ETH/USDT": np.random.randn(n),
}, index=index)

# 价格
price_panel = pd.DataFrame({
    "BTC/USDT": 100 + np.cumsum(np.random.randn(n) * 0.1),
    "ETH/USDT": 50 + np.cumsum(np.random.randn(n) * 0.05),
}, index=index)

# 一键评价
analyzer = FactorAnalyzer(factor_panel, price_panel)
report = analyzer.full_report(horizons=[1, 5, 10])
print(analyzer.summary_text(factor_name="random_factor"))

# 单独查看 IC
ic = analyzer.ic_analysis(horizons=[1, 5, 10])
print(ic["ic_decay"])
```

---

## 十、第 8 站：数据对齐 alignment/

### 核心问题

加密货币交易数据的到达时间是异步的:
- BTC/USDT 的 tick 可能在 10:00:00.123 到达
- ETH/USDT 的 tick 可能在 10:00:00.456 到达
- 10 档订单簿快照每 100ms 推送一次

直接合并这些不规则数据会引入偏差。对齐模块解决这个问题。

### 10.1 grid.py — 网格对齐

**核心思想**: 将不规则时间序列映射到等间距网格上（如 1 秒），空值用前一个值填充（forward fill）。

**关键参数:**
- `freq`: 网格间距（如 "1s", "100ms"）
- `max_gap`: 最大填充间隔，超过此间隔填 NaN（防止用过期数据）

| 行范围 | 内容 | 学什么 |
|--------|------|--------|
| grid_align() | 核心函数 | resample → ffill → max_gap 截断 |

```python
from factor_research.alignment.grid import grid_align
import pandas as pd

btc = pd.Series([100, 101, 102],
    index=pd.to_datetime(["10:00:00", "10:00:01", "10:00:03"], utc=True))
eth = pd.Series([50, 51],
    index=pd.to_datetime(["10:00:00", "10:00:02"], utc=True))

panel = grid_align({"BTC": btc, "ETH": eth}, freq="1s")
print(panel)
# BTC: 100, 101, 101(ffill), 102
# ETH: 50, 50(ffill), 51, 51(ffill)
```

### 10.2 refresh_time.py — 刷新时间采样

**核心思想**: 只在"所有标的都至少更新一次"的时刻取样。避免用过期数据。

**与 grid 的区别:**
- grid: 固定间距，可能用过期数据（ffill）
- refresh_time: 不固定间距，但保证每个值都"新鲜"

### 10.3 hayashi_yoshida.py — Hayashi-Yoshida 协方差估计

**核心思想**: 在两个异步时间序列之间直接计算协方差/相关性，无需先对齐。

**传统方法的问题:**
- 先 grid_align 再算相关 → 信息损失（ffill 引入自相关）
- Hayashi-Yoshida 利用所有重叠区间，无信息损失

**算法原理:**
```
HY_Cov(X, Y) = Σ ΔX_i × ΔY_j    (对所有时间区间重叠的 i, j)

时间区间重叠条件:
    X 的第 i 个区间 [t_{i-1}, t_i]
    Y 的第 j 个区间 [s_{j-1}, s_j]
    重叠 ⟺ t_{i-1} < s_j 且 s_{j-1} < t_i
```

**适用场景:**
- 高频数据的相关性估计
- 多标的的协方差矩阵构建
- 作为风险模型的输入

---

## 十一、第 9 站：因子计算引擎 core/engine.py

### 核心知识点

**设计模式:**
- **编排者 (Orchestrator)**: 引擎不做计算，只编排流程
- **数据路由**: 根据 FactorType 决定如何组织数据传给因子

**引擎流程:**
```
1. 从注册表获取因子实例
2. 读取 FactorMeta 获取数据需求
3. 根据因子类型准备数据（三种格式）
4. 调用 factor.compute(data)
5. 可选: 保存到 FactorStore
```

### 阅读重点

| 行范围 | 内容 | 学什么 |
|--------|------|--------|
| 48-56 | `_DATA_TYPE_TO_READER_METHOD` | DataType → DataReader 方法名的映射字典 |
| 86-143 | `compute_factor()` | 完整流程: 获取→准备→计算→保存 |
| 145-188 | `compute_all()` | 批量计算 + 异常捕获 + 统计 |
| 190-223 | `_prepare_data()` | **关键**: 根据因子类型分发到不同的数据组织方法 |
| 225-240 | `_prepare_timeseries_data()` | 时序因子: {symbol: {DataType: df}} |
| 242-257 | `_prepare_cross_sectional_data()` | 截面因子: {DataType: {symbol: df}} |
| 259-269 | `_prepare_cross_asset_data()` | 跨标的因子: 只读 input_symbols |
| 271-295 | `_read_data()` | DataRequest → DataReader 方法调用的映射 |

### 重点理解: 数据组织格式

三种因子类型的数据参数格式不同，这是引擎的核心职责:

```python
# 时序因子: {symbol: {DataType: DataFrame}}
{
    "BTC/USDT": {DataType.OHLCV: btc_ohlcv_df},
    "ETH/USDT": {DataType.OHLCV: eth_ohlcv_df},
}

# 截面/跨标的因子: {DataType: {symbol: DataFrame}}
{
    DataType.OHLCV: {
        "BTC/USDT": btc_ohlcv_df,
        "ETH/USDT": eth_ohlcv_df,
    }
}
```

为什么不同？
- 时序因子的 `compute_single(symbol, data)` 只需要单标的的数据
- 截面因子的 `compute(data)` 需要同时看到所有标的

---

## 十二、第 10 站：示例因子 factors/

### 12.1 imbalance.py — 订单簿不平衡度

**经济含义:**
- 买方挂单量 > 卖方 → imbalance > 0 → 短期上行压力
- 这是微观结构中最基础也最有效的因子

**阅读重点:**

| 行范围 | 内容 | 学什么 |
|--------|------|--------|
| 1-27 | 模块文档 | 经济含义、计算逻辑、范围 |
| 42-64 | `meta()` | 声明需要 10 档订单簿，输出 1s 频率 |
| 66-112 | `compute_single()` | bid/ask 列名构造 → 求和 → 不平衡度 → 降采样 |

**学习要点:**
- 动态构造列名: `[f"bid_qty_{i}" for i in range(10)]`
- 防除零: `total.replace(0, float("nan"))`
- 降采样: `.resample("1s").mean()` 从 100ms → 1s

### 12.2 returns.py — 多尺度收益率因子家族

**经济含义:**
- 短期收益率（5-10 bar）→ 可能有反转效应（均值回复）
- 中期收益率（30-60 bar）→ 可能有动量效应（趋势延续）

**学习要点: 参数化因子家族**

```python
@register_factor_family
class MultiScaleReturns(TimeSeriesFactor):
    """多尺度收益率因子"""

    _param_grid = {"lookback": [5, 10, 30, 60]}  # 声明参数网格

    def __init__(self, lookback: int = 5):
        self.lookback = lookback

    def meta(self):
        return FactorMeta(
            name=f"returns_{self.lookback}m",
            family="multi_scale_returns", ...)

    def compute_single(self, symbol, data):
        close = data[DataType.OHLCV]["close"]
        return (close / close.shift(self.lookback) - 1).dropna()
# → 装饰器自动注册 4 个因子变体: returns_5m, returns_10m, returns_30m, returns_60m
```

这个模式的好处:
- 一套计算逻辑 + 一行参数声明 → 自动生成 N 个变体
- 自动注册到全局注册表，支持 FamilyAnalyzer 批量扫参
- 添加新窗口只需在 `_param_grid` 列表中追加一个值

---

## 十三、第 11 站：测试 tests/

### 测试文件与被测模块对照

| 测试文件 | 被测模块 | 关键测试技巧 |
|----------|----------|-------------|
| test_types.py | core/types.py | 枚举值、dataclass 默认值 |
| test_base.py | core/base.py | Dummy 实现 + 面板验证 |
| test_registry.py | core/registry.py | 注册/获取/重名冲突/装饰器 |
| test_factor_store.py | store/factor_store.py | tmp_path + 存储/加载/序列化 |
| test_evaluation.py | evaluation/* | 合成数据 + 统计性质验证 |
| test_alignment.py | alignment/* | 小规模手算验证 |
| test_engine.py | core/engine.py | MockDataReader + 三种因子类型端到端 |
| test_catalog.py | store/catalog.py | tmp_path + 空目录/搜索/损坏 JSON |
| test_factors.py | factors/* | 具体因子 meta/compute/已知答案 |
| test_family_analyzer.py | evaluation/family_analyzer.py | 合成数据 + Dummy 参数化因子 + sweep/plot |

### 测试技巧

**技巧 1: 合成数据构造**

```python
@pytest.fixture
def factor_panel():
    """构造一个与收益正相关的因子面板"""
    np.random.seed(42)               # 固定随机种子 → 可重复
    n = 200
    index = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {"BTC/USDT": np.random.randn(n), "ETH/USDT": np.random.randn(n)},
        index=index,
    )
```

**技巧 2: 用数学性质验证**

```python
def test_correlation_range(self):
    """相关系数应在 [-1, 1] 之间"""
    corr = hy_correlation(x, y)
    assert -1 <= corr <= 1           # 不检查具体值，检查数学约束

def test_correlation_perfect(self):
    """完美正相关 → 相关系数 = 1"""
    corr = hy_correlation(x, x)      # 与自身相关 = 1
    assert corr == pytest.approx(1.0, abs=1e-6)
```

**技巧 3: pytest.approx 处理浮点误差**

```python
# 不要用 == 比较浮点数
assert result == pytest.approx(expected, abs=1e-10)
```

### 运行测试

```bash
# 全部 Phase 2a 测试
python -m pytest factor_research/tests/ -v

# 单个文件
python -m pytest factor_research/tests/test_evaluation.py -v

# 单个类/方法
python -m pytest factor_research/tests/test_evaluation.py::TestICAnalysis::test_ic_decay -v

# 全项目测试（Phase 1 + Phase 2a）
python -m pytest data_infra/tests/ factor_research/tests/ -v  # 预期 323 项全通过
```

---

## 十四、核心概念深入

### 14.1 什么是 IC (Information Coefficient)？

IC 是量化因子研究中最核心的评价指标，本质是"因子值与未来收益的秩相关"。

**直觉:**
如果你有一个因子，每天给出 5 个币种的打分。如果打分高的币种后续确实涨得多，说明因子有预测力。IC 就是度量这个"一致性"的数值。

**计算过程:**
```
时刻 t:
    因子值: BTC=0.5, ETH=-0.3, SOL=0.8, BNB=0.1, DOGE=-0.7
    前瞻收益: BTC=2%, ETH=-1%, SOL=3%, BNB=0.5%, DOGE=-2%

    因子排名: DOGE(1) < ETH(2) < BNB(3) < BTC(4) < SOL(5)
    收益排名: DOGE(1) < ETH(2) < BNB(3) < BTC(4) < SOL(5)

    IC_t = Spearman(因子排名, 收益排名) = 1.0 (完美一致)
```

**IC 参考基准:**
- |IC| < 0.02: 无预测力（随机噪声）
- |IC| ∈ [0.02, 0.05]: 弱预测力（可考虑）
- |IC| ∈ [0.05, 0.1]: 中等预测力（有价值）
- |IC| > 0.1: 强预测力（非常有价值，需要警惕过拟合）

**为什么用 Spearman 而非 Pearson？**
- Spearman 只看排名，不看具体值 → 对异常值稳健
- 投资决策关心"哪个最好"（排名），不关心"好多少"（具体值）

### 14.2 什么是 IC_IR？

IC_IR = IC 的平均值 / IC 的标准差

类比:
- IC 均值 = 平均预测力（类似平均收益）
- IC 标准差 = 预测力波动（类似收益波动）
- IC_IR = 预测力的夏普比（稳定性指标）

|IC_IR| > 0.5 是好因子的基本门槛。

### 14.3 什么是 IC 衰减？

在不同前瞻窗口 (h=1,5,10,30,60) 下计算 IC。

**理想的 IC 衰减曲线:**
```
h=1:  IC=0.08  ← 短期最强
h=5:  IC=0.06  ← 逐步衰减
h=10: IC=0.04
h=30: IC=0.02  ← 信号耗散
h=60: IC=0.01  ← 接近零
```

**衰减速度的含义:**
- 快速衰减 → 短期效应，适合高频策略
- 缓慢衰减 → 持久信号，适合低频策略
- 先低后高 → 延迟效应（不常见）

### 14.4 什么是分层回测 (Quantile Backtest)？

**思路:**
1. 每个时刻，按因子值将所有标的排序
2. 分成 N 组（如 5 组）：Q1（因子值最低）到 Q5（因子值最高）
3. 计算每组的平均收益

**好因子的表现:**
```
Q1（最低因子值）: 收益最低
Q2: ...
Q3: ...
Q4: ...
Q5（最高因子值）: 收益最高
```
即组收益单调递增。

**多空收益:**
多空收益 = Q5 的收益 - Q1 的收益

这模拟了"做多因子值最高的标的，做空因子值最低的标的"的策略。

### 14.5 三层 API 设计的哲学

```
第一层: FactorAnalyzer.full_report()
    → 适用: notebook 快速评估，一行代码出完整报告
    → 灵活性: 低（固定格式）
    → 使用频率: 最高

第二层: ic_analysis(), quantile_backtest(), ...
    → 适用: 深入某个维度，自定义参数
    → 灵活性: 中
    → 使用频率: 中

第三层: spearman_ic(), compute_forward_returns(), ...
    → 适用: 构建自定义分析流程
    → 灵活性: 高（纯函数，任意组合）
    → 使用频率: 低（研究型使用）
```

这个设计保证了:
- 新手用第一层就够了
- 进阶用户可以深入第二层调参
- 专家可以用第三层自由组合

---

## 十五、设计模式总结

### 模式 1: 统一面板格式

```
所有因子输出:
    pd.DataFrame
    index:   DatetimeIndex (UTC)
    columns: symbol 列表
    values:  float64 因子值
```
一种格式贯穿因子定义、存储、评价的全流程。

### 模式 2: 模板方法

```python
class TimeSeriesFactor:
    def compute(self, data):        # 模板方法（固定流程）
        for symbol in data:
            self.compute_single()   # 子类实现（变化部分）
```
分离"固定流程"和"可变逻辑"。

### 模式 3: 定义即注册

```python
@register_factor
class MyFactor(TimeSeriesFactor):
    ...
# 定义 ≡ 注册，无需额外步骤
```
减少遗漏注册的风险。

### 模式 4: 三层 API

```
一键报告 → 独立分析函数 → 底层纯函数原语
便利性递减，灵活性递增
```
满足不同层次的使用需求。

### 模式 5: 数据声明式

```python
data_requirements = [
    DataRequest(DataType.OHLCV, timeframe="1m", lookback_bars=60),
]
# 因子只声明"我需要什么"，引擎负责"怎么准备"
```
因子逻辑与数据获取解耦。

### 模式 6: 参数化因子家族

```python
@register_factor_family
class Base(TimeSeriesFactor):
    _param_grid = {"lookback": [5, 10, 30]}

    def __init__(self, lookback: int = 5):
        self.lookback = lookback
    ...
# → 声明一次，自动展开为 3 个变体
```
声明式参数网格 + 自动笛卡尔积展开，因子作者只写一个类。

---

## 十六、数据流全景图

```
                            DataReader (Phase 1)
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
              get_ohlcv()    get_ticks()   get_orderbook()  ...
                    │              │              │
                    └──────────────┼──────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │         FactorEngine         │
                    │  1. 读取 FactorMeta          │
                    │  2. _prepare_data()          │
                    │  3. factor.compute()         │
                    │  4. FactorStore.save()       │
                    └──────────────┼──────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
             TimeSeriesFactor  CrossSectional  CrossAsset
             compute_single()  compute()       compute()
                    │              │              │
                    └──────────────┼──────────────┘
                                   │
                      pd.DataFrame (因子面板)
                     index=timestamp, cols=symbols
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
              FactorStore    FactorAnalyzer    alignment/
              (持久化)       (评价体系)        (对齐工具)
                    │              │
              db/factors/    ┌─────┼─────┐
              ├─ output.pq   │     │     │
              └─ meta.json  ic  quantile tail  ...
                    │
                    ▼
             Phase 2b（模型/策略）
             唯一接口: FactorStore
```

---

## 十七、延伸练习

### 练习 1: 编写新因子（简单）

编写一个"已实现波动率"因子:
- 类型: TimeSeriesFactor
- 计算: 过去 N 根 K线的收益率标准差
- 用 `@register_factor` 注册
- 运行 `FactorRegistry().get("your_factor_name")` 验证注册成功

### 练习 2: 评价随机因子（简单）

用 `np.random.randn()` 构造一个随机因子面板，用 FactorAnalyzer 评价。
观察: 随机因子的 IC 应该接近 0，IC_IR 很低，分层回测无单调性。

### 练习 3: 构造有预测力的因子（中等）

在合成数据中构造一个有预测力的因子:
```python
# 提示: 让因子值与前瞻收益相关
signal = np.random.randn(n)
prices = 100 + np.cumsum(signal * 0.1 + noise * 0.01)
factor_values = signal  # 因子 = 未来收益的信号
```
用 FactorAnalyzer 评价，观察 IC 应该显著为正。

### 练习 4: 理解 IC 衰减（中等）

对练习 3 的因子，计算 h=1,5,10,30,60 的 IC 衰减曲线。
思考: 为什么 h 越大 IC 越低？（答: 因子信号随时间耗散）

### 练习 5: 编写截面因子（进阶）

编写一个"截面动量排名"因子:
- 类型: CrossSectionalFactor
- 计算: 每个时刻对所有标的的 5 根 K线收益率排名（rank(axis=1, pct=True)）
- 注意 data 参数的格式与 TimeSeriesFactor 不同

### 练习 6: 参数搜索（进阶）

对 MultiScaleReturns 因子家族，评价 lookback = 5,10,15,20,...,120 的所有变体。
绘制 IC 均值 vs lookback 的曲线，找到最优窗口。

### 练习 7: Hayashi-Yoshida 实验（进阶）

构造两个异步 tick 序列（时间戳不对齐），分别用以下方法计算相关性:
1. grid_align + Pearson 相关
2. Hayashi-Yoshida 相关

比较结果差异，理解 grid_align 的信息损失。

### 练习 8: 阅读测试理解行为（进阶）

只阅读 `test_evaluation.py` 的测试用例名称和断言，不看源码。
尝试推断每个评价函数的输入输出格式和返回结构。
然后对照源码验证你的推断。

---

**Phase 2a 学习完成后，你应该能:**
1. 理解因子研究的完整管道（定义→计算→存储→评价）
2. 独立编写新因子并评价
3. 解读 IC/IR/IC衰减/分层回测等评价指标
4. 理解三种因子类型的设计差异和适用场景
5. 使用数据对齐工具处理异步高频数据

---
---

# Phase 2a 进阶学习 — 因子研究方法论与实战深入

以下内容是 Phase 2a 的进阶学习材料，帮助你从"会用框架"提升到"理解为什么这么设计"。
建议在完成上面所有基础站点学习后再阅读本部分。

---

## 十八、因子研究方法论入门

### 18.1 什么是 Alpha 因子？

在量化投资中，**因子 (Factor)** 是一个能够预测资产未来收益的信号。更具体地说:

- **Beta 因子**: 解释市场整体收益的因素（如市场风险溢价）。持有任何资产都会暴露于 Beta。
- **Alpha 因子**: 扣除市场共同驱动后，仍能产生**超额收益**的信号。Alpha 是主动管理的核心价值。

在我们的框架中，所有因子（`TimeSeriesFactor`, `CrossSectionalFactor`, `CrossAssetFactor`）都以寻找 Alpha 为目标。因子值越高的标的，预期未来收益越高（或越低，取决于因子方向）。

**因子的数学表达:**

```
因子值 f(t, i): 在时刻 t，对标的 i 的一个数值打分
预测: E[r(t+h, i)] ∝ f(t, i)    （正因子）
   或 E[r(t+h, i)] ∝ -f(t, i)   （负因子）
```

### 18.2 因子研究的标准流程

因子研究在学术界和工业界有显著差异:

**学术界流程:**
```
1. 文献综述 → 找到理论假说
2. 数据收集 → 清洗、处理
3. 因子构造 → 按假说定义
4. 统计检验 → t-test, Fama-MacBeth 回归
5. 稳健性检验 → 子样本、不同市场、不同时期
6. 发表论文
```

**工业界流程:**
```
1. 因子灵感 → 来自理论、直觉、数据探索、竞品分析
2. 快速原型 → notebook 中计算因子值
3. 因子评价 → IC、分层回测、稳定性（本框架的核心功能）
4. 因子代码化 → 从 notebook 提取为 @register_factor 类
5. 组合构建 → 多因子加权 → 策略信号
6. 回测验证 → 模拟交易环境下的完整策略回测
7. 实盘部署 → 纸交易 → 小资金实盘
```

工业界的核心区别:
- **速度**: 论文要一年，产业可能一周就要出结论
- **评价标准**: 学术看 p-value，产业看 IC_IR 和扣费后净收益
- **数据驱动**: 产业更依赖数据探索发现信号，学术更依赖理论推导
- **迭代速度**: 产业需要快速验证和放弃不好的因子

### 18.3 Crypto 市场因子研究的特殊性

加密货币市场与传统股票/期货市场有很多本质差异，这些差异深刻影响因子研究的方法论:

**1. 24/7 不间断交易**

传统市场有开盘/收盘，日频数据自然以"交易日"为单位。Crypto 没有休市概念:
- "日收益率"可以定义为任意 24 小时窗口
- "隔夜效应"不存在
- 周末和节假日照常交易，但流动性可能显著下降
- **影响**: 我们选择以 UTC 00:00 为日界线，以 1 分钟 K 线为基础粒度

**2. 高波动率**

Crypto 资产的日波动率通常是传统资产的 3-10 倍:
- BTC 年化波动率 ~60-80%（标普 500 ~15-20%）
- 因子信号容易被噪声淹没 → 需要更高的 IC 门槛
- 极端行情频繁 → 尾部分析 (`tail_analysis()`) 尤为重要
- **影响**: 框架中包含尾部分析维度，以及 regime 分析（高波/低波分别看 IC）

**3. 标的数量少**

我们目前只有 5 个标的（BTC, ETH, SOL, BNB, DOGE），远少于股票市场的数千只:
- 截面 IC 的统计意义受限（5 个点的 Spearman 相关不够稳健）
- 分层回测中每组只有 1-2 个标的
- **影响**: 需要更谨慎地解读截面指标；时序因子可能比截面因子更有效
- **缓解**: 拉长时间维度（分钟频数据量大），用时序稳定性弥补截面宽度不足

**4. 独特的市场微观结构**

Crypto 交易所有许多传统市场没有的特征:
- **永续合约资金费率**: 每 8 小时结算一次，直接影响价格
- **杠杆交易普遍**: 大量爆仓事件导致价格剧烈波动
- **跨交易所套利**: 同一资产在不同交易所价格不完全一致
- **订单簿透明**: 完整的买卖盘信息可公开获取
- **影响**: 框架中包含微观结构因子（orderbook imbalance）和合约市场数据（funding rate, OI, long/short ratio）

**5. 无基本面锚定**

传统股票有财务报表、估值模型。Crypto 资产:
- 难以定义"内在价值"
- 更依赖技术面和链上数据
- 情绪驱动更强
- **影响**: 动量和微观结构因子通常比"价值型"因子更有效

---

## 十九、各评价维度的深度解读

### 19.1 为什么需要 6 个维度？

很多初学者会问："直接看 IC 不就行了？"答案是：**IC 只能告诉你因子平均有多强，但无法告诉你因子是否安全、稳定、可交易。**

一个好的因子必须通过多个维度的交叉验证，才能放心用于实盘:

```
                        IC 分析
                         /    \
                        /      \
              分层回测    尾部分析
                |            |
              稳定性分析   非线性分析
                        \   /
                       换手分析
```

### 19.2 各维度能发现什么、不能发现什么

#### 维度 1: IC 分析 (`ic_analysis`)

**能发现:**
- 因子的平均预测力（IC 均值）
- 预测力的稳定性（IC_IR = IC_mean / IC_std）
- 因子信号的有效期（IC 衰减曲线）
- 因子方向是否一致（IC 胜率）

**不能发现:**
- IC 高是否来自个别极端值（需要尾部分析）
- IC 高是否只在特定市场环境下成立（需要稳定性分析）
- 高 IC 是否可以实际交易获利（需要看换手和交易成本）

#### 维度 2: 分层回测 (`quantile_backtest`)

**能发现:**
- 因子值高低与收益的单调关系（组收益单调性）
- 因子的多空收益和夏普比
- 最大回撤和收益曲线形态

**不能发现:**
- 分层结果可能受单标的极端收益扭曲（标的少时尤其明显）
- 无法区分因子是"连续有效"还是"偶尔爆发"

#### 维度 3: 尾部分析 (`tail_analysis`)

**能发现:**
- 极端因子值（>90 分位）时的预测力是否更强
- 尾部信号的命中率
- MAE（最大逆向偏移）: 极端信号后的最大浮亏

**不能发现:**
- 尾部事件本身较少，统计上可能不稳健
- 无法反映因子在"正常区间"的表现

#### 维度 4: 稳定性分析 (`stability_analysis`)

**能发现:**
- 因子在不同市场环境下的表现（趋势 vs 震荡, 高波 vs 低波）
- IC 的时间演变趋势（是否在衰减？）
- IC 最大回撤（最差时期有多严重）
- 月度分解，发现季节性或失效月份

**不能发现:**
- 未来是否会出现从未见过的 regime
- 因子失效的根本原因

#### 维度 5: 非线性分析 (`nonlinear_analysis`)

**能发现:**
- 因子与收益的关系是否非线性（互信息 vs IC 的差异）
- 因子在不同取值区间的 IC 差异（条件 IC）
- 因子 Profile: 因子值分 bin 后每 bin 的平均收益

**不能发现:**
- 非线性的具体形式（只知道存在，不知道如何建模）
- 如果非线性很复杂，线性框架下可能丢失信息

#### 维度 6: 换手分析 (`turnover_analysis`)

**能发现:**
- 因子信号的自相关（高自相关 → 低换手 → 低交易成本）
- 排名变化率（标的排名在相邻时刻的变化幅度）
- 信号翻转率（多空方向翻转的频率）

**不能发现:**
- 具体的交易成本金额（需要知道手续费率和滑点模型）
- 换手分解到个别标的的贡献

### 19.3 评价指标之间的内在联系

指标之间不是孤立的，它们存在重要的交叉关系:

**关系 1: IC 高 + 换手高 → 检查扣费后净收益**

```
如果一个因子 IC = 0.06（不错），但自相关 = 0.1（每分钟信号几乎全换）
→ 每分钟都在全仓调仓
→ 手续费（maker 0.02% + taker 0.04%）× 大量换手 → 吃掉所有利润
→ 需要 turnover_analysis() 确认
```

**关系 2: IC 高 + 尾部 IC 更高 → 好信号**

```
整体 IC = 0.05，尾部 IC = 0.12
→ 因子在极端值时预测力翻倍
→ 可以设计"只在因子极端时交易"的策略
→ 减少换手，提高每笔交易的期望收益
```

**关系 3: IC 高 + 稳定性差 → 高风险因子**

```
整体 IC = 0.08，但趋势行情 IC = 0.15，震荡行情 IC = -0.02
→ 因子只在趋势行情中有效
→ 需要先判断当前是什么行情，再决定是否使用
→ 或者降低因子权重以控制风险
```

**关系 4: IC 高 + 非线性强 → 可能需要非线性模型**

```
IC = 0.04（一般），但 MI = 0.3（远高于线性解释）
→ 因子与收益存在非线性关系
→ 线性 IC 低估了因子价值
→ 可以考虑分段处理或用机器学习模型捕捉非线性
```

---

## 二十、常见陷阱和反模式

### 20.1 过拟合 (Overfitting)

**什么是过拟合？**

因子有太多参数，在历史数据上"记住"了噪声而非信号。样本内表现优异，样本外一塌糊涂。

**典型场景:**

```python
# 坏例子: 参数过多的因子
class OverfittedFactor(TimeSeriesFactor):
    def compute_single(self, symbol, data):
        close = data[DataType.OHLCV]["close"]
        volume = data[DataType.OHLCV]["volume"]

        # 14 个参数！每个都是"调出来的最优值"
        ma_fast = close.rolling(7).mean()          # 为什么是 7？
        ma_slow = close.rolling(23).mean()         # 为什么是 23？
        vol_threshold = volume.rolling(17).quantile(0.83)  # 为什么是 17 和 0.83？
        rsi = ...  # rsi_window=9, overbought=72, oversold=31
        # ... 更多自由参数 ...

        signal = (ma_fast > ma_slow) & (volume > vol_threshold) & (rsi < 72) ...
        return signal.astype(float)
```

**如何检测:**
- 样本内 IC 显著高于样本外 IC（参见练习 10）
- 因子参数微调后 IC 剧烈变化
- 参数取值缺乏经济学直觉

**经验法则:**
- 因子参数不超过 2-3 个
- 每个参数都应有经济学解释
- 参数应在一定范围内稳健（如 lookback = 5 和 lookback = 7 的 IC 应该接近）

### 20.2 前视偏差 (Lookahead Bias)

**什么是前视偏差？**

因子计算中意外使用了"未来信息"。这是量化研究中最严重的错误之一。

**典型场景:**

```python
# 错误: 用整段数据的均值做归一化
close = data["close"]
factor = (close - close.mean()) / close.std()
# close.mean() 包含了未来数据！

# 正确: 用滚动窗口
factor = (close - close.rolling(60).mean()) / close.rolling(60).std()
```

```python
# 错误: forward return 计算方向搞反
forward_ret = price.shift(5) / price - 1  # 这是过去收益，不是未来收益！
# 正确:
forward_ret = price.shift(-5) / price - 1  # shift(-5) 是未来价格
```

**如何检测:**
- 因子在 h=1 时 IC 异常高（|IC| > 0.3）→ 几乎一定是前视偏差
- 实盘收益远低于回测
- 用 `compute_forward_returns()` 时注意 shift 方向

### 20.3 生存者偏差 (Survivorship Bias)

**什么是生存者偏差？**

只用"活下来的"标的做研究，忽略了退市/归零的标的。

**在 Crypto 中的表现:**
- 只研究 BTC、ETH 等蓝筹币 → 忽略了大量归零的山寨币
- LUNA/UST 崩盘、FTT 归零这类事件在回测中消失
- 因子在蓝筹币上的 IC 可能虚高

**缓解方法:**
- 在我们的框架中，5 个标的都是蓝筹，生存者偏差的影响相对可控
- 但在扩展到更多标的时需要注意
- 保留历史数据中的退市标的（即使后来表现差）

### 20.4 交易成本忽略 (Transaction Cost Neglect)

**什么是交易成本忽略？**

因子评价只看信号质量（IC），忽略将信号转化为交易的成本。

**Crypto 交易成本结构:**

```
手续费: Binance maker 0.02%, taker 0.04% (有 BNB 抵扣)
滑点: 取决于订单大小和市场深度
  - 小单 (< $10k): 几乎无滑点
  - 中单 ($10k-$100k): 1-3 bps
  - 大单 (> $100k): 可能 5-10+ bps

单次交易总成本 ≈ 0.04% - 0.1%
```

**实际影响:**

```
假设因子 IC = 0.05, 月换手率 100%（每月全仓调仓一次）
  → 月交易成本 = 0.04% × 2（买+卖）= 0.08%
  → 年交易成本 = 0.08% × 12 = 0.96%
  → 对于年化超额收益 5% 的策略，成本占 20%

如果月换手率 2000%（每天全仓调仓一次）
  → 年交易成本 = 0.08% × 252 ≈ 20%
  → 吃掉大部分甚至全部利润
```

**框架中的对策:**
- `turnover_analysis()` 提供自相关和排名变化率
- 高换手因子需要更高的 IC 才有价值
- Phase 2b 中将引入交易成本模型

---

## 二十一、从因子到策略的衔接

### 21.1 Phase 2a → Phase 2b 的桥梁: FactorStore

Phase 2a（因子研究）和 Phase 2b（模型策略）之间通过 `FactorStore` 进行唯一的数据通信:

```
Phase 2a                          Phase 2b
┌─────────────────────┐          ┌─────────────────────┐
│ 因子开发             │          │ 策略构建             │
│ FactorEngine        │          │ 因子组合             │
│ FactorAnalyzer      │   ───>   │ 风险模型             │
│ notebooks           │ FactorStore │ 交易信号生成       │
│                     │          │ 回测引擎             │
└─────────────────────┘          └─────────────────────┘
```

**为什么 FactorStore 是唯一接口？**

1. **解耦**: Phase 2a 可以独立迭代因子，不影响策略代码
2. **版本控制**: FactorStore 中的 `meta.json` 记录了因子的版本、参数、计算逻辑
3. **复现性**: 任何策略都可以追溯到具体的因子版本
4. **标准化**: 所有因子的输出格式统一（面板 DataFrame），策略端无需适配不同格式

### 21.2 因子组合的基本思路

单个因子的预测力有限，实际策略通常组合多个因子:

**方法 1: 等权组合**

```python
# 最简单的方法: 所有因子等权
composite = (factor_1 + factor_2 + factor_3) / 3

# 注意: 需要先归一化（rank_normalize），消除量纲差异
from factor_research.evaluation.metrics import rank_normalize
f1_normed = rank_normalize(factor_1)
f2_normed = rank_normalize(factor_2)
composite = (f1_normed + f2_normed) / 2
```

优点: 简单稳健，无过拟合风险
缺点: 忽略了因子质量的差异

**方法 2: IC 加权**

```python
# 按因子的历史 IC 均值加权
weights = {"returns_5m": 0.06, "returns_30m": 0.03, "imbalance": 0.08}
# 归一化权重
total = sum(weights.values())
weights = {k: v / total for k, v in weights.items()}

composite = sum(w * rank_normalize(f) for f, w in zip(factors, weights.values()))
```

优点: IC 高的因子权重大，合理利用了因子质量信息
缺点: 使用了历史 IC，可能过拟合

**方法 3: 优化法（进阶）**

```python
# 最大化组合 IC（或最小化组合方差）
# 类似 Markowitz 均值-方差优化
# 需要因子协方差矩阵 → correlation_analysis() 提供
# 此方法需谨慎使用，容易过拟合
```

### 21.3 风险模型基本概念

在从因子到策略的过程中，风险模型决定了"持仓大小":

**协方差矩阵:**

```
              BTC    ETH    SOL    BNB    DOGE
BTC        [ 1.0    0.8    0.7    0.6    0.5  ]
ETH        [ 0.8    1.0    0.7    0.6    0.5  ]
SOL        [ 0.7    0.7    1.0    0.5    0.4  ]
BNB        [ 0.6    0.6    0.5    1.0    0.4  ]
DOGE       [ 0.5    0.5    0.4    0.4    1.0  ]

→ BTC 和 ETH 高度相关（0.8）
→ 同时做多 BTC 和 ETH 的分散化效果有限
→ 做多 BTC + 做空 ETH 可以对冲系统性风险
```

框架中的 `hayashi_yoshida.py` 可以在异步高频数据上计算精确的协方差矩阵。

**VIF (方差膨胀因子):**

VIF 衡量多因子之间的共线性程度:

```python
from factor_research.evaluation.correlation import vif_analysis

# VIF 解读:
# VIF = 1:  该因子与其他因子完全独立
# VIF < 5:  可接受
# VIF > 5:  存在多重共线性，因子冗余
# VIF > 10: 严重共线性，应删除该因子
```

为什么关心 VIF？
- 冗余因子浪费计算资源
- 在优化法中，共线因子会导致权重不稳定
- 增量 IC (`incremental_ic()`) 可以量化新因子的边际贡献

---

## 二十二、推荐阅读资料

### 22.1 经典书籍

**因子投资理论:**

1. **《Advances in Financial Machine Learning》** — Marcos Lopez de Prado
   - 量化金融机器学习的里程碑著作
   - 重点章节: 金融数据结构（bar 类型）、标签方法（triple barrier）、交叉验证（purged K-fold）
   - 与本项目的关联: 理解为什么不能用普通的 train/test split

2. **《Quantitative Equity Portfolio Management》** — Qian, Hua, Sorensen
   - 多因子模型的经典教材
   - 重点章节: 因子模型构建、风险模型、组合优化
   - 与本项目的关联: Phase 2b 因子组合和风险模型的理论基础

3. **《Efficiently Inefficient》** — Lasse Heje Pedersen
   - 对冲基金策略的学术视角
   - 重点章节: 动量策略、价值策略、做市策略
   - 与本项目的关联: 理解不同因子背后的经济学逻辑

**统计与计量经济学:**

4. **《The Elements of Statistical Learning》** — Hastie, Tibshirani, Friedman
   - 统计学习经典
   - 重点章节: 偏差-方差权衡、交叉验证、正则化
   - 与本项目的关联: 理解过拟合的理论基础

### 22.2 重要论文

1. **Hayashi, T. & Yoshida, N. (2005)** — "On covariance estimation of non-synchronously observed diffusion processes"
   - Hayashi-Yoshida 异步协方差估计量的原始论文
   - 与本项目的关联: `alignment/hayashi_yoshida.py` 的理论基础
   - 核心思想: 在两个非同步观测的时间序列之间计算协方差，无需先对齐

2. **Fama, E. & French, K. (1993)** — "Common risk factors in the returns on stocks and bonds"
   - 三因子模型的奠基论文
   - 理解因子如何解释资产收益的截面差异

3. **Harvey, C., Liu, Y., & Zhu, H. (2016)** — "... and the Cross-Section of Expected Returns"
   - 因子动物园（Factor Zoo）问题的经典警示
   - 数百个"显著"因子中，大部分是多重检验的假阳性
   - 与本项目的关联: 提醒我们警惕过拟合和数据挖掘

### 22.3 Python 量化开源项目

1. **Alphalens** (https://github.com/quantopian/alphalens)
   - Quantopian 出品的因子分析工具
   - 功能: IC 分析、分层回测、换手分析
   - 与本项目的关系: 我们的评价体系参考了 Alphalens 的设计思路，但针对 Crypto 做了定制（尾部分析、异步对齐等）

2. **Qlib** (https://github.com/microsoft/qlib)
   - 微软出品的 AI 量化投研平台
   - 功能: 数据管理、因子表达式引擎、模型训练、回测
   - 学习重点: 数据抽象层的设计、因子表达式引擎的效率优化

3. **VectorBT** (https://github.com/polanikov/vectorbt)
   - 高性能向量化回测框架
   - 学习重点: NumPy/Pandas 向量化技巧、避免 Python 循环

4. **CCXT** (https://github.com/ccxt/ccxt)
   - 我们在 Phase 1 中使用的交易所统一接口
   - 支持 100+ 交易所，统一 API 设计
   - 学习重点: REST vs WebSocket、异步编程模式

---

## 二十三、更多实战练习

### 练习 9: 因子衰减分析实战

**目标:** 理解不同因子的 IC 衰减特征差异。

**步骤:**

1. 构造一个**快速衰减因子**:

```python
import numpy as np, pandas as pd
from factor_research.evaluation.ic import ic_decay

np.random.seed(42)
n = 1000
index = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")

# 快速衰减因子: 只与下一根 K 线的收益相关
signal = np.random.randn(n)
noise = np.random.randn(n) * 0.5

# 价格: 下一根 K 线涨跌由 signal 决定，但效果迅速消失
returns = signal * 0.002 + noise * 0.01
prices_btc = 100 * np.exp(np.cumsum(returns))
prices_eth = 50 * np.exp(np.cumsum(np.random.randn(n) * 0.005))  # 噪声

factor_fast = pd.DataFrame({
    "BTC/USDT": signal + np.random.randn(n) * 0.3,
    "ETH/USDT": np.random.randn(n),
}, index=index)

price_panel = pd.DataFrame({
    "BTC/USDT": prices_btc,
    "ETH/USDT": prices_eth,
}, index=index)
```

2. 构造一个**慢速衰减因子**:

```python
# 慢速衰减因子: 基于长期趋势
trend_signal = pd.Series(signal).rolling(30).mean().fillna(0).values

factor_slow = pd.DataFrame({
    "BTC/USDT": trend_signal + np.random.randn(n) * 0.3,
    "ETH/USDT": np.random.randn(n),
}, index=index)
```

3. 分别计算 IC 衰减:

```python
decay_fast = ic_decay(factor_fast, price_panel, horizons=[1, 5, 10, 30, 60])
decay_slow = ic_decay(factor_slow, price_panel, horizons=[1, 5, 10, 30, 60])

print("快速衰减因子:")
print(decay_fast)
print("\n慢速衰减因子:")
print(decay_slow)

# 预期: fast 在 h=1 时 IC 最高，之后迅速下降
#       slow 在 h=1 时 IC 较低，但 h=30 时仍有 IC
```

4. 思考: 这两种因子分别适合什么交易频率？

---

### 练习 10: 过拟合检测

**目标:** 亲手体验过拟合现象，理解为什么参数越多不一定越好。

**步骤:**

1. 准备数据，分为训练集和测试集:

```python
import numpy as np, pandas as pd
from factor_research.evaluation.ic import ic_summary

np.random.seed(42)
n = 2000
index = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")

# 真实信号（弱信号）
true_signal = np.random.randn(n)
noise = np.random.randn(n) * 5
prices_btc = 100 * np.exp(np.cumsum(true_signal * 0.001 + noise * 0.001))
prices_eth = 50 * np.exp(np.cumsum(np.random.randn(n) * 0.005))

price_panel = pd.DataFrame({
    "BTC/USDT": prices_btc,
    "ETH/USDT": prices_eth,
}, index=index)

# 分割: 前 1000 条训练，后 1000 条测试
train_prices = price_panel.iloc[:1000]
test_prices = price_panel.iloc[1000:]
```

2. 构造不同复杂度的因子:

```python
close = price_panel["BTC/USDT"]

# 简单因子: 1 个参数 (lookback=10)
factor_simple = pd.DataFrame({
    "BTC/USDT": (close / close.shift(10) - 1),
    "ETH/USDT": np.random.randn(n),
}, index=index)

# 复杂因子: 5+ 个参数（在训练集上"优化"出来的）
ma1 = close.rolling(7).mean()
ma2 = close.rolling(23).mean()
ma3 = close.rolling(51).mean()
factor_complex = pd.DataFrame({
    "BTC/USDT": (ma1 - ma2) / ma3 * (close > ma1).astype(float),
    "ETH/USDT": np.random.randn(n),
}, index=index)
```

3. 比较训练集和测试集的 IC:

```python
from factor_research.evaluation.ic import ic_series, ic_summary

# 简单因子: 先计算 IC 时间序列，再汇总统计
train_ic_ts_simple = ic_series(factor_simple.iloc[:1000], train_prices, horizon=1)
train_ic_simple = ic_summary(train_ic_ts_simple)
test_ic_ts_simple = ic_series(factor_simple.iloc[1000:], test_prices, horizon=1)
test_ic_simple = ic_summary(test_ic_ts_simple)

# 复杂因子: 同样两步走
train_ic_ts_complex = ic_series(factor_complex.iloc[:1000], train_prices, horizon=1)
train_ic_complex = ic_summary(train_ic_ts_complex)
test_ic_ts_complex = ic_series(factor_complex.iloc[1000:], test_prices, horizon=1)
test_ic_complex = ic_summary(test_ic_ts_complex)

print(f"简单因子 — 训练IC: {train_ic_simple['ic_mean']:.4f}, "
      f"测试IC: {test_ic_simple['ic_mean']:.4f}")
print(f"复杂因子 — 训练IC: {train_ic_complex['ic_mean']:.4f}, "
      f"测试IC: {test_ic_complex['ic_mean']:.4f}")

# 预期: 复杂因子训练IC > 简单因子训练IC
#       但复杂因子测试IC 可能 < 简单因子测试IC → 过拟合！
```

4. 思考: 如何在不看测试集的情况下判断因子是否过拟合？

---

### 练习 11: 完整因子开发流程

**目标:** 从零完成一个因子的全生命周期: 灵感 → 探索 → 代码化 → 评价 → 入库。

**步骤:**

1. **灵感**: 已实现波动率（Realized Volatility）因子
   - 假说: 高波动后可能出现均值回复，低波动后可能爆发

2. **notebook 探索**:

```python
import numpy as np, pandas as pd

# 合成数据
np.random.seed(42)
n = 1000
index = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
prices = pd.DataFrame({
    "BTC/USDT": 100 + np.cumsum(np.random.randn(n) * 0.1),
    "ETH/USDT": 50 + np.cumsum(np.random.randn(n) * 0.05),
}, index=index)

# 计算已实现波动率
returns = prices.pct_change()
realized_vol = returns.rolling(20).std()

# 快速看一下 IC
from factor_research.evaluation.analyzer import FactorAnalyzer
analyzer = FactorAnalyzer(realized_vol.dropna(), prices)
print(analyzer.summary_text(factor_name="realized_vol_20"))
```

3. **代码化** — 创建因子类:

```python
# 在 factor_research/factors/volatility/ 目录下创建 realized_vol.py

from factor_research.core.base import TimeSeriesFactor
from factor_research.core.types import (
    DataType, DataRequest, FactorMeta, FactorType,
)
from factor_research.core.registry import register_factor


@register_factor
class RealizedVol20(TimeSeriesFactor):
    """20 根 K 线的已实现波动率"""

    def meta(self) -> FactorMeta:
        return FactorMeta(
            name="realized_vol_20",
            display_name="已实现波动率 (20 bar)",
            factor_type=FactorType.TIME_SERIES,
            category="volatility",
            description="过去 20 根 1m K 线收益率的标准差",
            data_requirements=[
                DataRequest(DataType.OHLCV, timeframe="1m", lookback_bars=25),
            ],
            output_freq="1m",
            params={"window": 20},
        )

    def compute_single(self, symbol, data):
        close = data[DataType.OHLCV]["close"]
        returns = close.pct_change()
        vol = returns.rolling(20).std().dropna()
        return vol
```

4. **验证注册**:

```python
from factor_research.core.registry import get_default_registry
reg = get_default_registry()  # 获取全局注册表（@register_factor 注册到此处）
print("realized_vol_20" in reg)  # True
```

5. **通过 FactorEngine 计算并入库**:

```python
from factor_research.core.engine import FactorEngine
engine = FactorEngine()
panel = engine.compute_factor("realized_vol_20", save=True)
print(panel.shape)
```

---

### 练习 11b: 参数化因子族与 FamilyAnalyzer

**目标:** 使用 `@register_factor_family` 创建参数化因子族，并用 `FamilyAnalyzer` 进行参数空间扫描。

**步骤:**

1. **创建参数化因子族**:

```python
# factor_research/factors/volatility/realized_vol.py

from factor_research.core.base import TimeSeriesFactor
from factor_research.core.registry import register_factor_family
from factor_research.core.types import *

@register_factor_family
class RealizedVol(TimeSeriesFactor):
    """已实现波动率因子族 — 不同窗口的波动率"""

    _param_grid = {"window": [10, 20, 30, 60]}  # 作者人工指定

    def __init__(self, window: int = 20):
        self.window = window

    def meta(self) -> FactorMeta:
        return FactorMeta(
            name=f"realized_vol_{self.window}",
            display_name=f"已实现波动率 ({self.window} bar)",
            factor_type=FactorType.TIME_SERIES,
            category="volatility",
            description=f"过去 {self.window} 根 K线收益率的标准差",
            data_requirements=[
                DataRequest(DataType.OHLCV, timeframe="1m", lookback_bars=self.window + 5),
            ],
            output_freq="1m",
            params={"window": self.window},
            family="realized_vol",  # 族名
        )

    def compute_single(self, symbol, data):
        close = data[DataType.OHLCV]["close"]
        returns = close.pct_change()
        vol = returns.rolling(self.window).std().dropna()
        return vol
# → 自动注册 4 个因子: realized_vol_10, realized_vol_20, realized_vol_30, realized_vol_60
```

2. **用 FamilyAnalyzer 扫参**:

```python
from factor_research.evaluation.family_analyzer import FamilyAnalyzer

# 准备数据（引擎提供便捷接口）
engine = FactorEngine()
data = engine.prepare_data(RealizedVol(), symbols=["BTC/USDT", "ETH/USDT"])

# 参数扫描
family = FamilyAnalyzer(
    factor_class=RealizedVol,
    data=data,
    price_panel=prices,
)
sweep_df = family.sweep()
print(sweep_df)  # 查看所有参数组合 × horizon 的轻量指标

# 可视化 + 筛选
family.plot_sensitivity(metric="ic_ir")
candidates = family.select(min_ic_ir=0.3, top_n=2)

# 钻取
report = family.detail(window=20)
```

**思考题:**
1. `_param_grid` 中的参数可以是非数字类型吗？（答: 可以，如字符串 "ema"/"sma"）
2. `FamilyAnalyzer.sweep()` 为什么不计算全部 6 个评价维度？（答: 保证扫描速度，全量评价留给 `detail()`）
3. 参数敏感性图中平坦区域意味着什么？（答: 参数稳健，不易过拟合）

---

### 练习 12: 多因子冗余检测

**目标:** 使用 `correlation_analysis` 和 `incremental_ic` 分析多个因子之间的冗余关系。

**步骤:**

1. 构造 5 个收益率因子:

```python
import numpy as np, pandas as pd

np.random.seed(42)
n = 500
index = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")

prices = pd.DataFrame({
    "BTC/USDT": 100 + np.cumsum(np.random.randn(n) * 0.1),
    "ETH/USDT": 50 + np.cumsum(np.random.randn(n) * 0.05),
    "SOL/USDT": 20 + np.cumsum(np.random.randn(n) * 0.08),
}, index=index)

close = prices["BTC/USDT"]

# 5 个不同 lookback 的收益率因子
factors = {}
for lb in [5, 10, 15, 20, 30]:
    name = f"returns_{lb}"
    factors[name] = pd.DataFrame({
        sym: (prices[sym] / prices[sym].shift(lb) - 1)
        for sym in prices.columns
    }, index=index)

# 去掉 NaN 行
valid_start = 30
for name in factors:
    factors[name] = factors[name].iloc[valid_start:]
price_valid = prices.iloc[valid_start:]
```

2. 计算因子相关矩阵:

```python
from factor_research.evaluation.correlation import correlation_analysis

# 构造因子字典（格式: {name: panel}）
corr_result = correlation_analysis(factors)
print("相关矩阵:")
print(corr_result["correlation_matrix"])
print("\nVIF:")
print(corr_result["vif"])

# 预期: returns_5 和 returns_10 相关性很高（0.7+）
#       returns_5 和 returns_30 相关性较低（0.3-0.5）
```

3. 使用增量 IC 分析新因子的边际贡献:

```python
from factor_research.evaluation.correlation import incremental_ic

# 假设已有 returns_5 和 returns_30，评估 returns_10 的边际贡献
result = incremental_ic(
    new_factor=factors["returns_10"],
    existing_factors=[factors["returns_5"], factors["returns_30"]],
    price_panel=price_valid,
    horizon=1,
)

print(f"原始 IC: {result['raw_ic']:.4f}")
print(f"增量 IC: {result['incremental_ic']:.4f}")
print(f"信息保留率: {result['info_retention']:.2%}")

# 如果 info_retention < 0.3 → returns_10 大部分信息已被其他因子覆盖 → 冗余
```

4. 基于分析结果进行因子筛选:

```python
# 筛选策略:
# - VIF > 5 的因子标记为"高共线性"
# - info_retention < 0.3 的因子标记为"冗余"
# - 在冗余因子中保留 IC 最高的那个

# 思考: 为什么不直接删除所有冗余因子？
# 答: 因为冗余因子在不同市场环境下可能有不同的表现
#     保留一定的冗余可以增加策略的稳健性
```

---

**Phase 2a 进阶学习完成后，你应该额外具备:**
1. 理解因子研究的方法论和 Crypto 市场的特殊性
2. 能够识别和避免过拟合、前视偏差等常见陷阱
3. 理解 6 个评价维度的深层含义和交叉关系
4. 了解从因子到策略（Phase 2b）的衔接路径
5. 具备独立完成因子全生命周期开发的能力

**祝学习顺利！**

---
---

# Phase 2b 模型策略 — 代码学习指引

本章节对照 `alpha_model/` 源码，系统讲解从**因子面板**到**可交易目标权重**的完整流程。

建议先完成 Phase 2a 的学习再进入本章。Phase 2b 的核心理念是：**框架做管道，模型做黑盒**——框架负责数据流编排（预处理 → 训练 → 信号 → 组合 → 回测），模型实现完全外接，通过 AlphaModel 协议交互。

---

## 目录

- [一、Phase 2b 整体结构](#一phase-2b-整体结构)
- [二、建议学习路线](#二建议学习路线-1)
- [三、第 1 站：核心类型 core/types.py](#三第-1-站核心类型-coretypespy)
- [四、第 2 站：预处理 preprocessing/](#四第-2-站预处理-preprocessing)
- [五、第 3 站：训练框架 training/](#五第-3-站训练框架-training)
- [六、第 4 站：信号生成 signal/](#六第-4-站信号生成-signal)
- [七、第 5 站：组合构建 portfolio/](#七第-5-站组合构建-portfolio)
- [八、第 6 站：向量化回测 backtest/](#八第-6-站向量化回测-backtest)
- [九、第 7 站：持久化 store/](#九第-7-站持久化-store)
- [十、第 8 站：参考模型 models/](#十第-8-站参考模型-models)
- [十一、第 9 站：AlphaPipeline 端到端 core/pipeline.py](#十一第-9-站alphapipeline-端到端-corepipelinepy)
- [十二、第 10 站：测试 tests/](#十二第-10-站测试-tests)
- [十三、核心概念深入](#十三核心概念深入)
- [十四、设计模式总结](#十四设计模式总结)
- [十五、数据流全景图](#十五数据流全景图)
- [十六、延伸练习](#十六延伸练习)

---

## 一、Phase 2b 整体结构

```
alpha_model/                     # Phase 2b 模型策略
├── __init__.py                  # 模块文档
├── config.py                    # 集中配置（路径、默认参数、年化常数）
├── utils.py                     # 辅助函数（load_price_panel）
│
├── core/                        # 核心协议与管道
│   ├── types.py                 # AlphaModel Protocol + 配置数据类
│   └── pipeline.py              # AlphaPipeline: 一键管道
│
├── preprocessing/               # 预处理
│   ├── alignment.py             # 多频率因子对齐
│   ├── transform.py             # 标准化工具箱 + 特征矩阵构建
│   └── selection.py             # 因子筛选（threshold / top_k / 族级）
│
├── training/                    # 训练框架
│   ├── splitter.py              # 时序切分器（Expanding / Rolling + embargo）
│   ├── walk_forward.py          # Walk-Forward 引擎
│   └── trainer.py               # 训练调度器（一站式接口）
│
├── signal/                      # 信号生成
│   ├── generator.py             # 预测值 → 标准化信号
│   └── smoother.py              # EMA 信号平滑
│
├── portfolio/                   # 组合构建
│   ├── beta.py                  # 滚动 beta 估计
│   ├── covariance.py            # Ledoit-Wolf 协方差矩阵估计
│   ├── constraints.py           # cvxpy 约束生成器
│   ├── risk_budget.py           # 波动率目标 (Vol Targeting)
│   └── constructor.py           # 凸优化组合构建器
│
├── backtest/                    # 回测
│   ├── vectorized.py            # 向量化回测 + 市场冲击模型
│   └── performance.py           # 绩效指标 + BacktestResult
│
├── store/                       # 持久化
│   ├── signal_store.py          # 信号/权重/元数据存储
│   └── model_store.py           # 模型对象持久化
│
├── models/                      # 参考模型实现（示例，非核心架构）
│   ├── linear_models.py         # SklearnModelWrapper（Ridge/Lasso/ElasticNet）
│   ├── tree_models.py           # LGBMModelWrapper / XGBModelWrapper
│   └── torch_base.py            # TorchModelBase（PyTorch 基类）
│
└── tests/                       # 单元测试
    ├── test_types.py            # 协议 + 配置类验证
    ├── test_preprocessing.py    # 对齐 + 标准化 + 特征矩阵 + 因子筛选
    ├── test_training.py         # 切分器 + Walk-Forward + Trainer
    ├── test_signal.py           # 信号生成 + EMA 平滑
    ├── test_portfolio.py        # beta + 协方差 + 约束 + 组合构建 + vol targeting
    ├── test_backtest.py         # 绩效指标 + 向量化回测 + 市场冲击
    ├── test_store.py            # SignalStore + ModelStore
    ├── test_models.py           # 参考模型封装
    └── test_pipeline.py         # AlphaPipeline 端到端
```

**依赖方向（只能向下依赖，不能向上）：**

```
pipeline.py → training/ + signal/ + portfolio/ + backtest/ + store/
                   ↓           ↓          ↓           ↓
              preprocessing/ → core/types.py ← config.py
                   ↓
         factor_research.store (Phase 2a)
         data_infra.data (Phase 1)
```

**与其他阶段的接口：**

```
Phase 2a (因子研究)           Phase 2b (模型策略)           Phase 3 (实盘)
FactorStore.load()  ──→  preprocessing/  ──→  ...  ──→  SignalStore  ──→  ...
                         (alignment,                    (weights.parquet)
                          transform,
                          selection)
```

---

## 二、建议学习路线

```
第1站 alpha_model/core/types.py          ← 理解协议和数据类
  ↓
第2站 alpha_model/preprocessing/ (3个文件) ← 理解数据怎么变成特征矩阵
  ↓
第3站 alpha_model/training/ (3个文件)      ← 理解 Walk-Forward 如何防止过拟合
  ↓
第4站 alpha_model/signal/ (2个文件)        ← 理解预测值怎么变成信号
  ↓
第5站 alpha_model/portfolio/ (5个文件)     ← 理解信号怎么变成目标权重（凸优化）
  ↓
第6站 alpha_model/backtest/ (2个文件)      ← 理解权重怎么计算 P&L
  ↓
第7站 alpha_model/store/ (2个文件)         ← 理解策略输出怎么持久化
  ↓
第8站 alpha_model/models/ (3个文件)        ← 理解参考模型封装（不是核心架构）
  ↓
第9站 alpha_model/core/pipeline.py       ← 理解如何串联全链路
  ↓
第10站 alpha_model/tests/ (8个文件)       ← 理解如何测试、学习 mock 技巧
```

---

## 三、第 1 站：核心类型 core/types.py

### 文件：`alpha_model/core/types.py`

这是整个 Phase 2b 的**地基**。所有模块都引用这里定义的协议和数据类。

**核心知识点：**

1. **Protocol（协议）vs ABC（抽象基类）**

Python 3.8+ 引入的 `typing.Protocol` 实现了**结构化子类型**（structural subtyping），也叫"鸭子类型的静态版本"。只要一个类实现了协议要求的方法，无需继承即可满足协议。

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class AlphaModel(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray | pd.Series: ...
```

为什么用 Protocol 而不用 ABC？因为 sklearn 原生模型已有 `fit/predict` 方法，无需继承任何基类即可直接使用：

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)
isinstance(model, AlphaModel)  # True! 天然满足协议
```

2. **`@runtime_checkable` 装饰器**

让 `isinstance(obj, AlphaModel)` 在运行时可用。没有此装饰器，`isinstance` 检查会抛 `TypeError`。

3. **AlphaModel 协议的方法分层**

```
必须实现（核心）:
    fit(X, y, **kwargs)     — 训练模型
    predict(X)              — 生成预测值

可选实现（增强）:
    save_model(path)        — 保存模型到指定目录
    load_model(path)        — 从指定目录加载模型
    get_feature_importance() — 返回因子重要性
    get_params()            — 返回模型参数
```

4. **TrainMode 和 WalkForwardMode 枚举**

```python
class TrainMode(Enum):
    POOLED = "pooled"           # 所有标的堆叠为一个训练集
    PER_SYMBOL = "per_symbol"   # 每个标的独立训练

class WalkForwardMode(Enum):
    EXPANDING = "expanding"     # 训练窗口起点固定，终点前进
    ROLLING = "rolling"         # 训练窗口固定大小，整体滑动
```

5. **TrainConfig 数据类**

```python
@dataclass
class TrainConfig:
    train_mode: TrainMode = TrainMode.POOLED
    wf_mode: WalkForwardMode = WalkForwardMode.EXPANDING
    target_horizon: int = 10        # 前瞻窗口（bar 数）
    train_periods: int = 5000       # 训练窗口长度
    test_periods: int = 1000        # 测试窗口长度
    purge_periods: int = 60         # 隔离期（必须 >= target_horizon）
```

`__post_init__` 中有严格的参数校验：
- `target_horizon >= 1`
- `purge_periods >= target_horizon`（防止 forward return 标签泄漏）

6. **PortfolioConstraints 数据类**

```python
@dataclass
class PortfolioConstraints:
    max_weight: float = 0.4         # 单标的最大绝对权重
    dollar_neutral: bool = True     # 多空等金额 Σw_i = 0
    beta_neutral: bool = False      # 对市场方向中性 β'w = 0
    beta_lookback: int = 60         # beta 估计窗口（天数）
    vol_target: float | None = None # 年化波动率目标
    vol_lookback: int = 60          # 波动率估计窗口
    leverage_cap: float = 2.0       # 最大杠杆 Σ|w_i|
    risk_aversion: float = 1.0      # 风险厌恶系数 λ
    turnover_penalty: float = 0.01  # 换手率惩罚系数 γ
```

三层约束的数学表达：
```
第一层: |w_i| ≤ max_weight        （防止单标的过度集中）
第二层: Σ w_i = 0                 （多空对冲）
第三层: β'w = 0                   （市场中性）
第四层: Σ|w_i| ≤ leverage_cap     （杠杆上限）
```

7. **ModelMeta 元数据类**

记录策略的完整配置，确保可复现性。提供 `to_dict()` 和 `from_dict()` 实现 JSON 序列化/反序列化往返。

**阅读重点：**

| 行范围 | 内容 | 学什么 |
|--------|------|--------|
| 33-73 | AlphaModel Protocol | Protocol 模式、`@runtime_checkable`、方法签名设计 |
| 112-121 | TrainMode / WalkForwardMode | Enum 类型的使用 |
| 128-176 | TrainConfig | dataclass + `__post_init__` 验证 |
| 178-232 | PortfolioConstraints | 三层约束的参数设计 |
| 235-328 | ModelMeta | 序列化/反序列化往返、`from_dict` 的 `get` 默认值技巧 |

**动手练习：**

```python
from alpha_model.core.types import *

# 1. 验证 sklearn 原生模型天然满足协议
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)
print(isinstance(model, AlphaModel))  # → True

# 2. 验证不完整的类不满足协议
class BadModel:
    def fit(self, X, y): pass
    # 没有 predict 方法
print(isinstance(BadModel(), AlphaModel))  # → False

# 3. 创建配置并尝试违反约束
try:
    TrainConfig(target_horizon=10, purge_periods=5)  # purge < horizon → 报错
except ValueError as e:
    print(f"预期的错误: {e}")

# 4. ModelMeta 序列化往返
meta = ModelMeta(
    name="test", factor_names=["f1", "f2"],
    target_horizon=10,
    train_config=TrainConfig(),
    constraints=PortfolioConstraints(),
)
d = meta.to_dict()
restored = ModelMeta.from_dict(d)
print(restored.name == meta.name)  # → True
```

**思考题：**
1. 为什么 `AlphaModel` 使用 Protocol 而不是 ABC？（答：让 sklearn 原生模型免继承直接使用）
2. 为什么 `purge_periods` 必须 >= `target_horizon`？（答：forward return 标签窗口会延伸到未来 h 个 bar，purge 必须覆盖这个窗口以防止标签泄漏）
3. `PortfolioConstraints` 中 `risk_aversion` 和 `turnover_penalty` 为什么允许为 0？（答：risk_aversion=0 意味着纯 alpha 最大化，turnover_penalty=0 意味着不惩罚换手）

---

## 四、第 2 站：预处理 preprocessing/

### 4.1 `alpha_model/preprocessing/alignment.py` — 多频率因子对齐

**核心知识点：**
- **不同频率因子的统一**：orderbook 因子可能是 10 秒频率，kline 因子是 1 分钟，需要对齐到统一时间网格
- **频率推断**：通过计算相邻时间戳的中位数差值自动推断
- **对齐策略**：取所有因子中最低频率（最大间隔）作为目标频率，高频向低频重采样
- **前向填充（ffill）**：高频因子对齐到低频网格时，用最近一个有效值填充
- **max_gap 控制**：限制最大前向填充步数，防止过度填充

```python
from alpha_model.preprocessing.alignment import align_factor_panels

# 有一个 1min 因子和一个 5min 因子
aligned = align_factor_panels(
    {"fast_factor": panel_1min, "slow_factor": panel_5min},
    fill_method="ffill",  # 前向填充
    max_gap=3,            # 最多填充 3 步
)
# 两个面板的时间索引现在完全相同（5min 频率）
```

**阅读重点：**

| 行范围 | 内容 | 学什么 |
|--------|------|--------|
| 21-38 | `_infer_freq` | 通过中位数推断频率（鲁棒于缺失行） |
| 70-82 | 自动推断 vs 手动指定 | `to_offset` 将字符串转为频率对象 |
| 84-106 | 时间网格构建 | 取交集 + `pd.date_range` 生成统一索引 |
| 112-136 | 逐面板对齐 | `reindex` + `ffill(limit=max_gap)` |

### 4.2 `alpha_model/preprocessing/transform.py` — 标准化工具箱 + 特征矩阵构建

**核心知识点：**

1. **标准化是工具箱，不是管道步骤**

这是一个重要的设计决策。标准化不硬编码在管道中，而是作为独立工具函数提供：
- 树模型（LightGBM/XGBoost）对单调变换不敏感，不需要标准化
- 线性模型需要标准化，但 expanding/rolling/截面 z-score 各有优劣
- 选择哪种标准化是"研究决策"，不应由框架替用户做选择

2. **四种标准化方式**

```python
# 时序标准化（逐列独立，无截面交互）
expanding_zscore(panel, min_periods=252)   # z = (x - expanding_mean) / expanding_std
rolling_zscore(panel, window=1000)         # z = (x - rolling_mean) / rolling_std

# 截面标准化（每行独立，无时序交互）
cross_sectional_zscore(panel)              # 每行做 z-score
cross_sectional_rank(panel)                # 每行做百分位排名 [0, 1]
```

3. **无未来信息保证**
- expanding/rolling：只看当前及之前的数据
- 截面标准化：在每个时刻独立计算，不涉及时序方向

4. **Winsorize 去极值**

```python
winsorize(panel, sigma=3.0, method="expanding")
# 将超过 ±3σ 的值截断到边界值
# method 可选: "expanding", "rolling", "cross_sectional"
```

5. **特征矩阵构建 `build_feature_matrix`**

将多个因子面板合并为模型输入的特征矩阵。两种模式的输出格式不同：

```python
# Pooled 模式: 所有 symbol 堆叠
X = build_feature_matrix(factor_panels, symbols, TrainMode.POOLED)
# X.index = MultiIndex(timestamp, symbol)
# X.columns = factor_names

# Per-Symbol 模式: 每个 symbol 独立
X = build_feature_matrix(factor_panels, symbols, TrainMode.PER_SYMBOL)
# X = {"BTC/USDT": DataFrame, "ETH/USDT": DataFrame, ...}
```

6. **`build_pooled_target` 构建 Pooled 目标变量**

将 `fwd_returns (timestamp x symbol)` 重塑为与 X 同样的 MultiIndex 格式：

```python
y = build_pooled_target(X, fwd_returns, symbols)
# y.index = MultiIndex(timestamp, symbol)，与 X 完全对齐
```

**阅读重点：**

| 行范围 | 内容 | 学什么 |
|--------|------|--------|
| 40-65 | `expanding_zscore` | `panel.expanding(min_periods).mean/std`，std=0 返回 NaN |
| 68-92 | `rolling_zscore` | expanding vs rolling 的权衡 |
| 99-138 | 截面标准化 | `panel.mean(axis=1)` 逐行计算、`sub/div` 广播 |
| 145-201 | `winsorize` | 三种方式的去极值，注意 `np.tile` 的形状对齐技巧 |
| 208-296 | `build_feature_matrix` | Pooled 堆叠逻辑、MultiIndex 构建、`set_index(append=True)` |
| 302-337 | `build_pooled_target` | 从面板格式到 MultiIndex Series 的转换 |

### 4.3 `alpha_model/preprocessing/selection.py` — 因子筛选

**核心知识点：**

因子筛选是**可选步骤**（树模型可跳过），支持三种模式：

1. **threshold 模式（三步过滤）：**
   - Step 1: 按 IC/MI/单调性评分，过滤低于阈值的因子
   - Step 2: VIF 过滤共线性因子（逐步移除 VIF 最高的）
   - Step 3: 贪心增量 IC 筛选（逐个加入，保留增量贡献够大的）

2. **top_k 模式：**
   - 对每个因子计算多维评分（IC、MI、单调性）
   - 归一化后加权综合评分
   - 排序取前 k 个

3. **族级筛选 `select_from_families`：**
   - 对每个因子族，选出最优变体
   - 跨族执行 `select_factors()` 进一步过滤

```python
from alpha_model.preprocessing.selection import select_factors, select_from_families

# threshold 模式
selected = select_factors(
    factor_panels, price_panel,
    mode="threshold", metric="ic",
    min_ic=0.02, max_vif=10.0,
    min_incremental_ic=0.005,
    min_factors=3,
)

# top_k 模式
selected = select_factors(
    factor_panels, price_panel,
    mode="top_k", top_k=5,
    score_weights={"ic": 0.5, "mi": 0.3, "monotonicity": 0.2},
)

# 族级筛选
selected = select_from_families(
    family_names=["momentum", "volatility"],
    price_panel=price_panel,
    family_select_metric="ic_mean",
)
```

**阅读重点：**

| 行范围 | 内容 | 学什么 |
|--------|------|--------|
| 42-81 | `_score_factor` | 复用 Phase 2a 的 `ic_analysis`, `nonlinear_analysis`, `quantile_backtest` |
| 88-215 | `_threshold_select` | VIF 逐步删除（while 循环）、贪心增量 IC |
| 222-274 | `_topk_select` | 多维评分归一化、加权综合 |
| 346-416 | `select_from_families` | 族内选优 + 跨族过滤的两层架构 |

---

## 五、第 3 站：训练框架 training/

### 5.1 `alpha_model/training/splitter.py` — 时序切分器

**核心知识点：**

1. **为什么不用 sklearn 的 TimeSeriesSplit？**

sklearn 的 `TimeSeriesSplit` 没有 **embargo period**（隔离期）。在金融时序中，训练集末尾的 forward return 标签与测试集开头的数据存在重叠（标签窗口延伸到了测试集），会导致**前瞻偏差**。

2. **Embargo period 的计算**

```python
embargo_periods = max(target_horizon, max_factor_lookback)
```

两个泄漏源都要覆盖：
- `target_horizon`：forward return 标签的窗口长度
- `max_factor_lookback`：因子计算中使用的最大 rolling window

3. **Fold 数据类**

```python
@dataclass
class Fold:
    fold_id: int
    train_start: int       # 训练集起始索引（含）
    train_end: int         # 训练集结束索引（不含）
    test_start: int        # 测试集起始索引（含）
    test_end: int          # 测试集结束索引（不含）
```

4. **两种模式的切分逻辑**

```
Expanding 模式（训练数据越来越多）:
Fold 0: |===TRAIN===|--embargo--|==TEST==|
Fold 1: |=========TRAIN=========|--embargo--|==TEST==|
Fold 2: |===============TRAIN===============|--embargo--|==TEST==|

Rolling 模式（训练窗口固定大小）:
Fold 0: |===TRAIN===|--embargo--|==TEST==|
Fold 1:       |===TRAIN===|--embargo--|==TEST==|
Fold 2:             |===TRAIN===|--embargo--|==TEST==|
```

```python
from alpha_model.training.splitter import TimeSeriesSplitter
from alpha_model.core.types import WalkForwardMode

splitter = TimeSeriesSplitter(
    train_periods=3000,
    test_periods=1000,
    target_horizon=10,
    max_factor_lookback=60,    # embargo = max(10, 60) = 60
    mode=WalkForwardMode.EXPANDING,
)

folds = splitter.split(n_samples=10000)
for fold in folds:
    print(f"Fold {fold.fold_id}: "
          f"train [{fold.train_start}:{fold.train_end}] "
          f"test [{fold.test_start}:{fold.test_end}] "
          f"gap={fold.test_start - fold.train_end}")
```

**阅读重点：**

| 行范围 | 内容 | 学什么 |
|--------|------|--------|
| 27-51 | Fold 数据类 | `@property` 计算属性 |
| 73-111 | `__init__` | 参数校验 + embargo 自动计算 |
| 113-210 | `split` | Expanding vs Rolling 的循环逻辑、最后一个 fold 的截断处理 |

### 5.2 `alpha_model/training/walk_forward.py` — Walk-Forward 引擎

**核心知识点：**

1. **WalkForwardResult 结果数据类**

```python
@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame         # 样本外预测面板 (timestamp × symbol)
    fold_metrics: list[dict]          # 每个 fold 的 IC 等评估指标
    feature_importance: pd.DataFrame  # 因子重要性（各 fold 平均）
    train_config: TrainConfig | None
```

2. **Pooled 模式的流程**

```python
# 按 timestamp 分组确定唯一时间戳数量（用于切分）
timestamps = X.index.get_level_values("timestamp").unique().sort_values()
folds = splitter.split(len(timestamps))

for fold in folds:
    # 用时间戳索引取切分
    train_ts = timestamps[fold.train_start:fold.train_end]
    test_ts = timestamps[fold.test_start:fold.test_end]

    # 布尔掩码选择行
    train_mask = X.index.get_level_values("timestamp").isin(train_ts)
    X_train = X.loc[train_mask]
    # ...

    # 每个 fold 使用模型的深拷贝，避免状态污染
    fold_model = copy.deepcopy(self.model)
    fold_model.fit(X_train, y_train)
    preds = fold_model.predict(X_test)
```

3. **关键设计要点**
- `copy.deepcopy(model)`：每个 fold 独立模型副本，避免状态互相污染
- NaN 行在训练时删除（`X_train.notna().all(axis=1) & y_train.notna()`）
- 因子重要性支持安全获取（`hasattr` + `try/except`）

4. **IC 计算方式**
- Pooled 模式：按 timestamp 分组计算截面 IC，取均值
- Per-Symbol 模式：直接计算 Spearman IC

**阅读重点：**

| 行范围 | 内容 | 学什么 |
|--------|------|--------|
| 111-217 | `_run_pooled` | MultiIndex 时间戳切分技巧、`deepcopy` 隔离 |
| 219-310 | `_run_per_symbol` | 每个 symbol 独立 Walk-Forward |
| 312-349 | `_compute_fold_ic` | 截面 IC 均值计算 |
| 351-379 | `_get_importance` / `_average_importance` | 安全获取 + 多 fold 平均 |

### 5.3 `alpha_model/training/trainer.py` — 训练调度器

一站式接口：从因子面板和价格面板出发，自动完成特征构建、目标变量计算、Walk-Forward 训练。

```python
from alpha_model.training.trainer import Trainer
from alpha_model.core.types import TrainConfig

trainer = Trainer(
    model=Ridge(alpha=1.0),
    train_config=TrainConfig(
        train_periods=3000, test_periods=1000,
        target_horizon=10,
    ),
    max_factor_lookback=60,
)
result = trainer.run(factor_panels, price_panel, symbols)
```

`Trainer.run()` 内部做三件事：
1. `build_feature_matrix(factor_panels, symbols, mode)` → X
2. `compute_forward_returns_panel(price_panel, horizon)` → y
3. `WalkForwardEngine(model, splitter, mode).run(X, y)` → result

**阅读重点：**

| 行范围 | 内容 | 学什么 |
|--------|------|--------|
| 60-141 | `run` | 自动推断 symbols、Pooled vs Per-Symbol 的 y 构建差异 |

---

## 六、第 4 站：信号生成 signal/

### 6.1 `alpha_model/signal/generator.py` — 预测值到标准化信号

**核心知识点：**

1. **默认不截断极端值**

这是一个有意的设计决策：模型输出的极端预测值可能正是信号强烈的证明。截断极端值会削弱策略的表达能力。真正的风控由 portfolio 层的约束（`max_weight`、`leverage_cap`）来实现。

2. **两种标准化方式**

```python
from alpha_model.signal.generator import generate_signal

# 截面 z-score（默认）
signal = generate_signal(predictions, method="cross_sectional_zscore")
# 每行: z = (x - mean) / std，均值 → 0，标准差 → 1

# 截面百分位排名
signal = generate_signal(predictions, method="cross_sectional_rank")
# 每行: rank ∈ [0, 1]，无分布假设
```

3. **可选截断**

```python
# 截面 z-score + 3σ 截断
signal = generate_signal(predictions, method="cross_sectional_zscore", clip_sigma=3.0)
```

### 6.2 `alpha_model/signal/smoother.py` — EMA 信号平滑

```python
from alpha_model.signal.smoother import ema_smooth

# halflife 控制衰减速度
smoothed = ema_smooth(signal, halflife=5)
# halflife 越小 → 响应越快 → 换手越高
# halflife 越大 → 越平滑 → 信号越滞后
```

底层使用 `pandas.DataFrame.ewm(halflife=halflife, min_periods=1).mean()`。

**阅读重点：**

| 文件 | 行范围 | 内容 | 学什么 |
|------|--------|------|--------|
| generator.py | 21-69 | `generate_signal` | 截面 z-score、`replace(0, np.nan)` 处理零方差 |
| smoother.py | 12-35 | `ema_smooth` | `ewm` 参数含义、halflife 与 alpha 的关系 |

---

## 七、第 5 站：组合构建 portfolio/

### 7.1 `alpha_model/portfolio/beta.py` — 滚动 Beta 估计

**核心知识点：**

```
beta_i = Cov(r_i, r_market) / Var(r_market)
```

使用 pandas 的 `rolling().cov()` 和 `rolling().var()` 实现：

```python
from alpha_model.portfolio.beta import rolling_beta

betas = rolling_beta(returns_panel, market_symbol="BTC/USDT", lookback=60)
# betas["BTC/USDT"] 恒为 1.0（市场自身）
# betas["ETH/USDT"] ≈ 0.8~1.2（与市场正相关）
```

### 7.2 `alpha_model/portfolio/covariance.py` — 协方差矩阵估计

**核心知识点：**

1. **Ledoit-Wolf Shrinkage**

样本协方差矩阵在 T/N（样本数/资产数）比较小时估计误差极大。Ledoit-Wolf 收缩将样本协方差向一个结构化目标"收缩"：

```
Σ_shrunk = δ × F + (1-δ) × S
F = 结构化目标（对角矩阵，只保留方差）
S = 样本协方差矩阵
δ = 最优收缩强度（数据驱动，自动计算）
```

2. **三种估计方法**

```python
from alpha_model.portfolio.covariance import estimate_covariance

# Ledoit-Wolf（推荐）
cov = estimate_covariance(returns_panel, lookback=60, method="ledoit_wolf")

# 样本协方差（仅用于对比）
cov = estimate_covariance(returns_panel, lookback=60, method="sample")

# 指数加权协方差
cov = estimate_covariance(returns_panel, lookback=60, method="exponential")
```

3. **滚动协方差 `rolling_covariance`**

逐时刻计算协方差矩阵序列，返回 `{timestamp: np.ndarray}` 字典。

**阅读重点：**

| 行范围 | 内容 | 学什么 |
|--------|------|--------|
| 35-86 | `estimate_covariance` | `LedoitWolf().fit(data.values).covariance_` 的用法 |
| 89-142 | `rolling_covariance` | 滚动窗口估计、dropna 后数据质量预警 |

### 7.3 `alpha_model/portfolio/constraints.py` — cvxpy 约束生成器

**核心知识点：**

所有约束被建模为 cvxpy 约束表达式，**联合传入优化器求解**。不再有顺序应用的问题——所有约束同时满足。

```python
import cvxpy as cp
from alpha_model.portfolio.constraints import build_constraints

w = cp.Variable(5)  # 5 个标的的权重
config = PortfolioConstraints(
    max_weight=0.4,
    dollar_neutral=True,
    beta_neutral=True,
    leverage_cap=2.0,
)
beta = np.array([1.0, 0.9, 1.2, 0.8, 1.1])

constraints = build_constraints(w, config, beta=beta)
# 返回的 constraints 列表:
# [cp.abs(w) <= 0.4,           第一层: 仓位上限
#  cp.sum(w) == 0,             第二层: dollar-neutral
#  beta @ w == 0,              第三层: beta-neutral
#  cp.norm(w, 1) <= 2.0]       第四层: 杠杆上限
```

**阅读重点：**

| 行范围 | 内容 | 学什么 |
|--------|------|--------|
| 40-78 | `build_constraints` | cvxpy 约束表达式写法、`TYPE_CHECKING` 延迟导入 |

### 7.4 `alpha_model/portfolio/risk_budget.py` — 波动率目标 (Vol Targeting)

**核心知识点：**

动态调整组合杠杆，使组合的预期年化波动率稳定在目标水平。

```
realized_vol = 组合最近 N 行的年化波动率
scale_factor = target_vol / realized_vol
adjusted_weights = original_weights * scale_factor

波动率高于目标 → 缩小仓位
波动率低于目标 → 放大仓位
```

受 `leverage_cap` 约束：缩放后 `Σ|w_i|` 不超过杠杆上限。

```python
from alpha_model.portfolio.risk_budget import apply_vol_target

adjusted = apply_vol_target(
    weights, price_panel,
    vol_target=0.15,     # 15% 年化波动率
    lookback=60,
    leverage_cap=2.0,
)
```

**阅读重点：**

| 行范围 | 内容 | 学什么 |
|--------|------|--------|
| 25-84 | `apply_vol_target` | 滞后权重计算组合收益率、`MINUTES_PER_YEAR` 年化、杠杆上限约束 |

### 7.5 `alpha_model/portfolio/constructor.py` — 凸优化组合构建器（核心）

**核心知识点：**

1. **Mean-Variance 优化的 QP 形式**

```
minimize    w'Σw - λα'w + γ||w - w_prev||₁
subject to  |w_i| ≤ max_weight
            Σw_i = 0                      (dollar-neutral, 可选)
            β'w = 0                       (beta-neutral, 可选)
            ||w||₁ ≤ leverage_cap

其中:
    w     = 权重向量（决策变量）
    Σ     = 协方差矩阵（Ledoit-Wolf 估计）
    α     = alpha 向量（信号值）
    λ     = 风险厌恶系数
    γ     = 换手率惩罚系数
    w_prev = 上一期权重
```

2. **逐期求解**

`PortfolioConstructor.construct()` 对信号面板的每个时间戳逐期求解 QP 问题，通过 `prev_w` 追踪上一期权重用于换手率惩罚：

```python
from alpha_model.portfolio.constructor import PortfolioConstructor

constructor = PortfolioConstructor(
    PortfolioConstraints(
        dollar_neutral=True,
        max_weight=0.4,
        risk_aversion=1.0,
        turnover_penalty=0.01,
    )
)
weights = constructor.construct(signal, price_panel)
```

3. **协方差矩阵的正半定保证**

优化求解前，检查协方差矩阵的最小特征值。如果非正半定，加微小正则项：

```python
min_eig = np.linalg.eigvalsh(cov_matrix).min()
if min_eig < 0:
    cov_matrix += (-min_eig + 1e-8) * np.eye(n)
```

4. **Infeasible 退化处理**

当约束矛盾导致优化问题无解（infeasible）时，退化为安全权重（dollar-neutral → 全零，否则等权）。

5. **Vol Targeting 事后缩放**

优化求解得到权重后，如果配置了 `vol_target`，再调用 `apply_vol_target` 做事后缩放。

**阅读重点：**

| 行范围 | 内容 | 学什么 |
|--------|------|--------|
| 56-170 | `construct` | 逐期求解循环、NaN 信号处理、协方差估计的无前瞻保证 |
| 172-232 | `_solve_single_period` | cvxpy 建模全过程、正半定修正、OSQP 求解器 |

**动手练习：**

```python
import cvxpy as cp
import numpy as np

# 手写一个最简单的 Mean-Variance 优化
n = 3
alpha = np.array([0.5, -0.3, 0.1])   # alpha 信号
cov = np.eye(3) * 0.01                # 简化为对角协方差

w = cp.Variable(n)
risk = cp.quad_form(w, cov)            # w'Σw
ret = alpha @ w                        # α'w
objective = cp.Minimize(risk - ret)    # 风险 - 收益

constraints = [
    cp.abs(w) <= 0.4,                  # 仓位上限
    cp.sum(w) == 0,                    # dollar-neutral
    cp.norm(w, 1) <= 2.0,             # 杠杆上限
]

prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.OSQP)
print(f"最优权重: {w.value}")
print(f"权重之和: {w.value.sum():.6f}")  # 应接近 0
```

---

## 八、第 6 站：向量化回测 backtest/

### 8.1 `alpha_model/backtest/vectorized.py` — 向量化回测

**核心知识点：**

1. **P&L 计算公式**

```
portfolio_return_t = Σ(w_{i,t-1} × r_{i,t})
```

用上一期（t-1）的权重乘以本期（t）的收益率。代码中通过 `weights.shift(1)` 实现滞后。

2. **两层交易成本模型**

```
第一层: 固定手续费
    fee = turnover × fee_rate
    turnover = Σ|Δw_i|（权重变化的绝对值之和）

第二层: 市场冲击（Square-root model）
    impact_i = impact_coeff × σ_i × √(|ΔV_i| / ADV_i)
    其中 ΔV_i = |Δw_i| × portfolio_value
```

3. **Square-root 市场冲击模型的直觉**

```python
from alpha_model.backtest.vectorized import estimate_market_impact

impact = estimate_market_impact(
    delta_weights,    # 权重变化面板
    adv_panel,        # 日均成交量面板 (USDT)
    volatility_panel, # 滚动波动率面板
    portfolio_value,  # 组合总资金
    impact_coeff=0.1, # 冲击系数
)
```

为什么用 `sqrt`？因为市场冲击与交易规模不是线性关系——小额交易几乎无冲击，大额交易冲击迅速增大但最终趋于饱和。`sqrt` 是实证中最常用的近似。

4. **向量化回测的局限**

向量化回测假设在 t-1 时刻设定的权重在 t 时刻完美执行。实际中有：
- 下单延迟
- 部分成交
- 滑点
这些需要 Phase 3 的事件驱动回测来模拟。

```python
from alpha_model.backtest.vectorized import vectorized_backtest

result = vectorized_backtest(
    weights, price_panel,
    fee_rate=0.0004,         # Binance taker 0.04%
    impact_coeff=0.1,
    adv_panel=None,          # None → 纯手续费模型
    portfolio_value=10000,
)
print(result.summary())
```

### 8.2 `alpha_model/backtest/performance.py` — 绩效指标

**核心知识点：**

复用 Phase 2a 的基础指标，新增策略级指标：

```python
# Phase 2a 复用
from factor_research.evaluation.metrics import (
    sharpe_ratio, max_drawdown, annualize_return,
    annualize_volatility, cumulative_returns,
)

# Phase 2b 新增
sortino_ratio(returns)          # 只惩罚下行波动率
calmar_ratio(returns)           # 年化收益 / |最大回撤|
max_drawdown_duration(returns)  # 最长回撤持续期（bar 数）
```

**Sortino ratio 的意义：**
Sharpe ratio 对正负波动率一视同仁，但投资者通常不介意"赚多了"。Sortino 只用下行波动率做分母，更符合直觉。

**BacktestResult 数据类：**

```python
@dataclass
class BacktestResult:
    equity_curve: pd.Series         # 净值曲线 (1.0 起始)
    returns: pd.Series              # 逐期净收益率
    turnover: pd.Series             # 逐期换手率
    weights_history: pd.DataFrame   # 权重历史
    gross_returns: pd.Series | None # 毛收益
    total_cost: float               # 总交易成本

    def summary(self) -> dict:
        # 返回 12 个绩效指标的字典
```

`summary()` 返回的完整指标列表：
```
annual_return        — 年化收益率
annual_volatility    — 年化波动率
sharpe_ratio         — 夏普比率
sortino_ratio        — Sortino 比率
calmar_ratio         — Calmar 比率
max_drawdown         — 最大回撤
max_drawdown_duration — 最长回撤持续期
avg_turnover         — 平均换手率
total_cost           — 总交易成本
win_rate             — 胜率
n_periods            — 总期数
total_return         — 总收益率
```

**阅读重点：**

| 文件 | 行范围 | 内容 | 学什么 |
|------|--------|------|--------|
| vectorized.py | 42-74 | `estimate_market_impact` | sqrt-model 公式实现 |
| vectorized.py | 77-153 | `vectorized_backtest` | `shift(1)` 滞后权重、`diff()` 计算换手率 |
| performance.py | 37-63 | `sortino_ratio` | 下行波动率的计算方式 |
| performance.py | 88-111 | `max_drawdown_duration` | `cummax` 求高水位、连续回撤计数 |

---

## 九、第 7 站：持久化 store/

### 9.1 `alpha_model/store/signal_store.py` — 策略输出持久化

**核心知识点：**

SignalStore 是 Phase 2b → Phase 3 的**唯一输出接口**。

存储结构：
```
db/signals/{strategy_name}/
├── weights.parquet        # 目标权重面板
├── signals.parquet        # 原始信号面板（调试用）
├── meta.json              # ModelMeta 序列化
└── performance.json       # BacktestResult.summary()
```

1. **原子写入**（与 Phase 1 一致）：先写 `.tmp` 目录，完成后 `os.rename` 原子替换，防止写入中断导致数据损坏。

2. **Parquet 格式**：带 `timestamp` 列，保持与 FactorStore 风格一致。

```python
from alpha_model.store.signal_store import SignalStore

store = SignalStore()

# 保存
store.save(
    strategy_name="ridge_momentum_v1",
    weights=weights_df,
    signals=signal_df,
    meta=model_meta,
    performance=backtest_result.summary(),
)

# 加载
weights = store.load_weights("ridge_momentum_v1")
meta = store.load_meta("ridge_momentum_v1")
perf = store.load_performance("ridge_momentum_v1")

# 列出所有策略
print(store.list_strategies())
```

### 9.2 `alpha_model/store/model_store.py` — 模型持久化

**核心知识点：**

存储结构：
```
db/models/{model_name}/
├── model/                 # 模型文件目录
│   └── model.joblib       # 或 model.pt / model.txt
├── meta.json              # ModelMeta
└── importance.json        # 因子重要性排名
```

1. **ModelStore 不假设模型的序列化格式**。它调用 `model.save_model(path)` 和 `model.load_model(path)`，由模型自行决定如何保存/加载。

2. **加载时需要 `model_factory`**：因为 ModelStore 不知道模型的类型，需要用户提供一个工厂函数创建空模型实例。

```python
from alpha_model.store.model_store import ModelStore

store = ModelStore()

# 保存
store.save("ridge_v1", model, meta, importance={"f1": 0.8, "f2": 0.2})

# 加载（需要提供工厂函数）
model, meta = store.load(
    "ridge_v1",
    model_factory=lambda: SklearnModelWrapper(Ridge(alpha=1.0)),
)

# 加载因子重要性
importance = store.load_importance("ridge_v1")
```

**阅读重点：**

| 文件 | 行范围 | 内容 | 学什么 |
|------|--------|------|--------|
| signal_store.py | 46-107 | `save` | 原子写入: `.tmp` 目录 → `os.rename` |
| signal_store.py | 160-175 | `_save_parquet` / `_load_parquet` | Parquet 格式约定 |
| model_store.py | 49-110 | `save` | `hasattr(model, "save_model")` 安全检查 |
| model_store.py | 112-150 | `load` | `model_factory()` 工厂模式 |

---

## 十、第 8 站：参考模型 models/

这些是**示例代码**，不是核心架构。用户完全可以不用这些封装，只要自己的模型实现了 `fit/predict` 即可。

### 10.1 `alpha_model/models/linear_models.py` — SklearnModelWrapper

封装 sklearn 的线性模型，补充 `save_model/load_model/get_feature_importance`：

```python
from alpha_model.models.linear_models import SklearnModelWrapper
from sklearn.linear_model import Ridge, Lasso, ElasticNet

model = SklearnModelWrapper(Ridge(alpha=1.0))
# 也可以: SklearnModelWrapper(Lasso(alpha=0.01))
# 也可以: SklearnModelWrapper(ElasticNet(alpha=0.1, l1_ratio=0.5))

model.fit(X, y)
preds = model.predict(X)
model.save_model(path)              # → model.joblib
importance = model.get_feature_importance()  # → |coef_|
```

### 10.2 `alpha_model/models/tree_models.py` — LGBMModelWrapper / XGBModelWrapper

```python
from alpha_model.models.tree_models import LGBMModelWrapper

model = LGBMModelWrapper(
    objective="regression",
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=100,
)
model.fit(X, y)
importance = model.get_feature_importance()  # → gain-based importance
model.save_model(path)  # → model.txt
```

XGBModelWrapper 类似，保存格式为 `model.json`。

注意：`lightgbm` 和 `xgboost` 是**可选依赖**。未安装时 import 模块不报错，但实例化时抛 `ImportError`。

### 10.3 `alpha_model/models/torch_base.py` — TorchModelBase

用户继承此基类，只需实现 `build_network()` 方法：

```python
from alpha_model.models.torch_base import TorchModelBase
import torch.nn as nn

class MyMLP(TorchModelBase):
    def build_network(self, n_features):
        return nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

model = MyMLP(
    val_ratio=0.2,     # 验证集比例（尾部按时间切分）
    patience=10,       # early stopping 容忍轮次
    max_epochs=100,
    batch_size=256,
    lr=1e-3,
    device="auto",     # 自动选 GPU/CPU
)
model.fit(X, y)
```

**`fit` 方法自动处理的完整流程：**
1. DataFrame → numpy float32
2. 删除 NaN 行
3. 按时间顺序拆分 train/val（尾部 `val_ratio` 为验证集）
4. Tensor → DataLoader
5. 训练循环（epoch, batch, gradient descent）
6. 验证集 early stopping
7. 加载 best checkpoint

**关键设计：验证集按时间顺序切分（不是随机切分），避免前瞻偏差。**

**阅读重点：**

| 文件 | 行范围 | 内容 | 学什么 |
|------|--------|------|--------|
| linear_models.py | 25-83 | `SklearnModelWrapper` | 通用封装模式、`joblib` 序列化 |
| tree_models.py | 19-91 | `LGBMModelWrapper` | `lgb.Dataset` + `lgb.train` 的用法 |
| torch_base.py | 89-177 | `fit` | 完整的 PyTorch 训练循环、early stopping |
| torch_base.py | 179-196 | `predict` | `eval()` + `no_grad()` + Tensor→numpy |

---

## 十一、第 9 站：AlphaPipeline 端到端 core/pipeline.py

### 文件：`alpha_model/core/pipeline.py`

AlphaPipeline 串联全链路，一键执行 8 个步骤：

```python
from sklearn.linear_model import Ridge
from alpha_model.core.pipeline import AlphaPipeline
from alpha_model.core.types import TrainConfig, PortfolioConstraints

pipeline = AlphaPipeline(
    model=Ridge(alpha=1.0),
    train_config=TrainConfig(
        train_periods=3000, test_periods=1000,
        target_horizon=10, purge_periods=60,
    ),
    constraints=PortfolioConstraints(
        dollar_neutral=True, max_weight=0.4,
        vol_target=0.15,
    ),
    factor_names=["momentum_10", "volatility_30", "mean_reversion_60"],
    signal_method="cross_sectional_zscore",
    max_factor_lookback=60,
)

# 执行
result = pipeline.run(price_panel, symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"])

# 查看绩效
print(result.summary())

# 保存
pipeline.save("ridge_momentum_v1")
```

**`run()` 的 8 个步骤：**

```
Step 1: 从 FactorStore 加载因子
        ├── 按 factor_names 直接加载
        └── 按 factor_families 族级筛选 → 自动选最优变体

Step 2: 因子筛选（可选, 配置了 selection_params 时执行）

Step 3: 因子对齐
        align_factor_panels(factor_panels)

Step 4: 构建特征矩阵 + 目标变量
        X = build_feature_matrix(...)
        y = compute_forward_returns_panel(...)

Step 5: Walk-Forward 训练
        TimeSeriesSplitter → WalkForwardEngine → predictions

Step 6: 信号生成
        generate_signal(predictions, method=...)

Step 7: 组合构建
        PortfolioConstructor(constraints).construct(signal, prices)

Step 8: 向量化回测
        vectorized_backtest(weights, prices)
```

**`save()` 方法：** 同时保存到 SignalStore（权重/信号/元数据/绩效）和 ModelStore（模型对象/重要性）。

**支持两种因子指定方式的组合使用：**

```python
pipeline = AlphaPipeline(
    model=Ridge(),
    train_config=TrainConfig(),
    constraints=PortfolioConstraints(),
    factor_names=["my_custom_factor"],    # 直接指定
    factor_families=["momentum"],          # 族级筛选
    # 两者可组合使用，结果合并
)
```

**阅读重点：**

| 行范围 | 内容 | 学什么 |
|--------|------|--------|
| 53-88 | `__init__` | 参数设计、至少指定一个因子来源 |
| 97-237 | `run` | 8 步管道编排、错误处理、日志记录 |
| 239-279 | `save` | SignalStore + ModelStore 双重保存 |

---

## 十二、第 10 站：测试 tests/

### 测试文件概览

| 文件 | 测试对象 | 关键技巧 |
|------|----------|----------|
| `test_types.py` | Protocol 检查 + 配置验证 | `isinstance(model, AlphaModel)` 协议运行时检查 |
| `test_preprocessing.py` | 对齐 + 标准化 + 特征矩阵 + 因子筛选 | `pd.testing.assert_frame_equal` 精确比较 |
| `test_training.py` | 切分器 + Walk-Forward + Trainer | `_SimpleModel` 测试替身 |
| `test_signal.py` | 信号生成 + EMA 平滑 | 统计性质验证（均值→0, 标准差→1） |
| `test_portfolio.py` | beta + 协方差 + 约束 + 组合构建 + vol target | cvxpy 约束测试、infeasible 退化 |
| `test_backtest.py` | 绩效指标 + 向量化回测 + 市场冲击 | 零权重→零收益、费率对比 |
| `test_store.py` | SignalStore + ModelStore | `tmp_path` fixture、往返一致性 |
| `test_models.py` | 参考模型封装 | `pytest.skip` 条件跳过、`tmp_path` 序列化往返 |
| `test_pipeline.py` | AlphaPipeline 端到端 | `monkeypatch` 重定向存储路径 |

### 核心测试模式

**1. 测试替身（Test Double）**

```python
class _SimpleModel:
    """简单均值模型用于测试"""
    def fit(self, X, y, **kwargs):
        self._mean = y.mean() if len(y) > 0 else 0.0
    def predict(self, X):
        return np.full(len(X), self._mean)
```

使用最简单的模型代替真实模型，隔离测试目标（Walk-Forward 引擎本身，而非模型效果）。

**2. monkeypatch 重定向存储路径**

```python
@pytest.fixture
def setup_stores(self, tmp_path, monkeypatch):
    # 重定向 FactorStore 路径
    monkeypatch.setattr(
        "factor_research.config.FACTOR_STORE_DIR", str(factor_dir),
    )
    monkeypatch.setattr(
        "factor_research.store.factor_store.FACTOR_STORE_DIR", str(factor_dir),
    )
    # 重定向 alpha_model 侧路径
    monkeypatch.setattr(
        "alpha_model.config.SIGNAL_STORE_DIR", tmp_path / "signals",
    )
    monkeypatch.setattr(
        "alpha_model.config.MODEL_STORE_DIR", tmp_path / "models",
    )
```

**关键点：** `monkeypatch.setattr` 必须同时 patch **定义处**和 **import 后的副本**两个位置，否则无参构造的类仍使用旧值。

**3. 可选依赖的条件跳过**

```python
@pytest.fixture(autouse=True)
def check_lgbm(self):
    try:
        import lightgbm
    except ImportError:
        pytest.skip("lightgbm 未安装")
```

**4. 不变量验证**

```python
# 训练集在测试集之前
assert fold.train_end <= fold.test_start

# embargo gap 足够大
gap = fold.test_start - fold.train_end
assert gap >= 30

# dollar-neutral: 权重之和接近 0
row_sums = weights.sum(axis=1).dropna()
assert row_sums.abs().max() < 0.05

# 仓位上限
assert weights.abs().max().max() < 0.3 + 0.01
```

**运行测试：**

```bash
# 运行 Phase 2b 全部测试
python -m pytest alpha_model/tests/ -v

# 运行单个测试文件
python -m pytest alpha_model/tests/test_training.py -v

# 运行特定测试类
python -m pytest alpha_model/tests/test_portfolio.py::TestPortfolioConstructor -v
```

---

## 十三、核心概念深入

### 13.1 Walk-Forward 验证 vs 传统交叉验证

**传统 k-fold 交叉验证的问题：**

```
传统 k-fold:
Fold 1: [TEST] [TRAIN] [TRAIN] [TRAIN] [TRAIN]
Fold 2: [TRAIN] [TEST] [TRAIN] [TRAIN] [TRAIN]
Fold 3: [TRAIN] [TRAIN] [TEST] [TRAIN] [TRAIN]
```

在金融时序中，这会导致**用未来数据训练模型来预测过去**，严重违反因果关系。

**Walk-Forward 验证：**

```
Walk-Forward:
Fold 0: [===TRAIN===]--embargo--[=TEST=]
Fold 1: [=========TRAIN=========]--embargo--[=TEST=]
Fold 2: [===============TRAIN===============]--embargo--[=TEST=]
```

核心保证：
1. **时序顺序**：训练集永远在测试集之前
2. **Embargo 隔离**：训练集和测试集之间有间隔，防止标签泄漏
3. **样本外预测**：每个数据点只在样本外被预测一次

### 13.2 为什么用 cvxpy 凸优化而不是顺序规则

**顺序规则的问题：**

```python
# 顺序规则方法（有缺陷）
weights = signal / signal.abs().sum()       # Step 1: 归一化
weights = weights.clip(-0.4, 0.4)           # Step 2: 截断
weights = weights - weights.mean()          # Step 3: dollar-neutral 调整
# 但 Step 3 之后可能破坏 Step 2 的仓位上限约束！
```

**cvxpy 凸优化的优势：**

所有约束**同时满足**，不存在顺序冲突。优化器在可行域内找到**全局最优解**（二次凸问题保证全局最优），同时考虑风险最小化、收益最大化和换手率控制。

### 13.3 Ledoit-Wolf Shrinkage 直觉

想象你估计了一个 5x5 的协方差矩阵。样本协方差矩阵 S 有 15 个自由参数（上三角），但可能只有几百个样本。估计误差很大。

Ledoit-Wolf 的做法是把 S 向一个简单的"目标矩阵" F 收缩：
- F = 对角矩阵（只保留方差，去掉所有相关性）
- 收缩后：Σ = δF + (1-δ)S

δ 越大，越信任结构化假设（"资产之间不相关"）；δ 越小，越信任样本数据。δ 的最优值由数据自动决定。

**直觉**：如果样本太少（T/N 小），δ 会较大（更多收缩）；如果样本充足，δ 会较小（更信任数据）。

### 13.4 Square-root 市场冲击模型

```
impact = coeff × σ × √(trade_value / ADV)
```

三个因素的直觉：
- **σ（波动率）**：高波动率的资产，订单簿更薄，冲击更大
- **trade_value / ADV（参与率）**：交易金额占日均成交量的比例越大，冲击越大
- **√（平方根）**：冲击随参与率增长但速率递减——第一个百分点的冲击最大

这是 Almgren-Chriss 模型的简化版，是量化行业最常用的市场冲击估计方法。

### 13.5 Vol Targeting 机制

**问题**：不同时期的市场波动率差异巨大（2020年3月 vs 2021年平静期），固定权重导致策略的波动率不可控。

**解决方案**：动态调整杠杆

```
realized_vol = portfolio.rolling(60).std() × √525960  (年化)
scale = target_vol / realized_vol
adjusted_weights = weights × scale
```

```
高波动率时期：scale < 1 → 减仓
低波动率时期：scale > 1 → 加仓

效果：策略的波动率更稳定，夏普比率通常更高
```

受 `leverage_cap` 约束：即使波动率很低，也不能无限放大杠杆。

### 13.6 无前瞻偏差的执行

Phase 2b 在多个层面防止前瞻偏差：

```
1. 时序切分（splitter.py）:
   embargo_periods = max(target_horizon, max_factor_lookback)
   训练集和测试集之间有隔离期

2. 标准化工具箱（transform.py）:
   expanding_zscore: 只看当前及之前
   rolling_zscore: 只看窗口内的过去数据
   cross_sectional: 每个时刻独立计算

3. 协方差估计（constructor.py）:
   returns_before = returns_panel.loc[:ts].iloc[:-1]
   严格只用当前时刻之前的数据（不含当前行）

4. Vol Targeting（risk_budget.py）:
   shifted_weights = weights.shift(1)
   用滞后一期的权重计算组合收益率

5. 回测 P&L（vectorized.py）:
   shifted_w = w.shift(1)
   用 t-1 时刻的权重乘 t 时刻的收益
```

---

## 十四、设计模式总结

| 模式 | 应用位置 | 解决的问题 |
|------|----------|------------|
| **Protocol（协议）** | `AlphaModel` | 模型零耦合——sklearn/PyTorch/手写均可接入 |
| **dataclass + `__post_init__`** | `TrainConfig`, `PortfolioConstraints`, `ModelMeta` | 配置集中管理 + 构造时校验 |
| **工厂模式** | `ModelStore.load(model_factory=...)` | 解耦模型创建和模型加载 |
| **深拷贝隔离** | `copy.deepcopy(model)` in Walk-Forward | 每个 fold 的模型独立，避免状态污染 |
| **原子写入** | SignalStore, ModelStore | `.tmp` + `os.rename`，防止写入中断损坏数据 |
| **凸优化建模** | `PortfolioConstructor` | 所有约束同时满足，cvxpy 联合求解 |
| **工具箱模式** | `transform.py` 的标准化函数 | 标准化不硬编码在管道中，用户自由选择 |
| **可选依赖** | `tree_models.py`, `torch_base.py` | `import lightgbm` 在方法内部，未安装不影响其他模块 |
| **Facade（门面）** | `AlphaPipeline`, `Trainer` | 一站式接口，内部编排多模块协作 |

---

## 十五、数据流全景图

```
                          Phase 2b 数据流
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  FactorStore.load()          DataReader.get_ohlcv()                 │
│       │                              │                              │
│       ▼                              ▼                              │
│  因子面板 dict                    价格面板                          │
│  {name: (ts × sym)}            (ts × sym)                          │
│       │                              │                              │
│  ┌────▼────────┐                     │                              │
│  │ selection   │ (可选)              │                              │
│  │ threshold   │                     │                              │
│  │ / top_k     │                     │                              │
│  └────┬────────┘                     │                              │
│       │                              │                              │
│  ┌────▼────────┐                     │                              │
│  │ alignment   │                     │                              │
│  │ 多频率对齐  │                     │                              │
│  └────┬────────┘                     │                              │
│       │                              │                              │
│  ┌────▼────────────────┐    ┌────────▼───────────┐                 │
│  │ build_feature_matrix │    │ forward_returns    │                 │
│  │ → X (特征矩阵)      │    │ → y (目标变量)     │                 │
│  └────┬────────────────┘    └────────┬───────────┘                 │
│       │                              │                              │
│  ┌────▼──────────────────────────────▼───────────┐                 │
│  │            Walk-Forward Engine                 │                 │
│  │  splitter.split() → folds                     │                 │
│  │  for fold in folds:                           │                 │
│  │    model = deepcopy(model)                    │                 │
│  │    model.fit(X_train, y_train)                │                 │
│  │    preds = model.predict(X_test)              │                 │
│  └────┬──────────────────────────────────────────┘                 │
│       │                                                             │
│       ▼ predictions (ts × sym)                                     │
│  ┌────────────────┐                                                 │
│  │ generate_signal │ → signal (ts × sym)                           │
│  │ 截面 z-score    │                                                │
│  └────┬───────────┘                                                 │
│       │                                                             │
│  ┌────▼──────────────────────────────────────────┐                 │
│  │         PortfolioConstructor                   │                 │
│  │  for ts in signal.index:                      │                 │
│  │    cov = estimate_covariance(returns[:ts])     │                 │
│  │    w = cvxpy.solve(                           │                 │
│  │      min w'Σw - λα'w + γ||w-w_prev||₁        │                 │
│  │      s.t. constraints                         │                 │
│  │    )                                          │                 │
│  │  apply_vol_target(weights, ...)               │                 │
│  └────┬──────────────────────────────────────────┘                 │
│       │                                                             │
│       ▼ weights (ts × sym)                                         │
│  ┌────────────────────┐                                             │
│  │ vectorized_backtest │ → BacktestResult                          │
│  │ P&L + 手续费       │   (equity, returns, turnover, summary)    │
│  │ + 市场冲击         │                                            │
│  └────┬───────────────┘                                             │
│       │                                                             │
│  ┌────▼────────┐  ┌────────────┐                                   │
│  │ SignalStore  │  │ ModelStore  │                                   │
│  │ → Phase 3   │  │ → 实盘推理  │                                   │
│  └─────────────┘  └────────────┘                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 十六、延伸练习

### 练习 1：手写一个满足 AlphaModel 协议的模型

```python
# 目标: 实现一个简单的动量因子均值模型
# 要求: 实现 fit/predict + save_model/load_model

class MomentumMeanModel:
    def fit(self, X, y, **kwargs):
        # 记住每个因子的均值系数
        # 提示: 使用 np.linalg.lstsq
        ...

    def predict(self, X):
        ...

    def save_model(self, path):
        # 提示: np.save
        ...

    def load_model(self, path):
        ...

# 验证: isinstance(MomentumMeanModel(), AlphaModel) 应为 True
```

### 练习 2：对比 Expanding vs Rolling Walk-Forward

```python
# 目标: 对同一数据集，分别用 Expanding 和 Rolling 模式训练
# 对比两者的 IC、Sharpe ratio 差异

from alpha_model.training.trainer import Trainer
from alpha_model.core.types import TrainConfig, WalkForwardMode

config_exp = TrainConfig(wf_mode=WalkForwardMode.EXPANDING, ...)
config_rol = TrainConfig(wf_mode=WalkForwardMode.ROLLING, ...)

result_exp = Trainer(model, config_exp).run(...)
result_rol = Trainer(model, config_rol).run(...)

# 对比每个 fold 的 IC
```

### 练习 3：实验不同 risk_aversion 对权重分配的影响

```python
# 目标: 固定信号，改变 λ (risk_aversion)，观察权重变化
# λ=0: 纯 alpha 最大化（忽略风险）
# λ=1: 均衡风险和收益
# λ=10: 极度风险厌恶

for lam in [0, 0.1, 1.0, 10.0]:
    constraints = PortfolioConstraints(risk_aversion=lam, ...)
    constructor = PortfolioConstructor(constraints)
    weights = constructor.construct(signal, prices)
    print(f"λ={lam}: avg |w|={weights.abs().mean().mean():.4f}")
```

### 练习 4：添加一个新的绩效指标

```python
# 目标: 在 performance.py 中添加 Information Ratio
# Information Ratio = mean(excess_return) / std(excess_return)
# 其中 excess_return = strategy_return - benchmark_return

def information_ratio(returns, benchmark_returns, periods_per_year):
    ...

# 然后在 BacktestResult.summary() 中添加此指标
```

### 练习 5：端到端 Pipeline 实验

```python
# 目标: 使用 AlphaPipeline 完成从因子到回测的完整流程
# 1. 选择 3-5 个因子
# 2. 使用 Ridge + Expanding Walk-Forward
# 3. Dollar-neutral + vol_target=0.15
# 4. 分析 summary() 中各指标的含义
# 5. 尝试换 LightGBM 模型，对比 Ridge 的结果

pipeline = AlphaPipeline(
    model=Ridge(alpha=1.0),
    train_config=TrainConfig(
        train_periods=3000, test_periods=1000,
        target_horizon=10, purge_periods=60,
    ),
    constraints=PortfolioConstraints(
        dollar_neutral=True, vol_target=0.15,
    ),
    factor_names=["..."],  # 填入你在 Phase 2a 中研究的因子
)
result = pipeline.run(price_panel)
print(result.summary())
```

---

**Phase 2b 学习完成后，你应该具备：**
1. 理解 Protocol 模式如何实现模型零耦合
2. 理解 Walk-Forward 验证防止过拟合的原理
3. 理解 cvxpy 凸优化如何同时满足多个组合约束
4. 理解从因子到目标权重的完整数据流
5. 能够独立构建和评估一个量化交易策略
6. 理解交易成本（手续费 + 市场冲击）对策略绩效的影响
7. 具备进入 Phase 3（实盘交易系统）的知识基础

**祝学习顺利！**
