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
├── config/                 # 配置层（最底层，无依赖）
│   ├── __init__.py
│   └── settings.py         # 全局配置，所有模块的参数来源
│
├── utils/                  # 工具层（依赖 config）
│   ├── logger.py           # 统一日志（控制台 + 按天轮转文件）
│   ├── time_utils.py       # 时间转换（ms ↔ datetime, 对齐, 周期转换）
│   ├── retry.py            # 重试装饰器（同步/异步, 异常分类, 指数退避）
│   └── heartbeat.py        # 运行状态监控（心跳日志 + 状态JSON文件）
│
├── data/                   # 数据层核心
│   ├── fetcher.py          # K线拉取（同步 REST, ccxt）
│   ├── tick_fetcher.py     # 逐笔成交拉取（异步 REST, ccxt async）
│   ├── orderbook_fetcher.py# 订单簿拉取（异步 WebSocket）
│   ├── market_fetcher.py   # 合约市场数据拉取（同步 REST, 合约API）
│   │
│   ├── kline_store.py      # K线存储（SQLite + WAL）
│   ├── tick_store.py       # 逐笔成交存储（Parquet 按天分区）
│   ├── orderbook_store.py  # 订单簿存储（Parquet + 内存缓冲刷盘）
│   ├── market_store.py     # 市场数据存储（SQLite 四张表）
│   │
│   ├── validator.py        # 数据校验（写入校验 + 完整性巡检）
│   ├── writer.py           # 统一写入入口（校验 → Store）
│   ├── reader.py           # 统一读取入口（路由 → Store/聚合）
│   └── aggregator.py       # 数据聚合（tick→OHLCV, 1m→5m/1h）
│
├── scripts/                # 采集脚本（独立进程）
│   ├── collect_klines.py   # K线持续采集
│   ├── collect_ticks.py    # 逐笔成交持续采集（异步）
│   ├── collect_orderbook.py# 订单簿持续采集（WebSocket）
│   ├── collect_market.py   # 合约市场数据持续采集
│   ├── backfill.py         # 历史数据回填
│   ├── check_data.py       # 数据质量巡检 + 自动修复
│   └── status.py           # 查看各采集器运行状态
│
├── tests/                  # 单元测试（14个文件, 125个测试用例）
├── main.py                 # 入口（用法说明）
└── requirements.txt        # 依赖清单
```

**依赖方向（只能向下依赖，不能向上）：**

```
scripts/  →  data/  →  utils/  →  config/
tests/  →  data/  →  utils/  →  config/
```

---

## 二、建议学习路线

```
第1站 config/settings.py            ← 理解所有配置项
  ↓
第2站 utils/ (4个文件)               ← 理解基础工具
  ↓
第3站 data/fetcher 系列 (4个文件)    ← 理解数据从哪来
  ↓
第4站 data/store 系列 (4个文件)      ← 理解数据存到哪去
  ↓
第5站 data/writer + reader + validator ← 理解门面模式
  ↓
第6站 data/aggregator.py            ← 理解数据聚合
  ↓
第7站 scripts/ (7个文件)             ← 理解如何组装成完整采集流程
  ↓
第8站 tests/ (14个文件)              ← 理解如何测试、学习mock技巧
```

每个站建议先看**模块顶部的文档字符串**，再看代码。
文档字符串是设计说明，代码是实现细节。

---

## 三、第 1 站：配置层 config/

### 文件：`config/settings.py`

**核心知识点：**
- **环境变量 + .env 文件**：敏感信息（API Key）通过 `python-dotenv` 从 `.env` 文件读取，不硬编码在源码中
- **集中配置管理**：所有可调参数在一个文件中，其他模块通过 `from config import settings` 引用
- **PROJECT_ROOT 的计算**：`Path(__file__).resolve().parent.parent`——从 `config/settings.py` 向上两级得到项目根目录

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

### 4.1 `utils/logger.py` — 统一日志

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

### 4.2 `utils/time_utils.py` — 时间处理

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
from utils.time_utils import *
from datetime import datetime, timezone

# 试试这些转换
dt = datetime(2024, 1, 15, 10, 23, 45, tzinfo=timezone.utc)
print(datetime_to_ms(dt))              # → 毫秒时间戳
print(ms_to_datetime(1705305600000))   # → datetime 对象
print(align_to_timeframe(dt, "5m"))    # → 10:20:00
print(align_to_timeframe(dt, "1h"))    # → 10:00:00
print(timeframe_to_seconds("4h"))      # → 14400
```

### 4.3 `utils/retry.py` — 重试与容错

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

### 4.4 `utils/heartbeat.py` — 运行状态监控

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

### 5.1 `data/fetcher.py` — K线拉取（同步 REST）

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

### 5.2 `data/tick_fetcher.py` — 逐笔成交拉取（异步 REST）

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

### 5.3 `data/orderbook_fetcher.py` — 订单簿拉取（异步 WebSocket）

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

### 5.4 `data/market_fetcher.py` — 合约市场数据拉取

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

### 6.1 `data/kline_store.py` — K线 SQLite 存储

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
from data.kline_store import KlineStore
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

### 6.2 `data/tick_store.py` — 逐笔成交 Parquet 存储

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

### 6.3 `data/orderbook_store.py` — 订单簿存储（带内存缓冲）

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

### 6.4 `data/market_store.py` — 市场数据 SQLite 存储

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

### 7.1 `data/writer.py` — 统一写入入口

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

### 7.2 `data/reader.py` — 统一读取入口

**核心知识点：**
- **路由模式**：`get_ohlcv("BTC/USDT", "5m")` 自动路由到 SQLite 读 1m + 降采样
- **三条读取路径**：1m 直读 / 标准周期降采样 / 亚分钟从 tick 聚合
- **按天分片防 OOM**：亚分钟周期的 tick 聚合按天处理，内存中同时只有一天的数据

**阅读重点：**

| 行 | 学什么 |
|----|--------|
| 61-97 | `get_ohlcv`：**三路分发**的路由逻辑 |
| 99-154 | `_ohlcv_from_ticks`：按天分片聚合的实现——内存友好 |

### 7.3 `data/validator.py` — 数据校验

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

### `data/aggregator.py`

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

**`collect_klines.py`**（最简单，建议第一个读）

| 行 | 学什么 |
|----|--------|
| 63-69 | 信号处理：`nonlocal running` 实现优雅退出 |
| 82-90 | 断点续传：查询最新 K线 时间 → 计算下一个 since |
| 117-120 | **可中断的 sleep**：`for _ in range(60): sleep(1)` 而不是 `sleep(60)` |

**`collect_ticks.py`**（异步版）

| 行 | 学什么 |
|----|--------|
| 43-115 | `async def run`：异步主循环，`asyncio.run(run(symbols))` 启动 |
| 63 | `heartbeat.set_status("catching_up")`：区分追赶中 vs 空闲状态 |
| 74-78 | 冷启动处理：首次采集时调用 `resolve_cold_start` |

**`collect_orderbook.py`**（回调模式）

| 行 | 学什么 |
|----|--------|
| 56-70 | `on_snapshot` 回调：WebSocket 每收到一个推送就触发 |
| 73-79 | 退出刷盘：`shutdown_handler` 中先 flush 再退出 |

**`collect_market.py`**（多指标采集）

| 行 | 学什么 |
|----|--------|
| 78-131 | 单轮四个指标的采集：每个指标独立 try/except，互不影响 |

### 9.3 工具脚本

**`backfill.py`** — 历史数据回填

| 行 | 学什么 |
|----|--------|
| 69-81 | K线批量回填：`fetch_ohlcv_batch` 自动分页 |
| 114-139 | Tick 逐批回填：直接调用 `fetch_trades`（单批）→ 立即写入 → 更新 current_id |

**`check_data.py`** — 数据质量巡检

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
# tests/test_fetcher.py
@pytest.fixture
def fetcher():
    with patch("data.fetcher.ccxt") as mock_ccxt:
        mock_exchange = MagicMock()
        mock_ccxt.binance.return_value = mock_exchange
        f = KlineFetcher()
        f.exchange = mock_exchange        # 替换真实交易所实例
        yield f
```

要点：
- `patch("data.fetcher.ccxt")` 替换的是 **fetcher 模块中导入的 ccxt**，不是全局的
- `mock_exchange.fetch_ohlcv.return_value = [...]` 预设返回值
- `mock_exchange.fetch_ohlcv.side_effect = [page1, page2]` 预设多次调用的不同返回

**技巧 2：AsyncMock 测试异步函数**

```python
# tests/test_tick_fetcher.py
mock_exchange = MagicMock()
mock_exchange.fetch_trades = AsyncMock()    # 异步方法用 AsyncMock
mock_exchange.close = AsyncMock()

# 运行异步测试
df = asyncio.run(fetcher.fetch_trades("BTC/USDT"))
```

**技巧 3：tempfile 隔离测试数据**

```python
# tests/test_kline_store.py
@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test_kline.db")
    return KlineStore(db_path)
```

每个测试用例都在独立的临时目录中运行，互不干扰。

### 10.3 运行测试

```bash
# 运行全部测试
python -m pytest tests/ -v

# 只运行某个文件
python -m pytest tests/test_fetcher.py -v

# 只运行某个类/方法
python -m pytest tests/test_fetcher.py::TestFetchOhlcv::test_returns_correct_columns -v

# 显示覆盖率
python -m pytest tests/ --cov=data --cov=utils --cov-report=term-missing
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
在 `settings.py` 中添加一个新交易对 `XRP/USDT`。思考：你需要改几个文件？

### 练习 2：手动测试 Store（简单）
在 Python REPL 中手动创建一个临时 KlineStore，写入数据，读出来验证。

### 练习 3：阅读测试理解行为（中等）
通读 `test_tick_store.py`，仅通过测试用例的名字和断言，推断 TickStore 的行为规范。然后对照源码验证你的推断。

### 练习 4：追踪一次完整采集（中等）
从 `collect_klines.py` 的 `main()` 开始，逐行追踪一次完整的 K线 采集流程：
`main → parse_args → KlineFetcher.fetch_ohlcv → DataWriter.write_ohlcv → validator.validate_ohlcv → KlineStore.write`

画出函数调用链，标注每个环节的输入和输出类型。

### 练习 5：写一个新的校验规则（中等）
在 `validator.py` 中为 `validate_ohlcv` 添加一条新规则：volume 不应超过 close × 1,000,000（防止异常大量）。然后在 `test_validator.py` 中添加对应的测试。

### 练习 6：理解 mock 模式（进阶）
阅读 `test_tick_fetcher.py`，回答：
- 为什么 `fetcher.exchange.fetch_trades` 要用 `AsyncMock` 而不是 `MagicMock`？
- `side_effect = [batch1, batch2]` 是什么意思？
- 测试中的 `asyncio.run()` 起什么作用？

---

**祝学习顺利！建议搭配 IDE 的"跳转到定义"功能阅读，可以快速在模块间跳转。**
