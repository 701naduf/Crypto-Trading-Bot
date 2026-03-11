"""
K线数据 SQLite 存储模块

只存储 1m K线（由采集脚本写入），其他周期由 DataReader 按需降采样生成。
启用 WAL (Write-Ahead Logging) 模式，支持读写并发:
    - 采集脚本持续写入的同时，DataReader 可以安全读取
    - 无需额外的锁机制

表结构:
    klines (
        symbol    TEXT,        -- 交易对，如 "BTC/USDT"
        timeframe TEXT,        -- 当前固定为 "1m"
        timestamp DATETIME,    -- K线开盘时间 (UTC)
        open      REAL,        -- 开盘价
        high      REAL,        -- 最高价
        low       REAL,        -- 最低价
        close     REAL,        -- 收盘价
        volume    REAL,        -- 成交量
        PRIMARY KEY (symbol, timeframe, timestamp)
    )

写入策略:
    INSERT OR IGNORE —— 主键冲突时跳过，实现天然幂等。
    重复写入同一根 K线 不会产生脏数据，也不会报错。

依赖: config.settings (KLINE_DB_PATH), utils.logger
"""

import sqlite3
from datetime import datetime, timezone

import pandas as pd

from data_infra.config import settings
from data_infra.utils.logger import get_logger

logger = get_logger(__name__)


class KlineStore:
    """
    SQLite K线存储

    负责 K线 数据的持久化和查询。
    一个实例对应一个 SQLite 数据库文件。
    """

    def __init__(self, db_path: str = None):
        """
        初始化 K线 存储

        Args:
            db_path: 数据库文件路径，默认使用 settings.KLINE_DB_PATH

        初始化时自动:
            1. 创建数据库文件（如不存在）
            2. 启用 WAL 模式
            3. 建表（如不存在）
        """
        self.db_path = db_path or settings.KLINE_DB_PATH

        # 确保目录存在
        import os
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # 初始化数据库
        self._init_db()

    def _init_db(self):
        """创建表和索引，启用 WAL 模式"""
        with sqlite3.connect(self.db_path) as conn:
            # 启用 WAL 模式
            # WAL 允许读写并发: 写操作不阻塞读操作
            # 这对采集脚本（持续写入）和 DataReader（随时读取）至关重要
            conn.execute("PRAGMA journal_mode=WAL")

            # 建表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS klines (
                    symbol    TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open      REAL NOT NULL,
                    high      REAL NOT NULL,
                    low       REAL NOT NULL,
                    close     REAL NOT NULL,
                    volume    REAL NOT NULL,
                    PRIMARY KEY (symbol, timeframe, timestamp)
                )
            """)

            conn.commit()

        logger.debug(f"K线数据库已初始化: {self.db_path}")

    def write(self, df: pd.DataFrame, symbol: str, timeframe: str) -> int:
        """
        写入 K线 数据

        使用 INSERT OR IGNORE，主键冲突时跳过（幂等写入）。
        适合增量采集场景: 每轮采集的数据可能与上一轮有重叠，
        重叠部分会被自动跳过。

        Args:
            df:        包含 [timestamp, open, high, low, close, volume] 的 DataFrame
                       timestamp 列应为 UTC datetime 或可解析的时间字符串
            symbol:    交易对，如 "BTC/USDT"
            timeframe: K线周期，如 "1m"

        Returns:
            实际新增的行数（不含被跳过的重复行）
        """
        if df.empty:
            return 0

        # 准备写入数据
        records = []
        for _, row in df.iterrows():
            # 确保 timestamp 是 ISO 格式字符串，便于 SQLite 存储和排序
            ts = row["timestamp"]
            if isinstance(ts, datetime):
                ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
            else:
                ts_str = str(ts)

            records.append((
                symbol, timeframe, ts_str,
                float(row["open"]), float(row["high"]),
                float(row["low"]), float(row["close"]),
                float(row["volume"]),
            ))

        with sqlite3.connect(self.db_path) as conn:
            # 获取写入前的行数（用于计算新增数）
            cursor = conn.execute(
                "SELECT COUNT(*) FROM klines WHERE symbol = ? AND timeframe = ?",
                (symbol, timeframe),
            )
            count_before = cursor.fetchone()[0]

            # 批量写入
            conn.executemany(
                """INSERT OR IGNORE INTO klines
                   (symbol, timeframe, timestamp, open, high, low, close, volume)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                records,
            )
            conn.commit()

            # 计算新增行数
            cursor = conn.execute(
                "SELECT COUNT(*) FROM klines WHERE symbol = ? AND timeframe = ?",
                (symbol, timeframe),
            )
            count_after = cursor.fetchone()[0]

        new_rows = count_after - count_before
        logger.debug(
            f"{symbol} {timeframe}: 写入 {len(records)} 行, "
            f"新增 {new_rows} 行, 跳过 {len(records) - new_rows} 行重复"
        )
        return new_rows

    def read(
        self,
        symbol: str,
        timeframe: str,
        start: datetime = None,
        end: datetime = None,
    ) -> pd.DataFrame:
        """
        读取 K线 数据

        Args:
            symbol:    交易对
            timeframe: K线周期
            start:     起始时间（包含），None 表示不限
            end:       结束时间（包含），None 表示不限

        Returns:
            DataFrame [timestamp, open, high, low, close, volume]
            按时间升序排列，timestamp 列为 UTC datetime
        """
        query = "SELECT timestamp, open, high, low, close, volume FROM klines WHERE symbol = ? AND timeframe = ?"
        params: list = [symbol, timeframe]

        if start is not None:
            query += " AND timestamp >= ?"
            params.append(start.strftime("%Y-%m-%d %H:%M:%S"))

        if end is not None:
            query += " AND timestamp <= ?"
            params.append(end.strftime("%Y-%m-%d %H:%M:%S"))

        query += " ORDER BY timestamp ASC"

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        # 将 timestamp 列转为 UTC datetime
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        return df

    def get_latest_timestamp(
        self, symbol: str, timeframe: str
    ) -> datetime | None:
        """
        获取最新 K线 的时间戳

        用于增量采集: 新一轮采集从此时间之后开始拉取。

        Args:
            symbol:    交易对
            timeframe: K线周期

        Returns:
            最新 K线 的 UTC datetime，无数据时返回 None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT MAX(timestamp) FROM klines WHERE symbol = ? AND timeframe = ?",
                (symbol, timeframe),
            )
            row = cursor.fetchone()

        if row[0] is None:
            return None

        # 解析为 UTC datetime
        return datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc
        )

    def count(self, symbol: str, timeframe: str) -> int:
        """
        统计指定币对和周期的 K线 总数

        Args:
            symbol:    交易对
            timeframe: K线周期

        Returns:
            K线数量
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM klines WHERE symbol = ? AND timeframe = ?",
                (symbol, timeframe),
            )
            return cursor.fetchone()[0]
