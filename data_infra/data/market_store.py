"""
合约市场数据 SQLite 存储模块

存储资金费率、持仓量、多空持仓比、主动买卖量等合约市场数据。
数据量极小（每日每币对几百条），SQLite 完全胜任。
与 kline.db 分开存储，使用独立的 market.db。

表结构:

    funding_rates (
        symbol         TEXT,
        timestamp      DATETIME,       -- ISO 格式 UTC 时间
        funding_rate   REAL,
        PRIMARY KEY (symbol, timestamp)
    )

    open_interest (
        symbol              TEXT,
        timestamp           DATETIME,
        open_interest       REAL,   -- 持仓量（合约张数）
        open_interest_value REAL,   -- 持仓价值（USDT）
        PRIMARY KEY (symbol, timestamp)
    )

    long_short_ratio (
        symbol           TEXT,
        timestamp        DATETIME,
        long_ratio       REAL,
        short_ratio      REAL,
        long_short_ratio REAL,
        PRIMARY KEY (symbol, timestamp)
    )

    taker_buy_sell (
        symbol         TEXT,
        timestamp      DATETIME,
        buy_vol        REAL,
        sell_vol       REAL,
        buy_sell_ratio REAL,
        PRIMARY KEY (symbol, timestamp)
    )

依赖: config.settings (MARKET_DB_PATH), utils.logger
"""

import os
import sqlite3
from datetime import datetime, timezone

import pandas as pd

from data_infra.config import settings
from data_infra.utils.logger import get_logger

logger = get_logger(__name__)


class MarketStore:
    """
    SQLite 合约市场数据存储

    管理四张表：资金费率、持仓量、多空持仓比、主动买卖量。
    每张表使用 (symbol, timestamp) 复合主键，INSERT OR IGNORE 实现幂等写入。
    """

    def __init__(self, db_path: str = None):
        """
        初始化市场数据存储

        Args:
            db_path: 数据库路径，默认使用 settings.MARKET_DB_PATH

        自动建表、启用 WAL 模式。
        """
        self.db_path = db_path or settings.MARKET_DB_PATH

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """创建四张表，启用 WAL 模式"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")

            # === 资金费率表 ===
            conn.execute("""
                CREATE TABLE IF NOT EXISTS funding_rates (
                    symbol       TEXT NOT NULL,
                    timestamp    DATETIME NOT NULL,
                    funding_rate REAL NOT NULL,
                    PRIMARY KEY (symbol, timestamp)
                )
            """)

            # === 持仓量表 ===
            conn.execute("""
                CREATE TABLE IF NOT EXISTS open_interest (
                    symbol              TEXT NOT NULL,
                    timestamp           DATETIME NOT NULL,
                    open_interest       REAL NOT NULL,
                    open_interest_value REAL NOT NULL,
                    PRIMARY KEY (symbol, timestamp)
                )
            """)

            # === 多空持仓比表 ===
            conn.execute("""
                CREATE TABLE IF NOT EXISTS long_short_ratio (
                    symbol           TEXT NOT NULL,
                    timestamp        DATETIME NOT NULL,
                    long_ratio       REAL NOT NULL,
                    short_ratio      REAL NOT NULL,
                    long_short_ratio REAL NOT NULL,
                    PRIMARY KEY (symbol, timestamp)
                )
            """)

            # === 主动买卖量表 ===
            conn.execute("""
                CREATE TABLE IF NOT EXISTS taker_buy_sell (
                    symbol         TEXT NOT NULL,
                    timestamp      DATETIME NOT NULL,
                    buy_vol        REAL NOT NULL,
                    sell_vol       REAL NOT NULL,
                    buy_sell_ratio REAL NOT NULL,
                    PRIMARY KEY (symbol, timestamp)
                )
            """)

            conn.commit()

        logger.debug(f"市场数据数据库已初始化: {self.db_path}")

    def _ts_to_str(self, ts) -> str:
        """将 timestamp 转为 ISO 格式字符串"""
        if isinstance(ts, datetime):
            return ts.strftime("%Y-%m-%d %H:%M:%S")
        return str(ts)

    # ===================================================================
    # 写入方法
    # ===================================================================

    def write_funding_rate(self, df: pd.DataFrame, symbol: str) -> int:
        """
        写入资金费率数据

        Args:
            df:     包含 [timestamp, funding_rate] 的 DataFrame
            symbol: 交易对

        Returns:
            新增行数
        """
        if df.empty:
            return 0

        records = [
            (symbol, self._ts_to_str(row["timestamp"]), float(row["funding_rate"]))
            for _, row in df.iterrows()
        ]

        return self._insert_many(
            "funding_rates",
            "(symbol, timestamp, funding_rate)",
            records,
            symbol,
        )

    def write_open_interest(self, symbol: str, data: dict) -> int:
        """
        写入持仓量快照

        Args:
            symbol: 交易对
            data:   {"timestamp": datetime, "open_interest": float,
                     "open_interest_value": float}

        Returns:
            新增行数（0 或 1）
        """
        record = [(
            symbol,
            self._ts_to_str(data["timestamp"]),
            float(data["open_interest"]),
            float(data["open_interest_value"]),
        )]

        return self._insert_many(
            "open_interest",
            "(symbol, timestamp, open_interest, open_interest_value)",
            record,
            symbol,
        )

    def write_long_short_ratio(self, df: pd.DataFrame, symbol: str) -> int:
        """
        写入多空持仓比

        Args:
            df:     包含 [timestamp, long_ratio, short_ratio, long_short_ratio] 的 DataFrame
            symbol: 交易对

        Returns:
            新增行数
        """
        if df.empty:
            return 0

        records = [
            (
                symbol,
                self._ts_to_str(row["timestamp"]),
                float(row["long_ratio"]),
                float(row["short_ratio"]),
                float(row["long_short_ratio"]),
            )
            for _, row in df.iterrows()
        ]

        return self._insert_many(
            "long_short_ratio",
            "(symbol, timestamp, long_ratio, short_ratio, long_short_ratio)",
            records,
            symbol,
        )

    def write_taker_buy_sell(self, df: pd.DataFrame, symbol: str) -> int:
        """
        写入主动买卖量

        Args:
            df:     包含 [timestamp, buy_vol, sell_vol, buy_sell_ratio] 的 DataFrame
            symbol: 交易对

        Returns:
            新增行数
        """
        if df.empty:
            return 0

        records = [
            (
                symbol,
                self._ts_to_str(row["timestamp"]),
                float(row["buy_vol"]),
                float(row["sell_vol"]),
                float(row["buy_sell_ratio"]),
            )
            for _, row in df.iterrows()
        ]

        return self._insert_many(
            "taker_buy_sell",
            "(symbol, timestamp, buy_vol, sell_vol, buy_sell_ratio)",
            records,
            symbol,
        )

    def _insert_many(
        self, table: str, columns: str, records: list[tuple], symbol: str
    ) -> int:
        """
        批量 INSERT OR IGNORE

        通过比较写入前后的行数来计算新增行数。

        Args:
            table:   表名
            columns: 列名字符串，如 "(symbol, timestamp, funding_rate)"
            records: 记录列表
            symbol:  交易对（用于统计）

        Returns:
            新增行数
        """
        placeholders = ", ".join(["?"] * len(records[0]))

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE symbol = ?", (symbol,)
            )
            count_before = cursor.fetchone()[0]

            conn.executemany(
                f"INSERT OR IGNORE INTO {table} {columns} VALUES ({placeholders})",
                records,
            )
            conn.commit()

            cursor = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE symbol = ?", (symbol,)
            )
            count_after = cursor.fetchone()[0]

        new_rows = count_after - count_before
        logger.debug(f"{symbol} {table}: 写入 {len(records)} 行, 新增 {new_rows} 行")
        return new_rows

    # ===================================================================
    # 读取方法
    # ===================================================================

    def read(
        self,
        table: str,
        symbol: str,
        start: datetime = None,
        end: datetime = None,
    ) -> pd.DataFrame:
        """
        通用读取接口

        Args:
            table:  表名，可选:
                    "funding_rates", "open_interest",
                    "long_short_ratio", "taker_buy_sell"
            symbol: 交易对
            start:  起始时间（包含）
            end:    结束时间（包含）

        Returns:
            DataFrame，按 timestamp 升序排列
        """
        # 安全校验: 防止 SQL 注入（table 来自外部参数）
        valid_tables = {
            "funding_rates", "open_interest",
            "long_short_ratio", "taker_buy_sell",
        }
        if table not in valid_tables:
            raise ValueError(
                f"无效的表名: '{table}'，有效值: {valid_tables}"
            )

        query = f"SELECT * FROM {table} WHERE symbol = ?"
        params: list = [symbol]

        if start is not None:
            query += " AND timestamp >= ?"
            params.append(start.strftime("%Y-%m-%d %H:%M:%S"))

        if end is not None:
            query += " AND timestamp <= ?"
            params.append(end.strftime("%Y-%m-%d %H:%M:%S"))

        query += " ORDER BY timestamp ASC"

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        return df

    def get_latest_timestamp(
        self, table: str, symbol: str
    ) -> datetime | None:
        """
        获取指定表指定币对的最新时间

        Args:
            table:  表名
            symbol: 交易对

        Returns:
            最新记录的 UTC datetime，无数据返回 None
        """
        valid_tables = {
            "funding_rates", "open_interest",
            "long_short_ratio", "taker_buy_sell",
        }
        if table not in valid_tables:
            raise ValueError(f"无效的表名: '{table}'")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                f"SELECT MAX(timestamp) FROM {table} WHERE symbol = ?",
                (symbol,),
            )
            row = cursor.fetchone()

        if row[0] is None:
            return None

        return datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc
        )
