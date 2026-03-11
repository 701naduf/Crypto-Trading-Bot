"""
逐笔成交 Parquet 存储模块

按 symbol/date 分区存储，每个交易对每天一个 Parquet 文件。
以 trade_id 为增量锚点，支持断点续传。

路径格式: db/ticks/{SYMBOL}/{YYYY-MM-DD}.parquet
    - SYMBOL: 将 "/" 替换为 "_"，如 "BTC/USDT" → "BTC_USDT"
    - 日期: 按 trade 的 UTC 时间戳确定归属日期

列结构:
    trade_id   int64          -- 全局递增的成交ID
    timestamp  datetime64[ms] -- 成交时间 (UTC)
    price      float64        -- 成交价格
    amount     float64        -- 成交数量
    side       string         -- "buy" 或 "sell"

写入策略:
    原子写入: 先写 .tmp 文件，成功后 rename 为正式文件。
    如果写入过程中崩溃，残留的 .tmp 文件会在下次初始化时被清理。
    写入时自动与已有数据合并去重（基于 trade_id）。

依赖: config.settings (TICK_DATA_DIR), utils.logger
"""

import os
from datetime import datetime, timezone

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# Parquet 文件的列 schema
# 定义统一的 schema 确保所有文件列类型一致
TICK_SCHEMA = pa.schema([
    ("trade_id", pa.int64()),
    ("timestamp", pa.timestamp("ms", tz="UTC")),
    ("price", pa.float64()),
    ("amount", pa.float64()),
    ("side", pa.string()),
])


class TickStore:
    """
    Parquet 逐笔成交存储

    每个交易对按天分区存储为独立的 Parquet 文件。
    支持原子写入、去重合并、断点续传。
    """

    def __init__(self, data_dir: str = None):
        """
        初始化 Tick 存储

        Args:
            data_dir: 存储根目录，默认使用 settings.TICK_DATA_DIR

        初始化时自动清理残留的 .tmp 文件（上次崩溃遗留的）。
        """
        self.data_dir = data_dir or settings.TICK_DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True)

        # 清理残留的 .tmp 文件
        self._cleanup_tmp_files()

    def _cleanup_tmp_files(self):
        """
        清理所有残留的 .tmp 文件

        这些文件是上次写入过程中崩溃遗留的半成品，不包含完整数据。
        直接删除是安全的，因为正式文件（无 .tmp 后缀）才是最终数据。
        """
        count = 0
        for root, _dirs, files in os.walk(self.data_dir):
            for f in files:
                if f.endswith(".tmp"):
                    os.remove(os.path.join(root, f))
                    count += 1

        if count > 0:
            logger.info(f"清理了 {count} 个残留的 .tmp 文件")

    def _symbol_dir(self, symbol: str) -> str:
        """
        获取指定币对的存储目录

        Args:
            symbol: 交易对，如 "BTC/USDT"

        Returns:
            目录路径，如 "db/ticks/BTC_USDT"
        """
        # "BTC/USDT" → "BTC_USDT"
        safe_name = symbol.replace("/", "_")
        path = os.path.join(self.data_dir, safe_name)
        os.makedirs(path, exist_ok=True)
        return path

    def _date_path(self, symbol: str, date_str: str) -> str:
        """
        获取指定日期的 Parquet 文件路径

        Args:
            symbol:   交易对
            date_str: 日期字符串，如 "2024-01-15"

        Returns:
            文件路径，如 "db/ticks/BTC_USDT/2024-01-15.parquet"
        """
        return os.path.join(self._symbol_dir(symbol), f"{date_str}.parquet")

    def write(self, df: pd.DataFrame, symbol: str) -> int:
        """
        写入逐笔成交数据（原子写入 + 去重合并）

        流程:
            1. 按日期拆分新数据
            2. 对每天的数据:
               a. 读取已有的 Parquet 文件（如存在）
               b. 合并新旧数据，按 trade_id 去重
               c. 写入 .tmp 文件
               d. rename .tmp 为正式文件（原子操作）

        Args:
            df:     包含 [trade_id, timestamp, price, amount, side] 的 DataFrame
            symbol: 交易对

        Returns:
            实际新增的记录数（去重后）
        """
        if df.empty:
            return 0

        # 确保 timestamp 是 UTC datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        elif df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

        # 按日期拆分: 根据 timestamp 的日期部分分组
        df["_date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
        total_new = 0

        for date_str, day_df in df.groupby("_date"):
            day_df = day_df.drop(columns=["_date"])
            new_count = self._write_day(day_df, symbol, date_str)
            total_new += new_count

        logger.debug(f"{symbol}: 写入 {len(df)} 条 tick, 新增 {total_new} 条")
        return total_new

    def _write_day(
        self, new_df: pd.DataFrame, symbol: str, date_str: str
    ) -> int:
        """
        写入单天数据（原子写入 + 去重）

        Returns:
            新增的记录数
        """
        file_path = self._date_path(symbol, date_str)
        tmp_path = file_path + ".tmp"

        # 读取已有数据（如存在）
        if os.path.exists(file_path):
            existing_df = pd.read_parquet(file_path)
            # 合并并按 trade_id 去重，保留最新的
            merged = pd.concat([existing_df, new_df], ignore_index=True)
            merged = merged.drop_duplicates(subset=["trade_id"], keep="last")
            new_count = len(merged) - len(existing_df)
        else:
            merged = new_df.drop_duplicates(subset=["trade_id"], keep="last")
            new_count = len(merged)

        # 按 trade_id 排序
        merged = merged.sort_values("trade_id").reset_index(drop=True)

        # 原子写入: 先写 .tmp 再 rename
        table = pa.Table.from_pandas(merged, schema=TICK_SCHEMA)
        pq.write_table(table, tmp_path, compression="snappy")
        os.replace(tmp_path, file_path)

        return new_count

    def read(
        self,
        symbol: str,
        start: datetime = None,
        end: datetime = None,
    ) -> pd.DataFrame:
        """
        读取指定日期范围的逐笔成交数据

        Args:
            symbol: 交易对
            start:  起始时间（包含），None 表示从最早开始
            end:    结束时间（包含），None 表示到最新

        Returns:
            DataFrame [trade_id, timestamp, price, amount, side]
            按 trade_id 升序排列
        """
        sym_dir = self._symbol_dir(symbol)

        if not os.path.exists(sym_dir):
            return pd.DataFrame(columns=["trade_id", "timestamp", "price", "amount", "side"])

        # 确定日期范围
        start_date = start.strftime("%Y-%m-%d") if start else None
        end_date = end.strftime("%Y-%m-%d") if end else None

        # 扫描目录下所有 .parquet 文件
        frames = []
        for f in sorted(os.listdir(sym_dir)):
            if not f.endswith(".parquet"):
                continue

            file_date = f.replace(".parquet", "")

            # 过滤日期范围
            if start_date and file_date < start_date:
                continue
            if end_date and file_date > end_date:
                continue

            file_path = os.path.join(sym_dir, f)
            df = pd.read_parquet(file_path)
            frames.append(df)

        if not frames:
            return pd.DataFrame(columns=["trade_id", "timestamp", "price", "amount", "side"])

        result = pd.concat(frames, ignore_index=True)

        # 精确过滤时间范围（文件按天分区，边界日期可能包含范围外数据）
        if start is not None:
            start_utc = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
            result = result[result["timestamp"] >= start_utc]
        if end is not None:
            end_utc = end if end.tzinfo else end.replace(tzinfo=timezone.utc)
            result = result[result["timestamp"] <= end_utc]

        return result.sort_values("trade_id").reset_index(drop=True)

    def get_latest_trade_id(self, symbol: str) -> int | None:
        """
        获取指定币对的最新 trade_id

        用于增量采集: 从此 trade_id 之后继续拉取。

        扫描该币对目录下最新（日期最大）的 Parquet 文件，
        读取其中最大的 trade_id。

        Args:
            symbol: 交易对

        Returns:
            最新的 trade_id，无数据时返回 None
        """
        sym_dir = self._symbol_dir(symbol)

        if not os.path.exists(sym_dir):
            return None

        # 找到最新的文件
        files = sorted(
            [f for f in os.listdir(sym_dir) if f.endswith(".parquet")],
            reverse=True,
        )

        if not files:
            return None

        # 从最新文件中获取最大 trade_id
        latest_file = os.path.join(sym_dir, files[0])
        df = pd.read_parquet(latest_file, columns=["trade_id"])

        if df.empty:
            return None

        return int(df["trade_id"].max())
