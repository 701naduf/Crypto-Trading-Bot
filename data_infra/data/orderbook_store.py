"""
订单簿快照 Parquet 存储模块

按 symbol/date 分区存储，每个交易对每天一个 Parquet 文件。
路径格式: db/orderbook/{SYMBOL}/{YYYY-MM-DD}.parquet

由于 100ms 粒度下数据量大（每币对每天 ~86.4万条），
采用内存缓冲 + 定期刷盘策略，避免每条数据都触发磁盘 I/O。

列结构（10 档示例，共 40 个数值列 + 1 个时间列）:
    timestamp    datetime64[ms]  -- 快照时间 (UTC)
    bid_price_0  float64         -- 买一价
    bid_qty_0    float64         -- 买一量
    ...
    bid_price_9  float64         -- 买十价
    bid_qty_9    float64         -- 买十量
    ask_price_0  float64         -- 卖一价
    ask_qty_0    float64         -- 卖一量
    ...
    ask_price_9  float64         -- 卖十价
    ask_qty_9    float64         -- 卖十量

为什么用扁平列而非嵌套结构:
    - Parquet 列式存储对扁平列压缩率最优
    - 读取时可以只加载需要的档位（如只读 bid_price_0/ask_price_0）
    - 直接映射为 DataFrame 列，无需拆包

依赖: config.settings (ORDERBOOK_DATA_DIR, ORDERBOOK_DEPTH), utils.logger
"""

import os
from datetime import datetime, timezone

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data_infra.config import settings
from data_infra.utils.logger import get_logger

logger = get_logger(__name__)


def _build_schema(depth: int) -> pa.Schema:
    """
    根据档位深度构建 Parquet schema

    Args:
        depth: 档位数（如 10）

    Returns:
        PyArrow Schema，包含 timestamp + depth*4 个数值列
    """
    fields = [("timestamp", pa.timestamp("ms", tz="UTC"))]

    for i in range(depth):
        fields.append((f"bid_price_{i}", pa.float64()))
        fields.append((f"bid_qty_{i}", pa.float64()))

    for i in range(depth):
        fields.append((f"ask_price_{i}", pa.float64()))
        fields.append((f"ask_qty_{i}", pa.float64()))

    return pa.schema(fields)


def _get_column_names(depth: int) -> list[str]:
    """获取所有列名（按 schema 顺序）"""
    cols = ["timestamp"]
    for i in range(depth):
        cols.extend([f"bid_price_{i}", f"bid_qty_{i}"])
    for i in range(depth):
        cols.extend([f"ask_price_{i}", f"ask_qty_{i}"])
    return cols


class OrderbookStore:
    """
    Parquet 订单簿存储

    采用内存缓冲 + 定期刷盘的写入策略:
        1. append() 将快照追加到内存缓冲
        2. 缓冲达到 buffer_size 条时自动刷盘
        3. 也可手动调用 flush() 强制刷盘
        4. 脚本退出时必须调用 flush_and_close() 确保数据不丢失
    """

    def __init__(self, data_dir: str = None, buffer_size: int = 10000):
        """
        初始化订单簿存储

        Args:
            data_dir:    存储根目录，默认使用 settings.ORDERBOOK_DATA_DIR
            buffer_size: 内存缓冲条数，达到后自动刷盘
                         默认 10000 条 ≈ 约 17 分钟 @100ms 推送频率
                         对 5 个币对，单个缓冲 ~10000 条 × 41列 × 8字节 ≈ 3.2MB
        """
        self.data_dir = data_dir or settings.ORDERBOOK_DATA_DIR
        self.depth = settings.ORDERBOOK_DEPTH
        self.buffer_size = buffer_size

        os.makedirs(self.data_dir, exist_ok=True)

        # schema 和列名（创建一次，反复使用）
        self._schema = _build_schema(self.depth)
        self._columns = _get_column_names(self.depth)

        # 内存缓冲: { "BTC/USDT": [row1, row2, ...], ... }
        # 每个 row 是一个 dict，对应一行数据
        self._buffers: dict[str, list[dict]] = {}

        # 清理残留 .tmp 文件
        self._cleanup_tmp_files()

    def _cleanup_tmp_files(self):
        """清理残留的 .tmp 文件"""
        count = 0
        for root, _dirs, files in os.walk(self.data_dir):
            for f in files:
                if f.endswith(".tmp"):
                    os.remove(os.path.join(root, f))
                    count += 1
        if count > 0:
            logger.info(f"订单簿: 清理了 {count} 个残留的 .tmp 文件")

    def _symbol_dir(self, symbol: str) -> str:
        """获取币对存储目录"""
        safe_name = symbol.replace("/", "_")
        path = os.path.join(self.data_dir, safe_name)
        os.makedirs(path, exist_ok=True)
        return path

    def _date_path(self, symbol: str, date_str: str) -> str:
        """获取指定日期的 Parquet 文件路径"""
        return os.path.join(self._symbol_dir(symbol), f"{date_str}.parquet")

    def append(self, symbol: str, snapshot: dict):
        """
        追加一条订单簿快照到内存缓冲

        Args:
            snapshot: 订单簿快照:
                {
                    "timestamp": datetime,              # UTC 时间
                    "bids": [[price, qty], ...],        # 买盘，价格降序
                    "asks": [[price, qty], ...],        # 卖盘，价格升序
                }

        当缓冲达到 buffer_size 时自动触发刷盘。
        """
        # 将嵌套结构展平为扁平 dict
        row = {"timestamp": snapshot["timestamp"]}

        bids = snapshot["bids"]
        asks = snapshot["asks"]

        for i in range(self.depth):
            if i < len(bids):
                row[f"bid_price_{i}"] = float(bids[i][0])
                row[f"bid_qty_{i}"] = float(bids[i][1])
            else:
                # 档位不足时填 0（不应发生，但防御性编程）
                row[f"bid_price_{i}"] = 0.0
                row[f"bid_qty_{i}"] = 0.0

            if i < len(asks):
                row[f"ask_price_{i}"] = float(asks[i][0])
                row[f"ask_qty_{i}"] = float(asks[i][1])
            else:
                row[f"ask_price_{i}"] = 0.0
                row[f"ask_qty_{i}"] = 0.0

        # 追加到缓冲
        if symbol not in self._buffers:
            self._buffers[symbol] = []
        self._buffers[symbol].append(row)

        # 检查是否需要自动刷盘
        if len(self._buffers[symbol]) >= self.buffer_size:
            self.flush(symbol)

    def flush(self, symbol: str = None):
        """
        将内存缓冲刷盘到 Parquet 文件

        Args:
            symbol: 指定刷盘的币对，None 则刷盘所有币对

        刷盘流程:
            1. 将缓冲数据按日期分组
            2. 对每天的数据:
               a. 读取已有文件（如存在）
               b. 追加新数据
               c. 原子写入（.tmp → rename）
            3. 清空缓冲
        """
        symbols = [symbol] if symbol else list(self._buffers.keys())

        for sym in symbols:
            buffer = self._buffers.get(sym, [])
            if not buffer:
                continue

            # 转为 DataFrame
            df = pd.DataFrame(buffer)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

            # 按日期分组刷盘
            df["_date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
            for date_str, day_df in df.groupby("_date"):
                day_df = day_df.drop(columns=["_date"])
                self._flush_day(day_df, sym, date_str)

            flushed_count = len(buffer)
            self._buffers[sym] = []
            logger.debug(f"{sym}: 刷盘 {flushed_count} 条订单簿快照")

    def _flush_day(self, new_df: pd.DataFrame, symbol: str, date_str: str):
        """将单天数据写入 Parquet（追加模式，原子写入）"""
        file_path = self._date_path(symbol, date_str)
        tmp_path = file_path + ".tmp"

        # 读取已有数据
        if os.path.exists(file_path):
            existing_df = pd.read_parquet(file_path)
            merged = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            merged = new_df

        # 按时间排序
        merged = merged.sort_values("timestamp").reset_index(drop=True)

        # 原子写入
        table = pa.Table.from_pandas(merged, schema=self._schema)
        pq.write_table(table, tmp_path, compression="snappy")
        os.replace(tmp_path, file_path)

    def read(
        self,
        symbol: str,
        start: datetime = None,
        end: datetime = None,
        levels: int = None,
    ) -> pd.DataFrame:
        """
        读取订单簿快照序列

        Args:
            symbol: 交易对
            start:  起始时间（包含）
            end:    结束时间（包含）
            levels: 只读取前 N 档，None 则读取全部档位
                    例如 levels=5 只返回买卖各 5 档（20 列 + timestamp）
                    可显著减少内存占用

        Returns:
            DataFrame，按 timestamp 升序排列
        """
        sym_dir = self._symbol_dir(symbol)

        if not os.path.exists(sym_dir):
            return pd.DataFrame(columns=self._columns)

        # 确定要读取的列
        if levels is not None and levels < self.depth:
            read_cols = ["timestamp"]
            for i in range(levels):
                read_cols.extend([f"bid_price_{i}", f"bid_qty_{i}"])
            for i in range(levels):
                read_cols.extend([f"ask_price_{i}", f"ask_qty_{i}"])
        else:
            read_cols = None  # 读取全部列

        # 确定日期范围
        start_date = start.strftime("%Y-%m-%d") if start else None
        end_date = end.strftime("%Y-%m-%d") if end else None

        frames = []
        for f in sorted(os.listdir(sym_dir)):
            if not f.endswith(".parquet"):
                continue

            file_date = f.replace(".parquet", "")
            if start_date and file_date < start_date:
                continue
            if end_date and file_date > end_date:
                continue

            file_path = os.path.join(sym_dir, f)
            df = pd.read_parquet(file_path, columns=read_cols)
            frames.append(df)

        if not frames:
            cols = read_cols if read_cols else self._columns
            return pd.DataFrame(columns=cols)

        result = pd.concat(frames, ignore_index=True)

        # 精确过滤时间范围
        if start is not None:
            start_utc = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
            result = result[result["timestamp"] >= start_utc]
        if end is not None:
            end_utc = end if end.tzinfo else end.replace(tzinfo=timezone.utc)
            result = result[result["timestamp"] <= end_utc]

        return result.sort_values("timestamp").reset_index(drop=True)

    def flush_and_close(self):
        """
        刷盘所有缓冲并释放资源

        脚本退出时必须调用此方法，确保内存中的数据不丢失。
        通常在信号处理函数（SIGTERM/SIGINT handler）中调用。
        """
        total = sum(len(buf) for buf in self._buffers.values())
        if total > 0:
            logger.info(f"订单簿退出刷盘: 共 {total} 条待写入")
        self.flush()
        logger.info("订单簿存储已关闭")

    def get_buffer_size(self, symbol: str = None) -> int:
        """
        获取当前缓冲区大小

        Args:
            symbol: 指定币对，None 则返回所有币对的总缓冲量

        Returns:
            缓冲中的记录数
        """
        if symbol:
            return len(self._buffers.get(symbol, []))
        return sum(len(buf) for buf in self._buffers.values())
