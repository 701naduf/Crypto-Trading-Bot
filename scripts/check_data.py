"""
数据质量巡检脚本

定期运行，检查所有采集数据的完整性和质量。
输出巡检报告到日志和控制台。

巡检内容:
    1. K线连续性: 1m K线的时间缺口检测
    2. Tick 同步状态: 本地 vs 交易所最新 trade_id
    3. 订单簿: 最新快照时间（判断采集是否在运行）
    4. 市场数据: 资金费率/OI 是否有缺失
    5. 数据量统计: 各表/文件的记录数和存储大小
    6. 跨源 K线 对比: 随机抽样与 API 比对

用法:
    python -m scripts.check_data                        # 运行巡检
    python -m scripts.check_data --symbols BTC/USDT     # 只检查指定币对
    python -m scripts.check_data --fix                  # 巡检并自动修复 K线缺口
"""

import argparse
import asyncio
import os
import random
from datetime import datetime, timedelta, timezone

from config import settings
from data.kline_store import KlineStore
from data.tick_store import TickStore
from data.market_store import MarketStore
from data.validator import check_kline_continuity, check_tick_sync_status
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="数据质量巡检")
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="指定检查的交易对"
    )
    parser.add_argument(
        "--fix", action="store_true",
        help="自动修复检测到的问题（当前支持 K线缺口回填）"
    )
    return parser.parse_args()


def check_klines(kline_store: KlineStore, symbols: list[str], fix: bool = False):
    """
    检查 K线 数据

    Args:
        kline_store: K线存储实例
        symbols: 交易对列表
        fix: 是否自动修复缺口
    """
    print("\n=== K线数据巡检 ===")

    all_gaps = {}  # symbol -> gaps，用于 --fix 修复

    for symbol in symbols:
        count = kline_store.count(symbol, "1m")
        latest = kline_store.get_latest_timestamp(symbol, "1m")

        # 连续性检查
        gaps = check_kline_continuity(kline_store, symbol, "1m")
        gap_info = f"缺口: {len(gaps)} 处" if gaps else "无缺口"

        if gaps:
            all_gaps[symbol] = gaps

        # 计算延迟
        lag = ""
        if latest:
            delay = (datetime.now(timezone.utc) - latest).total_seconds()
            lag = f", 延迟 {delay:.0f}s"

        print(f"  {symbol}: {count:,} 根, 最新 {latest or '无数据'}{lag}, {gap_info}")

        # 输出缺口详情
        for gap in gaps[:5]:  # 最多显示 5 处
            missing = gap["missing_bars"]
            print(f"    ↳ 缺口: {gap['start']} ~ {gap['end']} (缺 {missing} 根)")

    # --fix: 自动回填 K线 缺口
    if fix and all_gaps:
        _fix_kline_gaps(all_gaps)


def _fix_kline_gaps(all_gaps: dict):
    """
    自动回填 K线 缺口

    对每个有缺口的币对，调用 KlineFetcher 拉取缺失区间的数据并写入。

    Args:
        all_gaps: {symbol: [gap_dict, ...]} 格式的缺口信息
    """
    from data.fetcher import KlineFetcher
    from data.writer import DataWriter

    print("\n  --- 自动修复 K线缺口 ---")
    fetcher = KlineFetcher()
    writer = DataWriter()

    for symbol, gaps in all_gaps.items():
        for gap in gaps:
            start = gap["start"]
            end = gap["end"]
            missing = gap["missing_bars"]

            print(f"  回填 {symbol}: {start} ~ {end} (缺 {missing} 根) ...", end=" ")
            try:
                df = fetcher.fetch_ohlcv_batch(symbol, "1m", start, end)
                count = writer.write_ohlcv(df, symbol, "1m")
                print(f"写入 {count} 根")
            except Exception as e:
                print(f"失败: {e}")
                logger.error(f"K线缺口回填失败 {symbol}: {e}", exc_info=True)


def check_ticks(tick_store: TickStore, symbols: list[str]):
    """
    检查逐笔成交数据

    包括文件统计和交易所同步状态对比。
    """
    print("\n=== 逐笔成交巡检 ===")

    # 尝试获取交易所最新 trade_id 进行同步状态对比
    exchange_latest_ids = _fetch_exchange_latest_trade_ids(symbols)

    for symbol in symbols:
        latest_id = tick_store.get_latest_trade_id(symbol)

        # 统计文件数和大小
        sym_dir = tick_store._symbol_dir(symbol)
        files = [f for f in os.listdir(sym_dir) if f.endswith(".parquet")] if os.path.exists(sym_dir) else []
        total_size = sum(
            os.path.getsize(os.path.join(sym_dir, f))
            for f in files
        ) if files else 0

        # 同步状态
        exchange_id = exchange_latest_ids.get(symbol)
        sync_info = check_tick_sync_status(
            tick_store, symbol, exchange_latest_id=exchange_id
        )
        status = sync_info["status"]
        lag = sync_info.get("lag")
        lag_str = f", 落后 {lag:,} 笔" if lag is not None else ""

        print(
            f"  {symbol}: "
            f"latest_id={latest_id or '无数据'}, "
            f"状态={status}{lag_str}, "
            f"{len(files)} 个文件, "
            f"{total_size / 1024 / 1024:.1f} MB"
        )


def _fetch_exchange_latest_trade_ids(symbols: list[str]) -> dict:
    """
    从交易所获取各币对最新 trade_id

    使用异步 TickFetcher 获取最新一批 trades，提取最大 trade_id。
    如果获取失败（如网络问题），返回空 dict，不影响巡检其他项。

    Returns:
        {symbol: latest_trade_id} 字典
    """
    try:
        from data.tick_fetcher import TickFetcher

        async def _fetch():
            fetcher = TickFetcher()
            result = {}
            try:
                for symbol in symbols:
                    try:
                        raw = await fetcher.exchange.fetch_trades(symbol, limit=1)
                        if raw:
                            result[symbol] = max(int(t["id"]) for t in raw)
                    except Exception as e:
                        logger.debug(f"获取 {symbol} 最新 trade_id 失败: {e}")
            finally:
                await fetcher.close()
            return result

        return asyncio.run(_fetch())
    except Exception as e:
        logger.debug(f"获取交易所 trade_id 失败，跳过同步对比: {e}")
        return {}


def check_orderbook(symbols: list[str]):
    """检查订单簿数据"""
    print("\n=== 订单簿巡检 ===")

    for symbol in symbols:
        safe_name = symbol.replace("/", "_")
        sym_dir = os.path.join(settings.ORDERBOOK_DATA_DIR, safe_name)

        if not os.path.exists(sym_dir):
            print(f"  {symbol}: 无数据")
            continue

        files = sorted([f for f in os.listdir(sym_dir) if f.endswith(".parquet")])
        total_size = sum(
            os.path.getsize(os.path.join(sym_dir, f))
            for f in files
        )

        latest_file = files[-1] if files else "无"
        print(
            f"  {symbol}: "
            f"{len(files)} 个文件, "
            f"{total_size / 1024 / 1024:.1f} MB, "
            f"最新文件: {latest_file}"
        )


def check_market(market_store: MarketStore, symbols: list[str]):
    """检查合约市场数据"""
    print("\n=== 合约市场数据巡检 ===")

    tables = ["funding_rates", "open_interest", "long_short_ratio", "taker_buy_sell"]

    for symbol in symbols:
        parts = []
        for table in tables:
            latest = market_store.get_latest_timestamp(table, symbol)
            short_name = table.replace("_", " ")
            if latest:
                delay = (datetime.now(timezone.utc) - latest).total_seconds()
                parts.append(f"{short_name}: {delay:.0f}s前")
            else:
                parts.append(f"{short_name}: 无")

        print(f"  {symbol}: {', '.join(parts)}")


def check_kline_cross_validation(kline_store: KlineStore, symbols: list[str]):
    """
    跨源 K线 对比

    随机抽取若干时间点，调用 API 获取数据，与本地存储比对。
    用于检测本地数据是否与交易所一致（排除存储损坏或写入错误）。
    """
    from data.fetcher import KlineFetcher

    print("\n=== 跨源 K线 对比 ===")

    fetcher = KlineFetcher()
    sample_count = 3  # 每个币对抽样 3 个时间点

    for symbol in symbols:
        latest = kline_store.get_latest_timestamp(symbol, "1m")
        if latest is None:
            print(f"  {symbol}: 无本地数据，跳过")
            continue

        # 在本地数据范围内随机抽取时间点
        df_all = kline_store.read(symbol, "1m")
        if len(df_all) < 10:
            print(f"  {symbol}: 数据量过少 ({len(df_all)} 根)，跳过")
            continue

        # 随机选取 sample_count 个索引位置
        indices = random.sample(range(len(df_all)), min(sample_count, len(df_all)))
        mismatches = 0
        checked = 0

        for idx in indices:
            local_row = df_all.iloc[idx]
            ts = local_row["timestamp"]

            try:
                # 从 API 获取该时间点的 K线
                from utils.time_utils import datetime_to_ms
                since_ms = datetime_to_ms(ts.to_pydatetime())
                api_df = fetcher.fetch_ohlcv(symbol, "1m", since=since_ms, limit=1)

                if api_df.empty:
                    continue

                api_row = api_df.iloc[0]
                checked += 1

                # 比对 OHLCV（允许浮点精度误差）
                for col in ["open", "high", "low", "close", "volume"]:
                    local_val = float(local_row[col])
                    api_val = float(api_row[col])
                    if abs(local_val - api_val) > 1e-8 * max(abs(local_val), 1):
                        mismatches += 1
                        logger.warning(
                            f"K线不一致 {symbol} {ts}: "
                            f"{col} 本地={local_val} API={api_val}"
                        )
                        break

            except Exception as e:
                logger.debug(f"跨源对比 {symbol} {ts} 失败: {e}")

        if checked == 0:
            print(f"  {symbol}: 无法获取 API 数据进行对比")
        elif mismatches == 0:
            print(f"  {symbol}: 抽样 {checked} 个时间点，全部一致 ✓")
        else:
            print(f"  {symbol}: 抽样 {checked} 个时间点，{mismatches} 处不一致 ✗")


def main():
    args = parse_args()
    symbols = args.symbols or settings.SYMBOLS

    print(f"数据质量巡检 ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')})")
    print(f"交易对: {symbols}")

    kline_store = KlineStore()
    tick_store = TickStore()
    market_store = MarketStore()

    check_klines(kline_store, symbols, fix=args.fix)
    check_ticks(tick_store, symbols)
    check_orderbook(symbols)
    check_market(market_store, symbols)
    check_kline_cross_validation(kline_store, symbols)

    # --fix 对 tick/orderbook 无法自动修复，给出提示
    if args.fix:
        print("\n--- 修复提示 ---")
        print("  Tick 缺失: 请手动执行 python -m scripts.backfill --type tick --from-id <id>")
        print("  订单簿: 无法回填（Binance 不提供历史订单簿 API）")

    print("\n巡检完成。")


if __name__ == "__main__":
    main()
