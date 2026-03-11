"""
数据质量校验模块

提供两类校验能力:

1. 写入校验（实时）—— 在数据写入存储前调用，过滤不合格数据:
    - K线 (OHLCV): OHLC > 0, volume >= 0, high >= max(O,C), low <= min(O,C)
    - 逐笔成交 (Tick): price > 0, amount > 0, side in ("buy","sell"), trade_id > 0
    - 订单簿: 所有价格 > 0, 数量 >= 0, bid_0 < ask_0（买一低于卖一）
    - 市场数据: 各指标在合理范围内

2. 完整性巡检（定期）—— 由 check_data.py 调用:
    - K线连续性: 检测 1m K线的时间缺口
    - Tick 同步状态: 本地 vs 交易所最新 trade_id
    - 跨源对比: 随机抽样与 API 比对

校验规则设计原则:
    - 宁可漏放，不可误杀: 只过滤明显不合理的数据（如负数价格）
    - 过滤后记录日志: 方便排查数据源问题
    - 返回值分为 valid 和 invalid: 让调用方决定如何处理

依赖: utils.logger
"""

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


def validate_ohlcv(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    校验 K线 (OHLCV) 数据

    校验规则:
        1. open, high, low, close 必须 > 0
        2. volume 必须 >= 0
        3. high >= max(open, close)  — 最高价不低于开/收盘价
        4. low  <= min(open, close)  — 最低价不高于开/收盘价

    Args:
        df: 包含 [timestamp, open, high, low, close, volume] 列的 DataFrame

    Returns:
        (valid_df, invalid_df) —— 分别是通过和未通过校验的行
        invalid_df 为空 DataFrame 时表示全部通过
    """
    if df.empty:
        return df, df.iloc[0:0]

    # 规则 1: OHLC > 0
    positive_price = (
        (df["open"] > 0) &
        (df["high"] > 0) &
        (df["low"] > 0) &
        (df["close"] > 0)
    )

    # 规则 2: volume >= 0
    valid_volume = df["volume"] >= 0

    # 规则 3 & 4: high/low 与 open/close 的关系
    # high 应 >= open 和 close 中的较大值
    # low  应 <= open 和 close 中的较小值
    oc_max = df[["open", "close"]].max(axis=1)
    oc_min = df[["open", "close"]].min(axis=1)
    valid_range = (df["high"] >= oc_max) & (df["low"] <= oc_min)

    # 综合判定
    mask = positive_price & valid_volume & valid_range

    valid_df = df[mask].copy()
    invalid_df = df[~mask].copy()

    if not invalid_df.empty:
        logger.warning(
            f"OHLCV 校验: {len(invalid_df)} 行未通过 "
            f"(共 {len(df)} 行, 通过率 {len(valid_df)/len(df)*100:.1f}%)"
        )

    return valid_df, invalid_df


def validate_ticks(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    校验逐笔成交数据

    校验规则:
        1. trade_id > 0
        2. price > 0
        3. amount > 0
        4. side 必须是 "buy" 或 "sell"

    Args:
        df: 包含 [trade_id, timestamp, price, amount, side] 列的 DataFrame

    Returns:
        (valid_df, invalid_df)
    """
    if df.empty:
        return df, df.iloc[0:0]

    mask = (
        (df["trade_id"] > 0) &
        (df["price"] > 0) &
        (df["amount"] > 0) &
        (df["side"].isin(["buy", "sell"]))
    )

    valid_df = df[mask].copy()
    invalid_df = df[~mask].copy()

    if not invalid_df.empty:
        logger.warning(
            f"Tick 校验: {len(invalid_df)} 行未通过 "
            f"(共 {len(df)} 行, 通过率 {len(valid_df)/len(df)*100:.1f}%)"
        )

    return valid_df, invalid_df


def validate_orderbook(snapshot: dict, depth: int) -> bool:
    """
    校验单条订单簿快照

    校验规则:
        1. bids 和 asks 列表长度 == depth
        2. 所有价格 > 0
        3. 所有数量 >= 0
        4. bid_0 (买一价) < ask_0 (卖一价) —— 买卖盘不交叉

    Args:
        snapshot: 订单簿快照字典:
                  {"timestamp": datetime,
                   "bids": [[price, qty], ...],  # 价格降序
                   "asks": [[price, qty], ...]}  # 价格升序
        depth: 期望的档位深度（如 10）

    Returns:
        True 表示有效，False 表示无效
    """
    bids = snapshot.get("bids", [])
    asks = snapshot.get("asks", [])

    # 规则 1: 档位数量
    if len(bids) != depth or len(asks) != depth:
        logger.debug(
            f"订单簿档位不足: bids={len(bids)}, asks={len(asks)}, 期望={depth}"
        )
        return False

    # 规则 2 & 3: 价格 > 0, 数量 >= 0
    for side_name, levels in [("bids", bids), ("asks", asks)]:
        for price, qty in levels:
            if price <= 0:
                logger.debug(f"订单簿 {side_name} 价格异常: {price}")
                return False
            if qty < 0:
                logger.debug(f"订单簿 {side_name} 数量异常: {qty}")
                return False

    # 规则 4: 买一 < 卖一（买卖盘不交叉）
    best_bid = bids[0][0]
    best_ask = asks[0][0]
    if best_bid >= best_ask:
        logger.debug(f"订单簿买卖交叉: bid_0={best_bid} >= ask_0={best_ask}")
        return False

    return True


def validate_open_interest(data: dict) -> bool:
    """
    校验单条持仓量快照（dict 格式）

    持仓量数据以 dict 形式传入（非 DataFrame），需要单独的校验函数。

    校验规则:
        1. "timestamp" 键存在且值非 None
        2. open_interest > 0
        3. open_interest_value > 0

    Args:
        data: 持仓量快照字典:
              {"timestamp": datetime, "open_interest": float,
               "open_interest_value": float}

    Returns:
        True 表示有效，False 表示无效
    """
    # 规则 1: timestamp 存在且合法
    ts = data.get("timestamp")
    if ts is None:
        logger.warning("持仓量校验失败: timestamp 缺失或为 None")
        return False

    # 规则 2: 持仓量 > 0
    oi = data.get("open_interest", 0)
    if not isinstance(oi, (int, float)) or oi <= 0:
        logger.warning(f"持仓量校验失败: open_interest = {oi}")
        return False

    # 规则 3: 持仓价值 > 0
    oi_value = data.get("open_interest_value", 0)
    if not isinstance(oi_value, (int, float)) or oi_value <= 0:
        logger.warning(f"持仓量校验失败: open_interest_value = {oi_value}")
        return False

    return True


def validate_market_data(
    df: pd.DataFrame, data_type: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    校验合约市场数据

    根据 data_type 应用不同的校验规则:
        - "funding_rate":  资金费率在 [-0.1, 0.1] 范围内（极端情况下可达 ±5%）
        - "open_interest": 持仓量 > 0, 持仓价值 > 0
        - "long_short_ratio": 比率 > 0
        - "taker_buy_sell": 成交量 >= 0

    Args:
        df:        待校验的 DataFrame
        data_type: 数据类型标识

    Returns:
        (valid_df, invalid_df)
    """
    if df.empty:
        return df, df.iloc[0:0]

    if data_type == "funding_rate":
        # 资金费率通常在 -0.01 ~ 0.01，极端情况可达 ±0.05
        # 放宽到 ±0.1 以避免误杀
        mask = (df["funding_rate"].abs() <= 0.1)

    elif data_type == "open_interest":
        mask = (
            (df["open_interest"] > 0) &
            (df["open_interest_value"] > 0)
        )

    elif data_type == "long_short_ratio":
        mask = (
            (df["long_ratio"] >= 0) &
            (df["short_ratio"] >= 0) &
            (df["long_short_ratio"] > 0)
        )

    elif data_type == "taker_buy_sell":
        mask = (
            (df["buy_vol"] >= 0) &
            (df["sell_vol"] >= 0)
        )

    else:
        logger.warning(f"未知的市场数据类型: {data_type}，跳过校验")
        return df, df.iloc[0:0]

    valid_df = df[mask].copy()
    invalid_df = df[~mask].copy()

    if not invalid_df.empty:
        logger.warning(
            f"{data_type} 校验: {len(invalid_df)} 行未通过 "
            f"(共 {len(df)} 行)"
        )

    return valid_df, invalid_df


# =========================================================================
# 完整性巡检函数（由 check_data.py 调用）
# =========================================================================

def check_kline_continuity(
    store, symbol: str, timeframe: str = "1m"
) -> list[dict]:
    """
    检查 K线 时间连续性，找出缺口

    扫描指定币对的全部 K线，检测相邻两根 K线 之间是否存在
    超过一个周期的间隔（即缺失的 K线）。

    Args:
        store:     KlineStore 实例
        symbol:    交易对，如 "BTC/USDT"
        timeframe: K线周期，默认 "1m"

    Returns:
        缺口列表，每个元素:
        {
            "start": datetime,  # 缺口开始时间
            "end":   datetime,  # 缺口结束时间
            "missing_bars": int # 缺失的 K线 根数
        }
        空列表表示无缺口
    """
    from utils.time_utils import timeframe_to_seconds

    df = store.read(symbol, timeframe)
    if df.empty or len(df) < 2:
        return []

    # 计算相邻 K线 的时间差
    period_seconds = timeframe_to_seconds(timeframe)
    timestamps = df["timestamp"].sort_values()

    gaps = []
    for i in range(1, len(timestamps)):
        diff = (timestamps.iloc[i] - timestamps.iloc[i - 1]).total_seconds()
        expected = period_seconds

        # 允许 1 秒误差（时间戳精度问题）
        if diff > expected + 1:
            missing = int(diff / period_seconds) - 1
            gaps.append({
                "start": timestamps.iloc[i - 1],
                "end": timestamps.iloc[i],
                "missing_bars": missing,
            })

    if gaps:
        total_missing = sum(g["missing_bars"] for g in gaps)
        logger.info(
            f"{symbol} {timeframe} K线连续性检查: "
            f"发现 {len(gaps)} 处缺口，共缺失 {total_missing} 根"
        )
    else:
        logger.info(f"{symbol} {timeframe} K线连续性检查: 无缺口")

    return gaps


def check_tick_sync_status(
    store, symbol: str, exchange_latest_id: int | None = None
) -> dict:
    """
    检查逐笔成交的同步状态

    对比本地最新 trade_id 和交易所最新 trade_id，
    计算延迟的笔数。

    交易所最新 trade_id 由调用方负责获取并传入，
    本函数只做对比和状态判定。

    Args:
        store:              TickStore 实例
        symbol:             交易对
        exchange_latest_id: 交易所最新 trade_id（由调用方获取）

    Returns:
        {
            "symbol": str,
            "local_latest_id": int | None,
            "exchange_latest_id": int | None,
            "lag": int | None,          # 落后的笔数（估算）
            "status": str               # "synced" | "lagging" | "no_data" | "unknown"
        }
    """
    local_id = store.get_latest_trade_id(symbol)

    result = {
        "symbol": symbol,
        "local_latest_id": local_id,
        "exchange_latest_id": exchange_latest_id,
        "lag": None,
        "status": "no_data" if local_id is None else "unknown",
    }

    # 本地无数据，直接返回
    if local_id is None:
        return result

    # 无法获取交易所最新 ID，状态未知
    if exchange_latest_id is None:
        result["status"] = "unknown"
        return result

    # 计算 lag（trade_id 之差是估算值，因为 ID 可能不连续）
    lag = exchange_latest_id - local_id
    result["lag"] = lag

    if lag <= 0:
        result["status"] = "synced"
    else:
        result["status"] = "lagging"

    logger.info(
        f"{symbol} Tick 同步: 本地={local_id}, "
        f"交易所={exchange_latest_id}, lag={lag}, "
        f"状态={result['status']}"
    )

    return result
