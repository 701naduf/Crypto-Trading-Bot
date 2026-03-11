"""
订单簿快照采集模块（异步 WebSocket）

通过 Binance WebSocket 订阅 Partial Book Depth Stream，
接收 100ms 粒度的 10 档订单簿快照推送。

这是本项目中唯一使用 WebSocket 的模块。与 REST API 轮询不同，
WebSocket 是被动接收推送数据，需要维护长连接。

WebSocket 流名称: <symbol>@depth<levels>@<speed>
    例: btcusdt@depth10@100ms
    - symbol: 小写，无分隔符（"btcusdt" 而非 "BTC/USDT"）
    - levels: 5 / 10 / 20
    - speed:  100ms / 1000ms

Binance WebSocket 端点:
    wss://stream.binance.com:9443/ws           (单个流)
    wss://stream.binance.com:9443/stream?streams=  (多个流，合并连接)

连接管理:
    - 自动心跳: 定期发送 ping 帧保持连接活跃
    - 断线重连: 指数退避（WS_RECONNECT_DELAY × 2^n，上限 60 秒）
    - 多币对: 使用 combined stream 订阅多个币对，共用单个连接
    - 单个连接最多 200 个 stream

注意事项:
    - 订单簿是实时状态快照，无法回填历史
    - 从开始采集的那一刻起积累数据
    - WebSocket 断线期间的数据不可恢复

数据流: WebSocket 推送 → 解析 → 回调 (on_snapshot)
依赖: config.settings, utils.logger
被依赖: scripts/collect_orderbook.py
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Callable

import websockets

from data_infra.config import settings
from data_infra.utils.logger import get_logger

logger = get_logger(__name__)

# Binance WebSocket combined stream 端点
BINANCE_WS_URL = "wss://stream.binance.com:9443/stream?streams="


class OrderbookFetcher:
    """
    订单簿快照采集器（异步 WebSocket）

    维护 WebSocket 长连接，接收实时订单簿推送。
    通过 on_snapshot 回调将数据传给 DataWriter。
    """

    def __init__(
        self,
        symbols: list[str] = None,
        depth: int = None,
        speed: str = None,
    ):
        """
        初始化 WebSocket 采集器

        Args:
            symbols: 交易对列表，默认使用 settings.SYMBOLS
                     格式: ["BTC/USDT", "ETH/USDT", ...]
            depth:   订单簿档位深度（5/10/20），默认 settings.ORDERBOOK_DEPTH
            speed:   推送频率（"100ms"/"1000ms"），默认 settings.ORDERBOOK_UPDATE_SPEED
        """
        self.symbols = symbols or settings.SYMBOLS
        self.depth = depth or settings.ORDERBOOK_DEPTH
        self.speed = speed or settings.ORDERBOOK_UPDATE_SPEED

        # WebSocket 连接对象
        self._ws = None
        # 是否正在运行
        self._running = False
        # 重连计数器
        self._reconnect_count = 0

        # 构建 stream 名称映射: "btcusdt@depth10@100ms" → "BTC/USDT"
        # 用于将 WebSocket 消息中的 stream 名还原为标准交易对名
        self._stream_to_symbol = {}
        for symbol in self.symbols:
            stream_name = self._make_stream_name(symbol)
            self._stream_to_symbol[stream_name] = symbol

    def _make_stream_name(self, symbol: str) -> str:
        """
        将标准交易对名转为 Binance WebSocket stream 名称

        Args:
            symbol: 标准名称，如 "BTC/USDT"

        Returns:
            stream 名称，如 "btcusdt@depth10@100ms"
        """
        # "BTC/USDT" → "btcusdt"
        ws_symbol = symbol.replace("/", "").lower()
        return f"{ws_symbol}@depth{self.depth}@{self.speed}"

    def _build_ws_url(self) -> str:
        """
        构建 combined stream URL

        将所有币对的 stream 名用 "/" 连接:
        wss://stream.binance.com:9443/stream?streams=btcusdt@depth10@100ms/ethusdt@depth10@100ms/...
        """
        streams = "/".join(
            self._make_stream_name(s) for s in self.symbols
        )
        return BINANCE_WS_URL + streams

    async def start(self, on_snapshot: Callable):
        """
        启动 WebSocket 连接并开始接收数据

        此方法会阻塞（持续运行），直到调用 stop() 或发生不可恢复的错误。
        断线时会自动重连。

        Args:
            on_snapshot: 回调函数，每收到一个有效快照时调用
                签名: on_snapshot(symbol: str, data: dict) -> None
                data 格式:
                    {
                        "timestamp": datetime,          # 接收时间 (UTC)
                        "bids": [[price, qty], ...],    # 买盘，价格降序
                        "asks": [[price, qty], ...],    # 卖盘，价格升序
                    }
        """
        self._running = True
        self._reconnect_count = 0

        logger.info(
            f"订单簿采集启动: {len(self.symbols)} 个币对, "
            f"深度={self.depth}, 频率={self.speed}"
        )

        while self._running:
            try:
                await self._connect_and_receive(on_snapshot)
            except Exception as e:
                if not self._running:
                    # stop() 被调用，正常退出
                    break

                self._reconnect_count += 1
                wait = self._get_reconnect_delay()

                logger.warning(
                    f"WebSocket 断线 (第 {self._reconnect_count} 次): "
                    f"{type(e).__name__}: {e} | "
                    f"{wait:.1f}s 后重连"
                )

                await asyncio.sleep(wait)

        logger.info("订单簿采集已停止")

    async def _connect_and_receive(self, on_snapshot: Callable):
        """
        建立连接并持续接收消息

        Args:
            on_snapshot: 快照回调函数
        """
        url = self._build_ws_url()
        logger.info(f"WebSocket 连接中: {url[:80]}...")

        async with websockets.connect(
            url,
            ping_interval=settings.WS_PING_INTERVAL,
            ping_timeout=settings.WS_PING_INTERVAL * 2,
            close_timeout=10,
        ) as ws:
            self._ws = ws
            self._reconnect_count = 0  # 连接成功，重置计数

            logger.info("WebSocket 已连接，开始接收订单簿推送")

            async for message in ws:
                if not self._running:
                    break

                self._process_message(message, on_snapshot)

    def _process_message(self, message: str, on_snapshot: Callable):
        """
        处理单条 WebSocket 消息

        Binance combined stream 消息格式:
            {
                "stream": "btcusdt@depth10@100ms",
                "data": {
                    "lastUpdateId": 1234567890,
                    "bids": [["42000.00", "1.5"], ["41999.00", "0.8"], ...],
                    "asks": [["42001.00", "2.0"], ["42002.00", "1.2"], ...],
                }
            }

        注意: Binance 返回的价格和数量是字符串，需要转为 float。
        """
        try:
            msg = json.loads(message)
        except json.JSONDecodeError:
            logger.warning(f"WebSocket 消息解析失败: {message[:100]}")
            return

        stream_name = msg.get("stream", "")
        data = msg.get("data", {})

        # 查找对应的标准交易对名
        symbol = self._stream_to_symbol.get(stream_name)
        if symbol is None:
            logger.debug(f"未知的 stream: {stream_name}")
            return

        # 解析买卖盘数据: 字符串 → float
        # Binance 格式: [["42000.00", "1.5"], ...] → [[42000.0, 1.5], ...]
        bids = [[float(p), float(q)] for p, q in data.get("bids", [])]
        asks = [[float(p), float(q)] for p, q in data.get("asks", [])]

        snapshot = {
            "timestamp": datetime.now(timezone.utc),
            "bids": bids,
            "asks": asks,
        }

        on_snapshot(symbol, snapshot)

    def _get_reconnect_delay(self) -> float:
        """
        计算重连等待时间（指数退避）

        WS_RECONNECT_DELAY × 2^(n-1)，上限 60 秒。
        第1次: 5s, 第2次: 10s, 第3次: 20s, 第4次: 40s, 第5次+: 60s
        """
        delay = settings.WS_RECONNECT_DELAY * (2 ** (self._reconnect_count - 1))
        return min(delay, 60.0)

    async def stop(self):
        """
        停止采集，断开 WebSocket 连接

        设置 _running = False 后，receive 循环会在下一条消息后退出。
        同时关闭底层 WebSocket 连接。
        """
        self._running = False

        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass  # 忽略关闭时的异常

        logger.info("订单簿采集器已关闭")
