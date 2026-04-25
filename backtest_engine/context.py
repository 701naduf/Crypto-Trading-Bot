"""
MarketContextBuilder — 市场状态工厂

详见 docs/phase3_design.md §11.2。

职责：
  - 从 DataReader 读取原始行情，预聚合为 4 个面板：
      price_panel  / spread_panel / adv_panel / vol_panel
    以及一个 funding_panel（每 bar 摊销值，仅事件驱动模式用）。
  - 对外提供两种消费接口：
      build(t, ...)   逐步模式（事件循环每 bar 调用）
      build_panels()  批量模式（VECTORIZED + 持久化 + per-symbol cost 重算共用）

关键约束（§11.2.5）:
  - σ 是日化（日级标准差），与 Almgren-Chriss 配套
  - ADV 是 USD 单位（NaN-safe rolling sum/count）
  - spread 是 (ask-bid)/mid 的比率
  - funding_rate 是"下一次结算 rate / bars_per_8h"摊销值（不是真实 8h 事件）
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from execution_optimizer.config import MarketContext

if TYPE_CHECKING:
    from data_infra.data.reader import DataReader
    from backtest_engine.config import BacktestConfig

logger = logging.getLogger(__name__)


# §11.2.5 默认 min_periods 阈值（约 3.5 天 1m 数据，经验值）
DEFAULT_VOL_MIN_PERIODS = 5000


def _to_pd_freq(bar_freq: str) -> str:
    """
    把项目语义 bar_freq（"1m"/"5m"/"1h"）翻译为 pandas date_range freq 字符串。

    pandas >= 2.2 起 "1m" 表示 month-end（不是 minute），必须用 "1min"。
    """
    mapping = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h"}
    return mapping.get(bar_freq, bar_freq)


class MarketContextBuilder:
    """
    市场状态工厂

    `__init__` 立即预加载所有面板（取舍 A1，§11.2.4）；构造完成后不再访问 reader。
    """

    def __init__(
        self,
        reader: "DataReader",
        config: "BacktestConfig",
        *,
        spread_ffill_max_minutes: int = 5,
        lookback_days: int = 21,
        vol_min_periods: int = DEFAULT_VOL_MIN_PERIODS,
    ) -> None:
        """
        Args:
            reader:                   Phase 1 DataReader（须能 get_ohlcv / get_orderbook /
                                      get_funding_rate）
            config:                   BacktestConfig，提供 symbols / start / end / bar_freq /
                                      periods_per_year
            spread_ffill_max_minutes: orderbook gap 超过此阈值时 spread = NaN（不掩盖断线）
            lookback_days:            预热期天数（§11.2.4 取舍 C1：默认 21 天）
            vol_min_periods:          rolling std 最小有效样本量（避免短窗口估计误差过大）
        """
        self._config = config
        self._symbols = list(config.symbols)
        self._start = config.start
        self._end = config.end
        self._bar_freq = config.bar_freq
        self._periods_per_year = config.periods_per_year
        self._spread_ffill_max_minutes = spread_ffill_max_minutes
        self._lookback_days = lookback_days
        self._vol_min_periods = vol_min_periods

        # 推导关键尺度（§11.2.5）
        self._bars_per_day = self._periods_per_year / 365.25
        self._bars_per_8h = self._bars_per_day / 3.0

        # 实际 bar timestamps（含 lookback 热身期，用于 rolling 计算）
        self._earliest_needed = self._start - pd.Timedelta(days=lookback_days)
        pd_freq = _to_pd_freq(self._bar_freq)
        self._full_bar_index = pd.date_range(
            self._earliest_needed, self._end, freq=pd_freq, tz="UTC",
        )
        # 回测使用的 bar_timestamps（[start, end] 闭区间）
        self._eval_bar_index = pd.date_range(
            self._start, self._end, freq=pd_freq, tz="UTC",
        )

        # ── 预加载面板（取舍 A1） ──
        logger.info(
            "MarketContextBuilder __init__: %d 个 symbol × %d bars (含 %d 天热身)",
            len(self._symbols), len(self._full_bar_index), lookback_days,
        )

        self._price_panel = self._load_price_panel(reader)
        self._volume_panel = self._load_volume_panel(reader)

        # vol / ADV 面板（基于 OHLCV）
        self._vol_panel = self._compute_vol_panel(self._price_panel)
        self._adv_panel = self._compute_adv_panel(self._price_panel, self._volume_panel)

        # spread 面板（聚合 orderbook 后丢弃原始数据，B2 取舍）
        self._spread_panel = self._load_spread_panel(reader)

        # funding 面板（事件驱动模式才有意义；VECTORIZED 不读但保留接口一致）
        self._funding_panel = self._load_funding_panel(reader)

        # 数据完整性校验：若 [start, ...] 仍不足 vol_min_periods，警告
        first_eval_idx = self._vol_panel.index.searchsorted(self._start)
        if first_eval_idx < self._vol_min_periods:
            logger.warning(
                "热身期可能不够：start 之前仅 %d bars (vol_min_periods=%d)；"
                "回测开头部分 bar 的 σ 估计可能为 NaN",
                first_eval_idx, self._vol_min_periods,
            )

    # ------------------------------------------------------------------
    # 数据加载（私有）
    # ------------------------------------------------------------------

    def _load_price_panel(self, reader: "DataReader") -> pd.DataFrame:
        """读取每个 symbol 的 close 价，组装为 timestamp × symbol 面板"""
        cols = {}
        earliest_in_data: pd.Timestamp | None = None
        for sym in self._symbols:
            df = reader.get_ohlcv(
                sym, self._bar_freq, start=self._earliest_needed, end=self._end,
            )
            if df is None or len(df) == 0:
                raise KeyError(
                    f"DataReader 没有 {sym} 的 OHLCV 数据 "
                    f"({self._earliest_needed} ~ {self._end})"
                )
            df = df.set_index("timestamp") if "timestamp" in df.columns else df
            cols[sym] = df["close"]
            sym_earliest = df.index.min()
            if earliest_in_data is None or sym_earliest > earliest_in_data:
                earliest_in_data = sym_earliest

        # 检查热身期是否足够（基于原始数据的最早时间，而非 reindex 后的索引）
        if earliest_in_data is not None and earliest_in_data > self._earliest_needed:
            raise ValueError(
                f"价格数据起点 {earliest_in_data} 晚于所需热身期开始 "
                f"{self._earliest_needed}（lookback_days={self._lookback_days}）；"
                f"请增加数据采集或缩短 lookback_days"
            )

        panel = pd.DataFrame(cols)
        # 索引到全 bar 网格（缺失保留为 NaN，由 rolling 计算/校验自然处理）
        panel = panel.reindex(self._full_bar_index)
        return panel

    def _load_volume_panel(self, reader: "DataReader") -> pd.DataFrame:
        """读取每个 symbol 的 volume（USDT 量纲：close × volume_quote 或直接 volume）"""
        cols = {}
        for sym in self._symbols:
            df = reader.get_ohlcv(
                sym, self._bar_freq, start=self._earliest_needed, end=self._end,
            )
            df = df.set_index("timestamp") if "timestamp" in df.columns else df
            cols[sym] = df["volume"]

        panel = pd.DataFrame(cols).reindex(self._full_bar_index)
        return panel

    def _compute_vol_panel(self, price_panel: pd.DataFrame) -> pd.DataFrame:
        """
        日化 σ：returns.rolling(20*bars_per_day, min_periods=vol_min_periods).std() × √bars_per_day

        与 Almgren-Chriss 约定一致（σ 单位 = 1/√day）
        """
        window = max(int(round(20 * self._bars_per_day)), 1)
        returns = price_panel.pct_change()
        vol = returns.rolling(window, min_periods=self._vol_min_periods).std()
        vol = vol * np.sqrt(self._bars_per_day)
        return vol

    def _compute_adv_panel(
        self, price_panel: pd.DataFrame, volume_panel: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        ADV (USD)：NaN-safe (rolling_sum / rolling_count) × bars_per_day

        禁止用 sum/N 简化，避免窗口内 NaN 被当 0 导致 ADV 系统性低估。
        """
        notional = price_panel * volume_panel
        window = max(int(round(20 * self._bars_per_day)), 1)
        rolling_sum = notional.rolling(window, min_periods=self._vol_min_periods).sum()
        rolling_cnt = notional.rolling(window, min_periods=self._vol_min_periods).count()
        # 避免 0 计数除零
        adv = (rolling_sum / rolling_cnt.replace(0, np.nan)) * self._bars_per_day
        return adv

    def _load_spread_panel(self, reader: "DataReader") -> pd.DataFrame:
        """
        Spread 面板：每 bar 取 bar_close 时点之前最近一条 orderbook 快照的 (ask-bid)/mid

        B2 取舍：聚合后丢弃原始 orderbook 数据；ffill with max_gap 容忍小规模采集断线，
        gap 超过 spread_ffill_max_minutes 时 spread = NaN（不掩盖大规模断线）。
        """
        cols = {}
        max_gap = pd.Timedelta(minutes=self._spread_ffill_max_minutes)
        for sym in self._symbols:
            try:
                ob = reader.get_orderbook(
                    sym, start=self._earliest_needed, end=self._end, levels=1,
                )
            except Exception as e:
                logger.warning("orderbook 读取失败 sym=%s: %s；spread 全 NaN", sym, e)
                cols[sym] = pd.Series(np.nan, index=self._full_bar_index, dtype=float)
                continue

            if ob is None or len(ob) == 0:
                logger.warning("orderbook 为空 sym=%s；spread 全 NaN", sym)
                cols[sym] = pd.Series(np.nan, index=self._full_bar_index, dtype=float)
                continue

            ob = ob.set_index("timestamp") if "timestamp" in ob.columns else ob
            mid = (ob["ask_price_0"] + ob["bid_price_0"]) / 2.0
            spread_raw = (ob["ask_price_0"] - ob["bid_price_0"]) / mid
            spread_raw = spread_raw.dropna().sort_index()

            # Step 7 / B2: 仅 spread 用 bar_close 偏移
            # 设计 §11.2.5 + §八.8 / E.1: t 时刻语义 = bar_close（bar_open + bar_freq）
            # 取 bar_close 之前最近一条快照，tolerance 控制 max_gap
            close_offset = pd.Timedelta(_to_pd_freq(self._bar_freq))
            spread_aligned = self._snapshot_to_bar(
                spread_raw, self._full_bar_index, max_gap,
                close_offset=close_offset,
            )
            cols[sym] = spread_aligned

        return pd.DataFrame(cols, index=self._full_bar_index)

    @staticmethod
    def _snapshot_to_bar(
        snapshot_series: pd.Series,
        bar_index: pd.DatetimeIndex,
        max_gap: pd.Timedelta,
        close_offset: pd.Timedelta | None = None,
    ) -> pd.Series:
        """
        把高频快照对齐到 bar 网格：取 bar_t 之前最近一条；gap 超过 max_gap 时返回 NaN。

        Args:
            close_offset: Step 7 / B2 / Q3：仅 spread 传入；查询时点 = bar_index + offset
                          （bar_close 时刻）。其他面板（vol/adv/funding/price）不用此偏移。
                          None 时退化为旧行为（用 bar_index 直接查询）。

        实现：merge_asof(direction='backward', tolerance=max_gap)。
        """
        if len(snapshot_series) == 0:
            return pd.Series(np.nan, index=bar_index, dtype=float)

        # Step 7: 仅 spread 偏移到 bar_close（bar_open + bar_freq）
        query_ts = bar_index if close_offset is None else (bar_index + close_offset)

        snap_df = pd.DataFrame({
            "ts": snapshot_series.index,
            "value": snapshot_series.values,
        }).sort_values("ts")
        bar_df = pd.DataFrame({"ts": query_ts}).sort_values("ts")

        merged = pd.merge_asof(
            bar_df, snap_df, on="ts",
            direction="backward",
            tolerance=max_gap,
            allow_exact_matches=True,
        )
        # 返回值仍以 bar_index（bar_open）为 index，方便面板对齐
        # 注意：merged 已被 sort_values("ts") 排序，需要 reorder 回 bar_index 顺序
        merged_indexed = pd.Series(merged["value"].values, index=query_ts)
        return pd.Series(
            merged_indexed.reindex(query_ts).values,
            index=bar_index, dtype=float,
        )

    def _load_funding_panel(self, reader: "DataReader") -> pd.DataFrame:
        """
        每 bar 摊销 funding rate：next_settlement_rate / bars_per_8h

        实现：
          对每 symbol 读真实 funding rate 序列（index=settlement_ts，value=rate_per_8h）；
          每个 bar 找"下一次结算"的 rate（searchsorted side='right'）；
          除以 bars_per_8h 得每 bar 摊销成本；
          区间内已无下一次结算的 bar → NaN。
        """
        cols = {}
        bar_ts = self._full_bar_index
        for sym in self._symbols:
            try:
                df = reader.get_funding_rate(
                    sym, start=self._earliest_needed, end=self._end,
                )
            except Exception as e:
                logger.warning("funding_rate 读取失败 sym=%s: %s；funding 全 NaN", sym, e)
                cols[sym] = pd.Series(np.nan, index=bar_ts, dtype=float)
                continue

            if df is None or len(df) == 0:
                cols[sym] = pd.Series(np.nan, index=bar_ts, dtype=float)
                continue

            settle_ts = pd.to_datetime(df["timestamp"], utc=True).values
            settle_rates = df["funding_rate"].astype(float).values
            # 排序保险（DataReader 已 ASC，但显式保证）
            order = np.argsort(settle_ts)
            settle_ts = settle_ts[order]
            settle_rates = settle_rates[order]

            # 每个 bar 找严格大于自己的下一次结算（side='right'）
            bar_ts_np = bar_ts.values
            idx = np.searchsorted(settle_ts, bar_ts_np, side="right")
            valid = idx < len(settle_ts)
            next_rates = np.full(len(bar_ts), np.nan)
            next_rates[valid] = settle_rates[idx[valid]]

            cols[sym] = pd.Series(
                next_rates / self._bars_per_8h, index=bar_ts, dtype=float,
            )

        return pd.DataFrame(cols, index=bar_ts)

    # ------------------------------------------------------------------
    # 公共接口（消费）
    # ------------------------------------------------------------------

    def build(
        self,
        t: pd.Timestamp,
        current_weights: pd.Series,  # 保留参数（D1 前瞻设计），当前不使用
        portfolio_value: float,
    ) -> MarketContext:
        """
        逐步模式：事件驱动每 bar 调用一次。

        Args:
            t:                当前 bar timestamp
            current_weights:  调用方持有；当前 MarketContext 不消费此字段（D1 保留）
            portfolio_value:  调用方传入（PnLTracker.portfolio_value）

        Returns:
            MarketContext (execution_optimizer.config.MarketContext)
        """
        del current_weights  # 显式声明不使用，未来扩展（逐 symbol impact_coeff）会用

        spread = self._spread_panel.loc[t]
        vol = self._vol_panel.loc[t]
        adv = self._adv_panel.loc[t]

        funding = self._funding_panel.loc[t] if t in self._funding_panel.index else None
        if funding is not None and funding.isna().all():
            funding = None

        return MarketContext(
            timestamp=t,
            symbols=list(self._symbols),
            spread=spread.copy(),
            volatility=vol.copy(),
            adv=adv.copy(),
            portfolio_value=portfolio_value,
            funding_rate=(funding.copy() if funding is not None else None),
        )

    def build_panels(self) -> dict[str, pd.DataFrame]:
        """
        批量模式：VECTORIZED / B3 持久化 / compute_per_symbol_cost 共用

        Returns:
            dict（canonical keys，§12.6 对齐）:
              "price_panel":  全 bar × symbols close 价（仅 VECTORIZED 用）
              "spread_panel": 比率 (ask-bid)/mid，bps 形式
              "adv_panel":    USDT 单位
              "vol_panel":    日化 σ

        说明：
          - 切到 [start, end] 评估区间（去掉热身期）
          - 不返回 funding_panel（VECTORIZED 不处理 funding，属已知偏差）
        """
        idx = self._eval_bar_index
        return {
            "price_panel":  self._price_panel.reindex(idx),
            "spread_panel": self._spread_panel.reindex(idx),
            "adv_panel":    self._adv_panel.reindex(idx),
            "vol_panel":    self._vol_panel.reindex(idx),
        }

    # ------------------------------------------------------------------
    # 调试 / 测试 friendly properties（不属于 API 契约）
    # ------------------------------------------------------------------

    @property
    def bar_index(self) -> pd.DatetimeIndex:
        """全 bar 索引（含热身期）"""
        return self._full_bar_index

    @property
    def eval_bar_index(self) -> pd.DatetimeIndex:
        """评估区间 [start, end] 的 bar 索引"""
        return self._eval_bar_index
