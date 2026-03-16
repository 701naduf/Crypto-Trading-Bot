"""
Alpha 管道

串联 预处理 → 训练 → 信号生成 → 组合构建 → 回测 的完整流程。

Notebook 中可手动逐步调用各模块，也可用 AlphaPipeline 一键执行。
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from factor_research.store.factor_store import FactorStore
from factor_research.evaluation.metrics import compute_forward_returns_panel

from alpha_model.config import DEFAULT_SYMBOLS
from alpha_model.core.types import (
    AlphaModel,
    ModelMeta,
    PortfolioConstraints,
    TrainConfig,
    TrainMode,
)
from alpha_model.preprocessing.alignment import align_factor_panels
from alpha_model.preprocessing.selection import select_factors, select_from_families
from alpha_model.preprocessing.transform import build_feature_matrix, build_pooled_target
from alpha_model.training.splitter import TimeSeriesSplitter
from alpha_model.training.walk_forward import WalkForwardEngine, WalkForwardResult
from alpha_model.signal.generator import generate_signal
from alpha_model.portfolio.constructor import PortfolioConstructor
from alpha_model.backtest.vectorized import vectorized_backtest
from alpha_model.backtest.performance import BacktestResult
from alpha_model.store.signal_store import SignalStore
from alpha_model.store.model_store import ModelStore

logger = logging.getLogger(__name__)


class AlphaPipeline:
    """
    一键管道

    支持两种因子指定方式:
    - factor_names: 直接指定因子名列表
    - factor_families: 指定因子族名，自动选出最优变体

    两者可组合使用。
    """

    def __init__(
        self,
        model: AlphaModel,
        train_config: TrainConfig,
        constraints: PortfolioConstraints,
        factor_names: list[str] | None = None,
        factor_families: list[str] | None = None,
        selection_params: dict | None = None,
        signal_method: str = "cross_sectional_zscore",
        signal_clip: float | None = None,
        max_factor_lookback: int = 0,
    ):
        """
        Args:
            model:              满足 AlphaModel 协议的模型对象
            train_config:       训练配置
            constraints:        组合约束配置
            factor_names:       直接指定的因子名列表
            factor_families:    因子族名列表（自动选最优变体）
            selection_params:   传递给 select_factors / select_from_families 的参数
            signal_method:      信号标准化方式
            signal_clip:        信号截断阈值（None = 不截断）
            max_factor_lookback: 因子最大 lookback 窗口
        """
        if factor_names is None and factor_families is None:
            raise ValueError("必须指定 factor_names 或 factor_families 中的至少一个")

        self.model = model
        self.train_config = train_config
        self.constraints = constraints
        self.factor_names = factor_names
        self.factor_families = factor_families
        self.selection_params = selection_params or {}
        self.signal_method = signal_method
        self.signal_clip = signal_clip
        self.max_factor_lookback = max_factor_lookback

        # 运行后填充
        self._wf_result: WalkForwardResult | None = None
        self._signal: pd.DataFrame | None = None
        self._weights: pd.DataFrame | None = None
        self._bt_result: BacktestResult | None = None
        self._factor_names_used: list[str] = []

    def run(
        self,
        price_panel: pd.DataFrame | None = None,
        start: str | None = None,
        end: str | None = None,
        symbols: list[str] | None = None,
    ) -> BacktestResult:
        """
        执行完整管道:
        1. 从 FactorStore 加载因子（按 factor_names 或 factor_families）
        2. 因子筛选（如果配置了 selection_params）
        3. 对齐和特征矩阵构建
        4. Walk-Forward 训练与预测
        5. 信号生成
        6. 组合构建
        7. 向量化回测

        Args:
            price_panel: 价格面板 (timestamp × symbol)。
                         None 则通过 load_price_panel 自动加载。
            start:       起始时间（可选，用于裁切或自动加载）
            end:         结束时间（可选，用于裁切或自动加载）
            symbols:     标的列表，None 则使用默认

        Returns:
            BacktestResult
        """
        if symbols is None:
            symbols = DEFAULT_SYMBOLS

        # [P4/R2/R3] price_panel 为 None 时自动加载
        if price_panel is None:
            from alpha_model.utils import load_price_panel
            price_panel = load_price_panel(symbols, start=start, end=end)

        # 时间裁切
        if start is not None:
            price_panel = price_panel.loc[start:]
        if end is not None:
            price_panel = price_panel.loc[:end]

        store = FactorStore()

        # --- Step 1: 加载因子 ---
        logger.info("Step 1: 加载因子")
        factor_panels = {}

        if self.factor_names:
            for name in self.factor_names:
                try:
                    factor_panels[name] = store.load(name)
                except Exception as e:
                    logger.warning("因子 '%s' 加载失败: %s", name, e)

        if self.factor_families:
            # 族级筛选（内部已包含跨族 select_factors，无需外部再次筛选）
            family_panels = select_from_families(
                self.factor_families, price_panel,
                horizon=self.train_config.target_horizon,
                store=store,
                **self.selection_params,
            )
            overlap = set(factor_panels) & set(family_panels)
            if overlap:
                logger.warning(
                    "factor_names 与 factor_families 存在重叠因子（将被族级版本覆盖）: %s",
                    overlap,
                )
            factor_panels.update(family_panels)

        if not factor_panels:
            raise RuntimeError("没有成功加载任何因子")

        logger.info("加载了 %d 个因子: %s", len(factor_panels), list(factor_panels.keys()))

        # --- Step 2: 因子筛选（如果有额外的 selection_params 且指定了 factor_names）---
        if self.factor_names and self.selection_params.get("mode"):
            logger.info("Step 2: 因子筛选")
            factor_panels = select_factors(
                factor_panels, price_panel,
                horizon=self.train_config.target_horizon,
                **self.selection_params,
            )

        self._factor_names_used = list(factor_panels.keys())

        # --- Step 3: 对齐 ---
        logger.info("Step 3: 因子对齐")
        factor_panels = align_factor_panels(factor_panels)

        # --- Step 4: 构建特征矩阵 + 目标变量 ---
        logger.info("Step 4: 构建特征矩阵")
        X = build_feature_matrix(
            factor_panels, symbols, self.train_config.train_mode,
        )
        y_panel = compute_forward_returns_panel(
            price_panel, horizon=self.train_config.target_horizon,
        )

        if self.train_config.train_mode == TrainMode.POOLED:
            # [P3] 构建 pooled y（使用公共函数，消除与 Trainer 的内部耦合）
            y = build_pooled_target(X, y_panel, symbols)
        else:
            y = {
                sym: y_panel[sym].reindex(X[sym].index)
                for sym in symbols
                if sym in X and sym in y_panel.columns
            }

        # --- Step 5: Walk-Forward ---
        logger.info("Step 5: Walk-Forward 训练")
        splitter = TimeSeriesSplitter(
            train_periods=self.train_config.train_periods,
            test_periods=self.train_config.test_periods,
            target_horizon=self.train_config.target_horizon,
            max_factor_lookback=self.max_factor_lookback,
            mode=self.train_config.wf_mode,
        )
        engine = WalkForwardEngine(
            self.model, splitter, self.train_config.train_mode,
        )
        self._wf_result = engine.run(X, y, symbols)

        # --- Step 6: 信号生成 ---
        logger.info("Step 6: 信号生成")
        self._signal = generate_signal(
            self._wf_result.predictions,
            method=self.signal_method,
            clip_sigma=self.signal_clip,
        )

        # --- Step 7: 组合构建 ---
        logger.info("Step 7: 组合构建")
        constructor = PortfolioConstructor(self.constraints)
        self._weights = constructor.construct(self._signal, price_panel)

        # --- Step 8: 向量化回测 ---
        logger.info("Step 8: 向量化回测")
        self._bt_result = vectorized_backtest(self._weights, price_panel)

        summary = self._bt_result.summary()
        logger.info("回测完成: Sharpe=%.3f, MaxDD=%.3f, Return=%.3f",
                     summary.get("sharpe_ratio", 0),
                     summary.get("max_drawdown", 0),
                     summary.get("total_return", 0))

        return self._bt_result

    def save(self, strategy_name: str) -> None:
        """
        保存到 SignalStore + ModelStore

        Args:
            strategy_name: 策略名称
        """
        if self._bt_result is None:
            raise RuntimeError("请先调用 run() 执行管道")

        meta = ModelMeta(
            name=strategy_name,
            factor_names=self._factor_names_used,
            target_horizon=self.train_config.target_horizon,
            train_config=self.train_config,
            constraints=self.constraints,
        )

        # 保存信号/权重
        signal_store = SignalStore()
        signal_store.save(
            strategy_name=strategy_name,
            weights=self._weights,
            signals=self._signal,
            meta=meta,
            performance=self._bt_result.summary(),
        )

        # 保存模型
        model_store = ModelStore()
        importance = {}
        if self._wf_result and not self._wf_result.feature_importance.empty:
            importance = self._wf_result.feature_importance["mean_importance"].to_dict()
        model_store.save(
            model_name=strategy_name,
            model=self.model,
            meta=meta,
            importance=importance,
        )

        logger.info("策略 '%s' 已保存", strategy_name)
