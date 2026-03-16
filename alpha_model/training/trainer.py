"""
训练调度器

整合 build_feature_matrix + compute_forward_returns + WalkForwardEngine。
一站式接口：从因子面板和价格面板出发，完成特征构建、目标变量计算、
Walk-Forward 训练。

依赖:
    preprocessing.transform.build_feature_matrix
    factor_research.evaluation.metrics.compute_forward_returns_panel
    training.walk_forward.WalkForwardEngine
    training.splitter.TimeSeriesSplitter
"""

from __future__ import annotations

import logging

import pandas as pd

from factor_research.evaluation.metrics import compute_forward_returns_panel

from alpha_model.core.types import AlphaModel, TrainConfig, TrainMode
from alpha_model.preprocessing.transform import build_feature_matrix, build_pooled_target
from alpha_model.training.splitter import TimeSeriesSplitter
from alpha_model.training.walk_forward import WalkForwardEngine, WalkForwardResult

logger = logging.getLogger(__name__)


class Trainer:
    """
    训练调度器

    一站式接口：从因子面板和价格面板出发，完成特征构建、目标变量计算、
    Walk-Forward 训练。

    用法:
        trainer = Trainer(model=Ridge(), train_config=TrainConfig(...))
        result = trainer.run(factor_panels, price_panel, symbols)
    """

    def __init__(
        self,
        model: AlphaModel,
        train_config: TrainConfig,
        max_factor_lookback: int = 0,
    ):
        """
        Args:
            model:                满足 AlphaModel 协议的模型对象
            train_config:         训练配置
            max_factor_lookback:  因子中最大的 lookback 窗口长度
                                  传递给 TimeSeriesSplitter 用于 embargo 计算
        """
        self.model = model
        self.train_config = train_config
        self.max_factor_lookback = max_factor_lookback

    def run(
        self,
        factor_panels: dict[str, pd.DataFrame],
        price_panel: pd.DataFrame,
        symbols: list[str] | None = None,
    ) -> WalkForwardResult:
        """
        完整训练流程

        1. build_feature_matrix → X
        2. compute_forward_returns_panel → y
        3. WalkForwardEngine.run(X, y) → result

        Args:
            factor_panels: {factor_name: panel (timestamp × symbol)}
            price_panel:   价格面板 (timestamp × symbol)
            symbols:       要使用的标的列表，None 则使用所有可用标的

        Returns:
            WalkForwardResult
        """
        # --- 推断 symbols ---
        if symbols is None:
            # 取所有因子面板和价格面板共有的标的
            symbol_sets = [set(p.columns) for p in factor_panels.values()]
            symbol_sets.append(set(price_panel.columns))
            symbols = sorted(set.intersection(*symbol_sets))
            if not symbols:
                raise ValueError("因子面板和价格面板没有共同的标的")
            logger.info("自动推断标的: %s", symbols)

        # --- Step 1: 构建特征矩阵 ---
        logger.info("Step 1: 构建特征矩阵 (mode=%s)", self.train_config.train_mode.value)
        X = build_feature_matrix(
            factor_panels, symbols, self.train_config.train_mode,
        )

        # --- Step 2: 构建目标变量 ---
        logger.info(
            "Step 2: 计算 forward returns (horizon=%d)",
            self.train_config.target_horizon,
        )
        fwd_returns = compute_forward_returns_panel(
            price_panel, horizon=self.train_config.target_horizon,
        )

        if self.train_config.train_mode == TrainMode.POOLED:
            # X: MultiIndex(timestamp, symbol), 需要从 fwd_returns 构造同格式的 y
            y = build_pooled_target(X, fwd_returns, symbols)
        else:
            # Per-Symbol: y = {symbol: Series}
            y = {
                symbol: fwd_returns[symbol].reindex(X[symbol].index)
                for symbol in symbols
                if symbol in X and symbol in fwd_returns.columns
            }

        # --- Step 3: Walk-Forward ---
        logger.info("Step 3: Walk-Forward 训练")
        splitter = TimeSeriesSplitter(
            train_periods=self.train_config.train_periods,
            test_periods=self.train_config.test_periods,
            target_horizon=self.train_config.target_horizon,
            max_factor_lookback=self.max_factor_lookback,
            mode=self.train_config.wf_mode,
        )

        engine = WalkForwardEngine(
            model=self.model,
            splitter=splitter,
            train_mode=self.train_config.train_mode,
        )

        result = engine.run(X, y, symbols)
        result.train_config = self.train_config

        logger.info(
            "训练完成: %d 个 fold, 预测面板 %s",
            len(result.fold_metrics),
            result.predictions.shape,
        )
        return result

