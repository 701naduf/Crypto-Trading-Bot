"""
Walk-Forward 交叉验证

这是 Phase 2b 防止过拟合的核心机制。

工作流程:
    1. splitter 生成 N 个 fold
    2. 对每个 fold:
       a. 取 X_train, y_train（训练集）
       b. model.fit(X_train, y_train)
       c. preds = model.predict(X_test)
       d. 记录预测值和评估指标
    3. 拼接所有 fold 的预测，形成完整样本外序列

关键：WalkForwardEngine 只通过 AlphaModel 协议交互，
      不关心模型内部实现（sklearn/PyTorch/手写均可）。

依赖: core.types.AlphaModel, training.splitter.TimeSeriesSplitter
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from factor_research.evaluation.metrics import spearman_ic

from alpha_model.core.types import AlphaModel, TrainConfig, TrainMode
from alpha_model.training.splitter import TimeSeriesSplitter, Fold

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    """
    Walk-Forward 结果

    Attributes:
        predictions:              样本外预测面板 (timestamp × symbol)
        fold_metrics:             每个 fold 的评估指标
        feature_importance:       模型原生因子重要性（各 fold 平均）
        permutation_importance:   置换重要性（各 fold 平均），未计算时为空 DataFrame
        train_config:             记录训练配置，确保可复现
    """
    predictions: pd.DataFrame         # 样本外预测面板 (timestamp × symbol)
    fold_metrics: list[dict]          # [{fold_id, train_size, test_size, ic, ...}]
    feature_importance: pd.DataFrame  # 模型原生因子重要性（各 fold 平均）
    # [R1] 使用 default_factory，外部构造时可省略
    permutation_importance: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["mean_importance", "std_importance"])
    )
    train_config: TrainConfig | None = None  # 记录训练配置


class WalkForwardEngine:
    """
    Walk-Forward 引擎

    用法:
        model = Ridge(alpha=1.0)  # 任何满足 AlphaModel 协议的对象
        splitter = TimeSeriesSplitter(...)
        engine = WalkForwardEngine(model, splitter)
        result = engine.run(X, y, symbols)
    """

    def __init__(
        self,
        model: AlphaModel,
        splitter: TimeSeriesSplitter,
        train_mode: TrainMode = TrainMode.POOLED,
        compute_permutation_importance: bool = False,
        n_permutations: int = 5,
        permutation_random_state: int | None = None,
    ):
        """
        Args:
            model:                           满足 AlphaModel 协议的模型对象
            splitter:                        时序切分器
            train_mode:                      训练粒度
            compute_permutation_importance:  是否计算置换重要性（默认 False，不影响现有行为）。
                                             启用后每个 fold 需额外调用
                                             n_features × n_permutations 次 predict，
                                             大模型（如 PyTorch）时可能有显著开销。
            n_permutations:                  每个特征的随机打乱次数（默认 5）
            permutation_random_state:        置换重要性的随机种子。None 则不固定（默认），
                                             传入整数时结果可复现。
        """
        self.model = model
        self.splitter = splitter
        self.train_mode = train_mode
        self.compute_permutation_importance = compute_permutation_importance
        self.n_permutations = n_permutations
        self.permutation_random_state = permutation_random_state

    def run(
        self,
        X: pd.DataFrame | dict[str, pd.DataFrame],
        y: pd.Series | dict[str, pd.Series],
        symbols: list[str] | None = None,
    ) -> WalkForwardResult:
        """
        执行 Walk-Forward

        Pooled 模式:
            X: DataFrame with MultiIndex(timestamp, symbol)
            y: Series with same MultiIndex
            在全量堆叠数据上切分和训练

        Per-Symbol 模式:
            X: {symbol: DataFrame(index=timestamp, columns=factor_names)}
            y: {symbol: Series(index=timestamp)}
            对每个 symbol 独立执行 Walk-Forward，汇总结果

        Returns:
            WalkForwardResult
        """
        if self.train_mode == TrainMode.POOLED:
            return self._run_pooled(X, y, symbols)
        elif self.train_mode == TrainMode.PER_SYMBOL:
            return self._run_per_symbol(X, y, symbols)
        else:
            raise ValueError(f"不支持的 train_mode: {self.train_mode}")

    def _run_pooled(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        symbols: list[str] | None,
    ) -> WalkForwardResult:
        """Pooled 模式: 在堆叠数据上执行 Walk-Forward"""
        # 按 timestamp 分组确定唯一时间戳数量（用于切分）
        timestamps = X.index.get_level_values("timestamp").unique().sort_values()
        n_timestamps = len(timestamps)

        folds = self.splitter.split(n_timestamps)
        logger.info(
            "Pooled Walk-Forward: %d 个时间戳, %d 个 fold",
            n_timestamps, len(folds),
        )

        if self.compute_permutation_importance:
            logger.info(
                "Permutation importance 已启用 (n_permutations=%d)",
                self.n_permutations,
            )

        all_fold_metrics = []
        all_predictions = []
        all_importances = []
        all_perm_importances = []

        for fold in folds:
            # 用时间戳索引取切分
            train_ts = timestamps[fold.train_start:fold.train_end]
            test_ts = timestamps[fold.test_start:fold.test_end]

            ts_level = X.index.get_level_values("timestamp")
            train_mask = ts_level.isin(train_ts)
            test_mask = ts_level.isin(test_ts)

            X_train = X.loc[train_mask]
            y_train = y.loc[train_mask]
            X_test = X.loc[test_mask]
            y_test = y.loc[test_mask]

            # 跳过数据不足的 fold
            if len(X_train) == 0 or len(X_test) == 0:
                logger.warning("Fold %d 数据不足，跳过", fold.fold_id)
                continue

            # 删除 NaN 行（训练集）
            valid_train = X_train.notna().all(axis=1) & y_train.notna()
            X_train = X_train[valid_train]
            y_train = y_train[valid_train]

            if len(X_train) == 0:
                logger.warning("Fold %d 训练集清洗后为空，跳过", fold.fold_id)
                continue

            # 每个 fold 使用模型的深拷贝，避免状态污染
            fold_model = copy.deepcopy(self.model)

            # 训练
            logger.debug(
                "Fold %d: train=%d, test=%d",
                fold.fold_id, len(X_train), len(X_test),
            )
            fold_model.fit(X_train, y_train)

            # 预测
            preds = fold_model.predict(X_test)
            if isinstance(preds, np.ndarray):
                preds = pd.Series(preds, index=X_test.index)

            all_predictions.append(preds)

            # 评估: 按 symbol 计算 IC
            fold_ic = self._compute_fold_ic(preds, y_test)

            # 因子重要性
            importance = self._get_importance(fold_model, X_train.columns.tolist())
            all_importances.append(importance)

            # 置换重要性（可选）
            if self.compute_permutation_importance:
                perm_imp = self._compute_permutation_importance(
                    fold_model, X_test, y_test,
                    n_permutations=self.n_permutations,
                    random_state=self.permutation_random_state,
                )
                all_perm_importances.append(perm_imp)

            all_fold_metrics.append({
                "fold_id": fold.fold_id,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "train_start": str(train_ts[0]),
                "train_end": str(train_ts[-1]),
                "test_start": str(test_ts[0]),
                "test_end": str(test_ts[-1]),
                **fold_ic,
            })

        # 拼接预测
        if not all_predictions:
            raise RuntimeError("没有成功完成任何 fold")

        combined_preds = pd.concat(all_predictions)

        # 重塑为面板格式 (timestamp × symbol)
        if symbols is None:
            symbols = combined_preds.index.get_level_values("symbol").unique().tolist()
        pred_panel = combined_preds.unstack(level="symbol")
        if isinstance(pred_panel.columns, pd.MultiIndex):
            pred_panel.columns = pred_panel.columns.droplevel(0)
        pred_panel = pred_panel.reindex(columns=symbols)

        # 平均因子重要性
        importance_df = self._average_importance(all_importances)

        # 汇总置换重要性
        perm_kwargs = {}
        if self.compute_permutation_importance and all_perm_importances:
            perm_kwargs["permutation_importance"] = self._average_importance(
                all_perm_importances
            )

        return WalkForwardResult(
            predictions=pred_panel,
            fold_metrics=all_fold_metrics,
            feature_importance=importance_df,
            **perm_kwargs,
        )

    def _run_per_symbol(
        self,
        X: dict[str, pd.DataFrame],
        y: dict[str, pd.Series],
        symbols: list[str] | None,
    ) -> WalkForwardResult:
        """Per-Symbol 模式: 每个标的独立执行 Walk-Forward"""
        if symbols is None:
            symbols = list(X.keys())

        all_fold_metrics = []
        symbol_predictions = {}
        all_importances = []
        all_perm_importances = []

        for symbol in symbols:
            if symbol not in X or symbol not in y:
                logger.warning("标的 '%s' 数据缺失，跳过", symbol)
                continue

            X_sym = X[symbol]
            y_sym = y[symbol]
            n_samples = len(X_sym)

            try:
                folds = self.splitter.split(n_samples)
            except ValueError as e:
                logger.warning("标的 '%s' 样本不足: %s", symbol, e)
                continue

            logger.info("Per-Symbol Walk-Forward [%s]: %d 样本, %d fold", symbol, n_samples, len(folds))

            symbol_preds_list = []

            for fold in folds:
                X_train = X_sym.iloc[fold.train_start:fold.train_end]
                y_train = y_sym.iloc[fold.train_start:fold.train_end]
                X_test = X_sym.iloc[fold.test_start:fold.test_end]
                y_test = y_sym.iloc[fold.test_start:fold.test_end]

                # 删除 NaN
                valid_train = X_train.notna().all(axis=1) & y_train.notna()
                X_train = X_train[valid_train]
                y_train = y_train[valid_train]

                if len(X_train) == 0 or len(X_test) == 0:
                    continue

                fold_model = copy.deepcopy(self.model)
                fold_model.fit(X_train, y_train)

                preds = fold_model.predict(X_test)
                if isinstance(preds, np.ndarray):
                    preds = pd.Series(preds, index=X_test.index)

                symbol_preds_list.append(preds)

                # IC
                valid_test = y_test.notna() & preds.notna()
                if valid_test.sum() > 2:
                    ic = spearman_ic(preds[valid_test], y_test[valid_test])
                else:
                    ic = np.nan

                importance = self._get_importance(fold_model, X_train.columns.tolist())
                all_importances.append(importance)

                # 置换重要性（可选）
                if self.compute_permutation_importance:
                    perm_imp = self._compute_permutation_importance(
                        fold_model, X_test, y_test,
                        n_permutations=self.n_permutations,
                        random_state=self.permutation_random_state,
                    )
                    all_perm_importances.append(perm_imp)

                all_fold_metrics.append({
                    "fold_id": fold.fold_id,
                    "symbol": symbol,
                    "train_size": len(X_train),
                    "test_size": len(X_test),
                    "ic": ic,
                })

            if symbol_preds_list:
                symbol_predictions[symbol] = pd.concat(symbol_preds_list)

        if not symbol_predictions:
            raise RuntimeError("没有任何标的成功完成 Walk-Forward")

        # 合并为面板 (timestamp × symbol)
        pred_panel = pd.DataFrame(symbol_predictions)
        pred_panel = pred_panel.sort_index()

        importance_df = self._average_importance(all_importances)

        perm_kwargs = {}
        if self.compute_permutation_importance and all_perm_importances:
            perm_kwargs["permutation_importance"] = self._average_importance(
                all_perm_importances
            )

        return WalkForwardResult(
            predictions=pred_panel,
            fold_metrics=all_fold_metrics,
            feature_importance=importance_df,
            **perm_kwargs,
        )

    @staticmethod
    def _compute_fold_ic(
        preds: pd.Series,
        y: pd.Series,
    ) -> dict:
        """计算 fold 级别的 IC（截面 IC 的均值）"""
        valid = preds.notna() & y.notna()
        if valid.sum() < 3:
            return {"ic": np.nan}

        preds_v = preds[valid]
        y_v = y[valid]

        # 如果是 MultiIndex，按 timestamp 分组计算截面 IC
        if isinstance(preds_v.index, pd.MultiIndex):
            timestamps = preds_v.index.get_level_values("timestamp").unique()
            ics = []
            for ts in timestamps:
                mask = preds_v.index.get_level_values("timestamp") == ts
                p = preds_v[mask]
                r = y_v[mask]
                if len(p) > 2:
                    ics.append(spearman_ic(p, r))
            if ics:
                return {
                    "ic": np.nanmean(ics),
                    "ic_std": np.nanstd(ics),
                    "ic_ir": (
                        np.nanmean(ics) / np.nanstd(ics)
                        if np.nanstd(ics) > 0
                        else np.nan
                    ),
                }
        else:
            ic = spearman_ic(preds_v, y_v)
            return {"ic": ic}

        return {"ic": np.nan}

    @staticmethod
    def _get_importance(model: AlphaModel, feature_names: list[str]) -> dict[str, float]:
        """安全获取因子重要性"""
        if hasattr(model, "get_feature_importance"):
            try:
                return model.get_feature_importance()
            except Exception:
                pass
        # sklearn 模型可能直接有 coef_
        if hasattr(model, "coef_"):
            coef = np.abs(model.coef_)
            if len(coef) == len(feature_names):
                return dict(zip(feature_names, coef))
        return {}

    @staticmethod
    def _compute_permutation_importance(
        model: AlphaModel,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_permutations: int = 5,
        random_state: int | None = None,
    ) -> dict[str, float]:
        """
        计算单个 fold 的置换重要性

        对每个特征:
            1. 计算基准 IC（原始预测 vs 真实值）
            2. 随机打乱该特征列 N 次，每次重新预测并计算 IC
            3. 置换重要性 = 基准 IC - 打乱后 IC 的均值
            正值 → 该特征对预测有正贡献（打乱后 IC 下降）
            零/负值 → 该特征无贡献或有害

        计算开销: O(n_features × n_permutations) 次额外 predict 调用。

        Args:
            model:            已训练的模型
            X_test:           测试集特征矩阵
            y_test:           测试集目标变量
            n_permutations:   打乱次数
            random_state:     随机种子，None 则不固定，传入整数时结果可复现

        Returns:
            {feature_name: importance_score} 字典
        """
        # 过滤 NaN
        valid = X_test.notna().all(axis=1) & y_test.notna()
        X_valid = X_test[valid]
        y_valid = y_test[valid]

        if len(X_valid) < 3:
            return {}

        # 基准 IC
        base_preds = model.predict(X_valid)
        if isinstance(base_preds, np.ndarray):
            base_preds = pd.Series(base_preds, index=X_valid.index)
        base_ic = spearman_ic(base_preds, y_valid)

        if np.isnan(base_ic):
            return {}

        importance = {}
        rng = np.random.RandomState(random_state)

        for feature in X_valid.columns:
            ic_drops = []
            for _ in range(n_permutations):
                X_shuffled = X_valid.copy()
                X_shuffled[feature] = rng.permutation(X_shuffled[feature].values)
                shuffled_preds = model.predict(X_shuffled)
                if isinstance(shuffled_preds, np.ndarray):
                    shuffled_preds = pd.Series(shuffled_preds, index=X_valid.index)
                shuffled_ic = spearman_ic(shuffled_preds, y_valid)
                ic_drops.append(base_ic - shuffled_ic)

            importance[feature] = float(np.nanmean(ic_drops))

        return importance

    @staticmethod
    def _average_importance(
        importances: list[dict[str, float]],
    ) -> pd.DataFrame:
        """对多个 fold 的因子重要性取平均"""
        if not importances or all(not imp for imp in importances):
            return pd.DataFrame(columns=["mean_importance", "std_importance"])

        df = pd.DataFrame(importances)
        result = pd.DataFrame({
            "mean_importance": df.mean(),
            "std_importance": df.std(),
        })
        return result.sort_values("mean_importance", ascending=False)
