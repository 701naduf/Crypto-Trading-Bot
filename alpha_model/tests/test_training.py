"""
training/ 模块的单元测试

测试:
    - splitter.py: 时序切分器（Expanding/Rolling + embargo）
    - walk_forward.py: Walk-Forward 引擎
    - trainer.py: 训练调度器
"""

import pytest
import numpy as np
import pandas as pd

from alpha_model.core.types import TrainMode, WalkForwardMode, TrainConfig
from alpha_model.training.splitter import TimeSeriesSplitter, Fold
from alpha_model.training.walk_forward import WalkForwardEngine, WalkForwardResult
from alpha_model.training.trainer import Trainer


# ---------------------------------------------------------------------------
# 测试辅助
# ---------------------------------------------------------------------------

class _SimpleModel:
    """简单线性模型用于测试"""
    def fit(self, X, y, **kwargs):
        # 简单均值模型
        self._mean = y.mean() if len(y) > 0 else 0.0

    def predict(self, X):
        return np.full(len(X), self._mean)

    def get_feature_importance(self):
        return {}


def _make_panel(n_rows=500, symbols=None, seed=42):
    """生成测试用面板"""
    if symbols is None:
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=n_rows, freq="1min", tz="UTC")
    data = rng.randn(n_rows, len(symbols)) * 0.01
    return pd.DataFrame(data, index=idx, columns=symbols)


def _make_price_panel(n_rows=500, symbols=None, seed=42):
    """生成测试用价格面板（累积随机游走）"""
    if symbols is None:
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    rng = np.random.RandomState(seed)
    returns = rng.randn(n_rows, len(symbols)) * 0.001
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    idx = pd.date_range("2024-01-01", periods=n_rows, freq="1min", tz="UTC")
    return pd.DataFrame(prices, index=idx, columns=symbols)


# ---------------------------------------------------------------------------
# splitter.py
# ---------------------------------------------------------------------------

class TestTimeSeriesSplitter:
    """时序切分器"""

    def test_expanding_basic(self):
        """Expanding 模式基本切分"""
        splitter = TimeSeriesSplitter(
            train_periods=100, test_periods=50,
            target_horizon=10,
            mode=WalkForwardMode.EXPANDING,
        )
        folds = splitter.split(n_samples=500)
        assert len(folds) > 0

        # 第一个 fold 的训练起点为 0
        assert folds[0].train_start == 0
        assert folds[0].train_end == 100

    def test_expanding_train_grows(self):
        """Expanding 模式训练集应逐步增大"""
        splitter = TimeSeriesSplitter(
            train_periods=100, test_periods=50,
            target_horizon=10,
            mode=WalkForwardMode.EXPANDING,
        )
        folds = splitter.split(n_samples=500)
        for i in range(1, len(folds)):
            assert folds[i].train_size > folds[i - 1].train_size

    def test_rolling_basic(self):
        """Rolling 模式基本切分"""
        splitter = TimeSeriesSplitter(
            train_periods=100, test_periods=50,
            target_horizon=10,
            mode=WalkForwardMode.ROLLING,
        )
        folds = splitter.split(n_samples=500)
        assert len(folds) > 0

    def test_rolling_train_size_constant(self):
        """Rolling 模式训练集大小应恒定"""
        splitter = TimeSeriesSplitter(
            train_periods=100, test_periods=50,
            target_horizon=10,
            mode=WalkForwardMode.ROLLING,
        )
        folds = splitter.split(n_samples=500)
        for fold in folds:
            assert fold.train_size == 100

    def test_embargo_gap(self):
        """训练集和测试集之间应有 embargo gap"""
        splitter = TimeSeriesSplitter(
            train_periods=100, test_periods=50,
            target_horizon=10, max_factor_lookback=30,
            mode=WalkForwardMode.EXPANDING,
        )
        # embargo = max(10, 30) = 30
        assert splitter.embargo_periods == 30

        folds = splitter.split(n_samples=500)
        for fold in folds:
            gap = fold.test_start - fold.train_end
            assert gap >= 30, f"gap={gap} < embargo=30"

    def test_no_overlap(self):
        """训练集和测试集不应重叠"""
        splitter = TimeSeriesSplitter(
            train_periods=100, test_periods=50,
            target_horizon=10,
            mode=WalkForwardMode.EXPANDING,
        )
        folds = splitter.split(n_samples=500)
        for fold in folds:
            assert fold.train_end <= fold.test_start

    def test_insufficient_samples_raises(self):
        """样本不足应报错"""
        splitter = TimeSeriesSplitter(
            train_periods=100, test_periods=50,
            target_horizon=10,
        )
        with pytest.raises(ValueError, match="样本数"):
            splitter.split(n_samples=50)

    def test_n_splits(self):
        """n_splits 应返回正确的 fold 数量"""
        splitter = TimeSeriesSplitter(
            train_periods=100, test_periods=50,
            target_horizon=10,
            mode=WalkForwardMode.EXPANDING,
        )
        assert splitter.n_splits(500) == len(splitter.split(500))

    def test_embargo_covers_target_horizon(self):
        """embargo 应至少覆盖 target_horizon"""
        splitter = TimeSeriesSplitter(
            train_periods=100, test_periods=50,
            target_horizon=20,
        )
        assert splitter.embargo_periods >= 20

    def test_embargo_covers_factor_lookback(self):
        """embargo 应至少覆盖 max_factor_lookback"""
        splitter = TimeSeriesSplitter(
            train_periods=100, test_periods=50,
            target_horizon=10, max_factor_lookback=60,
        )
        assert splitter.embargo_periods >= 60

    def test_fold_dataclass_properties(self):
        """Fold 的 train_size 和 test_size 属性"""
        fold = Fold(fold_id=0, train_start=0, train_end=100, test_start=110, test_end=160)
        assert fold.train_size == 100
        assert fold.test_size == 50

    def test_invalid_params(self):
        """无效参数应报错"""
        with pytest.raises(ValueError):
            TimeSeriesSplitter(train_periods=0, test_periods=50, target_horizon=10)
        with pytest.raises(ValueError):
            TimeSeriesSplitter(train_periods=100, test_periods=0, target_horizon=10)
        with pytest.raises(ValueError):
            TimeSeriesSplitter(train_periods=100, test_periods=50, target_horizon=0)
        with pytest.raises(ValueError):
            TimeSeriesSplitter(train_periods=100, test_periods=50, target_horizon=10, max_factor_lookback=-1)


# ---------------------------------------------------------------------------
# walk_forward.py
# ---------------------------------------------------------------------------

class TestWalkForwardEngine:
    """Walk-Forward 引擎"""

    def test_pooled_basic(self):
        """Pooled 模式基本运行"""
        # 准备数据
        symbols = ["BTC/USDT", "ETH/USDT"]
        n = 500
        idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        rng = np.random.RandomState(42)

        # 构建 Pooled 格式
        pieces = []
        for sym in symbols:
            df = pd.DataFrame(
                {"f1": rng.randn(n), "f2": rng.randn(n)},
                index=idx,
            )
            df["symbol"] = sym
            pieces.append(df)
        X = pd.concat(pieces).set_index("symbol", append=True)
        X.index.names = ["timestamp", "symbol"]
        X = X.sort_index()

        y = pd.Series(rng.randn(len(X)), index=X.index)

        model = _SimpleModel()
        splitter = TimeSeriesSplitter(
            train_periods=100, test_periods=50,
            target_horizon=5,
        )
        engine = WalkForwardEngine(model, splitter, TrainMode.POOLED)
        result = engine.run(X, y, symbols)

        assert isinstance(result, WalkForwardResult)
        assert result.predictions.shape[1] == len(symbols)
        assert len(result.fold_metrics) > 0

    def test_nan_in_features_handled(self):
        """[T4] 特征矩阵含 NaN 时 Walk-Forward 应正常运行"""
        symbols = ["BTC/USDT", "ETH/USDT"]
        n = 500
        idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        rng = np.random.RandomState(42)

        pieces = []
        for sym in symbols:
            df = pd.DataFrame(
                {"f1": rng.randn(n), "f2": rng.randn(n)},
                index=idx,
            )
            # 注入 10% 随机 NaN
            mask = rng.rand(n, 2) < 0.1
            df.values[mask] = np.nan
            df["symbol"] = sym
            pieces.append(df)
        X = pd.concat(pieces).set_index("symbol", append=True)
        X.index.names = ["timestamp", "symbol"]
        X = X.sort_index()

        y = pd.Series(rng.randn(len(X)), index=X.index)

        model = _SimpleModel()
        splitter = TimeSeriesSplitter(
            train_periods=100, test_periods=50, target_horizon=5,
        )
        engine = WalkForwardEngine(model, splitter, TrainMode.POOLED)
        result = engine.run(X, y, symbols)

        assert isinstance(result, WalkForwardResult)
        assert result.predictions.shape[1] == len(symbols)

    def test_per_symbol_basic(self):
        """Per-Symbol 模式基本运行"""
        symbols = ["BTC/USDT", "ETH/USDT"]
        n = 500
        idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        rng = np.random.RandomState(42)

        X = {
            sym: pd.DataFrame(
                {"f1": rng.randn(n), "f2": rng.randn(n)},
                index=idx,
            )
            for sym in symbols
        }
        y = {
            sym: pd.Series(rng.randn(n), index=idx)
            for sym in symbols
        }

        model = _SimpleModel()
        splitter = TimeSeriesSplitter(
            train_periods=100, test_periods=50,
            target_horizon=5,
        )
        engine = WalkForwardEngine(model, splitter, TrainMode.PER_SYMBOL)
        result = engine.run(X, y, symbols)

        assert isinstance(result, WalkForwardResult)
        assert result.predictions.shape[1] == len(symbols)


# ---------------------------------------------------------------------------
# permutation importance
# ---------------------------------------------------------------------------

class TestPermutationImportance:
    """Permutation importance 测试"""

    def test_disabled_by_default(self):
        """默认不计算 permutation importance"""
        symbols = ["BTC/USDT", "ETH/USDT"]
        n = 500
        idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        rng = np.random.RandomState(42)

        pieces = []
        for sym in symbols:
            df = pd.DataFrame(
                {"f1": rng.randn(n), "f2": rng.randn(n)}, index=idx,
            )
            df["symbol"] = sym
            pieces.append(df)
        X = pd.concat(pieces).set_index("symbol", append=True)
        X.index.names = ["timestamp", "symbol"]
        X = X.sort_index()
        y = pd.Series(rng.randn(len(X)), index=X.index)

        model = _SimpleModel()
        splitter = TimeSeriesSplitter(
            train_periods=100, test_periods=50, target_horizon=5,
        )
        engine = WalkForwardEngine(model, splitter, TrainMode.POOLED)
        result = engine.run(X, y, symbols)
        # permutation_importance 应为空 DataFrame
        assert result.permutation_importance.empty

    def test_enabled_returns_results(self):
        """启用时应计算并返回非空结果"""
        symbols = ["BTC/USDT", "ETH/USDT"]
        n = 500
        idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        rng = np.random.RandomState(42)

        pieces = []
        for sym in symbols:
            df = pd.DataFrame(
                {"f1": rng.randn(n), "f2": rng.randn(n)}, index=idx,
            )
            df["symbol"] = sym
            pieces.append(df)
        X = pd.concat(pieces).set_index("symbol", append=True)
        X.index.names = ["timestamp", "symbol"]
        X = X.sort_index()
        y = pd.Series(rng.randn(len(X)), index=X.index)

        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
        splitter = TimeSeriesSplitter(
            train_periods=100, test_periods=50, target_horizon=5,
        )
        engine = WalkForwardEngine(
            model, splitter, TrainMode.POOLED,
            compute_permutation_importance=True,
            n_permutations=3,
            permutation_random_state=42,
        )
        result = engine.run(X, y, symbols)
        assert not result.permutation_importance.empty
        assert "mean_importance" in result.permutation_importance.columns
        assert len(result.permutation_importance) > 0

    def test_predictive_feature_has_positive_importance(self):
        """有预测力的因子的 permutation importance 应为正值"""
        symbols = ["BTC/USDT", "ETH/USDT"]
        n = 500
        idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        rng = np.random.RandomState(42)

        # f_signal 与 y 强相关，f_noise 为纯噪声
        y_raw = rng.randn(n) * 0.01
        pieces = []
        for sym in symbols:
            df = pd.DataFrame({
                "f_signal": y_raw + rng.randn(n) * 0.001,
                "f_noise": rng.randn(n) * 0.01,
            }, index=idx)
            df["symbol"] = sym
            pieces.append(df)
        X = pd.concat(pieces).set_index("symbol", append=True)
        X.index.names = ["timestamp", "symbol"]
        X = X.sort_index()

        y_pieces = []
        for sym in symbols:
            mi = pd.MultiIndex.from_arrays(
                [idx, [sym] * n], names=["timestamp", "symbol"],
            )
            y_pieces.append(pd.Series(y_raw, index=mi))
        y = pd.concat(y_pieces).sort_index()

        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
        splitter = TimeSeriesSplitter(
            train_periods=150, test_periods=50, target_horizon=5,
        )
        engine = WalkForwardEngine(
            model, splitter, TrainMode.POOLED,
            compute_permutation_importance=True,
            n_permutations=5,
            permutation_random_state=42,
        )
        result = engine.run(X, y, symbols)
        perm = result.permutation_importance
        # f_signal 的 importance 应 > f_noise
        assert perm.loc["f_signal", "mean_importance"] > perm.loc["f_noise", "mean_importance"]
        # f_signal 的 importance 应为正值
        assert perm.loc["f_signal", "mean_importance"] > 0

    def test_per_symbol_mode(self):
        """Per-Symbol 模式下 permutation importance 也应正常工作"""
        symbols = ["BTC/USDT", "ETH/USDT"]
        n = 500
        idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        rng = np.random.RandomState(42)

        # 使用有相关性的数据，确保 IC 非 NaN（_SimpleModel 返回常数，IC=NaN）
        y_base = rng.randn(n) * 0.01
        X = {
            sym: pd.DataFrame(
                {"f1": y_base + rng.randn(n) * 0.005, "f2": rng.randn(n)},
                index=idx,
            )
            for sym in symbols
        }
        y = {sym: pd.Series(y_base, index=idx) for sym in symbols}

        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
        splitter = TimeSeriesSplitter(
            train_periods=100, test_periods=50, target_horizon=5,
        )
        engine = WalkForwardEngine(
            model, splitter, TrainMode.PER_SYMBOL,
            compute_permutation_importance=True,
            n_permutations=3,
            permutation_random_state=42,
        )
        result = engine.run(X, y, symbols)
        assert not result.permutation_importance.empty
        assert "mean_importance" in result.permutation_importance.columns


# ---------------------------------------------------------------------------
# trainer.py
# ---------------------------------------------------------------------------

class TestTrainer:
    """训练调度器"""

    def test_basic_run(self):
        """基本训练流程"""
        symbols = ["BTC/USDT", "ETH/USDT"]
        price_panel = _make_price_panel(n_rows=500, symbols=symbols)
        factor_panels = {
            "f1": _make_panel(n_rows=500, symbols=symbols, seed=1),
            "f2": _make_panel(n_rows=500, symbols=symbols, seed=2),
        }

        model = _SimpleModel()
        config = TrainConfig(
            train_periods=100, test_periods=50,
            target_horizon=5, purge_periods=10,
        )
        trainer = Trainer(model, config)
        result = trainer.run(factor_panels, price_panel, symbols)

        assert isinstance(result, WalkForwardResult)
        assert result.train_config is config
