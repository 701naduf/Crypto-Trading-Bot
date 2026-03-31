"""
[T1] AlphaPipeline 端到端集成测试

使用合成数据验证完整管道:
    因子加载 → 对齐 → 特征矩阵 → Walk-Forward → 信号 → 组合 → 回测 → 保存/加载

[R4/R4'] 使用 monkeypatch 重定向所有存储路径到 tmp_path，
避免找不到因子或污染真实 db/ 目录。
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge

from alpha_model.core.pipeline import AlphaPipeline
from alpha_model.core.types import TrainConfig, PortfolioConstraints
from alpha_model.backtest.performance import BacktestResult
from alpha_model.models.linear_models import SklearnModelWrapper
from alpha_model.store.signal_store import SignalStore
from alpha_model.store.model_store import ModelStore
from factor_research.store.factor_store import FactorStore
from factor_research.core.types import FactorMeta, FactorType


class TestAlphaPipeline:
    """AlphaPipeline 端到端集成测试"""

    @pytest.fixture
    def setup_stores(self, tmp_path, monkeypatch):
        """创建临时存储环境并保存合成因子"""
        symbols = ["BTC/USDT", "ETH/USDT"]
        n = 500
        idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        rng = np.random.RandomState(42)

        # 1. 构建合成价格面板
        returns = rng.randn(n, len(symbols)) * 0.001
        price_panel = pd.DataFrame(
            100 * np.exp(np.cumsum(returns, axis=0)),
            index=idx, columns=symbols,
        )

        # 2. 构建合成因子面板并保存到临时 FactorStore
        factor_dir = tmp_path / "factors"

        # [R4'] patch FACTOR_STORE_DIR —— 必须同时 patch 两个位置:
        # 1. factor_research.config 模块（原始定义处）
        # 2. factor_research.store.factor_store 模块（import 后的副本）
        # 否则 FactorStore() 无参构造时仍会使用旧值
        monkeypatch.setattr(
            "factor_research.config.FACTOR_STORE_DIR",
            str(factor_dir),
        )
        monkeypatch.setattr(
            "factor_research.store.factor_store.FACTOR_STORE_DIR",
            str(factor_dir),
        )

        factor_store = FactorStore()
        factor_names = []
        for i in range(3):
            name = f"test_factor_{i}"
            panel = pd.DataFrame(
                rng.randn(n, len(symbols)) * 0.01,
                index=idx, columns=symbols,
            )
            meta = FactorMeta(
                name=name, display_name=name,
                factor_type=FactorType.TIME_SERIES, category="test",
                description="test factor",
                data_requirements=[],
                output_freq="1min",
            )
            factor_store.save(name, panel, meta)
            factor_names.append(name)

        # 3. patch alpha_model 侧的存储路径
        # 必须同时 patch config 模块和 store 模块的局部绑定（与 FactorStore 同理）[R1-ext]
        monkeypatch.setattr(
            "alpha_model.config.SIGNAL_STORE_DIR", tmp_path / "signals",
        )
        monkeypatch.setattr(
            "alpha_model.store.signal_store.SIGNAL_STORE_DIR", tmp_path / "signals",
        )
        monkeypatch.setattr(
            "alpha_model.config.MODEL_STORE_DIR", tmp_path / "models",
        )
        monkeypatch.setattr(
            "alpha_model.store.model_store.MODEL_STORE_DIR", tmp_path / "models",
        )

        return factor_names, price_panel, symbols

    def test_full_pipeline_run(self, setup_stores):
        """完整管道执行"""
        factor_names, price_panel, symbols = setup_stores
        pipeline = AlphaPipeline(
            model=Ridge(alpha=1.0),
            train_config=TrainConfig(
                train_periods=100, test_periods=50,
                target_horizon=5, purge_periods=10,
            ),
            constraints=PortfolioConstraints(vol_target=None),
            factor_names=factor_names,
        )
        result = pipeline.run(price_panel, symbols=symbols)

        # 验证输出类型
        assert isinstance(result, BacktestResult)
        # 验证净值曲线非空
        assert len(result.equity_curve) > 0
        # 验证 summary 包含所有指标
        summary = result.summary()
        assert "sharpe_ratio" in summary
        assert "max_drawdown" in summary
        assert "win_rate" in summary

    def test_pipeline_save_load(self, setup_stores):
        """保存后可正确加载"""
        factor_names, price_panel, symbols = setup_stores
        pipeline = AlphaPipeline(
            model=SklearnModelWrapper(Ridge(alpha=1.0)),
            train_config=TrainConfig(
                train_periods=100, test_periods=50,
                target_horizon=5, purge_periods=10,
            ),
            constraints=PortfolioConstraints(vol_target=None),
            factor_names=factor_names,
        )
        pipeline.run(price_panel, symbols=symbols)
        pipeline.save("test_strategy")

        # 验证 SignalStore 可加载
        signal_store = SignalStore()
        assert signal_store.exists("test_strategy")
        weights = signal_store.load_weights("test_strategy")
        assert isinstance(weights, pd.DataFrame)
        assert len(weights) > 0

        # 验证 ModelStore 可加载
        model_store = ModelStore()
        assert model_store.exists("test_strategy")
        meta = signal_store.load_meta("test_strategy")
        assert meta.name == "test_strategy"
        assert meta.factor_names == factor_names
