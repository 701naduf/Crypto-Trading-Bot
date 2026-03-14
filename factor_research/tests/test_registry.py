"""core/registry.py 的单元测试"""

import pytest

from factor_research.core.base import TimeSeriesFactor
from factor_research.core.registry import (
    FactorRegistry,
    get_default_registry,
    register_factor,
    register_factor_family,
)
from factor_research.core.types import (
    DataRequest,
    DataType,
    FactorMeta,
    FactorType,
)

import pandas as pd


# =========================================================================
# 测试用因子（不使用装饰器，手动注册）
# =========================================================================

class _TestFactor1(TimeSeriesFactor):
    def meta(self):
        return FactorMeta(
            name="test_factor_1",
            display_name="Test 1",
            factor_type=FactorType.TIME_SERIES,
            category="test_cat_a",
            description="Test factor 1",
            data_requirements=[DataRequest(DataType.OHLCV, timeframe="1m")],
            output_freq="1m",
        )

    def compute_single(self, symbol, data):
        return pd.Series(dtype=float)


class _TestFactor2(TimeSeriesFactor):
    def meta(self):
        return FactorMeta(
            name="test_factor_2",
            display_name="Test 2",
            factor_type=FactorType.TIME_SERIES,
            category="test_cat_b",
            description="Test factor 2",
            data_requirements=[DataRequest(DataType.TICK)],
            output_freq="1s",
        )

    def compute_single(self, symbol, data):
        return pd.Series(dtype=float)


# =========================================================================
# 参数化测试用因子
# =========================================================================

class _ParamFactor(TimeSeriesFactor):
    """单参数因子，用于测试参数化注册"""

    def __init__(self, lookback: int = 5):
        self.lookback = lookback

    def meta(self):
        return FactorMeta(
            name=f"param_factor_{self.lookback}",
            display_name=f"Param Factor ({self.lookback})",
            factor_type=FactorType.TIME_SERIES,
            category="test_param",
            description=f"Parameterized factor with lookback={self.lookback}",
            data_requirements=[DataRequest(DataType.OHLCV, timeframe="1m")],
            output_freq="1m",
            params={"lookback": self.lookback},
            family="param_factor_family",
        )

    def compute_single(self, symbol, data):
        return pd.Series(dtype=float)


class _MultiParamFactor(TimeSeriesFactor):
    """多参数因子，用于测试笛卡尔积展开"""

    def __init__(self, window: int = 10, method: str = "ema"):
        self.window = window
        self.method = method

    def meta(self):
        return FactorMeta(
            name=f"multi_{self.method}_{self.window}",
            display_name=f"Multi ({self.method}, {self.window})",
            factor_type=FactorType.TIME_SERIES,
            category="test_multi",
            description=f"Multi-param factor window={self.window} method={self.method}",
            data_requirements=[DataRequest(DataType.OHLCV, timeframe="1m")],
            output_freq="1m",
            params={"window": self.window, "method": self.method},
            family="multi_param_family",
        )

    def compute_single(self, symbol, data):
        return pd.Series(dtype=float)


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def registry():
    """每个测试用全新的注册表（清空全局状态）"""
    reg = FactorRegistry()
    reg.clear()
    return reg


# =========================================================================
# 测试
# =========================================================================

class TestFactorRegistry:

    def test_register_and_get(self, registry):
        registry.register(_TestFactor1)
        factor = registry.get("test_factor_1")
        assert isinstance(factor, _TestFactor1)

    def test_register_duplicate_different_class_raises(self, registry):
        registry.register(_TestFactor1)
        # 创建一个名字相同但类不同的因子
        class DuplicateFactor(TimeSeriesFactor):
            def meta(self):
                return FactorMeta(
                    name="test_factor_1",  # 同名
                    display_name="Dup",
                    factor_type=FactorType.TIME_SERIES,
                    category="test",
                    description="dup",
                    data_requirements=[],
                    output_freq="1m",
                )
            def compute_single(self, symbol, data):
                return pd.Series(dtype=float)

        with pytest.raises(ValueError, match="已被"):
            registry.register(DuplicateFactor)

    def test_register_same_class_twice_ok(self, registry):
        registry.register(_TestFactor1)
        registry.register(_TestFactor1)  # 相同类重复注册不报错
        assert len(registry) == 1

    def test_register_non_factor_raises(self, registry):
        with pytest.raises(TypeError):
            registry.register(str)

    def test_get_nonexistent_raises(self, registry):
        with pytest.raises(KeyError, match="未注册"):
            registry.get("nonexistent")

    def test_list_all(self, registry):
        registry.register(_TestFactor1)
        registry.register(_TestFactor2)
        metas = registry.list_all()
        assert len(metas) == 2
        names = [m.name for m in metas]
        assert "test_factor_1" in names
        assert "test_factor_2" in names

    def test_list_by_category(self, registry):
        registry.register(_TestFactor1)
        registry.register(_TestFactor2)
        cat_a = registry.list_by_category("test_cat_a")
        assert len(cat_a) == 1
        assert cat_a[0].name == "test_factor_1"

    def test_list_by_type(self, registry):
        registry.register(_TestFactor1)
        ts = registry.list_by_type(FactorType.TIME_SERIES)
        assert len(ts) == 1

    def test_contains(self, registry):
        registry.register(_TestFactor1)
        assert "test_factor_1" in registry
        assert "nonexistent" not in registry

    def test_len(self, registry):
        assert len(registry) == 0
        registry.register(_TestFactor1)
        assert len(registry) == 1

    def test_clear(self, registry):
        registry.register(_TestFactor1)
        registry.clear()
        assert len(registry) == 0


class TestRegisterDecorator:

    def test_decorator_registers_factor(self):
        """@register_factor 装饰器注册到默认注册表"""
        default_reg = get_default_registry()

        @register_factor
        class DecoratedFactor(TimeSeriesFactor):
            def meta(self):
                return FactorMeta(
                    name="decorated_test",
                    display_name="Decorated",
                    factor_type=FactorType.TIME_SERIES,
                    category="test",
                    description="test",
                    data_requirements=[],
                    output_freq="1m",
                )
            def compute_single(self, symbol, data):
                return pd.Series(dtype=float)

        assert "decorated_test" in default_reg
        factor = default_reg.get("decorated_test")
        assert isinstance(factor, DecoratedFactor)

    def test_decorator_returns_original_class(self):
        """装饰器不修改类本身"""
        @register_factor
        class MyFactor(TimeSeriesFactor):
            def meta(self):
                return FactorMeta(
                    name="my_test",
                    display_name="My",
                    factor_type=FactorType.TIME_SERIES,
                    category="test",
                    description="test",
                    data_requirements=[],
                    output_freq="1m",
                )
            def compute_single(self, symbol, data):
                return pd.Series(dtype=float)

        assert MyFactor.__name__ == "MyFactor"


# =========================================================================
# 参数化注册测试
# =========================================================================

class TestParameterizedRegistration:
    """参数化因子注册的测试（Batch 1 新增）"""

    def test_register_with_params(self, registry):
        """T-1: 带参数注册 → get() 返回正确实例"""
        registry.register(_ParamFactor, lookback=10)
        factor = registry.get("param_factor_10")
        assert isinstance(factor, _ParamFactor)
        assert factor.lookback == 10

    def test_same_class_different_params(self, registry):
        """T-2: 同类不同参数 → 注册为不同名称的因子"""
        registry.register(_ParamFactor, lookback=5)
        registry.register(_ParamFactor, lookback=10)
        assert len(registry) == 2
        assert "param_factor_5" in registry
        assert "param_factor_10" in registry
        # 各自返回正确参数的实例
        f5 = registry.get("param_factor_5")
        f10 = registry.get("param_factor_10")
        assert f5.lookback == 5
        assert f10.lookback == 10

    def test_same_class_same_params_idempotent(self, registry):
        """T-3: 同类同参数重复注册 → 幂等，不抛异常"""
        registry.register(_ParamFactor, lookback=10)
        registry.register(_ParamFactor, lookback=10)  # 不应报错
        assert len(registry) == 1

    def test_same_name_different_class_raises(self, registry):
        """T-4: 不同类注册同名因子 → ValueError"""
        registry.register(_ParamFactor, lookback=5)

        # 创建一个与 _ParamFactor(lookback=5) 同名的不同类
        class _ConflictFactor(TimeSeriesFactor):
            def meta(self):
                return FactorMeta(
                    name="param_factor_5",  # 同名!
                    display_name="Conflict",
                    factor_type=FactorType.TIME_SERIES,
                    category="test",
                    description="conflict",
                    data_requirements=[],
                    output_freq="1m",
                )
            def compute_single(self, symbol, data):
                return pd.Series(dtype=float)

        with pytest.raises(ValueError, match="已被"):
            registry.register(_ConflictFactor)

    def test_list_all_includes_parameterized(self, registry):
        """T-8: list_all() 返回所有参数变体"""
        registry.register(_ParamFactor, lookback=5)
        registry.register(_ParamFactor, lookback=10)
        registry.register(_ParamFactor, lookback=30)

        metas = registry.list_all()
        assert len(metas) == 3
        # 检查 meta.params 反映参数
        params_set = {m.params["lookback"] for m in metas}
        assert params_set == {5, 10, 30}


class TestRegisterFactorFamily:
    """register_factor_family 装饰器测试（Batch 1 新增）"""

    def test_single_param_grid(self):
        """T-5: 单参数 _param_grid → 注册 N 个因子"""
        reg = FactorRegistry()

        class _SingleGrid(TimeSeriesFactor):
            _param_grid = {"lookback": [1, 2, 3]}

            def __init__(self, lookback: int = 1):
                self.lookback = lookback

            def meta(self):
                return FactorMeta(
                    name=f"sg_{self.lookback}",
                    display_name=f"SG {self.lookback}",
                    factor_type=FactorType.TIME_SERIES,
                    category="test",
                    description="test",
                    data_requirements=[],
                    output_freq="1m",
                    params={"lookback": self.lookback},
                    family="single_grid",
                )

            def compute_single(self, symbol, data):
                return pd.Series(dtype=float)

        # 手动模拟装饰器行为（使用独立注册表而非全局注册表）
        for lookback in [1, 2, 3]:
            reg.register(_SingleGrid, lookback=lookback)

        assert len(reg) == 3
        assert "sg_1" in reg
        assert "sg_2" in reg
        assert "sg_3" in reg

    def test_multi_param_grid_cartesian(self):
        """T-6: 多参数 _param_grid → 笛卡尔积注册"""
        reg = FactorRegistry()

        # 手动做笛卡尔积（与装饰器逻辑一致）
        import itertools
        param_grid = {"window": [10, 20], "method": ["ema", "sma"]}
        keys = list(param_grid.keys())
        for combo in itertools.product(*[param_grid[k] for k in keys]):
            params = dict(zip(keys, combo))
            reg.register(_MultiParamFactor, **params)

        assert len(reg) == 4  # 2 × 2 = 4
        assert "multi_ema_10" in reg
        assert "multi_ema_20" in reg
        assert "multi_sma_10" in reg
        assert "multi_sma_20" in reg

    def test_no_param_grid_fallback(self):
        """T-7: 无 _param_grid → 退化为单因子注册"""
        reg = FactorRegistry()

        class _NoGrid(TimeSeriesFactor):
            # 没有 _param_grid 属性

            def meta(self):
                return FactorMeta(
                    name="no_grid_factor",
                    display_name="No Grid",
                    factor_type=FactorType.TIME_SERIES,
                    category="test",
                    description="test",
                    data_requirements=[],
                    output_freq="1m",
                )

            def compute_single(self, symbol, data):
                return pd.Series(dtype=float)

        reg.register(_NoGrid)
        assert len(reg) == 1
        assert "no_grid_factor" in reg

    def test_register_factor_family_decorator_on_default_registry(self):
        """register_factor_family 装饰器注册到默认注册表"""
        default_reg = get_default_registry()

        @register_factor_family
        class _FamilyDecorated(TimeSeriesFactor):
            _param_grid = {"n": [100, 200]}

            def __init__(self, n: int = 100):
                self.n = n

            def meta(self):
                return FactorMeta(
                    name=f"family_decorated_{self.n}",
                    display_name=f"FD {self.n}",
                    factor_type=FactorType.TIME_SERIES,
                    category="test",
                    description="test",
                    data_requirements=[],
                    output_freq="1m",
                    params={"n": self.n},
                    family="family_decorated",
                )

            def compute_single(self, symbol, data):
                return pd.Series(dtype=float)

        assert "family_decorated_100" in default_reg
        assert "family_decorated_200" in default_reg

    def test_register_factor_family_no_grid_on_default_registry(self):
        """register_factor_family 无 _param_grid 时退化为 register_factor"""
        default_reg = get_default_registry()

        @register_factor_family
        class _FamilyNoGrid(TimeSeriesFactor):
            def meta(self):
                return FactorMeta(
                    name="family_no_grid_test",
                    display_name="FNG",
                    factor_type=FactorType.TIME_SERIES,
                    category="test",
                    description="test",
                    data_requirements=[],
                    output_freq="1m",
                )

            def compute_single(self, symbol, data):
                return pd.Series(dtype=float)

        assert "family_no_grid_test" in default_reg


# =========================================================================
# 族操作测试
# =========================================================================

class TestFamilyOperations:
    """list_family / list_families 方法测试（Batch 1 新增）"""

    def test_list_family(self, registry):
        """T-9: list_family() 只返回指定族的因子"""
        # 注册 param_factor_family 族
        registry.register(_ParamFactor, lookback=5)
        registry.register(_ParamFactor, lookback=10)
        # 注册 multi_param_family 族
        registry.register(_MultiParamFactor, window=10, method="ema")
        # 注册独立因子（无族）
        registry.register(_TestFactor1)

        family_metas = registry.list_family("param_factor_family")
        assert len(family_metas) == 2
        names = {m.name for m in family_metas}
        assert names == {"param_factor_5", "param_factor_10"}

        # 另一个族
        multi_metas = registry.list_family("multi_param_family")
        assert len(multi_metas) == 1

        # 不存在的族
        assert registry.list_family("nonexistent") == []

    def test_list_families(self, registry):
        """T-10: list_families() 返回去重排序的族名列表，不含空字符串"""
        # 注册多个族
        registry.register(_ParamFactor, lookback=5)
        registry.register(_ParamFactor, lookback=10)
        registry.register(_MultiParamFactor, window=10, method="ema")
        # 注册独立因子（family=""）
        registry.register(_TestFactor1)

        families = registry.list_families()
        assert isinstance(families, list)
        assert "param_factor_family" in families
        assert "multi_param_family" in families
        # 空字符串不在列表中
        assert "" not in families
        # 已排序
        assert families == sorted(families)

    def test_family_field_in_meta(self, registry):
        """T-11: FactorMeta.family 字段正确性"""
        registry.register(_ParamFactor, lookback=5)
        registry.register(_TestFactor1)

        # 参数化因子的 family 字段
        param_factor = registry.get("param_factor_5")
        assert param_factor.meta().family == "param_factor_family"

        # 独立因子的 family 字段（默认空字符串）
        standalone = registry.get("test_factor_1")
        assert standalone.meta().family == ""
