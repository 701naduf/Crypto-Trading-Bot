"""
因子注册与发现模块

提供因子的注册、检索、批量发现功能。
支持两种注册方式:
    - @register_factor:        注册单个因子（无参或固定参数）
    - @register_factor_family: 注册参数化因子族（自动展开 _param_grid）

注册机制的设计目标:
    - 零侵入: 新增因子只需继承基类 + 加装饰器，不修改框架代码
    - 自动发现: 只要因子模块被 import，因子就自动注册
    - 唯一性: 因子名称在注册表内唯一，重复注册会抛出异常
    - 实例隔离: 每个 FactorRegistry 实例拥有独立的注册表，
      避免测试间状态泄露。全局行为通过 _default_registry 实现。
    - 参数化: 支持带参数的因子注册，同一个类可通过不同参数注册为多个因子

用法:
    # 单因子注册（无参）
    @register_factor
    class OrderbookImbalance(TimeSeriesFactor):
        def meta(self) -> FactorMeta:
            return FactorMeta(name="orderbook_imbalance", ...)
        ...

    # 因子族注册（参数化）
    @register_factor_family
    class MultiScaleReturns(TimeSeriesFactor):
        _param_grid = {"lookback": [5, 10, 30, 60]}

        def __init__(self, lookback: int = 5):
            self.lookback = lookback

        def meta(self) -> FactorMeta:
            return FactorMeta(
                name=f"returns_{self.lookback}m",
                family="multi_scale_returns",
                params={"lookback": self.lookback},
                ...
            )
        ...

    # 检索
    registry = get_default_registry()
    factor = registry.get("returns_10m")                       # 按名称获取因子实例
    all_metas = registry.list_all()                            # 列出所有已注册因子
    momentum_metas = registry.list_by_category("momentum")     # 按分类检索
    family_metas = registry.list_family("multi_scale_returns") # 按族检索
    families = registry.list_families()                        # 列出所有族名

    # 测试中使用独立注册表（无需 clear）
    test_reg = FactorRegistry()  # 全新的空注册表

依赖: core.base
被依赖: core.engine, factors/
"""

import itertools

from data_infra.utils.logger import get_logger

from .base import Factor
from .types import FactorMeta, FactorType

logger = get_logger(__name__)


class FactorRegistry:
    """
    因子注册表（实例级别存储）

    管理已注册的因子类及其构造参数。每个实例拥有独立的注册表字典。
    全局单例行为通过模块级 _default_registry 实现。

    实例级别设计的优势:
        - 测试中每个 FactorRegistry() 都是干净的，无需 clear()
        - 支持多注册表场景（如测试注册表 vs 生产注册表）
        - clear() 不影响其他代码持有的注册表引用

    内部存储:
        _registry: dict[str, tuple[type, dict]]  — {因子名称: (因子类, 构造参数)}
        注册的是类和参数，在 get() 时以 cls(**params) 实例化。
        对于无参因子，params 为空字典 {}，行为与改造前完全一致。
    """

    def __init__(self):
        """初始化空注册表"""
        self._registry: dict[str, tuple[type, dict]] = {}

    def register(self, factor_cls: type, **params) -> None:
        """
        注册一个因子类（可带参数）

        在注册时会:
            1. 以 factor_cls(**params) 实例化获取 meta()
            2. 检查名称是否已被占用
            3. 存入注册表

        Args:
            factor_cls: 因子类（必须是 Factor 的子类）
            **params:   构造参数。无参时为空字典，行为与改造前完全一致。

        Raises:
            TypeError:  如果 factor_cls 不是 Factor 的子类
            ValueError: 如果因子名称已被其他类注册
        """
        if not (isinstance(factor_cls, type) and issubclass(factor_cls, Factor)):
            raise TypeError(
                f"{factor_cls} 不是 Factor 的子类，无法注册"
            )

        # 以参数实例化获取 meta
        instance = factor_cls(**params)
        meta = instance.meta()

        if meta.name in self._registry:
            existing_cls, existing_params = self._registry[meta.name]
            # 允许同类同参数重复注册（模块重载场景）
            if existing_cls is factor_cls and existing_params == params:
                return
            raise ValueError(
                f"因子名称 '{meta.name}' 已被 {existing_cls.__name__} 注册，"
                f"不能被 {factor_cls.__name__} 重复注册"
            )

        self._registry[meta.name] = (factor_cls, params)
        logger.debug(f"因子已注册: {meta.name} ({factor_cls.__name__}, params={params})")

    def get(self, name: str) -> Factor:
        """
        按名称获取因子实例

        每次调用都会创建一个新实例（因子应当是无状态的）。
        使用注册时的参数进行实例化。

        Args:
            name: 因子名称（即 FactorMeta.name）

        Returns:
            Factor: 因子实例

        Raises:
            KeyError: 如果因子未注册
        """
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys())) or "(无)"
            raise KeyError(
                f"因子 '{name}' 未注册。已注册的因子: {available}"
            )
        cls, params = self._registry[name]
        return cls(**params)

    def list_all(self) -> list[FactorMeta]:
        """
        列出所有已注册因子的元数据

        Returns:
            按名称排序的 FactorMeta 列表
        """
        metas = []
        for name in sorted(self._registry.keys()):
            cls, params = self._registry[name]
            instance = cls(**params)
            metas.append(instance.meta())
        return metas

    def list_by_category(self, category: str) -> list[FactorMeta]:
        """
        按分类列出因子

        Args:
            category: 分类名（如 "microstructure", "momentum"）

        Returns:
            属于该分类的 FactorMeta 列表
        """
        return [m for m in self.list_all() if m.category == category]

    def list_by_type(self, factor_type: FactorType) -> list[FactorMeta]:
        """
        按因子类型列出

        Args:
            factor_type: FactorType 枚举值

        Returns:
            属于该类型的 FactorMeta 列表
        """
        return [m for m in self.list_all() if m.factor_type == factor_type]

    def list_family(self, family_name: str) -> list[FactorMeta]:
        """
        列出指定族的所有因子变体

        Args:
            family_name: 族名（如 "multi_scale_returns"）

        Returns:
            属于该族的 FactorMeta 列表，按 name 排序
        """
        return [m for m in self.list_all() if m.family == family_name]

    def list_families(self) -> list[str]:
        """
        列出所有已注册的因子族名称

        Returns:
            去重排序的族名列表（不含空字符串的独立因子）
        """
        families = set()
        for m in self.list_all():
            if m.family:
                families.add(m.family)
        return sorted(families)

    def clear(self) -> None:
        """
        清空注册表

        仅影响当前实例，不影响其他实例或默认注册表。
        主要用于测试场景。
        """
        self._registry.clear()

    def __len__(self) -> int:
        """返回已注册因子数量"""
        return len(self._registry)

    def __contains__(self, name: str) -> bool:
        """检查因子是否已注册"""
        return name in self._registry


# =========================================================================
# 模块级默认注册表（全局单例行为通过此实现）
# =========================================================================

_default_registry = FactorRegistry()


def get_default_registry() -> FactorRegistry:
    """
    获取模块级默认注册表

    所有 @register_factor / @register_factor_family 装饰器注册到此注册表。
    FactorEngine 默认使用此注册表。

    Returns:
        全局默认注册表实例
    """
    return _default_registry


def register_factor(cls):
    """
    单因子注册装饰器

    用于在因子类定义时自动注册到默认注册表。
    适用于无参数化需求的独立因子。

    用法:
        @register_factor
        class MyFactor(TimeSeriesFactor):
            def meta(self) -> FactorMeta:
                return FactorMeta(name="my_factor", ...)

            def compute_single(self, symbol, data):
                ...

    Args:
        cls: 被装饰的因子类

    Returns:
        原始类（不做任何修改，只是注册到默认注册表）
    """
    _default_registry.register(cls)
    return cls


def register_factor_family(cls):
    """
    因子族注册装饰器

    读取类属性 _param_grid，对参数做笛卡尔积，
    为每个参数组合调用 registry.register(cls, **params)。

    _param_grid 格式:
        dict[str, list]  — {参数名: 可选值列表}
        参数值可以是任意可哈希类型（int, float, str, tuple 等）
        由因子作者人工指定所有可选取值。

    如果类没有 _param_grid 属性或为空，行为退化为 @register_factor
    （注册单个默认参数实例）。

    用法:
        @register_factor_family
        class MultiScaleReturns(TimeSeriesFactor):
            _param_grid = {"lookback": [5, 10, 30, 60]}

            def __init__(self, lookback: int = 5):
                self.lookback = lookback

            def meta(self) -> FactorMeta:
                return FactorMeta(
                    name=f"returns_{self.lookback}m",
                    family="multi_scale_returns",
                    params={"lookback": self.lookback},
                    ...
                )

    Args:
        cls: 被装饰的因子类

    Returns:
        原始类（不做任何修改，只是注册到默认注册表）
    """
    param_grid = getattr(cls, "_param_grid", None)

    if not param_grid:
        # 无参数网格，退化为普通注册
        _default_registry.register(cls)
        return cls

    # 笛卡尔积展开
    keys = list(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]

    for combo in itertools.product(*value_lists):
        params = dict(zip(keys, combo))
        _default_registry.register(cls, **params)

    return cls
