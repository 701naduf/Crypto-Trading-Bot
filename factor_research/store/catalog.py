"""
因子元数据目录

统一管理所有已入库因子的元信息，提供检索和摘要功能。
元数据来自 FactorStore 中各因子的 meta.json 文件。

与 FactorRegistry 的区别:
    - FactorRegistry 管理的是"已注册的因子类"（内存中的代码）
    - FactorCatalog 管理的是"已入库的因子值"（磁盘上的数据）
    - 一个因子可以已注册但尚未计算入库
    - 一个因子可以已入库但代码已被删除（历史因子值仍可用）

用法:
    catalog = FactorCatalog()
    print(catalog.summary())                        # 打印所有因子摘要表
    metas = catalog.search(category="momentum")     # 按分类搜索
    meta = catalog.get_meta("orderbook_imbalance")  # 获取单个因子元数据

依赖: store.factor_store
被依赖: evaluation.correlation, notebooks
"""

import pandas as pd

from data_infra.utils.logger import get_logger

from ..core.types import FactorMeta
from .factor_store import FactorStore

logger = get_logger(__name__)


class FactorCatalog:
    """
    因子元数据目录

    扫描 FactorStore 中所有已入库因子的 meta.json，
    提供检索、筛选和摘要表功能。
    """

    def __init__(self, store: FactorStore = None):
        """
        初始化目录

        扫描所有已存储因子的元数据。

        Args:
            store: FactorStore 实例，默认新建（使用默认路径 db/factors/）
        """
        self._store = store or FactorStore()
        self._metas: dict[str, FactorMeta] = {}
        self._refresh()

    def _refresh(self) -> None:
        """
        重新扫描因子存储目录，加载所有元数据

        静默跳过元数据文件损坏的因子（只记录警告日志）。
        """
        self._metas.clear()
        for name in self._store.list_factors():
            try:
                meta = self._store.load_meta(name)
                self._metas[name] = meta
            except Exception as e:
                logger.warning(f"加载因子 '{name}' 的元数据失败: {e}")

    def summary(self) -> pd.DataFrame:
        """
        返回所有因子的摘要表

        包含每个因子的名称、展示名、类型、分类、输出频率、版本等信息。
        方便在 notebook 中快速浏览所有已入库因子。

        Returns:
            pd.DataFrame: 摘要表，按因子名称排序
        """
        if not self._metas:
            return pd.DataFrame(
                columns=["name", "display_name", "factor_type",
                         "category", "output_freq", "version", "description"]
            )

        rows = []
        for name in sorted(self._metas.keys()):
            meta = self._metas[name]
            rows.append({
                "name": meta.name,
                "display_name": meta.display_name,
                "factor_type": meta.factor_type.value,
                "category": meta.category,
                "output_freq": meta.output_freq,
                "version": meta.version,
                "description": meta.description,
            })

        return pd.DataFrame(rows)

    def search(
        self,
        category: str = None,
        factor_type: str = None,
        family: str = None,
    ) -> list[FactorMeta]:
        """
        按条件搜索因子

        Args:
            category:    分类名（如 "microstructure"）
            factor_type: 因子类型值（如 "time_series"）
            family:      因子族名（如 "multi_scale_returns"）

        Returns:
            满足条件的 FactorMeta 列表
        """
        results = list(self._metas.values())

        if category is not None:
            results = [m for m in results if m.category == category]

        if factor_type is not None:
            results = [m for m in results if m.factor_type.value == factor_type]

        if family is not None:
            results = [m for m in results if m.family == family]

        return sorted(results, key=lambda m: m.name)

    def get_meta(self, factor_name: str) -> FactorMeta:
        """
        获取单个因子的元数据

        Args:
            factor_name: 因子名称

        Returns:
            FactorMeta: 因子元数据

        Raises:
            KeyError: 如果因子未入库
        """
        if factor_name not in self._metas:
            available = ", ".join(sorted(self._metas.keys())) or "(无)"
            raise KeyError(
                f"因子 '{factor_name}' 未入库。已入库因子: {available}"
            )
        return self._metas[factor_name]

    def list_families(self) -> list[str]:
        """
        列出所有已入库的因子族名称

        Returns:
            去重排序的族名列表（不含独立因子的空字符串）
        """
        families = set()
        for meta in self._metas.values():
            if meta.family:
                families.add(meta.family)
        return sorted(families)

    def family_summary(self, family_name: str) -> pd.DataFrame:
        """
        指定因子族的参数-变体摘要表

        返回族内所有变体的参数对比，方便一眼看出参数覆盖范围。

        Args:
            family_name: 族名（如 "multi_scale_returns"）

        Returns:
            pd.DataFrame: 族内变体摘要表。空族返回空 DataFrame。

        示例输出:
            | name          | lookback | output_freq | version |
            |---------------|----------|-------------|---------|
            | returns_5m    | 5        | 1m          | 1.0     |
            | returns_10m   | 10       | 1m          | 1.0     |
        """
        family_metas = self.search(family=family_name)
        if not family_metas:
            return pd.DataFrame()

        rows = []
        for meta in family_metas:
            row = {"name": meta.name, **meta.params,
                   "output_freq": meta.output_freq, "version": meta.version}
            rows.append(row)
        return pd.DataFrame(rows)

    def __len__(self) -> int:
        """返回已入库因子数量"""
        return len(self._metas)

    def __contains__(self, factor_name: str) -> bool:
        """检查因子是否已入库"""
        return factor_name in self._metas
