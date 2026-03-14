"""store/catalog.py 的单元测试

覆盖: FactorCatalog 的完整功能
    - 空目录 / 有因子时的扫描行为
    - summary 列结构
    - search 按 category / factor_type 过滤
    - get_meta 正常 / 不存在
    - __contains__ / __len__
    - 损坏的 meta.json 容错
"""

import json

import numpy as np
import pandas as pd
import pytest

from factor_research.core.types import (
    DataRequest,
    DataType,
    FactorMeta,
    FactorType,
)
from factor_research.store.catalog import FactorCatalog
from factor_research.store.factor_store import FactorStore


# =========================================================================
# Fixtures
# =========================================================================

def _make_meta(name, category="test", factor_type=FactorType.TIME_SERIES,
               family="", params=None):
    """生成测试用 FactorMeta"""
    return FactorMeta(
        name=name,
        display_name=f"Display {name}",
        factor_type=factor_type,
        category=category,
        description=f"Description for {name}",
        data_requirements=[DataRequest(DataType.OHLCV, timeframe="1m")],
        output_freq="1m",
        params=params or {},
        family=family,
        version="1.0",
    )


def _make_panel(n=20):
    """生成测试用因子面板"""
    index = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {"BTC/USDT": np.random.randn(n), "ETH/USDT": np.random.randn(n)},
        index=index,
    )


@pytest.fixture
def store(tmp_path):
    return FactorStore(base_dir=str(tmp_path / "factors"))


# =========================================================================
# 测试
# =========================================================================

class TestFactorCatalog:

    def test_empty_catalog(self, store):
        """空目录: summary 为空 DataFrame, len 为 0"""
        catalog = FactorCatalog(store=store)
        assert len(catalog) == 0
        summary = catalog.summary()
        assert summary.empty

    def test_scan_after_save(self, store):
        """存储因子后扫描: summary 有正确行数"""
        store.save("factor_a", _make_panel(), _make_meta("factor_a"))
        store.save("factor_b", _make_panel(), _make_meta("factor_b", category="momentum"))
        catalog = FactorCatalog(store=store)
        assert len(catalog) == 2
        summary = catalog.summary()
        assert len(summary) == 2

    def test_summary_columns(self, store):
        """summary 包含所有必要列"""
        store.save("factor_a", _make_panel(), _make_meta("factor_a"))
        catalog = FactorCatalog(store=store)
        summary = catalog.summary()
        expected_cols = [
            "name", "display_name", "factor_type",
            "category", "output_freq", "version", "description",
        ]
        for col in expected_cols:
            assert col in summary.columns, f"缺少列: {col}"

    def test_search_by_category(self, store):
        """按 category 搜索"""
        store.save("f_mom", _make_panel(), _make_meta("f_mom", category="momentum"))
        store.save("f_vol", _make_panel(), _make_meta("f_vol", category="volatility"))
        store.save("f_mom2", _make_panel(), _make_meta("f_mom2", category="momentum"))
        catalog = FactorCatalog(store=store)

        results = catalog.search(category="momentum")
        assert len(results) == 2
        names = [m.name for m in results]
        assert "f_mom" in names
        assert "f_mom2" in names

    def test_search_by_factor_type(self, store):
        """按 factor_type 搜索"""
        store.save("f_ts", _make_panel(), _make_meta("f_ts", factor_type=FactorType.TIME_SERIES))
        store.save("f_cs", _make_panel(), _make_meta("f_cs", factor_type=FactorType.CROSS_SECTIONAL))
        catalog = FactorCatalog(store=store)

        results = catalog.search(factor_type="time_series")
        assert len(results) == 1
        assert results[0].name == "f_ts"

    def test_get_meta(self, store):
        """获取单个因子元数据"""
        meta_in = _make_meta("factor_a", category="momentum")
        store.save("factor_a", _make_panel(), meta_in)
        catalog = FactorCatalog(store=store)

        meta_out = catalog.get_meta("factor_a")
        assert meta_out.name == "factor_a"
        assert meta_out.category == "momentum"

    def test_get_meta_nonexistent_raises(self, store):
        """获取不存在的因子 → KeyError"""
        catalog = FactorCatalog(store=store)
        with pytest.raises(KeyError, match="未入库"):
            catalog.get_meta("nonexistent")

    def test_contains(self, store):
        """__contains__ 支持 in 操作"""
        store.save("factor_a", _make_panel(), _make_meta("factor_a"))
        catalog = FactorCatalog(store=store)
        assert "factor_a" in catalog
        assert "nonexistent" not in catalog

    def test_corrupted_meta_json_skipped(self, store, tmp_path):
        """损坏的 meta.json 被静默跳过，不影响其他因子"""
        # 正常存储一个因子
        store.save("good_factor", _make_panel(), _make_meta("good_factor"))

        # 手动创建一个损坏的因子目录
        bad_dir = tmp_path / "factors" / "bad_factor"
        bad_dir.mkdir(parents=True, exist_ok=True)
        # 创建空 output.parquet 使 list_factors 能发现它
        _make_panel().to_parquet(str(bad_dir / "output.parquet"))
        # 写入损坏的 meta.json
        with open(str(bad_dir / "meta.json"), "w") as f:
            f.write("{invalid json content")

        catalog = FactorCatalog(store=store)
        # good_factor 正常加载，bad_factor 被跳过
        assert "good_factor" in catalog
        assert "bad_factor" not in catalog
        assert len(catalog) == 1


class TestCatalogFamilyFeatures:
    """因子族相关的 Catalog 功能测试（Batch 2 新增）"""

    def test_search_by_family(self, store):
        """search(family=...) 按族名筛选"""
        store.save("ret_5m", _make_panel(),
                   _make_meta("ret_5m", family="returns", params={"lookback": 5}))
        store.save("ret_10m", _make_panel(),
                   _make_meta("ret_10m", family="returns", params={"lookback": 10}))
        store.save("imb", _make_panel(), _make_meta("imb"))  # 独立因子

        catalog = FactorCatalog(store=store)
        results = catalog.search(family="returns")
        assert len(results) == 2
        names = {m.name for m in results}
        assert names == {"ret_5m", "ret_10m"}

    def test_search_family_combined(self, store):
        """search 同时按 family + category 筛选"""
        store.save("ret_5m", _make_panel(),
                   _make_meta("ret_5m", category="momentum", family="returns"))
        store.save("vol_5m", _make_panel(),
                   _make_meta("vol_5m", category="volatility", family="vol"))
        catalog = FactorCatalog(store=store)

        results = catalog.search(category="momentum", family="returns")
        assert len(results) == 1
        assert results[0].name == "ret_5m"

    def test_list_families(self, store):
        """list_families() 返回去重排序的族名"""
        store.save("ret_5m", _make_panel(),
                   _make_meta("ret_5m", family="returns", params={"lookback": 5}))
        store.save("ret_10m", _make_panel(),
                   _make_meta("ret_10m", family="returns", params={"lookback": 10}))
        store.save("vol_5m", _make_panel(),
                   _make_meta("vol_5m", family="vol_family"))
        store.save("imb", _make_panel(), _make_meta("imb"))  # 独立因子

        catalog = FactorCatalog(store=store)
        families = catalog.list_families()
        assert families == ["returns", "vol_family"]
        assert "" not in families

    def test_list_families_empty(self, store):
        """无因子时 list_families() 返回空列表"""
        catalog = FactorCatalog(store=store)
        assert catalog.list_families() == []

    def test_family_summary(self, store):
        """family_summary() 返回族内变体摘要表"""
        store.save("ret_5m", _make_panel(),
                   _make_meta("ret_5m", family="returns", params={"lookback": 5}))
        store.save("ret_10m", _make_panel(),
                   _make_meta("ret_10m", family="returns", params={"lookback": 10}))
        catalog = FactorCatalog(store=store)

        summary = catalog.family_summary("returns")
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2
        assert "name" in summary.columns
        assert "lookback" in summary.columns
        assert set(summary["lookback"]) == {5, 10}

    def test_family_summary_nonexistent(self, store):
        """不存在的族 → 空 DataFrame"""
        catalog = FactorCatalog(store=store)
        summary = catalog.family_summary("nonexistent")
        assert summary.empty
