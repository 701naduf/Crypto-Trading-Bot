"""
因子值 Parquet 存储模块

负责因子面板的持久化存储和读取。每个因子独占一个目录，
内含一个 output.parquet（因子值面板）和一个 meta.json（因子元数据）。

路径格式:
    db/factors/{factor_name}/output.parquet
    db/factors/{factor_name}/meta.json

存储格式:
    output.parquet 的内容与因子面板 DataFrame 一一对应:
        - 列 "timestamp": DatetimeIndex (UTC)
        - 其余列: 各 symbol 的因子值（float64）
    读取时自动还原为 index=timestamp 的 DataFrame。

    meta.json 记录 FactorMeta 的所有字段，便于离线查询因子信息。

原子写入:
    继承 data_infra 的原子写入策略：先写 .tmp 文件，再 rename。
    防止写入中途崩溃导致数据损坏。

依赖: core.types, factor_research.config
被依赖: core.engine, store.catalog, evaluation.analyzer
"""

import json
import os
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from data_infra.utils.logger import get_logger

from ..config import FACTOR_STORE_DIR
from ..core.types import DataRequest, DataType, FactorMeta, FactorType

logger = get_logger(__name__)


class FactorStore:
    """
    因子值持久化存储

    管理因子面板的 Parquet 文件和元数据 JSON 文件。
    是因子计算管道和下游模型管道之间的唯一接口。

    目录结构示例:
        db/factors/
        ├── orderbook_imbalance_10s/
        │   ├── output.parquet
        │   └── meta.json
        ├── btc_lead_lag/
        │   ├── output.parquet
        │   └── meta.json
        └── ...
    """

    def __init__(self, base_dir: str = None):
        """
        初始化因子存储

        Args:
            base_dir: 因子存储根目录，默认 db/factors/
                      测试时可传入临时目录。
        """
        if base_dir is None:
            self._base_dir = Path(FACTOR_STORE_DIR)
        else:
            self._base_dir = Path(base_dir)

        # 确保根目录存在
        self._base_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"FactorStore 已初始化: {self._base_dir}")

    def _factor_dir(self, factor_name: str) -> Path:
        """获取因子目录路径"""
        return self._base_dir / factor_name

    def save(
        self,
        factor_name: str,
        panel: pd.DataFrame,
        meta: FactorMeta,
    ) -> None:
        """
        保存因子面板和元数据

        使用原子写入策略: 先写 .tmp 再 rename，防止崩溃损坏。

        Args:
            factor_name: 因子名称（用作目录名）
            panel:       因子面板 DataFrame (index=timestamp, columns=symbols)
            meta:        因子元数据

        Raises:
            ValueError: 如果面板为空
        """
        if panel.empty:
            raise ValueError(f"因子 '{factor_name}' 的面板为空，不保存")

        factor_dir = self._factor_dir(factor_name)
        factor_dir.mkdir(parents=True, exist_ok=True)

        # --- 保存因子面板 (Parquet, 原子写入) ---
        parquet_path = factor_dir / "output.parquet"
        tmp_path = factor_dir / "output.parquet.tmp"

        # 确保 index 名为 timestamp，重置为列以便存储
        save_df = panel.copy()
        save_df.index.name = "timestamp"
        save_df = save_df.reset_index()

        save_df.to_parquet(str(tmp_path), engine="pyarrow", index=False)
        # 原子替换: Windows 上 os.replace 也是原子的
        os.replace(str(tmp_path), str(parquet_path))

        # --- 保存元数据 (JSON) ---
        meta_path = factor_dir / "meta.json"
        meta_dict = self._meta_to_dict(meta)

        tmp_meta_path = factor_dir / "meta.json.tmp"
        with open(str(tmp_meta_path), "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, ensure_ascii=False, indent=2)
        os.replace(str(tmp_meta_path), str(meta_path))

        rows, cols = panel.shape
        logger.info(
            f"因子已保存: {factor_name} | "
            f"{rows} 行 × {cols} 列 | "
            f"路径: {parquet_path}"
        )

    def load(self, factor_name: str) -> pd.DataFrame:
        """
        加载因子面板

        读取 Parquet 文件，将 timestamp 列还原为 DatetimeIndex。

        Args:
            factor_name: 因子名称

        Returns:
            pd.DataFrame: 因子面板 (index=DatetimeIndex, columns=symbols)

        Raises:
            FileNotFoundError: 如果因子未存储
        """
        parquet_path = self._factor_dir(factor_name) / "output.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"因子 '{factor_name}' 的数据文件不存在: {parquet_path}"
            )

        df = pd.read_parquet(str(parquet_path))

        # 还原 timestamp 为 index
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
            df.index = pd.to_datetime(df.index, utc=True)

        return df

    def load_meta(self, factor_name: str) -> FactorMeta:
        """
        加载因子元数据

        Args:
            factor_name: 因子名称

        Returns:
            FactorMeta: 因子元数据实例

        Raises:
            FileNotFoundError: 如果元数据文件不存在
        """
        meta_path = self._factor_dir(factor_name) / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"因子 '{factor_name}' 的元数据文件不存在: {meta_path}"
            )

        with open(str(meta_path), "r", encoding="utf-8") as f:
            meta_dict = json.load(f)

        return self._dict_to_meta(meta_dict)

    def list_factors(self) -> list[str]:
        """
        列出所有已存储的因子名称

        扫描因子根目录下的子目录，只有包含 output.parquet 的才算有效因子。

        Returns:
            排序后的因子名称列表
        """
        if not self._base_dir.exists():
            return []

        factors = []
        for item in self._base_dir.iterdir():
            if item.is_dir() and (item / "output.parquet").exists():
                factors.append(item.name)
        return sorted(factors)

    def exists(self, factor_name: str) -> bool:
        """
        检查因子是否已存储

        Args:
            factor_name: 因子名称

        Returns:
            bool: True 表示因子数据文件存在
        """
        return (self._factor_dir(factor_name) / "output.parquet").exists()

    def load_family(self, family_name: str) -> dict[str, pd.DataFrame]:
        """
        加载指定因子族的所有已存储变体

        遍历已存储因子，根据 meta.json 中的 family 字段筛选。

        Args:
            family_name: 族名（如 "multi_scale_returns"）

        Returns:
            {factor_name: 因子面板} 字典。
            如果族内无因子或族不存在，返回空字典。
        """
        results = {}
        for name in self.list_factors():
            try:
                meta = self.load_meta(name)
                if meta.family == family_name:
                    results[name] = self.load(name)
            except Exception:
                pass  # 静默跳过损坏的因子
        return results

    def list_families(self) -> list[str]:
        """
        列出所有已存储的因子族名称

        Returns:
            去重排序的族名列表（不含独立因子的空字符串）
        """
        families = set()
        for name in self.list_factors():
            try:
                meta = self.load_meta(name)
                if meta.family:
                    families.add(meta.family)
            except Exception:
                pass
        return sorted(families)

    def delete(self, factor_name: str) -> None:
        """
        删除因子数据

        删除因子目录下的所有文件和目录本身。

        Args:
            factor_name: 因子名称

        Raises:
            FileNotFoundError: 如果因子不存在
        """
        factor_dir = self._factor_dir(factor_name)
        if not factor_dir.exists():
            raise FileNotFoundError(f"因子 '{factor_name}' 不存在，无法删除")

        # 删除目录下所有文件
        for f in factor_dir.iterdir():
            f.unlink()
        factor_dir.rmdir()

        logger.info(f"因子已删除: {factor_name}")

    # ------------------------------------------------------------------
    # 私有方法: FactorMeta 与 dict 的互转
    # ------------------------------------------------------------------

    @staticmethod
    def _meta_to_dict(meta: FactorMeta) -> dict:
        """
        将 FactorMeta 转换为可 JSON 序列化的字典

        需要特殊处理:
            - FactorType / DataType 枚举转为字符串值
            - DataRequest dataclass 转为字典
        """
        d = asdict(meta)
        # 枚举 → 字符串
        d["factor_type"] = meta.factor_type.value
        # DataRequest 列表中的枚举也需要转换
        reqs = []
        for req in meta.data_requirements:
            req_dict = asdict(req)
            req_dict["data_type"] = req.data_type.value
            reqs.append(req_dict)
        d["data_requirements"] = reqs
        return d

    @staticmethod
    def _dict_to_meta(d: dict) -> FactorMeta:
        """
        从字典还原 FactorMeta

        需要特殊处理:
            - 字符串值 → FactorType / DataType 枚举
            - 字典 → DataRequest dataclass
        """
        # 还原 FactorType 枚举
        d["factor_type"] = FactorType(d["factor_type"])

        # 还原 DataRequest 列表
        reqs = []
        for req_dict in d.get("data_requirements", []):
            req_dict["data_type"] = DataType(req_dict["data_type"])
            reqs.append(DataRequest(**req_dict))
        d["data_requirements"] = reqs

        return FactorMeta(**d)
