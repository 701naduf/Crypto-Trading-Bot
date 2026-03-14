"""
因子研究模块集中配置

将 factor_research 的可调参数集中管理，解耦对 data_infra 的直接依赖。
所有模块通过此文件读取配置，避免散落在各处的魔法数字。

修改指南:
    修改此文件中的常量即可全局生效，无需逐模块修改。
    但请注意：修改后需运行全量回归测试确认无破坏。

依赖: data_infra.config.settings (仅此一处引用)
"""

from pathlib import Path

from data_infra.config import settings

# ── 因子存储路径 ──
FACTOR_STORE_DIR = Path(settings.DB_DIR) / "factors"

# ── 引擎默认参数 ──
DEFAULT_SYMBOLS = settings.SYMBOLS

# ── 因子评价默认参数 ──
DEFAULT_HORIZONS = [1, 5, 10, 30, 60]
DEFAULT_N_GROUPS = 5
DEFAULT_TAIL_THRESHOLD = 0.9
DEFAULT_ROLLING_WINDOW = 60

# ── 最小样本数约束 ──
# IC 计算所需的最少截面观测数（标的数）
MIN_IC_OBSERVATIONS = 3
# 回归/相关性分析所需的最少样本数
MIN_REGRESSION_OBSERVATIONS = 10
