"""
时间序列切分器

为 Walk-Forward 训练提供严格无未来信息的训练/测试索引划分。

核心约束:
    1. 时序顺序：训练集永远在测试集之前
    2. Embargo period: 训练集和测试集之间的隔离期
       embargo = max(target_horizon, max_factor_lookback)
       - 防止 forward return 标签泄漏（标签窗口重叠）
       - 防止因子 lookback 窗口间接使用测试集数据

两种模式:
    Expanding: 训练起点固定(0)，随 fold 推进数据越来越多
    Rolling:   训练窗口固定大小，整体向前滑动

依赖: core.types.WalkForwardMode
"""

from __future__ import annotations

from dataclasses import dataclass

from alpha_model.core.types import WalkForwardMode


@dataclass
class Fold:
    """
    一个训练/测试切分

    Attributes:
        fold_id:     Fold 序号（从 0 开始）
        train_start: 训练集起始索引（含）
        train_end:   训练集结束索引（不含）
        test_start:  测试集起始索引（含）
        test_end:    测试集结束索引（不含）
    """
    fold_id: int
    train_start: int       # 训练集起始索引（含）
    train_end: int         # 训练集结束索引（不含）
    test_start: int        # 测试集起始索引（含）
    test_end: int          # 测试集结束索引（不含）

    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start

    @property
    def test_size(self) -> int:
        return self.test_end - self.test_start


class TimeSeriesSplitter:
    """
    时序切分器

    根据配置生成 Walk-Forward 的训练/测试切分。
    支持 Expanding 和 Rolling 两种模式。

    用法:
        splitter = TimeSeriesSplitter(
            train_periods=3000, test_periods=1000,
            target_horizon=10, max_factor_lookback=60,
            mode=WalkForwardMode.EXPANDING,
        )
        folds = splitter.split(n_samples=10000)
        for fold in folds:
            X_train = X[fold.train_start:fold.train_end]
            X_test = X[fold.test_start:fold.test_end]
    """

    def __init__(
        self,
        train_periods: int,
        test_periods: int,
        target_horizon: int,
        max_factor_lookback: int = 0,
        mode: WalkForwardMode = WalkForwardMode.EXPANDING,
    ):
        """
        Args:
            train_periods:        训练窗口长度
            test_periods:         测试窗口长度
            target_horizon:       预测目标的前瞻窗口长度
            max_factor_lookback:  因子中最大的 lookback 窗口长度
                                  （如 60 分钟 rolling window → 60）
            mode:                 切分模式

        Embargo period 自动计算:
            embargo_periods = max(target_horizon, max_factor_lookback)
        """
        if train_periods < 1:
            raise ValueError(f"train_periods 必须 >= 1, 收到 {train_periods}")
        if test_periods < 1:
            raise ValueError(f"test_periods 必须 >= 1, 收到 {test_periods}")
        if target_horizon < 1:
            raise ValueError(f"target_horizon 必须 >= 1, 收到 {target_horizon}")
        if max_factor_lookback < 0:
            raise ValueError(
                f"max_factor_lookback 必须 >= 0, 收到 {max_factor_lookback}"
            )

        self.train_periods = train_periods
        self.test_periods = test_periods
        self.target_horizon = target_horizon
        self.max_factor_lookback = max_factor_lookback
        self.mode = mode

        # Embargo period: 防止标签泄漏 + 因子 lookback 泄漏
        self.embargo_periods = max(target_horizon, max_factor_lookback)

    def split(self, n_samples: int) -> list[Fold]:
        """
        生成所有 fold

        算法:
            Expanding 模式:
                train_start 固定为 0
                第一个 fold: train_end = train_periods
                后续 fold: train_end = 上一个 fold 的 test_end
                test_start = train_end + embargo_periods
                test_end = test_start + test_periods

            Rolling 模式:
                第一个 fold: train_start=0, train_end=train_periods
                后续 fold: train 窗口前移，大小保持 train_periods
                test_start = train_end + embargo_periods
                test_end = test_start + test_periods

            终止条件: test_end > n_samples 时停止

        Args:
            n_samples: 总样本数

        Returns:
            Fold 列表

        Raises:
            ValueError: 样本数不足以生成至少一个 fold
        """
        min_required = self.train_periods + self.embargo_periods + self.test_periods
        if n_samples < min_required:
            raise ValueError(
                f"样本数 {n_samples} 不足以生成 fold，"
                f"至少需要 {min_required} "
                f"(train={self.train_periods} + embargo={self.embargo_periods} "
                f"+ test={self.test_periods})"
            )

        folds = []
        fold_id = 0

        if self.mode == WalkForwardMode.EXPANDING:
            # Expanding: 训练起点固定为 0，终点逐步扩展
            train_end = self.train_periods

            while True:
                test_start = train_end + self.embargo_periods
                test_end = test_start + self.test_periods

                if test_end > n_samples:
                    # 尝试缩短最后一个 fold 的测试窗口
                    # 但至少要有 1 个测试样本
                    if test_start < n_samples:
                        test_end = n_samples
                    else:
                        break

                folds.append(Fold(
                    fold_id=fold_id,
                    train_start=0,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                ))
                fold_id += 1

                # 下一个 fold 的训练集扩展到本 fold 的 test_end
                train_end = test_end

        elif self.mode == WalkForwardMode.ROLLING:
            # Rolling: 训练窗口固定大小，整体滑动
            train_start = 0
            train_end = self.train_periods

            while True:
                test_start = train_end + self.embargo_periods
                test_end = test_start + self.test_periods

                if test_end > n_samples:
                    if test_start < n_samples:
                        test_end = n_samples
                    else:
                        break

                folds.append(Fold(
                    fold_id=fold_id,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                ))
                fold_id += 1

                # 下一个 fold: 整体向前滑动 test_periods
                train_start = train_start + self.test_periods
                train_end = train_start + self.train_periods

        return folds

    def n_splits(self, n_samples: int) -> int:
        """返回 fold 数量"""
        return len(self.split(n_samples))

    def __repr__(self) -> str:
        return (
            f"TimeSeriesSplitter("
            f"mode={self.mode.value}, "
            f"train={self.train_periods}, "
            f"test={self.test_periods}, "
            f"embargo={self.embargo_periods})"
        )
