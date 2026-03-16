"""
参考模型实现（示例代码，非核心架构）

包含:
    linear_models.py  Ridge, Lasso, ElasticNet 等 sklearn 封装示例
    tree_models.py    LightGBM, XGBoost 封装示例
    torch_base.py     PyTorch 基础封装（含 val 拆分 + early stopping）

这些是示例代码，用户可以直接使用、继承修改、或完全忽略。
任何实现了 fit/predict 的对象都可以接入框架。
"""