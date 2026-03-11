"""
数据模块

包含数据采集、存储、读取、聚合、校验等子模块。
所有子模块仅依赖 config 和 utils，不依赖上层业务逻辑。

主要入口:
    from data.writer import DataWriter   # 采集脚本写入数据
    from data.reader import DataReader   # 上层模块读取数据
"""
