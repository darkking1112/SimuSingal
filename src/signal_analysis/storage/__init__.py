"""存储设施层：数据库 schema、迁移与工作区维护。

领域操作（资产、集合、标注、数据集）见 ``signal_analysis.data``；
本包不导入领域实现。schema 常量与 DDL 见 :mod:`.schema`，通用校验与
排序工具见 :mod:`.utils`，工作区维护（盘点、清理、历史登记）见
:mod:`.maintenance`。
"""
