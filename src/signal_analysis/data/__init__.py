"""数据领域层：资产、集合、标注、数据集的文件与数据库操作。

对上层（服务/界面/CLI）的公共入口是 :class:`.workspace.Workspace`（表操作与
领域规则），文件级读写见 :mod:`.io` / :mod:`.sigmf`，标注数据集见
:mod:`.annotations`，数据版本构建见 :mod:`.datasets`，采纳写入见 :mod:`.adoption`。
"""

from .assets import ShardWriter
from .workspace import Workspace

__all__ = ["ShardWriter", "Workspace"]
