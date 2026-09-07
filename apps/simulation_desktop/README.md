# 通信仿真实验独立应用

业务源码位于 [src/communication_sim](../../src/communication_sim)，共用基础位于 [src/common](../../src/common)。本目录包含专用 `main.py` 和正式发布元数据 `pyproject.toml`。

在仓库根目录安装开发环境后运行 `python -m communication_sim gui` 或 `python apps/simulation_desktop/main.py gui`。默认使用 `workspace_data/simulation`，不提供分析资产或原生信号插件入口。

独立发布使用 `python scripts/build_wheels.py simulation`。构建脚本只收集仿真源码与 common，不直接在本目录执行 `pip install .`。

使用生成的 wheel 执行 `python scripts/build_desktop.py simulation WHEEL_PATH`，输出 `dist/CommunicationSim`。正式部署建议使用独立虚拟环境。

当前是通用事件队列实验，专用通信模型仍待接入。功能和测试说明见[根 README](../../README.md)和[设计文档](../../docs/某星通信仿真系统_Python技术方案.md)。
