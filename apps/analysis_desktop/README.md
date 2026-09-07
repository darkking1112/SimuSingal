# 电磁信号分析独立应用

业务源码位于 [src/signal_analysis](../../src/signal_analysis)，共用基础位于 [src/common](../../src/common)。本目录包含专用 `main.py` 和正式发布元数据 `pyproject.toml`。

在仓库根目录安装开发环境后运行 `python -m signal_analysis gui` 或 `python apps/analysis_desktop/main.py gui`。默认使用 `workspace_data/analysis`。

独立发布使用 `python scripts/build_wheels.py analysis --compile-core`。构建脚本把本项目元数据、分析源码和 common 收集到临时构建目录，不收集仿真源码；不直接在本目录执行 `pip install .`。

使用生成的 wheel 执行 `python scripts/build_desktop.py analysis WHEEL_PATH`，输出 `dist/SignalAnalysis`。建议在独立虚拟环境安装正式 wheel：`python -m pip install 'WHEEL_PATH[gui]'`，把占位符换成真实文件路径。

功能、原生插件及测试说明见[根 README](../../README.md)和[设计文档](../../docs/电磁信号分析识别系统_Python技术方案.md)。
