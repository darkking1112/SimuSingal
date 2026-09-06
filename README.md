# SimuSignal

基于 Python 的离线数据分析与通用事件仿真基础工程。界面使用 PySide6 / PyQtGraph，核心计算可编译为 Python 二进制扩展，支持外置原生复制示例插件。

当前版本为 **0.1.0 基础原型**：已有数据导入、数学演示数据、统计与时频图形、消息队列仿真、运行记录和报告。自动调制识别、专用星地协议及正式算法 SDK 尚未实现；界面明确显示对应边界。

## 启动

需要 Python 3.10 以上。以下命令在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[gui,dev]'
python -m simusignal --workspace workspace_data gui
```

Windows 用 `python -m venv .venv` 创建环境，然后在 PowerShell 执行 `.venv\Scripts\Activate.ps1`，其余 Python 命令相同。图形环境不可用时，可使用命令行运行基础流程。

当前工作区已安装开发依赖，Linux 可直接执行：

```bash
.venv/bin/python -m simusignal gui
```

界面操作：生成演示或导入文件 → 选择资产 → 分析 → 查看波形、频谱、时频图和瀑布图 → 保存备注或导出报告。事件仿真页可修改通用队列参数、运行并拖动滑块回放；运行记录页可打开已保存的结果。

![数据分析工作台](docs/images/analysis-workbench.png)

![通用事件仿真工作台](docs/images/simulation-workbench.png)

## 命令行

`--workspace` 放在子命令前；省略时使用当前目录下的 `workspace_data`。

```bash
python -m simusignal demo --count 8192 --sample-rate 48000
python -m simusignal import data.npy --sample-rate 48000
python -m simusignal list
python -m simusignal analyze ASSET_ID --nfft 256
python -m simusignal simulate --messages 12 --duration 3
python -m simusignal export RUN_ID report.html
```

将 `ASSET_ID` / `RUN_ID` 替换为前一步返回的 ID。NPY 必须是一维实数或复数数值数组；CSV 必须无表头，一列为实数，两列为 I,Q。基础版单文件限制为 64 MiB、最多 1,000,000 个采样点，采样率由用户明确填写；不会猜测未知 `.bin` 的含义。

## 用户 DLL/SO 示例

先按[原生示例说明](examples/native_plugin/README.md)编译，再生成包含本机架构和库摘要的清单：

```bash
cmake -S examples/native_plugin -B /tmp/simusignal-native-demo
cmake --build /tmp/simusignal-native-demo --config Release
python -m simusignal plugin-manifest /tmp/simusignal-native-demo/libdemo_plugin.so /tmp/simusignal-native-demo/plugin.json
python -m simusignal native ASSET_ID /tmp/simusignal-native-demo/plugin.json
```

也可在数据分析页点击“加载原生复制插件清单”选择 `plugin.json`。只加载用户明确选择、符合演示 ABI 的库；输出作为新资产保存。正式 SDK 的生命周期接口仍以设计草案为准。

## 测试

```bash
QT_QPA_PLATFORM=offscreen python -m pytest -q
```

Windows PowerShell 先执行 `$env:QT_QPA_PLATFORM="offscreen"`，再运行 `python -m pytest -q`。原生测试需要 CMake 和 C 编译器；Unix 崩溃测试在没有 `cc` 时跳过。GUI 测试缺少图形依赖时跳过，跳过项不能视为通过。

## 核心编译和桌面打包

Linux 示例；Windows 在设置对应环境变量后执行相同 Python 构建命令，产物名随平台和 Python 版本变化：

```bash
SIMUSIGNAL_COMPILE_CORE=1 python -m build --wheel --no-isolation
python scripts/check_binary.py dist/simusignal-0.1.0-cp312-cp312-linux_x86_64.whl
python scripts/build_desktop.py dist/simusignal-0.1.0-cp312-cp312-linux_x86_64.whl
```

Windows 环境变量设置为 `$env:SIMUSIGNAL_COMPILE_CORE="1"`。示例只将 `_numeric.py` 编译为 `.pyd` / `.so`；其他应用层模块仍为 Python。二进制 wheel 不附带该核心 `.py`，核心源码继续保留在源码交付包中。编译并不承诺不可逆向或自动加速。

`build_desktop.py` 使用传入的编译 wheel 在临时目录构建目录式应用，默认输出到 `dist/SimuSignal`。应用支持同一可执行文件启动后台 worker；插件外置，不必重打主程序。Windows 和麒麟必须在对应目标环境单独构建和验收。

## 文档

- [基础工程实现与逐文件说明](docs/基础工程实现与文件说明.md)：实现范围、数据流、所有新增文件、测试和后续工作。
- [设计总览](docs/Python技术方案总览.md)
- [电磁信号分析识别系统设计](docs/电磁信号分析识别系统_Python技术方案.md)
- [某星通信仿真系统设计](docs/某星通信仿真系统_Python技术方案.md)
- [二进制化与原生插件接口设计](docs/核心模块二进制化与原生插件接口方案.md)
