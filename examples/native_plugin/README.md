# DLL/SO 与 Python 调用最小示例

本示例只复制 float32 数组，用于验证通用 C ABI、调用约定、输出容量检查和子进程调用，不包含信号算法。它与[生产插件接口草案](../../docs/核心模块二进制化与原生插件接口方案.md)中的完整 SDK 分开命名。

基础工作台现已通过 `simusignal.plugins` 接入本演示 ABI。库编译完成后，执行 `python -m simusignal plugin-manifest 动态库路径 清单路径` 生成本机清单，再在界面选择清单，或运行 `python -m simusignal native ASSET_ID 清单路径`。清单应位于动态库目录或其上级目录；库更新后需重新生成摘要。完整示例命令见[项目 README](../../README.md)。

文件包括 [C 头文件](demo_api.h)、[C 实现](demo_plugin.c)、[CMake 构建文件](CMakeLists.txt)和 [Python 宿主](host.py)。示例宿主只依赖 Python 标准库，无需 NumPy 或 Qt。

## Linux 构建和调用

在仓库根目录执行，需要 C 编译器、CMake 和 Python 3：

```bash
cmake -S examples/native_plugin -B /tmp/simusignal-native-demo
cmake --build /tmp/simusignal-native-demo --config Release
python3 examples/native_plugin/host.py /tmp/simusignal-native-demo/libdemo_plugin.so
```

## Windows 构建和调用

在已安装 CMake、Python 及 Visual Studio C++ 工具的环境中，从仓库根目录执行。下例针对 x64 的 Visual Studio 生成器，Python 也应为 x64：

```powershell
cmake -S examples/native_plugin -B build/native-demo -A x64
cmake --build build/native-demo --config Release
python examples/native_plugin/host.py build/native-demo/Release/demo_plugin.dll
```

使用其他 CMake 生成器时，调整架构参数和产物位置。Windows、麒麟及不同 CPU 架构需要分别构建和测试；本轮 Linux 结果不代替目标机验收。

## 预期结果

```json
{"abi": 1, "output": [1.0, -2.5, 0.0, 3.25], "checks_passed": 5}
```

五项自检为数组复制、容量不足且输出未变化、空输入、非法空输入指针及非法输出计数指针。计数单位为 float 元素；输入输出内存均由 Python 工作进程分配，库不保留指针。

宿主先启动子进程，子进程才加载本机库。库加载失败或 Python 层错误会返回任务失败；本机崩溃通过退出码处理；超过 10 秒终止子进程。此示例只测试固定的小数组，不是通用生产插件加载器。

独立 `host.py` 保持最小示例形式。工作台已实现基础清单检查、任务超时/取消、日志阈值及冻结程序 worker 入口；正式 ABI 生命周期、插件安装管理、完整资源配额和依赖链验证仍需补齐。`host.py` 使用脚本路径启动 worker；打包应用采用 `simusignal.tasks` 的同一可执行文件 worker 入口。

## 本轮验证记录

2026-09-06，在 Linux x86_64、CPython 3.12.3、GCC 13.3.0 环境通过 CMake 构建及上述五项自检。另使用临时测试库验证错误 ABI、缺少处理符号、子进程 SIGSEGV 和超时均由宿主报告失败，并验证缺失路径在加载前被拒绝。

这些结果只覆盖最小示例及本机环境；Windows DLL、指定麒麟系统、完整 SDK 和冻结后的桌面应用尚未验证。库文件保存在临时构建目录，不作为已交付的平台二进制包。
