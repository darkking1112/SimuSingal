# 调制识别：`custom/` 第三方模型接入指南（包装层契约与模板）

> 适用：把 `custom/` 下别人给的**推理脚本**接进 AMC 原始 IQ 通路的训练/导出/验收链路
> （`training/train_iq.py --arch <id>`）。训练策略与契约本体见
> [training/README.md](../../../training/README.md) §7，设计取舍见
> [模型扩展实施方案](调制识别_模型扩展实施方案.md)。

## 0. 一句话结论

`custom/` 里的文件**只需要留"模型定义"**：**一个 `nn.Module` 模型类 + 它引用的子模块/辅助函数 + 顶部 import**。
其余部分包装层**一个都不会调用**——`__main__` 演示块、论文配置字典（`get_config_rml2016`、`get_poet_config_*`、
`get_amc_net_config_rml2016`）、参数量统计 `numParams`、`device = torch.device(...)`、训练循环/评估/画图/数据加载/argparse。

接入时**不给 `custom/` 打补丁**：要么直接包装（路径 A，原文件一字不改），要么把模型部分拷成维护副本
（路径 B，原文件仍是只读来源）。两条路径都只改训练侧。

## 1. 加载机制：包装层到底做了什么

`training/amc_models/_custom.py` 只做一件事：**按文件路径加载** `custom/<名字>.py`。

```python
module = load_custom_module("MCLDNN")      # → custom/MCLDNN.py，模块名 amc_custom_MCLDNN
```

由此有四条硬约束（写自定义文件前先对照）：

| # | 事实 | 后果 |
| --- | --- | --- |
| 1 | 用 `importlib.util.spec_from_file_location` 按路径加载，模块名 `amc_custom_<文件名>`，同进程只加载一次；文件不存在直接 `FileNotFoundError`（不静默跳过） | 文件名必须 `custom/<名>.py`；不要指望"改名也能找到" |
| 2 | **不注入 `sys.path`**，`custom/` 不是包也没有 `__init__.py` | 文件里**不能**有兄弟文件导入（`import POET`）或相对导入（`from . import x`）；每个文件必须自包含 |
| 3 | 顶层语句在**第一次构建模型时执行一次**（模块级 `device = ...`、大张量、打印都算副作用） | 顶层只留 import 与类/函数定义；设备、随机种子、采样都由训练器统一处理 |
| 4 | 需要改的东西只能落在 `training/amc_models/<id>.py` | 原文件"不改"是硬约束；`tests/analysis/test_iq_training_tools.py` 还会校验 CV_TRN/POET 原文件保持带 timm 导入的原样 |

附带的依赖规则：原文件顶部 import 的第三方库（如 `timm`）必须写进目录条目的 `requires`，
否则缺依赖时是导入期 `ModuleNotFoundError`，而不是目录给出的"请在训练环境安装 X"提示
（`amc_models.missing_requirements()` 只检查 `requires` 里声明过的名字）。

## 2. `custom/` 文件：留什么、删什么

**必须保留**

1. 模型类本体（`nn.Module` 子类）与其 `forward`；
2. 该类**递归引用**的全部子模块类与辅助函数（例如 `CausalConv1d`、`PET`、`get_same_padding`、`stft_sig`）；
3. 这些类用到的顶部 import（`torch` / `torch.nn` / `numpy` / `math` …）。

**不需要（删掉不影响接入）**

| 不需要的内容 | 为什么 |
| --- | --- |
| `if __name__ == '__main__':` 演示/计时/形状打印块 | 包装层用模块名 `amc_custom_*` 加载，`__main__` 块**永远不会执行** |
| `get_config_rml2016` / `get_config_rml2018` / `get_poet_config_*` / `get_amc_net_config_rml2016` | 只被 `__main__` 块调用；参数默认值的唯一事实源是目录条目的 `ParamSpec.default` |
| `numParams(net)` | 参数量统计，项目里没有调用点 |
| 顶层 `device = torch.device('cuda' ...)` | 设备由训练器统一 `build_model(...).to(device)` 控制（`trainer.py`） |
| 训练循环、损失、优化器、评估指标、画图、数据加载/增强流水线、checkpoint 保存、argparse 入口 | 训练链路自己实现（`trainer.py`）；原文件里的这些代码在接入后无用武之地 |
| 为 RML2016/2018 写死的输出类别数、窗口长度 | 类别数与窗口由数据集契约传入（`num_classes` / `samples`），写死会直接报错或训出错误结构 |

**六个来源文件的实测对照**

| 文件 | 路径 | 必须保留的核心 | 实测不需要（本仓库无调用） |
| --- | --- | --- | --- |
| [custom/MCLDNN.py](../../../custom/MCLDNN.py) | A 直接包装 | `CausalConv1d`、`MCLDNN` | `__main__` 演示块 |
| [custom/PETCGDNN.py](../../../custom/PETCGDNN.py) | A 直接包装 | `PET`、`PETCGDNN` | `__main__` 计时块（`time` / `device`） |
| [custom/CV_TRN.py](../../../custom/CV_TRN.py) | B 维护副本 | `RelativePositionBias` … `CV_TRN`（含 `_init_weights`、`random_phase_offset`） | `get_config_rml2016/2018`、`numParams`、`__main__` 块 |
| [custom/POET.py](../../../custom/POET.py) | B 维护副本 | `InstanceAGC` … `POET`（14 个模块类） | `get_poet_config_rml2016/2018`、`numParams`、`__main__` 块 |
| [custom/AMC_Net.py](../../../custom/AMC_Net.py) | B 维护副本 | `Conv_Block`、`MultiScaleModule`、`TinyMLP`、`AdaCorrModule`、`FeaFusionModule`、`AMC_Net` | `get_amc_net_config_rml2016`、`__main__` 块 |
| [custom/ASCS.py](../../../custom/ASCS.py) | B 维护副本 | `get_same_padding`、`SamePaddingMaxPool2d` … `ClassifierNet`、`stft_sig`、`ASCS` | 顶层 `device = torch.device(...)`、`__main__` 块 |

> 上表"不需要"的项在**路径 A** 下几乎无副作用（`__main__` 块不执行、函数定义不调用），
> 所以直接包装时**不必**为了好看去改原文件；**路径 B** 的副本按规范应删干净
> （现状：`cv_trn.py` / `poet.py` 副本留了 `get_config_*` / `numParams`，无害但确实无调用）。

**自定义文件最小骨架**

```python
"""<论文/来源>；只保留模型定义（供 AMC 包装层按文件路径 import）。"""

import torch
import torch.nn as nn


class _Helper(nn.Module):
    ...                                    # 模型递归引用到的子模块/函数都留着


class ExampleNet(nn.Module):
    """输入 (B, 2, N)、输出 (B, C) logits。"""

    def __init__(self, num_classes=11, hidden_size=128):
        ...                                # 默认值写什么都行：包装层总是显式传参

    def forward(self, x):
        ...
        return logits                      # 不要 softmax：导出层会追加

# 不需要：__main__ 演示块、get_config_* / numParams、顶层 device、
# 训练/评估/画图/数据加载/checkpoint/argparse
```

## 3. 实现模块：唯一入口 `build()`

`training/amc_models/<id>.py` 必须提供（`training/amc_models/build_model()` 唯一调用的符号）：

```python
def build(*, num_classes: int, samples: int | None, params: dict) -> nn.Module
```

三条硬性约定：

1. **布局与输出**：返回的模型**直接接收 `(B, 2, N)`**（`iq_waveform_v1`：单位 RMS 的 I/Q 窗口），
   **输出 `(B, C)` logits**；softmax 由导出层追加（`SoftmaxWrapper`），训练、验证评分、导出、预检
   用的是同一个对象，不存在"训练-推理口径分叉"。
   原实现吃 `(B, N, 2)` 时用目录条目的 `input_layout="iq_channels_last_v1"` + `adapt()` 包一层。
2. **参数的唯一事实源是目录条目**：`params` 已经过 `validate_params`（未知键、范围、长度、奇偶、整除、
   交叉约束都在构建前拦下）并**填好默认值**；实现里只读 `params["名字"]`，不得再写默认值或第二套校验。
3. **结构依赖窗口长度时必须显式要 `samples`**：`if samples is None: raise ValueError(...)`，
   不要静默用 128 之类兜底（否则换窗口就训出错误结构）。
   `build_model()` 会从数据集契约把窗口长度传进来。

其它由注册表兜底的规则：`requires` 里的依赖缺失在**导入实现模块之前**报错并给出 pip 安装提示；
`exportable=False` 的模型会被服务层拒绝作为训练任务（`training_jobs.iq_plan`）。

## 4. 目录条目（spec）：字段与登记

目录条目放 `src/signal_analysis/algorithms/amc/ai_model/<id>.py`，**纯元数据、不许 import torch**
（桌面应用冻结包不含 torch；`tests/analysis/test_amc_model_catalog.py` 强制源码级 + 运行期零 torch）。

| 字段 | 说明 |
| --- | --- |
| `id` | 模型 ID（= `--arch` 取值）：实现侧由测试强制一致（`spec.implementation == "amc_models.<id>"` 且 `training/amc_models/<id>.py` 存在）；目录条目文件名按惯例同样取 `<id>.py` |
| `title` / `summary` / `layers` | 界面「模型架构」只读区与日志展示 |
| `implementation` | 必须写成 `amc_models.<id>` |
| `model_revision` | 结构或前向语义变更时 +1（历史实验口径判定） |
| `params` | `ParamSpec` 元组，默认值唯一事实源 |
| `samples` | `any` / `exact:<N>` / `min:<N>`，与数据集窗口校验 |
| `input_layout` | `iq_channels_first_v1`（原样）或 `iq_channels_last_v1`（`adapt()` 转置） |
| `requires` | 依赖名元组，训练环境缺失时给出安装提示 |
| `exportable` | 只有通过 ONNX 导出与验收后才可为 `True` |
| `notes` | 来源、与原文差异、窗口/耗时提醒 |
| `cross_validate` | 可选：参数间交叉约束（单个参数表达不了的规则） |

`ParamSpec` 的 `kind` 与可用约束：

| `kind` | 取值形态 | 约束字段 |
| --- | --- | --- |
| `int` | 整数 | `minimum` / `maximum` / `exclusive_minimum` / `exclusive_maximum` / `odd` |
| `float` | 浮点 | 同上 |
| `int-list` | 非空整数列表 | `length` / `minimum_length` / `maximum_length` / `element_minimum` / `element_odd` |
| `bool` | 布尔 | — |
| `choice` | 枚举 | `choices` |

通用字段：`help`（界面 tooltip 与 CLI 帮助）、`label`（显示名，留空用 `name`）、
`divides`（本参数必须整除另一个整数参数，如注意力头数整除隐层宽度）。

**登记（唯一需要动"框架代码"的地方）**

1. 在 [ai_model/__init__.py](../../../src/signal_analysis/algorithms/amc/ai_model/__init__.py) 里 import `SPEC as _X_SPEC` 并加进 `SPECS` 元组（元组顺序即界面顺序）；
2. 模型增删或参数 schema 变更时把 `CATALOG_VERSION` +1（应用与训练源码目录按它握手，不改会让界面
   以为训练环境还是旧目录）；
3. 其余**自动生效**：`--arch` 的 choices、界面模型下拉/参数选项卡/窗口预检（`amc_training_page.py`
   读 `available_models()` / `catalog_json()`）、任务启动前校验（`training_jobs.py` 读 `model_spec()`）
   都从这一份目录取，不需要改界面或服务层代码。

## 5. 两条路径：直接包装 or 维护副本

| 路径 | 适用条件（同时满足） | 训练侧文件形态 | 例子 |
| --- | --- | --- | --- |
| **A 直接包装** | ① 只用能写进 `requires` 的库；② 文件自包含（无兄弟导入）；③ `forward` 吃 `(B,2,N)` 或只差一次转置；④ 不改内部实现就能导出 ONNX | 20 行以内的 `build()`：加载文件 → 传参 →（必要时）`adapt()` | `mcldnn.py`、`petcgdnn.py` |
| **B 维护副本** | 命中任一条：timm 等不想引入的导入、设备写死、复数 FFT/STFT（opset 17 导出失败）、结构参数写死、顶层副作用 | 拷贝模型部分 + 必要改写 + `build()`；docstring **逐条列出"相对原文件的差异"** | `cv_trn.py`（timm→`_nn_utils`、构造期不写 CUDA）、`poet.py`（timm→`_nn_utils`）、`amc_net.py`（复数 FFT→实数 DFT 表、`deepcopy`→`clone`）、`ascs.py`（STFT 实数化、`input_size` 按窗口探测） |

路径 B 的纪律：**只改"跑不通/导出不了"的那几处**，模型结构与计算口径不变，并在 docstring 里量化
数值差异（现状：`amc_net` 实数化与 FFT 差 1e-5～1e-4，`ascs` 与 `torch.stft` 差约 2e-6，均为相对误差）。

## 6. 接入清单（5 步）与验证命令

| 步骤 | 产物 |
| --- | --- |
| 1 | `custom/<文件>.py`：原文件（已有则跳过，**不改**）；确认满足 §1 的四条约束 |
| 2 | `training/amc_models/<id>.py`：路径 A 包装或路径 B 副本 + `build()`（模板见 §7） |
| 3 | `src/signal_analysis/algorithms/amc/ai_model/<id>.py`：目录条目（模板见 §7） |
| 4 | `ai_model/__init__.py` 登记 `SPECS`，必要时 `CATALOG_VERSION` +1 |
| 5 | 跑下面四条命令，全绿才算接入完成 |

```bash
# 1) 目录一致性：ID ↔ 模块 ↔ 文件一一对应、目录零 torch、参数规则（秒级，先跑这个）
python -m pytest tests/analysis/test_amc_model_catalog.py -q

# 2) 结构/构建/导出/数值一致性（需要 torch / onnx / onnxruntime，即 [train]）
python -m pytest tests/analysis/test_iq_training_tools.py -q

# 3) 训练冒烟：类别数与窗口由数据集契约传入，不必手工指定
python training/train_iq.py --data training/data/iq --arch <id> --epochs 1 --seed 7

# 4) 验收：清单 + 图形状 + 四个确定性场景端到端
python training/verify_iq.py --manifest <onnx 目录>/iq_manifest.json --data training/data/iq
```

界面侧核对（可选）：在训练环境执行目录查询，确认新模型与依赖状态可见。

```bash
python -c "import sys; sys.path.insert(0,'src'); sys.path.insert(0,'training'); \
from signal_analysis.algorithms.amc.ai_model import catalog_json; import amc_models, json; \
payload = json.loads(catalog_json()); \
print([m['id'] for m in payload['models']]); \
print({m['id']: list(amc_models.missing_requirements(m['id'])) for m in payload['models']})"
```

## 7. 模板

| 模板 | 用法 |
| --- | --- |
| [training/amc_models/_template.py](../../../training/amc_models/_template.py) | 复制成 `training/amc_models/<id>.py`（路径 A 的包装示例写在 `build()` 里，路径 B 的说明在 docstring） |
| [ai_model/_template.py](../../../src/signal_analysis/algorithms/amc/ai_model/_template.py) | 复制成 `ai_model/<id>.py`，覆盖 `id` / `title` / `implementation` / `layers` / `params` 等 |
| 本文 §2「自定义文件最小骨架」 | `custom/` 侧该保留什么 |

两个 `_template.py` 都以 `_` 开头：**不会被加载、不在 `SPECS` 里**，也不会被一致性测试当作模型
（`training/amc_models` 的扫描跳过 `_` 前缀、`trainer.py`、`checkpoint.py`）。

## 8. 排查表

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `FileNotFoundError: 缺少第三方模型文件` | 文件名与 `load_custom_module("X")` 不一致 | 名字必须对应 `custom/X.py` |
| 构建模型时 `ModuleNotFoundError`（如 `timm`） | 原文件顶部导入未在 `requires` 里声明 | 路径 B 换成本地等价实现（`_nn_utils.py`），或写进 `requires` 并在训练环境安装 |
| ONNX 导出失败（复数 / FFT / STFT 相关） | opset 17 的现有导出器不支持复数算子 | 实数化（DFT 表作 buffer，参照 `amc_net.py` / `ascs.py`），或先标 `exportable=False` 停在离线实验 |
| 训练时 `mat1 and mat2 shapes cannot be multiplied` | 结构把窗口长度写死（如 ASCS 原实现 `input_size=72` 只对 128 点成立） | 构造期按窗口探测，或由 `samples` 传入（`build()` 里显式要求 `samples`） |
| `模型 X 固定要求 128 点窗口` / `要求窗口不少于 N 点` | `spec.samples` 与数据集窗口不符 | 调整 `samples` 约束或换数据集窗口 |
| 形状对但精度明显不对 | `input_layout` 与实际不符（I/Q 与时间维错位） | 让实现直接吃 `(B,2,N)`，或声明 `iq_channels_last_v1` 交给 `adapt()` |
| 界面看不到新模型 / 显示"未验证" | 没登记 `SPECS`、`CATALOG_VERSION` 未 +1、训练环境查询失败（依赖缺失） | 按 §4 登记；用 §6 的查询命令核对 |
| `模型 X 不认识参数：...` | 参数名与目录条目不一致 | 参数名以 `ParamSpec.name` 为准，实现只读 `params[name]` |
| 目录测试报 "必须保持零 torch" | 在 `ai_model/*.py` 里写了 `import torch` | 元数据与实现分家：torch 只能出现在 `training/amc_models/` |

## 9. 相关文件索引

| 文件 | 作用 |
| --- | --- |
| [custom/](../../../custom) | 第三方原文件（只读来源，接入时不改） |
| [training/amc_models/_custom.py](../../../training/amc_models/_custom.py) | 按文件路径加载 `custom/<名>.py`（`amc_custom_*`，不注入 `sys.path`） |
| [training/amc_models/_adapters.py](../../../training/amc_models/_adapters.py) | `adapt()` / `ChannelsLastAdapter` / 导出用 `SoftmaxWrapper` |
| [training/amc_models/_nn_utils.py](../../../training/amc_models/_nn_utils.py) | timm 两个工具的本地等价实现（`trunc_normal_` / `DropPath`） |
| [training/amc_models/__init__.py](../../../training/amc_models/__init__.py) | `build_model()` / `missing_requirements()` / `export_onnx()` |
| [ai_model/base.py](../../../src/signal_analysis/algorithms/amc/ai_model/base.py) | `ModelSpec` / `ParamSpec` / 参数与窗口校验 |
| [ai_model/__init__.py](../../../src/signal_analysis/algorithms/amc/ai_model/__init__.py) | `SPECS` 登记与 `CATALOG_VERSION` |
| [test_amc_model_catalog.py](../../../tests/analysis/test_amc_model_catalog.py) | 目录约束回归（ID 对应、零 torch、参数规则） |
| [test_iq_training_tools.py](../../../tests/analysis/test_iq_training_tools.py) | 结构/构建/导出/数值一致性回归 |
