# SimuSignal 两项目工作区

本仓库按方案总览实现 **两个独立应用、一个共用基础库**。

「模型训练」页面支持时频图检测框标注、RT-DETR/YOLO26s 检测训练及 CNN/TCN IQ 分类训练。
操作和环境配置见 [模型训练工作台](docs/模型训练工作台.md)。

| 项目 | 业务包 | 桌面入口 | 默认数据目录 |
| --- | --- | --- | --- |
| 电磁信号分析 | `src/signal_analysis` | `apps/analysis_desktop/main.py` | `workspace_data/analysis` |
| 通信仿真实验 | `src/communication_sim` | `apps/simulation_desktop/main.py` | `workspace_data/simulation` |
| 共用基础库 | `src/common` | 无业务启动入口 | 无共享业务数据库 |

## 开发安装与分别启动

在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
# 核心依赖 + gui + dev 
python -m pip install -e '.[gui,dev]' 
# 完整训练环境（含固定版本 TorchSig 数据生成及 ONNX 验收）
python -m pip install -e '.[gui,dev,ml,train]'
python -m signal_analysis gui
python -m communication_sim gui
```

Windows 用 `python -m venv .venv` 创建环境，PowerShell 中执行 `.venv\Scripts\Activate.ps1` 后使用同样的 Python 命令。根项目只用于统一开发，两套正式发布清单在各自的 `apps` 目录。

安装后也可分别运行 `signal-analysis gui`、`communication-sim gui`，或执行两个 `apps/.../main.py`。

## 电磁信号分析项目

支持一维数值 NPY、无表头 CSV（一列实数或两列 I,Q）、交织 IQ 二进制（.bin/.raw/.iq，显式指定 int16/float32 与大小端）、通用统计、频谱/时频图与星座图/瀑布图、数据备注、运行历史、JSON/HTML 报告和原生复制示例插件。分析有两种方式：一键概览整段数据，或按可调速度实时播放。波形与频谱使用固定显示范围（幅度、时窗、频宽、动态范围），可在界面调整，播放时保持不变。数字/模拟判定为启发式，可在界面手动纠正；实数数据默认只显示非负频率，检测与跳频页同样提供“自动/双边/仅正频率”选择。实时播放的瀑布图为固定时长的滚动窗口，时间窗可选 10 ms～20 s。

```bash
python -m signal_analysis demo --count 8192 --sample-rate 48000
python -m signal_analysis import data.npy --sample-rate 48000
python -m signal_analysis import iq.bin --sample-rate 48000 --binary-dtype int16 --endian little
python -m signal_analysis list
python -m signal_analysis analyze ASSET_ID --nfft 256
python -m signal_analysis export RUN_ID report.html
```

把 ID 替换为前一步输出。单文件限制为 512 MiB，最多 16,000,000 个采样点。采样率明确填写，不猜测未知 BIN 文件。

### 信号 IQ 生成（测试信号源）

界面"IQ 信号生成"页，用于生成测试检测、参数估计与调制识别算法的 IQ 基带信号：支持 AM、FM、SSB、2ASK、QPSK、16QAM、64QAM 与跳频信号（FH-2FSK 模拟遥控链路、FH-OFDM 模拟图传链路）。IQ 为复基带记录，不设置载频，"频点"指基带频率偏移。可设置信号持续时间、采样率、随机种子、每信号功率（dBFS）、目标带宽、频点、跳速、符号速率等；自动按调制样式与目标带宽推导消息带宽、频偏、滚降成形等参数。支持一次 IQ 中包含最多 16 种信号并独立设置参数，以及带限背景噪声。

提供**数学双音演示**按钮：不建模通信链路，直接按页内"采样率 × 持续时间"生成一对可复现的复基带双音资产（源类型 `generated:tones_v1`，含少量噪声），点数与页内"预计 N 个复采样"一致；演示不使用信号列表与背景噪声设置，生成后在左侧资产列表选中，可切到"数据分析"页查看。

噪声为**带内信噪比**：噪声功率在设定的噪声带宽内均匀分布（功率谱密度 $`N0 = P_{noise} / B_{noise} `$ ），信号的带内噪声取 $`N0 × 该信号实际占用带宽`$，因此某信号的带内 $SNR = 其平均功率 ÷（`N0 × B_{actual}`）$。界面填写的 `snr_db` 指最强的那个信号（按实测平均功率判定）的带内 SNR，其余信号按各自实际带宽与重叠频带折算，结果记录在摘要的 `signals[].snr_inband_db` 中；若噪声带宽未覆盖最强信号的占用频带会直接报错，避免实测 SNR 与填写值不符。纯噪声（无信号）时改为填写噪声的绝对总功率（dBFS）。摘要同时给出 `noise.power_dbfs_per_hz`、`noise.snr_definition`（当前为 `inband_snr_v1`）与 `noise.snr_reference_index` 以便复核。可导出 NPY、CSV（两列 I,Q）或交织 IQ 二进制（int16/float32、大小端可选）。

```bash
python -m signal_analysis --workspace /tmp/iqws generate spec.json
```

`spec.json` 形如 `{"sample_rate": 1e6, "duration": 0.2, "seed": 0, "noise": {"enabled": true, "bandwidth": 1e6, "snr_db": 20}, "signals": [{"mode": "qpsk", "offset": 100000, "power_dbfs": -10, "bandwidth": 200000}], "name": "QPSK测试", "export": {"format": "iq16", "endian": "little"}}`，生成结果保存为工作目录内数据资产并可同时导出到 `exports/`。

SigMF 双文件读写使用正式依赖 `sigmf==1.11.1`：在生成页选择“SigMF 双文件”，或把上述规格中的 `export` 改为 `{"format": "sigmf"}`。输出为同名 `.sigmf-meta`（元数据）与 `.sigmf-data`（小端 complex float32 IQ），两者须一起保存和分发。已有环境更新依赖可运行 `python -m pip install -e ".[gui]"`。

GUI 导入可选择 `.sigmf-meta` 或 `.sigmf-data`，自动读取采样率。CLI 示例：

```bash
python -m signal_analysis import path/to/recording.sigmf-meta
```

SigMF 不必指定 `--sample-rate`；显式指定时必须与文件一致。其他格式仍需该参数。导入支持单通道 `cf32/cf64/ci16` 的大小端连续 IQ 双文件，内部转换为 complex64；暂不支持 `.sigmf` 归档、多通道或外部数据引用。导入的原始元数据及生成摘要关联资产保存；生成摘要也写入导出文件的 `core:description`，不虚构射频载频。详细限制见设计文档 §4.1。

### 信号检测与参数估计（能量检测基线）

界面“信号检测”页，或 CLI `detect`：STFT 时频图 → 噪声本底估计 → 自适应门限 → 形态学闭运算合并 → 逐目标给出中心频率、占用带宽、时间范围、功率与带内 SNR；同一载波上的跳频信道按会话合并为一条记录。结果是冻结的 `detect_result_v1` 结构，可用 `evaluation.evaluate_detections` 与 IQ 生成摘要中的真值逐项评分（召回、精确率、中心频率 MAE、带宽相对误差）。

```bash
python -m signal_analysis detect ASSET_ID --nfft 512 --threshold-db 3 --max-detections 32
```

界面上时频图的横轴是频率、纵轴是时间，与平均功率谱密度、检测框共用同一频率范围，可用“频率显示”选择：`自动`（默认）在频谱关于 0 Hz 镜像时（实数记录）只画非负频率，复数 IQ 保留双边；`双边` 固定 −fs/2～+fs/2；`仅正频率` 固定 0～fs/2。实数记录不读原始样本、只按 PSD 的镜像关系判断，因此不依赖 `detect_result_v1` 契约里的额外字段。检测框按**数据坐标**绘制，若某次检出本身跨过 0 Hz（例如硬门控信号的开关瞬态展宽），红框会真实地压在负半轴上，这不是绘图错误。

### 跳频逐跳参数估计（`detect-hops`）

界面“跳频参数”页，或 CLI `detect-hops`：同一份 STFT 时频图上做逐帧门限游程，再以时频脊线跟踪把属于同一部发射机的跳接成轨道，对每跳用细网格复算占用带宽与带内 SNR，最后给出跳频点、跳时刻、驻留时间、跳速、占空比与会话分组。结果是独立契约 `fh_hops_v1`（算法 `hop_track_v1`）。

```bash
python -m signal_analysis detect-hops ASSET_ID --nfft 512 --threshold-db 6
python -m signal_analysis detect-hops ASSET_ID --max-hops 64 --no-sessions
```

逐跳页的时频图同样按“横轴频率、纵轴时间”绘制，并用与检测页一致的“频率显示”选择频率范围：平均功率谱密度、时频图与逐跳框始终共用同一范围，同色的逐跳框属于同一会话。

驻留时间与跳速存在**可分辨下限**（帧间距与最小驻留共同决定），不可分辨时结果会显式给出 `resolvable=false` 与 `reason`，而不是静默给出乐观数字。逐跳真值只在生成器产出且记录了生成摘要时存在（`hop_truth`），否则报告标“不适用”。

同一契约还有 **AI 逐跳通路** `ml-detect-hops`（算法标识 `ml_detect_hops:<模型 id>`，需清单声明 `label_semantics=per_hop_v1`）：网络在时频图上直接给出逐跳候选框，框之后的驻留/功率/带宽/SNR 重测与会话归并**复用同一套逐跳逻辑**，因此 AI 与传统逐跳的物理量可直接并排比较，报告里另外给出 AI 逐跳/传统逐跳/会话级三列指标；`--no-traditional` 可关掉传统逐跳基线，`--no-sessions` 时连会话级基线也不跑。每条 AI 逐跳明细多一个 `model_confidence`（网络分数，**不是概率、未标定**）供追溯。

### AI 检测（可选，ONNX Runtime）

```bash
python -m pip install -e ".[ml]"           # onnxruntime；未安装时界面 AI 入口自动禁用
python -m signal_analysis ml-manifest model.onnx detector.json --image-size 1024 --nfft 512 \
    --framework yolox --license Apache-2.0 --dataset "本项目合成数据集"
python -m signal_analysis ml-detect ASSET_ID detector.json --score-threshold 0.25
```

模型清单声明输入图像契约（尺寸、STFT 点数、动态范围、归一化方式）与输出框格式，推理时强制与清单一致而不是静默改变输入；`ml-detect` 默认同时跑一遍能量检测基线，两条路径的指标可直接对照。逐跳模型用 `ml-detect-hops`（清单需 `--label-semantics per_hop_v1`）：

```bash
python -m signal_analysis ml-manifest hops.onnx hops.json --image-size 1024 --nfft 512 \
    --label-semantics per_hop_v1 --license Apache-2.0
# --workspace 是全局选项，必须放在子命令之前；清单是位置参数
python -m signal_analysis --workspace workspace_data/analysis ml-detect-hops ASSET_ID hops.json
```

会话级清单与逐跳清单不能互换：用错粒度会在推理前直接报错，而不是静默给出“一跳等于整段传输”的假结果。训练、导出与验收脚本见 [`training/`](training/README.md)（该目录不随 wheel 分发，训练依赖 `.[train]`；逐跳模型的数据集与训练参数见其 §2.1 / §3.1）。

### 调制识别（A09 六类）

界面“调制识别”页，或 CLI `amc-classify`：在检测结果给出的频带（或手工填写的中心频率/占用带宽）上把信号搬到零频、抽取到与带宽匹配的分析率，提取 34 维冻结特征（包络统计、瞬时频率/相位、高阶累积量、谱对称性等），再由模型输出 A09 六类字典 `FM / SSB / 2ASK / QPSK / 16QAM / 64QAM` 的分数与置信度。

```bash
python -m signal_analysis amc-classify ASSET_ID --offset-hz 40000 --bandwidth-hz 30000
python -m signal_analysis amc-classify ASSET_ID --model model.json          # 自带模型
python -m signal_analysis amc-manifest model.json amc.json --id dut --version 1.0.0
```

随包分发一个**线性基线**模型（`amc-linear-default`，34 维特征上的多项逻辑回归，含温度标定），无需 `.[ml]` 即可离线使用；仓库内实测（800 次/类合成场景，验证集）准确率 0.8250、macro F1 0.8246，带内 SNR ≥ 10 dB 时 ≥ 0.97，主要误差来自 10 dB 以下 16QAM 与 64QAM 之间的混淆。结果结构为冻结的 `amc_classify_v1`，同时给出传统启发式对照（数字/模拟、恒包络/非恒包络），字段 `pending` 明确列出**尚未确认项**：识别准确率的合格门限尚未确定，因此只报原始指标而不做通过/不通过判定。

生成数据会带上生成器真值并逐条计命中（`truth_hit`）；导入数据或原生插件产出没有真值，此时接口返回 `不适用` 并**保留失败样本计数**，不静默丢弃。`am` 与跳频样式不在 A09 六类字典内，按“不适用”计入统计而不算识别错误。训练、导出与验收脚本见 [`training/README.md`](training/README.md) §6。

#### 原始 IQ 通路（`amc_iq_classify_v1`）

同一页面改选一个**原始 IQ 模型清单**（`input.contract = iq_waveform_v1`）即可走第二条通路：不提取 34 维特征，
而是把“搬到零频 → 抽取到与带宽匹配的分析率 → 取中一段定长窗口 → 单位 RMS 归一化”后的 `(2, N)` 复数波形
直接交给 CNN/TCN 分类器。前端抽取口径与特征通路**共用同一份实现**（抽取比 8.0、低通抽头 65），
所以同一带宽下两条通路的分析带宽与滤波器完全一致；差别只在“交给判别器的东西”。

```bash
python -m signal_analysis amc-iq-classify ASSET_ID --model training/runs/iq/iq_manifest.json
python -m signal_analysis amc-iq-classify ASSET_ID --model iq_manifest.json \
    --offset-hz 0 --bandwidth-hz 30000 --threads 4
python -m signal_analysis amc-iq-manifest onnx/iq.onnx iq_manifest.json \
    --id iq-cnn-v1 --version 0.1.0 --samples 1024
```

类别字典由清单声明（`a09` 六类，或 `custom` 自定义且不超过 64 类），窗口长度必须与清单 `input.samples` 一致：
短于 64 点或长于 65536 点直接报错，**不补零、不截断**——短窗口补零会让“看起来像噪声”的输入也能出高置信度结果。
结果是冻结的 `amc_iq_classify_v1`，含波形摘要（窗口起点、RMS、峰值因数、`snr_estimate_db`）、各类分数与可信度提示；
与特征通路一样，`pending` 明确列出**识别准确率的合格门限尚未确认**，所以只报原始指标、不做通过判定。
这条通路没有 34 维特征向量，因此不提供传统启发式对照行（对照只对确定性特征有意义）。
数据集的构建（可混入 TorchSig 补充数据）、训练与验收见 [`training/README.md`](training/README.md) §7。

### 算法对比与离线报告

界面“算法对比”页把同一对象在不同路径上的取值并排列出（环节 / 对象 / 指标 / 取值），便于直接看出差异而不是各看各的结果；某一侧没有产出时显示 `--`，不填 0、不猜测。

导出的 HTML 报告（`export RUN_ID report.html`，CLI 与冻结产物同接口）除页尾完整原始 JSON 外，按结果类型附指标表：检测与 AI 检测给出“检测结果 / 传统基线”两列的真实目标数、匹配、漏警、虚警、精确率、召回、F1、中心频率 MAE、带宽相对误差与带内信噪比 MAE；调制识别给出预测类别、置信度、置信度差、是否可靠、判定说明，以及真值命中与真值带内信噪比，并把 `pending` 中**尚未确认项**单列。真值不可用按“不适用”计数，缺失字段显示 `--`。所有文本经 HTML 转义，报告为自包含离线文件。

### 工作区数据管理

界面“数据管理”页（在“跳频参数”之后、“运行记录”之前）盘点工作目录的占用并给出安全清理：

* 按 `assets/`、`runs/`、`jobs/`、`exports/`、`catalog.sqlite3`、页面设置分别统计字节与文件数，另可把 `training/data`、`training/runs` 或模型目录加为**额外目录**（只统计容量，不建索引）；
* 核对索引与磁盘是否一致：未入库文件、入库但文件缺失、未入库运行目录、原子写 `.tmp` 残留、源资产已删除；
* 按“任务保留天数”列出可清理的**过期任务**与**无主文件**，需先“预览清理”再逐条勾选才能删除，删除前二次确认；
  已入库的信号资产、运行目录与 `exports/` **永不删除**，运行中任务一律跳过；
* **扫描与导出都是只读的**，不写入运行记录（不会让工作目录越扫越大）；每次清理往 `maintenance.log` 追加一行审计记录。

统计口径、报告字段、清理规则与已知局限见 [`docs/数据管理页面.md`](docs/数据管理页面.md)。

![独立分析界面](docs/images/analysis-workbench.png)

## 通信仿真项目

提供独立的通用消息中继队列实验、场景参数、事件回放和报告，不显示信号资产侧栏或原生信号插件按钮。

```bash
python -m communication_sim simulate --messages 12 --duration 3
python -m communication_sim list
python -m communication_sim export RUN_ID simulation.html
```

当前模型为 A→中继队列→B 的通用离散事件演示。专用星地协议、跳频、同步、TDMA、编码和语音模型尚未实现。

![独立仿真界面](docs/images/simulation-workbench.png)

## 数据目录与旧数据

两个命令均支持在子命令前指定 `--workspace`，例如：

```bash
python -m signal_analysis --workspace /tmp/analysis-demo demo
python -m communication_sim --workspace /tmp/simulation-demo simulate
```

每个数据目录保存 `project.json` 标识所属项目，仿真数据库不创建分析资产表。

分析目录里另有 `maintenance.json`（数据管理页的额外目录与任务保留天数，默认 30 天，上限 3650 天）与
`maintenance.log`（清理审计 JSONL），均可在“数据管理”页里维护。

仓库根目录下的 `workspace_data/{assets,runs,jobs,catalog.sqlite3}` 是**没有 `project.json` 的旧版布局残留**，
两个项目都不会读取或清理它；“数据管理”页只统计不删除，确认无用后请手动删除。

## 原生插件（分析项目）

```bash
cmake -S examples/native_plugin -B /tmp/simusignal-native-demo
cmake --build /tmp/simusignal-native-demo --config Release
python -m signal_analysis plugin-manifest /tmp/simusignal-native-demo/libdemo_plugin.so /tmp/simusignal-native-demo/plugin.json
python -m signal_analysis native ASSET_ID /tmp/simusignal-native-demo/plugin.json
```

也可在分析界面选择插件清单。Windows 编译步骤见[原生示例说明](examples/native_plugin/README.md)。当前为 `demo_copy_f32` 演示 ABI，正式生命周期 SDK 仍待开发。

## 按项目测试和基准

```bash
QT_QPA_PLATFORM=offscreen python -m pytest -q
python -m pytest tests/common -q
python -m pytest tests/analysis -q
python -m pytest tests/simulation -q
python benchmarks/run_smoke.py analysis
python benchmarks/run_smoke.py simulation
```

Windows PowerShell 先设置 `$env:QT_QPA_PLATFORM="offscreen"`。该设置**只用于自动化测试**：设过之后同一个会话里再启动图形界面不会显示窗口，需先执行 `Remove-Item Env:QT_QPA_PLATFORM` 或另开一个终端。缺少 GUI 依赖或原生编译器的跳过项不能视为通过。基准脚本只记录基础流程耗时，不代表合同性能验收。

## 独立构建与发布

分别构建两个 wheel，源码从总览约定的 `src` 目录收集，项目元数据读取各自 `apps/.../pyproject.toml`：

```bash
python scripts/build_wheels.py analysis --compile-core
python scripts/build_wheels.py simulation
```

分析 wheel 包含 `common + signal_analysis`，按 `packaging/numeric_core.json` 将七个职责模块及 `_numeric.py` 兼容入口编译为八个扩展并排除核心明文；仿真 wheel 包含 `common + communication_sim`，不依赖 SimPy 以外的业务计算库。GUI 依赖单独声明。建议在独立虚拟环境安装和升级各项目的 wheel；共用源码随各自 wheel 分发。

在 Linux CPython 3.12 上，使用生成的文件名分别构建目录式桌面程序：

```bash
python scripts/check_binary.py dist/wheels/signal_analysis-0.2.0-cp312-cp312-linux_x86_64.whl
python scripts/build_desktop.py analysis dist/wheels/signal_analysis-0.2.0-cp312-cp312-linux_x86_64.whl
python scripts/build_desktop.py simulation dist/wheels/communication_sim-0.2.0-py3-none-any.whl
```

输出分别为 `dist/SignalAnalysis`、`dist/CommunicationSim`，各自通过自身可执行文件启动 worker，外置插件不要求重打分析程序。Windows/麒麟需单独构建验证。当前仅分析数值模块完成 Cython 编译，仿真模块仍为 Python；源码包继续保留核心源码，不承诺二进制不可逆向。

## 设计与文件说明

- [方案总览与实际目录](docs/Python技术方案总览.md)
- [逐文件说明、依赖与测试](docs/基础工程实现与文件说明.md)
- [分析项目设计](docs/电磁信号分析识别系统_Python技术方案.md)
- [仿真项目设计](docs/某星通信仿真系统_Python技术方案.md)
- [二进制与原生插件设计](docs/核心模块二进制化与原生插件接口方案.md)

### 算法设计文档

信号检测与调制识别各自的传统算法、AI 算法独立成篇，内容包含算法思路、公式推导、流程图、
输入输出参数、设计局限与可改进方向（含改进所需文献）、当前参考文献：

| 任务 | 传统算法 | AI 算法 |
| --- | --- | --- |
| 信号检测（会话级，时频域） | [传统能量检测](docs/algorithms/信号检测_传统能量检测.md) | [AI 时频图检测](docs/algorithms/信号检测_AI时频图检测.md) |
| 信号检测（跳频逐跳） | [跳频逐跳参数估计](docs/algorithms/信号检测_跳频逐跳参数估计.md) | [AI 时频图检测](docs/algorithms/信号检测_AI时频图检测.md) §7（`ml-detect-hops`，需清单 `per_hop_v1`） |
| 调制识别（A09 六类，特征通路） | [传统特征与启发式判定](docs/algorithms/调制识别_传统特征与启发式判定.md) | [AI 特征学习](docs/algorithms/调制识别_AI特征学习.md) |
| 调制识别（原始 IQ 通路） | 无（无确定特征可对照） | [AI 特征学习](docs/algorithms/调制识别_AI特征学习.md) §7 |

五篇文档以**已实现代码**为准逐项核对公式与常量；其中会话级与逐跳两条检测路径共用同一份
STFT 与带内信噪比口径（`inband_snr_v1`），只是粒度不同（一条链路 vs 一跳）；
A09 调制识别的两条路径共用同一份
34 维确定性特征契约（`amc_feature_vector_v1`），因此“传统”与“AI”的差别只体现在最后的判别层，
两者可直接同口径比较。原始 IQ 通路是另一条输入契约（`iq_waveform_v1` / `amc_iq_classify_v1`）：
它不共用 34 维特征，而是让网络直接学波形，因此**不与特征通路做同口径对比**，只与自己的基线比。
识别准确率的合格门限仍为**待确认项**，文档只给原始指标与局限，不作通过判定。

数值实现按[模块拆分与回归说明](docs/数值算法模块拆分与回归说明.md)维护；算法函数使用统一中文说明，旧 `_numeric` 与 `ml.amc` 特征接口继续兼容。
