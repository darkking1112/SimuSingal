# AMC 识别训练

版本：0.1.1。更新日期：2026-10-06。

「AMC 识别训练」是信号分析桌面应用的第八个标签页，负责**调制识别（AMC）任务的数据标注与外部训练**：查看并修改当前集合内所选资产的 AMC 类别与目标参数，选好「训练集」「验证集」两个**信号集合**后启动 `training/` 下的 IQ 波形分类训练与验收，并在「实验与日志」里查看指标、加载验收通过的模型。它对应[00 技术方案](00电磁信号分析识别系统_Python技术方案.md) §4.4「数据评估与信号识别训练」的机器侧训练入口：产物是 ONNX 模型与 `iq_manifest.json` 清单，由「调制识别」页的 AI 入口加载推理（见[调制识别](05调制识别.md)）。

训练数据**直接取自信号集合的当前内容**：本页只做标注、配置、实验与日志，**不生成、不导出任何数据集，也不登记数据版本**。数据生成统一在「IQ 信号生成」页的「信号集合生成」子页完成（见[IQ 信号生成页面](02IQ信号生成页面.md)），环境准备与操作步骤见[模型训练工作台](模型训练工作台.md)，集合／目标／标注的数据模型见[数据库设计](数据库设计.md)。

页面与「信号检测训练」（固定任务 `detection`）是两个相邻的顶层页：本页固定任务为 `iq`，模型为 `cnn` 或 `tcn`，标注子页名为「信号标注」。**左栏门禁、跨页运行槽、进度归一化、进程树收尾与可复现性口径与检测页完全一致**，本页只写差异并链到[信号检测训练](07信号检测训练.md)。

数据流向：左侧「信号集合」→ 训练集／验证集 → 运行目录 `training/runs/<时间-id>/collection_data/`（`iq_dataset.json` + `iq_dataset.npz`，本次运行当时的内部输入快照）→ 外部训练与验收 → `model/` 下的 ONNX、清单与指标 → 「加载验收通过的模型」写入「调制识别」页。

---

## 1. 目标与边界

| 本页负责 | 本页不负责 |
| --- | --- |
| 信号标注：当前集合内左侧选中资产的 AMC 类别与目标参数 | 生成或修改集合内容（「IQ 信号生成 → 信号集合生成」） |
| 训练配置：训练集／验证集集合、CNN/TCN、启动前预检 | 导出训练集、构建 `iq_dataset` 目录、登记数据版本 |
| 外部训练进程的启动、进度、日志与取消 | 集合、标注集、类别字典的新建与归档 |
| 进程树收尾与运行槽释放 | 数据资产与集合的浏览、容量盘点与清理（「数据管理」页） |
| 历史实验浏览、指标（含按 SNR 分档）展示、加载模型到识别页 | 模型推理本身（「调制识别」页）与特征通路分类器的训练 |

三条硬约束：

1. **集合是唯一数据来源。** 训练集／验证集都必须是左侧存在且未归档的具体集合；页面没有数据集目录、数据来源字段，也没有生成或导出入口。
2. **追加式版本 + 同一事务。** 保存时先校验标签字段，再在同一事务里写目标参数版本与 AMC 标签：类别非法、时间范围越界等校验失败**一条记录都不留**，不存在「参数已写、标签失败」的半成品。
3. **只认已确认类别。** 训练输入只收 `class_state=known` 的目标；`unknown` 与 `out_of_taxonomy` 不进入训练，数量记在运行目录的卡片与跳过统计里。

---

## 2. 页面入口与共享上下文

### 2.1 标签页位置与门禁

本页是注册表第 8 页，排在「信号检测训练」之后。与检测页相同的部分（不再复述，见[信号检测训练](07信号检测训练.md) §2）：

- 左侧「信号集合」为**全部资产**或**零散资产**时本页标签置灰，正停在本页时自动跳回「信号导入」；标签置灰不替代配置校验。
- 左栏选集、按信号名称查找、「未标注检测信息／未标注AMC信息」过滤、分页与只读目标详情共用同一面板；过滤口径是「资产有目标但没有任何该任务的标签」，不做逐资产 N+1 查询。
- 训练集默认跟随左侧当前集合（页内可改选，两页各自保存），验证集默认不预设。
- 切集合、切资产、切目标、开始训练、关窗之前都先保存未保存的编辑，失败则回滚选择并留在原地。

本页差异：

| 项 | 取值 |
| --- | --- |
| 固定任务 / 模型 | `iq` / `cnn`、`tcn`（无任务切换下拉） |
| 注册名（owner）与标题 | 「AMC 识别训练」（历史记录与横幅都按它归属） |
| 标注子页 | 「信号标注」（检测页为「数据标注」） |
| 历史任务过滤 | 只列 `iq` 记录；`asset` 记录不属于本页 |
| 曲线标题 | AMC 训练 loss / IQ 验证准确率（0–1） |

### 2.2 跨页共享的运行槽

两页各持一个运行器、共用窗口级协调器：同一时间只允许一个训练运行，另一页启动会被拒绝并提示占用页，不产生第二个进程；运行期间本页冻结配置输入与标注表格，横幅、日志与取消仍可用。运行槽的获取、绑定任务、失败回滚与释放规则见[信号检测训练](07信号检测训练.md) §2.3、§6.5。

---

## 3. 「信号标注」子页

### 3.1 区块

| 区块 | 内容 |
| --- | --- |
| 标题栏 | 当前资产名 · 集合『<标注集名>』；「保存当前标注」按钮 |
| 目标表格 | 只列 `for_amc=1` 的目标（含会话与逐跳层级），每行一个目标；只读，选中行后到下方表单编辑 |
| 编辑表单 | 类别状态、调制类别、频率下限/上限、起始/结束采样点、带内 SNR、原始调制标注 |
| 状态栏 | 适用目标数、待标注数、是否已修改/已保存；保存结果与错误提示 |

目标表格的 7 列：

| 列 | 含义 |
| --- | --- |
| 目标 | 目标键与粒度，例如 `s0（session）`、`s0.h3（hop）` |
| 频率范围 | 当前参考参数版本的频率上下限（Hz）；未知显示「未提供」 |
| 时间范围 | 起止采样点；未知显示「未提供」 |
| SNR | 当前版本的 `snr_db`（dB）；未知显示「未提供」 |
| 调制 | 当前版本的调制标注；未知显示「未提供」 |
| 类别状态 | 字典内类别 / 未知调制 / 字典外类别；没有标签显示「未标注」 |
| 当前类别 | 最新 AMC 标签的类别名；没有标签显示「未标注」 |

### 3.2 可编辑字段与候选来源

| 字段 | 规则 |
| --- | --- |
| 类别状态 | 三态：`known`（字典内类别）、`unknown`（未知调制）、`out_of_taxonomy`（字典外类别） |
| 调制类别 | 可编辑下拉，**候选来自当前集合 AMC 标注集的类别字典**（`taxonomies.classes_json`），不是写死的 A09；`known` 时必须是字典内名称，`out_of_taxonomy` 时必须保留原始类名 |
| 频率下限 / 上限 | 数值（Hz），留空表示未提供；必须同时给出且上限大于下限，中心频率与带宽是这两个边界的派生值（数据层会拒绝与边界不一致的取值），页面不单独提供手填入口 |
| 起始 / 结束采样点 | 整数，留空表示未提供；必须同时给出，结束采样点要大于起始且不得超过资产采样点数 |
| 带内 SNR | 数值（dB），口径沿用参考参数的 `snr_definition`；留空表示未提供 |
| 原始调制标注 | 自由文本，例如 `QPSK`；用于保留外部或导入时的原始标注 |

表单里没有出现的字段（跳频标记、跳速、波形模式、符号率、功率、名义带宽、`params_json` 等）保持只读沿用，不在本页编辑。

### 3.3 保存：同一事务写版本与标签

点「保存当前标注」调用 `append_amc_annotation`：

1. 先校验标签字段（类别状态枚举、`known` 的类名在字典内、`out_of_taxonomy` 必须带原始类名、时间范围等），任何一项不通过直接抛错；
2. 校验通过后在**同一个连接/事务**里写目标参数版本（`source="manual"`）与该版本上的 AMC 标签（`source="manual"`）。

因此「非法保存」不会留下任何参数版本或标签。参数版本是**完整快照**，表单未暴露的字段用 `carryover_fields` 从上一版本沿用（`params_json` 会从库里的 JSON 文本解码回对象，避免二次转义）；表单编辑的 6 个字段显式排除在沿用之外。

保存成功后刷新表格与状态栏（消息「已保存 · 类别与参数已追加新版本」），并同步左栏只读目标详情。历史版本不修改、不删除；重新标注就是追加新版本。

### 3.4 只读摘要与缺标注集

「训练配置」里的 IQ 数据契约摘要（只读）显示：

- 窗口长度：1024 samples（`DEFAULT_IQ_SAMPLES`）；
- 通道：2；归一化：`unit_rms`；
- 类别顺序：`FM / SSB / 2ASK / QPSK / 16QAM / 64QAM`（A09 六类）或当前集合标注集的类别字典；
- 来源：未选集合时提示「未选择集合：类别字典显示 A09 默认值」；选中集合时提示「类别字典来自集合『<名>』的 AMC 标注集」。

**类别顺序即模型输出下标顺序**，所以摘要里的顺序不是装饰信息。集合缺少 AMC 标注集时，标注子页与摘要都只提示去「IQ 信号生成 → 信号集合生成」勾选 AMC 标注后重新生成，**不自动创建标注集**。本页不导入 Torch、不加载 NPZ，只读写工作区数据库。

### 3.5 切换、冻结与关窗

- **切换目标行**：先保存当前表单内容；保存失败把选中行恢复为原目标并保留表单，提示「未切换目标，修正后重新保存」，不静默丢弃修改。
- **切换资产 / 切换左侧集合**：同样先保存，失败回滚选择并留在原地（与检测页一致）。
- **运行期间**：配置输入与标注表格冻结；关窗前先保存，保存失败阻止关闭并可重试。

---

## 4. 「训练配置」与预检

### 4.1 字段

与检测页共用的字段：训练源码根目录、训练环境 Python、训练集（信号集合）、验证集（信号集合）、训练设备（cpu/cuda）、轮数（默认 20）、批大小（默认 8）、学习率（默认 0.001）、随机种子（默认 7）、**模型名称（留空=自动）**。本页差异：

| 字段 | 说明 |
| --- | --- |
| 模型 | 当前**模型目录**登记的模型 ID，默认 `cnn`；目录里每个模型一个文件（目录条目 + 实现），页面只读文件里的元数据。现有：`cnn`（IQCNN 步长卷积）、`tcn`（IQTCN 膨胀因果残差）、`mcldnn`（MCLDNN，I/Q 图像 + 因果卷积 + LSTM）、`petcgdnn`（PET 相位旋转 + CNN + GRU；窗口长度必须在 16 点以上） |
| 模型目录 | 「刷新模型列表」按钮 + 状态行：按「训练源码根目录」+「训练环境 Python」在训练环境里查询目录（含每个模型的依赖是否满足）。查询失败时保留应用内置目录，状态行注明"未与训练环境核对"；依赖或目录版本对不上时不允许起任务 |
| 模型架构（只读） | 由目录渲染的架构文本：结构分层、参数与默认值、窗口约束、依赖与说明；切换模型即刷新 |
| 模型名称（留空=自动） | 入库到模型库（`training/models/<名称>/`）时用的名称。留空按 `<模型类型>-<训练用途>-<UTC 时间>` 自动命名，本页用途固定为 `amc`（如 `cnn-amc-20261006T105213`）；手动填写时在预检里校验合法性（1–64 字符、不含 `/ \ : * ? " < > |`）与重名 |
| 初始权重 / RT-DETR 目录与 YAML / 时频图边长 | **不存在**：这些是检测专属字段 |
| IQ 数据契约摘要 | 只读（§3.4） |
| IQ 窗口长度 `samples` | **可改**，默认 1024（`DEFAULT_IQ_SAMPLES`）：写进配置后由 worker 按它装配数据快照，`train_iq.py` 再校验数据集声明的窗口与模型约束一致。模型声明 `exact:<N>` 时输入框自动对齐到唯一合法值，`min:<N>` 则在预检里拦截过短窗口；窗口长度必须与清单 `input.samples` 一致，训练、验收与推理只有一份实现 |
| 结构参数（可选） | 复选框「覆盖默认结构参数（高级）」**默认关闭**：关闭时配置里不出现 `model_params`，一律用模型目录声明的默认值（cnn 通道 32,64,128 / 核长 7；tcn 通道 64 / 核长 3；dropout 0.1）。勾选后才写入 `model_params`（JSON），由 `services/training_jobs.iq_tuning_args` 按目录参数 schema 校验后转成 `train_iq.py` 的 `--model-params`——页面预检与 worker 执行**共用同一份逻辑**。结构超参没有搜索证据，改动即视为新的实验口径，需要重新验收 |
| 权重衰减 / 早停轮数 | 公共训练参数（与模型结构无关），同样只在勾选高级后才写入配置，走 `--weight-decay` / `--patience` |
| 学习率调度 / 存优判据 / checkpoint | 公共训练参数（勾选高级后写 `scheduler`、`monitor`、`save_checkpoint`）：`cosine`（默认）或 `plateau`（按验证损失降半）；`accuracy`（默认）或 `val_loss`（最低验证损失存优并早停）；勾选 checkpoint 时把最佳权重写到运行目录 `model/iq_checkpoint.pt`（带模型 ID、结构版本、结构参数、类别、窗口），ONNX 与清单不受影响 |

页内说明：「训练数据取自信号集合：选择训练集与验证集集合即可，页内不生成数据、不导出训练集。“信号标注”查看并修改当前集合内左侧选中资产的参数与调制类别。」

集合下拉框下方另有**输入提示行**：选择两个集合后显示「训练集 N 个样本（M 条资产），跳过 字典外类别 X、未标注 Y；验证集 …」——`build_rows` 为 AMC 统计被跳过的标签状态（`unlabeled` / `unknown` / `out_of_taxonomy`），提示行直接读同一份 `inputs_plan` 装配结果，避免"资产数不少、训练样本却少一截"的原因只能到运行目录里查。选择不可训练（如两集合相同）时该行显示拒绝原因。

### 4.2 预检清单

集合级检查与检测页**共用同一个函数**，因此文案与拒绝条件一致（完整表见[信号检测训练](07信号检测训练.md) §4.2）：任务与模型、训练 Python/worker、集合选择与存在性、缺 AMC 标注集、可训练样本、防泄漏（两集合 `origin_group_id` 重叠）。AMC 差异：

| 检查 | 拒绝文案 / 行为 |
| --- | --- |
| 任务与模型 | 「AMC 识别训练 只接受 iq 训练任务」；模型不在当前目录（默认 `cnn`/`tcn`）内时「请选择当前页面支持的模型」 |
| 类别字典一致 | 「训练集与验证集的类别字典不同，无法一起训练：请统一后重试」 |
| 类别数 | 「AMC 训练至少需要 2 个类别：请检查训练集的类别字典」（训练集标注集的类别数 < 2 时拒绝） |
| 已确认类别 | 字典外与未知标签**不阻止训练**：它们被跳过并计数，只有「一个可训练样本都没有」才拒绝 |
| 结构参数（勾选后） | `model_params` 按模型目录的参数 schema 校验（未知参数、长度、区间、奇数、整除等规则都声明在目录里）：cnn 需 3 个正整数通道、核长不小于 3 的奇数；tcn 通道 1～3 个、核长不小于 2；dropout ∈ [0,1)；权重衰减 ≥ 0、早停轮数 ≥ 1。非法值在起任务前报错，不创建实验目录 |
| 窗口长度 | 按模型目录的窗口约束（`any`/`exact:<N>`/`min:<N>`）校验：「模型 <id> 固定要求 <N> 点窗口，数据集是 <M> 点」等；页面同时在输入框上按 `exact` 约束对齐 |
| 模型依赖与目录版本 | 训练环境缺依赖时「训练环境缺少 <id> 的依赖：…」；配置里的 `catalog_version` 与训练源码目录不一致时「模型目录版本不一致（界面 X / 训练源码 Y）：请在「训练配置」里重新点「刷新模型列表」再开始训练」 |
| 快照为空 | 窗口生成全部失败时由 worker 报「没有可训练的 AMC 样本（需要已确认的类别，且频带内样本足够长）：<原因>（N 条）」 |
| 模型名称 | 名称不合法时给出名称规则报错；模型库里已有同名模型时报「模型名称已存在：<名称>，请换一个名称或留空自动命名」 |

预检失败不占槽、不创建实验目录，错误同时显示在配置页红字与实验页状态。

### 4.3 训练配置字段详解

本节逐项说明「训练配置」里每个输入**到底影响什么**。整份配置会**原样**写进运行目录的
`experiment.json`，事后可在实验页或文件里核对，不存在"页面填了 A、实际跑了 B"。
一次完整实跑的各字段取值与结果见[调制识别：AI 特征学习](algorithms/调制识别_AI特征学习.md) §8「项目实例」。

**共用字段**（控件与检测页同一组）

| 字段 | 默认值 | 作用 | 影响与建议 |
| --- | --- | --- | --- |
| 训练源码根目录 `repository` | 从本页代码位置向上回溯的仓库根（`parents[4]`） | worker 从哪里加载 `training/` 脚本 | 只决定"用哪份代码"，不改变数据；换目录前确认其中有 `training/train_iq.py`、`training/amc_models/`，且 `src/signal_analysis/algorithms/amc/ai_model/` 的模型目录与该实现匹配 |
| 训练环境 Python `python` | 源码运行 = 当前解释器；冻结（打包）版 = 空，需手选 | 用哪个解释器跑 worker | 必须已装 `torch`、`onnx`、`onnxruntime`；环境不对会在「校验环境与数据」阶段失败（不是训练中途失败），日志里有缺包提示 |
| 训练集 `train_collection_id` | 左栏当前选中的集合（随左栏切换同步） | 训练样本来源 | 实际样本数 = 集合内 `class_state=known` 的目标数（下拉框下方的提示行直接给出），不是"资产数"；页内改选会保持到下次左栏切换 |
| 验证集 `val_collection_id` | **空选**（刻意不预设） | 早停与最佳轮次选择；验收脚本报告的准确率 | **不是按比例从训练集切的**，而是另一个完整集合——所以报出来的验证准确率天生是**跨集合**指标；空选或与训练集相同都不能启动，两集合同源（`origin_group_id` 重叠）会被预检拒绝 |
| 训练设备 `device` | cpu | 计算设备 | cpu 慢但稳定、可复现；cuda 需要环境里是 CUDA 版 torch，且结果与 cpu 有细微差异 |
| 轮数 `epochs` | 20 | **上限**，不是"一定跑完" | 验证准确率连续 `patience` 轮不提升即早停，并回滚到**最佳轮**权重再导出；同时它还是余弦退火的周期，所以设得太小会双重截断，设大一些更稳（只多花时间） |
| 批大小 `batch` | 8 | 每步样本数 | 每轮步数 ≈ 样本数 ÷ 批大小；几千样本时 8 偏小、CPU 上慢，实例里用 64；内存吃紧再调小 |
| 学习率 `lr` | 0.001 | AdamW 初始学习率 | **最敏感的一项**：界面下限是 `1e-7`，按成 1e-7 会让 loss 卡在 1.78 几乎不降（历史实验踩过）；训练用同一学习率配合余弦退火在 `epochs` 内衰减到 0 |
| 随机种子 `seed` | 7 | 权重初始化 + 每轮打乱顺序 | 决定这次训练的可复现性（同种子同数据同参数结果一致）；换种子只应带来几个百分点抖动 |

**AMC 专属字段**

| 字段 | 默认值 | 作用 | 影响与建议 |
| --- | --- | --- | --- |
| 模型 `arch` | `cnn` | 网络结构 | 列表来自模型目录：`cnn` = 步长卷积堆叠，`tcn` = 膨胀因果残差；同一份数据上实测 cnn 更好，窗口很长时可试 tcn。「刷新模型列表」后训练目录里新增的模型会出现在这里，缺依赖的模型会标出原因 |
| IQ 数据契约（只读） | — | 回显当前集合的窗口长度/通道排布/类别字典 | 只是回显；真正的契约校验在 worker 与 `train_iq.py` 里（契约不符直接报错，不训练） |
| IQ 窗口长度 `samples` | 1024 | 送进网络的采样点数：决定数据快照的切片长度与模型输入长度 | 必须与清单 `input.samples` 一致——改窗口长度等于换契约，训练、验收、推理三处必须一起改；模型有窗口约束时在这里拦截（`exact` 约束自动对齐到唯一合法值） |
| 覆盖默认结构参数（高级） | **关闭** | 打开后才把结构参数写进配置 | 关闭时配置里**不出现** `model_params`，完全沿用模型目录声明的默认值；结构超参没有搜索证据，改动即新的实验口径，需重新验收 |

**可覆盖的参数（勾选「覆盖默认结构参数」后才生效）**

前三行是**模型结构参数**（约束由模型目录声明）；其余五行是**公共训练配置**（与模型无关，
由 `services/training_jobs.iq_tuning_args` 校验）：

| 字段 | 默认值 | 作用 | 约束（模型目录声明，`services/training_jobs.iq_tuning_args` 校验） |
| --- | --- | --- | --- |
| 卷积通道 `channels` | cnn `32,64,128`；tcn `64` | 各级特征宽度 | cnn 需 **3 个正整数**（如 `64,128,256`）；tcn 1～3 个正整数；留空 = 用目录默认 |
| 卷积核长 `kernel` | cnn 7；tcn 3 | 感受野 | cnn 需 **≥3 的奇数**；tcn **≥2**；留空 = 用目录默认 |
| Dropout `dropout` | 0.10 | 全连接前的丢弃率 | 取值 `[0, 0.9]`；小集合加大可抑过拟合，过大转为欠拟合 |
| 权重衰减 `weight_decay` | 1e-4 | AdamW 的 L2 强度 | `≥ 0`；0 = 不加正则 |
| 早停轮数 `patience` | 8 | 真值判据连续多少轮不改善就停 | `≥ 1`；调大只会多跑几轮，通常收益很小 |
| 学习率调度 `scheduler` | cosine | 学习率怎么降 | `cosine` = 以轮数为周期退火（默认）；`plateau` = `ReduceLROnPlateau(val_loss)`，每 5 轮不改善降半、下限 `1e-7`（`custom/训练参数.md` 的口径） |
| 存优判据 `monitor` | accuracy | 早停与最佳权重（导出用）看哪个指标 | `accuracy` = 验证准确率最高（默认）；`val_loss` = 验证损失最低。`metrics.json` 两种口径下都记录最佳验证损失与准确率 |
| 另存权重 `save_checkpoint` | 关 | 是否把最佳权重另存为 checkpoint | 写入运行目录 `model/iq_checkpoint.pt`（含模型 ID/结构版本/结构参数/类别顺序/窗口长度）；加载时逐项核对，结构参数变了就拒绝 |

**页面不控制的项**（别以为"没入口 = 没发生"）：验证比例（由两个集合的关系决定，页内不按比例划分，
也不产生 `test` 划分）、类别权重（各类等权，靠集合本身均衡）、**数据增强**（时移/相位/频偏/噪声注入
都没接）、优化器与损失（固定 AdamW + 交叉熵 + 余弦退火）、`train_iq.py` 的 `--min-snr`
（命令行支持"只用高 SNR 样本训练"，页面未暴露）、以及清单里的
`--identifier` / `--version` / `--opset` 等元信息（页面使用默认值 `iq-cnn` / `0.1.0` / opset 17）。

---

## 5. 训练数据流（集合 → 运行目录）

### 5.1 链路

1. GUI 组装的配置带 `train_collection_id` / `val_collection_id`、`task="iq"`、`arch`、`device`、`epochs`、`batch`、`lr`、`seed` 与 `workspace`；没有数据集路径、来源与旧集合字段。
2. worker 首阶段「构建训练输入」：`inputs_plan` 复算预检（任务名 `iq` 映射到标注集任务 `amc`），取两个集合的 AMC 标注集，`build_rows` 装配候选行时**只保留 `class_state=known` 的目标**（`unknown`、`out_of_taxonomy`、未标注目标跳过）；训练集行标 `split=train`、验证集行标 `split=val`，**不按比例重划、不产生 `test` 划分**。
3. `iq_snapshot` 把行实体化到 `runs/<id>/collection_data/`：每行按提取范围取 IQ，经推理端同一份 `iq_waveform` 前端口径生成 `(2, N)` float32 单位 RMS 窗口；窗口凑不满（样本不足等）时该行跳过并把异常文本计入 `skipped`；一个有效窗口都没有时直接报错。随后写 `iq_dataset.npz` 与 `iq_dataset.json`。
4. `training_inputs.json` = `plan_summary(plan)`，记录两个集合的 ID／名称、标注集 ID 与启动时的样本统计；`experiment.json` 的 `config` 保存页面配置原样。**不写 `dataset_version_id`**。
5. 之后的阶段由 `iq_plan` 生成：在所选 Python 环境里校验数据契约与 CUDA 可用性 → `train_iq.py` 训练并导出（页面勾选「覆盖默认结构参数」时附加 `--model-params`，权重衰减/早停轮数走 `--weight-decay/--patience`）→ `verify_iq.py` 验收。运行目录内的 `collection_data/` 只是本次输入快照，不作为用户可见的「导出的训练集」，也不在 `datasets/` 或数据版本表登记。

### 5.2 运行目录布局（AMC 部分）

| 路径 | 写入者 | 内容 |
| --- | --- | --- |
| `runs/<id>/collection_data/iq_dataset.npz` | worker | `waveforms (M,2,N) float32`、`labels`、`split`、`source`（值为 `collection`）、`snr_db`、`offset_hz`、`bandwidth_hz`、`sample_rate_hz`、`index` |
| `runs/<id>/collection_data/iq_dataset.json` | worker | 契约 `iq_waveform_v1`、通道排布 `iq_channels_first_v1`、类别字典与类别顺序、窗口长度、划分统计（策略 `collection`）、来源统计、`skipped` 明细 |
| `runs/<id>/model/iq.onnx` | `train_iq.py` | ONNX 模型：输入 `(1, 2, N)` float32，输出 `(1, C)` 概率（softmax 在图内） |
| `runs/<id>/model/iq_manifest.json` | `train_iq.py` | 模型清单：契约、类别顺序、前端口径、训练信息（各划分样本数、轮数、优化参数、种子、设备、最佳轮次与最佳验证准确率） |
| `runs/<id>/model/metrics.json` | `train_iq.py` | `validation`（准确率、Macro-F1、每类指标、混淆矩阵、`per_snr` 分档）、`history`、口径说明 |
| `runs/<id>/verification.json` | `verify_iq.py --json` | 验收报告：清单与图形状、确定性场景端到端、可复现、数据集独立验证 |
| `runs/<id>/training_inputs.json`、`environment.json`、`worker.log`、`experiment.json` | worker / 运行器 | 与检测页相同（见[信号检测训练](07信号检测训练.md) §5.2、§6.4） |

划分统计里 `test` 恒为 0：训练与验收都只使用 `train` 与 `val`，且两边都必须非空（`train` 或 `val` 为空时快照直接报错）。

### 5.3 可复现性声明

与检测页同口径：训练输入取自集合的**当前**内容，集合成员、目标参数或标签之后变化，历史实验**不保证**逐位复现，也不会自动重跑或回滚；同一份固定种子与参数下的 `iq_dataset.npz` 本身可逐字节复现（训练脚本侧的性质），但本页不承诺跨运行复现集合的输入。严格复现需要另行引入数据版本与不可变清单。

---

## 6. 运行、进度与收尾

启动、占槽、任务登记、失败回滚与实验记录落盘与检测页完全一致，见[信号检测训练](07信号检测训练.md) §6.1、§6.6。AMC 差异：

- **阶段序列**：「构建训练输入」→「生成 IQ 窗口 i/N」（输入快照进度）→「训练输入就绪：M 个样本」→「校验环境与数据」→「训练与导出」→「模型验收」。
- **轮次事件**：`train_iq.py --events` 每轮打印 `TRAIN_EVENT {"epoch": n, "loss": x, "validation_accuracy": a}`；运行器归一化为 `done=min(n, epochs)`、`total=epochs`，文案带轮次与 loss，事件同时存入记录供**验证准确率**曲线使用（检测侧的轮次事件只有轮次与 loss，因此检测页第二条曲线为空）。
- **进度语义**：每次事件都显式写 `done/total`，换阶段清零；阶段没有进度信息时用 `0/0` 表示忙碌态（详见[信号检测训练](07信号检测训练.md) §6.2）。
- **取消与关窗**：请求停止受管进程树并等 2 秒，未退出则强制结束；关窗 TERM 等 1 秒、强杀再等 1 秒；进程树未确认退出就保持占槽、阻止关窗并可重试。Windows 用 Job Object + 启动门事件并跟踪未进入 Job 的后代，POSIX 由 worker 建立新会话、按进程组终止（只在一处 `setsid`）。
- **首阶段**：快照阶段同样在终止范围内；被中断时运行目录会保留半成品，便于复核，不视为有效实验。

---

## 7. 「实验与日志」

| 区域 | 内容 |
| --- | --- |
| 历史下拉 | 只列本页 `iq` 记录，按时间倒序；空列表清理旧展示与加载状态 |
| 状态 | 「成功/失败/已停止/已中断/运行中 · 错误或目录」 |
| 指标 | 读取 `model/metrics.json`，去掉 `history` 后按 AMC 口径展示：验证集准确率、Macro-F1（有则显示）、混淆矩阵（行：真值；列：预测）与类别顺序、按 SNR 统计（`per_snr` 列表按 `low_db/high_db/count/accuracy` 渲染；也兼容字典形状） |
| 曲线 | loss 曲线与验证准确率曲线（与轮次事件同源） |
| 日志 | 该实验 `worker.log` 的末尾片段，UTF-8 → GBK 逐级解码 |
| 动作 | 「停止任务」；「加载验收通过的模型」仅在记录成功、`iq_manifest.json` 存在时可用 |

「加载验收通过的模型」把 `model/iq_manifest.json` 路径写入「调制识别」页的 AI 入口并切换到该页；清单缺失或不可解析时页面报错，不改动路径。验收报告 `verification.json` 与 `per_snr` 分档保存在实验目录，供复核；验收通过只表示链路正确，**不代表精度达标**（识别准确率的合格门限仍是待确认项）。

训练成功后模型还会自动入库到 `training/models/<模型名称>/`（命名规则与入库过程见[信号检测训练](07信号检测训练.md) §6.7）。「调制识别」页的识别模型下拉按用途 `amc` 过滤模型库条目，选中即填入清单路径，页面再按清单契约分流到特征通路或原始 IQ 通路；未登记的清单用下拉里的「浏览本地文件…」指定。模型的查看、重命名与删除在「数据管理 → 模型管理」。

---

## 8. 操作流程

1. 在「IQ 信号生成 → 信号集合生成」批量生成并自动标注（AMC 标注默认启用），或用「信号导入」导入录制（导入路径只登记 AMC 标签）后加入集合。
2. 左栏选中该集合；用「未标注AMC信息」过滤或按信号名称查找待标注资产。
3. 切到「AMC 识别训练 → 信号标注」：选中目标行，改类别状态与调制类别（候选来自集合类别字典），必要时修正频率范围、时间范围、SNR 与原始调制标注；点「保存当前标注」（版本与标签同一事务写入）。
4. 切到「训练配置」：填训练源码根目录与训练 Python，选模型（CNN/TCN）、训练集（默认跟随左侧集合）与验证集（显式选择、与训练集不同且不同源）、设备/轮数/批大小/学习率/种子；核对只读 IQ 契约与类别顺序。
5. 点「开始训练 → 验收」：通过预检后自动切到「实验与日志」，观察阶段、日志、loss 与验证准确率；需要中断时点横幅取消或「停止任务」。
6. 结束后核对指标与按 SNR 分档；点「加载验收通过的模型」，到「调制识别」页对所选资产执行识别。

---

## 9. 常见问题（FAQ）

- **类别候选为什么不是固定 A09？** 候选来自当前集合 AMC 标注集的类别字典：生成时默认写 A09 六类（FM / SSB / 2ASK / QPSK / 16QAM / 64QAM），但更宽的字典同样合法；摘要里的顺序就是模型输出下标顺序。
- **`unknown` 与 `out_of_taxonomy` 会进训练吗？** 不会。训练只收已确认的 `known` 类别；未知与字典外标签被跳过并把数量记进 `iq_dataset.json` 的统计与 `skipped`，不会静默当成某一类。
- **保存失败会留下半成品吗？** 不会：标签字段先校验，通过后才在同一事务里写参数版本与标签；时间范围越界、类别名不在字典内等失败都不留写入。
- **为什么表单里没有「中心频率」「带宽」？** 它们由频率上下限派生，页面不提供会造成不一致的手填入口；表格里的频率范围就是数据层保存的边界。
- **为什么填了 AMC 标注，检测训练还是用不了？** 两类标签是分开的：导入路径只写 AMC 标签（`for_detection=0`），检测训练需要到「信号检测训练 → 数据标注」点「启用检测标注」再画框保存。
- **训练集／验证集可以共用一批信号吗？** 不可以。两集合必须不同，且不能有同源（同一 `origin_group_id`）的资产，否则预检按数据泄漏拒绝。
- **验证集只有一类可以吗？** 可以启动，但类别数少于 2 的训练集类别字典会被预检拒绝（「AMC 训练至少需要 2 个类别」）；验证集覆盖不全时指标参考价值有限。
- **为什么窗口长度不能改？** `iq_waveform_v1` 是训练-推理共同契约：窗口长度、通道排布、归一化与抽取比只有一份实现，改动必须同时改标注、训练与推理，本页只读显示。
- **一条录制会切成多少训练样本？** **只切一个**：每条 AMC 标注目标调用一次 `iq_waveform`，取分析后序列**居中**的 1024 点，其余样本丢弃（一条录制里若有多个被标注目标，则每个目标各出一个窗口）。所以 4000 条集合就产出 4000 个训练样本，集合体积（十几 GB）与训练用量（几十 MB）差得很远——录制是给检测、复算、换参数重取窗用的，不是按训练样本存的。
- **一个 10000 点的待识别数据会被切片多次投票吗？** 不会。推理同样是"居中取一个 1024 点窗口"：先按分析中心/带宽做混频-低通-抽取，再取居中 1024 点送进 ONNX 一次，**没有滑窗、没有多窗口概率平均**；样本数不足 1024 直接报错（不补零）。这也意味着"多窗融合"目前是改进方向而非既有能力。
- **模型存在哪里？能不能只留一个文件？** 运行目录 `training/runs/<时间-id>/model/` 下是**一对**：`iq.onnx`（网络权重）+ `iq_manifest.json`（契约、类别顺序、前端口径、sha256、训练信息）。加载时必须选清单，它与 ONNX 必须在同一目录（清单里的 `library` 是相对路径，并会校验 sha256），单独拷 ONNX 不可用；`metrics.json`（逐轮与每类指标）、`verification.json`（验收报告）只用于回看，可以另外归档。
- **训练跑了多少参数？** 默认 `IQCNN` 在 6 类、1024 点窗口上是 **103,270** 个可训练参数（导出后 BatchNorm 折进卷积，ONNX 里 102,822 个张量元素、约 419 KB）——它是个小网络，别指望靠加大网络解决数据质量问题。
- **实验页为什么只有验证集、没有测试集？** 训练输入不产生 `test` 划分：训练集行 → `train`、验证集行 → `val`；测试集不参与模型选择。
- **进度为什么会在换阶段时清零？** 阶段事件与轮次事件每次显式写 `done/total`，换阶段清零是刻意行为，避免横幅停在上一阶段（详见[信号检测训练](07信号检测训练.md) §6.2）。
- **取消训练要多久？** 先请求进程树停止并等 2 秒，未退出才强杀；进程树未确认退出就保持占槽并允许重试。
- **训练数据能复现吗？** 不承诺跨运行复现：输入取自集合当前内容；同一次运行可用运行目录里的 `collection_data/` 复核。
- **页面能训练 TorchSig 混入数据吗？** 训练脚本 CLI 支持混入（显式类别映射），但页面不暴露该入口，也没有任何生成入口。
- **日志出现乱码？** 新实验按 UTF-8 写出，历史记录按 GBK 写出，读取时 UTF-8 → GBK → 替换逐级回退。

---

## 10. 与代码 / 测试的对应

| 功能 | 代码 | 测试 |
| --- | --- | --- |
| 页面与共享基类 | `src/signal_analysis/ui/pages/amc_training_page.py`、`ui/pages/training_common.py` | `tests/analysis/test_training_page_split.py` |
| 页面固定任务与独立 owner | `amc_training_page.TASK/OWNER/HISTORY_TASKS`、`ui/main_window.py`（页面注册表） | `test_training_page_split.py::test_training_pages_have_fixed_tasks_and_independent_owners` |
| 类别状态三态与参数编辑 | `amc_training_page`（`CLASS_STATES`、`COLUMNS`、`_write_target`） | `test_training_page_split.py::test_amc_annotation_edits_class_and_parameters` |
| 类别候选来自集合字典 | `amc_training_page._fill_class_choices` + `data/labels.py::get_taxonomy` | `test_labels_store.py::test_amc_label_states_and_validation` |
| 同事务写版本与标签（非法零写入） | `data/labels.py::append_amc_annotation`、`data/targets.py::_append_target_version` | `test_labels_store.py::test_amc_annotation_writes_version_and_label_in_one_transaction`、`test_training_page_split.py::test_amc_save_keeps_unedited_fields_and_writes_nothing_when_invalid` |
| 未编辑字段沿用 | `data/targets.py::carryover_fields` | `test_labels_store.py::test_carryover_fields_keep_unedited_values_and_decode_params` |
| 切换 AMC 目标先保存与回滚 | `amc_training_page._target_selection_changed`/`_asset_selection_changed` | `test_training_page_split.py::test_switching_amc_target_saves_and_blocks_on_failure` |
| 训练输入预检（类别字典一致、类别数、防泄漏） | `services/training_inputs.py::inputs_plan`、`amc_training_page.validate_task_config` | `test_training_snapshot.py::test_inputs_plan_reports_missing_sets_and_same_collection`、`::test_inputs_plan_rejects_shared_origin_group` |
| AMC 快照（只收 known、跳过计数） | `services/training_snapshot.py::iq_snapshot`/`snapshot` | `test_training_snapshot.py::test_iq_snapshot_keeps_only_confirmed_classes` |
| 数据层行装配（只收 known） | `data/datasets.py::build_rows` | `test_dataset_versions.py::test_amc_dataset_version_uses_target_windows` |
| IQ 契约与清单 | `contracts/iq.py`（`iq_waveform_v1`/`iq_channels_first_v1`/`unit_rms`/A09）、`training/train_iq.py` | `test_iq_training_tools.py::test_iq_dataset_card_matches_inference_contract`、`::test_train_iq_load_dataset_validates_the_contract` |
| 训练、导出与验收阶段 | `services/training_jobs.py::iq_plan`（含高级超参 `iq_tuning_args`）、`training/desktop_worker.py`、`training/train_iq.py`、`training/verify_iq.py` | `test_iq_training_tools.py::test_train_classifier_learns_and_exports_consistently`、`::test_iq_dataset_is_byte_reproducible`、`::test_iq_plan_passes_advanced_tuning_args` |
| 高级训练参数（页面开关与校验） | `ui/pages/amc_training_page.py::add_task_fields`/`configuration`、`services/training_jobs.py::iq_tuning_args` | `test_training_page_split.py::test_amc_advanced_training_fields_are_opt_in`、`test_iq_training_tools.py::test_iq_plan_passes_advanced_tuning_args` |
| 模型目录驱动展示（列表、架构、窗口、依赖） | `algorithms/amc/ai_model/`（`describe_model`/`spec_json`/`check_model_samples`）、`ui/pages/amc_training_page.py::_apply_catalog`/`_refresh_catalog`、`services/training_jobs.py::query_catalog`/`preflight_iq` | `test_amc_model_catalog.py`、`test_training_page_split.py::test_amc_model_catalog_drives_architecture_window_and_parameters`、`test_iq_training_tools.py::test_query_catalog_reads_the_catalog_from_a_training_environment`、`::test_preflight_iq_fills_window_and_rejects_unknown_model` |
| 端到端（集合直读 → 训练 → 加载） | `training/desktop_worker.py` + 运行器 | `test_training_workbench.py::test_gui_external_iq_training_from_collections_verify_and_load`、`::test_iq_test_partition_does_not_enter_validation` |
| 运行器进度/日志/收尾与运行槽 | `ui/training_runner.py`、`ui/training_coordinator.py`、`ui/training_process.py` | `test_training_page_split.py::test_progress_reports_epochs_and_resets_on_stage_change`、`::test_stop_and_close_terminate_worker_and_spawned_child`、`test_progress_channel.py` |
| 指标展示（含 per_snr）与加载 | `ui/pages/training_common.py::metric_text`/`load_model`、`services/training_jobs.py`（`list_experiments`/`decode_worker_log`） | `test_training_page_split.py::test_metric_text_shows_per_snr_buckets`、`test_iq_training_tools.py::test_train_iq_per_snr_buckets_are_explicit_about_empty_bands` |
| 识别侧契约（加载后的推理） | `contracts/iq.py`、`ml/iq.py`、`ui/pages/amc_page.py` | `test_amc.py` |
| 模型命名、入库与管理 | `services/model_store.py`、`ui/pages/training_common.py::publish_model`、`ui/pages/models_page.py` | `test_model_store.py`、`test_models_gui.py` |

---

## 11. 已知局限与后续方向

- **不保证跨运行复现。** 输入取自集合当前内容；严格复现需要另行引入数据版本与不可变清单。
- **实采跳频的逐跳人工标注未实现**，本页的「目标」选项目前只能来自生成器或导入时写入的目标。
- **页面不暴露 TorchSig 混入**（那是训练脚本 CLI 的能力），也没有任何生成入口；页面不支持在线生成数据。
- **CNN/TCN 之外的通路（如特征通路线性/Transformer 分类器与 `train_amc.py`）不在本页**；本页固定 `iq` 任务与 `iq_waveform_v1` 契约（特征通路内容见[传统特征与启发式判定](algorithms/调制识别_传统特征与启发式判定.md) §3、§5.6）。
- **识别准确率的合格门限仍是待确认项**：指标只描述当前数据分布；低信噪比下 16QAM 与 64QAM 易混（见工作台与 `training/README.md` 的实测记录），验收通过不代表精度达标。
- **一条录制只用"居中一个窗口"**：没有滑窗提取、没有跨窗融合，也没有短录制（短于窗口长度）的兼容路径（直接报错）。训练样本数因此等于"被标注的目标数"，集合里绝大多数采样点未被使用；改进方向是快照侧多窗口提取 + 推理侧概率平均，以及为 AMC 生成更短的专用集合。
- **无数据增强、无超参搜索**：时移/相位/频偏/噪声注入等保标签增强未接入；高级参数只是"手工覆盖默认值"，没有网格或贝叶斯搜索证据（对比实验见[调制识别：AI 特征学习](algorithms/调制识别_AI特征学习.md) §8）。
- **未做真实显示器上的人工布局与交互验收**（GUI 流程由离屏集成测试覆盖），CUDA 与冻结构建需在目标环境另行验收。
- 尚无多实验 A/B 汇总界面；误分类明细与更丰富的评测（如按类别/带宽分组的报表）属于后续方向。
