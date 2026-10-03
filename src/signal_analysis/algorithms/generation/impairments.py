"""生成器损伤项的**预留接口**（方案 §6.4）：目前全部未实现，界面置灰，配方里用到即报错。

为什么先留接口
--------------

生成页的“损伤”区要先把频偏、相位噪声、IQ 不平衡、多径这几项摆出来，但项目生成器
（``algorithms.generation.iqgen.generate_iq``）还没有这些能力。这里把每一项的参数形状、施加顺序、
对真值的影响写在代码里，做到：

* 界面按 :data:`IMPAIRMENTS` 逐项渲染，``implemented=False`` 的整行置灰并提示原因；
  某项实现后只需把 ``implemented`` 改成 ``True`` 并补 :meth:`Impairment.apply`，
  界面会自动启用，不需要再改表单代码；
* 配方校验（``recipes.check_generator_support``）通过 :func:`ensure_supported`
  拒绝未实现的项——**不静默忽略**，否则数据里没有损伤而配方里写着有，复现就成了谎言；
* 执行器（``collection_gen``）在 :func:`apply_impairments` 处留好了调用位置。

实现时必须遵守的约定（方案 §6.4）
--------------------------------

1. **默认全部关闭；关闭时输出与当前逐位一致**，并通过
   ``scripts/check_numeric_equivalence.py``。数值基线口径见
   ``docs/电磁信号分析和识别/基础工程实现与文件说明.md`` 的「数值模块与算法边界」。
2. **固定施加顺序**：基带调制与成形 → 频偏 → 相位噪声 → 多径 → 叠加噪声 → IQ 不平衡。
   噪声功率按多径之后的信号功率定标，使 ``inband_snr_v1`` 口径不变。
   即 :data:`IMPAIRMENTS` 的声明顺序就是施加顺序。
3. **每项设上限并校验**；组合后占用频带必须仍在采样带宽内，超出则重抽或报错。
4. 损伤参数写入生成摘要（``generation``）并进入目标的 ``params_json``；摘要版本升级，
   旧摘要仍可读取。
5. 逐信号的项（频偏、相位噪声、多径）必须在 ``algorithms.generation.iqgen.generate_iq`` 的逐信号
   循环里、``_synthesize`` 之后、叠加进记录之前调用（``stage="signal"``）；记录级的项
   （IQ 不平衡）在叠加噪声之后调用（``stage="record"``）。这意味着需要给数值核心增加
   钩子，属于可编译模块，需要同步回归。
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Field:
    """一个可抽样的损伤参数：``range`` 对应 ``uniform``，``choice`` 对应 ``choice``。"""

    key: str
    label: str
    kind: str = "range"          # "range" | "choice"
    unit: str = ""
    low: float = 0.0
    high: float = 1.0
    choices: tuple = ()


@dataclass(frozen=True)
class Impairment:
    """一项损伤的声明；子类化或替换实例即可实现，无需改界面。"""

    name: str
    label: str
    recipe_key: str              # 在配方 ``impairments`` 组里的键
    stage: str                   # "signal"（逐信号，发射端/信道）| "record"（记录级，接收端）
    summary: str                 # 物理含义与对真值的影响（显示在提示里）
    fields: tuple = field(default_factory=tuple)
    implemented: bool = False
    reason: str = "项目生成器尚未实现该损伤"

    def validate(self, params):
        """校验抽出的参数；实现时在此处拒绝越界值（约定 3）。"""
        raise NotImplementedError(self.reason)

    def apply(self, samples, sample_rate, params, rng, truth):
        """把损伤施加到 ``samples``（complex128 一维），返回新的样本数组。

        ``truth`` 是该信号的真值字典（逐信号项会读写 ``center_hz`` 等，使目标参考
        参数含损伤后的实际值而 ``nominal_*`` 保持名义值）；记录级项可传 ``None``。
        """
        raise NotImplementedError(self.reason)


#: 声明顺序即施加顺序（约定 2）。
IMPAIRMENTS = {
    "cfo": Impairment(
        name="cfo", label="频偏", recipe_key="cfo_ratio", stage="signal",
        summary="信号载频偏离名义值：占用频带中心随之平移，center_hz 取含频偏的实际值，"
                "nominal_center_hz 保持名义值。检测脚本已有 --cfo-ratio，AMC 与本生成器尚无。",
        fields=(Field("cfo_ratio", "相对频偏（占采样率）", "range", "", 0.0, 0.001),),
        reason="本生成器尚未实现频偏（检测脚本 build_dataset.py 有 --cfo-ratio，AMC 脚本没有）"),
    "phase_noise": Impairment(
        name="phase_noise", label="相位噪声", recipe_key="phase_noise", stage="signal",
        summary="发射端本振相噪：不改变占用频带，只影响星座质量；档位或噪声谱参数写入摘要。",
        fields=(Field("phase_noise", "档位", "choice", choices=("off", "low", "high")),),
        reason="本生成器尚未实现相位噪声"),
    "multipath": Impairment(
        name="multipath", label="多径", recipe_key="multipath", stage="signal",
        summary="径数、时延扩展与衰落类型：改变信号功率与频谱形状，带内 SNR 与占用带宽按"
                "实测重算；真值带宽仍为信号占用带宽。",
        fields=(Field("multipath", "类型", "choice", choices=("off", "two_ray", "rayleigh")),
                Field("delay_spread_s", "时延扩展", "range", "s", 0.0, 2e-5)),
        reason="本生成器尚未实现多径"),
    "iq_imbalance": Impairment(
        name="iq_imbalance", label="IQ 不平衡", recipe_key="iq_imbalance", stage="record",
        summary="接收端幅度/相位不平衡：在 -f 处产生镜像分量。镜像**不标为目标**，配方需限定"
                "不平衡范围，评估中记录镜像存在的样本。",
        fields=(Field("gain_db", "幅度不平衡", "range", "dB", 0.0, 1.0),
                Field("phase_deg", "相位偏差", "range", "°", 0.0, 5.0)),
        reason="本生成器尚未实现 IQ 不平衡"),
}

#: 生成器参数里也有预留项：独立符号率目前由 带宽 / 滚降 推导（``plan_signal``）。
RESERVED_SIGNAL_PARAMETERS = {
    "symbol_rate_baud": {
        "label": "符号率", "unit": "baud", "low": 5e3, "high": 1e5,
        "reason": "生成器尚未支持独立符号率：数字样式的符号率由带宽与滚降系数推导；"
                  "同时给出带宽与符号率需先做一致性校验（不一致直接报错）后才能启用",
    },
}

_BY_KEY = {item.recipe_key: item for item in IMPAIRMENTS.values()}
#: 配方 ``impairments`` 组里“嵌套参数”的子键（归属于同名外层项，如 iq_imbalance.gain_db）。
_SUBKEYS = {"gain_db": "iq_imbalance", "phase_deg": "iq_imbalance",
            "delay_spread_s": "multipath"}


def ensure_supported(tree):
    """配方 ``impairments`` 组里出现未实现的项就报错；未知键同样报错。"""
    for key in (tree or {}):
        item = _BY_KEY.get(key) or IMPAIRMENTS.get(_SUBKEYS.get(key, ""))
        if item is None:
            raise ValueError(f"impairments.{key}：未知的损伤项；可用："
                             f"{' / '.join(_BY_KEY)}")
        if not item.implemented:
            raise ValueError(f"损伤项“{item.label}”（impairments.{key}）不可用：{item.reason}。"
                             "请从生成参数中移除（界面里该项为置灰）")


def apply_impairments(samples, sample_rate, drawn, rng, truth=None, *, stage):
    """按声明顺序施加本阶段已抽样的损伤；没有请求时原样返回（约定 1）。

    执行器在生成每条录制后调用。``drawn`` 是 ``draw_record`` 抽出的
    ``impairments`` 组。当前全部未实现，因此请求非空时直接抛错，而不是悄悄跳过。
    """
    if not drawn:
        return samples
    ensure_supported(drawn)
    for item in IMPAIRMENTS.values():
        if item.stage != stage:
            continue
        params = {key: value for key, value in drawn.items()
                  if _BY_KEY.get(key) is item or _SUBKEYS.get(key) == item.name}
        if params:
            item.validate(params)
            samples = item.apply(samples, sample_rate, params, rng, truth)
    return samples
