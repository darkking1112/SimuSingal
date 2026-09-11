"""适配器基类与"从权重到清单"的共用流水线。

一个适配器只需要回答四件事：

1. **它是谁**（:class:`detectors.registry.AdapterInfo`：名称、许可证、上游地址）；
2. **本机能不能用**（``is_available`` / ``install_hint``）；
3. **怎么把权重变成 nn.Module**（``load_model``）；
4. **原始输出是什么布局**（``layout`` 以及可选的三件套预处理参数）。

剩下的事——包装、空跑校验、导出 ONNX、写清单、许可证闸门——全部由
:meth:`DetectorAdapter.export` 统一完成，避免每个框架各写一遍而引入口径差。
"""

from __future__ import annotations

import abc
import json
from pathlib import Path

from . import labels as label_tools
from .contract import layout_spec
from .dataset import contract_of, load_dataset, split_records
from .registry import COPYLEFT, READ_ME_HINT, AdapterInfo, require_available

DEFAULT_OPSET = 17


class DetectorAdapter(abc.ABC):
    """第三方检测器的接入描述。子类只需实现 :meth:`load_model` 与可用性检查。"""

    #: 静态描述（子类必须覆盖）
    info: AdapterInfo

    #: 原始输出的列布局（见 :mod:`detectors.contract`）
    layout = "pixel_xyxy"

    #: 图内通道复制次数（ImageNet 预训练主干需要 3）
    channel_repeat = 1

    #: 图内输入缩放：乘该系数得到框架期望的量纲。
    #: ``255.0`` 表示框架在 Python 侧做 ``/255``（Ultralytics）；
    #: ``1.0`` 表示框架直接吃原始灰度值（YOLOX，其官方 demo 不做 /255）。
    input_scale = 1.0

    #: 图内标准化（与 :attr:`channel_repeat` 同长度的序列）。
    #: Ultralytics 只做 ``/255``，**不做** ImageNet 均值方差，所以是 None。
    mean = None
    std = None

    #: 预处理必须由调用方显式声明。不同分支口径不一致时置 True：
    #: 缺少 ``--input-scale`` 直接报错，而不是套一个猜的默认值把口径错到底。
    prep_required = False

    #: ``prep_required`` 为真时在报错里给出的"选哪个"提示
    prep_hint = ""

    #: 导出原生数据集时默认使用的格式（``yolo`` / ``coco``）
    dataset_format = "coco"

    #: 权重说明（命令行帮助与报错使用）
    weights_hint = "框架自己的权重文件"

    def describe(self):
        data = self.info.as_dict()
        data.update({
            "layout": self.layout,
            "channel_repeat": self.channel_repeat,
            "input_scale": self.input_scale,
            "mean": list(self.mean) if self.mean else None,
            "std": list(self.std) if self.std else None,
            "prep_required": bool(self.prep_required),
            "dataset_format": self.dataset_format,
            "available": bool(self.is_available()),
            "weights_hint": self.weights_hint,
        })
        return data

    # ------------------------------------------------------------------ 可用性

    @abc.abstractmethod
    def is_available(self):
        """本机是否已具备运行该框架的最小依赖。"""

    @abc.abstractmethod
    def install_hint(self):
        """依赖缺失时给出的可执行安装命令。"""

    # ------------------------------------------------------------------ 模型加载

    @abc.abstractmethod
    def load_model(self, weights, *, device="cpu", image_size=1024, max_boxes=32):
        """把权重加载成 ``nn.Module``（返回的模块必须能接受 ``(B, C, H, W)``）。"""

    def configure(self, **options):
        """接收框架专属选项（如 ``--framework-path`` / ``--exp-file``）。"""
        self.options = dict(options)
        return self

    def _resolve_prep(self, *, channel_repeat=None, input_scale=None, mean=None, std=None):
        """合并调用方入参与适配器默认值；``prep_required`` 时强制显式声明。"""
        if self.prep_required and input_scale is None:
            raise SystemExit(
                f"{self.info.name}（{self.info.title}）的输入预处理在不同实现分支之间不一致，"
                "必须显式给 --input-scale（必要时再加 --input-mean / --input-std），"
                "不能套用默认值。\n" + (self.prep_hint or "") + "\n" + READ_ME_HINT)
        return {
            "channel_repeat": self.channel_repeat if channel_repeat is None else channel_repeat,
            "input_scale": self.input_scale if input_scale is None else input_scale,
            "mean": self.mean if mean is None else mean,
            "std": self.std if std is None else std,
        }

    def dataset_format_for(self, fmt=None):
        return str(fmt or self.dataset_format).strip().lower()

    def export_dataset(self, data_root, records, card, output, fmt=None):
        """把项目数据集导出成该框架的原生标注格式。"""
        return label_tools.export_dataset(data_root, records, card, output,
                                          self.dataset_format_for(fmt), contract_of(card))

    def check_available(self, allow_copyleft=False, runtime=True):
        """校验运行环境与许可证；返回 ``(适配器, 是否为有传染性的许可证)``。"""
        require_available(self, allow_copyleft, runtime=runtime)
        return self, self.info.license in COPYLEFT

    # ------------------------------------------------------------------ 原生 ONNX 导出

    def export_native_onnx(self, weights, *, image_size, output, opset=DEFAULT_OPSET):
        """用框架自带的导出器产出原生 ONNX（默认实现：不支持，交给 ``--onnx`` 模式）。"""
        raise SystemExit(
            f"{self.info.name} 的权重需要先用它自带的导出脚本转成 ONNX，"
            "再用 --onnx 模式做契约改写。\n  " + self.install_hint())

    # ------------------------------------------------------------------ 主流水线

    def export(self, *, weights, data_root, output, image_size=0, max_boxes=32,
               layout=None, opset=DEFAULT_OPSET, model_id="", version="0.1.0",
               device="cpu", allow_copyleft=False, channel_repeat=None,
               input_scale=None, mean=None, std=None, notes="", onnx_name="detector.onnx",
               manifest_name="detector.json", export_dataset=False, dataset_output="",
               dataset_format=None, probe=False):
        """加载权重 → 包装 → 空跑校验 → 导出 ONNX → 写清单。"""
        self.check_available(allow_copyleft)
        card, records = load_dataset(data_root)
        contract = contract_of(card)
        size = int(image_size or contract["image_size"])
        if size != int(contract["image_size"]):
            raise SystemExit(
                f"导出尺寸必须等于数据集契约 {contract['image_size']}，"
                "否则训练图像参数三件套与推理端不再一致")
        layout_name = str(layout or self.layout)
        spec = layout_spec(layout_name)

        from .torch_export import (build_contract_head, describe_onnx, export_contract_onnx,
                                  probe_detector, require_torch, validate_head)

        torch = require_torch()
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        model = self.load_model(weights, device=device, image_size=size, max_boxes=max_boxes)

        prep = self._resolve_prep(channel_repeat=channel_repeat, input_scale=input_scale,
                                  mean=mean, std=std)
        probe_result = None
        if probe:
            probe_result = probe_detector(model, image_size=size, torch=torch, **prep)

        head = build_contract_head(model, layout=layout_name, image_size=size,
                                   max_boxes=max_boxes, torch=torch, **prep)
        validation = validate_head(head, image_size=size, max_boxes=max_boxes, torch=torch)

        onnx_path = output / onnx_name
        export_contract_onnx(head, onnx_path, image_size=size, opset=opset, torch=torch)
        graph = describe_onnx(onnx_path)

        dataset_summary = None
        if export_dataset:
            target = Path(dataset_output) if dataset_output else output / "dataset"
            dataset_summary = self.export_dataset(data_root, records, card, target,
                                                  dataset_format)

        manifest = self._write_manifest(
            output / manifest_name, onnx_path, contract=contract, card=card, records=records,
            spec=spec, validation=validation, graph=graph, opset=opset, size=size,
            model_id=model_id or f"{self.info.name}-detector", version=version,
            weights=weights, notes=notes, prep=prep, dataset_summary=dataset_summary,
            allow_copyleft=allow_copyleft)
        return {"manifest": manifest, "manifest_path": str(output / manifest_name),
                "onnx_path": str(onnx_path), "graph": graph, "validation": validation,
                "probe": probe_result, "dataset": dataset_summary,
                "framework": self.describe()}

    # ------------------------------------------------------------------ ONNX 契约改写

    def export_onnx(self, *, source, output, image_size=0, max_boxes=32, layout=None,
                    model_id="", version="0.1.0", allow_copyleft=False, channel_repeat=None,
                    input_scale=None, mean=None, std=None, transpose=None, data_root="",
                    labels=("emitter",), spectrogram_nfft=None, dynamic_range_db=None,
                    notes="", weights="", onnx_name="detector.onnx",
                    manifest_name="detector.json"):
        """把框架自己的导出图改写成契约图并写清单（**不需要 torch**）。

        这是接入 YOLOX / RT-DETR / Ultralytics 的主路径：框架负责训练与导出，
        我们负责"补输入预处理 + 补输出几何转换"，权重一个字节都不动。

        本路径不需要框架的 Python 包（图已经是导出的成品），只需 onnx/onnxruntime。
        """
        self.check_available(allow_copyleft, runtime=False)
        from signal_analysis.ml.manifest import DEFAULT_DYNAMIC_RANGE_DB, DEFAULT_NFFT

        if data_root:
            card, records = load_dataset(data_root)
            contract = contract_of(card)
            size = int(image_size or contract["image_size"])
            if size != int(contract["image_size"]):
                raise SystemExit(
                    f"导出尺寸必须等于数据集契约 {contract['image_size']}")
        else:
            card, records, contract = {}, [], {}
            size = int(image_size)
            if size <= 0:
                raise SystemExit("未给 --data 时必须显式给 --imgsz")
            contract = {"image_size": size, "labels": list(labels),
                        "spectrogram_nfft": int(spectrogram_nfft or DEFAULT_NFFT),
                        "dynamic_range_db": float(dynamic_range_db
                                                  if dynamic_range_db is not None
                                                  else DEFAULT_DYNAMIC_RANGE_DB)}

        layout_name = str(layout or self.layout)
        spec = layout_spec(layout_name)
        prep = self._resolve_prep(channel_repeat=channel_repeat, input_scale=input_scale,
                                  mean=mean, std=std)

        from .onnx_contract import rewrite

        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        native = source
        if not str(source).lower().endswith(".onnx"):
            native = self.export_native_onnx(weights or source, image_size=size, output=output)
        onnx_path = output / onnx_name
        _, graph = rewrite(native, onnx_path, layout=layout_name, image_size=size,
                           max_boxes=max_boxes, transpose=transpose, **prep)

        dataset_summary = None
        if data_root:
            dataset_summary = {"root": str(data_root), "samples": len(records),
                               "splits": card.get("splits") or {}}
        manifest = self._write_manifest(
            output / manifest_name, onnx_path, contract=contract, card=card, records=records,
            spec=spec, validation=graph, graph=graph,
            opset=_graph_opset(graph), size=size,
            model_id=model_id or f"{self.info.name}-detector", version=version,
            weights=weights or source, notes=notes, prep=prep,
            dataset_summary=dataset_summary, allow_copyleft=allow_copyleft)
        return {"manifest": manifest, "manifest_path": str(output / manifest_name),
                "onnx_path": str(onnx_path), "graph": graph, "validation": graph,
                "probe": {"kind": "graph", "shape": graph["outputs"][0]["shape"]},
                "dataset": dataset_summary, "framework": self.describe()}

    def _write_manifest(self, manifest_path, onnx_path, *, contract, card, records, spec,
                        validation, graph, opset, size, model_id, version, weights, notes,
                        prep, dataset_summary, allow_copyleft):
        from signal_analysis.ml import write_model_manifest

        splits = card.get("splits") or {}
        training = {
            "framework": self.info.name,
            "framework_title": self.info.title,
            "upstream": self.info.upstream,
            "weights": str(weights),
            "license": self.info.license,
            "distributable": bool(self.info.distributable),
            "layout": spec["name"],
            "layout_columns": spec["columns_desc"],
            "preprocess": {
                "channel_repeat": int(prep["channel_repeat"]),
                "input_scale": float(prep["input_scale"]),
                "mean": list(prep["mean"]) if prep["mean"] else None,
                "std": list(prep["std"]) if prep["std"] else None,
                "note": "以上预处理已写进导出图，推理端只喂 [0,1] 单通道时频图",
            },
            "dataset": {"root": str(card.get("generator_script", "")),
                        "samples": len(records), "splits": splits},
            "validation": validation,
        }
        if dataset_summary:
            training["native_dataset"] = dataset_summary
        note_lines = [
            f"{self.info.title}（{self.info.license}）经 training/detectors 包装导出",
            f"输出布局 {spec['name']}：{spec['columns_desc']}",
            f"标签由 band_to_box 生成，y 方向不做翻转（图像行 0 = +fs/2）",
        ]
        if not self.info.distributable:
            note_lines.append(
                f"许可证 {self.info.license} 具有传染性：本权重不得随产品分发"
                f"（--allow-copyleft 已显式确认）")
        if dataset_summary:
            note_lines.append(label_tools.QUANTIZATION_NOTE)
        if notes:
            note_lines.append(str(notes))
        return write_model_manifest(
            manifest_path, onnx_path, identifier=model_id, version=version, image_size=size,
            opset=opset, labels=list(contract["labels"]), training=training,
            notes="；".join(note_lines),
            spectrogram_nfft=int(contract["spectrogram_nfft"]),
            dynamic_range_db=float(contract["dynamic_range_db"]))


def _graph_opset(graph):
    """从 :func:`onnx_contract.describe` 的结果里取默认域的 opset。"""
    for entry in graph.get("opset", []):
        if entry.get("domain") in ("", "ai.onnx"):
            return int(entry.get("version", 0))
    return 0


def summary_table(adapters):
    """把适配器（对象或 :meth:`DetectorAdapter.describe` 的字典）渲染成等宽表格。"""
    header = ("名称", "框架", "许可证", "可分发", "可用", "布局")
    rows = [header]
    for adapter in adapters:
        if isinstance(adapter, dict):
            name, title = adapter["name"], adapter["title"]
            license_name = adapter["license"]
            distributable = adapter["distributable"]
            layout = adapter.get("layout", "")
            available = "是" if adapter.get("available") else "否"
        else:
            info = adapter.info
            name, title, license_name = info.name, info.title, info.license
            distributable = info.distributable
            layout = adapter.layout
            available = "是" if adapter.is_available() else "否"
        rows.append((name, title, license_name, "是" if distributable else "否",
                     available, layout))
    widths = [max(len(str(row[index])) for row in rows) for index in range(len(header))]
    lines = []
    for position, row in enumerate(rows):
        lines.append("  ".join(str(cell).ljust(width) for cell, width in zip(row, widths)))
        if position == 0:
            lines.append("  ".join("-" * width for width in widths))
    return "\n".join(lines)


def dump_json(payload):
    return json.dumps(payload, ensure_ascii=False, indent=2)


def available_splits(records):
    """数据集里出现过的 split 名（导出原生格式时报告用）。"""
    return sorted({str(record.get("split", "train")).lower() for record in records})


def select_split(records, split):
    return split_records(records, split)
