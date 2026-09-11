"""把第三方检测器接到本系统契约上的统一入口。

三条路径，按你的框架"顺手程度"选：

======================  ==========================================  ==================
路径                    适用                                        命令
======================  ==========================================  ==================
``--mode torch``        权重能直接载成 ``nn.Module``（如 ``tiny``）  ``--weights x.pt``
``--mode onnx``         你已用框架自带脚本导出了 ONNX               ``--onnx x.onnx``
``--mode auto``（默认）  自动判断（``.onnx`` → onnx，其余按适配器）  ``--weights x.onnx``
======================  ==========================================  ==================

``--dataset-only`` 只做数据集导出（练手前先对齐标注口径），不碰模型。

常用示例::

    # 看有哪些框架、哪些布局
    .venv/bin/python training/export_contract.py --list-frameworks
    .venv/bin/python training/export_contract.py --list-layouts

    # 把项目数据集导成 COCO（给 RT-DETR 用）
    .venv/bin/python training/export_contract.py --framework rtdetr \
        --data workspace_data/analysis --dataset-only --dataset-output /tmp/rtdetr-data

    # 把框架导出的 ONNX 改写成契约图 + 模型清单
    .venv/bin/python training/export_contract.py --framework rtdetr --onnx runs/rtdetr.onnx \
        --layout normalized_cxcywh --imgsz 1024 --max-boxes 32 \
        --output workspace_data/analysis/models/rtdetr

``--allow-copyleft`` 是针对 AGPL 框架的显式确认：不加它，AGPL 适配器一律拒绝导出，
避免"不小心把传染性许可证的权重带进发行包"（training/README.md §8）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT / "src", REPO_ROOT / "training"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from detectors import base as adapter_base  # noqa: E402
from detectors import contract, registry  # noqa: E402
from detectors.dataset import load_dataset  # noqa: E402

MODES = ("auto", "torch", "onnx")


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="export_contract.py",
        description="把第三方检测器（RT-DETR / YOLOX / Ultralytics / tiny）接成契约 ONNX。")
    parser.add_argument("--framework", default="", help="适配器名，见 --list-frameworks")
    parser.add_argument("--data", default="", help="项目数据集目录（build_dataset.py 的输出）")
    parser.add_argument("--output", default="", help="输出目录（存放 detector.onnx 与 detector.json）")
    parser.add_argument("--weights", default="", help="权重或原生 ONNX 路径")
    parser.add_argument("--onnx", default="", help="原生 ONNX 路径（等价于 --weights *.onnx）")
    parser.add_argument("--mode", default="auto", choices=MODES)

    parser.add_argument("--layout", default="", help="原生输出布局，见 --list-layouts")
    parser.add_argument("--imgsz", type=int, default=0, help="输入边长；给了 --data 则以契约为准")
    parser.add_argument("--max-boxes", type=int, default=32, help="导出的最大检测数（1～256）")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--model-id", default="")
    parser.add_argument("--version", default="0.1.0")
    parser.add_argument("--notes", default="")

    parser.add_argument("--channel-repeat", type=int, default=None, choices=(1, 3))
    parser.add_argument("--input-scale", type=float, default=None,
                        help="图内输入缩放，YOLO 系在 Python 侧做 /255，所以这里是 255")
    parser.add_argument("--input-mean", type=float, nargs="+", default=None)
    parser.add_argument("--input-std", type=float, nargs="+", default=None)
    parser.add_argument("--transpose", default="auto", choices=("auto", "yes", "no"),
                        help="原生输出是否为 (N, 列, 候选框)；auto 时按形状自动判断")

    parser.add_argument("--dataset-only", action="store_true", help="只导出原生数据集，不导出模型")
    parser.add_argument("--dataset-output", default="")
    parser.add_argument("--dataset-format", default="", choices=("", "yolo", "coco"))

    parser.add_argument("--allow-copyleft", action="store_true",
                        help="显式确认接受 AGPL 等传染性许可证（仅限内部评测）")
    parser.add_argument("--probe", action="store_true", help="torch 路径下先打印原生输出形状")
    parser.add_argument("--framework-path", default="",
                        help="git clone 下来的框架仓库路径（YOLOX / RT-DETR）")
    parser.add_argument("--exp-file", default="", help="YOLOX 的 exp 文件路径")
    parser.add_argument("--list-frameworks", action="store_true")
    parser.add_argument("--list-layouts", action="store_true")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    return parser.parse_args(argv)


def _transpose_option(value):
    return {"auto": None, "yes": True, "no": False}[value]


def _print_layouts(as_json=False):
    entries = contract.describe_layouts()
    if as_json:
        print(json.dumps(entries, ensure_ascii=False, indent=2))
        return 0
    print("原生输出布局（--layout 的取值）：\n")
    for entry in entries:
        print(f"  {entry['name']}")
        print(f"    列数    : {entry['columns']}（{entry['columns_desc']}）")
        print(f"    几何    : {entry['geometry']}；尺度: {entry['scale']}")
        print(f"    含类别  : {'是' if entry['has_class'] else '否'}")
        print(f"    说明    : {entry['description']}")
        print(f"    典型    : {entry['examples']}\n")
    return 0


def _print_frameworks(as_json=False):
    adapters = [registry.lookup(name) for name in registry.names()]
    if as_json:
        print(json.dumps([adapter.describe() for adapter in adapters],
                         ensure_ascii=False, indent=2))
        return 0
    print(adapter_base.summary_table(adapters))
    print("\n许可证说明见 training/README.md §8；AGPL 适配器需显式 --allow-copyleft。")
    return 0


def _pick_mode(args, adapter, source):
    if args.mode != "auto":
        return args.mode
    if str(source).lower().endswith(".onnx"):
        return "onnx"
    if adapter.info.name == "tiny":
        return "torch"
    return "onnx"


def _report(payload, as_json):
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    print(f"框架      : {payload['framework']['title']}（{payload['framework']['license']}）")
    print(f"布局      : {payload['framework']['layout']}")
    print(f"ONNX      : {payload['onnx_path']}")
    print(f"模型清单  : {payload['manifest_path']}")
    graph = payload.get("graph") or {}
    for value in graph.get("inputs", []):
        print(f"  输入    : {value['name']} {value['shape']}")
    for value in graph.get("outputs", []):
        print(f"  输出    : {value['name']} {value['shape']}")
    if payload.get("probe"):
        print(f"原生输出  : {payload['probe']}")
    if payload.get("dataset"):
        print(f"原生数据集: {payload['dataset']}")
    if not payload["framework"].get("distributable", True):
        print("\n⚠ 该框架许可证具有传染性（AGPL-3.0）：产物仅限内部评测，不得进入发行包。")
    return 0


def main(argv=None):
    args = _parse_args(argv)
    if args.list_layouts:
        return _print_layouts(args.json)
    if args.list_frameworks:
        return _print_frameworks(args.json)
    if not args.framework:
        raise SystemExit("请用 --framework 指定适配器（--list-frameworks 查看取值）")
    adapter = registry.lookup(args.framework)
    adapter.configure(framework_path=args.framework_path, exp_file=args.exp_file)

    if args.dataset_only:
        if not args.data:
            raise SystemExit("--dataset-only 需要 --data")
        # 只导出「我们自己的标注」，不需要框架的 Python 包，也没有许可证问题
        card, records = load_dataset(args.data)
        target = Path(args.dataset_output) if args.dataset_output else Path(args.data) / "native"
        summary = adapter.export_dataset(args.data, records, card, target,
                                         args.dataset_format or None)
        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            print(f"{adapter.info.title}（{adapter.info.license}）原生数据集已导出：")
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if not args.output:
        raise SystemExit("导出需要 --output 指定输出目录")
    source = args.onnx or args.weights
    mode = _pick_mode(args, adapter, source)
    transpose = _transpose_option(args.transpose)
    common = dict(image_size=args.imgsz, max_boxes=args.max_boxes, layout=args.layout or None,
                  model_id=args.model_id, version=args.version, allow_copyleft=args.allow_copyleft,
                  channel_repeat=args.channel_repeat, input_scale=args.input_scale,
                  mean=args.input_mean, std=args.input_std, data_root=args.data,
                  notes=args.notes)

    if mode == "onnx":
        if not source:
            raise SystemExit("onnx 模式需要 --onnx 或 --weights 指向原生 ONNX")
        if not args.data and not args.imgsz:
            raise SystemExit("未给 --data 时必须显式给 --imgsz")
        result = adapter.export_onnx(source=source, output=args.output, weights=args.weights,
                                     transpose=transpose, **common)
    else:
        if not args.data:
            raise SystemExit("torch 模式需要 --data（尺寸与标签都以数据集契约为准）")
        if transpose is not None:
            raise SystemExit("--transpose 只对 onnx 模式有意义")
        result = adapter.export(weights=args.weights, output=args.output, opset=args.opset,
                                device="cpu", probe=args.probe, **common)

    return _report(result, args.json)


if __name__ == "__main__":  # pragma: no cover - CLI 入口
    raise SystemExit(main())
