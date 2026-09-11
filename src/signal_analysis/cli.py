"""Command line and packaged-application entry point."""

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(description="SignalAnalysis 电磁信号离线分析")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--workspace", default="workspace_data/analysis", help="本地工作目录")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("gui", help="启动桌面工作台")
    demo = commands.add_parser("demo", help="生成数学双音数据")
    demo.add_argument("--count", type=int, default=8192)
    demo.add_argument("--sample-rate", type=float, default=48000)
    imp = commands.add_parser("import", help="导入 NPY、CSV、交织 IQ 或 SigMF 双文件")
    imp.add_argument("path")
    imp.add_argument("--sample-rate", type=float, help="非 SigMF 必填；SigMF 自动读取，显式指定时须一致")
    imp.add_argument("--binary-dtype", choices=("int16", "float32"),
                     help="交织 IQ 二进制（.bin/.raw/.iq）的数据类型，必须显式指定")
    imp.add_argument("--endian", choices=("little", "big"), default="little",
                     help="交织 IQ 二进制的字节序，默认 little")
    generate = commands.add_parser("generate", help="按 JSON 规格生成测试用 IQ 信号")
    generate.add_argument("spec", help="JSON 规格文件：sample_rate/duration/seed/noise/signals/name/export")
    analysis = commands.add_parser("analyze", help="通用统计和图形计算")
    analysis.add_argument("asset_id")
    analysis.add_argument("--nfft", type=int, default=256)
    detect = commands.add_parser("detect", help="信号检测与参数估计（能量检测基线）")
    detect.add_argument("asset_id")
    detect.add_argument("--nfft", type=int, default=512, help="STFT 点数，16～4096，默认 512")
    detect.add_argument("--threshold-db", type=float, help="检测门限，高于本底该值（dB），默认 3")
    detect.add_argument("--band-threshold-db", type=float,
                        help="带宽测量门限（dB），默认取检测门限的一半")
    detect.add_argument("--min-bandwidth", type=float, help="最小占用带宽（Hz），默认 3 个频点")
    detect.add_argument("--min-duration", type=float, help="最小持续时间（s），默认 0")
    detect.add_argument("--max-detections", type=int, help="最多保留的目标数，1～256，默认 32")
    detect.add_argument("--merge-bins", type=int, help="形态学闭运算半径（频点），默认按 FFT 点数推导")
    ml_detect = commands.add_parser("ml-detect", help="AI 检测（ONNX 模型清单 + 时频图推理）")
    ml_detect.add_argument("asset_id")
    ml_detect.add_argument("manifest", help="模型清单 JSON（含 .onnx 相对路径与 SHA-256）")
    ml_detect.add_argument("--nfft", type=int, help="时频图 STFT 点数，默认取清单声明值")
    ml_detect.add_argument("--threshold-db", type=float, help="检测门限（dB），同时用于传统基线")
    ml_detect.add_argument("--score-threshold", type=float, help="模型置信度阈值，默认 0.25")
    ml_detect.add_argument("--iou-threshold", type=float, help="重叠框去重 IoU 阈值，默认 0.5")
    ml_detect.add_argument("--max-detections", type=int, help="最多保留的目标数，1～256，默认 32")
    ml_detect.add_argument("--min-bandwidth", type=float, help="最小占用带宽（Hz），默认 0")
    ml_detect.add_argument("--min-duration", type=float, help="最小持续时间（s），默认 0")
    ml_detect.add_argument("--threads", type=int, help="ONNX Runtime 计算线程数")
    ml_detect.add_argument("--no-baseline", action="store_true", help="不附带传统能量检测基线")
    native = commands.add_parser("native", help="运行显式选择的原生复制演示插件")
    native.add_argument("asset_id")
    native.add_argument("manifest")
    amc = commands.add_parser("amc-classify",
                              help="A09 六类调制识别（确定性特征 + 线性基线或 ONNX 分类器）")
    amc.add_argument("asset_id")
    amc.add_argument("--model", help="模型 JSON 或 ONNX 清单；默认使用内置线性基线")
    amc.add_argument("--offset-hz", type=float, help="分析频带中心（Hz），默认 0（基带中心）")
    amc.add_argument("--bandwidth-hz", type=float, help="分析频带带宽（Hz），默认整段采样带宽")
    amc.add_argument("--threads", type=int, help="ONNX Runtime 计算线程数（仅清单模型有效）")
    manifest = commands.add_parser("plugin-manifest", help="为复制示例生成当前平台清单")
    manifest.add_argument("library")
    manifest.add_argument("output")
    model_manifest = commands.add_parser("ml-manifest", help="为已有 .onnx 模型生成检测清单")
    model_manifest.add_argument("model", help=".onnx 模型文件（需与清单同目录）")
    model_manifest.add_argument("output", help="要写入的清单 JSON")
    model_manifest.add_argument("--id", dest="identifier", help="模型标识，默认取文件名")
    model_manifest.add_argument("--version", default="0.1.0")
    model_manifest.add_argument("--image-size", type=int, default=1024, choices=(64, 128, 256, 512, 1024, 2048))
    model_manifest.add_argument("--nfft", type=int, default=512, help="训练时频图使用的 STFT 点数")
    model_manifest.add_argument("--dynamic-range", type=float, default=60.0, help="图像动态范围（dB）")
    model_manifest.add_argument("--opset", type=int, default=0)
    model_manifest.add_argument("--label", action="append", help="类别名，可重复；默认 emitter")
    model_manifest.add_argument("--framework", help="训练框架（如 yolox / rt-detr）")
    model_manifest.add_argument("--license", help="模型与训练代码的许可证（如 Apache-2.0）")
    model_manifest.add_argument("--dataset", help="训练数据来源与许可证说明")
    model_manifest.add_argument("--notes", default="")
    amc_manifest = commands.add_parser("amc-manifest", help="为已有调制识别 ONNX 分类器生成清单")
    amc_manifest.add_argument("model", help=".onnx 分类器文件（需与清单同目录）")
    amc_manifest.add_argument("output", help="要写入的清单 JSON")
    amc_manifest.add_argument("--id", dest="identifier", help="模型标识，默认 amc-linear-default")
    amc_manifest.add_argument("--version", default="0.1.0")
    amc_manifest.add_argument("--opset", type=int, default=17)
    amc_manifest.add_argument("--dataset", help="训练数据来源与许可证说明")
    amc_manifest.add_argument("--notes", default="")
    commands.add_parser("list", help="列出最近数据")
    export = commands.add_parser("export", help="导出 JSON/HTML 报告")
    export.add_argument("run_id")
    export.add_argument("path")
    worker = commands.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("request")
    worker.add_argument("response")
    args = parser.parse_args(argv)
    try:
        if args.command == "worker":
            from .tasks import worker_main
            return worker_main(args.request, args.response)
        if args.command in (None, "gui"):
            try:
                from .gui import launch
            except ImportError as exc:
                print(f"桌面依赖不可用：{exc}。请安装项目的 [gui] 依赖。", file=sys.stderr)
                return 2
            return launch(args.workspace)
        from .storage import Workspace
        from .tasks import run_job
        workspace = Workspace(args.workspace)
        # 命令行用连字符（ml-detect），服务动作名用下划线（ml_detect）
        request = {"workspace": str(workspace.root), "action": args.command.replace("-", "_")}
        if args.command == "list":
            result = workspace.list_assets()
        elif args.command == "export":
            from common.reports import export_report
            result = {"path": export_report(workspace.get_run(args.run_id), args.path)}
        elif args.command == "plugin-manifest":
            from .plugins import create_demo_manifest
            result = create_demo_manifest(args.library, args.output)
        elif args.command == "ml-manifest":
            from .ml import write_model_manifest
            training = {key: value for key, value in (("framework", args.framework),
                                                      ("license", args.license),
                                                      ("dataset", args.dataset))
                        if value}
            result = write_model_manifest(
                args.output, args.model,
                identifier=args.identifier or Path(args.model).stem,
                version=args.version, image_size=args.image_size, opset=args.opset,
                labels=args.label, training=training, notes=args.notes,
                spectrogram_nfft=args.nfft, dynamic_range_db=args.dynamic_range)
        elif args.command == "amc-manifest":
            from .ml import write_amc_manifest
            kwargs = {"identifier": args.identifier or "amc-linear-default",
                      "version": args.version, "opset": args.opset, "notes": args.notes}
            if args.dataset:
                kwargs["training"] = {"dataset": args.dataset}
            manifest, _ = write_amc_manifest(args.output, args.model, **kwargs)
            result = {"manifest": str(Path(args.output)), "contract": manifest["contract"],
                      "sha256": manifest["sha256"], "input": manifest["input"],
                      "output": manifest["output"]}
        else:
            if args.command in ("demo", "import"):
                request["sample_rate"] = args.sample_rate
            if args.command == "demo":
                request["count"] = args.count
            elif args.command == "import":
                request["path"] = args.path
                if args.binary_dtype:
                    request["binary_dtype"] = args.binary_dtype
                    request["endian"] = args.endian
            elif args.command == "generate":
                request.update(json.loads(Path(args.spec).read_text(encoding="utf-8")))
                request["action"] = "generate"
                request["workspace"] = str(workspace.root)
            elif args.command == "analyze":
                request.update(asset_id=args.asset_id, nfft=args.nfft)
            elif args.command == "detect":
                request["asset_id"] = args.asset_id
                config = {"nfft": args.nfft}
                for key, value in (("threshold_db", args.threshold_db),
                                   ("band_threshold_db", args.band_threshold_db),
                                   ("min_bandwidth_hz", args.min_bandwidth),
                                   ("min_duration_s", args.min_duration),
                                   ("max_detections", args.max_detections),
                                   ("merge_bins", args.merge_bins)):
                    if value is not None:
                        config[key] = value
                request["config"] = config
            elif args.command == "ml-detect":
                request.update(asset_id=args.asset_id, manifest=args.manifest)
                request["with_baseline"] = not args.no_baseline
                if args.threads is not None:
                    request["threads"] = args.threads
                config = {}
                for key, value in (("nfft", args.nfft),
                                   ("threshold_db", args.threshold_db),
                                   ("score_threshold", args.score_threshold),
                                   ("iou_threshold", args.iou_threshold),
                                   ("max_detections", args.max_detections),
                                   ("min_bandwidth_hz", args.min_bandwidth),
                                   ("min_duration_s", args.min_duration)):
                    if value is not None:
                        config[key] = value
                request["config"] = config
            elif args.command == "amc-classify":
                request["asset_id"] = args.asset_id
                if args.model:
                    request["model"] = args.model
                if args.threads is not None:
                    request["threads"] = args.threads
                config = {}
                for key, value in (("offset_hz", args.offset_hz),
                                   ("bandwidth_hz", args.bandwidth_hz)):
                    if value is not None:
                        config[key] = value
                request["config"] = config
            elif args.command == "native":
                request.update(asset_id=args.asset_id, manifest=args.manifest)
            result = run_job(request)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
