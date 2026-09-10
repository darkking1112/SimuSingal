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
    native = commands.add_parser("native", help="运行显式选择的原生复制演示插件")
    native.add_argument("asset_id")
    native.add_argument("manifest")
    manifest = commands.add_parser("plugin-manifest", help="为复制示例生成当前平台清单")
    manifest.add_argument("library")
    manifest.add_argument("output")
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
        request = {"workspace": str(workspace.root), "action": args.command}
        if args.command == "list":
            result = workspace.list_assets()
        elif args.command == "export":
            from common.reports import export_report
            result = {"path": export_report(workspace.get_run(args.run_id), args.path)}
        elif args.command == "plugin-manifest":
            from .plugins import create_demo_manifest
            result = create_demo_manifest(args.library, args.output)
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
