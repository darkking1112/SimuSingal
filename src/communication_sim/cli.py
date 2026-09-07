"""Standalone simulation command line and worker entry."""
import argparse
import json
import sys
from . import __version__

def main(argv=None):
    parser = argparse.ArgumentParser(description="CommunicationSim 通用事件仿真实验")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--workspace", default="workspace_data/simulation")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("gui", help="启动仿真桌面")
    simulation = commands.add_parser("simulate", help="通用消息中继队列演示")
    simulation.add_argument("--messages", type=int, default=12)
    simulation.add_argument("--duration", type=float, default=3.0)
    commands.add_parser("list", help="列出仿真实验")
    export = commands.add_parser("export", help="导出仿真报告")
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
                print(f"桌面依赖不可用：{exc}", file=sys.stderr)
                return 2
            return launch(args.workspace)
        from .storage import Workspace
        from .tasks import run_job
        workspace = Workspace(args.workspace)
        if args.command == "list":
            result = workspace.list_runs()
        elif args.command == "export":
            from common.reports import export_report
            result = {"path": export_report(workspace.get_run(args.run_id), args.path)}
        else:
            result = run_job({"workspace": str(workspace.root), "action": "simulate",
                              "scenario": {"messages": args.messages, "duration_s": args.duration}})
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
