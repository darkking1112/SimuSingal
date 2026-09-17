#!/usr/bin/env python3
"""训练环境自检与补装：对应 GUI「模型训练」页所需的外部 Python 环境。

用**训练环境自己的解释器**运行本脚本（就是 GUI 里「训练环境 Python」所填的那个）：

    Windows:      .venv\\Scripts\\python.exe scripts\\check_training_env.py
    Linux/macOS:  .venv/bin/python scripts/check_training_env.py

`--install` 会补装缺失的、可安全自动安装的包（onnx / onnxruntime / ultralytics /
RT-DETR 依赖 / torchsig）；torch 与 torchvision 不自动安装（需要按 CPU / CUDA
选择具体版本），脚本只给出命令提示。

退出码：0 = 全部就绪；1 = 存在缺失项（含版本不符与源码缺失）。
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: 训练相关包：(显示名, pip 安装名, 组件, 要求)
#: 要求为 None（只查存在）/ {"min": (x, y)} / {"exact": "x.y.z"}
PACKAGES = (
    ("torch", "torch", "common", {"min": (2, 2)}),
    ("torchvision", "torchvision", "common", {"min": (0, 17)}),
    ("onnx", "onnx", "common", {"min": (1, 16)}),
    ("onnxruntime", "onnxruntime", "common", {"min": (1, 17)}),
    ("ultralytics", "ultralytics", "yolo", None),
    ("PyYAML", "PyYAML", "rtdetr", None),
    ("scipy", "scipy", "rtdetr", None),
    ("faster-coco-eval", "faster-coco-eval", "rtdetr", None),
    ("pycocotools", "pycocotools", "rtdetr", None),
    ("tensorboard", "tensorboard", "rtdetr", None),
    ("torchsig", "torchsig==2.2.0", "torchsig", {"exact": "2.2.0"}),
)

#: 组件分组标题（检查结果按此顺序打印）
GROUPS = (
    ("common", "通用依赖（检测与 IQ 训练共用）"),
    ("yolo", "YOLO26s（Ultralytics）"),
    ("rtdetr", "RT-DETR（lyuwenyu v2）"),
    ("torchsig", "TorchSig 数据生成"),
)

#: 可自动补装的 pip 规格；仅对缺失项执行。torch / torchvision 不在内。
PIP_SPECS = {
    "onnx": "onnx>=1.16,<2",
    "onnxruntime": "onnxruntime>=1.17,<2",
    "ultralytics": "ultralytics",
    "PyYAML": "PyYAML",
    "scipy": "scipy",
    "faster-coco-eval": "faster-coco-eval",
    "pycocotools": "pycocotools",
    "tensorboard": "tensorboard",
    "torchsig": "torchsig==2.2.0",
}

#: 补装顺序：先公共与轻量项，再装体积较大的 ultralytics / torchsig
INSTALL_ORDER = ("onnx", "onnxruntime", "PyYAML", "scipy", "faster-coco-eval",
                 "pycocotools", "tensorboard", "ultralytics", "torchsig")

#: 缺失时跳过自动安装（依赖 torch，需要用户先按 CPU/CUDA 选择装好）
NEEDS_TORCH = ("ultralytics", "torchsig")

#: GUI 训练流程依赖的源码文件（相对仓库根）
SOURCE_FILES = (
    "training/desktop_worker.py",
    "training/train_yolox.py",
    "training/train_iq.py",
    "training/rtdetr_desktop.py",
    "training/build_dataset.py",
    "training/build_torchsig.py",
    "training/ingest_torchsig.py",
    "training/verify_onnx.py",
    "training/verify_iq.py",
    "training/detectors/dataset.py",
    "training/detectors/labels.py",
)


def release(text):
    """版本字符串 -> 可比较的数字组（忽略 +cpu 等本地后缀与预发布标签）。"""
    match = re.match(r"\s*(\d+(?:\.\d+)*)", text or "")
    return tuple(int(part) for part in match.group(1).split(".")) if match else ()


def installed(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def check_packages():
    """返回 {显示名: (状态, 版本)}；状态为 ok / old / missing。"""
    result = {}
    for display, _, _, requirement in PACKAGES:
        version = installed(display)
        if version is None:
            result[display] = ("missing", "-")
            continue
        if requirement and "min" in requirement:
            if release(version) < tuple(requirement["min"]):
                result[display] = ("old", version)
                continue
        if requirement and "exact" in requirement and version != requirement["exact"]:
            result[display] = ("old", version)
            continue
        result[display] = ("ok", version)
    return result


def package_spec(name):
    for display, pip_name, _, _ in PACKAGES:
        if display == name:
            return pip_name
    return name


def print_packages(results):
    for key, title in GROUPS:
        print(f"\n[{title}]")
        for display, _, group, requirement in PACKAGES:
            if group != key:
                continue
            state, version = results[display]
            if state == "ok":
                print(f"  [OK]   {display} {version}")
            elif state == "missing":
                print(f"  [缺失] {display}（pip 包名：{package_spec(display)}）")
            else:
                want = requirement["min"] if "min" in requirement else requirement["exact"]
                want = ".".join(map(str, want)) if isinstance(want, tuple) else want
                print(f"  [过低] {display} {version}（需要 {'≥ ' if 'min' in requirement else ''}{want}）")


def print_cuda(results):
    print("\n[CUDA]")
    if results.get("torch", ("missing",))[0] == "missing":
        print("  [--]   未安装 torch，跳过 CUDA 检测")
        return
    try:
        import torch  # noqa: PLC0415 - 只在确实装了 torch 时导入，且要拿运行时状态
    except Exception as exc:  # noqa: BLE001 - torch 损坏也要给出可读信息
        print(f"  [缺失] import torch 失败：{exc}")
        return
    if torch.cuda.is_available():
        print(f"  [OK]   CUDA 可用：{torch.version.cuda} · {torch.cuda.get_device_name(0)}")
        print("         训练设备可直接选 cuda")
    else:
        print("  [--]   未检测到可用 CUDA（训练设备请选 cpu；或用 CUDA 版 torch 重装环境）")


def print_sources():
    print("\n[训练源码]")
    missing = [name for name in SOURCE_FILES if not (REPO / name).is_file()]
    if not missing:
        print(f"  [OK]   {REPO / 'training'} 下 {len(SOURCE_FILES)} 个入口文件齐全")
    else:
        for name in missing:
            print(f"  [缺失] {name}")
    return missing


def print_rtdetr(args):
    if not args.rtdetr and not args.rtdetr_config:
        return []
    print("\n[RT-DETR 路径（命令行传入）]")
    problems = []
    if args.rtdetr:
        directory = Path(args.rtdetr)
        marker = directory / "src/core/yaml_config.py"
        if marker.is_file():
            print(f"  [OK]   {directory}（src/core/yaml_config.py 存在）")
        else:
            problems.append("rtdetr 目录")
            print(f"  [缺失] {directory} 下找不到 src/core/yaml_config.py（rtdetrv2_pytorch 目录选错了？）")
    if args.rtdetr_config:
        config = Path(args.rtdetr_config)
        if config.is_file():
            print(f"  [OK]   {config}")
        else:
            problems.append("rtdetr 模型 YAML")
            print(f"  [缺失] {config}")
    return problems


def group_ready(results, group):
    return all(results[display][0] == "ok"
               for display, _, item_group, _ in PACKAGES if item_group == group)


def print_summary(results, source_missing):
    print("\n=== 结论 ===")
    reasons = []
    if group_ready(results, "common") and not source_missing:
        print("  IQ 分类训练（CNN / TCN）: 就绪")
    else:
        bad = [d for d, _, g, _ in PACKAGES if g == "common" and results[d][0] != "ok"]
        reasons.extend(bad)
        print(f"  IQ 分类训练（CNN / TCN）: 缺少 {', '.join(bad) or '训练源码'}")
    if group_ready(results, "common") and group_ready(results, "yolo"):
        print("  YOLO26s 检测训练: 就绪")
    else:
        bad = [d for d, _, g, _ in PACKAGES if g == "yolo" and results[d][0] != "ok"]
        reasons.extend(bad)
        print(f"  YOLO26s 检测训练: 缺 {', '.join(bad)}"
              + ("（另需一个训练数据集）" if group_ready(results, "yolo") else ""))
    if group_ready(results, "common") and group_ready(results, "rtdetr"):
        print("  RT-DETR 检测训练: 依赖就绪（还需在界面选择 rtdetrv2_pytorch 目录与模型 YAML）")
    else:
        bad = [d for d, _, g, _ in PACKAGES if g == "rtdetr" and results[d][0] != "ok"]
        reasons.extend(bad)
        print(f"  RT-DETR 检测训练: 缺 {', '.join(bad)}；另需 rtdetrv2_pytorch 目录与模型 YAML")
    if group_ready(results, "common") and group_ready(results, "torchsig"):
        print("  TorchSig 数据生成: 就绪")
    else:
        bad = [d for d, _, g, _ in PACKAGES if g == "torchsig" and results[d][0] != "ok"]
        reasons.extend(bad)
        print(f"  TorchSig 数据生成: 缺 {', '.join(bad) or 'torch / torchvision'}")
    return reasons


def print_guidance(results):
    print("\n=== 训练工作台填写指引 ===")
    print(f"  训练源码根目录 : {REPO}")
    print(f"  训练环境 Python: {sys.executable}")
    print("  YOLO26s        : 模型选 yolo26s；权重留空会自动下载 yolo26s.pt"
          "（需要能访问 GitHub，直连不畅时先配置代理）")
    print("  RT-DETR        : 选 lyuwenyu/RT-DETR 的 rtdetrv2_pytorch 目录与模型 YAML，例如：")
    print("                   git clone --depth 1 https://github.com/lyuwenyu/RT-DETR.git D:\\RT-DETR")
    print("  CPU 环境       : 训练设备选 cpu；建议先用 200~500 条数据跑通链路再扩大规模")


def pip_install(spec):
    print(f"  $ python -m pip install {spec}")
    result = subprocess.run([sys.executable, "-m", "pip", "install",
                             "--disable-pip-version-check", "--no-input", spec])
    return result.returncode == 0


def install_missing(results):
    """补装缺失项；返回 (成功列表, 跳过列表, 失败列表)。"""
    torch_missing = results.get("torch", ("missing",))[0] != "ok"
    targets = [name for name in INSTALL_ORDER
               if name in PIP_SPECS and results.get(name, ("missing",))[0] != "ok"]
    done, skipped, failed = [], [], []
    if torch_missing:
        for name in list(targets):
            if name in NEEDS_TORCH:
                targets.remove(name)
                skipped.append(name)
        print("  [提示] torch 缺失或版本不符：请先在仓库根执行 "
              'python -m pip install -e ".[train,ml]"（按 CPU/CUDA 需要选择 torch 版本）')
    if not targets:
        print("  没有可自动补装的缺失项。")
        return done, skipped, failed
    for name in targets:
        if pip_install(PIP_SPECS[name]):
            done.append(name)
        else:
            failed.append(name)
    return done, skipped, failed


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="SimuSignal 训练环境自检 / 补装（在训练环境的解释器里运行）")
    parser.add_argument("--install", action="store_true",
                        help="补装缺失的可自动安装包（torch / torchvision 只提示不安装）")
    parser.add_argument("--rtdetr", metavar="DIR",
                        help="顺带检查 rtdetrv2_pytorch 目录（GUI 中要填的字段）")
    parser.add_argument("--rtdetr-config", metavar="YAML",
                        help="顺带检查 RT-DETRv2 模型 YAML（GUI 中要填的字段）")
    args = parser.parse_args(argv)

    print("=== SimuSignal 训练环境检查 ===")
    print(f"解释器 : {sys.executable}")
    print(f"版本   : {sys.version.split()[0]}（{sys.platform}）")
    print(f"仓库   : {REPO}")
    if not (REPO / "training/desktop_worker.py").is_file():
        print("  [警告] 没找到 training/desktop_worker.py：本脚本需放在仓库 scripts/ 目录下使用")

    results = check_packages()
    print_packages(results)
    print_cuda(results)
    source_missing = print_sources()
    rtdetr_problems = print_rtdetr(args)

    if args.install:
        print("\n=== 补装缺失项 ===")
        done, skipped, failed = install_missing(results)
        if done or failed:
            results = check_packages()
        print("\n=== 补装后状态 ===")
        for name in done:
            state, version = results[name]
            print(f"  {name}: {version}" if state == "ok" else f"  {name}: 仍未就绪（{version}）")
        for name in skipped:
            print(f"  {name}: 已跳过（需要先安装 torch）")
        for name in failed:
            print(f"  {name}: 安装失败，请手动重试：python -m pip install {PIP_SPECS[name]}")

    reasons = print_summary(results, source_missing)
    reasons.extend(rtdetr_problems)
    print_guidance(results)

    problems = [name for name, (state, _) in results.items() if state != "ok"]
    if problems or source_missing:
        print(f"\n结果：存在缺失（{len(problems)} 个包 / {len(source_missing)} 个源码文件）"
              "；补装可运行：python scripts/check_training_env.py --install")
        return 1
    print("\n结果：全部就绪。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
