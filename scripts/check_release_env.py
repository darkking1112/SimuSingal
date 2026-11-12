#!/usr/bin/env python3
"""发布环境自检：构建机与目标机共用。

构建机（默认）检查编译器、构建依赖和运行时依赖：

    python scripts/check_release_env.py --project analysis --stage build --full

目标机只检查运行时依赖：

    python share/check_release_env.py --stage runtime

退出码：0 = 全部通过（允许警告）；1 = 存在失败项。
`--install` 只补装缺失的 pip 包，不自动执行 apt/winget 等系统命令。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "packaging/release.json"
PREFIX = {"ok": "[OK]  ", "warn": "[WARN]", "fail": "[FAIL]"}
REPORT_SCHEMA = "simusignal_release_env_v1"

REQUIREMENT_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)\s*"
    r"(?:(?P<op>===|==|>=|<=|~=|>|<)\s*(?P<version>[^;,\s]+))?\s*"
    r"(?:;\s*(?P<marker>.+))?$"
)
MARKER_RE = re.compile(
    r"""^\s*(?P<name>python_version|sys_platform)\s*(?P<op>==|!=|>=|<=|>|<)\s*["'](?P<value>[^"']+)["']\s*$"""
)
VERSION_RE = re.compile(r"(\d+(?:\.\d+)*)")


def version_tuple(text):
    """把版本字符串转成可比较的数字元组，忽略预发布与本地后缀。"""
    match = VERSION_RE.search(text or "")
    return tuple(int(part) for part in match.group(1).split(".")) if match else ()


def compare_versions(left, right):
    """比较两个版本；长度不同时按 0 补齐。"""
    a, b = version_tuple(left), version_tuple(right)
    length = max(len(a), len(b))
    a += (0,) * (length - len(a))
    b += (0,) * (length - len(b))
    return (a > b) - (a < b)


def satisfies(installed, op, required):
    """按 pip 的常用比较符判断已安装版本是否满足要求。"""
    if not op:
        return True
    if op in ("==", "==="):
        return installed == required or compare_versions(installed, required) == 0
    if op == "~=":
        parts = version_tuple(required)
        if len(parts) < 2 or compare_versions(installed, required) < 0:
            return False
        upper = parts[:-2] + (parts[-2] + 1,)
        return compare_versions(installed, ".".join(str(part) for part in upper)) < 0
    result = compare_versions(installed, required)
    return {"<": result < 0, "<=": result <= 0, ">": result > 0, ">=": result >= 0}.get(op, False)


def parse_requirement(line):
    """解析一行 requirements；注释和空行返回 None。"""
    text = line.split("#", 1)[0].strip()
    if not text:
        return None
    match = REQUIREMENT_RE.match(text)
    if not match:
        return None
    return {
        "raw": text,
        "name": match.group("name"),
        "op": match.group("op") or "",
        "version": match.group("version") or "",
        "marker": (match.group("marker") or "").strip(),
    }


def marker_applies(marker, environment=None):
    """求值本仓库用到的简单环境标记；无法识别的标记按“适用”处理。"""
    if not marker:
        return True
    environment = environment or {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "sys_platform": sys.platform,
    }
    match = MARKER_RE.match(marker)
    if not match:
        return True
    name, op, value = match.group("name"), match.group("op"), match.group("value")
    current = environment.get(name)
    if current is None:
        return True
    if name == "sys_platform":
        return {"==": current == value, "!=": current != value}.get(op, True)
    result = compare_versions(current, value)
    return {"==": result == 0, "!=": result != 0, "<": result < 0,
            "<=": result <= 0, ">": result > 0, ">=": result >= 0}.get(op, True)


def read_requirements(path):
    """读取 requirements 文件，过滤不适用于当前环境的行。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    result = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        requirement = parse_requirement(line)
        if requirement is None:
            raise ValueError(f"{path}:{number} 无法解析：{line.strip()}")
        if marker_applies(requirement["marker"]):
            result.append(requirement)
    return result


def requirement_spec(requirement):
    """还原 pip 可执行的包规格（含环境标记）。"""
    spec = requirement["name"] + (f"{requirement['op']}{requirement['version']}" if requirement["op"] else "")
    if requirement["marker"]:
        spec += f"; {requirement['marker']}"
    return spec


def installed_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def platform_key():
    """返回 analysis 项目的平台键：linux / win32。"""
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "win32":
        return "win32"
    return sys.platform


def machine_key():
    machine = platform.machine().lower()
    return {"amd64": "x86_64", "x64": "x86_64", "arm64": "aarch64"}.get(machine, machine)


def host_platform():
    return f"{platform_key()}-{machine_key()}"


def record(records, status, category, message, detail=None):
    records.append({"status": status, "category": category, "message": message, "detail": detail or ""})


def load_config(path, project):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if project not in config:
        raise KeyError(f"发布配置中没有项目：{project}")
    return config[project]


def check_platform(records, target):
    host = platform_key()
    if target != host:
        record(records, "fail", "平台", f"目标 {target} 与宿主 {host} 不一致",
               "Python 扩展不能交叉编译，请在目标系统上构建")
        return False
    system = platform.system() or host
    record(records, "ok", "平台", f"{host}（{system} {platform.release()}）",
           f"machine={machine_key()} bits={8 * struct_calcsize()}")
    return True


def struct_calcsize():
    import struct
    return struct.calcsize("P")


def check_python(records, expected):
    implementation = sys.implementation.name
    actual = f"{sys.version_info.major}.{sys.version_info.minor}"
    abi = sysconfig.get_config_var("SOABI") or sysconfig.get_platform()
    if implementation != "cpython":
        record(records, "fail", "Python", f"需要 CPython，当前是 {implementation}")
        return
    if actual != expected:
        record(records, "fail", "Python", f"需要 {expected}.x，当前是 {sys.version.split()[0]}")
        return
    if struct_calcsize() != 8:
        record(records, "fail", "Python", "需要 64 位解释器")
        return
    record(records, "ok", "Python", f"{sys.version.split()[0]} {implementation} 64 位", abi)


def check_packages(records, requirements, stage):
    missing = []
    for requirement in requirements:
        name = requirement["name"]
        installed = installed_version(name)
        if installed is None:
            record(records, "fail", stage, f"{name} 未安装", requirement_spec(requirement))
            missing.append(requirement)
        elif requirement["op"] and not satisfies(installed, requirement["op"], requirement["version"]):
            record(records, "fail", stage,
                   f"{name}={installed} 不满足 {requirement['op']}{requirement['version']}",
                   requirement_spec(requirement))
            missing.append(requirement)
        else:
            record(records, "ok", stage, f"{name}={installed}")
    return missing


def check_compiler(records, full):
    system = platform_key()
    if system == "linux":
        tools = {"gcc": "gcc", "g++": "g++", "make": "make", "ld": "ld"}
        missing = [name for name, executable in tools.items() if shutil.which(executable) is None]
        if missing:
            record(records, "fail", "编译器", "缺少：" + ", ".join(missing),
                   "Debian/Ubuntu: sudo apt-get install build-essential python3-dev")
            return
        versions = []
        for name, executable in (("gcc", "gcc"), ("g++", "g++")):
            result = subprocess.run([executable, "--version"], capture_output=True, text=True)
            versions.append(result.stdout.splitlines()[0] if result.stdout else name)
        record(records, "ok", "编译器", "; ".join(versions))
        if not full:
            return
        with tempfile.TemporaryDirectory(prefix="simusignal-cc-") as folder:
            source = Path(folder) / "probe.c"
            output = Path(folder) / "libprobe.so"
            source.write_text("int simusignal_env_probe(void) { return 42; }\n", encoding="utf-8")
            result = subprocess.run([shutil.which("gcc"), "-shared", "-fPIC", str(source), "-o", str(output)],
                                    capture_output=True, text=True)
            if result.returncode != 0 or not output.is_file():
                record(records, "fail", "编译器", "共享库编译探针失败", result.stderr.strip()[:500])
            else:
                record(records, "ok", "编译器", "共享库编译探针通过")
        return
    if system == "win32":
        compiler = shutil.which("cl")
        if compiler is None:
            vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / \
                "Microsoft Visual Studio/Installer/vswhere.exe"
            hint = "安装 Visual Studio 2022 Build Tools（C++ 工作负载 + Windows SDK 10）"
            if vswhere.is_file():
                hint += f"；已检测到 vswhere：{vswhere}"
            record(records, "fail", "编译器", "未找到 cl.exe", hint)
            return
        record(records, "ok", "编译器", f"{compiler}")
        if full:
            with tempfile.TemporaryDirectory(prefix="simusignal-cc-") as folder:
                source = Path(folder) / "probe.c"
                source.write_text("int simusignal_env_probe(void) { return 42; }\n", encoding="utf-8")
                result = subprocess.run([compiler, "/nologo", "/c", str(source)],
                                        capture_output=True, text=True, cwd=folder)
                if result.returncode != 0:
                    record(records, "fail", "编译器", "MSVC 编译探针失败", result.stdout.strip()[:500])
                else:
                    record(records, "ok", "编译器", "MSVC 编译探针通过")
        return
    record(records, "fail", "编译器", f"不支持的系统：{system}")


def check_system_libraries(records):
    if platform_key() != "linux":
        return
    try:
        output = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True).stdout
    except OSError:
        record(records, "warn", "系统库", "无法执行 ldconfig，跳过 GUI 运行库检查")
        return
    for library, hint in (("libGL.so.1", "libgl1"),
                          ("libxcb-cursor.so.0", "libxcb-cursor0")):
        if library in output:
            record(records, "ok", "系统库", library)
        else:
            record(records, "warn", "系统库", f"未找到 {library}",
                   f"GUI 启动可能需要安装 {hint}；无界面运行可忽略")


def check_gui(records):
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    code = ("import PySide6, pyqtgraph\n"
            "print(PySide6.__version__, pyqtgraph.__version__)")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    if result.returncode != 0:
        record(records, "fail", "GUI", "PySide6 / pyqtgraph 导入失败", result.stderr.strip()[:500])
    else:
        record(records, "ok", "GUI", f"offscreen 导入通过（{result.stdout.strip()}）")


def check_source(records, config):
    manifest = ROOT / "packaging/numeric_core.json"
    try:
        names = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        record(records, "fail", "源码", f"无法读取 {manifest.relative_to(ROOT)}", str(exc))
        return
    missing = []
    for module in names:
        path = ROOT / "src" / (module.replace(".", "/") + ".py")
        if not path.is_file():
            missing.append(str(path.relative_to(ROOT)))
    if missing:
        record(records, "fail", "源码", "核心模块源码缺失", ", ".join(missing))
    else:
        record(records, "ok", "源码", f"{len(names)} 个核心模块源码齐全")
    training = ROOT / "packaging/training_core.json"
    try:
        training_config = json.loads(training.read_text(encoding="utf-8"))
        modules = training_config.get("modules", [])
        missing = [item["source"] for item in modules
                   if not (ROOT / item["source"]).is_file()]
        package_count = 0
        for package in training_config.get("packages", []):
            source_root = ROOT / package["source_root"]
            if not source_root.is_dir():
                missing.append(package["source_root"])
                continue
            exclude = set(package.get("exclude", []))
            package_count += sum(1 for path in source_root.glob("*.py")
                                 if path.name not in exclude)
        if missing:
            record(records, "fail", "源码", "训练模块源码缺失", ", ".join(missing))
        else:
            record(records, "ok", "源码",
                   f"{len(modules) + package_count} 个训练模块源文件齐全")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        record(records, "fail", "源码", "无法读取 packaging/training_core.json", str(exc))
    model = ROOT / "src/signal_analysis/algorithms/amc/amc_default.json"
    if model.is_file():
        record(records, "ok", "源码", "内置 AMC 模型存在")
    else:
        record(records, "fail", "源码", "缺少内置 AMC 模型", str(model.relative_to(ROOT)))
    app = ROOT / "apps" / config["app_dir"]
    if (app / "main.py").is_file() and (app / "pyproject.toml").is_file():
        record(records, "ok", "源码", f"{config['app_dir']} 入口与元数据存在")
    else:
        record(records, "fail", "源码", f"{config['app_dir']} 入口或 pyproject.toml 缺失")


def check_git(records, strict):
    def run(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    commit = run("rev-parse", "--short", "HEAD")
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    status = run("status", "--porcelain")
    if commit.returncode != 0:
        record(records, "warn", "Git", "无法读取提交信息")
        return
    detail = f"branch={branch.stdout.strip()} commit={commit.stdout.strip()}"
    if status.stdout.strip():
        status_name = "fail" if strict else "warn"
        record(records, status_name, "Git", "工作区有未提交改动", detail)
    else:
        record(records, "ok", "Git", "工作区干净", detail)


def check_disk(records, output, minimum_gb):
    output = Path(output)
    probe = output
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError as exc:
        record(records, "fail", "磁盘", f"无法读取 {probe} 的磁盘空间", str(exc))
        return
    free_gb = usage.free / (1024 ** 3)
    if free_gb < minimum_gb:
        record(records, "fail", "磁盘", f"剩余 {free_gb:.1f} GiB，需要至少 {minimum_gb} GiB")
    else:
        record(records, "ok", "磁盘", f"剩余 {free_gb:.1f} GiB（输出 {output}）")
    try:
        output.mkdir(parents=True, exist_ok=True)
        probe_file = output / ".simusignal-env-check"
        probe_file.write_text("ok", encoding="utf-8")
        probe_file.unlink()
        record(records, "ok", "磁盘", "输出目录可写")
    except OSError as exc:
        record(records, "fail", "磁盘", "输出目录不可写", str(exc))


def check_optional(records):
    version = installed_version("onnxruntime")
    if version:
        record(records, "ok", "可选", f"onnxruntime={version}")
    else:
        record(records, "warn", "可选", "onnxruntime 未安装（仅 AI 检测需要）")


def print_report(records, project, stage, target):
    print(f"发布环境自检：{project} / {stage} / {target}")
    for item in records:
        line = f"{PREFIX[item['status']]} {item['category']:<6} {item['message']}"
        print(line)
        if item["detail"]:
            print(f"       {item['detail']}")
    counts = {status: sum(1 for item in records if item["status"] == status)
              for status in ("ok", "warn", "fail")}
    print(f"结果：{counts['ok']} 项通过，{counts['warn']} 项警告，{counts['fail']} 项失败")


def freeze_requirements(path, requirements):
    """把已安装版本写成新的 requirements 文件，保留原环境标记。"""
    lines = [f"# Frozen by scripts/check_release_env.py at {datetime.now(timezone.utc).isoformat()}"]
    for requirement in requirements:
        version = installed_version(requirement["name"])
        if version is None:
            lines.append(f"# 缺失：{requirement_spec(requirement)}")
            continue
        line = f"{requirement['name']}=={version}"
        if requirement["marker"]:
            line += f"; {requirement['marker']}"
        lines.append(line)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写入 {path}")


def install_missing(requirements):
    missing = [requirement for requirement in requirements
               if installed_version(requirement["name"]) is None
               or (requirement["op"] and not satisfies(
                   installed_version(requirement["name"]), requirement["op"], requirement["version"]))]
    if not missing:
        return []
    specs = [requirement_spec(requirement) for requirement in missing]
    print("将安装：" + " ".join(specs))
    subprocess.run([sys.executable, "-m", "pip", "install", *specs], check=True)
    return specs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="analysis")
    parser.add_argument("--stage", choices=("build", "runtime"), default="build")
    parser.add_argument("--target", default="auto", help="auto / linux / win32（也接受 windows）")
    parser.add_argument("--python", default=None, help="要求的 CPython 次版本，默认取 packaging/release.json")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(ROOT / "dist/release"), help="发布输出目录")
    parser.add_argument("--json", type=Path, help="把自检结果写入 JSON")
    parser.add_argument("--freeze", type=Path, help="把已安装版本写成 requirements 文件")
    parser.add_argument("--install", action="store_true", help="自动补装缺失的 pip 包")
    parser.add_argument("--full", action="store_true", help="执行编译探针")
    parser.add_argument("--strict", action="store_true", help="把警告视为失败")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, args.project)
    except (OSError, KeyError, ValueError) as exc:
        print(f"无法读取发布配置：{exc}", file=sys.stderr)
        return 2
    target_platform = platform_key() if args.target == "auto" else (
        "win32" if args.target == "windows" else args.target)
    if target_platform not in ("linux", "win32"):
        print(f"不支持的目标平台：{args.target}", file=sys.stderr)
        return 2
    target = f"{target_platform}-{machine_key()}"
    expected_python = args.python or config.get("python", "3.12")
    platform_config = config["platforms"].get(target_platform, {})
    build_path = ROOT / config["requirements"]["build"].format(platform=target_platform)
    runtime_path = ROOT / config["requirements"]["runtime"].format(platform=target_platform)
    build_requirements = read_requirements(build_path) if args.stage == "build" else []
    runtime_requirements = read_requirements(runtime_path)

    if args.install:
        try:
            install_missing(build_requirements + runtime_requirements)
        except subprocess.CalledProcessError as exc:
            print(f"依赖安装失败：{exc}", file=sys.stderr)
            return 1

    records = []
    if not check_platform(records, target_platform):
        print_report(records, args.project, args.stage, target)
        return 1
    check_python(records, expected_python)
    if args.stage == "build":
        check_compiler(records, args.full)
        check_source(records, config)
    check_packages(records, build_requirements, "构建依赖")
    check_packages(records, runtime_requirements, "运行时依赖")
    check_gui(records)
    check_system_libraries(records)
    check_disk(records, args.output, config.get("min_free_gb", 3))
    check_optional(records)
    if args.stage == "build":
        check_git(records, args.strict)

    print_report(records, args.project, args.stage, target)
    failed = any(item["status"] == "fail" for item in records)
    warned = any(item["status"] == "warn" for item in records)
    report = {
        "schema": REPORT_SCHEMA,
        "project": args.project,
        "stage": args.stage,
        "target": target,
        "python": sys.version.split()[0],
        "abi": sysconfig.get_config_var("SOABI") or sysconfig.get_platform(),
        "ok": not failed and not (args.strict and warned),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checks": records,
        "requirements": {"build": str(build_path), "runtime": str(runtime_path)},
        "platform": platform_config,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.freeze:
        freeze_requirements(args.freeze, build_requirements + runtime_requirements)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
