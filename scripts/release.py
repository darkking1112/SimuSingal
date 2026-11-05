#!/usr/bin/env python3
"""一键发布 signal_analysis：编译核心、校验、打包入口并归档。

用法：

    python scripts/release.py analysis --mode portable
    python scripts/release.py analysis --mode system
    python scripts/release.py analysis --mode both --no-archive

产物写入 ``dist/release``；Linux 为 ``tar.gz``，Windows 为 ``zip``。
Python 扩展不能交叉编译，``--target`` 必须与当前宿主一致。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
from pathlib import Path
from zipfile import ZipFile

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import check_release_env  # noqa: E402

ROOT = SCRIPTS.parent
DEFAULT_CONFIG = ROOT / "packaging/release.json"


class ReleaseError(RuntimeError):
    pass


def log(stage, message):
    print(f"[{stage}] {message}", flush=True)


def package_version(root=ROOT):
    import re
    text = (Path(root) / "src/signal_analysis/__init__.py").read_text(encoding="utf-8")
    match = re.search(r"""__version__\s*=\s*["']([^"']+)["']""", text)
    if not match:
        raise ReleaseError("无法从 src/signal_analysis/__init__.py 读取版本号")
    return match.group(1)


def git_value(*args):
    try:
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    except OSError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def load_config(path=DEFAULT_CONFIG):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def project_config(config, project):
    if project not in config:
        raise ReleaseError(f"发布配置中没有项目：{project}")
    return config[project]


def python_tag():
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def artifact_name(executable, version, platform_label, mode):
    suffix = "" if mode == "portable" else f"-{mode}"
    return f"{executable}-{version}-{python_tag()}-{platform_label}{suffix}"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_member(archive, member):
    digest = hashlib.sha256()
    with archive.open(member) as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def core_manifest(wheel):
    """返回 8 个核心模块的 wheel 成员、文件名与哈希。"""
    names = json.loads(
        (ROOT / "packaging/numeric_core.json").read_text(encoding="utf-8"))
    records = []
    with ZipFile(wheel) as archive:
        members = archive.namelist()
        for module in names:
            stem = module.replace(".", "/")
            matches = [name for name in members
                       if name.startswith(stem + ".") and name.endswith((".so", ".pyd"))]
            if len(matches) != 1:
                raise ReleaseError(f"核心模块产物不唯一：{module}: {matches}")
            member = matches[0]
            records.append({"module": module, "member": member,
                            "file": Path(member).name,
                            "sha256": sha256_member(archive, member)})
    return records


def extract_core(wheel, destination):
    """把核心扩展解到 lib/，便于发布包内直接核对与集成。"""
    with ZipFile(wheel) as archive:
        for record in core_manifest(wheel):
            target = Path(destination) / record["member"]
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(record["member"]) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    return destination


def run(command, cwd=ROOT, env=None):
    printable = " ".join(str(part) for part in command)
    print(f"$ {printable}", flush=True)
    result = subprocess.run([str(part) for part in command], cwd=str(cwd), env=env)
    if result.returncode != 0:
        raise ReleaseError(f"命令失败（退出码 {result.returncode}）：{printable}")
    return result


def build_wheel(work):
    output = Path(work) / "wheels"
    shutil.rmtree(output, ignore_errors=True)
    run([sys.executable, ROOT / "scripts/build_wheels.py", "analysis",
         "--compile-core", "--output", output])
    wheels = sorted(output.glob("*.whl"))
    if len(wheels) != 1:
        raise ReleaseError(f"预期 1 个 wheel，实际 {len(wheels)} 个：{wheels}")
    return wheels[0]


def verify_wheel(wheel, work, skip_tests):
    report = Path(work) / "reports/binary.xml"
    command = [sys.executable, ROOT / "scripts/check_binary.py", wheel,
               "--junitxml", report]
    if skip_tests:
        command.append("--skip-tests")
    run(command)


def build_portable(wheel, work):
    output = Path(work) / "portable"
    run([sys.executable, ROOT / "scripts/build_desktop.py", "analysis", wheel,
         "--output", output])
    return output


def build_system(wheel, target):
    lib = Path(target) / "lib"
    run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-cache-dir",
         "--no-warn-script-location", "--target", lib, wheel])
    # pip 会为目标入口生成指向构建机解释器的脚本，发布包改用 bin/ 下的启动器。
    generated_scripts = lib / "bin"
    if generated_scripts.is_dir():
        shutil.rmtree(generated_scripts)
    bin_dir = Path(target) / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        launcher = bin_dir / "SignalAnalysis.cmd"
        launcher.write_text(
            "@echo off\r\n"
            "setlocal\r\n"
            'set "ROOT=%~dp0.."\r\n'
            'set "PYTHONPATH=%ROOT%\\lib;%PYTHONPATH%"\r\n'
            'set "PYTHONNOUSERSITE=1"\r\n'
            'if defined SIMUSIGNAL_PYTHON (set "PY=%SIMUSIGNAL_PYTHON%") else (set "PY=python")\r\n'
            '"%PY%" -m signal_analysis %*\r\n'
            "exit /b %ERRORLEVEL%\r\n",
            encoding="utf-8")
    else:
        launcher = bin_dir / "SignalAnalysis"
        launcher.write_text(
            "#!/bin/sh\n"
            "set -eu\n"
            'bin_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\n'
            'root_dir=$(dirname -- "$bin_dir")\n'
            'python=${SIMUSIGNAL_PYTHON:-python3}\n'
            'PYTHONPATH="$root_dir/lib${PYTHONPATH:+:$PYTHONPATH}" \\\n'
            "PYTHONNOUSERSITE=1 \\\n"
            'exec "$python" -m signal_analysis "$@"\n',
            encoding="utf-8")
        launcher.chmod(0o755)
    return launcher


def write_share(target, project, config, wheel, platform_key):
    share = Path(target) / "share"
    requirements = share / "packaging/requirements"
    scripts = share / "scripts"
    requirements.mkdir(parents=True, exist_ok=True)
    scripts.mkdir(parents=True, exist_ok=True)
    for kind in ("build", "runtime"):
        source = ROOT / config["requirements"][kind].format(platform=platform_key)
        shutil.copyfile(source, requirements / source.name)
    shutil.copyfile(ROOT / "scripts/check_release_env.py", scripts / "check_release_env.py")
    (scripts / "check_release_env.py").chmod(0o755)
    shutil.copyfile(ROOT / "LICENSE", share / "LICENSE")
    shutil.copyfile(ROOT / "packaging/numeric_core.json", share / "packaging/numeric_core.json")
    (share / "packaging/release.json").write_text(json.dumps(
        {project: config}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    wheel_dir = Path(target) / "wheel"
    wheel_dir.mkdir(exist_ok=True)
    shutil.copyfile(wheel, wheel_dir / Path(wheel).name)
    return share


def write_manifest(target, project, config, wheel, mode, version, platform_key, started):
    core = core_manifest(wheel)
    entry_suffix = config["platforms"][platform_key]["exe_suffix"]
    entry = Path(target) / "bin" / (config["executable"] + entry_suffix)
    if mode == "system" and platform_key == "win32":
        entry = Path(target) / "bin" / (config["executable"] + ".cmd")
    manifest = {
        "schema": "simusignal_release_v1",
        "project": project,
        "name": config["executable"],
        "version": version,
        "mode": mode,
        "platform": f"{platform_key}-{check_release_env.machine_key()}",
        "python": sys.version.split()[0],
        "abi": sysconfig.get_config_var("SOABI") or sysconfig.get_platform(),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": {
            "branch": git_value("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": git_value("rev-parse", "HEAD"),
        },
        "toolchain": {
            "python": sys.version.split()[0],
            "cython": check_release_env.installed_version("Cython"),
            "pyinstaller": check_release_env.installed_version("pyinstaller"),
        },
        "core_modules": core,
        "wheel": {"file": Path(wheel).name, "sha256": sha256_file(wheel)},
        "entry": {"file": str(entry.relative_to(target)).replace(os.sep, "/"),
                  "sha256": sha256_file(entry) if entry.is_file() else None},
        "excluded": json.loads((ROOT / "packaging/projects.json").read_text(encoding="utf-8"))[
            project].get("exclude", []),
        "duration_s": round(time.time() - started, 1),
    }
    path = Path(target) / "share/manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def write_checksums(target):
    target = Path(target)
    lines = []
    for path in sorted(target.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{sha256_file(path)}  {path.relative_to(target).as_posix()}")
    (target / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(target, project, config, mode, version, platform_key):
    suffix = config["platforms"][platform_key]["exe_suffix"]
    entry = config["executable"] + (suffix if mode == "portable" else (".cmd" if platform_key == "win32" else ""))
    lines = [
        f"{config['executable']} {version}",
        "",
        "启动：",
        f"  bin/{entry} gui",
    ]
    if mode == "system":
        lines += [
            "",
            "system 模式需要目标机先安装 CPython 3.12 与依赖：",
            "  python -m pip install -r share/packaging/requirements/runtime-"
            f"{platform_key}-py312.txt",
            "可用 SIMUSIGNAL_PYTHON 指定解释器：",
            f"  SIMUSIGNAL_PYTHON=/path/to/python3 bin/{entry} gui",
        ]
    lines += [
        "",
        "目标机自检：",
        "  python share/scripts/check_release_env.py --stage runtime --output .",
        "",
        "运行时自检：",
        f"  bin/{entry} selftest --expect-binary",
        "",
        "说明：",
        "  share/packaging/requirements/ 为目标机 Python 依赖清单。",
        "  share/manifest.json 记录核心模块、wheel、入口与工具链哈希。",
        "  核心模块为 .so/.pyd：portable 使用 bin/_internal/ 内副本，lib/ 保留一份供核对。",
        "",
    ]
    (Path(target) / "README-RELEASE.txt").write_text("\n".join(lines), encoding="utf-8")


def smoke(target, config, mode, version, platform_key):
    suffix = config["platforms"][platform_key]["exe_suffix"]
    if mode == "portable":
        entry = Path(target) / "bin" / (config["executable"] + suffix)
    elif platform_key == "win32":
        entry = Path(target) / "bin" / (config["executable"] + ".cmd")
    else:
        entry = Path(target) / "bin" / config["executable"]
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    if mode == "system":
        # 构建机冒烟用当前解释器；目标机由启动器或 SIMUSIGNAL_PYTHON 指定。
        env.setdefault("SIMUSIGNAL_PYTHON", sys.executable)
    for arguments in config.get("smoke", []):
        log("smoke", f"{entry.name} {' '.join(arguments)}")
        command = [str(entry), *arguments]
        if sys.platform == "win32" and str(entry).lower().endswith(".cmd"):
            command = ["cmd", "/c", *command]
        result = subprocess.run(command, cwd=str(target), env=env,
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise ReleaseError(
                f"入口冒烟失败：{entry} {' '.join(arguments)}\n"
                f"stdout:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-2000:]}")
        if arguments == ["--version"] and version not in result.stdout:
            raise ReleaseError(f"--version 输出不包含 {version}：{result.stdout!r}")
        if arguments[0] == "selftest" and '"ok": true' not in result.stdout:
            raise ReleaseError(f"selftest 未通过：{result.stdout[-2000:]}")


def assemble_portable(work, target, wheel, project, config, version, platform_key, started):
    source = build_portable(wheel, work) / config["executable"]
    if not source.is_dir():
        raise ReleaseError(f"PyInstaller 输出不存在：{source}")
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    (target / "bin").mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        shutil.move(str(item), target / "bin" / item.name)
    extract_core(wheel, target / "lib")
    write_share(target, project, config, wheel, platform_key)
    write_readme(target, project, config, "portable", version, platform_key)
    write_manifest(target, project, config, wheel, "portable", version, platform_key, started)


def assemble_system(work, target, wheel, project, config, version, platform_key, started):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    build_system(wheel, target)
    extract_core(wheel, target / "lib")
    write_share(target, project, config, wheel, platform_key)
    write_readme(target, project, config, "system", version, platform_key)
    write_manifest(target, project, config, wheel, "system", version, platform_key, started)


def archive(target, platform_key, config):
    target = Path(target)
    fmt = config["platforms"][platform_key]["archive"]
    base = target.parent / target.name
    if fmt == "gztar":
        path = shutil.make_archive(str(base), "gztar", root_dir=str(target.parent),
                                   base_dir=target.name)
    else:
        path = shutil.make_archive(str(base), "zip", root_dir=str(target.parent),
                                   base_dir=target.name)
    digest = sha256_file(path)
    Path(str(path) + ".sha256").write_text(f"{digest}  {Path(path).name}\n", encoding="utf-8")
    return Path(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", choices=("analysis",), nargs="?", default="analysis")
    parser.add_argument("--mode", choices=("portable", "system", "both"), default="portable")
    parser.add_argument("--target", choices=("auto", "linux", "win32", "windows"), default="auto")
    parser.add_argument("--python", default=None, help="要求的 CPython 次版本")
    parser.add_argument("--output", type=Path, default=ROOT / "dist/release")
    parser.add_argument("--work", type=Path, help="保留并使用指定构建目录（默认临时目录）")
    parser.add_argument("--skip-env-check", action="store_true")
    parser.add_argument("--skip-tests", action="store_true", help="跳过分发前的隔离 pytest")
    parser.add_argument("--no-archive", action="store_true")
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args(argv)
    started = time.time()
    config = project_config(load_config(), args.project)
    target_key = check_release_env.platform_key()
    if target_key not in config["platforms"]:
        raise ReleaseError(f"当前系统 {target_key} 不在发布平台清单中（仅支持 Linux/Windows）")
    requested = "win32" if args.target in ("win32", "windows") else args.target
    if requested not in ("auto", target_key):
        raise ReleaseError(
            f"目标 {args.target} 与宿主 {target_key} 不一致；Python 扩展不能交叉编译")
    version = package_version()
    expected_python = args.python or config.get("python", "3.12")
    platform_label = f"{target_key}-{check_release_env.machine_key()}"
    args.output.mkdir(parents=True, exist_ok=True)
    work = Path(args.work) if args.work else Path(
        tempfile.mkdtemp(prefix="simusignal-release-"))
    if args.work:
        work.mkdir(parents=True, exist_ok=True)
    try:
        if not args.skip_env_check:
            log("1/6", "环境自检")
            run([sys.executable, ROOT / "scripts/check_release_env.py",
                 "--project", args.project, "--stage", "build",
                 "--target", target_key, "--python", expected_python,
                 "--output", args.output, "--json", work / "reports/env.json", "--full"])
        log("2/6", "编译二进制核心 wheel")
        wheel = build_wheel(work)
        log("3/6", "校验 wheel：核心扩展、明文排除与隔离回归")
        verify_wheel(wheel, work, args.skip_tests)
        modes = ("portable", "system") if args.mode == "both" else (args.mode,)
        results = []
        for index, mode in enumerate(modes, start=1):
            log("4/6", f"组装 {mode} 发布包")
            name = artifact_name(config["executable"], version, platform_label, mode)
            target = args.output / name
            if target.exists():
                shutil.rmtree(target)
            if mode == "portable":
                assemble_portable(work, target, wheel, args.project, config, version,
                                  target_key, started)
            else:
                assemble_system(work, target, wheel, args.project, config, version,
                                target_key, started)
            log("5/6", f"冒烟测试 {target.name}")
            smoke(target, config, mode, version, target_key)
            write_checksums(target)
            archive_path = None
            if not args.no_archive:
                archive_path = archive(target, target_key, config)
            results.append({"mode": mode, "directory": str(target),
                            "archive": str(archive_path) if archive_path else None,
                            "archive_sha256": sha256_file(archive_path) if archive_path else None})
        log("6/6", "完成")
        summary = {"schema": "simusignal_release_result_v1", "project": args.project,
                   "version": version, "platform": platform_label,
                   "python": sys.version.split()[0], "wheel": str(wheel),
                   "wheel_sha256": sha256_file(wheel),
                   "duration_s": round(time.time() - started, 1),
                   "artifacts": results}
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if args.keep_work:
            print(f"构建目录：{work}")
        return 0
    except ReleaseError as exc:
        print(f"发布失败：{exc}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"发布失败：命令返回 {exc.returncode}", file=sys.stderr)
        return 1
    finally:
        if not args.work and not args.keep_work:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReleaseError as exc:
        print(f"发布失败：{exc}", file=sys.stderr)
        raise SystemExit(1)
    except OSError as exc:
        print(f"发布失败：{exc}", file=sys.stderr)
        raise SystemExit(1)
    except ValueError as exc:
        print(f"发布失败：{exc}", file=sys.stderr)
        raise SystemExit(1)
