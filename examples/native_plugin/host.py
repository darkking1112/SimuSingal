"""Small native-call demo. All library loading takes place in a child process."""

import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import subprocess
import sys


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def worker(library_path):
    # Keep the Windows dependency search handle alive until this process exits.
    dll_directory = None
    if os.name == "nt":
        dll_directory = os.add_dll_directory(str(library_path.parent))
    try:
        lib = ct.CDLL(str(library_path))
        lib.demo_abi_version.argtypes = []
        lib.demo_abi_version.restype = ct.c_uint32
        check(lib.demo_abi_version() == 1, "unsupported demo ABI")

        copy = lib.demo_copy_f32
        copy.argtypes = [
            ct.POINTER(ct.c_float), ct.c_uint64,
            ct.POINTER(ct.c_float), ct.c_uint64,
            ct.POINTER(ct.c_uint64),
        ]
        copy.restype = ct.c_int32

        values = [1.0, -2.5, 0.0, 3.25]
        source = (ct.c_float * len(values))(*values)
        output = (ct.c_float * len(values))()
        written = ct.c_uint64()
        status = copy(source, len(values), output, len(values), ct.byref(written))
        check(status == 0 and written.value == len(values), "copy failed")
        check(list(output) == values, "unexpected output")

        small = (ct.c_float * 1)(99.0)
        status = copy(source, len(values), small, 1, ct.byref(written))
        check(status == 2 and written.value == 0, "capacity error not reported")
        check(small[0] == 99.0, "output changed on capacity error")

        status = copy(None, 0, None, 0, ct.byref(written))
        check(status == 0 and written.value == 0, "empty input failed")
        status = copy(None, 1, output, len(values), ct.byref(written))
        check(status == 1 and written.value == 0, "null input not rejected")
        status = copy(source, len(values), output, len(values), None)
        check(status == 1, "null written pointer not rejected")

        return {"abi": 1, "output": list(output), "checks_passed": 5}
    finally:
        if dll_directory is not None:
            dll_directory.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    library_path = args.library.resolve()
    if not library_path.is_file():
        parser.error(f"library does not exist: {library_path}")

    if args.worker:
        try:
            print(json.dumps(worker(library_path)))
            return 0
        except Exception as exc:
            print(f"plugin error: {exc}", file=sys.stderr)
            return 1

    # No native library is loaded by the parent. A crash/timeout ends this task.
    try:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), str(library_path), "--worker"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except subprocess.TimeoutExpired:
        print("plugin timeout (10 seconds)", file=sys.stderr)
        return 1
    if result.returncode != 0:
        print(f"plugin worker failed: exit={result.returncode}", file=sys.stderr)
        print(result.stderr.strip(), file=sys.stderr)
        return 1
    try:
        report = json.loads(result.stdout)
        check(isinstance(report, dict) and report.get("checks_passed") == 5,
              "unexpected worker report")
    except (ValueError, RuntimeError) as exc:
        print(f"invalid plugin report: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
