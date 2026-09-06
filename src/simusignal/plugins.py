"""Demo C ABI adapter. Load and call ONLY from an isolated worker."""

import ctypes as ct
import json
import os
from pathlib import Path
import platform
import struct

import numpy as np

from .core_api import validate_samples
from .storage import file_digest


def create_demo_manifest(library, output):
    library = Path(library).resolve(strict=True)
    output = Path(output).resolve()
    if not library.is_relative_to(output.parent):
        raise ValueError("清单应放在动态库所在目录或其上级目录")
    manifest = {
        "schema_version": 1, "id": "demo.copy_f32", "version": "1.0.0",
        "adapter": "demo_copy_f32", "abi_version": 1,
        "platform": platform.system(), "machine": platform.machine().lower(),
        "bits": struct.calcsize("P") * 8, "calling_convention": "cdecl",
        "library": str(library.relative_to(output.parent)), "sha256": file_digest(library),
    }
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def read_manifest(path):
    path = Path(path).resolve(strict=True)
    if path.stat().st_size > 64 * 1024:
        raise ValueError("插件清单超过 64 KiB")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {"schema_version": 1, "adapter": "demo_copy_f32", "abi_version": 1,
                "platform": platform.system(), "machine": platform.machine().lower(),
                "bits": struct.calcsize("P") * 8, "calling_convention": "cdecl"}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"插件清单不兼容：{key}")
    for field in ("library", "id", "version", "sha256"):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise ValueError(f"缺少插件字段：{field}")
    relative = Path(manifest["library"])
    if relative.is_absolute():
        raise ValueError("动态库必须使用插件目录内相对路径")
    library = (path.parent / relative).resolve(strict=True)
    if not library.is_relative_to(path.parent):
        raise ValueError("动态库路径越过插件目录")
    if file_digest(library) != manifest["sha256"]:
        raise ValueError("动态库摘要与清单不符")
    return manifest, library


def call_demo_plugin(manifest_path, samples):
    manifest, library = read_manifest(manifest_path)
    x = validate_samples(samples)
    data = x.view(np.float32)
    output = np.zeros_like(data)
    dll_dir = os.add_dll_directory(str(library.parent)) if os.name == "nt" else None
    try:
        lib = ct.CDLL(str(library))
        lib.demo_abi_version.argtypes = []
        lib.demo_abi_version.restype = ct.c_uint32
        if lib.demo_abi_version() != 1:
            raise ValueError("动态库实际 ABI 不兼容")
        copy = lib.demo_copy_f32
        pointer = ct.POINTER(ct.c_float)
        copy.argtypes = [pointer, ct.c_uint64, pointer, ct.c_uint64, ct.POINTER(ct.c_uint64)]
        copy.restype = ct.c_int32
        written = ct.c_uint64()
        status = copy(data.ctypes.data_as(pointer), data.size,
                      output.ctypes.data_as(pointer), output.size, ct.byref(written))
        if status != 0 or written.value != output.size:
            raise ValueError(f"原生插件返回错误：status={status}, written={written.value}")
        result = validate_samples(output.view(np.complex64))
        return manifest, result
    finally:
        if dll_dir is not None:
            dll_dir.close()
