"""发布包运行时自检：验证核心模块、内置模型与一次数值闭环。"""

from __future__ import annotations

import hashlib
import platform
import sys
from importlib import import_module
from pathlib import Path

from . import __version__

#: 与 packaging/numeric_core.json 保持一致的运行时副本；tests/common 会校验两者同步。
CORE_MODULES = (
    "signal_analysis.algorithms.dsp.base",
    "signal_analysis.algorithms.dsp.spectrum",
    "signal_analysis.algorithms.dsp.preprocess",
    "signal_analysis.algorithms.generation.iqgen",
    "signal_analysis.algorithms.amc.features",
    "signal_analysis.algorithms.amc.heuristic",
    "signal_analysis.algorithms.detection.energy",
    "signal_analysis.algorithms.detection.hops",
)

BINARY_SUFFIXES = (".so", ".pyd")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _platform_label():
    machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "x64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
    return f"{sys.platform}-{machine}"


def _core_records():
    records = []
    errors = []
    for name in CORE_MODULES:
        try:
            module = import_module(name)
            path = Path(module.__file__).resolve()
            binary = path.suffix in BINARY_SUFFIXES
            records.append({"name": name, "file": str(path), "binary": binary,
                            "sha256": _sha256(path)})
        except Exception as exc:  # noqa: BLE001 - 自检需要收集所有失败面
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    return records, errors


def _numeric_records():
    import numpy as np

    from .algorithms.detection.energy import detect_signals
    from .algorithms.dsp.spectrum import analyze
    from .algorithms.generation.iqgen import generate_iq

    sample_rate = 48000.0
    samples, _ = generate_iq(
        sample_rate, 0.05,
        [{"mode": "qpsk", "offset": 2000.0, "bandwidth": 4000.0}],
        noise={"enabled": True, "snr_db": 20.0}, seed=7)
    stats, _ = analyze(samples, sample_rate, nfft=64)
    detection, _ = detect_signals(samples, sample_rate, {"nfft": 64})
    checks = {
        "sample_count": int(samples.size),
        "finite": bool(np.all(np.isfinite(samples))),
        "statistics_contract": stats.get("algorithm") == "generic_statistics_v1",
        "detection_contract": detection.get("contract") == "detect_result_v1",
        "sample_count_matches": stats.get("sample_count") == int(samples.size),
    }
    digest = hashlib.sha256(
        np.round(np.abs(samples[:1024]).astype(np.float64), 6).tobytes()).hexdigest()
    return {
        "sample_rate_hz": sample_rate,
        "sample_count": int(samples.size),
        "contracts": [stats.get("algorithm"), detection.get("contract")],
        "checks": checks,
        "checksum": digest,
    }


def run_selftest(expect_binary=False):
    """返回自检结果字典；不抛异常，失败写入 ``errors`` 且 ``ok=False``。"""
    result = {
        "schema": "simusignal_selftest_v1",
        "ok": False,
        "version": __version__,
        "python": sys.version.split()[0],
        "platform": _platform_label(),
        "expect_binary": bool(expect_binary),
        "binary": False,
        "core_modules": [],
        "model": None,
        "numeric": None,
        "errors": [],
    }
    records, errors = _core_records()
    result["core_modules"] = records
    result["errors"].extend(errors)
    result["binary"] = bool(records) and all(item["binary"] for item in records)
    if expect_binary and not result["binary"]:
        result["errors"].append("预期核心模块为 .so/.pyd，但存在源码模块")

    try:
        from .algorithms.amc.feature_model import AMC_MODEL_CONTRACT, default_model_path, load_model
        path = default_model_path()
        model = load_model(path)
        if model.get("contract") != AMC_MODEL_CONTRACT:
            raise ValueError(f"模型契约不匹配：{model.get('contract')!r}")
        result["model"] = {"id": model.get("id"), "version": model.get("version"),
                           "file": str(path), "sha256": _sha256(path)}
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"内置模型: {type(exc).__name__}: {exc}")

    try:
        numeric = _numeric_records()
        result["numeric"] = numeric
        checks = numeric["checks"]
        if not (checks["finite"] and checks["statistics_contract"]
                and checks["detection_contract"] and checks["sample_count_matches"]):
            result["errors"].append("数值闭环检查未全部通过")
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"数值闭环: {type(exc).__name__}: {exc}")

    result["ok"] = not result["errors"]
    return result
