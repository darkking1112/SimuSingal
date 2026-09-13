"""保存或核对数值重构基线：完整摘要、数组字节摘要、类型及异常消息。

在修改前使用 --record；修改后用相同命令去掉 --record。基线仅适用于
相同 Python/NumPy 环境；--source 可指向源码目录或 wheel 解包目录。
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np


def encode(value):
    """将结果编码为可比较 JSON；数组同时保留形状、类型和逐字节摘要。"""
    if isinstance(value, np.ndarray):
        return {"array_dtype": value.dtype.str, "shape": list(value.shape),
                "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}
    if isinstance(value, dict):
        return {"dict": [[str(key), encode(item)] for key, item in value.items()]}
    if isinstance(value, (list, tuple)):
        return {type(value).__name__: [encode(item) for item in value]}
    if isinstance(value, np.generic):
        return {"scalar_dtype": value.dtype.str, "value": encode(value.item())}
    if isinstance(value, complex):
        return {"complex": [value.real.hex(), value.imag.hex()]}
    if isinstance(value, float):
        return {"float": value.hex()}
    return value


def snapshot():
    """固定种子覆盖生成、分析、两种检测、特征、IQ 前处理及异常出口。"""
    from signal_analysis import _numeric as n
    from signal_analysis.ml import amc, iq

    results = {}

    def record(label, function, *args, **kwargs):
        try:
            results[label] = encode(function(*args, **kwargs))
        except Exception as exc:
            results[label] = {"error": type(exc).__name__, "message": str(exc)}

    rate = 1_000_000.0
    for index, mode in enumerate(n.MODES):
        spec = {"mode": mode, "offset": 100_000.0, "bandwidth": 100_000.0,
                "power_dbfs": -10.0}
        x, generation = n.generate_iq(rate, 0.05, [spec],
                                     noise={"bandwidth": rate, "snr_db": 20.0}, seed=index)
        results[mode + "/generation"] = encode((x, generation))
        record(mode + "/plan", n.plan_signal, spec, rate)
        record(mode + "/analysis", n.analyze, x, rate)
        record(mode + "/spectrum", n.spectrum_row, x, 73, rate=rate)
        record(mode + "/classify", n.classify_modulation, x)
        record(mode + "/detect", n.detect_signals, x, rate)
        record(mode + "/hops", n.detect_hops, x, rate)
        record(mode + "/hops_without_baseline", n.detect_hops, x, rate, with_sessions=False)
        record(mode + "/features", amc.extract_features, x, rate, 100_000.0, 100_000.0)
        record(mode + "/preprocess", amc._mix_and_decimate, amc._validate(x),
               rate, 100_000.0, 100_000.0)
        record(mode + "/iq", iq.iq_waveform, x, rate, 100_000.0, 100_000.0)

    rng = np.random.default_rng(123)
    scenes = {
        "short": np.array([1 + 2j], dtype=np.complex64),
        "zero": np.zeros(4096, dtype=np.complex64),
        "noise": rng.standard_normal(8192) + 1j * rng.standard_normal(8192),
    }
    scenes["multi"] = n.generate_iq(rate, 0.1, [
        {"mode": "fm", "offset": -200_000, "bandwidth": 50_000},
        {"mode": "qpsk", "offset": 200_000, "bandwidth": 50_000}], seed=7)[0]
    # 恒定频点的长驻留对应连续复用同一信道时的观测极限。
    scenes["same_channel"] = np.exp(2j * np.pi * 0.1 * np.arange(65536))
    scenes["fast_hops"] = n.generate_iq(rate, 0.05, [
        {"mode": "fh_rc", "offset": 0, "bandwidth": 400_000,
         "hop_rate": 10_000}], seed=8)[0]
    for name, x in scenes.items():
        for function in (n.analyze, n.detect_signals, n.detect_hops, amc.extract_features):
            record(name + "/" + function.__name__, function, x, rate)
    for name, x in {"empty": [], "nan": [float("nan")], "matrix": [[1]],
                    "text": ["bad"], "zero": [0]}.items():
        for function in (n.validate_samples, n.classify_modulation, amc._validate):
            record("invalid/" + name + "/" + function.__name__, function, x)
    for name, config in {"unknown": {"oops": 1}, "nfft": {"nfft": True},
                         "threshold": {"threshold_db": -1}}.items():
        for function in (n.detect_signals, n.detect_hops):
            record("config/" + name + "/" + function.__name__, function,
                   scenes["noise"], rate, config)
    record("demo", n.make_demo)
    record("feature_vector", amc.feature_vector, {key: i for i, key in enumerate(amc.AMC_FEATURES)})
    results["feature_order"] = encode(amc.AMC_FEATURES)
    results["signatures"] = {f.__name__: str(inspect.signature(f)) for f in (
        n.generate_iq, n.plan_signal, n.analyze, n.classify_modulation, n.detect_signals,
        n.detect_hops, n.spectrum_row, amc.extract_features, amc.feature_vector, iq.iq_waveform)}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] / "src")
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    current = snapshot()
    if args.record:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        # 防止误覆盖唯一的拆分前基线。
        with args.baseline.open("x", encoding="utf-8") as output:
            json.dump(current, output, ensure_ascii=False, indent=2)
        print(f"Recorded {len(current)} cases: {args.baseline}")
    else:
        expected = json.loads(args.baseline.read_text(encoding="utf-8"))
        differences = [key for key in expected.keys() | current.keys()
                       if expected.get(key) != current.get(key)]
        if differences:
            raise AssertionError(f"Numerical differences: {differences}")
        print(f"Exact equality: {len(current)} cases (including array bytes and exceptions)")


if __name__ == "__main__":
    main()
