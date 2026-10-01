"""TorchSig 接入：生成 bundle（仅 Linux）、导入 bundle 为集合资产并换成目标。

TorchSig 依赖 ``torchsig`` 且要在 Linux 下运行，所以是唯一需要外部进程的生成引擎：

1. ``run_generation``：用所选 Python 环境跑 ``training/build_torchsig.py`` 生成
   ``torchsig_bundle_v1``（Windows 直接提示“只能在 Linux 下使用”，不做任何降级）；
2. ``import_bundle``：**纯 NumPy**，任何平台都能跑——把 bundle 里的 IQ 写成分片资产，
   每个信号实例换成一个目标：起止时间（存采样点）、中心频率与带宽（由频带边界派生）、
   带内 SNR（按 ``inband_snr_v1`` 重测；测不了时退回 TorchSig 标称值并标明口径）。

类别映射：TorchSig 的 ``class_name`` 属于它自己的类别体系，本项目不猜映射。用户给出
``{TorchSig 类名: A09 类别}`` 映射时写成对应的规范调制名（如 ``qpsk`` → ``QPSK``），
映射不到的保留原类名，AMC 标注记为“字典外”。TorchSig 没有跳频，不产生逐跳目标。

bundle 格式定义见 ``training/torchsig_bundle.py``（训练目录不随包分发，这里按同一份
格式自行读取）。
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

from .collection_gen import SHARD_BYTES, Reporter, annotate_assets, prepare_task_sets
from .datasets import CLASS_TO_MODULATION
from .recipes import distribution_bounds, literal_values

BUNDLE_VERSION = "torchsig_bundle_v1"
ENV_NAME = "torchsig_env.json"
_PROGRESS = re.compile(r"已生成 (\d+)/(\d+)")

LINUX_ONLY = ("TorchSig 只能在 Linux 环境下使用（它依赖 torchsig，需要 Linux）；当前系统不是 "
              "Linux。可在 Linux/WSL/远程机上用 training/build_torchsig.py 生成 bundle，"
              "再回到这里“导入 TorchSig bundle”。")


def is_linux():
    return sys.platform.startswith("linux")


# ---------------------------------------------------------------------------
# 环境设置与探测
# ---------------------------------------------------------------------------


def _default_repository():
    root = Path(__file__).resolve().parents[2]
    return str(root) if (root / "training" / "build_torchsig.py").is_file() else ""


def read_env(workspace):
    """TorchSig 环境设置：训练源码根目录 + 装有 torchsig 的 Python；损坏时回退默认。"""
    env = {"repository": _default_repository(),
           "python": "" if getattr(sys, "frozen", False) else sys.executable}
    try:
        data = json.loads((Path(workspace.root) / "training" / ENV_NAME)
                          .read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return env
    if isinstance(data, dict):
        for key in env:
            if isinstance(data.get(key), str) and data[key].strip():
                env[key] = data[key].strip()
    return env


def write_env(workspace, repository, python):
    folder = Path(workspace.root) / "training"
    folder.mkdir(exist_ok=True)
    path = folder / ENV_NAME
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"repository": str(repository).strip(),
                                     "python": str(python).strip()},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _resolve_env(workspace, request):
    env = dict(read_env(workspace))
    supplied = request.get("torchsig_env")
    if isinstance(supplied, dict):
        env.update({key: str(value).strip() for key, value in supplied.items()
                    if key in env and str(value).strip()})
    python = shutil.which(env["python"]) if env["python"] else None
    script = Path(env["repository"]).expanduser() / "training" / "build_torchsig.py"
    if not python:
        raise ValueError("TorchSig 环境的 Python 不存在；请在“TorchSig 环境…”里设置")
    if not script.is_file():
        raise ValueError("TorchSig 环境的训练源码根目录缺少 training/build_torchsig.py；"
                         "请在“TorchSig 环境…”里设置")
    return {"python": python, "script": str(script.resolve()),
            "repository": str(script.resolve().parents[1])}


def preflight(workspace, request):
    """生成前的环境检查：非 Linux 或环境缺失直接报错（此时还没有建集合）。"""
    if not is_linux():
        raise ValueError(LINUX_ONLY)
    return _resolve_env(workspace, request)


def probe(workspace, request):
    """测试 TorchSig 环境：平台、Python、源码脚本、``import torchsig`` 与版本。"""
    result = {"kind": "torchsig_probe", "ok": False, "linux": is_linux(),
              "version": None, "message": ""}
    if not result["linux"]:
        result["message"] = LINUX_ONLY
        return result
    try:
        env = _resolve_env(workspace, request)
    except ValueError as exc:
        result["message"] = str(exc)
        return result
    try:
        done = subprocess.run(
            [env["python"], "-c",
             "import torchsig; print(getattr(torchsig, '__version__', 'unknown'))"],
            capture_output=True, timeout=110, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["message"] = f"无法运行所选 Python：{exc}"
        return result
    if done.returncode != 0:
        tail = done.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or ["?"]
        result["message"] = ("所选 Python 无法 import torchsig："
                             f"{tail[0]}（安装：pip install -e '.[train]' 或 torchsig==2.2.0）")
        return result
    result.update(ok=True, version=done.stdout.decode("utf-8", "replace").strip(),
                  message=f"TorchSig {done.stdout.decode('utf-8', 'replace').strip()} 可用",
                  python=env["python"], repository=env["repository"])
    return result


# ---------------------------------------------------------------------------
# 配方 → build_torchsig.py 参数
# ---------------------------------------------------------------------------


def build_arguments(recipe, bundle):
    """把 TorchSig 引擎的配方压成 ``build_torchsig.py`` 命令行（分布压成区间）。"""
    record = recipe.get("record") or {}
    signals = recipe.get("signals") or {}
    options = recipe.get("torchsig") or {}
    rate = distribution_bounds(record["sample_rate_hz"])[0] if "sample_rate_hz" in record \
        else 1_000_000.0
    arguments = ["--output", str(bundle), "--count", str(int(recipe["count"])),
                 "--seed", str(int(recipe["base_seed"])), "--sample-rate", repr(float(rate))]
    if "duration_s" in record:
        arguments += ["--num-iq-samples",
                      str(max(2, int(round(rate * distribution_bounds(record["duration_s"])[0]))))]
    if "count" in signals:
        low, high = distribution_bounds(signals["count"])
        arguments += ["--signals-range", f"{int(low)},{int(high)}"]
    if "snr_db" in signals:
        low, high = distribution_bounds(signals["snr_db"])
        arguments += ["--snr-range", f"{low!r},{high!r}"]
    if "bandwidth_ratio" in signals:
        low, high = distribution_bounds(signals["bandwidth_ratio"])
        arguments += ["--bandwidth-ratio-range", f"{low!r},{high!r}"]
    generators = literal_values(options.get("signal_generators") or {"fixed": "all"})[0]
    arguments += ["--signal-generators", str(generators or "all")]
    level = literal_values(options.get("impairment_level") or {"fixed": None})[0]
    if level is not None:
        arguments += ["--impairment-level", str(int(level))]
    return arguments


def _run_build(env, arguments, log_path, reporter, total):
    """运行 build_torchsig.py；返回 ``"cancelled"`` 或 ``None``，失败抛 ``ValueError``。"""
    command = [env["python"], "-u", env["script"], *arguments]
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    with log_path.open("wb") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   cwd=env["repository"], env=environment)
        while process.poll() is None:
            if reporter.cancelled():
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                return "cancelled"
            text = log_path.read_bytes()[-4000:].decode("utf-8", "replace")
            found = _PROGRESS.findall(text)
            done = int(found[-1][0]) if found else 0
            reporter.emit(done, total, f"TorchSig 生成 bundle {done}/{total}")
            time.sleep(0.3)
    if process.returncode != 0:
        tail = log_path.read_bytes()[-1200:].decode("utf-8", "replace").strip()
        raise ValueError(f"TorchSig 生成失败（退出码 {process.returncode}）：{tail}")
    return None


# ---------------------------------------------------------------------------
# bundle 导入
# ---------------------------------------------------------------------------


def load_mapping(source):
    """``{TorchSig 类名: A09 类别}``；类别必须属于 A09 六类，键按不区分大小写匹配。"""
    from .ml.amc import AMC_CLASSES

    if not source:
        return {}
    if isinstance(source, dict):
        data = source
    else:
        try:
            data = json.loads(Path(str(source)).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"类别映射 JSON 无法读取：{exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("类别映射必须是 {TorchSig 类名: A09 类别} 的 JSON 对象")
    outside = sorted({str(value) for value in data.values() if value not in AMC_CLASSES})
    if outside:
        raise ValueError(f"映射目标必须属于 A09 类别（{' / '.join(AMC_CLASSES)}）："
                         f"{' / '.join(outside)}")
    return {str(key).strip().lower(): str(value) for key, value in data.items()}


def read_bundle_manifest(bundle):
    path = Path(bundle) / "manifest.json"
    if not path.is_file():
        raise ValueError(f"{bundle} 不是有效的 bundle（缺少 manifest.json）")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"bundle 清单无法解析：{exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("bundle_version") != BUNDLE_VERSION:
        raise ValueError(f"bundle 格式版本应为 {BUNDLE_VERSION}")
    rate = manifest.get("sample_rate_hz")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not rate > 0:
        raise ValueError("bundle 清单缺少有效的 sample_rate_hz")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("bundle 清单里没有 records")
    return manifest


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _component_band(component, rate, sample_count):
    """信号实例 → 频带与起止采样点；无法落成合法目标时返回 ``(None, 原因)``。"""
    if not isinstance(component, dict):
        return None, "malformed_component"
    low, high = _number(component.get("lower_freq")), _number(component.get("upper_freq"))
    if low is None or high is None:
        center = _number(component.get("center_freq"))
        width = _number(component.get("bandwidth"))
        if width is None:
            width = _number(component.get("occupied_bandwidth"))
        if center is None or width is None:
            return None, "missing_geometry"
        low, high = center - width / 2.0, center + width / 2.0
    low, high = min(low, high), max(low, high)
    nyquist = rate / 2.0
    if high <= -nyquist or low >= nyquist:
        return None, "out_of_band"
    low, high = max(low, -nyquist), min(high, nyquist)
    if high - low <= 0:
        return None, "too_narrow"
    start = _number(component.get("start_in_samples"))
    length = _number(component.get("duration_in_samples"))
    if start is not None and length is not None:
        begin, end = int(round(start)), int(round(start + length))
    else:  # 采样点字段缺失才退回秒制（TorchSig 原生的 start/stop 是比例，不可直接用）
        begin = int(round((_number(component.get("start")) or 0.0) * rate))
        stop = _number(component.get("stop"))
        end = sample_count if stop is None else int(round(stop * rate))
    begin, end = max(begin, 0), min(end, sample_count)
    if end <= begin:
        return None, "too_short"
    return {"f_low_hz": low, "f_high_hz": high, "sample_start": begin, "sample_end": end}, None


def _read_iq(path):
    array = np.load(path, allow_pickle=False)
    if array.ndim == 2 and array.shape[1] == 2:
        array = array[:, 0] + 1j * array[:, 1]
    if array.ndim != 1 or array.size < 2:
        raise ValueError(f"{path.name} 应为单通道 IQ 序列，实际形状 {array.shape}")
    if not np.iscomplexobj(array):
        array = array.astype(np.float64).astype(np.complex64)
    return np.ascontiguousarray(array, dtype=np.complex64)


def _measure(samples, rate, bands):
    """按 ``inband_snr_v1`` 重测各频带的功率与 SNR；测不了的项返回 ``None``。"""
    from .ml.tensor import measure_band, spectral_context

    try:
        summary, arrays = spectral_context(samples, rate, {"nfft": 512})
    except ValueError:
        return [None] * len(bands)
    results = []
    for band in bands:
        try:
            measured = measure_band(arrays, summary, {
                "f_low_hz": band["f_low_hz"], "f_high_hz": band["f_high_hz"],
                "t_start_s": band["sample_start"] / rate, "t_end_s": band["sample_end"] / rate})
            results.append({"power_dbfs": _number(measured.get("power_dbfs")),
                            "snr_db": _number(measured.get("snr_db"))})
        except (ValueError, KeyError):
            results.append(None)
    return results


def _add_targets(workspace, asset_id, components, bands, measured, mapping, unmapped):
    scope = "session" if len(bands) > 1 else "whole_record"
    for order, (component, band, info) in enumerate(zip(components, bands, measured)):
        name = str(component.get("class_name") or "").strip()
        mapped = mapping.get(name.lower()) if name else None
        if name and mapped is None:
            unmapped[name] = unmapped.get(name, 0) + 1
        modulation = CLASS_TO_MODULATION[mapped] if mapped else (name or None)
        target = workspace.add_target(asset_id, f"s{order}", scope, for_detection=1,
                                      for_amc=1 if name else 0)
        nominal = _number(component.get("snr_db"))
        snr = (info or {}).get("snr_db")
        definition = "inband_snr_v1"
        if snr is None and nominal is not None:
            snr, definition = nominal, "torchsig_nominal"
        center = round((band["f_low_hz"] + band["f_high_hz"]) / 2.0, 6)
        width = round(band["f_high_hz"] - band["f_low_hz"], 6)
        workspace.append_target_version(
            target["id"], source="external", sample_start=band["sample_start"],
            sample_end=band["sample_end"], f_low_hz=band["f_low_hz"],
            f_high_hz=band["f_high_hz"], nominal_center_hz=center,
            nominal_bandwidth_hz=width, modulation=modulation, is_hopping=0, snr_db=snr,
            snr_definition=definition if snr is not None else None,
            power_dbfs=(info or {}).get("power_dbfs"),
            params_json={"torchsig": {"class_name": name or None, "snr_nominal_db": nominal}})


def import_bundle(workspace, bundle, collection, task_sets, reporter, *, mapping=None,
                  recipe_row=None, created_by=None, max_seconds=None):
    """把 bundle 写成集合里的分片资产并建立目标，随后按标注集写初始标签。"""
    bundle = Path(bundle)
    manifest = read_bundle_manifest(bundle)
    rate = float(manifest["sample_rate_hz"])
    mapping = mapping or {}
    records = manifest["records"]
    total = len(records)
    started = time.monotonic()
    writer, shard_bytes, shards = None, 0, 0
    pending, noise_ids = [], set()
    failures, unmapped, totals = {}, {}, {"detection": 0, "amc": 0}
    created = sessions = skipped_components = 0
    stopped = None

    def flush():
        if pending:
            counts = annotate_assets(workspace, task_sets, pending, source="external",
                                     noise_only=noise_ids)
            for key, value in counts.items():
                totals[key] += value
            pending.clear()
            noise_ids.clear()

    def fail(reason):
        failures[reason] = failures.get(reason, 0) + 1

    try:
        for position, entry in enumerate(records):
            if reporter.cancelled():
                stopped = "cancelled"
                break
            if max_seconds and time.monotonic() - started > float(max_seconds):
                stopped = "time_limit"
                break
            try:
                samples = _read_iq(bundle / str(entry["iq"]))
                meta_name = entry.get("meta")
                meta = json.loads((bundle / str(meta_name)).read_text(encoding="utf-8")) \
                    if meta_name else {}
                components = meta.get("components") or []
                if isinstance(components, dict):
                    components = list(components.values())
                bands, kept = [], []
                for component in components:
                    band, reason = _component_band(component, rate, int(samples.size))
                    if band is None:
                        skipped_components += 1
                        fail(f"信号实例被跳过：{reason}")
                        continue
                    bands.append(band)
                    kept.append(component)
            except (OSError, ValueError, KeyError) as exc:
                fail(f"记录读取失败：{exc}")
                continue
            if writer is None:
                shards += 1
                writer = workspace.create_shard(
                    name=f"{collection['name']} · TorchSig 分片 {shards}",
                    created_by=created_by)
            metadata = {"torchsig": {"bundle": bundle.name, "record_index": position,
                                     "torchsig_version": manifest.get("torchsig_version")}}
            if recipe_row is not None:
                metadata["recipe"] = {"recipe_id": recipe_row["id"], "index": position}
            asset = writer.append(samples, rate, f"{collection['name']} · {position:06d}",
                                  "generated:torchsig_v1", source_kind="generated",
                                  metadata=metadata)
            workspace.add_collection_member(collection["id"], asset["id"])
            _add_targets(workspace, asset["id"], kept, bands,
                         _measure(samples, rate, bands), mapping, unmapped)
            created += 1
            sessions += len(bands)
            pending.append(asset["id"])
            if not bands:
                noise_ids.add(asset["id"])
            shard_bytes += int(samples.nbytes)
            if shard_bytes >= SHARD_BYTES:
                writer.seal()
                writer, shard_bytes = None, 0
                flush()
            reporter.emit(position + 1, total, f"导入 TorchSig 记录 {position + 1}/{total}")
    finally:
        if writer is not None:
            writer.seal()
    reporter.emit(total, total, "写入标注…", force=True)
    flush()
    return {"created": created, "failures": failures, "sessions": sessions, "hops": 0,
            "labels": totals, "stopped": stopped, "shards": shards,
            "unmapped": dict(sorted(unmapped.items())),
            "skipped_components": skipped_components}


def run_generation(workspace, recipe, collection, recipe_row, task_sets, reporter, request):
    """TorchSig 引擎：外部进程生成 bundle → 导入集合；临时 bundle 用完即删。"""
    if not is_linux():
        raise ValueError(LINUX_ONLY)
    env = _resolve_env(workspace, request)
    mapping = load_mapping(request.get("mapping") or request.get("mapping_path"))
    folder = Path(workspace.root) / "training" / "torchsig_tmp" / uuid.uuid4().hex
    folder.mkdir(parents=True)
    try:
        total = int(recipe["count"])
        reporter.emit(0, total, "启动 TorchSig 环境…", force=True)
        status = _run_build(env, build_arguments(recipe, folder / "bundle"),
                            folder / "build.log", reporter, total)
        if status == "cancelled":
            return {"created": 0, "failures": {}, "sessions": 0, "hops": 0,
                    "labels": {"detection": 0, "amc": 0}, "stopped": "cancelled", "shards": 0}
        return import_bundle(workspace, folder / "bundle", collection, task_sets, reporter,
                             mapping=mapping, recipe_row=recipe_row,
                             created_by=request.get("created_by"),
                             max_seconds=request.get("max_seconds"))
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def import_action(workspace, request, resolve_collection):
    """导入已有 bundle（任何平台）：``bundle_path`` + 可选映射 + 目标集合 + 标注勾选。"""
    bundle = Path(str(request.get("bundle_path") or ""))
    manifest = read_bundle_manifest(bundle)
    mapping = load_mapping(request.get("mapping") or request.get("mapping_path"))
    labels = {"detection": "session_v1" if request.get("detection", True) else None,
              "amc": bool(request.get("amc", True))}
    reporter = Reporter(request.get("job_dir"))
    started = time.monotonic()
    collection = resolve_collection()
    if collection is None:
        raise ValueError("请指定目标集合（新建或追加）")
    task_sets = prepare_task_sets(workspace, collection, labels)
    outcome = import_bundle(workspace, bundle, collection, task_sets, reporter,
                            mapping=mapping, created_by=request.get("created_by"),
                            max_seconds=request.get("max_seconds"))
    reasons = sorted(outcome["failures"].items(), key=lambda item: -item[1])
    return {"kind": "generate_collection", "engine": "torchsig_import",
            "requested": len(manifest["records"]), "created": outcome["created"],
            "failed": sum(outcome["failures"].values()),
            "failure_reasons": [{"reason": reason, "count": count}
                                for reason, count in reasons[:5]],
            "stopped": outcome["stopped"], "collection_id": collection["id"],
            "collection_name": collection["name"], "recipe_id": None,
            "sessions": outcome["sessions"], "hops": 0, "labels": outcome["labels"],
            "shards": outcome["shards"], "unmapped": outcome["unmapped"],
            "elapsed_s": round(time.monotonic() - started, 3)}
