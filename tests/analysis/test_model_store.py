"""模型库：命名、入库（硬链接/复制）、盘点、重命名与删除，以及既有运行的登记。"""
import json
from pathlib import Path

import pytest

from signal_analysis.services import model_store


def _write_run(workspace, run_id, *, task="iq", arch="cnn", classes=("fm", "qpsk"),
               label_semantics=None, library="iq.onnx", metrics=None, status="success"):
    """造一次训练运行的落盘产物：experiment.json + model/<清单> + model/<onnx>。"""
    directory = Path(workspace) / "training" / "runs" / run_id
    model_dir = directory / "model"
    model_dir.mkdir(parents=True)
    if task == "iq":
        manifest_name, contract = "iq_manifest.json", "iq_waveform_v1"
        payload = {"schema_version": 1, "task": "amc_iq", "id": "iq-cnn", "version": "0.1.0",
                   "contract": contract, "library": library,
                   "input": {"samples": 1024, "channels": 2},
                   "output": {"classes": list(classes)},
                   "training": {"arch": arch, "epochs": 60, "batch_size": 64,
                                "learning_rate": 0.001, "dataset_samples": 4167,
                                "best_validation_accuracy": 0.72}}
    else:
        manifest_name, contract = "detector.json", "tf_image_v1"
        training = {"framework": "rtdetr", "dataset": {"samples": 128}}
        if label_semantics:
            training["label_semantics"] = label_semantics
        payload = {"schema_version": 1, "id": "rtdetr-detector", "version": "0.1.0",
                   "contract": contract, "library": library, "labels": ["emitter"],
                   "input": {"image_size": 256, "spectrogram_nfft": 512,
                             "dynamic_range_db": 60.0},
                   "training": training}
    (model_dir / manifest_name).write_text(json.dumps(payload, ensure_ascii=False),
                                           encoding="utf-8")
    (model_dir / library).write_bytes(b"onnx-bytes-" + run_id.encode())
    if metrics is not None:
        (model_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (directory / "experiment.json").write_text(json.dumps({
        "id": Path(run_id).name, "status": status, "stage": "完成", "metrics": [], "error": None,
        "config": {"task": task, "arch": arch, "workspace": str(workspace)},
    }, ensure_ascii=False), encoding="utf-8")
    return directory


def test_validate_name_rejects_path_and_reserved(tmp_path):
    for bad in ("", "   ", "a/b", "a\\b", "..", ".hidden", "a:b", "x" * 65):
        with pytest.raises(model_store.ModelError):
            model_store.validate_name(bad)
    assert model_store.validate_name("  cnn-amc-20260101T000000  ") == "cnn-amc-20260101T000000"


def test_default_name_is_type_purpose_timestamp(tmp_path):
    from datetime import datetime, timezone

    created = datetime(2026, 10, 6, 10, 52, 13, tzinfo=timezone.utc)
    assert model_store.default_name("cnn", "amc", created) == "cnn-amc-20261006T105213"
    assert model_store.default_name("YOLO26s", "hop", created) == "yolo26s-hop-20261006T105213"
    with pytest.raises(model_store.ModelError):
        model_store.default_name("cnn", "unknown", created)


def test_publish_from_run_registers_and_links_library(tmp_path):
    directory = _write_run(tmp_path, "20261006T105213-7a781642", metrics={"validation": {
        "accuracy": 0.7225, "macro_f1": 0.713814}})
    entry = model_store.publish_from_run(tmp_path, directory)
    assert entry["name"] == "cnn-amc-20261006T105213"
    assert entry["model_type"] == "cnn" and entry["purpose"] == "amc"
    assert entry["purpose_title"] == "调制识别"
    assert entry["status"] == "ok" and entry["size_bytes"] > 0
    assert entry["params"]["window_samples"] == 1024
    assert entry["params"]["classes"] == ["fm", "qpsk"]
    assert entry["metrics"]["accuracy"] == 0.7225

    target = Path(entry["directory"])
    assert (target / "manifest.json").is_file()
    assert json.loads((target / "manifest.json").read_text(encoding="utf-8"))["library"] == "iq.onnx"
    assert (target / "metrics.json").is_file()
    # 同盘优先硬链接，避免大模型复制占双份空间
    assert (target / "iq.onnx").stat().st_ino == (directory / "model/iq.onnx").stat().st_ino


def test_publish_rejects_duplicate_manual_name(tmp_path):
    _write_run(tmp_path, "20261006T105213-7a781642")
    _write_run(tmp_path, "20261006T110000-aaaaaaaa")
    model_store.publish_from_run(tmp_path, tmp_path / "training/runs/20261006T105213-7a781642",
                                 name="我的模型")
    with pytest.raises(model_store.ModelError, match="已存在"):
        model_store.publish_from_run(tmp_path, tmp_path / "training/runs/20261006T110000-aaaaaaaa",
                                     name="我的模型")


def test_publish_same_run_updates_in_place(tmp_path):
    directory = _write_run(tmp_path, "20261006T105213-7a781642")
    first = model_store.publish_from_run(tmp_path, directory, name="我的模型")
    (directory / "model/iq.onnx").write_bytes(b"retrained")
    second = model_store.publish_from_run(tmp_path, directory, name="我的模型")
    assert second["name"] == first["name"]
    assert second["sha256"] != first["sha256"]
    assert len(model_store.list_models(tmp_path)) == 1


def test_auto_named_collision_gets_suffix(tmp_path):
    created = _write_run(tmp_path, "20261006T105213-7a781642")
    other = _write_run(tmp_path, "20261006T105213-bbbbbbbb")
    assert model_store.publish_from_run(tmp_path, created)["name"] == "cnn-amc-20261006T105213"
    assert model_store.publish_from_run(tmp_path, other)["name"] == "cnn-amc-20261006T105213-2"


def test_detection_purpose_splits_session_and_hop(tmp_path):
    session = _write_run(tmp_path, "20261006T110000-aaaaaaaa", task="detection", arch="rtdetr",
                         library="detector.onnx")
    hop = _write_run(tmp_path, "20261006T110001-bbbbbbbb", task="detection", arch="rtdetr",
                     library="detector.onnx", label_semantics="per_hop_v1")
    assert model_store.publish_from_run(tmp_path, session)["purpose"] == "detect"
    assert model_store.publish_from_run(tmp_path, hop)["purpose"] == "hop"


def test_reconcile_registers_existing_runs_once(tmp_path):
    _write_run(tmp_path, "20261006T103044-1f27b913")
    _write_run(tmp_path, "20261006T105213-7a781642", metrics={"validation": {"accuracy": 0.7}})
    _write_run(tmp_path, "20260915T223518-b59737bd", task="torchsig", arch="rtdetr", status="success")
    _write_run(tmp_path, "20261006T120000-cccccccc", status="failed")
    first = model_store.reconcile(tmp_path)
    assert sorted(first["added"]) == ["cnn-amc-20261006T103044", "cnn-amc-20261006T105213"]
    assert first["failed"] == []
    # 幂等：重复扫描不重复登记，也不重复改名
    second = model_store.reconcile(tmp_path)
    assert second == {"added": [], "updated": [], "failed": []}
    assert len(model_store.list_models(tmp_path)) == 2


def test_reconcile_updates_changed_model(tmp_path):
    directory = _write_run(tmp_path, "20261006T105213-7a781642")
    model_store.reconcile(tmp_path)
    (directory / "model/iq.onnx").write_bytes(b"new-weights")
    result = model_store.reconcile(tmp_path)
    assert result["updated"] == ["cnn-amc-20261006T105213"]
    assert result["added"] == []


def test_list_and_inspect_report_damage(tmp_path):
    directory = _write_run(tmp_path, "20261006T105213-7a781642")
    entry = model_store.publish_from_run(tmp_path, directory)
    assert model_store.inspect_model(tmp_path, entry["name"])["digest_ok"] is True
    (Path(entry["directory"]) / "iq.onnx").unlink()
    damaged = model_store.load_model(tmp_path, entry["name"])
    assert damaged["status"] == "missing" and damaged["size_bytes"] == 0
    (Path(entry["directory"]) / model_store.ENTRY_NAME).write_text("{", encoding="utf-8")
    assert model_store.load_model(tmp_path, entry["name"])["status"] == "corrupt"


def test_rename_and_delete_are_confined_to_models_root(tmp_path):
    directory = _write_run(tmp_path, "20261006T105213-7a781642")
    entry = model_store.publish_from_run(tmp_path, directory)
    other = model_store.publish_from_run(tmp_path, _write_run(tmp_path, "20261006T110000-aaaaaaaa"))
    renamed = model_store.rename_model(tmp_path, entry["name"], "巡检模型")
    assert renamed["name"] == "巡检模型"
    assert (Path(renamed["directory"]) / "iq.onnx").is_file()
    assert json.loads((Path(renamed["directory"]) / "model.json").read_text(
        encoding="utf-8"))["name"] == "巡检模型"
    with pytest.raises(model_store.ModelError, match="已存在"):
        model_store.rename_model(tmp_path, renamed["name"], other["name"])
    with pytest.raises(model_store.ModelError):
        model_store.rename_model(tmp_path, renamed["name"], "../escape")

    outside = Path(tmp_path) / "training" / "runs" / "20261006T105213-7a781642" / "model"
    model_store.delete_model(tmp_path, renamed["name"])
    assert not Path(renamed["directory"]).exists()
    assert outside.is_dir()  # 删除模型不动训练运行目录
    with pytest.raises(model_store.ModelError, match="不存在"):
        model_store.delete_model(tmp_path, renamed["name"])


def test_run_directory_helper(tmp_path):
    directory = _write_run(tmp_path, "20261006T105213-7a781642")
    entry = model_store.publish_from_run(tmp_path, directory)
    assert model_store.run_directory(tmp_path, entry) == directory
    assert model_store.run_directory(tmp_path, {}) is None
