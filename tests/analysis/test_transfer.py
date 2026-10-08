"""模型包 / 集合包（含标注）的导出导入：跨工作区迁移的完整回归。

覆盖：
* 模型包：导出 → 体检 → 导入另一个工作区（身份、摘要、可读清单保持一致）、改名与重名后缀；
* 集合包：导出 → 导入（资产、目标与参考参数、AMC/检测标签、覆盖度、来源分组、类别字典）；
* 安全：篡改包（摘要不符）、zip-slip 路径、包类型不匹配一律拒绝；
* CLI：``model-export`` / ``collection-import`` / ``package-inspect`` 的返回码与输出。

模型清单在这里按契约造（含真实 sha256），不依赖 torch：导入校验只读清单与摘要，
不会加载 ONNX。
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from signal_analysis.core_api import generate_iq
from signal_analysis.data import Workspace
from signal_analysis.services import model_store, transfer

pytestmark = []


# --------------------------------------------------------------------------- 造数据


def _iq_manifest(library_bytes, *, classes=("fm", "qpsk"), samples=1024, identifier="iq-cnn"):
    return {"schema_version": 1, "task": "amc_iq", "id": identifier, "version": "0.1.0",
            "contract": "iq_waveform_v1", "runtime": "onnxruntime",
            "library": "iq.onnx",
            "sha256": hashlib.sha256(library_bytes).hexdigest(),
            "input": {"name": "iq", "samples": samples, "channels": 2,
                      "layout": "iq_channels_first_v1"},
            "output": {"name": "scores", "classes": list(classes)},
            "preprocess": {"normalization": "unit_rms"},
            "training": {"arch": "cnn", "epochs": 3, "best_validation_accuracy": 0.9}}


def _detector_manifest(library_bytes):
    return {"schema_version": 1, "id": "rtdetr-detector", "version": "0.1.0",
            "contract": "tf_image_v1", "runtime": "onnxruntime", "library": "detector.onnx",
            "sha256": hashlib.sha256(library_bytes).hexdigest(), "labels": ["emitter"],
            "input": {"image_size": 256, "channels": 1, "spectrogram_nfft": 512,
                      "dynamic_range_db": 60.0,
                      "layout": "time_frequency_grayscale_v1"},
            "output": {"name": "detections"},
            "training": {"framework": "rtdetr", "label_semantics": "session_v1"}}


def publish_run(root, run_id, *, task="iq", arch="cnn", name=None):
    """写一次训练运行并入库，返回模型库条目名。"""
    directory = Path(root) / "training" / "runs" / run_id
    model_dir = directory / "model"
    model_dir.mkdir(parents=True)
    if task == "iq":
        library = f"iq-{run_id}.onnx".encode()
        payload = _iq_manifest(library)
        (model_dir / "iq.onnx").write_bytes(library)
        manifest_name = "iq_manifest.json"
    else:
        library = f"detector-{run_id}.onnx".encode()
        payload = _detector_manifest(library)
        (model_dir / "detector.onnx").write_bytes(library)
        manifest_name = "detector.json"
    (model_dir / manifest_name).write_text(json.dumps(payload, ensure_ascii=False),
                                           encoding="utf-8")
    (model_dir / "metrics.json").write_text(json.dumps({"validation": {"accuracy": 0.9}}),
                                            encoding="utf-8")
    (directory / "experiment.json").write_text(json.dumps({
        "id": run_id, "status": "success", "stage": "完成", "metrics": [], "error": None,
        "config": {"task": task, "arch": arch, "workspace": str(root)},
    }, ensure_ascii=False), encoding="utf-8")
    entry = model_store.publish_from_run(root, directory, name=name)
    return entry["name"]


def add_asset(workspace, mode="fm", seed=1, *, name=None, duration=0.02, group=None):
    samples, summary = generate_iq(
        200_000.0, duration,
        [{"mode": mode, "offset": 0.0, "bandwidth": 50_000.0, "power_dbfs": -6.0}],
        {"enabled": True, "snr_db": 15.0}, seed)
    return workspace.add_samples(samples, 200_000.0, name or f"样本{seed}",
                                 f"generated:iq_{mode}_v1",
                                 metadata={"generation": summary},
                                 origin_group_id=group)


def build_collection(workspace, *, name="集合甲"):
    """一个带 AMC 标注、检测标注与覆盖度的小集合（3 条资产）。"""
    collection = workspace.create_collection(name, description="迁移测试")
    assets = [add_asset(workspace, mode, seed=index * 7 + 1, group=f"grp-{index}")
              for index, mode in enumerate(("fm", "qpsk", "qam16"), start=1)]
    workspace.add_collection_members(collection["id"], [asset["id"] for asset in assets])
    amc = workspace.create_task_set(collection["id"], "amc")
    detection = workspace.create_task_set(collection["id"], "detection")
    for index, asset in enumerate(assets):
        target = workspace.list_targets(asset["id"])[0]
        workspace.append_amc_label(amc["id"], target["id"], source="manual",
                                   class_state="known", class_name=("fm", "qpsk", "qam16")[index])
        workspace.append_detection_label(detection["id"], target["id"], source="manual",
                                         class_name="emitter", include=True)
    workspace.set_asset_coverage(detection["id"], assets[0]["id"], "complete")
    workspace.set_asset_coverage(detection["id"], assets[1]["id"], "partial")
    return collection, assets


def collection_snapshot(workspace, collection_id):
    """把集合的可迁移内容摊平，便于源/目标逐项对比。"""
    task_sets = workspace.list_task_sets(collection_id)
    labels = {}
    for task_set in task_sets:
        for row in workspace.current_labels(task_set["task"], task_set["id"], include_all=True):
            labels.setdefault(row["target_id"], []).append(
                (task_set["task"], row.get("class_name"), row.get("class_state"),
                 row.get("include"), row.get("source")))
    rows = []
    for asset_id in workspace.collection_asset_ids(collection_id):
        asset = workspace.get_asset(asset_id)
        targets = workspace.list_targets(asset_id)
        rows.append({
            "name": asset["name"], "rate": asset["sample_rate"],
            "kind": asset.get("source_kind"), "group": asset.get("origin_group_id"),
            "count": asset.get("sample_count"),
            "targets": sorted((t["target_key"], t["scope"], t["for_detection"], t["for_amc"],
                               (t.get("current") or {}).get("modulation"),
                               (t.get("current") or {}).get("waveform_mode"),
                               (t.get("current") or {}).get("snr_db"),
                               (t.get("current") or {}).get("f_low_hz"))
                              for t in targets),
            "labels": sorted(item for t in targets for item in labels.get(t["id"], [])),
            "coverage": sorted((task_set["task"],
                                (workspace.get_asset_coverage(task_set["id"], asset_id) or {}
                                 ).get("coverage"))
                               for task_set in task_sets),
        })
    return rows


# --------------------------------------------------------------------------- 模型包


def test_model_package_round_trip_into_another_workspace(tmp_path):
    source = tmp_path / "source"
    publish_run(source, "20261008T000000-aaaaaaaa", name="cnn-amc-1")
    publish_run(source, "20261008T000001-bbbbbbbb", task="detection", arch="rtdetr",
                name="rtdetr-detect-1")
    archive = tmp_path / "models.zip"

    report = transfer.export_models(source, archive)
    assert sorted(report["models"]) == ["cnn-amc-1", "rtdetr-detect-1"]
    assert report["size_bytes"] > 0
    inspected = transfer.inspect_model_package(archive)
    assert all(item["ok"] for item in inspected["models"])
    assert {item["task"] for item in inspected["models"]} == {"iq", "detection"}

    target = tmp_path / "target"
    result = transfer.import_models(target, archive)
    assert {item["name"] for item in result["models"]} == {"cnn-amc-1", "rtdetr-detect-1"}
    entries = {entry["name"]: entry for entry in model_store.list_models(target)}
    source_entries = {entry["name"]: entry for entry in model_store.list_models(source)}
    for name, purpose in (("cnn-amc-1", "amc"), ("rtdetr-detect-1", "detect")):
        entry = entries[name]
        assert entry["status"] == "ok" and entry["purpose"] == purpose
        assert entry["sha256"] == source_entries[name]["sha256"]      # 权重摘要逐字节一致
        assert model_store.inspect_model(target, name)["digest_ok"] is True
    assert entries["cnn-amc-1"]["params"]["window_samples"] == 1024


def test_model_import_rename_and_never_overwrites(tmp_path):
    source = tmp_path / "source"
    publish_run(source, "20261008T000000-aaaaaaaa", name="cnn-amc-1")
    archive = tmp_path / "one.zip"
    transfer.export_models(source, archive)

    target = tmp_path / "target"
    assert transfer.import_models(target, archive)["models"][0]["name"] == "cnn-amc-1"
    again = transfer.import_models(target, archive)["models"][0]["name"]
    assert again == "cnn-amc-1 (2)"          # 不覆盖，自动加序号
    renamed = transfer.import_models(target, archive, name="我的新模型")["models"][0]
    assert renamed["name"] == "我的新模型"
    entry = model_store.load_model(target, "我的新模型")
    assert entry["status"] == "ok" and entry["name"] == "我的新模型"
    with pytest.raises(transfer.TransferError, match="名称"):
        transfer.import_models(target, archive, name="bad/name")

    many = tmp_path / "many.zip"
    publish_run(source, "20261008T000002-cccccccc", name="cnn-amc-2")
    transfer.export_models(source, many)
    with pytest.raises(transfer.TransferError, match="不能指定目标名称"):
        transfer.import_models(target, many, name="再来一个")


def test_model_package_rejects_tampering_and_wrong_kind(tmp_path):
    source = tmp_path / "source"
    publish_run(source, "20261008T000000-aaaaaaaa", name="cnn-amc-1")
    archive = tmp_path / "models.zip"
    transfer.export_models(source, archive)

    # 篡改一个文件的内容（摘要不符）
    broken = tmp_path / "broken.zip"
    with zipfile.ZipFile(archive) as source_zip, \
            zipfile.ZipFile(broken, "w", zipfile.ZIP_STORED) as target_zip:
        for info in source_zip.infolist():
            payload = source_zip.read(info.filename)
            if info.filename.endswith("metrics.json"):
                payload = payload.replace(b"0.9", b"0.1")
            target_zip.writestr(info.filename, payload)
    assert not all(item["ok"] for item in transfer.inspect_model_package(broken)["models"])
    with pytest.raises(transfer.TransferError, match="摘要不符"):
        transfer.import_models(tmp_path / "target", broken)

    # zip-slip：条目名带上级目录
    hostile = tmp_path / "hostile.zip"
    with zipfile.ZipFile(hostile, "w") as handle:
        handle.writestr("../evil.txt", b"boom")
    with pytest.raises(transfer.TransferError, match="条目名非法"):
        transfer.inspect_model_package(hostile)

    # 包类型不匹配：集合包当模型包
    workspace = Workspace(tmp_path / "ws")
    collection, _ = build_collection(workspace)
    collection_zip = tmp_path / "collection.zip"
    transfer.export_collection(workspace, collection["id"], collection_zip)
    with pytest.raises(transfer.TransferError, match="不是一个.*集合包"):
        transfer.import_models(tmp_path / "target", collection_zip)
    with pytest.raises(transfer.TransferError, match="不是一个.*模型包"):
        transfer.import_collection(workspace, archive)


def test_model_package_reports_missing_model(tmp_path):
    source = tmp_path / "source"
    publish_run(source, "20261008T000000-aaaaaaaa", name="cnn-amc-1")
    with pytest.raises(transfer.TransferError, match="没有模型"):
        transfer.export_models(source, tmp_path / "x.zip", names=["nope"])


# --------------------------------------------------------------------------- 集合包


def test_collection_package_round_trip_keeps_annotations(tmp_path):
    source = Workspace(tmp_path / "source")
    collection, _ = build_collection(source)
    archive = tmp_path / "collection.zip"
    report = transfer.export_collection(source, collection["id"], archive)
    assert report["assets"] == 3 and report["targets"] == 3
    assert report["labels"] == 6 and report["coverage"] == 2

    inspected = transfer.inspect_collection_package(archive)
    assert inspected["ok"] and inspected["counts"]["assets"] == 3
    assert inspected["collection"]["name"] == "集合甲"
    assert [item["task"] for item in inspected["task_sets"]] == ["amc", "detection"]

    target = Workspace(tmp_path / "target")
    result = transfer.import_collection(target, archive)
    assert (result["assets"], result["targets"], result["labels"], result["coverage"]) \
        == (3, 3, 6, 2)
    assert result["skipped_labels"] == []
    imported = next(row for row in target.list_collections() if row["name"] == "集合甲")
    assert collection_snapshot(source, collection["id"]) == \
        collection_snapshot(target, imported["id"])

    # 再导入一次：新建而不是覆盖
    again = transfer.import_collection(target, archive)
    assert again["collection"] == "集合甲 (2)"
    assert len(target.list_collections()) == 2


def test_collection_import_renames_and_reports_counts(tmp_path):
    source = Workspace(tmp_path / "source")
    collection, _ = build_collection(source)
    archive = tmp_path / "collection.zip"
    transfer.export_collection(source, collection["id"], archive)

    target = Workspace(tmp_path / "target")
    result = transfer.import_collection(target, archive, name="换台机器的集合")
    assert result["collection"] == "换台机器的集合"
    row = target.get_collection(result["collection_id"])
    assert row["name"] == "换台机器的集合"
    assert target.asset_label_status(target.collection_asset_ids(result["collection_id"])[0])
    summary = target.collection_summary(result["collection_id"])
    assert summary["asset_count"] == 3 and summary["target_count"] == 3


def test_collection_package_rejects_tampered_assets(tmp_path):
    source = Workspace(tmp_path / "source")
    collection, _ = build_collection(source)
    archive = tmp_path / "collection.zip"
    transfer.export_collection(source, collection["id"], archive)

    broken = tmp_path / "broken.zip"
    with zipfile.ZipFile(archive) as source_zip, \
            zipfile.ZipFile(broken, "w", zipfile.ZIP_STORED) as target_zip:
        for info in source_zip.infolist():
            payload = source_zip.read(info.filename)
            if info.filename.startswith("assets/"):
                payload = payload + b"\x00"          # 资产内容被改
            target_zip.writestr(info.filename, payload)
    with pytest.raises(transfer.TransferError, match="摘要不符"):
        transfer.import_collection(Workspace(tmp_path / "target"), broken)

    missing = tmp_path / "missing.zip"
    with zipfile.ZipFile(archive) as source_zip, \
            zipfile.ZipFile(missing, "w", zipfile.ZIP_STORED) as target_zip:
        for info in source_zip.infolist():
            if info.filename.startswith("assets/"):
                continue
            target_zip.writestr(info.filename, source_zip.read(info.filename))
    with pytest.raises(transfer.TransferError, match="缺少资产文件"):
        transfer.import_collection(Workspace(tmp_path / "target2"), missing)


def test_collection_import_failure_leaves_no_debris(tmp_path):
    """导入是整体事务：中途失败不能留下半个集合，也不能留下已落盘的资产文件。"""
    source = Workspace(tmp_path / "source")
    collection, _ = build_collection(source)
    archive = tmp_path / "collection.zip"
    transfer.export_collection(source, collection["id"], archive)

    broken = tmp_path / "broken-late.zip"
    with zipfile.ZipFile(archive) as source_zip, \
            zipfile.ZipFile(broken, "w", zipfile.ZIP_STORED) as target_zip:
        names = [info.filename for info in source_zip.infolist()]
        late = sorted(name for name in names if name.startswith("assets/"))[-1]
        for name in names:
            info = source_zip.getinfo(name)
            payload = source_zip.read(name)
            if name == late:                         # 最后一条资产才损坏
                payload = payload[:-8] + b"\x00" * 8
            target_zip.writestr(info, payload)

    target = Workspace(tmp_path / "target")
    with pytest.raises(transfer.TransferError, match="摘要不符"):
        transfer.import_collection(target, broken)
    assert target.list_collections(include_archived=True) == []
    assert target.list_task_sets() == []
    assert target.list_assets() == []
    assert list((target.root / "assets").glob("*")) == []
    assert target.list_taxonomies("amc") == [] or all(
        row["name"] != "六类" for row in target.list_taxonomies("amc"))
    assert transfer.import_collection(target, archive)["assets"] == 3


def test_workspace_batch_rolls_back_on_error(tmp_path):
    """批量事务是数据层的通用能力：失败回滚、可重入、退出后恢复逐条提交。"""
    workspace = Workspace(tmp_path / "ws")
    with pytest.raises(RuntimeError):
        with workspace.batch():
            with workspace.batch():                  # 可重入
                workspace.create_collection("会被回滚")
            raise RuntimeError("模拟失败")
    assert workspace.list_collections() == []

    with workspace.batch():
        workspace.create_collection("批量内可见")
        assert [row["name"] for row in workspace.list_collections()] == ["批量内可见"]
    assert [row["name"] for row in workspace.list_collections()] == ["批量内可见"]
    workspace.create_collection("事务外追加")
    assert len(workspace.list_collections()) == 2


def test_collection_import_preserves_origin_group_for_leak_checks(tmp_path):
    """来源分组必须随包迁移，否则训练前的泄漏检查会失效。"""
    from signal_analysis.services.training_inputs import inputs_plan

    source = Workspace(tmp_path / "source")
    collection, _ = build_collection(source)
    archive = tmp_path / "collection.zip"
    transfer.export_collection(source, collection["id"], archive)

    target = Workspace(tmp_path / "target")
    first = transfer.import_collection(target, archive)
    second = transfer.import_collection(target, archive)
    groups = {target.get_asset(asset_id)["origin_group_id"]
              for asset_id in target.collection_asset_ids(first["collection_id"])}
    assert groups == {"grp-1", "grp-2", "grp-3"}
    with pytest.raises(ValueError, match="同源"):
        inputs_plan(target, "amc", first["collection_id"], second["collection_id"])


def test_default_package_name_is_cross_platform_safe():
    """默认文件名由集合/模型名拼出，必须避开 Windows 禁用字符与路径分隔符。"""
    name = transfer.default_package_name('集合包-训练:集/合*\\ "甲"', stamp="20261008T120000")
    assert name == "集合包-训练_集_合__ _甲_-20261008T120000.zip"
    assert not set('<>:"/\\|?*') & set(name)
    assert transfer.default_package_name("   ", stamp="20261008T120000") \
        == "package-20261008T120000.zip"


def test_collection_export_requires_assets(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    empty = workspace.create_collection("空的")
    with pytest.raises(transfer.TransferError, match="没有任何资产"):
        transfer.export_collection(workspace, empty["id"], tmp_path / "x.zip")


# --------------------------------------------------------------------------- CLI


def test_cli_pack_commands(tmp_path, capsys):
    from signal_analysis import cli

    source = tmp_path / "source"
    publish_run(source, "20261008T000000-aaaaaaaa", name="cnn-amc-1")
    workspace = Workspace(source)
    collection, _ = build_collection(workspace)

    pack = tmp_path / "models.zip"
    assert cli.main(["--workspace", str(source), "model-export",
                     "--output", str(pack)]) == 0
    assert json.loads(capsys.readouterr().out)["models"] == ["cnn-amc-1"]

    target = tmp_path / "target"
    assert cli.main(["--workspace", str(target), "model-import", str(pack)]) == 0
    assert json.loads(capsys.readouterr().out)["models"][0]["name"] == "cnn-amc-1"
    assert cli.main(["--workspace", str(target), "package-inspect", str(pack)]) == 0
    assert json.loads(capsys.readouterr().out)["kind"] == transfer.MODEL_PACKAGE_KIND

    collection_pack = tmp_path / "collection.zip"
    assert cli.main(["--workspace", str(source), "collection-export",
                     "--collection", "集合甲", "--output", str(collection_pack)]) == 0
    assert json.loads(capsys.readouterr().out)["assets"] == 3
    assert cli.main(["--workspace", str(target), "collection-import",
                     str(collection_pack), "--name", "导入集合"]) == 0
    assert json.loads(capsys.readouterr().out)["collection"] == "导入集合"
    assert cli.main(["--workspace", str(target), "package-inspect",
                     str(collection_pack)]) == 0
    assert json.loads(capsys.readouterr().out)["kind"] == transfer.COLLECTION_PACKAGE_KIND

    # 错误路径：找不到集合、包不可用，都返回 1 并打印原因
    assert cli.main(["--workspace", str(source), "collection-export",
                     "--collection", "不存在", "--output", str(tmp_path / "x.zip")]) == 1
    assert "找不到集合" in capsys.readouterr().err
    assert cli.main(["--workspace", str(target), "model-import",
                     str(collection_pack)]) == 1
    assert "不是一个" in capsys.readouterr().err   # 类型不符：提示用对应的导入方式


# --------------------------------------------------------------------------- GUI


@pytest.mark.gui
def test_gui_model_and_collection_package_buttons(tmp_path, monkeypatch):
    """界面按钮走通：模型包导出→另一个工作区导入；集合包导出→导入并刷新列表。"""
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from PySide6 import QtWidgets

    from signal_analysis.ui import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    source_root = tmp_path / "source"
    publish_run(source_root, "20261008T000000-aaaaaaaa", name="cnn-amc-1")
    window = MainWindow(source_root)
    try:
        # 模型包导出：把保存对话框固定到临时路径
        model_pack = tmp_path / "models.zip"
        monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                            staticmethod(lambda *args, **kwargs: (str(model_pack), "")))
        window.models_export_button.click()
        app.processEvents()
        assert model_pack.is_file()
        assert "已导出 1 个模型" in window.model_hint.text()

        # 集合：先造一个有标注的集合，再导出
        collection, _ = build_collection(window.workspace)
        window.refresh_collections_panel()
        collection_pack = tmp_path / "collection.zip"
        monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                            staticmethod(lambda *args, **kwargs: (str(collection_pack), "")))
        app.processEvents()
        window.export_collection_button.click()
        app.processEvents()
        assert collection_pack.is_file()
        assert "导出" in window.collection_detail.toPlainText()

        # 在另一个工作区导入模型包与集合包
        target = MainWindow(tmp_path / "target")
        try:
            # 导入前的确认框：离屏环境下统一回答"是"
            monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                                staticmethod(lambda *args, **kwargs:
                                             QtWidgets.QMessageBox.StandardButton.Yes))
            monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                                staticmethod(lambda *args, **kwargs: (str(model_pack), "")))
            target.models_import_button.click()
            app.processEvents()
            assert "已导入 1 个模型" in target.model_hint.text()
            assert [entry["name"] for entry in model_store.list_models(target.workspace.root)] \
                == ["cnn-amc-1"]

            monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                                staticmethod(lambda *args, **kwargs: (str(collection_pack), "")))
            target.import_collection_button.click()
            app.processEvents()
            assert [row["name"] for row in target.workspace.list_collections()] == ["集合甲"]
            assert "已导入集合" in target.status.text()
        finally:
            target.close()
        window.refresh_collections_panel()
        assert collection["id"]
    finally:
        window.close()
        app.processEvents()
