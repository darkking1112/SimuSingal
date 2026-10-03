"""信号集合生成的 GUI 流程（表单 → 执行器 → 集合/标注）与训练页数据来源。"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6 import QtWidgets

from signal_analysis.ui import MainWindow
from signal_analysis.ui.pages.collection_gen_page import _BundleImportDialog
from signal_analysis.data import Workspace
from signal_analysis.tasks import run_job

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))


def wait_job(app, window, timeout=60.0):
    deadline = time.monotonic() + timeout
    while window.active_job is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.02)
    app.processEvents()
    assert window.active_job is None, "GUI 任务未在预期时间内结束"


def small_settings(panel, count=4, modes=("qpsk", "fm")):
    """把页面参数改成小批量、少信号的组合，方便 GUI 测试快速跑完。"""
    panel.count.setValue(count)
    panel.settings.update({"signals": (1, 1), "snr": (15.0, 25.0),
                           "bandwidth_ratio": (0.05, 0.08), "power": (-10.0, -6.0),
                           "duration": (0.05, 0.08), "modes": list(modes)})


@pytest.mark.gui
def test_generation_from_form_creates_collection_with_auto_labels(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(tmp_path)
    window.show()
    try:
        panel = window.gen_panel
        small_settings(panel, count=4, modes=("qpsk", "fh_rc"))
        panel.collection.setCurrentIndex(panel.collection.findData("__new__"))
        panel.collection_name.setText("界面集合")
        panel.start_generation()
        wait_job(app, window)
        assert window.last_result["kind"] == "generate_collection"
        assert window.last_result["created"] == 4
        text = panel.result_label.text()
        assert "新建资产 4" in text and "目标集合：界面集合" in text
        assert "逐跳" in text and "初始标注" in text
        assert "集合生成完成" in window.status.text()
    finally:
        window.close()
        app.processEvents()

    workspace = Workspace(tmp_path)
    collection = next(item for item in workspace.list_collections()
                      if item["name"] == "界面集合")
    assert collection["source_kind"] == "generated" and collection["recipe_id"]
    recipe = json.loads(workspace.get_recipe(collection["recipe_id"])["recipe_json"])
    assert recipe["labels"]["detection"] == "per_hop_v1"
    task_sets = {item["task"]: item for item in workspace.list_task_sets(collection["id"])}
    assert task_sets["detection"]["label_semantics"] == "per_hop_v1"
    assert workspace.current_labels("amc", task_sets["amc"]["id"], include_all=True)


@pytest.mark.gui
def test_append_to_existing_collection_and_missing_name(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(tmp_path)
    window.show()
    try:
        panel = window.gen_panel
        small_settings(panel, count=2, modes=("qpsk",))
        panel.collection.setCurrentIndex(panel.collection.findData("__new__"))
        panel.start_generation()
        assert window.active_job is None and "请填写集合名称" in panel.status.text()
        collection = window.workspace.create_collection("既有集合")
        window.refresh_collections()
        index = panel.collection.findData(collection["id"])
        assert index >= 0
        panel.collection.setCurrentIndex(index)
        panel.start_generation()
        wait_job(app, window)
        assert window.last_result["collection_id"] == collection["id"]
        assert window.last_result["created"] == 2
    finally:
        window.close()
        app.processEvents()
    workspace = Workspace(tmp_path)
    assert workspace.count_assets(collection_id=collection["id"]) == 2


@pytest.mark.gui
def test_training_pages_use_collections_without_generation_or_exports(tmp_path):
    """两个训练页只用集合：没有生成入口、没有导出入口、没有数据集来源下拉。"""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(tmp_path)
    window.show()
    try:
        page = window.detection_training_page
        amc_page = window.amc_training_page
        texts = [button.text() for button in page.findChildren(QtWidgets.QPushButton)]
        assert not any("准备检测数据" in text for text in texts)
        assert not any("导出训练数据" in text for text in texts)
        assert page.configuration()["task"] == "detection"
        assert amc_page.configuration()["task"] == "iq"
        for training_page in (page, amc_page):
            for field in ("data", "source", "collection", "export_button"):
                assert not hasattr(training_page, field), field
        assert set(page.configuration()) == {
            "repository", "python", "task", "arch", "workspace", "train_collection_id",
            "val_collection_id", "weights", "framework_path", "framework_config",
            "device", "epochs", "batch", "lr", "seed", "image_size", "nfft"}
        assert amc_page.sections.tabText(0) == "信号标注"

        workspace = window.workspace
        result = run_job({"workspace": str(workspace.root), "action": "generate_collection",
                          "collection_name": "训练用集合",
                          "recipe": {"contract": "gen_recipe_v1", "engine": "project",
                                     "base_seed": 3, "count": 6,
                                     "record": {"sample_rate_hz": {"fixed": 200000.0},
                                                "duration_s": {"fixed": 0.05}},
                                     "signals": {"count": {"fixed": 1},
                                                 "mode": {"balanced": ["qpsk", "fm"]},
                                                 "bandwidth_ratio": {"fixed": 0.06},
                                                 "snr_db": {"uniform": [12, 25]},
                                                 "power_dbfs": {"fixed": -8.0}},
                                     "labels": {"detection": "session_v1", "amc": True}}},
                       timeout=120)
        window.refresh_collections()
        index = window.collection_combo.findData(result["collection_id"])
        assert index >= 0
        window.collection_combo.setCurrentIndex(index)
        # 训练集默认跟随侧栏集合；两页共用同一集合（验证集需另外指定）
        assert page.train_collection.currentData() == result["collection_id"]
        assert amc_page.train_collection.currentData() == result["collection_id"]
        # 训练输入直接来自集合：不产生导出目录，也不登记数据版本
        assert not (workspace.root / "training" / "datasets").exists()
        assert workspace.list_dataset_versions() == []
    finally:
        window.close()
        app.processEvents()


@pytest.mark.gui
def test_torchsig_environment_dialog_and_bundle_import(tmp_path):
    from torchsig_bundle import component, record, write_bundle

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(tmp_path)
    window.show()
    try:
        panel = window.gen_panel
        dialog = panel.env_dialog or None
        panel.open_environment()
        dialog = panel.env_dialog
        assert dialog is not None
        dialog.test()
        if not sys.platform.startswith("linux"):
            assert "只能在 Linux" in dialog.status.text()
        dialog.repository.setText(str(tmp_path / "missing"))
        dialog.save()
        from signal_analysis.integrations.torchsig import read_env

        assert read_env(window.workspace)["repository"] == str(tmp_path / "missing")
        dialog.close()

        # bundle 导入（任何平台）：清单对话框 → 后台任务 → 集合与目标
        bundle = tmp_path / "bundle"
        rate = 100000.0
        comps = [component(class_name="QPSK", center_freq=-20000.0, bandwidth=10000.0,
                           start_in_samples=100, duration_in_samples=4000, snr_db=18.0)]
        iq = (1 + 1j) * __import__("numpy").ones(10000, dtype="complex64")
        write_bundle(bundle, [(record(index=index, iq=iq, components=comps), iq)
                              for index in range(3)], sample_rate_hz=rate)
        mapping = tmp_path / "map.json"
        mapping.write_text('{"QPSK": "qpsk"}', encoding="utf-8")
        import_dialog = _BundleImportDialog(panel, bundle)
        import_dialog.mapping.setText(str(mapping))
        import_dialog.collection_name.setText("bundle 集合")
        request = import_dialog.request()
        assert request["bundle_path"] == str(bundle) and request["mapping"] == {"qpsk": "qpsk"}
        window.start_job("torchsig_import", **request)
        wait_job(app, window)
        result = window.last_result
        assert result["kind"] == "generate_collection" and result["created"] == 3
        assert result["collection_name"] == "bundle 集合"
        workspace = Workspace(tmp_path)
        collection = workspace.get_collection(result["collection_id"])
        assets = workspace.collection_asset_ids(collection["id"])
        targets = workspace.list_targets(assets[0])
        assert targets and targets[0]["current"]["source"] == "external"
    finally:
        window.close()
        app.processEvents()
