"""模型库的界面接线：三个页面的模型下拉、模型管理子页与既有模型登记。"""
import json
from pathlib import Path

import pytest

from signal_analysis.services import model_store

pytestmark = pytest.mark.gui


def _write_run(workspace, run_id, *, task="iq", arch="cnn", label_semantics=None,
               library="iq.onnx"):
    """造一次成功训练的落盘产物，供模型库登记使用。"""
    directory = Path(workspace) / "training" / "runs" / run_id
    model_dir = directory / "model"
    model_dir.mkdir(parents=True)
    if task == "iq":
        manifest_name = "iq_manifest.json"
        payload = {"schema_version": 1, "task": "amc_iq", "id": "iq-cnn", "version": "0.1.0",
                   "contract": "iq_waveform_v1", "library": library,
                   "input": {"samples": 1024, "channels": 2},
                   "output": {"classes": ["fm", "qpsk"]},
                   "training": {"arch": arch, "epochs": 60, "batch_size": 64,
                                "learning_rate": 0.001, "best_validation_accuracy": 0.72}}
    else:
        manifest_name = "detector.json"
        training = {"framework": "rtdetr", "dataset": {"samples": 128}}
        if label_semantics:
            training["label_semantics"] = label_semantics
        payload = {"schema_version": 1, "id": "rtdetr-detector", "version": "0.1.0",
                   "contract": "tf_image_v1", "library": library, "labels": ["emitter"],
                   "input": {"image_size": 256, "spectrogram_nfft": 512,
                             "dynamic_range_db": 60.0},
                   "training": training}
    (model_dir / manifest_name).write_text(json.dumps(payload, ensure_ascii=False),
                                           encoding="utf-8")
    (model_dir / library).write_bytes(b"onnx-" + run_id.encode())
    (directory / "experiment.json").write_text(json.dumps({
        "id": run_id, "status": "success", "stage": "完成", "metrics": [], "error": None,
        "config": {"task": task, "arch": arch, "workspace": str(workspace)},
    }, ensure_ascii=False), encoding="utf-8")
    return directory


def _window(tmp_path):
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from PySide6 import QtWidgets

    from signal_analysis.ui import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(tmp_path / "ws")
    window.show()
    return app, window


def _items(combo):
    return [combo.itemText(index) for index in range(combo.count())]


def _names(combo):
    """下拉里已登记模型的名称（标签首段，去掉“ · 类型 · 日期”后缀）。"""
    return [item.split(" · ")[0].split("（")[0].split("：")[-1] for item in _items(combo)]


def test_startup_registers_existing_models_into_matching_pickers(tmp_path):
    workspace = tmp_path / "ws"
    _write_run(workspace, "20261006T105213-7a781642")
    _write_run(workspace, "20261006T110000-aaaaaaaa", task="detection", arch="rtdetr",
               library="detector.onnx")
    _write_run(workspace, "20261006T110001-bbbbbbbb", task="detection", arch="rtdetr",
               library="detector.onnx", label_semantics="per_hop_v1")
    app, window = _window(tmp_path)
    try:
        assert "cnn-amc-20261006T105213" in _names(window.amc_picker)
        assert "rtdetr-detect-20261006T110000" in _names(window.ml_picker)
        assert "rtdetr-hop-20261006T110001" in _names(window.hops_picker)
        # 用途过滤：检测页看不到 AMC 模型，识别页也看不到检测与逐跳模型
        assert "cnn-amc-20261006T105213" not in _names(window.ml_picker)
        assert "rtdetr-detect-20261006T110000" not in _names(window.amc_picker)
        assert "rtdetr-hop-20261006T110001" not in _names(window.ml_picker)

        # 下拉选中即把清单路径写回 QLineEdit（识别页据此分流原始 IQ 通路）
        index = window.amc_picker.findData(str(workspace / "training/models"
                                                / "cnn-amc-20261006T105213/manifest.json"))
        window.amc_picker.setCurrentIndex(index)
        assert window.amc_model.text().endswith("cnn-amc-20261006T105213/manifest.json")
        # 直接写入未登记清单时下拉显示“未登记”，不丢失外部模型
        window.ml_manifest.setText("/tmp/outside/detector.json")
        assert "未登记" in window.ml_picker.currentText()
    finally:
        window.close()
        app.processEvents()


def test_models_tab_lists_detail_and_actions(tmp_path, monkeypatch):
    from PySide6 import QtWidgets

    workspace = tmp_path / "ws"
    _write_run(workspace, "20261006T105213-7a781642")
    app, window = _window(tmp_path)
    try:
        assert window.storage_tables.tabText(1) == "模型管理"
        assert window.models_table.rowCount() == 1
        assert window.models_table.item(0, 0).text() == "cnn-amc-20261006T105213"
        assert window.models_table.item(0, 2).text() == "调制识别"
        assert window.models_table.item(0, 7).text() == "正常"
        detail = window.model_detail.toPlainText()
        assert "cnn-amc-20261006T105213" in detail and "SHA-256" in detail
        assert "window_samples=1024" in detail

        # 在页面中使用：切到调制识别页并填好清单
        window.use_selected_model()
        assert window.tabs.tabText(window.tabs.currentIndex()) == "调制识别"
        assert window.amc_model.text().endswith("manifest.json")

        # 重命名：只改模型库目录名，训练运行目录不动
        monkeypatch.setattr(QtWidgets.QInputDialog, "getText",
                            staticmethod(lambda *args, **kwargs: ("巡检模型", True)))
        window.rename_selected_model()
        assert window.models_table.item(0, 0).text() == "巡检模型"
        assert "巡检模型" in _names(window.amc_picker)
        run_model = workspace / "training/runs/20261006T105213-7a781642/model/iq.onnx"
        assert run_model.is_file()

        # 删除：只删模型库条目
        monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                            staticmethod(lambda *args, **kwargs:
                                         QtWidgets.QMessageBox.StandardButton.Yes))
        window.delete_selected_model()
        assert window.models_table.rowCount() == 0
        assert not (workspace / "training/models/巡检模型").exists()
        assert run_model.is_file()
    finally:
        window.close()
        app.processEvents()


def test_scan_button_registers_models_added_after_startup(tmp_path):
    workspace = tmp_path / "ws"
    app, window = _window(tmp_path)
    try:
        assert window.models_table.rowCount() == 0
        _write_run(workspace, "20261006T105213-7a781642")
        window.scan_model_library()
        assert window.models_table.rowCount() == 1
        assert "cnn-amc-20261006T105213" in window.model_hint.text()
    finally:
        window.close()
        app.processEvents()


def test_training_page_blocks_duplicate_manual_model_name(tmp_path):
    workspace = tmp_path / "ws"
    _write_run(workspace, "20261006T105213-7a781642")
    app, window = _window(tmp_path)
    try:
        page = window.amc_training_page
        assert page.model_name.placeholderText().startswith("留空自动命名")
        page.validate_model_name({"model_name": ""})
        page.validate_model_name({"model_name": "新模型"})
        with pytest.raises(ValueError, match="已存在"):
            page.validate_model_name({"model_name": "cnn-amc-20261006T105213"})
        with pytest.raises(model_store.ModelError):
            page.validate_model_name({"model_name": "a/b"})
    finally:
        window.close()
        app.processEvents()


def test_model_row_exposes_only_the_picker(tmp_path):
    """三页的模型行只留下拉：首项文案统一，路径输入框隐藏、“选择…”按钮删除。"""
    from signal_analysis.ui.model_picker import DEFAULT_EMPTY_TEXT

    app, window = _window(tmp_path)
    try:
        assert DEFAULT_EMPTY_TEXT == "默认（不使用已登记模型）"
        for picker in (window.ml_picker, window.amc_picker, window.hops_picker):
            assert picker.itemText(0) == DEFAULT_EMPTY_TEXT
            assert picker.itemText(picker.count() - 1) == "浏览本地文件…"
        # 路径仍由隐藏输入框承载：外部清单照样能写进去并被下拉识别
        for field in (window.ml_manifest, window.amc_model, window.hops_manifest):
            assert field.isHidden()
        window.ml_manifest.setText("/tmp/outside/detector.json")
        assert "未登记" in window.ml_picker.currentText()
        # “选择…”按钮已被下拉里的“浏览本地文件…”取代
        for removed in ("ml_choose", "amc_choose", "hops_ml_choose"):
            assert not hasattr(window, removed)
        # 补充说明（默认通路是什么）改挂在 tooltip 上，不再依赖看不见的占位文本
        assert "传统能量检测" in window.ml_picker.toolTip()
        assert "内置线性基线" in window.amc_picker.toolTip()
        assert "per_hop_v1" in window.hops_picker.toolTip()
    finally:
        window.close()
        app.processEvents()
