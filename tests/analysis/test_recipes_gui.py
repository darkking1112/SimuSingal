"""生成参数表单（配置弹框）与高级 JSON 导入导出的 GUI 测试（方案 §6.3）。"""
import json
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6 import QtWidgets

from signal_analysis.collection_gen_gui import CollectionGenPanel, SignalParamsDialog
from signal_analysis.gui import MainWindow
from signal_analysis.recipes import check_generator_support, validate_recipe


def wait_job(app, window, timeout=30.0):
    deadline = time.monotonic() + timeout
    while window.active_job is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.02)
    app.processEvents()
    assert window.active_job is None, "GUI 任务未在预期时间内结束"


@pytest.fixture
def window(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    widget = MainWindow(tmp_path)
    widget.show()
    yield app, widget
    widget.close()
    app.processEvents()


@pytest.mark.gui
def test_generation_page_is_a_form_with_greyed_impairments(window):
    app, widget = window
    panel = widget.gen_panel
    assert isinstance(panel, CollectionGenPanel)
    generator = widget.tabs.widget(widget._page_index("IQ 信号生成"))
    assert generator.isAncestorOf(panel)
    # 两个子页：单个信号生成 / 信号集合生成
    sections = generator.findChild(QtWidgets.QTabWidget, "gen_sections")
    assert sections is not None and sections.count() == 2
    assert sections.tabText(1) == "信号集合生成"
    # 损伤先显示但置灰；接口在 impairments 里
    assert panel.impairment_checks and not any(
        check.isEnabled() for check in panel.impairment_checks.values())
    # 默认：检测=逐跳（会话+逐跳目标）、AMC 勾选
    recipe = panel.recipe()
    assert recipe["labels"] == {"detection": "per_hop_v1", "amc": True}
    assert recipe["engine"] == "project" and recipe["count"] == 200
    assert recipe["signals"]["mode"]["balanced"]  # 默认类别均衡
    assert "hopping" in recipe
    # 页面上没有可手写 JSON 的编辑框；明细在配置弹框里
    assert not any(not item.isReadOnly()
                   for item in panel.findChildren(QtWidgets.QPlainTextEdit))
    dialog = SignalParamsDialog(panel, panel.settings, "project")
    assert dialog.windowTitle() == "生成参数 · 信号与采样"
    dialog.reject()


@pytest.mark.gui
def test_preview_export_and_load_json(window, tmp_path, monkeypatch):
    app, widget = window
    panel = widget.gen_panel
    panel.count.setValue(20)
    panel.preview_recipe()
    wait_job(app, widget)
    assert "样本数 20" in panel.preview.toPlainText()
    assert "未合成 IQ" in widget.status.text()
    # 高级：导出生成参数 JSON（内部格式，仅用于迁移/交接）
    target = tmp_path / "生成参数.json"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *args, **kwargs: (str(target), "JSON")))
    panel.export_recipe()
    assert target.is_file()
    exported = json.loads(target.read_text(encoding="utf-8"))
    validate_recipe(exported)
    check_generator_support(exported)
    # 高级：载入会回填表单（而不是替换成不可编辑的 JSON）
    exported["count"] = 5
    exported["signals"]["mode"] = {"choice": ["fm"]}
    exported["labels"]["detection"] = "session_v1"
    edited = tmp_path / "编辑.json"
    edited.write_text(json.dumps(exported, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *args, **kwargs: (str(edited), "JSON")))
    panel.load_recipe()
    assert panel.count.value() == 5
    assert panel.settings["modes"] == ["fm"] and not panel.settings["balanced"]
    assert panel.detection.isChecked() and panel.detection_scope.currentData() == "session_v1"
    again = panel.recipe()
    assert again["count"] == 5 and again["signals"]["mode"] == {"choice": ["fm"]}
    assert again["labels"] == {"detection": "session_v1", "amc": True}
    # 非法 JSON 不改变现有配置
    broken = tmp_path / "坏.json"
    broken.write_text('{"contract": "gen_recipe_v9"}', encoding="utf-8")
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *args, **kwargs: (str(broken), "JSON")))
    panel.load_recipe()
    assert "生成参数无效" in panel.status.text()
    assert panel.count.value() == 5


@pytest.mark.gui
def test_form_validation_reports_before_any_job(window):
    app, widget = window
    panel = widget.gen_panel
    panel.settings["modes"] = []
    panel.preview_recipe()
    assert widget.active_job is None and "至少选择一种调制类型" in panel.status.text()
    panel.settings["modes"] = ["qpsk"]
    panel.collection.setCurrentIndex(panel.collection.findData("__new__"))
    panel.start_generation()
    assert widget.active_job is None and "请填写集合名称" in panel.status.text()
    # 损伤项：配方里用到即报错（不静默忽略）
    panel.collection_name.setText("集合 X")
    recipe = panel.recipe()
    recipe["impairments"] = {"cfo_ratio": {"uniform": [0, 0.001]}}
    with pytest.raises(ValueError, match="不可用"):
        check_generator_support(recipe)


@pytest.mark.gui
def test_torchsig_engine_is_linux_only(window, tmp_path):
    app, widget = window
    panel = widget.gen_panel
    panel.engine.setCurrentIndex(panel.engine.findData("torchsig"))
    assert panel.detection_scope.currentData() == "session_v1"  # 没有跳频：固定会话级
    with pytest.raises(ValueError, match="类别映射"):
        panel.recipe()
    mapping = tmp_path / "map.json"
    mapping.write_text('{"QPSK": "qpsk"}', encoding="utf-8")
    panel.settings["torchsig"]["mapping"] = str(mapping)
    panel.collection.setCurrentIndex(panel.collection.findData("__new__"))
    panel.collection_name.setText("TorchSig 集合")
    panel.start_generation()
    if not sys.platform.startswith("linux"):
        assert widget.active_job is None and "只能在 Linux" in panel.status.text()
    assert panel.recipe()["torchsig"]["signal_generators"] == {"fixed": "all"}
