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

from signal_analysis.ui import MainWindow
from signal_analysis.ui.pages import collection_gen_page
from signal_analysis.ui.pages.collection_gen_page import (CollectionGenPanel, ProjectParamsDialog,
                                                          TorchSigParamsDialog, params_dialog)
from signal_analysis.algorithms.generation.recipes import check_generator_support, validate_recipe


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
def test_generation_page_is_a_form_with_engine_specific_dialogs(window):
    app, widget = window
    panel = widget.gen_panel
    assert isinstance(panel, CollectionGenPanel)
    generator = widget.tabs.widget(widget._page_index("IQ 信号生成"))
    assert generator.isAncestorOf(panel)
    # 两个子页：单个信号生成 / 信号集合生成
    sections = generator.findChild(QtWidgets.QTabWidget, "gen_sections")
    assert sections is not None and sections.count() == 2
    assert sections.tabText(1) == "信号集合生成"
    # 默认：检测=逐跳（会话+逐跳目标）、AMC 勾选
    recipe = panel.recipe()
    assert recipe["labels"] == {"detection": "per_hop_v1", "amc": True}
    assert recipe["engine"] == "project" and recipe["count"] == 200
    assert recipe["signals"]["mode"]["balanced"]  # 默认类别均衡
    assert "hopping" in recipe
    # 页面上没有可手写 JSON 的编辑框；明细在配置弹框里
    assert not any(not item.isReadOnly()
                   for item in panel.findChildren(QtWidgets.QPlainTextEdit))
    # 页面上不再有损伤分组：它属于项目引擎的配置弹框
    assert not hasattr(panel, "impairment_checks")


@pytest.mark.gui
def test_config_dialog_depends_on_engine(window):
    """“配置…”按引擎打开不同弹框：损伤项只在项目引擎那个里，且全部置灰（尚未实现）。"""
    app, widget = window
    panel = widget.gen_panel

    project = params_dialog(panel, panel.settings, "project")
    assert isinstance(project, ProjectParamsDialog)
    assert project.windowTitle() == "生成参数 · 信号与采样（项目引擎）"
    assert project.impairment_checks and not any(
        check.isEnabled() for check in project.impairment_checks.values())
    assert list(project.impairment_checks) == ["cfo", "phase_noise", "multipath", "iq_imbalance"]
    # 项目引擎专属：调制轮换、功率、跳速与损伤
    assert project.modes.count() == 9 and hasattr(project, "hop_low")
    # TorchSig 专属的东西不在这个弹框里
    assert not hasattr(project, "generators") and not hasattr(project, "mapping")
    project.reject()

    torchsig = params_dialog(panel, panel.settings, "torchsig")
    assert isinstance(torchsig, TorchSigParamsDialog)
    assert torchsig.windowTitle() == "生成参数 · 信号与采样（TorchSig）"
    # TorchSig 没有跳频，也没有 impairments 组（扰动只有 0/1/2 三档）
    assert not hasattr(torchsig, "impairment_checks")
    assert not hasattr(torchsig, "hop_low") and not hasattr(torchsig, "modes")
    assert torchsig.impairment.currentData() is None and torchsig.generators.text() == "all"
    torchsig.reject()

    # 未知引擎退回项目引擎，不会崩
    assert isinstance(params_dialog(panel, panel.settings, "unknown"), ProjectParamsDialog)


@pytest.mark.gui
def test_config_button_opens_the_dialog_of_the_current_engine(window, monkeypatch):
    app, widget = window
    panel = widget.gen_panel
    opened = []

    class _Recorder:
        def __init__(self, parent, settings, engine):
            opened.append(engine)

        def exec(self):
            return QtWidgets.QDialog.DialogCode.Rejected

    monkeypatch.setattr(collection_gen_page, "params_dialog",
                        lambda parent, settings, engine: _Recorder(parent, settings, engine))
    panel.configure_signals()
    panel.engine.setCurrentIndex(panel.engine.findData("torchsig"))
    panel.configure_signals()
    panel.engine.setCurrentIndex(panel.engine.findData("project"))
    panel.configure_signals()
    assert opened == ["project", "torchsig", "project"]


@pytest.mark.gui
def test_config_dialog_values_are_applied_per_engine(window, monkeypatch):
    """确认后按引擎写回表单：项目引擎回写调制/功率/跳速，TorchSig 回写信号族/扰动/映射。"""
    app, widget = window
    panel = widget.gen_panel

    class _Accepted:
        def __init__(self, values):
            self._values = values

        def exec(self):
            return QtWidgets.QDialog.DialogCode.Accepted

        def values(self):
            return dict(self._values)

    project_values = {"rate": 2_000_000.0, "duration": (0.1, 0.2), "signals": (1, 1),
                      "snr": (0.0, 10.0), "bandwidth_ratio": (0.05, 0.4),
                      "modes": ["qpsk"], "balanced": False,
                      "power": (-20.0, -8.0), "hop": (50.0, 50.0)}
    monkeypatch.setattr(collection_gen_page, "params_dialog",
                        lambda *args: _Accepted(project_values))
    panel.configure_signals()
    assert panel.settings["modes"] == ["qpsk"] and not panel.settings["balanced"]
    assert panel.settings["hop"] == (50.0, 50.0) and panel.settings["rate"] == 2_000_000.0
    # TorchSig 的设置不被项目引擎的弹框动到
    assert panel.settings["torchsig"] == {"generators": "all", "impairment": None,
                                          "mapping": ""}

    torchsig_values = {"rate": 500_000.0, "duration": (0.1, 0.2), "signals": (1, 2),
                       "snr": (5.0, 15.0), "bandwidth_ratio": (0.1, 0.2),
                       "torchsig": {"generators": "qpsk,ook", "impairment": 1},
                       "torchsig_mapping": "/tmp/map.json"}
    monkeypatch.setattr(collection_gen_page, "params_dialog",
                        lambda *args: _Accepted(torchsig_values))
    panel.configure_signals()
    assert panel.settings["torchsig"] == {"generators": "qpsk,ook", "impairment": 1,
                                          "mapping": "/tmp/map.json"}
    # 项目引擎的设置保持不变（两套设置共存，按引擎各取所需）
    assert panel.settings["modes"] == ["qpsk"] and panel.settings["rate"] == 500_000.0
    panel.engine.setCurrentIndex(panel.engine.findData("torchsig"))
    assert "qpsk,ook" in panel.signal_summary.text()


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
