"""拆分后的训练页：命名、进入条件、集合标注、切换保护、跨页占用与进程树监管。"""
import ctypes
import json
import os
from pathlib import Path
import sys
import time

import pytest

from signal_analysis.contracts.iq import DEFAULT_IQ_SAMPLES
from signal_analysis.data.datasets import build_rows
from signal_analysis.ui.pages.training_common import metric_text
from signal_analysis.ui.training_process import PROCESS_BOOTSTRAP
from test_collection_gen import generate, make_recipe


@pytest.fixture
def training_window(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from PySide6 import QtWidgets
    from signal_analysis.ui import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(tmp_path / "workspace")
    yield app, window
    window.close()
    app.processEvents()


def wait_for(app, condition, timeout=10):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    app.processEvents()
    assert condition(), "条件未在预期时间内满足"


def use_collection(window, app, *, name="训练集", count=2, detection="session_v1",
                   amc=True):
    """生成一个集合并把侧栏切到它（训练页只在选中集合时可用）。"""
    result = generate(window.workspace,
                      make_recipe(count=count,
                                  labels={"detection": detection, "amc": amc}),
                      collection_name=name)
    window.refresh_collections()
    index = window.collection_combo.findData(result["collection_id"])
    assert index >= 0
    window.collection_combo.setCurrentIndex(index)
    window.refresh_assets()
    app.processEvents()
    return result["collection_id"]


def switch_collection(window, app, collection_id):
    """把侧栏切到指定集合（不走生成流程，便于用自定义资产/标注集）。"""
    window.refresh_collections()
    index = window.collection_combo.findData(collection_id)
    assert index >= 0
    window.collection_combo.setCurrentIndex(index)
    window.refresh_assets()
    app.processEvents()


def fake_config(page, repository):
    config = page.configuration()
    config.update(repository=str(repository), python=sys.executable)
    return config


@pytest.mark.gui
def test_training_pages_have_fixed_tasks_and_independent_owners(training_window):
    _, window = training_window
    assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == [
        "信号导入", "IQ 信号生成", "态势显示", "信号检测", "调制识别", "跳频参数",
        "信号检测训练", "AMC 识别训练", "数据管理", "算法对比", "运行记录"]
    detection = window.detection_training_page
    amc = window.amc_training_page
    assert detection.TITLE == "信号检测训练" and amc.TITLE == "AMC 识别训练"
    assert detection.configuration()["task"] == "detection"
    assert amc.configuration()["task"] == "iq"
    assert detection.OWNER != amc.OWNER
    # 两页各有一个标注子页，AMC 的叫“信号标注”；检测专属字段不得出现在 AMC 页
    assert detection.sections.tabText(0) == "数据标注"
    assert amc.sections.tabText(0) == "信号标注"
    assert not hasattr(amc, "weights")
    assert not hasattr(amc, "framework_config")
    # 训练页不再有数据集/来源/导出入口
    for page in (detection, amc):
        for field in ("data", "source", "collection", "export_button"):
            assert not hasattr(page, field), field
        assert {"train_collection_id", "val_collection_id"} <= set(page.configuration())
    assert not hasattr(window, "training_page")


@pytest.mark.gui
def test_amc_advanced_training_fields_are_opt_in(training_window):
    """高级结构超参默认不写入配置；勾选后才带上，并受 iq_tuning_args 校验。"""
    app, window = training_window
    page = window.amc_training_page
    use_collection(window, app, name="高级参数集合", count=2)
    config = page.configuration()
    assert page.advanced_toggle.isChecked() is False
    for key in ("model_params", "weight_decay", "patience", "scheduler", "monitor",
                "save_checkpoint"):
        assert key not in config, key          # 默认沿用模型目录声明的默认值
    assert not page.iq_channels.isEnabled()

    page.advanced_toggle.setChecked(True)
    app.processEvents()
    assert page.iq_channels.isEnabled()
    page.iq_channels.setText("64,128,256")
    page.iq_kernel.setValue(9)
    page.iq_dropout.setValue(0.20)
    page.iq_weight_decay.setValue(0.001)
    page.iq_patience.setValue(4)
    page.iq_scheduler.setCurrentIndex(page.iq_scheduler.findData("plateau"))
    page.iq_monitor.setCurrentIndex(page.iq_monitor.findData("val_loss"))
    page.iq_checkpoint.setChecked(True)
    config = page.configuration()
    assert config["model_params"] == {"channels": [64, 128, 256], "kernel": 9, "dropout": 0.20}
    assert config["weight_decay"] == 0.001 and config["patience"] == 4
    assert config["scheduler"] == "plateau" and config["monitor"] == "val_loss"
    assert config["save_checkpoint"] is True

    # 校验与执行侧共用同一份逻辑：cnn 通道数不对 / 核长为偶数直接拒绝
    from signal_analysis.services.training_jobs import iq_tuning_args
    page.iq_channels.setText("64,128")
    with pytest.raises(ValueError, match="3 个正整数"):
        iq_tuning_args(page.configuration(), "cnn")
    page.iq_channels.setText("")
    page.iq_kernel.setValue(4)
    with pytest.raises(ValueError, match="奇数"):
        iq_tuning_args(page.configuration(), "cnn")


@pytest.mark.gui
def test_amc_model_catalog_drives_architecture_window_and_parameters(training_window):
    """模型目录驱动模型列表、架构展示、窗口长度与参数表单；刷新按训练环境核对。"""
    app, window = training_window
    page = window.amc_training_page
    from signal_analysis.algorithms.amc.ai_model import (CATALOG_VERSION, model_spec,
                                                         spec_json)
    assert page.model_options() == ["cnn", "tcn", "mcldnn", "petcgdnn", "cv_trn", "poet",
                                        "amc_net", "ascs"]
    assert "IQCNN" in page.model_arch.toPlainText()
    assert page.iq_channels is not None and page.iq_kernel is not None
    assert page.model_hint.text() == ""                 # cnn 无窗口/依赖问题
    config = page.configuration()
    assert config["samples"] == DEFAULT_IQ_SAMPLES
    assert config["catalog_version"] == CATALOG_VERSION

    train_id = use_collection(window, app, name="目录训练集")
    val_id = use_collection(window, app, name="目录验证集")
    page.train_collection.setCurrentIndex(page.train_collection.findData(train_id))
    page.val_collection.setCurrentIndex(page.val_collection.findData(val_id))
    page.validate_training_config(page.configuration())   # 合法配置：先建 inputs_plan
    app.processEvents()

    # 按训练环境刷新：目录来自训练环境子进程（依赖状态随目录返回）
    page.repository.setText(str(Path(__file__).resolve().parents[2]))
    page.python.setText(sys.executable)
    page._refresh_catalog()
    app.processEvents()
    assert "训练环境" in page.catalog_status.text()
    assert page.model_options() == ["cnn", "tcn", "mcldnn", "petcgdnn", "cv_trn", "poet",
                                        "amc_net", "ascs"]

    # 伪目录：exact 窗口约束联动输入框；缺依赖的模型显示原因并拒绝起任务
    fake = dict(spec_json(model_spec("cnn")))
    fake.update(title="CV_TRN（示意）", samples="exact:128", missing=["timm"])
    page._apply_catalog({"catalog_version": CATALOG_VERSION, "models": [fake]})
    app.processEvents()
    assert page.samples.value() == 128                  # 只有一个合法窗口长度
    assert "timm" in page.model_hint.text() and "不可训练" in page.model_hint.text()
    with pytest.raises(ValueError, match="缺少 cnn 的依赖"):
        page.validate_task_config(page.configuration())

    # 依赖齐备但窗口长度非法：同一处校验拒绝（手动改回 256 点）
    page._apply_catalog({"catalog_version": CATALOG_VERSION,
                         "models": [{**fake, "missing": []}]})
    app.processEvents()
    assert page.model_hint.text() == ""
    with pytest.raises(ValueError, match="固定要求 128 点"):
        page.validate_task_config({**page.configuration(), "samples": 256})


@pytest.mark.gui
def test_training_tabs_need_a_selected_collection(training_window):
    app, window = training_window
    tabs = window.tabs
    detection_index = window._page_index("信号检测训练")
    amc_index = window._page_index("AMC 识别训练")
    assert not tabs.isTabEnabled(detection_index) and not tabs.isTabEnabled(amc_index)

    collection_id = use_collection(window, app)
    assert tabs.isTabEnabled(detection_index) and tabs.isTabEnabled(amc_index)
    page = window.detection_training_page
    assert page.train_collection.currentData() == collection_id
    assert page.val_collection.currentData() is None  # 验证集必须显式选择
    assert not page.train_button.isEnabled()

    # 停在训练页时切回“全部资产”→ 置灰并跳回“信号导入”
    tabs.setCurrentIndex(detection_index)
    window.collection_combo.setCurrentIndex(0)
    app.processEvents()
    assert tabs.currentIndex() == window._page_index("信号导入")
    assert not tabs.isTabEnabled(detection_index)
    # 离开集合时保留上一次选择（页面已不可进入），重新选中集合后按侧栏同步
    assert page.train_collection.currentData() == collection_id
    window.collection_combo.setCurrentIndex(
        window.collection_combo.findData(collection_id))
    app.processEvents()
    assert tabs.isTabEnabled(detection_index)
    assert page.train_collection.currentData() == collection_id


@pytest.mark.gui
def test_sidebar_label_filters_find_unlabeled_assets(training_window):
    app, window = training_window
    labeled = use_collection(window, app, name="只标检测", count=2,
                             detection="session_v1", amc=False)
    assert window.assets.count() == 2
    window.label_filter.setCurrentIndex(window.label_filter.findData("amc"))
    window.refresh_assets()
    assert window.assets.count() == 2  # AMC 未标注
    assert "已过滤" in window.asset_total.text()
    window.label_filter.setCurrentIndex(window.label_filter.findData("detection"))
    window.refresh_assets()
    assert window.assets.count() == 0  # 检测已标注
    assert "没有资产" in window.asset_info.text()
    window.label_filter.setCurrentIndex(0)
    window.refresh_assets()
    assert window.assets.count() == 2
    assert window.collection_combo.currentData() == labeled


@pytest.mark.gui
def test_detection_annotation_writes_collection_labels(training_window):
    app, window = training_window
    collection_id = use_collection(window, app, count=2)
    window.assets.setCurrentRow(0)
    app.processEvents()
    page = window.detection_training_page
    assert page._asset_id is not None and page._task_set_id is not None
    asset_id, task_set_id = page._asset_id, page._task_set_id
    before = len(window.workspace.list_targets(asset_id))

    page.add_box()
    assert page.dirty
    assert page.save_annotation()
    targets = window.workspace.list_targets(asset_id)
    assert len(targets) == before + 1
    added = targets[-1]
    label = window.workspace.current_label("detection", task_set_id, added["id"])
    assert label is not None and int(label["include"]) == 1
    assert label["source"] == "manual"

    # 删框再保存 → 追加负标注（历史版本保留）
    page.select_roi(page._boxes[-1]["roi"])
    page.delete_box()
    assert page.save_annotation()
    label = window.workspace.current_label("detection", task_set_id, added["id"])
    assert int(label["include"]) == 0
    assert len(window.workspace.label_revisions("detection", task_set_id,
                                               added["id"])) == 2

    # 标记无信号 → 覆盖度 complete
    page.mark_no_signal()
    coverage = window.workspace.get_asset_coverage(task_set_id, asset_id)
    assert coverage["coverage"] == "complete"
    assert coverage["negative_kind"] == "no_signal"
    assert window.collection_combo.currentData() == collection_id


@pytest.mark.gui
def test_detection_annotation_hints_missing_task_set(training_window):
    app, window = training_window
    use_collection(window, app, name="只有 AMC", count=2, detection=None, amc=True)
    window.assets.setCurrentRow(0)
    app.processEvents()
    page = window.detection_training_page
    assert page._task_set_id is None
    assert "没有检测标注集" in page.annotation_status.text()
    assert "信号集合生成" in page.annotation_status.text()


@pytest.mark.gui
def test_amc_annotation_edits_class_and_parameters(training_window):
    app, window = training_window
    use_collection(window, app, count=2)
    window.assets.setCurrentRow(0)
    app.processEvents()
    page = window.amc_training_page
    assert page.target_table.rowCount() >= 1
    target = page.selected_target()
    assert target is not None
    task_set_id = page._task_set_id
    before = len(window.workspace.label_revisions("amc", task_set_id, target["id"]))

    page.class_state.setCurrentIndex(page.class_state.findData("known"))
    page.class_name.setCurrentText("qpsk")
    page.snr.setText("12.5")
    page.modulation.setText("QPSK")
    assert page.dirty
    assert page.save_annotation()
    label = window.workspace.current_label("amc", task_set_id, target["id"])
    assert label["class_state"] == "known" and label["class_name"] == "qpsk"
    assert label["source"] == "manual"
    assert len(window.workspace.label_revisions("amc", task_set_id,
                                                target["id"])) == before + 1
    version = window.workspace.current_target_version(target["id"])
    assert version["snr_db"] == pytest.approx(12.5)
    assert version["modulation"] == "QPSK"
    assert version["source"] == "manual"

    # 参数非法时给出可执行提示，不写入
    page.snr.setText("不是数字")
    page.mark_dirty()
    assert not page.save_annotation()
    assert "应为数值" in page.annotation_status.text()


@pytest.mark.gui
def test_training_runner_slot_rejects_other_page_without_second_process(
        training_window, tmp_path):
    app, window = training_window
    repository = tmp_path / "repo"
    worker = repository / "training" / "desktop_worker.py"
    worker.parent.mkdir(parents=True)
    worker.write_text("import time\nprint('running', flush=True)\ntime.sleep(30)\n",
                      encoding="utf-8")
    first = window.detection_training_page
    second = window.amc_training_page
    assert first.runner.start(
        fake_config(first, repository), owner=first.OWNER, task_kind="training",
        validate_config=lambda _config: None)
    assert window.tasks.count() == 1
    before = list((window.workspace.root / "training" / "runs").iterdir())

    assert not second.runner.start(
        fake_config(second, repository), owner=second.OWNER, task_kind="training",
        validate_config=lambda _config: None)
    assert first.runner.active
    assert not second.runner.active
    assert window.tasks.count() == 1
    assert list((window.workspace.root / "training" / "runs").iterdir()) == before
    assert window.training_coordinator.slot["owner"] == first.OWNER

    first.runner.stop()
    wait_for(app, lambda: not first.runner.active)
    assert window.tasks.count() == 0


@pytest.mark.gui
def test_history_filters_by_task_and_never_loads_asset_runs(training_window):
    app, window = training_window
    use_collection(window, app)  # 训练页需要选中集合才可用
    root = window.workspace.root / "training" / "runs"
    records = (
        ("detect", "detection", "rtdetr"),
        ("asset", "asset", "rtdetr"),
        ("iq", "iq", "cnn"),
        ("yolo", "detection", "yolo26s"),
    )
    for identifier, task, arch in records:
        directory = root / identifier
        model = directory / "model"
        model.mkdir(parents=True)
        (directory / "experiment.json").write_text(json.dumps({
            "id": identifier, "status": "success",
            "config": {"task": task, "arch": arch}, "metrics": [],
        }), encoding="utf-8")
        (model / ("iq_manifest.json" if task == "iq" else "detector.json")).write_text(
            "{}", encoding="utf-8")

    detection = window.detection_training_page
    amc = window.amc_training_page
    detection.refresh_history()
    assert {detection.history.itemData(i)["config"]["task"]
            for i in range(detection.history.count())} == {"detection", "asset"}
    asset_index = next(i for i in range(detection.history.count())
                       if detection.history.itemData(i)["id"] == "asset")
    detection.history.setCurrentIndex(asset_index)
    assert not detection.load_button.isEnabled()

    amc.refresh_history()
    assert [amc.history.itemData(i)["config"]["task"]
            for i in range(amc.history.count())] == ["iq"]
    assert amc.load_button.isEnabled()


@pytest.mark.gui
@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object process-tree validation")
@pytest.mark.parametrize("close_window", [False, True])
def test_stop_and_close_terminate_worker_and_spawned_child(
        training_window, tmp_path, close_window):
    app, window = training_window
    repository = tmp_path / "repo"
    worker = repository / "training" / "desktop_worker.py"
    worker.parent.mkdir(parents=True)
    worker.write_text(
        "from pathlib import Path\n"
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(60)'])\n"
        "Path(sys.argv[1]).with_name('child.pid').write_text(str(child.pid))\n"
        "print('child-started', flush=True)\n"
        "time.sleep(60)\n",
        encoding="utf-8")
    page = window.detection_training_page
    assert page.runner.start(
        fake_config(page, repository), owner=page.OWNER, task_kind="training",
        validate_config=lambda _config: None)
    wait_for(app, lambda: page.runner.active and page.runner.directory is not None
             and (page.runner.directory / "child.pid").is_file())
    child_pid = int((page.runner.directory / "child.pid").read_text(encoding="utf-8"))
    run = page.runner._run
    assert run.process_tree.active_count(run.process) >= 2, \
        "Job Object 未纳入 worker 创建的子进程"
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel32.WaitForSingleObject.restype = ctypes.c_ulong
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    child_handle = kernel32.OpenProcess(0x00101000, False, child_pid)
    assert child_handle, f"无法打开受管子进程句柄：{ctypes.get_last_error()}"
    if close_window:
        window.close()
        assert not page.runner.active
    else:
        assert page.runner.stop()
        wait_for(app, lambda: not page.runner.active, timeout=12)
    try:
        wait_for(app, lambda: kernel32.WaitForSingleObject(child_handle, 0) == 0, timeout=3)
    finally:
        kernel32.CloseHandle(child_handle)


# --------------------------------------------------------------- 第三轮修复（第 11 节）
@pytest.mark.gui
def test_switching_asset_saves_detection_annotation_and_blocks_on_failure(
        training_window, monkeypatch):
    app, window = training_window
    use_collection(window, app, count=2)
    window.assets.setCurrentRow(0)
    app.processEvents()
    page = window.detection_training_page
    first_asset, task_set_id = page._asset_id, page._task_set_id
    before = len(window.workspace.list_targets(first_asset))

    page.add_box()
    window.assets.setCurrentRow(1)
    app.processEvents()
    # 切换前自动保存：新框已落库
    assert page._asset_id != first_asset
    assert len(window.workspace.list_targets(first_asset)) == before + 1
    added = window.workspace.list_targets(first_asset)[-1]
    label = window.workspace.current_label("detection", task_set_id, added["id"])
    assert label is not None and int(label["include"]) == 1

    # 保存失败时留在原资产（不静默丢弃修改）
    window.assets.setCurrentRow(1)
    app.processEvents()
    page = window.detection_training_page
    second_asset = page._asset_id

    def broken(*_args, **_kwargs):
        raise ValueError("模拟写库失败")

    monkeypatch.setattr(window.workspace, "append_detection_label", broken)
    page.add_box()
    assert page.dirty
    window.assets.setCurrentRow(0)
    app.processEvents()
    assert page._asset_id == second_asset, "保存失败时不得切换资产"
    assert page.dirty and window.selected_asset()["id"] == second_asset
    assert "未切换资产" in page.annotation_status.text()


@pytest.mark.gui
def test_switching_amc_target_saves_and_blocks_on_failure(training_window, monkeypatch):
    app, window = training_window
    result = generate(window.workspace, make_recipe(count=1), collection_name="AMC 目标")
    switch_collection(window, app, result["collection_id"])
    window.assets.setCurrentRow(0)
    app.processEvents()
    page = window.amc_training_page
    asset = window.workspace.list_assets(collection_id=result["collection_id"])[0]
    extra = window.workspace.add_target(asset["id"], "切换", "segment", for_amc=True)
    window.workspace.append_target_version(extra["id"], source="manual",
                                           sample_start=0, sample_end=100,
                                           f_low_hz=1e6, f_high_hz=2e6)
    page.refresh_annotation()
    app.processEvents()
    assert page.target_table.rowCount() >= 2
    page.target_table.selectRow(0)
    app.processEvents()
    first = page.selected_target()
    task_set_id = page._task_set_id

    page.snr.setText("42.0")
    page.target_table.selectRow(1)
    app.processEvents()
    # 切换目标前自动保存到原目标
    label = window.workspace.current_label("amc", task_set_id, first["id"])
    assert label is not None
    assert window.workspace.current_target_version(first["id"])["snr_db"] == \
        pytest.approx(42.0)

    def broken(*_args, **_kwargs):
        raise ValueError("模拟写库失败")

    monkeypatch.setattr(window.workspace, "append_amc_annotation", broken)
    page.target_table.selectRow(1)
    app.processEvents()
    page.target_table.selectRow(0)
    app.processEvents()
    page.snr.setText("43.0")
    page.target_table.selectRow(1)
    app.processEvents()
    assert page.loaded_target()["id"] == first["id"], "保存失败时不得切换目标"
    assert page.dirty and "未切换目标" in page.annotation_status.text()


@pytest.mark.gui
def test_saving_a_box_keeps_unedited_target_parameters(training_window):
    """本页只编辑时间与频率：SNR／调制／跳频标记等必须沿用旧版本（第 11 节）。"""
    app, window = training_window
    use_collection(window, app, count=1, detection="session_v1", amc=False)
    window.assets.setCurrentRow(0)
    app.processEvents()
    page = window.detection_training_page
    workspace = window.workspace
    target = page._targets[0]
    previous = target["current"]
    workspace.append_target_version(
        target["id"], source="manual",
        sample_start=previous["sample_start"], sample_end=previous["sample_end"],
        f_low_hz=previous["f_low_hz"], f_high_hz=previous["f_high_hz"],
        snr_db=12.5, modulation="QPSK", is_hopping=1, symbol_rate_baud=2000.0,
        waveform_mode="cw", params_json={"样式": "probe"})
    page.refresh_annotation()
    app.processEvents()
    page.mark_dirty()
    assert page.save_annotation()

    current = workspace.current_target_version(target["id"])
    assert current["sample_start"] is not None and current["f_low_hz"] is not None
    assert current["snr_db"] == pytest.approx(12.5)
    assert current["modulation"] == "QPSK"
    assert current["is_hopping"] == 1
    assert current["waveform_mode"] == "cw" and current["symbol_rate_baud"] == 2000.0
    assert json.loads(current["params_json"]) == {"样式": "probe"}


@pytest.mark.gui
def test_amc_save_keeps_unedited_fields_and_writes_nothing_when_invalid(training_window):
    app, window = training_window
    use_collection(window, app, count=1)
    window.assets.setCurrentRow(0)
    app.processEvents()
    page = window.amc_training_page
    workspace = window.workspace
    target = page.loaded_target()
    previous = target["current"]
    workspace.append_target_version(
        target["id"], source="manual",
        sample_start=previous.get("sample_start"), sample_end=previous.get("sample_end"),
        f_low_hz=previous.get("f_low_hz"), f_high_hz=previous.get("f_high_hz"),
        snr_db=previous.get("snr_db"), is_hopping=1, hop_rate_hz=300.0,
        power_dbfs=-9.0, waveform_mode="cw", symbol_rate_baud=4000.0)
    page.refresh_annotation(select_id=target["id"])
    app.processEvents()

    # 类别非法：保存失败，且参数版本与标签都不写入
    versions_before = len(workspace.list_target_versions(target["id"]))
    revisions_before = len(workspace.label_revisions("amc", page._task_set_id, target["id"]))
    page.class_state.setCurrentIndex(page.class_state.findData("known"))
    page.class_name.setCurrentText("绝对不在字典里的类别")
    page.snr.setText("33.0")
    assert not page.save_annotation()
    assert "字典内名称" in page.annotation_status.text()
    assert len(workspace.list_target_versions(target["id"])) == versions_before
    assert len(workspace.label_revisions("amc", page._task_set_id,
                                         target["id"])) == revisions_before
    assert workspace.current_target_version(target["id"])["snr_db"] != pytest.approx(33.0)

    # 改为合法类别：参数与标签一起写入，未编辑字段沿用旧值
    page.class_name.setCurrentText("qpsk")
    page.mark_dirty()
    assert page.save_annotation()
    current = workspace.current_target_version(target["id"])
    assert current["snr_db"] == pytest.approx(33.0)
    assert current["is_hopping"] == 1 and current["hop_rate_hz"] == pytest.approx(300.0)
    assert current["power_dbfs"] == pytest.approx(-9.0)
    assert current["waveform_mode"] == "cw" and current["symbol_rate_baud"] == 4000.0
    label = workspace.current_label("amc", page._task_set_id, target["id"])
    assert label["class_name"] == "qpsk" and label["class_state"] == "known"
    assert label["target_version_id"] == current["id"]


@pytest.mark.gui
def test_detection_save_confirms_coverage_and_negates_pending_targets(training_window):
    """保存即确认覆盖完整，并把没有框的适用目标记为无信号（第 11 节）。"""
    app, window = training_window
    use_collection(window, app, count=1, detection="session_v1", amc=False)
    window.assets.setCurrentRow(0)
    app.processEvents()
    workspace = window.workspace
    asset_id = window.detection_training_page._asset_id

    # 模拟手工加入集合的资产：资产没有覆盖度记录，集合的检测标注集是新建的
    manual = workspace.create_collection("手工集合")
    workspace.add_collection_member(manual["id"], asset_id)
    task_set = workspace.create_task_set(manual["id"], "detection",
                                         label_semantics="session_v1")
    switch_collection(window, app, manual["id"])
    window.assets.setCurrentRow(0)
    app.processEvents()
    page = window.detection_training_page
    assert page._task_set_id == task_set["id"]

    page.add_box()
    assert page.save_annotation()
    coverage = workspace.get_asset_coverage(task_set["id"], asset_id)
    assert coverage["coverage"] == "complete" and coverage["negative_kind"] is None
    rows, stats = build_rows(workspace, task_set["id"])
    assert rows, f"保存后资产应可进入训练：{stats['excluded']}"
    assert not [item for item in stats["excluded"] if item["asset_id"] == asset_id]

    # 「标记无信号」：整条资产作为负样本进入训练
    page.mark_no_signal()
    coverage = workspace.get_asset_coverage(task_set["id"], asset_id)
    assert coverage["coverage"] == "complete"
    assert coverage["negative_kind"] == "no_signal"
    rows, stats = build_rows(workspace, task_set["id"])
    assert [row for row in rows if row["asset_id"] == asset_id and row["negative"]]
    assert not [item for item in stats["excluded"] if item["asset_id"] == asset_id]


@pytest.mark.gui
def test_progress_reports_epochs_and_resets_on_stage_change(training_window, tmp_path):
    app, window = training_window
    use_collection(window, app, count=1)
    page = window.detection_training_page
    repository = tmp_path / "repo"
    worker = repository / "training" / "desktop_worker.py"
    worker.parent.mkdir(parents=True)
    worker.write_text(
        "import json, time\n"
        "def event(**values):\n"
        "    print('TRAIN_EVENT ' + json.dumps(values), flush=True)\n"
        "event(stage='构建训练输入 (2/2)', message='构建训练输入 (2/2)', done=2, total=2)\n"
        "event(epoch=1, loss=0.5)\n"
        "event(stage='YOLO26s 训练')\n"
        "time.sleep(0.4)\n"
        "event(epoch=2, loss=0.4)\n"
        "time.sleep(30)\n", encoding="utf-8")
    config = fake_config(page, repository)
    config["epochs"] = 10
    assert page.runner.start(config, owner=page.OWNER, task_kind="training",
                             validate_config=lambda _state: None)
    try:
        wait_for(app, lambda: page.runner.record["stage"] == "YOLO26s 训练"
                 and len(page.runner.record["metrics"]) == 1)
        progress = window.tasks.latest().progress
        assert progress["stage"] == "YOLO26s 训练"
        assert progress["done"] == 0 and progress["total"] == 0, "换阶段必须清零"
        assert progress["message"] == "YOLO26s 训练"
        wait_for(app, lambda: len(page.runner.record["metrics"]) == 2)
        progress = window.tasks.latest().progress
        assert progress["done"] == 2 and progress["total"] == 10
        assert progress["message"].startswith("训练第 2/10 轮")
    finally:
        page.runner.stop()
        wait_for(app, lambda: not page.runner.active)


def test_only_the_worker_creates_a_posix_session():
    """POSIX 上只允许一处 setsid：bootstrap 再调一次会因已是会话首进程而失败。"""
    assert "setsid" not in PROCESS_BOOTSTRAP
    source = (Path(__file__).resolve().parents[2] / "training" / "desktop_worker.py")
    assert "os.setsid()" in source.read_text(encoding="utf-8")


@pytest.mark.gui
def test_switching_collection_saves_and_rolls_back_on_failure(training_window, monkeypatch):
    """侧栏切换集合前先保存；保存失败时集合选择回滚（第 11 节）。"""
    app, window = training_window
    first = use_collection(window, app, name="集合一", count=1, detection="session_v1",
                           amc=False)
    second = use_collection(window, app, name="集合二", count=1, detection="session_v1",
                            amc=False)
    page = window.detection_training_page
    window.assets.setCurrentRow(0)
    app.processEvents()
    asset_id, task_set_id = page._asset_id, page._task_set_id
    targets_before = len(window.workspace.list_targets(asset_id))
    page.add_box()
    assert page.dirty

    switch_collection(window, app, first)
    assert window.collection_combo.currentData() == first
    assert not page.dirty
    # 切换集合前先保存：新框已落库
    assert len(window.workspace.list_targets(asset_id)) == targets_before + 1
    added = window.workspace.list_targets(asset_id)[-1]
    assert window.workspace.current_label("detection", task_set_id, added["id"]) is not None

    window.assets.setCurrentRow(0)
    app.processEvents()
    page.add_box()
    assert page.dirty

    def broken(*_args, **_kwargs):
        raise ValueError("模拟写库失败")

    monkeypatch.setattr(window.workspace, "append_detection_label", broken)
    switch_collection(window, app, second)
    assert window.collection_combo.currentData() == first, "保存失败必须回滚集合选择"
    assert page.dirty and window.current_collection_id() == first


def test_metric_text_shows_per_snr_buckets():
    """训练报告写 ``per_snr``（字典或列表），展示必须按该键读取。"""
    listed = metric_text({"validation": {"accuracy": .9, "per_snr": [
        {"low_db": -5.0, "high_db": 0.0, "count": 3, "accuracy": .5},
        {"low_db": 5.0, "high_db": None, "count": 0, "accuracy": None}]}})
    assert "按 SNR 统计" in listed
    assert "-5~0 dB：50.00%（3 个样本）" in listed
    assert "≥5 dB：未提供（0 个样本）" in listed
    bucketed = metric_text({"validation": {"accuracy": .8,
                                           "per_snr": {"0~5 dB": .7}}})
    assert "0~5 dB：70.00%" in bucketed
    assert "按 SNR 统计" not in metric_text({"validation": {"accuracy": .8}})
