"""Annotation persistence, snapshot isolation, and real GUI process lifecycle."""
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pytest

from signal_analysis.annotations import AnnotationDataset, create_dataset, validate_boxes


def sample_dataset(root):
    dataset = create_dataset(root, size=128)
    records = []
    for i, split in enumerate(("train", "val")):
        np.save(root / "images" / f"{i}.npy", np.zeros((128, 128), dtype=np.float32))
        records.append({"image": f"images/{i}.npy", "boxes": [], "split": split,
                        "annotation_status": "pending"})
    (root / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return AnnotationDataset(root)


def test_annotation_snapshot_preserves_original_and_freezes_labels(tmp_path):
    root = tmp_path / "dataset"
    dataset = sample_dataset(root)
    with pytest.raises(ValueError, match="未确认"):
        dataset.snapshot(tmp_path / "pending")
    box = [.5, .25, .4, .2, 1, 0]
    dataset.save(0, [box], "train")
    dataset.save(1, [], "val")
    path = dataset.snapshot(tmp_path / "snapshot")
    snapshot = AnnotationDataset(path)
    assert snapshot.record(0)["boxes"] == [box]
    assert snapshot.record(1)["annotation_status"] == "reviewed"
    assert json.loads((root / "samples.jsonl").read_text().splitlines()[0])["boxes"] == []
    dataset.save(0, [[.2, .2, .1, .1, 1, 0]], "train")
    assert AnnotationDataset(path).record(0)["boxes"] == [box]
    assert len(snapshot.card["annotation_snapshot"]["sha256"]) == 64


@pytest.mark.parametrize("box", [[.1, .1, .5, .5, 1, 0], [.5, .5, 0, .2, 1, 0],
                                 [.5, .5, .1, .1, 1, 2], [.5, float("nan"), .1, .1, 1, 0]])
def test_invalid_annotations_are_rejected(box):
    with pytest.raises(ValueError):
        validate_boxes([box])


def test_dataset_image_path_cannot_escape(tmp_path):
    dataset = sample_dataset(tmp_path / "data")
    dataset.records[0]["image"] = "../../outside.npy"
    with pytest.raises(ValueError, match="超出"):
        dataset.image(0)


def test_native_check_detects_wrong_input_scaling(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))
    from desktop_evaluate import verify_native, match_boxes
    from detectors.onnx_contract import rewrite

    class Native(torch.nn.Module):
        def forward(self, x):
            value = x.mean().reshape(1, 1, 1)
            left = value * 10
            raw = torch.cat([left, left, left + 20, left + 20,
                             value * 0 + .8, value * 0], dim=-1)
            return raw.repeat(1, 32, 1)

    native = tmp_path / "native.onnx"
    torch.onnx.export(Native(), torch.zeros(1, 3, 128, 128), str(native),
                      opset_version=17, dynamo=False,
                      input_names=["images"], output_names=["detections"])
    for scale in (1, 255):
        output = tmp_path / f"scale{scale}.onnx"
        rewrite(native, output, layout="pixel_xyxy", image_size=128, max_boxes=32,
                channel_repeat=3, input_scale=scale)
        if scale == 1:
            assert verify_native(native, output, 128)["max_error"] < 1e-4
            from desktop_evaluate import session
            assert session(output).get_inputs()[0].name == "images"
        else:
            with pytest.raises(ValueError, match="数值不一致"):
                verify_native(native, output, 128)
    box = [.5, .5, .2, .2, .9, 0]
    assert match_boxes([box, box], [box]) == (1, 1, 0)
    assert match_boxes([], [box]) == (0, 0, 1)


def test_iq_test_partition_does_not_enter_validation():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))
    from train_iq import _split
    data = np.arange(3)
    train, _, _, val, _, _ = _split(data, data, np.array(["train", "val", "test"]), data)
    assert train.tolist() == [0]
    assert val.tolist() == [1]


def wait_process(app, page, timeout=30):
    deadline = time.monotonic() + timeout
    while page.process is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    app.processEvents()
    assert page.process is None, page.log.toPlainText()[-2000:]


@pytest.fixture
def window(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from PySide6.QtWidgets import QApplication
    from signal_analysis.gui import MainWindow
    app = QApplication.instance() or QApplication([])
    widget = MainWindow(tmp_path / "workspace")
    yield app, widget
    widget.close()
    app.processEvents()


@pytest.mark.gui
def test_gui_roi_geometry_roundtrip_and_empty_confirmation(window, tmp_path):
    app, widget = window
    dataset = sample_dataset(tmp_path / "data")
    page = widget.training_page
    page.set_dataset(dataset.root)
    page.add_box(box=[.6, .2, .3, .1, 1, 0])
    assert page.save_annotation()
    assert np.allclose(AnnotationDataset(dataset.root).record(0)["boxes"][0], [.6, .2, .3, .1, 1, 0])
    page.samples.setCurrentRow(1)
    page.clear_boxes()
    assert page.save_annotation()
    assert AnnotationDataset(dataset.root).record(1)["annotation_status"] == "reviewed"
    page.dataset.snapshot(tmp_path / "ready")


@pytest.mark.gui
def test_gui_external_iq_training_export_verify_and_load(window):
    pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    app, widget = window
    page = widget.training_page
    config = page.configuration()
    config.update(task="iq", source="generator", arch="cnn", epochs=1,
                  per_class=3, samples=128, batch=4, snr_low=15, snr_high=25)
    page.start(config)
    assert page.process is not None
    wait_process(app, page, timeout=60)
    assert page.record["status"] == "success", page.log.toPlainText()[-4000:]
    assert page.record["metrics"][0]["epoch"] == 1
    assert (page.directory / "verification.json").is_file()
    assert page.load_button.isEnabled()
    page.load_model()
    assert widget.tabs.currentIndex() == widget._page_index("调制识别")
    assert Path(widget.amc_model.text()).is_file()


@pytest.mark.gui
def test_failed_start_and_cancel_are_persisted(window, tmp_path):
    app, widget = window
    page = widget.training_page
    repo = tmp_path / "repo"
    (repo / "training").mkdir(parents=True)
    worker = repo / "training/desktop_worker.py"
    worker.write_text("raise RuntimeError('intentional failure')\n")
    config = page.configuration()
    config.update(task="generate", repository=str(repo), python=sys.executable)
    page.start(config)
    wait_process(app, page)
    assert page.record["status"] == "failed"
    assert "intentional failure" in page.log.toPlainText()
    worker.write_text("import time, os\nif os.name == 'posix': os.setsid()\nprint('ready', flush=True)\ntime.sleep(60)\n")
    page.start(config)
    deadline = time.monotonic() + 5
    while "ready" not in page.log.toPlainText() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    page.stop()
    wait_process(app, page)
    assert json.loads((page.directory / "experiment.json").read_text())["status"] == "stopped"
