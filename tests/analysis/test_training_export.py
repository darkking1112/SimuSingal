"""所选集合 → 训练数据集（检测 AnnotationDataset / AMC iq_dataset）。"""
import sys
from pathlib import Path

import numpy as np
import pytest

from signal_analysis.data.annotations import AnnotationDataset
from signal_analysis.data import Workspace
from signal_analysis.tasks import run_job

from test_collection_gen import generate, make_recipe

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))


def export(workspace, collection_id, **request):
    return run_job({"workspace": str(workspace.root), "action": "export_training_data",
                    "collection_id": collection_id, **request}, timeout=120)


@pytest.fixture
def collection(tmp_path):
    workspace = Workspace(tmp_path)
    recipe = make_recipe(count=10, signals={
        "count": {"fixed": 1}, "mode": {"balanced": ["qpsk", "fm", "fh_rc"]},
        "bandwidth_ratio": {"uniform": [0.05, 0.1]}, "snr_db": {"uniform": [15, 25]},
        "power_dbfs": {"fixed": -10.0}},
        record={"sample_rate_hz": {"fixed": 200000.0}, "duration_s": {"fixed": 0.2}})
    result = generate(workspace, recipe)
    assert result["created"] == 10
    return workspace, result["collection_id"]


def test_detection_export_is_a_valid_per_hop_annotation_dataset(collection, tmp_path):
    workspace, collection_id = collection
    result = export(workspace, collection_id, task="detection", image_size=256)
    assert result["label_semantics"] == "per_hop_v1"
    assert result["samples"] == 10 and result["splits"]["train"] and result["splits"]["val"]
    assert result["boxes"] > 10  # 跳频会话展开成逐跳框
    dataset = AnnotationDataset(result["path"])
    assert dataset.card["contract"]["label_semantics"] == "per_hop_v1"
    assert dataset.card["source"]["dataset_version_id"] == result["dataset_version_id"]
    assert all(dataset.record(i)["annotation_status"] == "reviewed"
               for i in range(len(dataset.records)))
    # 训练页的后续链路：快照 → 训练脚本的数据集加载器
    snapshot = dataset.snapshot(tmp_path / "snapshot")
    from detectors.dataset import load_dataset

    card, records = load_dataset(snapshot)
    assert card["contract"]["label_semantics"] == "per_hop_v1" and len(records) == 10
    # 版本已记录，可追溯
    assert workspace.get_dataset_version(result["dataset_version_id"])["status"] == "ready"


def test_iq_export_keeps_only_confirmed_a09_classes(collection):
    workspace, collection_id = collection
    result = export(workspace, collection_id, task="iq", samples=256)
    assert result["samples"] > 0 and result["splits"]["train"] and result["splits"]["val"]
    import json

    root = Path(result["path"])
    card = json.loads((root / "iq_dataset.json").read_text(encoding="utf-8"))
    assert card["contract"]["samples"] == 256 and card["contract"]["class_set"] == "a09"
    with np.load(root / "iq_dataset.npz") as store:
        assert store["waveforms"].shape[1:] == (2, 256)
        assert set(store["labels"].astype(str)) <= {"qpsk", "fm"}  # 跳频样式没有 A09 类别
    from train_iq import load_dataset

    loaded = load_dataset(root)
    assert len(loaded[1]) == result["samples"]


def test_export_reports_missing_annotations_and_tiny_collections(tmp_path):
    workspace = Workspace(tmp_path)
    bare = workspace.create_collection("没有标注集")
    with pytest.raises(Exception, match="没有检测标注集"):
        export(workspace, bare["id"], task="detection")
    tiny = generate(workspace, make_recipe(count=1, labels={"detection": "session_v1"}),
                    collection_name="太小")
    with pytest.raises(Exception, match="样本太少"):
        export(workspace, tiny["collection_id"], task="detection")
