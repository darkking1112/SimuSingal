"""Application operations executed by the worker; shared by GUI and CLI.

本包把原先单文件 ``services.py`` 按动作域拆分：:mod:`.imports`（导入链）、
:mod:`.truth`（真值附加）、:mod:`.detection` / :mod:`.amc`（检测与识别编排）、
:mod:`.generation`（生成与分析）、:mod:`.datasets`（数据集与标注）、
:mod:`.management`（维护与迁移）；执行器 :mod:`.collection_gen` /
:mod:`.training_export` / :mod:`.training_jobs` 也收在本包内。

``execute`` 是唯一入口（``tasks.run_job`` 调用）；旧模块级的导入辅助名
（``_attach_hop_truth``、``_normalized_target_rows`` 等）继续可导入，供测试与
历史调用复用。
"""

from ..data import Workspace
from . import amc, datasets, detection, generation, management  # noqa: F401
from .imports import (_apply_import_targets, _ensure_initial_labels, _inspect_import_file,
                      _normalized_target_rows, import_file, import_files, import_inspect,
                      import_manifest)
from .truth import (_attach_amc_truth, _attach_hop_truth, _attach_iq_truth, _attach_truth,
                    _class_from_version, _generated_name, _generation_targets,
                    _signal_targets, _target_hop_truth, _target_signal_truth)


def execute(request):
    workspace = Workspace(request["workspace"])
    action = request["action"]
    if action == "demo":
        return generation.demo(workspace, request)
    if action == "import":
        return import_file(workspace, request)
    if action == "import_inspect":
        return import_inspect(request)
    if action == "import_manifest":
        return import_manifest(request)
    if action == "import_files":
        return import_files(workspace, request)
    if action == "generate":
        return generation.generate(workspace, request)
    if action == "analyze":
        return generation.analyze(workspace, request)
    if action == "detect":
        return detection.detect(workspace, request)
    if action == "detect_hops":
        return detection.detect_hops(workspace, request)
    if action == "ml_detect":
        return detection.ml_detect(workspace, request)
    if action == "ml_detect_hops":
        return detection.ml_detect_hops(workspace, request)
    if action == "amc_classify":
        return amc.amc_classify(workspace, request)
    if action == "amc_iq_classify":
        return amc.amc_iq_classify(workspace, request)
    if action == "native":
        return management.native(workspace, request)
    if action == "storage_report":
        return management.storage_report(workspace, request)
    if action == "storage_cleanup":
        return management.storage_cleanup(workspace, request)
    if action == "recipe_preview":
        return generation.recipe_preview(workspace, request)
    if action == "recipe_save":
        return generation.recipe_save(workspace, request)
    if action == "dataset_build":
        return datasets.dataset_build(workspace, request)
    if action == "dataset_verify":
        return datasets.dataset_verify(workspace, request)
    if action == "dataset_bootstrap_labels":
        return datasets.dataset_bootstrap_labels(workspace, request)
    if action == "adopt_result":
        return datasets.adopt_result(workspace, request)
    if action == "generate_collection":
        return generation.generate_collection(workspace, request)
    if action == "torchsig_import":
        return generation.torchsig_import(workspace, request)
    if action == "torchsig_probe":
        return generation.torchsig_probe(workspace, request)
    if action == "export_training_data":
        return datasets.export_training_data(workspace, request)
    if action == "migrate_legacy":
        return management.migrate_legacy(workspace, request)
    raise ValueError(f"不支持的任务类型：{action}")
