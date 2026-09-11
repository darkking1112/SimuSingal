"""AI/机器学习接入层：检测（P3）与调制识别（P4）。

设计要点：

* **只依赖 NumPy 与标准库**；``onnxruntime`` 是可选依赖（``pip install .[ml]``），
  且只在工作进程内惰性加载，因此默认环境无需安装即可运行传统检测路径。
* **契约与训练一致**：时频图的排布、归一化、边框坐标约定由
  :mod:`signal_analysis.ml.tensor` 唯一定义，训练脚本（``training/``）复用
  同一模块生成标签，避免"训练-推理"口径分叉；调制识别的特征向量由
  :mod:`signal_analysis.ml.amc` 的 ``AMC_FEATURES`` 固定顺序，训练脚本同样直接
  复用 ``extract_features``。
* **结果契约不变**：检测输出仍是冻结的 ``detect_result_v1``，可以直接用
  :func:`signal_analysis.evaluation.evaluate_detections` 与传统检测并排评分；
  识别输出为 ``amc_classify_v1``，自带传统启发式对照（数字/模拟）。
"""

from .amc import (
    AMC_CLASSES,
    AMC_FEATURE_CONTRACT,
    AMC_FEATURES,
    AMC_MANIFEST_SCHEMA_VERSION,
    AMC_MODEL_CONTRACT,
    AMC_ONNX_CONTRACT,
    AMC_RESULT_CONTRACT,
    CLASS_LABELS,
    DEFAULT_MODEL_ID,
    ModelError,
    amc_classify,
    default_model_path,
    evaluate_model,
    extract_features,
    feature_vector,
    fit_model,
    load_default_model,
    load_model,
    mode_to_class,
    onnx_scores,
    predict,
    read_amc_manifest,
    save_model,
    write_amc_manifest,
)
from .decode import (
    BOX_COLUMNS,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_SCORE_THRESHOLD,
    boxes_to_bands,
    non_max_suppression,
    parse_model_output,
)
from .detector import CONTEXT_KEYS, ML_KEYS, ml_detect
from .manifest import (
    DEFAULT_IMAGE_SIZE,
    IMAGE_LAYOUT,
    INPUT_CONTRACT,
    MANIFEST_SCHEMA_VERSION,
    OUTPUT_LAYOUT,
    ManifestError,
    read_model_manifest,
    write_model_manifest,
)
from .runtime import (
    ModelRunner,
    RuntimeUnavailable,
    available,
    load_runner,
    runtime_module,
    runtime_version,
)
from .tensor import (
    DYNAMIC_RANGE_DB,
    IMAGE_SIZE,
    band_to_box,
    box_to_band,
    detection_image,
    image_db,
    measure_band,
    spectral_context,
)

__all__ = [
    "AMC_CLASSES",
    "AMC_FEATURES",
    "AMC_FEATURE_CONTRACT",
    "AMC_MANIFEST_SCHEMA_VERSION",
    "AMC_MODEL_CONTRACT",
    "AMC_ONNX_CONTRACT",
    "AMC_RESULT_CONTRACT",
    "BOX_COLUMNS",
    "CLASS_LABELS",
    "CONTEXT_KEYS",
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_IOU_THRESHOLD",
    "DEFAULT_MODEL_ID",
    "DEFAULT_SCORE_THRESHOLD",
    "DYNAMIC_RANGE_DB",
    "IMAGE_LAYOUT",
    "IMAGE_SIZE",
    "INPUT_CONTRACT",
    "MANIFEST_SCHEMA_VERSION",
    "ML_KEYS",
    "ManifestError",
    "ModelError",
    "ModelRunner",
    "OUTPUT_LAYOUT",
    "RuntimeUnavailable",
    "amc_classify",
    "available",
    "band_to_box",
    "box_to_band",
    "boxes_to_bands",
    "default_model_path",
    "detection_image",
    "evaluate_model",
    "extract_features",
    "feature_vector",
    "fit_model",
    "image_db",
    "load_default_model",
    "load_model",
    "load_runner",
    "measure_band",
    "ml_detect",
    "mode_to_class",
    "non_max_suppression",
    "onnx_scores",
    "parse_model_output",
    "predict",
    "read_amc_manifest",
    "read_model_manifest",
    "runtime_module",
    "runtime_version",
    "save_model",
    "spectral_context",
    "write_amc_manifest",
    "write_model_manifest",
]
