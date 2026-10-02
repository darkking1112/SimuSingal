"""训练与推理共享的数据格式和约束（契约层）。

只依赖标准库与 NumPy：不导入推理运行时、算法实现或 GUI。细分职责：

* :mod:`.manifest` / :mod:`.decode`：检测模型清单与 ``normalized_boxes_v1`` 框布局；
* :mod:`.image`：``tf_image_v1`` 坐标与布局（``band_to_box``/``box_to_band``）；
* :mod:`.amc`：A09 类别、特征顺序与分类器清单；
* :mod:`.iq`：``iq_waveform_v1`` 常量、类别工具与分类器清单；
* :mod:`.preprocess`：前段（抽取比/低通抽头）共享口径。

常用符号在本模块汇总导出；细分符号请从具体子模块导入。
"""

from .manifest import (
    ALLOWED_IMAGE_SIZES,
    DEFAULT_DYNAMIC_RANGE_DB,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_LABEL_SEMANTICS,
    DEFAULT_NFFT,
    IMAGE_LAYOUT,
    INPUT_CONTRACT,
    LABEL_SEMANTICS,
    LABEL_SEMANTICS_FIELD,
    MANIFEST_SCHEMA_VERSION,
    MAX_MANIFEST_BYTES,
    OUTPUT_LAYOUT,
    RUNTIME,
    ManifestError,
    read_model_manifest,
    write_model_manifest,
)

from .decode import (
    BOX_COLUMNS,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_SCORE_THRESHOLD,
    boxes_to_bands,
    non_max_suppression,
    parse_model_output,
)

from .image import (
    DYNAMIC_RANGE_DB,
    IMAGE_SIZE,
    band_bins,
    band_to_box,
    box_to_band,
)

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
    read_amc_manifest,
    write_amc_manifest,
)

from .iq import (
    CLASS_SET_A09,
    CLASS_SET_CUSTOM,
    DEFAULT_IQ_SAMPLES,
    IQ_LAYOUT,
    IQ_MANIFEST_SCHEMA_VERSION,
    IQ_NORMALIZATION,
    IQ_ONNX_CONTRACT,
    IQ_RESULT_CONTRACT,
    IQ_WAVEFORM_CONTRACT,
    IQModelError,
    class_labels,
    class_set_name,
    read_iq_manifest,
    write_iq_manifest,
)

from .preprocess import (
    LOWPASS_TAPS,
    MAX_ANALYSIS_SAMPLES,
    MIN_WORK_SAMPLES,
    SAMPLES_PER_BAND,
)

__all__ = [
    "ALLOWED_IMAGE_SIZES", "AMC_CLASSES", "AMC_FEATURES", "AMC_FEATURE_CONTRACT",
    "AMC_MANIFEST_SCHEMA_VERSION", "AMC_MODEL_CONTRACT", "AMC_ONNX_CONTRACT",
    "AMC_RESULT_CONTRACT", "BOX_COLUMNS", "CLASS_LABELS", "CLASS_SET_A09",
    "CLASS_SET_CUSTOM", "DEFAULT_DYNAMIC_RANGE_DB", "DEFAULT_IMAGE_SIZE",
    "DEFAULT_IOU_THRESHOLD", "DEFAULT_IQ_SAMPLES", "DEFAULT_LABEL_SEMANTICS",
    "DEFAULT_MODEL_ID", "DEFAULT_NFFT", "DEFAULT_SCORE_THRESHOLD", "DYNAMIC_RANGE_DB",
    "IMAGE_LAYOUT", "IMAGE_SIZE", "INPUT_CONTRACT", "IQModelError", "IQ_LAYOUT",
    "IQ_MANIFEST_SCHEMA_VERSION", "IQ_NORMALIZATION", "IQ_ONNX_CONTRACT",
    "IQ_RESULT_CONTRACT", "IQ_WAVEFORM_CONTRACT", "LABEL_SEMANTICS",
    "LABEL_SEMANTICS_FIELD", "LOWPASS_TAPS", "MANIFEST_SCHEMA_VERSION",
    "MAX_ANALYSIS_SAMPLES", "MAX_MANIFEST_BYTES", "MIN_WORK_SAMPLES", "ModelError",
    "ManifestError", "OUTPUT_LAYOUT", "RUNTIME", "SAMPLES_PER_BAND", "band_bins",
    "band_to_box", "box_to_band", "boxes_to_bands", "class_labels", "class_set_name",
    "non_max_suppression", "parse_model_output", "read_amc_manifest", "read_iq_manifest",
    "read_model_manifest", "write_amc_manifest", "write_iq_manifest", "write_model_manifest",
]
