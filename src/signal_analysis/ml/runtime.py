"""ONNX Runtime 加载（惰性导入，只在工作进程内发生）。

``onnxruntime`` 是可选依赖（``pip install .[ml]``）。未安装时本模块给出
可执行的中文提示，传统能量检测路径完全不受影响；导入本模块本身也不会
加载推理运行时，因此仓库的默认测试环境无需安装它。
"""

from __future__ import annotations

import numpy as np

from .manifest import MIN_RUNTIME_VERSION, ManifestError

_RUNTIME_HINT = ("未安装 onnxruntime，AI 检测不可用。请安装可选依赖后重试："
                 "pip install 'simusignal[ml]'（或 pip install onnxruntime>=1.17）")


class RuntimeUnavailable(RuntimeError):
    """推理运行时缺失或版本不满足清单要求。"""


def runtime_module():
    """导入并返回 ``onnxruntime`` 模块，缺失时抛出带安装提示的异常。"""
    try:
        import onnxruntime  # noqa: PLC0415 - 惰性导入是刻意设计
    except ImportError as exc:  # pragma: no cover - 取决于运行环境
        raise RuntimeUnavailable(_RUNTIME_HINT) from exc
    return onnxruntime


def runtime_version():
    """返回已安装的运行时版本字符串；未安装时为 ``None``。"""
    try:
        return str(runtime_module().__version__)
    except RuntimeUnavailable:
        return None


def available():
    """推理运行时是否可用（GUI/CLI 用它决定是否展示 AI 入口）。"""
    return runtime_version() is not None


def _version_tuple(text):
    parts = []
    for chunk in str(text).split("."):
        digits = "".join(character for character in chunk if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def check_version(version=None, minimum=MIN_RUNTIME_VERSION):
    """校验运行时版本不低于清单要求，返回版本字符串。"""
    current = version if version is not None else runtime_version()
    if current is None:
        raise RuntimeUnavailable(_RUNTIME_HINT)
    if _version_tuple(current) < _version_tuple(minimum):
        raise ManifestError(f"onnxruntime 版本过低（当前 {current}，需要 >= {minimum}）")
    return current


class ModelRunner:
    """模型会话：保证输入按清单契约整形，输出保持原始形状。

    ``run(image)`` 接受 :func:`~signal_analysis.ml.tensor.detection_image` 产出的
    ``(size, size)`` 单通道灰度图，返回网络原始输出数组。会话在首次调用时
    创建，名字、版本与摘要全部来自清单，便于结果追溯。
    """

    def __init__(self, manifest, library, *, threads=None, version=None):
        self.manifest = dict(manifest)
        self.library = str(library)
        self.threads = None if threads is None else int(threads)
        self.runtime_version = check_version(version)
        self._session = None

    @property
    def model_name(self):
        return f"{self.manifest['id']}@{self.manifest['version']}"

    def _create(self):
        if self._session is not None:
            return self._session
        runtime = runtime_module()
        options = runtime.SessionOptions()
        if self.threads:
            options.intra_op_num_threads = self.threads
            options.inter_op_num_threads = 1
        self._session = runtime.InferenceSession(
            self.library, sess_options=options, providers=["CPUExecutionProvider"])
        return self._session

    def _output_name(self):
        wanted = self.manifest.get("output", {}).get("name")
        names = [output.name for output in self._create().get_outputs()]
        if wanted and wanted in names:
            return wanted
        return names[0] if names else None

    def run(self, image):
        """执行一次推理；``image`` 形状为 ``(size, size)``，取值 ``[0, 1]``。"""
        array = np.asarray(image, dtype=np.float32)
        size = int(self.manifest.get("input", {}).get("image_size", array.shape[-1]))
        if array.shape != (size, size):
            raise ValueError(f"模型输入应为 {size}×{size} 灰度图，实际为 {array.shape}")
        tensor = array.reshape(1, 1, size, size)
        input_name = self.manifest.get("input", {}).get("name") or "images"
        session = self._create()
        names = {item.name for item in session.get_inputs()}
        if input_name not in names:
            if not names:
                raise ManifestError("模型没有输入节点")
            input_name = sorted(names)[0]
        return session.run([self._output_name()], {input_name: tensor})[0]


def load_runner(manifest_path, *, threads=None):
    """按清单路径加载模型会话：``(runner, manifest, 模型绝对路径)``。"""
    from .manifest import read_model_manifest  # 局部导入，避免循环引用

    manifest, library = read_model_manifest(manifest_path)
    runner = ModelRunner(manifest, library, threads=threads)
    return runner, manifest, library
