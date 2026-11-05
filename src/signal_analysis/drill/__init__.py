"""Manual signal-analysis drill: catalog, waveform generation and scoring.

This package is deliberately independent of Qt and of the workspace database;
the desktop page is a thin view over :class:`DrillSession`.
"""

from .catalog import (AUX_INDEX, AUX_LABELS, CLASSES, CLASS_BY_KEY, GROUP_ORDER,
                      KEYS, SignalClass, SignalSpec, classes_for_group,
                      grouped_classes)
from .session import AnswerResult, DrillSession, Round

__all__ = [
    "AUX_INDEX", "AUX_LABELS", "CLASSES", "CLASS_BY_KEY", "GROUP_ORDER",
    "KEYS", "SignalClass", "SignalSpec", "AnswerResult", "DrillSession",
    "Round", "classes_for_group", "grouped_classes",
]
