"""Stable API shared by source and binary builds; no GUI imports."""

from ._numeric import analyze, make_demo, validate_rate, validate_samples

__all__ = ["analyze", "make_demo", "validate_rate", "validate_samples"]
