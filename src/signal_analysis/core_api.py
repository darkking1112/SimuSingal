"""Stable API shared by source and binary builds; no GUI imports."""

from ._numeric import (MAX_SAMPLES, MODE_NAMES, analyze, classify_modulation, generate_iq,
                       make_demo, plan_signal, spectrum_row, validate_rate, validate_samples)

__all__ = ["MAX_SAMPLES", "MODE_NAMES", "analyze", "classify_modulation", "generate_iq",
           "make_demo", "plan_signal", "spectrum_row", "validate_rate", "validate_samples"]
