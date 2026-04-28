"""
shape_templates.py — produce normalized cumulative target profiles
per shape for length N.

Each profile is length N, range roughly [-1, 1] (after detrend),
centered, suitable as a target trajectory for sequential slot fill.
"""
from __future__ import annotations

import numpy as np

from phase_A_shape_diagnostic import SHAPES


def template_profile(shape: str, n: int) -> np.ndarray:
    """Return cumulative-trajectory template of length n for given shape.
    Values are detrended (linear fit removed) and roughly normalized
    so peak-to-trough magnitude is ~1."""
    if n < 2:
        return np.zeros(n)
    x = np.linspace(0, 1, n)

    if shape == "flat":
        prof = np.zeros(n)
    elif shape == "linear_up":
        prof = x  # rises 0→1
    elif shape == "linear_dn":
        prof = -x  # falls 0→-1
    elif shape == "U":
        # Parabola dipping in the middle
        prof = (2 * x - 1) ** 2 - 0.5  # ranges [-0.5, 0.5]
    elif shape == "inv_U":
        prof = 0.5 - (2 * x - 1) ** 2
    elif shape == "W":
        # Two dips: sin(2πx) shifted
        prof = -np.cos(2 * np.pi * x) * 0.5 + 0.0
        prof = -np.abs(prof) + np.abs(prof).mean()  # smooth two dips
    elif shape == "M":
        prof = np.cos(2 * np.pi * x) * 0.5
    elif shape == "other":
        # No constraint — flat zero profile (no shape pull)
        prof = np.zeros(n)
    else:
        prof = np.zeros(n)

    # Detrend: subtract linear fit
    if n >= 2:
        slope, intercept = np.polyfit(x, prof, 1)
        prof = prof - (slope * x + intercept)

    return prof
