"""util.py - Shared utility functions used across the AttributionLens pipeline.

Common helpers that appear in multiple modules live here to avoid repetition.
Import from this module (or from the ``util`` package directly via ``util``)
rather than redefining these in individual signal or scoring modules.
"""

from __future__ import annotations


def clamp01(value: float) -> float:
    """Clamp a float to the closed interval [0.0, 1.0].

    Args:
        value (float): The value to clamp.

    Returns:
        float: value clamped to [0.0, 1.0].
    """
    return max(0.0, min(1.0, value))
