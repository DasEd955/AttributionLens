"""util - Shared utility functions for the AttributionLens pipeline.

Re-exports the public helpers from util.util so callers can import directly
from the package (``from util import clamp01``) rather than from the submodule.

Current exports:
  clamp01   Clamp a float to [0.0, 1.0].
"""

from util.util import clamp01

__all__ = ["clamp01"]
