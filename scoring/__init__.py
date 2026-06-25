"""scoring - Confidence scoring, grounding signal, and transparency label generation.

Combines signal probabilities into calibrated verdicts and reader-facing labels,
with grounding-based confidence modifiers for human provenance signals.

Current submodules:
  scoring       Confidence scorer - merges two independent classification signals
                into one calibrated verdict with confidence weighting. Signal 3
                (grounding) acts as a confidence modifier via grounding_factor.
  labels        Transparency label generator - maps verdicts and confidence values
                to reader-facing text variants.
  grounding_signal (in signals/) - Experiential content grounding analysis that
                measures human provenance specificity (temporal anchors, spatial
                references, sensory observations, firsthand epistemics).
"""
