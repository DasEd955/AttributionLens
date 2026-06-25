"""util - Shared utility functions for the AttributionLens pipeline.

Re-exports the public helpers from util.util so callers can import directly
from the package (``from util import clamp01``) rather than from the submodule.

Current exports:
  clamp01            Clamp a float to [0.0, 1.0].
  NEUTRAL_SCORE      Neutral fence value (0.5) shared across all signals.
  MIN_RELIABLE_WORDS Minimum word count below which signal features are noisy.
  WORD_RE            Compiled regex for extracting word tokens.
  SENTENCE_SPLIT_RE  Compiled regex for splitting text into sentences.
  extract_words      Extract lowercased word tokens from text.
  split_sentences    Split text into non-empty sentence strings.
"""

from util.util import (
    clamp01,
    NEUTRAL_SCORE,
    MIN_RELIABLE_WORDS,
    WORD_RE,
    SENTENCE_SPLIT_RE,
    extract_words,
    split_sentences,
)

__all__ = [
    "clamp01",
    "NEUTRAL_SCORE",
    "MIN_RELIABLE_WORDS",
    "WORD_RE",
    "SENTENCE_SPLIT_RE",
    "extract_words",
    "split_sentences",
]
