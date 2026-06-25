"""util.py - Shared utility functions used across the AttributionLens pipeline.

Common helpers that appear in multiple modules live here to avoid repetition.
Import from this module (or from the ``util`` package directly via ``util``)
rather than redefining these in individual signal or scoring modules.
"""

from __future__ import annotations
import re


# ---------------------------------------------------------------------------
# Constants shared across signal modules
# ---------------------------------------------------------------------------

# The neutral fence. A signal that has nothing to say returns 0.5 (max
# ambiguity). All three signals speak the same language to the confidence
# scorer so this value must be a single source of truth rather than three
# independent literals that happen to agree.
NEUTRAL_SCORE: float = 0.5

# Below this many words the structural features (variance, TTR) and grounding
# features are too noisy to trust (Section 4, "Short-text instability"). Signals
# still return a reading, but they blend it toward the neutral fence so they do
# not overly assert on a handful of words.
MIN_RELIABLE_WORDS: int = 40


# ---------------------------------------------------------------------------
# Compiled regex patterns shared across signal modules
# ---------------------------------------------------------------------------

# A "word" is a run of letters/apostrophes; this avoids counting punctuation or
# stray digits as vocabulary when computing the type-token ratio or grounding
# hit densities.
WORD_RE: re.Pattern = re.compile(r"[A-Za-z']+")

# Sentence terminators used to split text into sentences.
SENTENCE_SPLIT_RE: re.Pattern = re.compile(r"[.!?]+(?:\s+|$)")


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def clamp01(value: float) -> float:
    """Clamp a float to the closed interval [0.0, 1.0].

    Args:
        value (float): The value to clamp.

    Returns:
        float: value clamped to [0.0, 1.0].
    """
    return max(0.0, min(1.0, value))


def extract_words(text: str) -> list[str]:
    """Extract lowercase word tokens from text for vocabulary measures.

    Tokens are runs of letters and apostrophes; digits and punctuation are
    excluded so the type-token ratio reflects genuine vocabulary rather than
    formatting artifacts. Words are lowercased so case variants count as one type.

    Args:
        text (str): The raw submitted text.

    Returns:
        list[str]: Lowercased word tokens in order of appearance.
    """
    return [w.lower() for w in WORD_RE.findall(text)]


def split_sentences(text: str) -> list[str]:
    """Split text into non-empty, stripped sentence strings.

    Sentences are separated on runs of terminal punctuation (``.!?``). Text with
    no terminal punctuation is treated as a single sentence so a one line input
    still yields a measurable unit rather than an empty list.

    Args:
        text (str): The raw submitted text.

    Returns:
        list[str]: Non-empty sentence strings with surrounding whitespace removed.
    """
    parts = SENTENCE_SPLIT_RE.split(text)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences or ([text.strip()] if text.strip() else [])
