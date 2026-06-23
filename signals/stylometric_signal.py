"""stylometric_signal.py - Signal 2; deterministic stylometric heuristics (pure Python, no LLM).

Per planning.md Section 4 (Signal 2) and Section 5 (scoring inputs):

  * This signal measures STRUCTURE, not meaning. It computes measurable
    statistical properties of the text and maps them to a probability ``p_ai``
    in [0, 1]. It understands nothing about what the text says, which is
    exactly why it is not fooled by the register bias that trips the LLM
    (Section 4, "Why the pairing works").
  * Four features are computed (Section 4, Signal 2):
      - sentence length variance (burstiness)  -> low variance reads AI-like
      - type-token ratio (vocabulary diversity) -> low diversity reads AI-like
      - punctuation density and variety          -> flat punctuation reads AI-like
      - mean sentence complexity (words/sentence) -> used alongside variance
    Each feature is mapped to the "AI-like" end of its expected range, then the
    four subscores are combined into a single ``p_ai_style``.
  * Like Signal 1 this is NEVER a binary flag. It emits a probability plus the
    raw feature dict so the confidence scorer (Section 5) and a human reviewer
    (Section 8) can see exactly what drove the number.
  * Short text instability (Section 4 blind spots): variance and type-token
    ratio are noisy on very short inputs, so on too little text the signal pulls
    toward the neutral fence rather than asserting a confident structural read.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from util import clamp01

# Below this many words the structural features (variance, TTR) are too noisy to
# trust (Section 4, "Short-text instability"). We still return a reading, but we
# blend it toward the neutral fence so the signal does not overly assert on a
# handful of words. This mirrors the LLM signal's refusal to fake certainty.
MIN_RELIABLE_WORDS = 40

# The neutral fence. A signal that has nothing to say returns 0.5 (max
# ambiguity), matching llm_signal.NEUTRAL_SCORE so both signals speak the same
# language to the confidence scorer.
NEUTRAL_SCORE = 0.5

# Sentence terminators used to split text into sentences.
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+(?:\s+|$)")
# A "word" is a run of letters/apostrophes; this avoids counting punctuation or
# stray digits as vocabulary when computing the type-token ratio.
_WORD_RE = re.compile(r"[A-Za-z']+")
# Punctuation marks we track for the punctuation variety feature.
_PUNCTUATION = set(".,;:!?-—()\"'")


@dataclass
class StylometricSignalResult:
    """The structured output of the stylometric signal.

    Matches the ``signals.stylometric`` block of the /submit response contract
    (planning.md Section 3): a probability and the raw feature dict that
    produced it. The features are surfaced so a human reviewer working an
    appeal (Section 8) can see the structural evidence, not just the number.
    """

    p_ai: float                          # Probability in [0, 1] that text is AI-generated
    features: dict = field(default_factory=dict)  # Raw + per-feature sub-score values

    def to_dict(self) -> dict:
        """Serialize the result to the ``signals.stylometric`` response contract shape.

        Returns:
            dict: Keys ``p_ai`` (float) and ``features`` (dict of raw metrics
                  and their individual AI-leaning sub-scores).
        """
        return {
            "p_ai": self.p_ai,
            "features": self.features,
        }


def _split_sentences(text: str) -> list[str]:
    """Split text into non-empty, stripped sentence strings.

    Sentences are separated on runs of terminal punctuation (``.!?``). Text with
    no terminal punctuation is treated as a single sentence so a one line input
    still yields a measurable unit rather than an empty list.

    Args:
        text (str): The raw submitted text.

    Returns:
        list[str]: Non-empty sentence strings with surrounding whitespace removed.
    """
    parts = _SENTENCE_SPLIT_RE.split(text)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences or ([text.strip()] if text.strip() else [])


def _words(text: str) -> list[str]:
    """Extract lowercase word tokens from text for vocabulary measures.

    Tokens are runs of letters and apostrophes; digits and punctuation are
    excluded so the type-token ratio reflects genuine vocabulary rather than
    formatting artifacts. Words are lowercased so case variants count as one type.

    Args:
        text (str): The raw submitted text.

    Returns:
        list[str]: Lowercased word tokens in order of appearance.
    """
    return [w.lower() for w in _WORD_RE.findall(text)]


def _burstiness_subscore(sentence_word_counts: list[int]) -> float:
    """Map sentence length variance (burstiness) to an AI-leaning subscore.

    Human writing mixes long and short sentences (high variance / high
    burstiness); AI writing is more uniform (low variance). We compute the
    coefficient of variation (std / mean) of sentence word counts so the measure
    is scale independent, then invert it: LOW variability -> HIGH AI-likeness.

    Args:
        sentence_word_counts (list[int]): Word count of each sentence.

    Returns:
        float: AI-leaning subscore in [0, 1]; higher means more uniform/AI-like.
                Returns NEUTRAL_SCORE when fewer than two sentences exist (variance
                is undefined / uninformative).
    """
    if len(sentence_word_counts) < 2:
        return NEUTRAL_SCORE
    mean = sum(sentence_word_counts) / len(sentence_word_counts)
    if mean == 0:
        return NEUTRAL_SCORE
    variance = sum((c - mean) ** 2 for c in sentence_word_counts) / len(sentence_word_counts)
    std = variance ** 0.5
    cv = std / mean  # Coefficient of Variation: scale independent burstiness
    # A CV around 0.6+ is typical of varied human prose; near 0 is robotic.
    # Map cv in [0, 0.6] linearly to AI-likeness in [1, 0], clamping past 0.6.
    return clamp01(1.0 - (cv / 0.6))


def _ttr_subscore(words: list[str]) -> float:
    """Map type-token ratio (vocabulary diversity) to an AI-leaning subscore.

    Type-token ratio is unique words / total words. Higher diversity reads as
    human; lower diversity (repetition, narrow range) reads as AI. Because TTR
    falls naturally as texts get longer, the raw ratio is interpreted against a
    moderate band rather than an absolute cutoff.

    Args:
        words (list[str]): The word tokens of the text.

    Returns:
        float: AI-leaning subscore in [0, 1]; higher means lower diversity/AI-like.
                Returns NEUTRAL_SCORE when there are no words.
    """
    if not words:
        return NEUTRAL_SCORE
    ttr = len(set(words)) / len(words)
    # Raw TTR is dominated by length: short submissions sit near 0.85-0.92 simply
    # because there are too few words to repeat, while a long uniform passage can
    # fall well below 0.5. A single fixed [0.4, 0.7] band (an earlier draft) was
    # MISCALIBRATED for this regime: it pinned every short input to 0.0, so the
    # feature silently contributed nothing. We band against the range where TTR
    # actually varies for the texts this service sees (~0.55 diverse down to
    # ~0.92 repetitive-for-its-length). Higher diversity is human-leaning, so we
    # invert: ttr at/above 0.92 -> 0 AI-likeness, ttr at/below 0.55 -> 1. On short
    # text TTR clusters high, landing this near the low-AI end, which is the
    # HONEST reading: TTR carries little information there (Section 4 blind spot),
    # so it must not fabricate separation it does not have.
    return clamp01((0.92 - ttr) / (0.92 - 0.55))


def _punctuation_subscore(text: str) -> float:
    """Map punctuation density and variety to an AI-leaning subscore.

    AI text often has flatter, more predictable punctuation. We combine two
    cheap measures: the VARIETY of distinct punctuation marks used (more variety
    is human-leaning) and the DENSITY of punctuation per word (extremely sparse
    punctuation is AI-leaning). The two are averaged into one subscore.

    Args:
        text (str): The raw submitted text.

    Returns:
        float: AI-leaning sub-score in [0, 1]; higher means flatter/AI-like.
    """
    distinct_marks = sum(1 for mark in _PUNCTUATION if mark in text)
    # Up to ~6 distinct marks reads as varied human punctuation; 0-1 is flat.
    variety_ai = clamp01(1.0 - (distinct_marks / 6.0))

    word_count = len(_WORD_RE.findall(text))
    punct_count = sum(1 for ch in text if ch in _PUNCTUATION)
    if word_count == 0:
        density_ai = NEUTRAL_SCORE
    else:
        density = punct_count / word_count
        # ~0.15 punctuation marks per word is normal prose; far below that is
        # unusually flat. Map density in [0, 0.15] to AI-likeness in [1, 0].
        density_ai = clamp01(1.0 - (density / 0.15))

    return (variety_ai + density_ai) / 2.0


def _complexity_subscore(sentence_word_counts: list[int]) -> float:
    """Map mean sentence complexity (words per sentence) to an AI-leaning sub-score.

    Used alongside burstiness so the signal measures variability rather than raw
    length. Very long, uniformly complex sentences lean slightly AI; this is a
    weak secondary cue and is deliberately mapped to a narrow, gentle range so it
    never dominates the structural read on its own.

    Args:
        sentence_word_counts (list[int]): Word count of each sentence.

    Returns:
        float: AI-leaning sub-score in [0, 1]. Returns NEUTRAL_SCORE when no
                sentences exist.
    """
    if not sentence_word_counts:
        return NEUTRAL_SCORE
    mean_len = sum(sentence_word_counts) / len(sentence_word_counts)
    # Sentences averaging ~12 words are unremarkable; ~28+ words sustained is
    # mildly AI-leaning (long, even clauses). Map [12, 28] to [0.5, 1.0] and
    # short choppy means (< 12) gently toward human.
    if mean_len <= 12:
        return clamp01(mean_len / 24.0)  # 12 -> 0.5, shorter -> below 0.5
    return clamp01(0.5 + (mean_len - 12) / 32.0)  # 28 -> ~1.0


def analyze_stylometry(text: str) -> StylometricSignalResult:
    """Run the stylometric (Signal 2) heuristics on ``text``.

    Computes the four Section 4 features (burstiness, type-token ratio,
    punctuation, mean complexity), maps each to its AI-leaning end, and combines
    them into a single ``p_ai`` probability. The raw metrics and per-feature
    subscores are returned in ``features`` for transparency and auditing.

    The four subscores are combined with weights that reflect how informative
    each feature is: burstiness and type-token ratio are the strongest
    structural cues and carry most of the weight; punctuation is a moderate cue;
    mean complexity is a weak secondary cue.

    On text below ``MIN_RELIABLE_WORDS`` the combined score is blended toward
    NEUTRAL_SCORE (Section 4, short-text instability) so the signal does not
    overly assert on too little data; a ``too_short`` flag is recorded in features.

    Args:
        text (str): The raw submitted text to analyze.

    Returns:
        StylometricSignalResult: ``p_ai`` in [0, 1] plus the feature dict. This
            function never raises on ordinary input; empty or whitespace text
            returns NEUTRAL_SCORE with the short-text flag set.
    """
    sentences = _split_sentences(text)
    words = _words(text)
    sentence_word_counts = [len(_WORD_RE.findall(s)) for s in sentences]

    burstiness = _burstiness_subscore(sentence_word_counts)
    ttr = _ttr_subscore(words)
    punctuation = _punctuation_subscore(text)
    complexity = _complexity_subscore(sentence_word_counts)

    # Weighted combination. Burstiness and TTR are the load-bearing structural
    # features; punctuation is secondary; complexity is a light tiebreaker.
    raw_p_ai = (
        0.35 * burstiness
        + 0.35 * ttr
        + 0.20 * punctuation
        + 0.10 * complexity
    )

    too_short = len(words) < MIN_RELIABLE_WORDS
    if too_short:
        # Blend toward the neutral fence proportionally to how short the text is:
        # near zero words -> almost fully neutral; near the threshold -> mostly
        # the real reading. This keeps the signal honest on thin input.
        trust = len(words) / MIN_RELIABLE_WORDS
        p_ai = (trust * raw_p_ai) + ((1 - trust) * NEUTRAL_SCORE)
    else:
        p_ai = raw_p_ai

    features = {
        # Raw, human-readable metrics (Section 11 / Section 8: a reviewer reads these).
        "num_sentences": len(sentences),
        "num_words": len(words),
        "type_token_ratio": round(len(set(words)) / len(words), 4) if words else None,
        "mean_sentence_length": round(sum(sentence_word_counts) / len(sentence_word_counts), 2)
        if sentence_word_counts else None,
        "too_short": too_short,
        # Per-feature AI-leaning sub-scores, so a divergent combined number can
        # be traced back to the feature that drove it.
        "subscores": {
            "burstiness": round(burstiness, 4),
            "type_token_ratio": round(ttr, 4),
            "punctuation": round(punctuation, 4),
            "complexity": round(complexity, 4),
        },
    }

    return StylometricSignalResult(p_ai=clamp01(p_ai), features=features)
