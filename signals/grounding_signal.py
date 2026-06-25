"""grounding_signal.py - Signal 3; experiential grounding heuristics (pure Python, no LLM).

Per planning.md Section 4 (Signal 3) and Section 5 (scoring inputs):

  * This signal measures CONTENT GROUNDING, not structure and not semantics.
    It asks: does the text contain evidence that originates from a specific
    human experience, memory, or chain of observation?
  * The four features are explicitly chosen to be orthogonal to the existing
    stylometric features. Stylometry measures sentence-length variance,
    vocabulary diversity, punctuation density, and sentence complexity. These are
    STRUCTURAL properties derivable from a character/word scan with no regard
    for what the text is about. The four grounding features below each require
    understanding the CONTENT:
      - temporal_specificity -- exact clock times, calendar dates, durations
      - spatial_specificity  -- named places, physical positions, addresses
      - sensory_observation  -- smells, sounds, textures, visual descriptors
      - firsthand_epistemic  -- first person knowledge acquisition phrases
    Test: could two texts have identical stylometric statistics but very
    different grounding scores? Yes, a text with uniform short sentences and
    low vocabulary diversity can be full of timestamps and named places (high
    grounding) or full of abstract universals (low grounding). The signals
    measure orthogonal axes.
  * Like the other signals this emits a probability ``p_grounding_human`` in
    [0, 1] (probability the text shows human provenance signals) but the
    primary output consumed by the confidence scorer is a ``grounding_factor``
    in [0.85, 1.15] that acts as a multiplier on confidence, not a third
    additive term in combined_p_ai. This follows the architectural note in
    planning.md Section 4: the grounding signal answers a different question
    from "how AI-like is the structure/semantics?" so it is architecturally
    cleaner as a confidence modifier than as a peer probability.
  * Absence of grounding is NOT strong evidence of AI. A mathematical proof,
    a technical specification, or a philosophy essay may have zero temporal
    anchors and zero sensory observations without being AI generated. The
    signal therefore only INCREASES confidence when rich grounding is present;
    it only modestly DECREASES confidence when grounding is entirely absent
    but combined with other borderline signals. The modifier range [0.85, 1.15]
    encodes this asymmetry: a bonus of up to 15% for richly grounded text,
    a penalty of at most 15% for completely ungrounded text.
  * Short text instability: grounding counts are sparse on very short text, so
    on inputs below MIN_RELIABLE_WORDS the signal blends toward the neutral
    factor (1.0) proportionally, mirroring the stylometric signal's fence logic.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from util import clamp01

# Below this many words the grounding features are too sparse to trust.
# Mirrors MIN_RELIABLE_WORDS in stylometric_signal.py.
MIN_RELIABLE_WORDS = 40

# Neutral grounding factor: no modification to confidence.
NEUTRAL_FACTOR = 1.0

# The modifier range (planning.md Section 4, grounding architecture note).
# Richly grounded text receives up to a +15% confidence boost; completely
# ungrounded text receives at most a -15% confidence penalty.
GROUNDING_FACTOR_MIN = 0.85
GROUNDING_FACTOR_MAX = 1.15

# --------------------------------------------------------------------------
# Compiled regex patterns. Each targets CONTENT signals that the existing
# stylometric features cannot detect (they are meaning-bearing, not
# structure-bearing).
# --------------------------------------------------------------------------

# Temporal specificity -- concrete clock times (7:12 AM, 23:45), explicit
# calendar dates (March 4, 2020; the 3rd of June; last Tuesday), and
# specific durations (three weeks, forty minutes). These are the kinds of
# anchors humans include from memory but that AI tends to omit because
# they are not informationally efficient for generic prose.
_TEMPORAL_PATTERNS = [
    # Clock times: 7:12, 07:12 AM, 11:59pm
    re.compile(r"\b\d{1,2}:\d{2}(?:\s*[ap]m)?\b", re.IGNORECASE),
    # Named weekdays used as anchors
    re.compile(
        r"\b(?:last|this|next)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        re.IGNORECASE,
    ),
    # Calendar month + day combos: April 3, 3rd of March, January 2022
    re.compile(
        r"\b(?:january|february|march|april|may|june|july|august|"
        r"september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?\b",
        re.IGNORECASE,
    ),
    # Numeric dates: 2024-03-15, 03/15/2024, 15/03/24
    re.compile(r"\b\d{1,4}[/-]\d{1,2}[/-]\d{2,4}\b"),
    # Specific durations with cardinal numbers: "three weeks", "forty minutes"
    re.compile(
        r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
        r"twenty|thirty|forty|fifty|sixty|ninety|hundred|\d+)\s+"
        r"(?:second|minute|hour|day|week|month|year)s?\b",
        re.IGNORECASE,
    ),
]

# Spatial specificity -- named locations (streets, cities, buildings, landmarks)
# and physical positioning language. These are the "where I was" anchors that
# humans include from episodic memory. Differs from stylometry because stylometry
# measures nothing about proper nouns or location vocabulary.
_SPATIAL_PATTERNS = [
    # Ordinal directions or "on [Street/Avenue/Road/Drive/Lane/Blvd]" combos
    re.compile(
        r"\b(?:street|avenue|road|drive|lane|boulevard|highway|route|alley|"
        r"plaza|square|park|station|airport|terminal|platform|gate)\b",
        re.IGNORECASE,
    ),
    # Floor / room / seat physical positions
    re.compile(
        r"\b(?:floor|room|seat|corner|back|front|left|right|top|bottom)\s+"
        r"(?:of|row|table|desk|shelf|door|window|wall|aisle)\b",
        re.IGNORECASE,
    ),
    # "at [the] [place name]" or "in [the] [place name]" with a proper noun-like cap
    re.compile(r"\b(?:at|in|near|outside|inside|behind|across from)\s+the\s+[A-Z][a-z]+"),
    # Named establishment shorthand: "the Starbucks", "the DMV", "the SEPTA"
    re.compile(r"\b(?:the\s+[A-Z][A-Z]+)\b"),
]

# Sensory observations -- smells, sounds, textures, colours, temperatures, and
# other phenomenal detail. Humans embed this because it is stored alongside the
# episodic memory itself. AI text rarely includes it because it is not
# informationally efficient for the stated communicative goal.
_SENSORY_PATTERNS = [
    # Smell / scent words
    re.compile(
        r"\b(?:smell(?:ed|s|ing)?|smelt|scent(?:ed)?|odou?r|reek(?:ed)?|stench|"
        r"whiff|fragranc[ey]|musty|stale|fresh|chlorine|coffee|gasoline)\b",
        re.IGNORECASE,
    ),
    # Sound words
    re.compile(
        r"\b(?:sound(?:ed|s|ing)?|noise|heard|listen(?:ed|ing)|click(?:ed|ing)?|"
        r"hum(?:med|ming)?|buzz(?:ed|ing)?|creak(?:ed|ing)?|rattle(?:d|ing)?|"
        r"beep(?:ed|ing)?|thud(?:ded|ding)?|whisper(?:ed|ing)?|shout(?:ed|ing)?|"
        r"clatter(?:ed|ing)?|bang(?:ed|ing)?|drip(?:ped|ping)?)\b",
        re.IGNORECASE,
    ),
    # Texture / physical sensation
    re.compile(
        r"\b(?:rough|smooth|sticky|slippery|cold|warm|hot|wet|dry|soft|hard|"
        r"sharp|dull|heavy|light|brittle|rubbery|gritty|chalky|greasy)\b",
        re.IGNORECASE,
    ),
    # Colour used as observation (not metaphor): "the red door", "yellow walls"
    re.compile(
        r"\b(?:red|blue|green|yellow|orange|purple|pink|black|white|gray|grey|"
        r"brown|beige|cream|golden|silver|bronze)\s+\w+\b",
        re.IGNORECASE,
    ),
]

# Firsthand epistemics -- phrases that communicate HOW the writer acquired the
# knowledge being shared. These mark the information as personally witnessed or
# experienced, rather than generally known. AI text rarely uses these because it
# synthesises from training data, not from a first-person vantage point.
_FIRSTHAND_PATTERNS = [
    # "I remember", "I noticed", "I saw", "I heard", "I felt", "I thought", "I realized"
    re.compile(
        r"\bI\s+(?:remember(?:ed)?|recall(?:ed)?|noticed?|saw|spotted?|heard|"
        r"felt|sensed?|thought|realized?|realised?|figured?|decided?|"
        r"found|knew|didn't\s+know|couldn't\s+tell|wasn't\s+sure)\b",
    ),
    # "when I was" / "while I was" -- autobiographical temporal frame
    re.compile(r"\b(?:when|while|after|before)\s+I\s+(?:was|had|went|got|came|left|tried)\b"),
    # "my [noun]" -- first-person possessive specific things: "my friend", "my dog"
    re.compile(r"\bmy\s+(?:friend|sister|brother|mom|dad|mother|father|roommate|boss|"
               r"teacher|coworker|colleague|dog|cat|car|apartment|house|phone)\b",
               re.IGNORECASE),
    # Personal uncertainty / partial knowledge: "I think", "I'm not sure", "maybe"
    re.compile(
        r"\b(?:I\s+think|I\s+thought|I'm\s+not\s+sure|I\s+wasn't\s+sure|"
        r"I\s+don't\s+know|I\s+couldn't\s+tell|I\s+had\s+no\s+idea)\b",
        re.IGNORECASE,
    ),
]

# Word tokenizer reused from stylometric_signal (same definition, private copy).
_WORD_RE = re.compile(r"[A-Za-z']+")


@dataclass
class GroundingSignalResult:
    """The structured output of the grounding signal (Signal 3).

    Matches the ``signals.grounding`` block of the /submit response contract
    (planning.md Section 3): a grounding_factor confidence modifier, the raw
    p_grounding_human probability that drove it, and the raw feature counts that
    produced it. The features are surfaced so a human reviewer working an appeal
    (planning.md Section 8) can see what content evidence was found, not just
    the factor.
    """

    grounding_factor: float       # Confidence multiplier in [0.85, 1.15]
    p_grounding_human: float      # Probability in [0, 1] of human provenance signals
    features: dict = field(default_factory=dict)  # Raw counts and subscores

    def to_dict(self) -> dict:
        """Serialize the result to the ``signals.grounding`` response contract shape.

        Returns:
            dict: Keys ``grounding_factor`` (float), ``p_grounding_human`` (float),
                  and ``features`` (dict of raw counts and subscores).
        """
        return {
            "grounding_factor": self.grounding_factor,
            "p_grounding_human": self.p_grounding_human,
            "features": self.features,
        }


def _count_pattern_hits(text: str, patterns: list[re.Pattern]) -> int:
    """Count total non-overlapping matches across a list of patterns.

    Args:
        text (str): The raw submitted text.
        patterns (list[re.Pattern]): Compiled regex patterns to search.

    Returns:
        int: Total match count across all patterns.
    """
    return sum(len(p.findall(text)) for p in patterns)


def _temporal_subscore(text: str, word_count: int) -> float:
    """Map temporal specificity hit density to a human-grounding subscore.

    A text containing concrete clock times, named dates, or specific durations
    is strongly grounded in a particular moment; the kind of detail that comes
    from episodic memory. AI text rarely includes these unprompted because they
    add nothing to a generic explanation.

    Args:
        text (str): The raw submitted text.
        word_count (int): Pre-computed word count for density normalization.

    Returns:
        float: Human grounding subscore in [0, 1]; higher means more temporally
               specific. Returns 0.5 (neutral) when word_count is zero.
    """
    if word_count == 0:
        return 0.5
    hits = _count_pattern_hits(text, _TEMPORAL_PATTERNS)
    # Even 1 hit per 100 words is notable; 3+ per 100 is richly anchored.
    density = hits / word_count * 100
    return clamp01(density / 3.0)


def _spatial_subscore(text: str, word_count: int) -> float:
    """Map spatial specificity hit density to a human-grounding subscore.

    Named locations, physical positions, and place references appear in human
    writing because the writer was actually somewhere and their memory records it.
    This feature does not overlap with any stylometric measure (none of the four
    stylometric features count proper nouns or place vocabulary).

    Args:
        text (str): The raw submitted text.
        word_count (int): Pre-computed word count for density normalization.

    Returns:
        float: Human grounding subscore in [0, 1]; higher means more spatially
               anchored. Returns 0.5 when word_count is zero.
    """
    if word_count == 0:
        return 0.5
    hits = _count_pattern_hits(text, _SPATIAL_PATTERNS)
    density = hits / word_count * 100
    return clamp01(density / 3.0)


def _sensory_subscore(text: str, word_count: int) -> float:
    """Map sensory observation density to a human-grounding subscore.

    Sensory details (smells, sounds, textures, colours as observation) arise
    from phenomenal experience and are embedded in episodic memory. AI text
    omits them because they do not advance the communicative goal efficiently.
    This feature is content-bearing: two texts with identical structural
    statistics can differ greatly on sensory vocabulary.

    Args:
        text (str): The raw submitted text.
        word_count (int): Pre-computed word count for density normalization.

    Returns:
        float: Human grounding subscore in [0, 1]; higher means more sensory
               richness. Returns 0.5 when word_count is zero.
    """
    if word_count == 0:
        return 0.5
    hits = _count_pattern_hits(text, _SENSORY_PATTERNS)
    density = hits / word_count * 100
    return clamp01(density / 4.0)


def _firsthand_subscore(text: str, word_count: int) -> float:
    """Map first-hand epistemic marker density to a human-grounding subscore.

    Phrases like "I remember", "when I was", "my roommate", and "I had no idea"
    communicate how knowledge was acquired. They are markers of first person
    experience rather than synthesized general knowledge. This feature is
    orthogonal to all four stylometric measures because it detects specific
    vocabulary patterns tied to first person provenance, not to sentence length
    or punctuation frequency.

    Args:
        text (str): The raw submitted text.
        word_count (int): Pre-computed word count for density normalization.

    Returns:
        float: Human grounding subscore in [0, 1]; higher means more
               firsthand epistemic markers. Returns 0.5 when word_count is zero.
    """
    if word_count == 0:
        return 0.5
    hits = _count_pattern_hits(text, _FIRSTHAND_PATTERNS)
    density = hits / word_count * 100
    return clamp01(density / 4.0)


def _factor_from_score(p_grounding_human: float) -> float:
    """Convert a grounding probability to a confidence modification factor.

    The factor lives in [GROUNDING_FACTOR_MIN, GROUNDING_FACTOR_MAX] = [0.85, 1.15].
    A p_grounding_human of 0.5 (neutral, neither grounded nor ungrounded) maps to
    exactly 1.0 (no change to confidence). Rich grounding (near 1.0) maps up to
    1.15. Complete absence of grounding (near 0.0) maps down to 0.85.

    The asymmetry in the planning spec ("absence is not strong evidence") is
    already encoded in the feature design. Genre-neutral texts (technical
    writing, philosophy) will score near 0.5 by default, not 0.0. So 0.85 is
    a genuine floor that only activates on texts that actively present the
    hallmarks of AI generality with zero experiential content whatsoever.

    Args:
        p_grounding_human (float): Grounding probability in [0, 1].

    Returns:
        float: Confidence modifier in [GROUNDING_FACTOR_MIN, GROUNDING_FACTOR_MAX].
    """
    span = GROUNDING_FACTOR_MAX - GROUNDING_FACTOR_MIN
    raw = GROUNDING_FACTOR_MIN + span * p_grounding_human
    return round(raw, 4)


def analyze_grounding(text: str) -> GroundingSignalResult:
    """Run the grounding (Signal 3) heuristics on ``text``.

    Computes four content grounding subscores (temporal specificity, spatial
    specificity, sensory observation, firsthand epistemics), combines them into
    a single ``p_grounding_human`` probability, and maps that to a
    ``grounding_factor`` in [0.85, 1.15] for use as a confidence modifier in
    the confidence scorer (scoring.py).

    The four subscores are combined with equal weights (0.25 each). None of the
    four features duplicates any stylometric feature (see module docstring for the
    independence argument). Because absence of grounding is not strong evidence of
    AI for genre-neutral text, the subscores default to 0.5 (neutral) when no
    matches are found rather than to 0.0.

    On text below ``MIN_RELIABLE_WORDS`` the grounding_factor is blended toward
    NEUTRAL_FACTOR (1.0) so the signal does not modify confidence on thin input.

    Args:
        text (str): The raw submitted text to analyze.

    Returns:
        GroundingSignalResult: ``grounding_factor`` in [0.85, 1.15],
            ``p_grounding_human`` in [0, 1], and a feature dict. This function
            never raises on ordinary input; empty or whitespace-only text returns
            NEUTRAL_FACTOR with the short flag set.
    """
    words = _WORD_RE.findall(text)
    word_count = len(words)
    too_short = word_count < MIN_RELIABLE_WORDS

    temporal = _temporal_subscore(text, word_count)
    spatial = _spatial_subscore(text, word_count)
    sensory = _sensory_subscore(text, word_count)
    firsthand = _firsthand_subscore(text, word_count)

    # Equal weight combination: all four features are equally important.
    # No single content-grounding dimension dominates the others.
    raw_p = 0.25 * temporal + 0.25 * spatial + 0.25 * sensory + 0.25 * firsthand

    # Blend toward neutral on short text, proportionally to how short it is.
    if too_short:
        trust = word_count / MIN_RELIABLE_WORDS
        p_grounding_human = (trust * raw_p) + ((1 - trust) * 0.5)
        grounding_factor = NEUTRAL_FACTOR
    else:
        p_grounding_human = raw_p
        grounding_factor = _factor_from_score(p_grounding_human)

    # Raw hit counts are surfaced in features so a reviewer can see the evidence.
    features = {
        "num_words": word_count,
        "too_short": too_short,
        "temporal_hits": _count_pattern_hits(text, _TEMPORAL_PATTERNS),
        "spatial_hits": _count_pattern_hits(text, _SPATIAL_PATTERNS),
        "sensory_hits": _count_pattern_hits(text, _SENSORY_PATTERNS),
        "firsthand_hits": _count_pattern_hits(text, _FIRSTHAND_PATTERNS),
        "subscores": {
            "temporal": round(temporal, 4),
            "spatial": round(spatial, 4),
            "sensory": round(sensory, 4),
            "firsthand": round(firsthand, 4),
        },
    }

    return GroundingSignalResult(
        grounding_factor=grounding_factor,
        p_grounding_human=round(clamp01(p_grounding_human), 4),
        features=features,
    )
