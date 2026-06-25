"""test_grounding_signal.py - Unit tests for Signal 3, the grounding heuristics in grounding_signal.py.

Verifies the spec contract from planning.md Section 4 (Signal 3):
  analyze_grounding returns a grounding_factor, a p_grounding_human, and a feature dict.
  grounding_factor is always in [0.85, 1.15].
  p_grounding_human is always in [0.0, 1.0].
  The four features (temporal, spatial, sensory, firsthand) are each computed independently.
  A richly grounded personal narrative scores higher than generic abstract prose.
  Short text is flagged and the grounding_factor collapses to NEUTRAL_FACTOR (1.0).
  Independence from stylometric features: two texts with similar structure but
    different content grounding must produce meaningfully different grounding scores.
  to_dict returns the correct contract shape.
"""

import pytest
from signals.grounding_signal import (
    GROUNDING_FACTOR_MAX,
    GROUNDING_FACTOR_MIN,
    MIN_RELIABLE_WORDS,
    NEUTRAL_FACTOR,
    GroundingSignalResult,
    analyze_grounding,
    _temporal_subscore,
    _spatial_subscore,
    _sensory_subscore,
    _firsthand_subscore,
)

# A richly grounded personal narrative: specific clock time, named place,
# sensory details, first-person epistemic markers, incidental observations.
GROUNDED_TEXT = (
    "I got to the station at 7:10 because Google Maps said the train was delayed. "
    "It wasn't. The doors closed right as I reached platform 4. The platform smelled "
    "faintly like coffee and diesel. I remember checking my watch twice before I "
    "realized the train had already left. My roommate had warned me about that line "
    "being unreliable. I stood in the back corner of the waiting room for about "
    "twenty minutes, watching the pigeons on the overhead beams. When I was finally "
    "on the next train, I noticed the seat fabric was sticky, probably from the "
    "heat. I had no idea the trip would take an extra forty minutes."
)

# A generic, abstract AI style passage with no temporal anchors, no named
# locations, no sensory observations, and no first-person epistemic markers.
ABSTRACT_TEXT = (
    "Public transportation systems play an important role in urban mobility. "
    "Reliable scheduling is essential for commuters who depend on these services "
    "on a daily basis. When disruptions occur, it is important for transit "
    "authorities to communicate delays clearly and promptly. Passengers benefit "
    "from accurate real-time information, which allows them to make informed "
    "decisions about their travel plans. The relationship between infrastructure "
    "investment and service reliability has been documented in numerous studies. "
    "Effective management of transit systems contributes positively to both "
    "economic productivity and environmental sustainability across communities."
)

# Two texts designed to have SIMILAR stylometric statistics but DIFFERENT grounding.
# Both use uniform short sentences and restricted vocabulary (similar structure).
# But STRUCTURAL_GROUNDED has temporal and sensory content; STRUCTURAL_BARE does not.
# This is the key independence test.
STRUCTURAL_GROUNDED = (
    "At 8:15 AM I arrived. The room smelled like bleach. I noticed the yellow walls. "
    "My friend was already there. I had no idea what to say. The floor was cold. "
    "I remember standing by the window. It was March 4th. I heard a loud click. "
    "The seat felt rough. I waited twenty minutes. Outside, the street was wet."
)

STRUCTURAL_BARE = (
    "The process began. The system checked. The output was produced. Results came back. "
    "The status was updated. A confirmation was sent. The record was stored. "
    "Verification was complete. The workflow ended. The log was written. "
    "Processing finished. The queue was cleared."
)


# --------------------------------------------------------------------------
# Basic Contract Tests
# --------------------------------------------------------------------------

def test_returns_result_dataclass():
    """analyze_grounding returns a GroundingSignalResult, not a raw number."""
    result = analyze_grounding(GROUNDED_TEXT)
    assert isinstance(result, GroundingSignalResult)


def test_grounding_factor_always_in_range():
    """grounding_factor is always in [GROUNDING_FACTOR_MIN, GROUNDING_FACTOR_MAX]."""
    for text in (GROUNDED_TEXT, ABSTRACT_TEXT, STRUCTURAL_BARE, ""):
        result = analyze_grounding(text)
        assert GROUNDING_FACTOR_MIN <= result.grounding_factor <= GROUNDING_FACTOR_MAX


def test_p_grounding_human_always_in_unit_interval():
    """p_grounding_human is always in [0, 1]."""
    for text in (GROUNDED_TEXT, ABSTRACT_TEXT, STRUCTURAL_BARE, ""):
        result = analyze_grounding(text)
        assert 0.0 <= result.p_grounding_human <= 1.0


def test_features_dict_present_with_all_keys():
    """The feature dict contains the four subscore keys and the raw hit counts."""
    features = analyze_grounding(GROUNDED_TEXT).features
    for key in ("temporal_hits", "spatial_hits", "sensory_hits", "firsthand_hits",
                "num_words", "too_short"):
        assert key in features, f"Expected key '{key}' in features"
    assert set(features["subscores"].keys()) == {
        "temporal", "spatial", "sensory", "firsthand",
    }


def test_to_dict_matches_contract():
    """to_dict returns exactly the three keys in the signals.grounding response block."""
    d = analyze_grounding(GROUNDED_TEXT).to_dict()
    assert set(d.keys()) == {"grounding_factor", "p_grounding_human", "features"}


# --------------------------------------------------------------------------
# Directional Separation Test
# --------------------------------------------------------------------------

def test_grounded_narrative_scores_higher_than_abstract():
    """A richly grounded personal narrative scores higher p_grounding_human than generic prose.

    This is the core directional contract: the signal must separate text that
    originates from a specific human experience from text that synthesises
    general knowledge without any experiential anchors.
    """
    grounded = analyze_grounding(GROUNDED_TEXT).p_grounding_human
    abstract = analyze_grounding(ABSTRACT_TEXT).p_grounding_human
    assert grounded > abstract, (
        f"Expected grounded ({grounded:.3f}) > abstract ({abstract:.3f})"
    )


def test_grounded_narrative_factor_above_neutral():
    """Richly grounded text must produce a grounding_factor above 1.0."""
    result = analyze_grounding(GROUNDED_TEXT)
    assert result.grounding_factor > NEUTRAL_FACTOR


def test_abstract_text_factor_below_neutral():
    """Generic abstract text with no grounding markers must produce a factor below 1.0."""
    result = analyze_grounding(ABSTRACT_TEXT)
    assert result.grounding_factor < NEUTRAL_FACTOR


# --------------------------------------------------------------------------
# Independence from Stylometry Test
# --------------------------------------------------------------------------

def test_similar_structure_different_grounding():
    """Two structurally similar texts produce meaningfully different grounding scores.

    This is the key orthogonality proof from planning.md Section 4:
    'Could two texts have identical stylometric statistics but very different
    provenance scores? If yes, then it is distinct.'
    STRUCTURAL_GROUNDED and STRUCTURAL_BARE have similar short-sentence structure
    and restricted vocabulary (similar burstiness, TTR) but very different
    content grounding (times, sensory details, first-person markers vs. none).
    """
    grounded_score = analyze_grounding(STRUCTURAL_GROUNDED).p_grounding_human
    bare_score = analyze_grounding(STRUCTURAL_BARE).p_grounding_human
    # The grounded text should score meaningfully higher than the bare text.
    assert grounded_score > bare_score + 0.1, (
        f"Expected structural_grounded ({grounded_score:.3f}) to exceed "
        f"structural_bare ({bare_score:.3f}) by at least 0.1"
    )


# --------------------------------------------------------------------------
# Short Text Behaviour
# --------------------------------------------------------------------------

def test_short_text_flagged_and_factor_is_neutral():
    """Text below MIN_RELIABLE_WORDS is flagged too_short and grounding_factor returns 1.0."""
    short = "I smelled coffee at 7 AM near the station."
    result = analyze_grounding(short)
    assert result.features["too_short"] is True
    assert result.grounding_factor == NEUTRAL_FACTOR


def test_long_text_not_flagged_short():
    """A passage above MIN_RELIABLE_WORDS is not flagged as too short."""
    result = analyze_grounding(GROUNDED_TEXT)
    assert result.features["num_words"] >= MIN_RELIABLE_WORDS
    assert result.features["too_short"] is False


def test_empty_text_returns_neutral_without_raising():
    """Whitespace-only text returns NEUTRAL_FACTOR and the short flag, never raising."""
    result = analyze_grounding("   ")
    assert result.grounding_factor == NEUTRAL_FACTOR
    assert result.features["too_short"] is True


# --------------------------------------------------------------------------
# Individual Subscore Feature Tests
# --------------------------------------------------------------------------

def test_temporal_subscore_detects_clock_time():
    """_temporal_subscore finds a specific clock time as a temporal anchor."""
    text = "I arrived at 7:12 AM."
    score = _temporal_subscore(text, word_count=5)
    assert score > 0.0


def test_temporal_subscore_detects_calendar_date():
    """_temporal_subscore detects a calendar month-day combination."""
    text = "It happened on March 4th when everything changed."
    score = _temporal_subscore(text, word_count=9)
    assert score > 0.0


def test_temporal_subscore_detects_specific_duration():
    """_temporal_subscore detects a cardinal numeric duration."""
    text = "I waited thirty minutes outside the building before leaving."
    score = _temporal_subscore(text, word_count=10)
    assert score > 0.0


def test_spatial_subscore_detects_platform_reference():
    """_spatial_subscore finds a physical location word like 'platform'."""
    text = "I ran to platform 4 but the doors were already closed."
    score = _spatial_subscore(text, word_count=12)
    assert score > 0.0


def test_sensory_subscore_detects_smell():
    """_sensory_subscore finds a smell observation."""
    text = "The platform smelled faintly like coffee and diesel fuel."
    score = _sensory_subscore(text, word_count=10)
    assert score > 0.0


def test_sensory_subscore_detects_sound():
    """_sensory_subscore finds a sound observation."""
    text = "The coffee machine kept clicking for about twenty minutes."
    score = _sensory_subscore(text, word_count=10)
    assert score > 0.0


def test_sensory_subscore_detects_texture():
    """_sensory_subscore finds a texture observation."""
    text = "The seat felt rough and sticky in the summer heat."
    score = _sensory_subscore(text, word_count=10)
    assert score > 0.0


def test_firsthand_subscore_detects_i_remember():
    """_firsthand_subscore finds the first-hand marker 'I remember'."""
    text = "I remember checking my watch twice before the train arrived."
    score = _firsthand_subscore(text, word_count=11)
    assert score > 0.0


def test_firsthand_subscore_detects_i_had_no_idea():
    """_firsthand_subscore finds the uncertainty marker 'I had no idea'."""
    text = "I had no idea the trip would take an extra forty minutes."
    score = _firsthand_subscore(text, word_count=12)
    assert score > 0.0


def test_firsthand_subscore_detects_my_friend():
    """_firsthand_subscore finds first-person possessive relationship markers."""
    text = "My roommate had warned me that this line was always late."
    score = _firsthand_subscore(text, word_count=13)
    assert score > 0.0


# --------------------------------------------------------------------------
# Zero Word Edge Case for each Subscore Helper
# --------------------------------------------------------------------------

def test_subscores_neutral_on_empty_text():
    """Each subscore returns 0.5 when word_count is zero (max ambiguity, not zero)."""
    assert _temporal_subscore("", 0) == 0.5
    assert _spatial_subscore("", 0) == 0.5
    assert _sensory_subscore("", 0) == 0.5
    assert _firsthand_subscore("", 0) == 0.5
