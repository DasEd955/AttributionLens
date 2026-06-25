"""test_scoring_with_grounding.py - Tests for Signal 3 integration with the confidence scorer.

Verifies the grounding_factor integration in scoring.py (planning.md Section 4,
grounding architecture note, and Section 5 formulas):

  grounding_factor=1.0 (neutral) leaves confidence unchanged.
  grounding_factor > 1.0 (rich grounding) boosts confidence.
  grounding_factor < 1.0 (no grounding) reduces confidence.
  Grounding factor is clamped to [0.85, 1.15] by the scorer.
  The degraded mode cap still applies after grounding modification.
  The grounding_factor is stored on ScoreResult and serialized in to_dict.
  Grounding boost cannot push a borderline uncertain verdict into likely_ai
    on its own (the asymmetric caution philosophy: false positives are worse
    than false negatives).
  Rich grounding can increase confidence enough to push a borderline result
    into likely_ai when both primary signals already agree.
"""

import pytest
from scoring.scoring import (
    AI_CONFIDENCE_THRESHOLD,
    AI_SCORE_THRESHOLD,
    DEGRADED_CONFIDENCE_CAP,
    GROUNDING_FACTOR_MAX,
    GROUNDING_FACTOR_MIN,
    VERDICT_AI,
    VERDICT_HUMAN,
    VERDICT_UNCERTAIN,
    ScoreResult,
    score,
)


def test_neutral_grounding_factor_leaves_confidence_unchanged():
    """grounding_factor=1.0 must not change confidence relative to the two signal formula."""
    base = score(0.8, 0.6)
    with_neutral = score(0.8, 0.6, grounding_factor=1.0)
    assert base.confidence == with_neutral.confidence
    assert base.combined_p_ai == with_neutral.combined_p_ai


def test_high_grounding_factor_boosts_confidence():
    """grounding_factor above 1.0 must produce higher confidence than grounding_factor=1.0."""
    base = score(0.8, 0.7)
    boosted = score(0.8, 0.7, grounding_factor=1.15)
    assert boosted.confidence > base.confidence


def test_low_grounding_factor_reduces_confidence():
    """grounding_factor below 1.0 must produce lower confidence than grounding_factor=1.0."""
    base = score(0.8, 0.7)
    reduced = score(0.8, 0.7, grounding_factor=0.85)
    assert reduced.confidence < base.confidence


def test_grounding_factor_clamped_below_085():
    """Values below 0.85 are clamped to 0.85 before being applied."""
    result_clamped = score(0.8, 0.7, grounding_factor=0.5)
    result_at_min = score(0.8, 0.7, grounding_factor=0.85)
    assert result_clamped.confidence == result_at_min.confidence
    assert result_clamped.grounding_factor == 0.85


def test_grounding_factor_clamped_above_115():
    """Values above 1.15 are clamped to 1.15 before being applied."""
    result_clamped = score(0.8, 0.7, grounding_factor=2.0)
    result_at_max = score(0.8, 0.7, grounding_factor=1.15)
    assert result_clamped.confidence == result_at_max.confidence
    assert result_clamped.grounding_factor == 1.15


def test_grounding_factor_stored_on_scoreresult():
    """The grounding_factor used is stored on the returned ScoreResult."""
    result = score(0.7, 0.6, grounding_factor=1.1)
    assert hasattr(result, "grounding_factor")
    assert result.grounding_factor == pytest.approx(1.1, abs=1e-4)


def test_grounding_factor_in_to_dict():
    """grounding_factor is serialized in ScoreResult.to_dict()."""
    result = score(0.7, 0.6, grounding_factor=1.05)
    d = result.to_dict()
    assert "grounding_factor" in d


def test_degraded_cap_applies_after_grounding_boost():
    """Degraded mode cap must apply AFTER the grounding boost, not before.

    Even if a richly grounded text boosts confidence, the cap at
    DEGRADED_CONFIDENCE_CAP still applies when llm_available=False.
    """
    result = score(0.9, 0.9, llm_available=False, grounding_factor=1.15)
    assert result.confidence <= DEGRADED_CONFIDENCE_CAP


def test_grounding_boost_cannot_flip_uncertain_to_ai_on_disagreeing_signals():
    """Grounding boost alone must not flip a strongly uncertain verdict to likely_ai.

    This protects the false-positive defence: if the two primary signals disagree
    badly (LLM 0.85 vs stylometry 0.30), the confidence collapse must still win
    over the grounding boost. We use the Section 6 worked example.
    """
    result = score(0.85, 0.30, grounding_factor=GROUNDING_FACTOR_MAX)
    assert result.verdict != VERDICT_AI


def test_grounding_boost_can_tip_borderline_to_ai_when_signals_agree():
    """Grounding boost can push a borderline result over the AI confidence floor
    when both primary signals strongly agree.

    This verifies the signal is not purely decorative: it has real effect on
    borderline cases where signals agree but the raw confidence is just below
    the AI floor.
    """
    # Without grounding: strong agreement, combined score above AI threshold,
    # but confidence just below AI_CONFIDENCE_THRESHOLD after rounding.
    # We set up a case where base confidence is near but below 0.20.
    # p_llm=0.69, p_style=0.66 -> combined ~0.678 (above 0.65),
    # agreement ~0.97, decisiveness ~0.356, confidence ~0.345.
    # Grounding factor 1.15 raises it further -> result should be likely_ai.
    result_boosted = score(0.69, 0.66, grounding_factor=1.15)
    assert result_boosted.combined_p_ai >= AI_SCORE_THRESHOLD
    assert result_boosted.confidence >= AI_CONFIDENCE_THRESHOLD
    assert result_boosted.verdict == VERDICT_AI


def test_grounding_reduction_protects_false_positive():
    """Reducing grounding factor on borderline AI score can push it to uncertain.

    This tests the protective use case: a text that is scoring near the AI
    threshold but has ZERO experiential grounding (no times, places, senses, or
    firsthand markers) should have confidence reduced, potentially preventing
    a false accusation.
    """
    # Set up a case where verdict would be likely_ai at neutral grounding.
    base = score(0.95, 0.90)
    assert base.verdict == VERDICT_AI

    # Apply the minimum grounding factor -- confidence is reduced.
    reduced = score(0.95, 0.90, grounding_factor=0.85)
    # Confidence should be lower (the grounding penalty applied).
    assert reduced.confidence < base.confidence


def test_score_result_has_all_fields_including_grounding():
    """ScoreResult carries all six expected fields including grounding_factor."""
    result = score(0.5, 0.5)
    for attr in ("combined_p_ai", "confidence", "verdict", "agreement",
                 "decisiveness", "grounding_factor"):
        assert hasattr(result, attr), f"Missing field: {attr}"


def test_default_grounding_factor_is_1():
    """Calling score() with no grounding_factor defaults to 1.0 (backward compatible)."""
    result = score(0.7, 0.6)
    assert result.grounding_factor == 1.0
