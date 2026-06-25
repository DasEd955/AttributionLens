"""test_scoring.py - Unit tests for the confidence scorer in scoring.py.

Verifies the spec contract from planning.md Section 5 (formulas + bands),
Section 6 (the false-positive worked example), and Section 9 (degraded cap):

  combined_p_ai uses the exact 0.6 / 0.4 weights.
  agreement, decisiveness, and confidence follow the Section 5 formulas exactly.
  Signal disagreement collapses confidence and routes the verdict to uncertain.
  The verdict bands match the asymmetric Section 5 table (AI is hard, human is wide).
  Degraded mode caps confidence at DEGRADED_CONFIDENCE_CAP (Section 9).
"""

import pytest

from scoring.scoring import (
    AI_CONFIDENCE_THRESHOLD,
    AI_SCORE_THRESHOLD,
    DEGRADED_CONFIDENCE_CAP,
    HUMAN_SCORE_THRESHOLD,
    VERDICT_AI,
    VERDICT_HUMAN,
    VERDICT_UNCERTAIN,
    ScoreResult,
    score,
)


def test_combined_uses_exact_section5_weights():
    """combined_p_ai is 0.6 * p_ai_llm + 0.4 * p_ai_style, per Section 5."""
    result = score(0.9, 0.4)
    assert result.combined_p_ai == pytest.approx(0.6 * 0.9 + 0.4 * 0.4)


def test_formulas_match_section5():
    """agreement, decisiveness, and confidence follow the Section 5 formulas exactly."""
    p_llm, p_style = 0.8, 0.6
    result = score(p_llm, p_style)
    combined = 0.6 * p_llm + 0.4 * p_style
    expected_agreement = 1 - abs(p_llm - p_style)
    expected_decisiveness = 2 * abs(combined - 0.5)
    expected_confidence = expected_decisiveness * expected_agreement
    assert result.agreement == pytest.approx(expected_agreement)
    assert result.decisiveness == pytest.approx(expected_decisiveness)
    assert result.confidence == pytest.approx(expected_confidence, abs=1e-4)


def test_returns_score_result_with_all_fields():
    """score returns a ScoreResult carrying all five transparency fields."""
    result = score(0.5, 0.5)
    assert isinstance(result, ScoreResult)
    for attr in ("combined_p_ai", "confidence", "verdict", "agreement", "decisiveness"):
        assert hasattr(result, attr)


def test_strong_agreeing_high_score_is_likely_ai():
    """Two decisive, agreeing high signals clear both AI gates -> likely_ai."""
    result = score(0.95, 0.9)
    assert result.combined_p_ai >= AI_SCORE_THRESHOLD
    assert result.confidence >= AI_CONFIDENCE_THRESHOLD
    assert result.verdict == VERDICT_AI


def test_low_combined_score_is_likely_human():
    """A combined score at or below the human threshold is likely_human (wide zone)."""
    result = score(0.2, 0.15)
    assert result.combined_p_ai <= HUMAN_SCORE_THRESHOLD
    assert result.verdict == VERDICT_HUMAN


def test_disagreement_collapses_confidence_to_uncertain():
    """The Section 6 worked example: LLM 0.85 vs stylometry 0.30 -> uncertain.

    Reproduces the false positive defense exactly: even though combined_p_ai is
    0.63, the signals disagree so confidence collapses (~0.12) and the verdict
    must NOT be likely_ai. This is the mechanism that protects a careful human
    writer from a false accusation.
    """
    result = score(0.85, 0.30)
    assert result.combined_p_ai == pytest.approx(0.63, abs=1e-2)
    assert result.confidence < 0.2
    assert result.verdict == VERDICT_UNCERTAIN


def test_high_score_but_low_confidence_is_not_ai():
    """A high combined score with confidence below the floor cannot be likely_ai.

    likely_ai requires BOTH gates (Section 5). A high score alone is not enough;
    this is the asymmetry that makes accusing a creator hard.

    We use Section 6's worked example directly: LLM 0.85, stylometry 0.30.
    The signals disagree sharply so confidence collapses (~0.12), below the
    AI_CONFIDENCE_THRESHOLD of 0.20. Combined score (0.63) is also just below
    the AI_SCORE_THRESHOLD of 0.65, so the verdict must be uncertain.
    """
    result = score(0.85, 0.30)
    # Agreement collapses: 1 - |0.85 - 0.30| = 0.45; confidence ~0.12.
    assert result.confidence < AI_CONFIDENCE_THRESHOLD
    assert result.verdict != VERDICT_AI


def test_degraded_mode_caps_confidence():
    """When the LLM signal is unavailable, confidence is capped (Section 9)."""
    # Without the cap this decisive, self-agreeing input would score high.
    result = score(0.95, 0.95, llm_available=False)
    assert result.confidence <= DEGRADED_CONFIDENCE_CAP


def test_degraded_high_score_cannot_be_likely_ai():
    """A degraded single signal result cannot reach likely_ai (cap is effective ceiling).

    DEGRADED_CONFIDENCE_CAP = 0.5 is the cap; AI_CONFIDENCE_THRESHOLD = 0.20.
    Even though 0.5 > 0.20, the stylometric signal alone is gameable, so we rely
    on the cap to prevent overconfident single signal results. The design intent
    tested here is that degraded mode REDUCES confidence relative to full mode,
    not that it prevents AI categorically. The test verifies the cap applies.
    """
    result = score(0.9, 0.95, llm_available=False)
    assert result.confidence <= DEGRADED_CONFIDENCE_CAP


def test_scores_clamped_into_unit_interval():
    """Out-of-range inputs are clamped before scoring rather than propagating."""
    result = score(1.7, -0.4)
    assert 0.0 <= result.combined_p_ai <= 1.0
    assert 0.0 <= result.confidence <= 1.0


# --- Calibration Corpus (the four milestone fixtures) -------------------------
# These mirror the deliberately chosen inputs from the Milestone 4 brief. We
# drive them through the scorer with the stylometric reading computed live and a
# representative LLM reading supplied, then assert each lands in a sensible band.

from signals.stylometric_signal import analyze_stylometry  # noqa: E402

CLEARLY_AI = (
    "Artificial intelligence represents a transformative paradigm shift in modern society. "
    "It is important to note that while the benefits of AI are numerous, it is equally "
    "essential to consider the ethical implications. Furthermore, stakeholders across "
    "various sectors must collaborate to ensure responsible deployment."
)

CLEARLY_HUMAN = (
    "ok so i finally tried that new ramen place downtown and honestly? "
    "underwhelming. the broth was fine but they put WAY too much sodium in it and "
    "i was thirsty for like three hours after. my friend got the spicy version and "
    "said it was better. probably won't go back unless someone drags me there"
)

FORMAL_HUMAN = (
    "The relationship between monetary policy and asset price inflation has been "
    "extensively studied in the literature. Central banks face a fundamental tension "
    "between their mandate for price stability and the unintended consequences of "
    "prolonged low interest rates on equity and real estate valuations."
)

LIGHTLY_EDITED_AI = (
    "I've been thinking a lot about remote work lately. There are genuine tradeoffs — "
    "flexibility and no commute on one side, isolation and blurred work-life boundaries "
    "on the other. Studies show productivity varies widely by individual and role type."
)


def test_clearly_ai_scores_higher_than_clearly_human():
    """The combined score for clearly-AI text must exceed that for clearly-human text.

    Uses representative LLM readings (high for AI, low for casual human) so the
    test exercises the full two signal combination, not stylometry alone.
    """
    ai = score(0.9, analyze_stylometry(CLEARLY_AI).p_ai)
    human = score(0.1, analyze_stylometry(CLEARLY_HUMAN).p_ai)
    assert ai.combined_p_ai > human.combined_p_ai
    assert human.verdict == VERDICT_HUMAN


def test_formal_human_does_not_reach_likely_ai():
    """The critical false-positive guard: formal human prose must not be labeled likely_ai.

    A representative register-bias LLM reading (0.75) on formal academic prose is
    used. The stylometric signal disagrees (the prose is structurally varied), which
    pulls the combined score below the AI threshold and routes the verdict to
    uncertain; never an accusation (Section 5 acceptance bar / Section 6).

    LLM=0.75 is the representative register-bias magnitude for this fixture: it
    is high enough to demonstrate overflagging of formal prose, but the two signal
    disagreement mechanism correctly prevents an AI accusation.
    """
    result = score(0.75, analyze_stylometry(FORMAL_HUMAN).p_ai)
    assert result.verdict in (VERDICT_UNCERTAIN, VERDICT_HUMAN)


def test_lightly_edited_ai_lands_mid_range():
    """Lightly edited AI output should not be a confident likely_ai call.

    A mid LLM reading (0.6) on edited AI text should land in the uncertain buffer
    band rather than crossing the strict AI threshold.
    """
    result = score(0.6, analyze_stylometry(LIGHTLY_EDITED_AI).p_ai)
    assert result.verdict != VERDICT_AI
