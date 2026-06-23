"""scoring.py - Confidence scorer; combines the two signals into one calibrated verdict.

Per planning.md Section 5 (confidence scoring) and Section 9 (graceful degradation).

The scorer separates two distinct quantities that the rest of the system must
never conflate:

  * ``combined_p_ai`` - the best estimate that the text is AI generated, [0, 1].
  * ``confidence``    - how much to TRUST that estimate, [0, 1]. This is what
                        drives the user-facing label.

Exact formulas (Section 5):

    combined_p_ai = w_llm * p_ai_llm + w_style * p_ai_style   (w_llm=0.6, w_style=0.4)
    agreement     = 1 - abs(p_ai_llm - p_ai_style)            in [0, 1]
    decisiveness  = 2 * abs(combined_p_ai - 0.5)              in [0, 1]
    confidence    = decisiveness * agreement

Confidence is high only when the estimate is far from the 0.5 fence AND the two
independent signals agree. Disagreement actively collapses confidence and pushes
the verdict to "uncertain". That is the mechanism (Section 6) that protects a
careful human writer from a false AI accusation.

Verdict bands are asymmetric ON PURPOSE (Section 5): declaring "AI" is hard,
defaulting to "human"/"uncertain" is easy, because a false AI accusation is the
worst error this system can make.

Degraded mode (Section 9): when the LLM signal did not run, confidence is capped
at DEGRADED_CONFIDENCE_CAP so a single, gameable structural signal can never
present as a high-confidence verdict.
"""

from __future__ import annotations
from dataclasses import dataclass
from util import clamp01

# Signal weights (Section 5). The LLM is weighted higher because it captures
# more of what distinguishes the classes, but stylometry keeps real weight so it
# can pull the verdict back when the LLM overly flags formal prose.
W_LLM = 0.6
W_STYLE = 0.4

# Verdict band thresholds (Section 5 table). Asymmetric by design.
AI_SCORE_THRESHOLD = 0.75        # combined_p_ai must reach this for likely_ai
AI_CONFIDENCE_THRESHOLD = 0.65   # AND confidence must reach this for likely_ai
HUMAN_SCORE_THRESHOLD = 0.40     # combined_p_ai at or below this is likely_human

# Graceful degradation ceiling (Section 9). With only the stylometric signal
# available, no result may report higher confidence than this.
DEGRADED_CONFIDENCE_CAP = 0.5

# Verdict label strings (also the audit-log ``verdict`` values, Section 11).
VERDICT_AI = "likely_ai"
VERDICT_HUMAN = "likely_human"
VERDICT_UNCERTAIN = "uncertain"


@dataclass
class ScoreResult:
    """The combined output of the confidence scorer.

    Carries the three quantities the response contract (planning.md Section 3)
    and the audit log (Section 11) need: the combined probability, the
    confidence in it, and the resulting verdict band. ``agreement`` and
    ``decisiveness`` are retained for transparency so a reviewer can see why the
    confidence landed where it did.
    """

    combined_p_ai: float       # Weighted probability the text is AI generated, [0, 1]
    confidence: float          # Trust in the estimate, [0, 1]
    verdict: str               # One of VERDICT_AI / VERDICT_HUMAN / VERDICT_UNCERTAIN
    agreement: float           # 1 - |p_ai_llm - p_ai_style|, [0, 1]
    decisiveness: float        # 2 * |combined_p_ai - 0.5|, [0, 1]

    def to_dict(self) -> dict:
        """Serialize the scorer output for the response and the audit log.

        Returns:
            dict: Keys ``combined_p_ai``, ``confidence``, ``verdict``,
                  ``agreement``, ``decisiveness``.
        """
        return {
            "combined_p_ai": self.combined_p_ai,
            "confidence": self.confidence,
            "verdict": self.verdict,
            "agreement": self.agreement,
            "decisiveness": self.decisiveness,
        }


def _decide_verdict(combined_p_ai: float, confidence: float) -> str:
    """Map a combined score and confidence to one of the three verdict bands.

    Implements the asymmetric Section 5 table exactly: ``likely_ai`` requires
    BOTH a high combined score and high confidence; ``likely_human`` requires
    only a low combined score (the human zone is intentionally wide); everything
    else falls into the ``uncertain`` buffer band.

    The order of checks matters: the strict AI gate is evaluated first, then the
    wide human zone, then uncertain as the catch-all.

    Args:
        combined_p_ai (float): The weighted combined probability in [0, 1].
        confidence (float): The confidence value in [0, 1].

    Returns:
        str: VERDICT_AI, VERDICT_HUMAN, or VERDICT_UNCERTAIN.
    """
    if combined_p_ai >= AI_SCORE_THRESHOLD and confidence >= AI_CONFIDENCE_THRESHOLD:
        return VERDICT_AI
    if combined_p_ai <= HUMAN_SCORE_THRESHOLD:
        return VERDICT_HUMAN
    return VERDICT_UNCERTAIN


def score(
    p_ai_llm: float,
    p_ai_style: float,
    *,
    llm_available: bool = True,
) -> ScoreResult:
    """Combine the two signal probabilities into a calibrated verdict.

    Applies the Section 5 formulas: a weighted combined probability, an
    agreement term (how close the two signals are), a decisiveness term (how far
    the combined score sits from the 0.5 fence), and ``confidence`` as their
    product. The verdict band is then derived from the combined score and
    confidence via the asymmetric Section 5 table.

    When ``llm_available`` is False the system is running on the stylometric
    signal alone (Section 9). In that degraded mode the reported confidence is
    capped at DEGRADED_CONFIDENCE_CAP so a single gameable signal can never
    surface as a high confidence verdict; the verdict is recomputed against the
    capped confidence, which in practice routes most single signal results to
    ``uncertain``.

    Args:
        p_ai_llm (float): Signal 1 probability in [0, 1]. Ignored for the
            agreement term's protective effect when ``llm_available`` is False,
            but still clamped and combined so the structural reading flows through.
        p_ai_style (float): Signal 2 probability in [0, 1].
        llm_available (bool, optional): Whether the LLM signal ran. When False,
            confidence is capped (Section 9). Defaults to True.

    Returns:
        ScoreResult: The combined probability, confidence, verdict, and the
            agreement/decisiveness terms that produced the confidence.
    """
    p_ai_llm = clamp01(p_ai_llm)
    p_ai_style = clamp01(p_ai_style)

    combined_p_ai = clamp01(W_LLM * p_ai_llm + W_STYLE * p_ai_style)
    agreement = 1.0 - abs(p_ai_llm - p_ai_style)
    decisiveness = 2.0 * abs(combined_p_ai - 0.5)
    confidence = clamp01(decisiveness * agreement)

    # Section 9: with only the structural signal, honesty forbids high confidence.
    if not llm_available:
        confidence = min(confidence, DEGRADED_CONFIDENCE_CAP)

    verdict = _decide_verdict(combined_p_ai, confidence)

    return ScoreResult(
        combined_p_ai=round(combined_p_ai, 4),
        confidence=round(confidence, 4),
        verdict=verdict,
        agreement=round(agreement, 4),
        decisiveness=round(decisiveness, 4),
    )
