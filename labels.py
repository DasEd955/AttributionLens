"""labels.py - Transparency label generator; maps a verdict and confidence to reader-facing text.

Per planning.md Section 7 (transparency label design):

  * Readers never see a raw probability. A "0.62" means nothing to a
    non-technical user and invites false precision. The three named states
    carry the meaning instead.
  * There are exactly three label variants, and their text is FIXED by the spec:
      - high_confidence_ai      verdict likely_ai with confidence at/above 0.65
      - high_confidence_human   verdict likely_human
      - uncertain               verdict uncertain, or any low confidence result
  * Every variant hedges ("likely", "probably") and names that the estimate is
    automated and contestable. The system never speaks with more certainty than
    it has, and the uncertain variant never reads as an accusation.

The mapping mirrors the verdict bands the scorer (scoring.py) produces, so the
label is a faithful surface for the verdict already recorded in the audit log
(Section 11) rather than a second, independent judgment.
"""

from __future__ import annotations
from dataclasses import dataclass
from scoring import VERDICT_AI, VERDICT_HUMAN

# Label variant identifiers (also the audit-log ``label_variant`` values,
# Section 11). These are the machine-readable names of the three Section 7 variants.
VARIANT_AI = "high_confidence_ai"
VARIANT_HUMAN = "high_confidence_human"
VARIANT_UNCERTAIN = "uncertain"

# Confidence floor that the AI variant requires (planning.md Section 7, Variant A).
# Mirrors scoring.AI_CONFIDENCE_THRESHOLD; an AI verdict that somehow arrives with
# lower confidence falls back to the uncertain variant rather than accusing.
AI_VARIANT_CONFIDENCE_FLOOR = 0.65

# The exact reader-facing text of each variant, verbatim from planning.md
# Section 7. Each tuple is (title, body). The title is the bold headline a reader
# sees; the body is the hedged explanation beneath it.
_VARIANT_TEXT = {
    VARIANT_AI: (
        "AI generated content likely",
        "Our analysis suggests this text was probably created with significant "
        "help from an AI tool. This is an automated estimate, not a certainty. "
        "The creator can contest it.",
    ),
    VARIANT_HUMAN: (
        "Likely human written",
        "Our analysis found no strong signs of AI generation in this text. This "
        "is an automated estimate and is not a guarantee.",
    ),
    VARIANT_UNCERTAIN: (
        "Attribution uncertain",
        "We could not confidently determine whether this text was written by a "
        "person or generated with AI. Please treat this as incomplete context "
        "rather than a verdict about the creator.",
    ),
}


@dataclass
class TransparencyLabel:
    """The reader-facing transparency label for a decision.

    Matches the ``label`` block of the /submit response contract (planning.md
    Section 3): a machine-readable ``variant`` identifier and the exact ``text``
    a non-technical reader sees. ``text`` is the title and body of Section 7
    joined into a single readable string.
    """

    variant: str                     # One of VARIANT_AI / VARIANT_HUMAN / VARIANT_UNCERTAIN
    text: str                        # The exact reader-facing label text (Section 7)

    def to_dict(self) -> dict:
        """Serialize the label to the ``label`` response contract shape.

        Returns:
            dict: Keys ``variant`` (str) and ``text`` (str).
        """
        return {
            "variant": self.variant,
            "text": self.text,
        }


def _select_variant(verdict: str, confidence: float) -> str:
    """Choose the label variant for a verdict and confidence value.

    Implements the Section 7 mapping. The AI variant is gated on BOTH a
    ``likely_ai`` verdict and a confidence at or above AI_VARIANT_CONFIDENCE_FLOOR,
    so a low confidence result can never surface the accusatory variant. A
    ``likely_human`` verdict maps to the human variant; everything else (the
    uncertain verdict, or any result that fails the AI gate) maps to uncertain.

    Args:
        verdict (str): The scorer verdict, one of scoring.VERDICT_* values.
        confidence (float): The confidence value in [0, 1].

    Returns:
        str: VARIANT_AI, VARIANT_HUMAN, or VARIANT_UNCERTAIN.
    """
    if verdict == VERDICT_AI and confidence >= AI_VARIANT_CONFIDENCE_FLOOR:
        return VARIANT_AI
    if verdict == VERDICT_HUMAN:
        return VARIANT_HUMAN
    return VARIANT_UNCERTAIN


def generate_label(verdict: str, confidence: float) -> TransparencyLabel:
    """Map a verdict and confidence to the reader-facing transparency label.

    Selects one of the three Section 7 variants and returns the exact, fixed
    label text for it. The text is never assembled from the score; it is chosen
    from the spec's verbatim variants so the reader sees the wording the design
    committed to, with no raw probability leaking through.

    Args:
        verdict (str): The scorer verdict, one of scoring.VERDICT_* values.
        confidence (float): The confidence value in [0, 1], used to gate the
            AI variant per Section 7.

    Returns:
        TransparencyLabel: The selected ``variant`` identifier and the exact
            reader-facing ``text`` (title and body) for it.
    """
    variant = _select_variant(verdict, confidence)
    title, body = _VARIANT_TEXT[variant]
    return TransparencyLabel(variant=variant, text=f"{title}\n{body}")
