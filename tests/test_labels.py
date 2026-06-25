"""test_labels.py - Unit tests for labels.py (transparency label generator).

Verifies the Section 7 design: that the verdict and confidence select the
correct one of the three variants, that the AI variant is gated on confidence,
that the reader-facing text matches the spec verbatim, and that no raw
probability leaks into the text.

Coverage:
  Each of the three variants is reachable and carries its exact Section 7 text.
  The AI variant requires confidence at/above the floor; below it falls back to uncertain.
  A likely_human verdict always maps to the human variant.
  No variant text contains a raw numeric probability.
"""

from labels import (
    VARIANT_AI,
    VARIANT_HUMAN,
    VARIANT_UNCERTAIN,
    generate_label,
)
from scoring import VERDICT_AI, VERDICT_HUMAN, VERDICT_UNCERTAIN

# The exact Section 7 text, verbatim, used to confirm the generator does not
# drift from the spec wording.
EXPECTED_AI_TEXT = (
    "AI generated content likely\n"
    "Our analysis suggests this text was probably created with significant "
    "help from an AI tool. This is an automated estimate, not a certainty. "
    "The creator can contest it."
)
EXPECTED_HUMAN_TEXT = (
    "Likely human written\n"
    "Our analysis found no strong signs of AI generation in this text. This "
    "is an automated estimate and is not a guarantee."
)
EXPECTED_UNCERTAIN_TEXT = (
    "Attribution uncertain\n"
    "We could not confidently determine whether this text was written by a "
    "person or generated with AI. Please treat this as incomplete context "
    "rather than a verdict about the creator."
)


def test_high_confidence_ai_variant_text_matches_spec():
    """A likely_ai verdict with confidence at/above the floor yields the exact Variant A text."""
    label = generate_label(VERDICT_AI, 0.80)
    assert label.variant == VARIANT_AI
    assert label.text == EXPECTED_AI_TEXT


def test_human_variant_text_matches_spec():
    """A likely_human verdict yields the exact Variant B text."""
    label = generate_label(VERDICT_HUMAN, 0.90)
    assert label.variant == VARIANT_HUMAN
    assert label.text == EXPECTED_HUMAN_TEXT


def test_uncertain_variant_text_matches_spec():
    """An uncertain verdict yields the exact Variant C text."""
    label = generate_label(VERDICT_UNCERTAIN, 0.30)
    assert label.variant == VARIANT_UNCERTAIN
    assert label.text == EXPECTED_UNCERTAIN_TEXT


def test_low_confidence_ai_falls_back_to_uncertain():
    """A likely_ai verdict below the confidence floor never surfaces the accusatory variant.

    Section 7 gates Variant A on confidence >= AI_VARIANT_CONFIDENCE_FLOOR (0.20).
    A would-be AI verdict arriving with confidence below 0.20 falls back to the
    uncertain variant rather than accusing the creator.
    """
    label = generate_label(VERDICT_AI, 0.10)
    assert label.variant == VARIANT_UNCERTAIN
    assert label.text == EXPECTED_UNCERTAIN_TEXT


def test_all_three_variants_are_reachable():
    """Driving the three verdict/confidence regimes produces three distinct variants."""
    variants = {
        generate_label(VERDICT_AI, 0.80).variant,
        generate_label(VERDICT_HUMAN, 0.90).variant,
        generate_label(VERDICT_UNCERTAIN, 0.30).variant,
    }
    assert variants == {VARIANT_AI, VARIANT_HUMAN, VARIANT_UNCERTAIN}


def test_label_text_never_contains_a_raw_probability():
    """No variant leaks a raw numeric probability to the reader (Section 7)."""
    for verdict, confidence in (
        (VERDICT_AI, 0.80),
        (VERDICT_HUMAN, 0.90),
        (VERDICT_UNCERTAIN, 0.30),
    ):
        text = generate_label(verdict, confidence).text
        assert not any(ch.isdigit() for ch in text)
