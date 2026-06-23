"""test_stylometric_signal.py - Unit tests for Signal 2, the stylometric heuristics in stylometric_signal.py.

Verifies the spec contract from planning.md Section 4 (Signal 2):
  analyze_stylometry returns a probability and a raw feature dict, never a binary flag.
  p_ai is always clamped to [0, 1].
  The four Section-4 features (burstiness, type-token ratio, punctuation, complexity) are computed.
  Uniform, low diversity text scores more AI-like than varied, diverse text.
  Short text (below MIN_RELIABLE_WORDS) is blended toward the neutral fence and flagged.
"""

from signals.stylometric_signal import (
    MIN_RELIABLE_WORDS,
    NEUTRAL_SCORE,
    StylometricSignalResult,
    analyze_stylometry,
)

# A long, uniform, low diversity passage: even sentence lengths, repeated
# vocabulary, flat punctuation. The structural fingerprint of AI-leaning text.
UNIFORM_TEXT = (
    "The system processes the data. The system stores the data. The system "
    "returns the data. The system logs the data. The system checks the data. "
    "The system reads the data. The system writes the data. The system sends "
    "the data. The system keeps the data. The system moves the data."
)

# A varied, diverse passage: mixed sentence lengths, wide vocabulary, varied
# punctuation. The structural fingerprint of human-leaning text.
VARIED_TEXT = (
    "Honestly? I had no idea what to expect. The morning crawled by, slow and "
    "gray, until — out of nowhere — a courier showed up with this absurd, "
    "gigantic parcel; I signed for it, baffled. Inside: forty tiny brass bells, "
    "each wrapped in tissue. Who sends bells? My grandmother, apparently, with a "
    "note that simply read 'ring when lonely.' I laughed until I cried."
)


def test_returns_score_and_features_not_a_flag():
    """analyze_stylometry returns a float p_ai and a feature dict, not a boolean."""
    result = analyze_stylometry(VARIED_TEXT)
    assert isinstance(result, StylometricSignalResult)
    assert isinstance(result.p_ai, float)
    assert not isinstance(result.p_ai, bool)
    assert isinstance(result.features, dict)


def test_p_ai_always_in_unit_interval():
    """p_ai is clamped to [0, 1] for both uniform and varied inputs."""
    for text in (UNIFORM_TEXT, VARIED_TEXT):
        result = analyze_stylometry(text)
        assert 0.0 <= result.p_ai <= 1.0


def test_features_include_all_section4_metrics():
    """The feature dict exposes the four Section 4 subscores plus raw metrics."""
    features = analyze_stylometry(VARIED_TEXT).features
    assert set(features["subscores"].keys()) == {
        "burstiness", "type_token_ratio", "punctuation", "complexity",
    }
    # Raw, reviewer-facing metrics are present too.
    for key in ("num_sentences", "num_words", "type_token_ratio", "mean_sentence_length"):
        assert key in features


def test_uniform_text_scores_more_ai_than_varied_text():
    """A uniform, low diversity passage scores higher (more AI-like) than a varied one.

    This is the core directional contract of the structural signal: it must
    separate statistically smooth text from bursty, diverse human text.
    """
    uniform = analyze_stylometry(UNIFORM_TEXT).p_ai
    varied = analyze_stylometry(VARIED_TEXT).p_ai
    assert uniform > varied


def test_short_text_is_flagged_and_pulled_toward_neutral():
    """Text below MIN_RELIABLE_WORDS is flagged too_short and blended toward NEUTRAL_SCORE."""
    short = analyze_stylometry("Short and choppy. Very short indeed.")
    assert short.features["too_short"] is True
    # Blended toward the fence: should sit nearer 0.5 than an extreme.
    assert abs(short.p_ai - NEUTRAL_SCORE) < 0.35


def test_long_text_is_not_flagged_short():
    """A passage above MIN_RELIABLE_WORDS is not flagged as too short."""
    result = analyze_stylometry(UNIFORM_TEXT)
    assert result.features["num_words"] >= MIN_RELIABLE_WORDS
    assert result.features["too_short"] is False


def test_empty_text_returns_neutral_without_raising():
    """Whitespace-only text returns NEUTRAL_SCORE and the short flag, never raising."""
    result = analyze_stylometry("   ")
    assert result.p_ai == NEUTRAL_SCORE
    assert result.features["too_short"] is True


def test_ttr_subscore_not_silently_pinned_on_short_text():
    """The type-token ratio subscore must contribute on short text, not pin to 0.0.

    Regression guard for a real calibration bug found in Milestone 4: an earlier
    TTR band ([0.4, 0.7]) was miscalibrated for the short submissions this
    service sees, where raw TTR sits near 0.85-0.92. Every short input saturated
    the band and the feature silently returned 0.0, contributing nothing. The
    band was recalibrated to the regime where TTR actually varies; this test
    asserts at least one realistic short input yields a nonzero TTR subscore.
    """
    typical_short = (
        "The system processes the data efficiently. It stores results in a "
        "reliable database and returns a structured response to the caller."
    )
    subscore = analyze_stylometry(typical_short).features["subscores"]["type_token_ratio"]
    assert subscore > 0.0


def test_to_dict_matches_contract():
    """to_dict returns exactly the two keys required by the Section 3 stylometric block."""
    d = analyze_stylometry(VARIED_TEXT).to_dict()
    assert set(d.keys()) == {"p_ai", "features"}
