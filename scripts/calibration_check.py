"""calibration_check.py - Manual calibration harness for the two signal confidence scorer.

This is a developer tool, not part of the request path. It runs the four
deliberately chosen Milestone 4 fixtures (clearly AI, clearly human, formal
human, lightly edited AI) through Signal 2 and the confidence scorer and prints
each fixture's individual signal scores alongside the combined result.

Per the Milestone 4 brief: when a fixture produces a score that does not match
intuition, printing BOTH signal scores separately (plus the per-feature
stylometric sub-scores) is how you find which signal is misbehaving. The LLM
score is supplied as a fixed, representative reading here so the harness is
deterministic and never calls Groq; only the structural signal is computed live.

Run from the repo root:

    python scripts/calibration_check.py
"""

from __future__ import annotations
import sys
from pathlib import Path

# Make the repo root importable so this script runs from anywhere, mirroring the
# path bootstrap in tests/conftest.py.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scoring import score
from signals.stylometric_signal import analyze_stylometry

# The four calibration fixtures from the Milestone 4 brief, paired with a fixed,
# representative LLM reading for each (high for AI, low for casual human, and the
# register-biased high read on formal human prose that the scorer must defuse).
FIXTURES = {
    "CLEARLY_AI": (
        "Artificial intelligence represents a transformative paradigm shift in modern "
        "society. It is important to note that while the benefits of AI are numerous, it "
        "is equally essential to consider the ethical implications. Furthermore, "
        "stakeholders across various sectors must collaborate to ensure responsible "
        "deployment.",
        0.90,
    ),
    "CLEARLY_HUMAN": (
        "ok so i finally tried that new ramen place downtown and honestly? underwhelming. "
        "the broth was fine but they put WAY too much sodium in it and i was thirsty for "
        "like three hours after. my friend got the spicy version and said it was better. "
        "probably won't go back unless someone drags me there",
        0.10,
    ),
    "FORMAL_HUMAN": (
        "The relationship between monetary policy and asset price inflation has been "
        "extensively studied in the literature. Central banks face a fundamental tension "
        "between their mandate for price stability and the unintended consequences of "
        "prolonged low interest rates on equity and real estate valuations.",
        0.80,
    ),
    "LIGHTLY_EDITED_AI": (
        "I've been thinking a lot about remote work lately. There are genuine tradeoffs, "
        "flexibility and no commute on one side, isolation and blurred work-life "
        "boundaries on the other. Studies show productivity varies widely by individual "
        "and role type.",
        0.60,
    ),
}


def run() -> None:
    """Score every calibration fixture and print a per-fixture breakdown.

    For each fixture, computes the live stylometric probability, combines it
    with the fixture's fixed LLM reading via the confidence scorer, and prints
    the individual signal scores, the combined score, the confidence, the
    verdict, and the stylometric subscores. Pure output: returns nothing and
    writes only to stdout.
    """
    header = f"{'fixture':<18}{'p_llm':>7}{'p_style':>9}{'combined':>10}{'conf':>7}{'verdict':>15}"
    print(header)
    print("-" * len(header))
    for name, (text, p_llm) in FIXTURES.items():
        style = analyze_stylometry(text)
        result = score(p_llm, style.p_ai)
        print(
            f"{name:<18}{p_llm:>7.2f}{style.p_ai:>9.3f}"
            f"{result.combined_p_ai:>10.3f}{result.confidence:>7.3f}{result.verdict:>15}"
        )
        print(f"    stylometric sub-scores: {style.features['subscores']}")


if __name__ == "__main__":
    run()
