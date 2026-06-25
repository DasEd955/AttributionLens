"""gen_architecture_diagram.py - One-time generator for the AttributionLens architecture diagram.

Produces one PNG diagram in the util/ directory that documents the full
repository architecture:

  Repo-Architecture.png — the complete five tier pipeline:
    HTTP client -> Flask app (app.py) with rate limiting and input validation
    -> three orthogonal detection signals running in parallel (LLM via Groq,
    Stylometric pure-Python, Grounding pure-Python) -> confidence scorer
    (scoring.py) combining signals into combined_p_ai + confidence + verdict
    -> transparency label generator (labels.py) -> SQLite audit log
    (audit_log.py) persisting all scores and appeal state. A separate panel
    shows the React + Vite + Recharts analytics dashboard consuming three
    read-only dashboard endpoints. An appeals sidebar illustrates the
    appeal review review path.

The diagram is rendered with Pillow using a dark background palette. Layout
helpers make_box() and make_varrow()/make_harrow() are returned as closures
capturing the ImageDraw instance so the drawing surface is self contained.
font() and center() are module-level utilities.

The Windows TrueType fonts (Arial, Arial Bold, Consolas) are hardcoded paths
under C:/Windows/Fonts/; running this on a non-Windows machine requires
substituting compatible font files.

Run with:
    python scripts/gen_architecture_diagram.py

from the repo root. The output PNG is written to util/Repo-Architecture.png.
Requires: Pillow (pip install pillow).
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Make the repo root importable when run as a script.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Shared Palette ────────────────────────────────────────────────────────────

BG     = (18,  18,  22)
INK    = (235, 236, 238)
SUB    = (170, 175, 185)
ARROW  = (130, 136, 148)
DIMMED = (110, 115, 125)

# Tier accent colours
C_CLIENT  = (38,  90,  60)   # green   – HTTP client / browser
C_FLASK   = (33,  86, 166)   # blue    – Flask app.py
C_LLM     = (100, 60, 150)   # purple  – Signal 1: LLM
C_STYLE   = (50,  95, 160)   # steel   – Signal 2: Stylometric
C_GROUND  = (48, 118,  90)   # teal    – Signal 3: Grounding
C_SCORE   = (140, 100,  30)  # gold    – Scoring layer
C_LABEL   = (155,  60,  40)  # red     – Transparency labels
C_AUDIT   = (55,   55,  70)  # slate   – Audit log / SQLite
C_DASH    = (35,  100, 120)  # cyan    – React dashboard
C_APPEAL  = (90,   60,  30)  # amber   – Appeals / human review
C_SIDE    = (38,   38,  48)  # dark    – sidebar annotation panels


# ── Font Helpers ──────────────────────────────────────────────────────────────

F  = "C:/Windows/Fonts/arial.ttf"
FB = "C:/Windows/Fonts/arialbd.ttf"
FM = "C:/Windows/Fonts/consola.ttf"


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    """Load a TrueType font from the given path at the given point size.

    Args:
        path (str): Absolute path to the .ttf font file.
        size (int): Point size to load.

    Returns:
        ImageFont.FreeTypeFont: The loaded font object.
    """
    return ImageFont.truetype(path, size)


f_title  = font(FB, 34)
f_sub    = font(F,  17)
f_stage  = font(FB, 22)
f_detail = font(F,  15)
f_small  = font(F,  13)
f_smallb = font(FB, 14)
f_label  = font(FB, 13)
f_tiny   = font(F,  12)


def center(
    draw: ImageDraw.ImageDraw,
    cx: float,
    y: float,
    text: str,
    fnt: ImageFont.FreeTypeFont,
    fill: tuple,
) -> None:
    """Draw text horizontally centered on the given x coordinate.

    Args:
        draw (ImageDraw.ImageDraw): The draw context to render into.
        cx (float): The x coordinate of the desired center.
        y (float): The top y coordinate for the text.
        text (str): The string to render.
        fnt (ImageFont.FreeTypeFont): The font to use.
        fill (tuple[int, int, int]): RGB fill color.
    """
    w = draw.textlength(text, font=fnt)
    draw.text((cx - w / 2, y), text, font=fnt, fill=fill)


# ── Layout Closures ───────────────────────────────────────────────────────────

def make_box(d: ImageDraw.ImageDraw):
    """Return a box drawing closure bound to the given ImageDraw context.

    The returned box() function draws a rounded rectangle with an optional title,
    subtitle, and list of detail lines stacked vertically from the top of the box.

    Args:
        d (ImageDraw.ImageDraw): The draw context to bind the closure to.

    Returns:
        Callable: box(x, y, w, h, fill, title, lines, *, title_font, line_font,
            title_fill, line_fill, radius, align_center, sub) that draws one
            diagram block and returns its bounding rect as (x, y, x+w, y+h).
    """
    def box(
        x, y, w, h, fill,
        title=None, lines=None, *,
        title_font=f_stage,
        line_font=f_detail,
        title_fill=INK,
        line_fill=(225, 228, 234),
        radius=13,
        align_center=True,
        sub=None,
    ):
        d.rounded_rectangle([x, y, x + w, y + h], radius=radius, fill=fill)
        cx = x + w / 2
        cy = y + 13
        if title:
            if align_center:
                center(d, cx, cy, title, title_font, title_fill)
            else:
                d.text((x + 14, cy), title, font=title_font, fill=title_fill)
            cy += title_font.size + 7
        if sub:
            if align_center:
                center(d, cx, cy, sub, f_small, SUB)
            else:
                d.text((x + 14, cy), sub, font=f_small, fill=SUB)
            cy += f_small.size + 7
        for ln in (lines or []):
            if align_center:
                center(d, cx, cy, ln, line_font, line_fill)
            else:
                d.text((x + 14, cy), ln, font=line_font, fill=line_fill)
            cy += line_font.size + 5
        return (x, y, x + w, y + h)
    return box


def make_varrow(d: ImageDraw.ImageDraw):
    """Return a vertical downward arrow closure bound to the given ImageDraw context.

    The returned varrow() function draws a line with a filled triangle arrowhead
    pointing downward, with an optional pill label beside the shaft.

    Args:
        d (ImageDraw.ImageDraw): The draw context to bind the closure to.

    Returns:
        Callable: varrow(cx, y0, y1, label=None) that draws one vertical connector.
    """
    def varrow(cx, y0, y1, label=None):
        d.line([cx, y0, cx, y1], fill=ARROW, width=3)
        d.polygon([(cx - 7, y1 - 10), (cx + 7, y1 - 10), (cx, y1)], fill=ARROW)
        if label:
            w = d.textlength(label, font=f_label)
            pad = 6
            ly = (y0 + y1) / 2 - f_label.size / 2
            d.rectangle(
                [cx + 13, ly - pad + 2, cx + 13 + w + 2 * pad, ly + f_label.size + pad - 2],
                fill=BG,
            )
            d.text((cx + 13 + pad, ly), label, font=f_label, fill=(200, 205, 215))
    return varrow


def make_harrow(d: ImageDraw.ImageDraw):
    """Return a horizontal rightward arrow closure bound to the given ImageDraw context.

    The returned harrow() function draws a line with a filled triangle arrowhead
    pointing right, with an optional label above the shaft.

    Args:
        d (ImageDraw.ImageDraw): The draw context to bind the closure to.

    Returns:
        Callable: harrow(x0, x1, y, label=None) that draws one horizontal connector.
    """
    def harrow(x0, x1, y, label=None):
        d.line([x0, y, x1, y], fill=ARROW, width=2)
        d.polygon([(x1 - 9, y - 6), (x1 - 9, y + 6), (x1, y)], fill=ARROW)
        if label:
            w = d.textlength(label, font=f_tiny)
            mx = (x0 + x1) / 2
            d.text((mx - w / 2, y - f_tiny.size - 4), label, font=f_tiny, fill=DIMMED)
    return harrow


# ── Per-Tier Section Drawers ──────────────────────────────────────────────────

def _draw_title(d: ImageDraw.ImageDraw, W: int) -> None:
    """Draw the diagram title and subtitle centered at the top of the canvas.

    Args:
        d (ImageDraw.ImageDraw): Active draw context.
        W (int): Canvas width in pixels.
    """
    center(d, W / 2, 22, "AttributionLens  —  Repository Architecture", f_title, INK)
    center(
        d, W / 2, 65,
        "AI authorship detector  ·  Flask API + three orthogonal signals"
        " + SQLite audit log + React dashboard",
        f_sub, SUB,
    )


def _draw_client_tier(
    box, varrow, LX: float, BW: float, CX: float
) -> int:
    """Draw Tier 1 — HTTP client block and the arrow leaving it.

    Args:
        box (Callable): Bound box drawing closure.
        varrow (Callable): Bound vertical arrow closure.
        LX (float): Left x of the primary column.
        BW (float): Width of primary column boxes.
        CX (float): Center x of the primary column.

    Returns:
        int: The bottom y of the tier box (before the arrow gap).
    """
    y = 110
    box(
        LX, y, BW, 72, C_CLIENT,
        title="HTTP Client  (browser / curl / any caller)",
        lines=["POST /submit  ·  POST /appeal  ·  GET /content/<id>  ·  GET /health"],
    )
    varrow(CX, y + 72, y + 72 + 38, "JSON request")
    return y + 72


def _draw_flask_tier(
    box, varrow, LX: float, BW: float, CX: float, client_bottom: int
) -> int:
    """Draw Tier 2 — Flask app.py block and the arrow leaving it.

    Args:
        box (Callable): Bound box drawing closure.
        varrow (Callable): Bound vertical arrow closure.
        LX (float): Left x of the primary column.
        BW (float): Width of primary column boxes.
        CX (float): Center x of the primary column.
        client_bottom (int): Bottom y of the Tier 1 box.

    Returns:
        int: The bottom y of the tier box (before the arrow gap).
    """
    y = client_bottom + 38 + 38
    box(
        LX, y, BW, 148, C_FLASK,
        title="Flask Application  (app.py)",
        sub="entry point · wires routes to signals and audit log",
        lines=[
            "Flask-Limiter: 10/hr submit · 5/hr appeal · 100/hr global (per IP)",
            "Input validation: text length [1, 50 000 chars], JSON shape",
            "Prompt injection defence: submitted text wrapped in data delimiters",
            "POST /submit orchestrates signals → scorer → label → audit log",
            "Graceful degradation: proceeds on Groq failure, caps confidence ≤ 0.5",
        ],
        line_font=f_small,
    )
    varrow(CX, y + 148, y + 148 + 38, "validated text")
    return y + 148


def _draw_signal_tier(
    box, varrow, harrow,
    LX: float, BW: float, AX: float, AW: float,
    flask_bottom: int,
) -> tuple[int, int]:
    """Draw Tier 3 — three parallel detection signal blocks and their connectors.

    Also draws the Groq external service annotation and the shared util helpers
    annotation in the annotations column (AX/AW). Returns the bottom y of the
    signal boxes and the y coordinate at which the scorer tier should start.

    Args:
        box (Callable): Bound box drawing closure.
        varrow (Callable): Bound vertical arrow closure.
        harrow (Callable): Bound horizontal arrow closure.
        LX (float): Left x of the primary column.
        BW (float): Width of primary column boxes.
        AX (float): Left x of the annotations column.
        AW (float): Width of the annotations column.
        flask_bottom (int): Bottom y of the Flask tier box.

    Returns:
        tuple[int, int]: (signal_box_bottom_y, scorer_start_y).
    """
    SIG_W = 196
    SIG_H = 178
    GAP   = 18
    SIG_Y = flask_bottom + 38 + 38

    # Signal 1 — LLM
    s1x = LX
    box(
        s1x, SIG_Y, SIG_W, SIG_H, C_LLM,
        title="Signal 1",
        sub="llm_signal.py",
        lines=[
            "LLM semantic analysis",
            "Groq llama-3.3-70b",
            "→ p_ai_llm ∈ [0, 1]",
            "+ rationale string",
            "+ available flag",
            "",
            "Weight in scorer: 60%",
        ],
        line_font=f_small,
    )

    # Signal 2 — Stylometric
    s2x = s1x + SIG_W + GAP
    box(
        s2x, SIG_Y, SIG_W, SIG_H, C_STYLE,
        title="Signal 2",
        sub="stylometric_signal.py",
        lines=[
            "Pure Python heuristics",
            "4 structural features:",
            "  burstiness · TTR",
            "  punct density",
            "  sentence complexity",
            "→ p_ai_style ∈ [0, 1]",
            "Weight in scorer: 40%",
        ],
        line_font=f_small,
    )

    # Signal 3 — Grounding
    s3x = s2x + SIG_W + GAP
    box(
        s3x, SIG_Y, SIG_W, SIG_H, C_GROUND,
        title="Signal 3",
        sub="grounding_signal.py",
        lines=[
            "Pure Python content",
            "grounding heuristics",
            "4 specificity dims:",
            "  temporal · spatial",
            "  sensory · firsthand",
            "→ grounding_factor",
            "  ∈ [0.85, 1.15]",
        ],
        line_font=f_small,
    )

    # Arrows: Flask → each signal
    for sx in [s1x + SIG_W // 2, s2x + SIG_W // 2, s3x + SIG_W // 2]:
        varrow(sx, flask_bottom + 38, SIG_Y)

    scorer_y = SIG_Y + SIG_H + 42

    # Arrows: each signal → scorer
    for sx in [s1x + SIG_W // 2, s2x + SIG_W // 2, s3x + SIG_W // 2]:
        varrow(sx, SIG_Y + SIG_H, scorer_y)

    # Annotations column — Groq external service (top, beside Signal 1)
    GROQ_H = 74
    box(
        AX, SIG_Y, AW, GROQ_H, C_SIDE,
        title="External: Groq API",
        title_font=f_smallb,
        sub=".env  ·  GROQ_API_KEY",
        lines=["llama-3.3-70b-versatile", "unavailable → degraded mode"],
        line_font=f_tiny, align_center=False,
    )
    harrow(s1x + SIG_W, AX, SIG_Y + 32, "API call")

    # Annotations column — util helpers (below Groq, with a clear gap)
    UTIL_H = 100
    util_y = SIG_Y + GROQ_H + 10
    box(
        AX, util_y, AW, UTIL_H, C_SIDE,
        title="util/util.py  — shared helpers",
        title_font=f_smallb,
        lines=[
            "NEUTRAL_SCORE=0.5",
            "MIN_RELIABLE_WORDS=40",
            "WORD_RE · SENTENCE_SPLIT_RE",
            "clamp01() · extract_words()",
            "split_sentences()",
        ],
        line_font=f_tiny, align_center=False,
    )

    return SIG_Y + SIG_H, scorer_y


def _draw_scoring_tier(
    box, varrow, LX: float, BW: float, CX: float, scorer_y: int
) -> int:
    """Draw Tier 4a — confidence scorer block and the arrow leaving it.

    Args:
        box (Callable): Bound box drawing closure.
        varrow (Callable): Bound vertical arrow closure.
        LX (float): Left x of the primary column.
        BW (float): Width of primary column boxes.
        CX (float): Center x of the primary column.
        scorer_y (int): Top y for this tier block.

    Returns:
        int: The bottom y of the tier box.
    """
    box(
        LX, scorer_y, BW, 130, C_SCORE,
        title="Confidence Scorer  (scoring/scoring.py)",
        sub="combines two probability signals + grounding modifier",
        lines=[
            "combined_p_ai = 0.6 × p_ai_llm + 0.4 × p_ai_style",
            "agreement = 1 − |p_ai_llm − p_ai_style|"
            "  ·  decisiveness = 2×|combined_p_ai − 0.5|",
            "confidence = decisiveness × agreement × grounding_factor  (clamped [0, 1])",
            "Verdict: likely_ai ≥ 0.65  ·  likely_human ≤ 0.40  ·  else uncertain",
        ],
        line_font=f_small,
    )
    varrow(CX, scorer_y + 130, scorer_y + 130 + 38, "ScoreResult")
    return scorer_y + 130


def _draw_label_tier(
    box, varrow,
    LX: float, BW: float, CX: float, AX: float, AW: float,
    scorer_y: int, scorer_bottom: int,
) -> int:
    """Draw Tier 4b — transparency label generator block, arrow, and test annotation.

    Args:
        box (Callable): Bound box-drawing closure.
        varrow (Callable): Bound vertical-arrow closure.
        LX (float): Left x of the primary column.
        BW (float): Width of primary column boxes.
        CX (float): Center x of the primary column.
        AX (float): Left x of the annotations column.
        AW (float): Width of the annotations column.
        scorer_y (int): Top y of the scorer tier (used to anchor the tests box).
        scorer_bottom (int): Bottom y of the scorer tier box.

    Returns:
        int: The bottom y of the tier box.
    """
    label_y = scorer_bottom + 38 + 38
    box(
        LX, label_y, BW, 96, C_LABEL,
        title="Transparency Label Generator  (scoring/labels.py)",
        sub="maps verdict + confidence → one of 3 reader-facing label variants",
        lines=[
            "high_confidence_ai · high_confidence_human · uncertain",
            "AI variant gated on confidence ≥ 0.20  (false-positive guard)",
        ],
        line_font=f_small,
    )
    varrow(CX, label_y + 96, label_y + 96 + 38, "TransparencyLabel")

    # Annotations column — test suite anchored at scorer_y so it never
    # overlaps the Groq/util boxes which sit at signal tier level
    tests_h = (label_y + 96) - scorer_y
    box(
        AX, scorer_y, AW, tests_h, C_SIDE,
        title="tests/  — 132 tests, 13 files",
        title_font=f_smallb,
        lines=[
            "conftest.py: FakeGroqClient · isolated DB",
            "test_llm/stylometric/grounding_signal.py",
            "test_scoring.py",
            "test_scoring_with_grounding.py",
            "test_labels.py · test_audit_log.py",
            "test_submit/appeal/rate_limit_route.py",
            "test_dashboard.py (24 tests)",
        ],
        line_font=f_tiny, align_center=False,
    )

    return label_y + 96


def _draw_audit_tier(
    box, varrow, harrow,
    LX: float, BW: float, CX: float,
    APX: float, APW: float,
    label_bottom: int,
) -> int:
    """Draw Tier 5 — SQLite audit log block, arrow, and appeals sidebar.

    Args:
        box (Callable): Bound box drawing closure.
        varrow (Callable): Bound vertical arrow closure.
        harrow (Callable): Bound horizontal arrow closure.
        LX (float): Left x of the primary column.
        BW (float): Width of primary column boxes.
        CX (float): Center x of the primary column.
        APX (float): Left x of the appeals sidebar.
        APW (float): Width of the appeals sidebar.
        label_bottom (int): Bottom y of the label tier box.

    Returns:
        int: The bottom y of the audit tier box.
    """
    audit_y = label_bottom + 38 + 38
    AUDIT_H = 138
    box(
        LX, audit_y, BW, AUDIT_H, C_AUDIT,
        title="SQLite Audit Log  (audit_log/audit_log.py)",
        sub="audit_log/audit_log.db  ·  16-column decisions + 5-column appeals tables",
        lines=[
            "record_decision(): persists all signal scores, verdict, label, status",
            "get_decision() / set_status(): powers human-reviewer GET /content/<id>",
            "record_appeal() / get_appeals(): appeals workflow (under_review flip)",
            "get_dashboard_stats|timeseries|scatter(): read-only analytics queries",
        ],
        line_font=f_small,
    )
    varrow(CX, audit_y + AUDIT_H, audit_y + AUDIT_H + 38, "JSON response to caller")

    # Left panel — appeals sidebar: header + 3 cards, all contained within audit tier height
    SIDEBAR_TOTAL = AUDIT_H        # match the audit box height exactly
    HEADER_H      = 32
    CARD_H        = 62
    CARD_GAP      = 4
    # 3 cards + 2 gaps + header + gap below header
    cards_block = 3 * CARD_H + 2 * CARD_GAP
    header_gap  = SIDEBAR_TOTAL - HEADER_H - cards_block

    box(
        APX, audit_y, APW, HEADER_H, C_APPEAL,
        title="Appeal Review Path", title_font=f_smallb,
    )
    appeal_items = [
        ("POST /appeal",      "creator files appeal",   ["content_id + reasoning", "→ status: under_review"]),
        ("GET /content/<id>", "human reviewer fetches", ["full decision record:", "scores + label + appeals"]),
        ("set_status()",      "never auto-corrects;",   ["always escalates to", "a human reviewer"]),
    ]
    ap_y = audit_y + HEADER_H + max(header_gap, 4)
    for title_t, sub_t, lines_t in appeal_items:
        box(
            APX, ap_y, APW, CARD_H, C_SIDE,
            title=title_t, title_font=f_smallb,
            sub=sub_t, lines=lines_t,
            line_font=f_tiny, align_center=False,
        )
        ap_y += CARD_H + CARD_GAP

    harrow(LX, APX + APW, audit_y + HEADER_H + 20, "appeal path")

    return audit_y + AUDIT_H


def _draw_response_tier(
    box, LX: float, BW: float, audit_bottom: int
) -> int:
    """Draw the final JSON response block at the bottom of the primary column.

    Args:
        box (Callable): Bound box-drawing closure.
        LX (float): Left x of the primary column.
        BW (float): Width of primary column boxes.
        audit_bottom (int): Bottom y of the audit tier box.

    Returns:
        int: The bottom y of the response block.
    """
    resp_y = audit_bottom + 38 + 38
    box(
        LX, resp_y, BW, 68, C_CLIENT,
        title="JSON Response  →  HTTP caller",
        lines=["verdict · confidence · combined_p_ai · label_variant · label_text · content_id"],
        line_font=f_small,
    )
    return resp_y + 68


def _draw_dashboard_panel(
    box, harrow,
    DX: float, DW: float,
    audit_y: int,
    LX: float, BW: float,
) -> None:
    """Draw the right-side React dashboard panel with component annotations.

    The panel is anchored so its top aligns with the audit log tier, keeping it
    clearly separated from the annotations column above.

    Args:
        box (Callable): Bound box-drawing closure.
        harrow (Callable): Bound horizontal-arrow closure.
        DX (float): Left x of the dashboard column.
        DW (float): Width of the dashboard column.
        audit_y (int): Top y of the audit log tier (used to anchor the panel).
        LX (float): Left x of the primary column (for arrow source).
        BW (float): Width of primary column boxes (for arrow source).
    """
    HEADER_H = 44
    APP_H    = 68
    COMP_H   = 66
    COMP_GAP = 8

    header_y = audit_y
    box(
        DX, header_y, DW, HEADER_H, C_DASH,
        title="React + Vite + Recharts  Dashboard",
        title_font=f_smallb,
    )

    app_y = header_y + HEADER_H + COMP_GAP
    box(
        DX, app_y, DW, APP_H, C_DASH,
        title="App.jsx  — root component",
        title_font=f_smallb,
        lines=["Promise.all 3 endpoints on mount · 2×2 grid", "loading / error state · refresh button"],
        line_font=f_tiny, align_center=False,
    )

    comp_y = app_y + APP_H + COMP_GAP
    components = [
        ("MetricCards.jsx",     "GET /api/dashboard/stats",     ["total · verdicts · appeal rate · grounding-boost %"]),
        ("VerdictBarChart.jsx", "GET /api/dashboard/timeseries", ["stacked bar · daily AI / human / uncertain"]),
        ("AppealsChart.jsx",    "GET /api/dashboard/stats",     ["grouped bar · appeals by verdict"]),
        ("SignalHeatmap.jsx",   "GET /api/dashboard/scatter",   ["scatter · p_ai_llm vs p_ai_style · coloured by verdict"]),
    ]
    for name, endpoint, desc in components:
        box(
            DX, comp_y, DW, COMP_H, C_SIDE,
            title=name, title_font=f_smallb,
            sub=endpoint, lines=desc,
            line_font=f_tiny, align_center=False,
        )
        comp_y += COMP_H + COMP_GAP

    # Arrow: audit log → dashboard (read queries)
    harrow(LX + BW, DX, audit_y + HEADER_H // 2, "read-only queries")


def _draw_footer(
    d: ImageDraw.ImageDraw, CX: float, resp_bottom: int
) -> None:
    """Draw the two line flow summary footer below the response block.

    Args:
        d (ImageDraw.ImageDraw): Active draw context.
        CX (float): Center x of the primary column (for centering text).
        resp_bottom (int): Bottom y of the response block.
    """
    footer_y = resp_bottom + 22
    center(
        d, CX, footer_y,
        "Flow: POST /submit → rate limit + validate → Signal 1 (LLM) ∥ Signal 2 (Stylometry)"
        " ∥ Signal 3 (Grounding) → scorer → label → audit log → JSON response",
        f_small, SUB,
    )
    center(
        d, CX, footer_y + 22,
        "Appeal path: POST /appeal → set_status(under_review) → GET /content/<id> (human reviewer)",
        f_small, DIMMED,
    )


# ── Top Level Diagram Generator ───────────────────────────────────────────────

def gen_repo_diagram(out_path: str | os.PathLike) -> None:
    """Render the AttributionLens repository architecture diagram and save it as a PNG.

    Orchestrates the per-tier helper functions to build a 1700×1900 pixel dark-
    background diagram that captures the full five tier pipeline: HTTP client →
    Flask → three parallel signals → confidence scorer + label generator →
    SQLite audit log → JSON response.

    Column layout (left to right):
      - APX/APW : left appeals sidebar (human-in-the-loop path)
      - LX/BW   : primary pipeline column (all main tier boxes)
      - AX/AW   : annotations column (Groq, util helpers, test suite)
      - DX/DW   : React dashboard column (anchored to audit tier)

    A two line footer below the response block summarises the end-to-end flow.

    Args:
        out_path (str | os.PathLike): Destination path for the output PNG.

    Side effects:
        Writes a 1700×1900 PNG to out_path and prints the path and dimensions
        to stdout.
    """
    W, H = 1700, 1900
    img = Image.new("RGB", (W, H), BG)
    d   = ImageDraw.Draw(img)

    box    = make_box(d)
    varrow = make_varrow(d)
    harrow = make_harrow(d)

    # Column geometry — four non-overlapping vertical strips
    APX = 14           # appeals sidebar left-x
    APW = 148          # appeals sidebar width
    LX  = APX + APW + 20   # primary pipeline left-x
    BW  = 640          # primary pipeline width
    CX  = LX + BW / 2  # primary pipeline center-x
    AX  = LX + BW + 20  # annotations column left-x
    AW  = 220          # annotations column width
    DX  = AX + AW + 20  # dashboard column left-x
    DW  = W - DX - 18   # dashboard column width

    _draw_title(d, W)

    client_bottom = _draw_client_tier(box, varrow, LX, BW, CX)
    flask_bottom  = _draw_flask_tier(box, varrow, LX, BW, CX, client_bottom)

    _sig_bottom, scorer_y = _draw_signal_tier(
        box, varrow, harrow, LX, BW, AX, AW, flask_bottom
    )

    # Signal tier label drawn here where the real draw context d is in scope
    SIG_Y = flask_bottom + 38 + 38
    center(d, CX, SIG_Y - 22,
           "Three orthogonal detection signals  (run in parallel)", f_tiny, DIMMED)

    scorer_bottom = _draw_scoring_tier(box, varrow, LX, BW, CX, scorer_y)
    label_bottom  = _draw_label_tier(box, varrow, LX, BW, CX, AX, AW, scorer_y, scorer_bottom)

    # Derive audit_y before calling _draw_audit_tier so we can pass it to dashboard
    audit_y      = label_bottom + 38 + 38
    audit_bottom = _draw_audit_tier(box, varrow, harrow, LX, BW, CX, APX, APW, label_bottom)
    resp_bottom  = _draw_response_tier(box, LX, BW, audit_bottom)

    _draw_dashboard_panel(box, harrow, DX, DW, audit_y, LX, BW)
    _draw_footer(d, CX, resp_bottom)

    img.save(out_path)
    print("wrote", out_path, img.size)


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    out  = here.parent / "util" / "Repo-Architecture.png"
    gen_repo_diagram(out)
