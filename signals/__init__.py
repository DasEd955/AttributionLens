"""signals - Detection signal modules for the Provenance Guard attribution pipeline.

Each submodule implements one independent classification signal that returns a
probability in [0, 1] plus metadata. Signals are never binary flags; the
confidence scorer in app.py combines them asymmetrically (LLM 60%, stylometric
40%, per planning.md Section 5).

Current signals:
  llm_signal    Signal 1 - LLM semantic/stylistic analysis via Groq.
  (stylometric  Signal 2 - Heuristic structural analysis, Milestone 4.)
"""
