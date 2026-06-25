"""signals - Detection signal modules for the Provenance Guard attribution pipeline.

Each submodule implements one independent classification signal that returns a
probability in [0, 1] plus metadata. Signals are never binary flags; the
confidence scorer in scoring.py combines them: Signals 1 and 2 contribute to
combined_p_ai (LLM 60%, stylometric 40%, per planning.md Section 5), and Signal
3 acts as a confidence modifier via grounding_factor in [0.85, 1.15].

Current signals:
  llm_signal          Signal 1 - LLM semantic/stylistic analysis via Groq.
  stylometric_signal  Signal 2 - Heuristic structural analysis (pure Python).
  grounding_signal    Signal 3 - Experiential content-grounding analysis (pure Python).
"""
