"""
verification.py
────────────────────────────────────────────────────────────────────────────
Fact-level verification of the scratch LLM's generated phrasing against the
deterministic, already-computed facts. The LLM is allowed to improve
wording; it is NOT allowed to change numbers, IDs, names, or counts.

Used for BOTH:
  * query_engine.EngineResult (multi-hop / aggregation answers)
  * the plain structured_facts string test.py already builds

If verification fails, callers should discard the generated text and show
the structured facts alone — the answer is never blocked by a bad
generation, only downgraded to "just the facts".
"""

from __future__ import annotations

import re
from typing import Iterable, Tuple

_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_ID_RE = re.compile(r"\bENQ\d{2,6}\b", re.I)


def _numbers(text: str) -> set:
    return set(_NUM_RE.findall(text or ""))


def _ids(text: str) -> set:
    return {m.upper() for m in _ID_RE.findall(text or "")}


def verify_answer(generated: str, facts: str,
                   required_words: Iterable[str] = ()) -> Tuple[bool, str]:
    """
    Returns (ok, reason). `ok` is False if the generated text introduces
    numbers or enquiry IDs that do not appear anywhere in `facts` — a
    strong signal of hallucinated figures, since every real number in
    this system originates from a pandas computation in `facts`.
    """
    if not generated or not generated.strip():
        return False, "empty generation"

    gen_numbers = _numbers(generated)
    fact_numbers = _numbers(facts)
    bad_numbers = gen_numbers - fact_numbers
    if bad_numbers:
        return False, f"generated numbers not found in facts: {sorted(bad_numbers)}"

    gen_ids = _ids(generated)
    fact_ids = _ids(facts)
    bad_ids = gen_ids - fact_ids
    if bad_ids:
        return False, f"generated IDs not found in facts: {sorted(bad_ids)}"

    for w in required_words:
        if w and w.lower() not in generated.lower():
            return False, f"required entity '{w}' missing from generation"

    # groundedness: generation should share real vocabulary with the facts,
    # not be an unrelated sentence
    fact_words = set(re.findall(r"[a-zA-Z]{4,}", (facts or "").lower()))
    gen_words = set(re.findall(r"[a-zA-Z]{4,}", generated.lower()))
    if fact_words and not (fact_words & gen_words):
        return False, "generation shares no vocabulary with the facts"

    return True, "ok"
