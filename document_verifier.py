"""
document_verifier.py
═══════════════════════════════════════════════════════════════════════════════
Applies the same anti-hallucination verification to document answers
that verification.py already applies to sales data answers.

Pipeline:
  DOCUMENT → PARSER → RETRIEVAL → STRUCTURED FACTS → SCRATCH LLM
  → DOCUMENT FACT VERIFIER → FINAL ANSWER

If verification fails: return grounded context or abstain.

Key checks:
  1. Every number in generated text must appear in the retrieved chunks.
  2. Every proper noun must appear in retrieved text (or document filename).
  3. Generated answer shares vocabulary with the chunks (groundedness).
  4. Answer does not introduce IDs / codes not present in retrieved context.
"""

from __future__ import annotations

import re
from typing import List, Tuple

_NUM_RE  = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_CODE_RE = re.compile(r"\b[A-Z]{2,}\d{2,}\b")   # e.g. INV001, REF42
_CAP_WORD_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")


def _numbers(text: str) -> set:
    return set(_NUM_RE.findall(text or ""))


def _codes(text: str) -> set:
    return {m.upper() for m in _CODE_RE.findall(text or "")}


def _cap_words(text: str) -> set:
    return set(_CAP_WORD_RE.findall(text or ""))


def _vocab_words(text: str) -> set:
    return set(re.findall(r"[a-zA-Z]{4,}", (text or "").lower()))


def verify_document_answer(
    generated: str,
    retrieved_chunks: List[str],
    document_filename: str = "",
    strict: bool = True,
) -> Tuple[bool, str]:
    """
    Returns (ok, reason).
    `ok` is False if the generated text introduces facts not supported by
    the retrieved chunks.
    """
    if not generated or not generated.strip():
        return False, "empty generation"

    context = " ".join(retrieved_chunks) + " " + document_filename
    context_lower = context.lower()

    # ── Check 1: introduced numbers ──────────────────────────────────────────
    gen_nums = _numbers(generated)
    ctx_nums = _numbers(context)
    bad_nums = gen_nums - ctx_nums
    if bad_nums and strict:
        return False, f"numbers not found in context: {sorted(bad_nums)}"

    # ── Check 2: introduced codes / IDs ──────────────────────────────────────
    gen_codes = _codes(generated)
    ctx_codes = _codes(context)
    bad_codes = gen_codes - ctx_codes
    if bad_codes and strict:
        return False, f"codes/IDs not found in context: {sorted(bad_codes)}"

    # ── Check 3: vocabulary groundedness ─────────────────────────────────────
    gen_words = _vocab_words(generated)
    ctx_words = _vocab_words(context)
    if ctx_words and not (gen_words & ctx_words):
        return False, "generated text shares no vocabulary with retrieved context"

    # ── Check 4: proper nouns must appear in context ──────────────────────────
    if strict:
        gen_caps = _cap_words(generated)
        # Allow words from the filename (e.g. "Invoice", "Report")
        allowed_caps = _cap_words(context) | _cap_words(document_filename)
        # Also allow common English words that might be capitalised
        _COMMON_CAPS = {"The", "This", "Based", "Below", "From", "According",
                        "Here", "Note", "Please", "See", "Yes", "No", "Not",
                        "Computed", "Fact", "Summary", "Document", "File"}
        allowed_caps |= _COMMON_CAPS
        bad_caps = gen_caps - allowed_caps
        if bad_caps:
            return False, f"proper nouns not found in context: {sorted(bad_caps)[:5]}"

    return True, "ok"


def ground_answer_or_fallback(
    generated: str,
    retrieved_chunks: List[str],
    document_filename: str = "",
    fallback_context: str = "",
) -> str:
    """
    Verify the generated answer. If verification passes, return it.
    If not, return the raw retrieved context (always grounded).
    """
    ok, reason = verify_document_answer(generated, retrieved_chunks, document_filename)
    if ok:
        return generated.strip()
    # Fallback: deterministic context, never the hallucinated generation
    if fallback_context:
        return f"Based on the document content:\n\n{fallback_context[:1200]}"
    if retrieved_chunks:
        return "Based on the document content:\n\n" + "\n\n".join(retrieved_chunks[:3])
    return "I could not find relevant content in the document to answer that question."
