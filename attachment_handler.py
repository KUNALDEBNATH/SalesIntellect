"""
attachment_handler.py
═══════════════════════════════════════════════════════════════════════════════
Handles uploaded file attachments (documents + images) for the sales chatbot.

ALL Qwen / transformers references removed.
Uses ScratchLLM for any text generation needed during document Q&A.

Supported file types
────────────────────
  Documents : PDF (.pdf), Word (.docx), Text (.txt), CSV (.csv), Excel (.xlsx/.xls)
  Images    : PNG, JPG, JPEG, WEBP

Size limits (enforced before processing):
  Documents : 15 MB
  Images    :  8 MB

Architecture
────────────
  1. Uploaded file → saved to a temp path.
  2. document_parser.py  extracts text / rows + builds a TF-IDF retriever.
  3. vision_parser.py    handles images (OCR → scratch LLM).
  4. A follow-up store lets the user continue asking questions about the
     same document without re-uploading it.
"""

from __future__ import annotations

import os
import re
import tempfile
import threading
import time
from typing import Dict, Optional

_DEFAULT_SESSION = "default"

# ── Scratch LLM ──────────────────────────────────────────────────────────────
from scratch_llm import ScratchLLM as _ScratchLLM

_scratch_llm  = _ScratchLLM()
_llm_lock     = threading.Lock()
_llm_loaded   = False


def _ensure_llm() -> bool:
    global _llm_loaded
    if _llm_loaded:
        return True
    with _llm_lock:
        if not _llm_loaded:
            _llm_loaded = _scratch_llm.load()
    return _llm_loaded


# ── Document / vision helpers ─────────────────────────────────────────────────
import document_parser
import vision_parser
from document_parser import ParsedDocument, parse_document, retrieve_relevant_rows_or_chunks, compute_quick_stat

# ── Domain helper (for "is this clearly a sales question?") ──────────────────
import domain_guard

# ── Anti-hallucination verification (numbers / codes / proper nouns must
# actually appear in the retrieved content) ──────────────────────────────────
from document_verifier import ground_answer_or_fallback


# ════════════════════════════════════ LIMITS ══════════════════════════════════

MAX_DOC_BYTES   = 15 * 1024 * 1024     # 15 MB
MAX_IMAGE_BYTES =  8 * 1024 * 1024     #  8 MB

DOC_EXTENSIONS   = {".pdf", ".docx", ".txt", ".csv", ".xlsx", ".xls"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

# ════════════════════════════════ FOLLOW-UP STORE ═════════════════════════════
#
# Keyed by session_id. This used to be a single pair of module-level
# variables (_stored_doc / _turns_since_doc), which meant EVERY caller of
# api.py shared one uploaded document — customer A uploads an invoice,
# customer B asks an unrelated follow-up, and B's "summarize it again"
# would silently answer from A's file. Per-session storage fixes that;
# callers that never pass a session_id fall back to "default" so single-
# user / CLI usage is unaffected.

_stored_docs: Dict[str, ParsedDocument] = {}
_stored_doc_lock = threading.Lock()
_turns_since_doc: Dict[str, int] = {}


def _store_document(doc: ParsedDocument, session_id: str = _DEFAULT_SESSION) -> None:
    with _stored_doc_lock:
        _stored_docs[session_id]      = doc
        _turns_since_doc[session_id]  = 0


def clear_stored_document(session_id: str = _DEFAULT_SESSION) -> None:
    with _stored_doc_lock:
        _stored_docs.pop(session_id, None)
        _turns_since_doc.pop(session_id, None)


def has_stored_document(session_id: str = _DEFAULT_SESSION) -> bool:
    with _stored_doc_lock:
        return _stored_docs.get(session_id) is not None


def mark_other_turn(session_id: str = _DEFAULT_SESSION) -> None:
    """Called by api.py after any non-attachment turn."""
    with _stored_doc_lock:
        if _stored_docs.get(session_id) is not None:
            _turns_since_doc[session_id] = _turns_since_doc.get(session_id, 0) + 1
            if _turns_since_doc[session_id] > 6:
                # Auto-expire after 6 unrelated turns
                _stored_docs.pop(session_id, None)
                _turns_since_doc.pop(session_id, None)


# ══════════════════════════════════ ERROR TYPE ════════════════════════════════

class AttachmentError(Exception):
    """Raised for user-facing errors (bad file type, too large, parse fail)."""


# ══════════════════════════════ CONTEXT FOLLOW-UP ════════════════════════════

_CONTEXT_FOLLOWUP_SIGNALS = {
    "this", "it", "the document", "the file", "the pdf", "the csv",
    "the sheet", "the image", "the photo", "the picture", "the table",
    "from this", "in this", "of this", "about this", "summarize it",
    "summarise it", "summarize again", "summarise again",
    "explain it", "explain this", "read it", "read this",
    "translate it", "translate this", "above", "uploaded",
    "now from this", "now from the",
}

_CONTEXT_PRONOUNS = {"it", "this", "that", "these", "those", "them"}


def looks_like_context_followup(query: str) -> bool:
    """
    True when the query appears to be a follow-up about the previously
    uploaded file — uses pronouns or explicit document references —
    rather than a fresh, independent question.
    """
    q_low    = query.lower().strip()
    q_tokens = set(re.findall(r"[a-zA-Z]+", q_low))

    # Explicit document phrases
    for sig in _CONTEXT_FOLLOWUP_SIGNALS:
        if sig in q_low:
            return True

    # Very short messages that contain a bare pronoun and no sales vocabulary
    if len(q_tokens) <= 6 and q_tokens & _CONTEXT_PRONOUNS:
        if not domain_guard.has_sales_keywords(query):
            return True

    return False


# ═══════════════════════════════ ANSWER BUILDER ═══════════════════════════════

def _build_doc_answer(doc: ParsedDocument, query: str) -> str:
    """
    Core RAG answer builder for an uploaded document/spreadsheet.

    1. Retrieve top-k relevant chunks / rows via TF-IDF.
    2. For tables, also try to compute a direct statistic (avg / count).
    3. Assemble context → scratch LLM → answer.
       Falls back to the structured context string if the LLM produces nothing.
    """
    chunks      = retrieve_relevant_rows_or_chunks(doc, query, top_k=6)
    quick_stat  = compute_quick_stat(doc, query) if doc.kind == "table" else None

    context_parts = []
    if quick_stat:
        context_parts.append(f"Computed fact: {quick_stat}")
    if chunks:
        context_parts.append("Relevant content:\n" + "\n---\n".join(chunks[:5]))
    elif doc.kind == "text" and doc.raw_text:
        context_parts.append("Document text (first 1500 chars):\n" +
                              doc.raw_text[:1500])
    elif doc.kind == "table" and doc.row_texts:
        context_parts.append("Table rows:\n" +
                              "\n".join(doc.row_texts[:10]))

    context = "\n\n".join(context_parts).strip()
    if not context:
        context = doc.summary

    if _ensure_llm():
        prompt = (
            f"Document: {doc.filename}\n"
            f"Summary : {doc.summary}\n\n"
            f"{context}\n\n"
            f"Question: {query}\n\n"
            "Answer using only the content above. "
            "Be concise and factual. One to three sentences."
        )
        answer = _scratch_llm.generate(prompt, max_new=150, temperature=0.4, top_p=0.9)
        if answer and len(answer.strip()) >= 15:
            BAD = ["i don't know", "i cannot", "as an ai", "i was trained",
                   "no information", "not able to"]
            if not any(b in answer.lower() for b in BAD):
                # Groundedness check: reject (and fall back to raw context)
                # if the generated sentence introduces a number, code/ID, or
                # proper noun that doesn't actually appear in what was
                # retrieved — the same anti-hallucination guarantee
                # verification.py already gives the sales-data pipeline.
                fallback_context = (chunks[0] if chunks else doc.summary)
                verify_against = ([quick_stat] if quick_stat else []) + chunks
                return ground_answer_or_fallback(
                    answer.strip(), verify_against, document_filename=doc.filename,
                    fallback_context=fallback_context,
                )

    # Fallback: return the structured context directly
    if quick_stat:
        return quick_stat + "\n\n" + (chunks[0] if chunks else doc.summary)
    if chunks:
        return "Based on the document:\n\n" + "\n\n".join(chunks[:3])
    return doc.summary


# ══════════════════════════════ MAIN PUBLIC API ═══════════════════════════════

def handle_attachment(uploaded_file, query: str, chatbot=None,
                       session_id: str = _DEFAULT_SESSION) -> dict:
    """
    Process an uploaded file and return a result dict:
      { answer, intent, elapsed, filename }

    `uploaded_file` is a Django InMemoryUploadedFile or TemporaryUploadedFile.
    `session_id` scopes the stored-document follow-up cache to one caller.
    Raises AttachmentError for user-facing problems.
    """
    t0       = time.time()
    filename = uploaded_file.name or "upload"
    ext      = os.path.splitext(filename)[1].lower()

    # ── Validate extension ────────────────────────────────────────────────────
    if ext not in DOC_EXTENSIONS and ext not in IMAGE_EXTENSIONS:
        raise AttachmentError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(sorted(DOC_EXTENSIONS | IMAGE_EXTENSIONS))}."
        )

    # ── Validate size ─────────────────────────────────────────────────────────
    limit = MAX_IMAGE_BYTES if ext in IMAGE_EXTENSIONS else MAX_DOC_BYTES
    size  = uploaded_file.size
    if size > limit:
        mb = limit // (1024 * 1024)
        raise AttachmentError(
            f"File too large ({size // 1024} KB). "
            f"Maximum allowed for this type: {mb} MB."
        )

    # ── Save to temp file ─────────────────────────────────────────────────────
    suffix  = ext
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        # ── Image path ────────────────────────────────────────────────────────
        if ext in IMAGE_EXTENSIONS:
            effective_query = query or "Describe this image."
            answer = vision_parser.analyze_image(tmp_path, effective_query)
            elapsed = round(time.time() - t0, 3)
            return {
                "answer":   answer,
                "intent":   "image_analysis",
                "elapsed":  elapsed,
                "filename": filename,
            }

        # ── Document path ─────────────────────────────────────────────────────
        try:
            doc = parse_document(tmp_path, filename)
        except ValueError as exc:
            raise AttachmentError(str(exc))

        _store_document(doc, session_id=session_id)

        effective_query = query or "Summarise this document."
        answer          = _build_doc_answer(doc, effective_query)

        elapsed = round(time.time() - t0, 3)
        return {
            "answer":   answer,
            "intent":   "document_analysis",
            "elapsed":  elapsed,
            "filename": filename,
        }

    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def answer_from_stored_document(query: str, chatbot=None,
                                 session_id: str = _DEFAULT_SESSION) -> Optional[dict]:
    """
    Answer a follow-up question using the previously uploaded document,
    without requiring the user to re-upload the file.

    Returns None if there is no stored document for this session.
    """
    with _stored_doc_lock:
        doc = _stored_docs.get(session_id)

    if doc is None:
        return None

    t0     = time.time()
    answer = _build_doc_answer(doc, query)
    return {
        "answer":   answer,
        "intent":   "document_followup",
        "elapsed":  round(time.time() - t0, 3),
        "filename": doc.filename,
    }
