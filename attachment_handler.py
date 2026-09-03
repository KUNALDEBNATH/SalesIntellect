"""
attachment_handler.py  ── DOCUMENT UNDERSTANDING EDITION
═══════════════════════════════════════════════════════════════════════════════
Handles uploaded file attachments (documents + images) for the sales chatbot.

WHAT CHANGED (vs previous version)
────────────────────────────────────
OLD architecture:
  parse_document → TF-IDF chunks → TextRank/extractive QA → ScratchLLM polish
  → verification → raw extracted text returned to user

NEW architecture:
  parse_document → StructuredDocument → DocumentRepresentation (semantic)
  → DocumentAnswerEngine (deterministic) → ScratchLLM polish (optional)
  → verification → coherent, grounded answer

The key change: DocumentAnswerEngine replaces the TF-IDF retrieval + TextRank
path. It answers from the document's SEMANTIC REPRESENTATION (purpose, section
summaries, facts, conclusions) rather than by selecting raw sentences.

This is the same philosophy as SmartFallbackEngine for sales data:
  GUARANTEE a correct, grounded answer from the data structure first.
  Let the neural model POLISH LANGUAGE only if trained for the task.
  NEVER confuse retrieval with understanding.

Session storage now stores DocumentRepresentation (semantic) instead of
ParsedDocument (raw text + TF-IDF index), so follow-up questions are answered
from the rich semantic memory — not by re-running TF-IDF every time.

Supported file types
────────────────────
  Documents : PDF (.pdf), Word (.docx), Text (.txt), CSV (.csv), Excel (.xlsx/.xls)
  Images    : PNG, JPG, JPEG, WEBP

Size limits:
  Documents : 15 MB
  Images    :  8 MB
"""

from __future__ import annotations

import os
import re
import tempfile
import threading
import time
from typing import Dict, Optional, Tuple

_DEFAULT_SESSION = "default"

# ── Scratch LLM (shared singleton from test.py) ───────────────────────────────
try:
    from test import _scratch_llm, _load_llm as _ensure_llm
except ImportError:
    from scratch_llm import ScratchLLM as _ScratchLLM
    _scratch_llm = _ScratchLLM()
    _llm_lock = threading.Lock()
    _llm_loaded = False

    def _ensure_llm() -> bool:
        global _llm_loaded
        if _llm_loaded:
            return True
        with _llm_lock:
            if not _llm_loaded:
                _llm_loaded = _scratch_llm.load()
        return _llm_loaded


# ── Document understanding pipeline ──────────────────────────────────────────
import document_parser
import vision_parser

from document_parser import parse_document, ParsedDocument

# The new semantic understanding layer
from document_understanding import (
    StructuredDocument,
    DocumentRepresentation,
    parse_document_structure,
    build_document_representation,
    build_table_representation,
)

# The new deterministic answer engine (SmartFallbackEngine for documents)
from document_answer_engine import answer_from_representation

# Domain guard and verification (kept unchanged)
import domain_guard
from document_verifier import ground_answer_or_fallback

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_DOC_BYTES   = 15 * 1024 * 1024
MAX_IMAGE_BYTES =  8 * 1024 * 1024

DOC_EXTENSIONS   = {".pdf", ".docx", ".txt", ".csv", ".xlsx", ".xls"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT MEMORY  (per-session semantic storage)
# ══════════════════════════════════════════════════════════════════════════════
#
# OLD: stored ParsedDocument (raw text + TF-IDF index)
# NEW: stores DocumentRepresentation (semantic: purpose, sections, facts,
#      conclusions, entities, summaries, document map)
#
# This means follow-up questions like:
#   "What is the main contribution?" → section_repr.claims
#   "What evidence supports it?" → section_repr.evidence
#   "Explain the conclusion section" → section QA from repr
# are answered instantly from memory WITHOUT re-running any extraction.

class DocumentMemory:
    """
    Per-session document semantic memory.
    Replaces the bare _stored_docs dict with a typed, thread-safe store.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._representations: Dict[str, DocumentRepresentation] = {}
        self._raw_texts: Dict[str, str] = {}          # kept for general_qa fallback
        self._turns_since_upload: Dict[str, int] = {}
        self._MAX_IDLE_TURNS = 10

    def store(self, session_id: str, rep: DocumentRepresentation, raw_text: str = "") -> None:
        with self._lock:
            self._representations[session_id] = rep
            self._raw_texts[session_id] = raw_text
            self._turns_since_upload[session_id] = 0

    def get(self, session_id: str) -> Optional[Tuple[DocumentRepresentation, str]]:
        with self._lock:
            rep = self._representations.get(session_id)
            raw = self._raw_texts.get(session_id, "")
            return (rep, raw) if rep is not None else None

    def has_document(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._representations

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._representations.pop(session_id, None)
            self._raw_texts.pop(session_id, None)
            self._turns_since_upload.pop(session_id, None)

    def tick_idle(self, session_id: str) -> None:
        """Call after each non-document turn. Auto-expires after MAX_IDLE_TURNS."""
        with self._lock:
            if session_id not in self._representations:
                return
            self._turns_since_upload[session_id] = self._turns_since_upload.get(session_id, 0) + 1
            if self._turns_since_upload[session_id] > self._MAX_IDLE_TURNS:
                self._representations.pop(session_id, None)
                self._raw_texts.pop(session_id, None)
                self._turns_since_upload.pop(session_id, None)


_DOC_MEMORY = DocumentMemory()


# Public interface (same names as before so api.py doesn't need changes)
def _store_document_repr(rep: DocumentRepresentation, raw_text: str,
                          session_id: str = _DEFAULT_SESSION) -> None:
    _DOC_MEMORY.store(session_id, rep, raw_text)


def clear_stored_document(session_id: str = _DEFAULT_SESSION) -> None:
    _DOC_MEMORY.clear(session_id)


def has_stored_document(session_id: str = _DEFAULT_SESSION) -> bool:
    return _DOC_MEMORY.has_document(session_id)


def mark_other_turn(session_id: str = _DEFAULT_SESSION) -> None:
    _DOC_MEMORY.tick_idle(session_id)


# ══════════════════════════════════════════════════════════════════════════════
# CONTEXT FOLLOW-UP DETECTION (unchanged from previous version)
# ══════════════════════════════════════════════════════════════════════════════

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
    q_low    = query.lower().strip()
    q_tokens = set(re.findall(r"[a-zA-Z]+", q_low))
    for sig in _CONTEXT_FOLLOWUP_SIGNALS:
        if sig in q_low:
            return True
    if len(q_tokens) <= 6 and q_tokens & _CONTEXT_PRONOUNS:
        if not domain_guard.has_sales_keywords(query):
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# ERROR TYPE
# ══════════════════════════════════════════════════════════════════════════════

class AttachmentError(Exception):
    """Raised for user-facing errors (bad file type, too large, parse fail)."""


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT INGESTION  (parse → structure → understand → store)
# ══════════════════════════════════════════════════════════════════════════════

def _ingest_document(tmp_path: str, filename: str, debug: bool = False) -> Tuple[DocumentRepresentation, str]:
    """
    Full document ingestion pipeline:
      1. Parse raw text / table data (using existing document_parser)
      2. Build StructuredDocument (sections, paragraphs, headings)
      3. Build DocumentRepresentation (semantic: purpose, summaries, facts, …)

    Returns (DocumentRepresentation, raw_text).
    """
    ext = os.path.splitext(filename)[1].lower()

    # --- Step 1: Extract raw text using existing extractors ---
    try:
        parsed = parse_document(tmp_path, filename)
    except ValueError as e:
        raise AttachmentError(str(e))

    # --- Step 2: Table documents (CSV/Excel) ---
    if parsed.kind == "table" and parsed.dataframe is not None:
        rep = build_table_representation(parsed.dataframe, filename)
        return rep, parsed.summary

    # --- Step 3: Text documents ---
    raw_text = parsed.raw_text
    if not raw_text or not raw_text.strip():
        raise AttachmentError(
            f"Could not extract readable text from '{filename}'. "
            "The file may be empty, image-only (scanned), or corrupted. "
            "For scanned PDFs, please use an image upload instead."
        )

    # Build structure
    structured = parse_document_structure(raw_text, filename)

    # Build semantic representation
    rep = build_document_representation(structured, debug=debug)

    return rep, raw_text


# ══════════════════════════════════════════════════════════════════════════════
# ANSWER BUILDER  (deterministic first, neural polish optional)
# ══════════════════════════════════════════════════════════════════════════════

def _build_doc_answer(rep: DocumentRepresentation, query: str,
                       raw_text: str = "", debug: bool = False) -> str:
    """
    Build a document answer using the new architecture.

    Step 1 (MANDATORY): DocumentAnswerEngine produces a deterministic,
      grounded answer from the DocumentRepresentation. This always succeeds.
      It is the equivalent of SmartFallbackEngine for sales data.

    Step 2 (OPTIONAL): If ScratchLLM has been trained on document tasks
      (task tokens exist in its vocabulary), it may rephrase the deterministic
      answer into more fluent language. The neural output is verified before use.
      If verification fails, the deterministic answer is returned unchanged.

    The neural model NEVER generates the factual content.
    It ONLY polishes language on top of already-correct facts.
    """
    # --- Step 1: Deterministic answer (always runs) ---
    det_answer = answer_from_representation(rep, query, raw_text=raw_text, debug=debug)

    if debug:
        print(f"\n[AttachmentHandler] Deterministic answer ({len(det_answer)} chars):")
        print(det_answer[:300])

    # --- Step 2: Optional neural language polishing ---
    # Only attempt if: (a) model is loaded, (b) answer is a plain one-liner
    # that might benefit from better phrasing, (c) document is not too long.
    # For structured answers (bullet lists, section headers) skip neural —
    # the structure is intentional and the model would flatten it.
    is_structured = bool(re.search(r"^(\*\*|•|-|\d+\.)", det_answer, re.M))
    is_short_factual = len(det_answer) < 300 and not is_structured

    if is_short_factual and _ensure_llm():
        # Build a minimal prompt for language polishing only
        polish_prompt = (
            f"<TASK=DOCUMENT_ANSWER>\n"
            f"Document: {rep.filename}\n"
            f"Document type: {rep.document_type}\n"
            f"Grounded answer: {det_answer}\n\n"
            f"Question: {query}\n\n"
            f"Rephrase the grounded answer naturally in 1-3 sentences."
        )
        neural_out = _scratch_llm.generate_for_document(
            polish_prompt, max_new=150, temperature=0.4, top_p=0.9
        )

        if neural_out and len(neural_out.strip()) >= 15:
            BAD = ["i don't know", "i cannot", "as an ai", "i was trained",
                   "no information", "not able to", "i do not have"]
            if not any(b in neural_out.lower() for b in BAD):
                # Verify: neural output must be grounded in the deterministic answer
                # (We verify against det_answer, not raw chunks, since det_answer
                # is already verified against the document representation)
                ok, _ = _quick_verify(neural_out, det_answer + " " + rep.global_summary)
                if ok:
                    if debug:
                        print(f"[AttachmentHandler] Neural polish accepted: {neural_out[:100]}")
                    return neural_out.strip()

    return det_answer


def _quick_verify(generated: str, ground_truth: str) -> Tuple[bool, str]:
    """
    Lightweight verification: generated text shares vocabulary with ground_truth.
    More permissive than document_verifier.py since ground_truth is already
    the deterministic answer (not raw retrieval chunks).
    """
    gen_words = set(re.findall(r"[a-zA-Z]{4,}", generated.lower()))
    ref_words = set(re.findall(r"[a-zA-Z]{4,}", ground_truth.lower()))
    if not gen_words:
        return False, "no content"
    overlap = len(gen_words & ref_words) / len(gen_words)
    if overlap < 0.3:
        return False, f"low overlap ({overlap:.2f})"
    return True, "ok"


# ══════════════════════════════════════════════════════════════════════════════
# DEBUG MODE
# ══════════════════════════════════════════════════════════════════════════════

def _debug_pipeline(rep: DocumentRepresentation, query: str, raw_text: str) -> str:
    """
    Returns a debug trace of the full document understanding pipeline.
    Activated by prefixing the query with "DEBUG:" in development.
    """
    lines = [
        "═══ DOCUMENT UNDERSTANDING DEBUG ═══",
        f"File        : {rep.filename}",
        f"Type        : {rep.document_type}",
        f"Main topic  : {rep.main_topic}",
        f"Purpose     : {rep.purpose[:120]}",
        f"Word count  : {rep.total_words}",
        f"Sections    : {len(rep.sections)}",
        "",
        "─── SECTIONS ───",
    ]
    for s in rep.sections:
        lines.append(f"  [{s.order}] {s.title}  (importance={s.importance:.2f}, {s.word_count}w)")
        if s.summary:
            lines.append(f"      Summary: {s.summary[:100]}...")
        if s.facts:
            lines.append(f"      Facts  : {s.facts[0][:80]}")
        if s.conclusions:
            lines.append(f"      Concl  : {s.conclusions[0][:80]}")
    lines += [
        "",
        "─── DOCUMENT-LEVEL ───",
        f"  Key facts   : {len(rep.key_facts)}",
        f"  Key claims  : {len(rep.key_claims)}",
        f"  Evidence    : {len(rep.evidence)}",
        f"  Conclusions : {len(rep.conclusions)}",
        f"  Entities    : {rep.entities[:6]}",
        "",
        "─── GLOBAL SUMMARY ───",
        rep.global_summary[:400],
        "",
        "─── SHORT SUMMARY ───",
        rep.short_summary,
        "",
        "─── KEY POINTS ───",
    ]
    for pt in rep.key_points[:5]:
        lines.append(f"  • {pt}")
    lines += [
        "",
        "─── DOCUMENT MAP ───",
        rep.document_map,
        "",
        "─── ANSWER FOR QUERY ───",
        f"Query: {query}",
        "",
        answer_from_representation(rep, query, raw_text=raw_text, debug=True),
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def handle_attachment(uploaded_file, query: str, chatbot=None,
                       session_id: str = _DEFAULT_SESSION,
                       debug: bool = False) -> dict:
    """
    Process an uploaded file and return a result dict:
      { answer, intent, elapsed, filename }

    `uploaded_file` is a Django InMemoryUploadedFile or TemporaryUploadedFile.
    `session_id` scopes the document memory to one caller.
    `debug` enables pipeline tracing (for development use only).
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
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext)
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
        rep, raw_text = _ingest_document(tmp_path, filename, debug=debug)

        # Store semantic representation in document memory
        _store_document_repr(rep, raw_text, session_id=session_id)

        effective_query = query or "Summarise this document."

        # Debug mode (development only)
        if debug or effective_query.lower().startswith("debug:"):
            clean_query = re.sub(r"^debug:\s*", "", effective_query, flags=re.I).strip()
            answer = _debug_pipeline(rep, clean_query or "Summarise this document.", raw_text)
        else:
            answer = _build_doc_answer(rep, effective_query, raw_text=raw_text, debug=debug)

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
                                 session_id: str = _DEFAULT_SESSION,
                                 debug: bool = False) -> Optional[dict]:
    """
    Answer a follow-up question using the previously uploaded document's
    semantic memory, without requiring the user to re-upload the file.

    Returns None if there is no stored document for this session.

    KEY IMPROVEMENT over the previous version:
    Follow-up questions are answered from DocumentRepresentation (semantic
    memory) — not by re-running TF-IDF on raw text every time. This means:
    - "What is the main contribution?" → pulls from rep.key_claims
    - "What evidence supports it?" → pulls from rep.evidence
    - "Explain the conclusion" → pulls from section_repr for 'conclusion'
    All of these are O(1) memory lookups, not re-extraction.
    """
    stored = _DOC_MEMORY.get(session_id)
    if stored is None:
        return None

    rep, raw_text = stored
    t0 = time.time()

    if debug or query.lower().startswith("debug:"):
        clean_query = re.sub(r"^debug:\s*", "", query, flags=re.I).strip() or "Summarise this document."
        answer = _debug_pipeline(rep, clean_query, raw_text)
    else:
        answer = _build_doc_answer(rep, query, raw_text=raw_text, debug=debug)

    return {
        "answer":   answer,
        "intent":   "document_followup",
        "elapsed":  round(time.time() - t0, 3),
        "filename": rep.filename,
    }
