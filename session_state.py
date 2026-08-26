"""
session_state.py
═══════════════════════════════════════════════════════════════════════════════
Structured per-session state replacing the bare history = [] pattern.

Resolves pronouns and context references using typed fields (not keyword
search over raw history strings). Each session is isolated — document
context from session A can never leak into session B.

Usage
─────
    state = ConversationState()

    # After answering a sales query:
    state.set_sales_context(
        customer="Rahul", enquiry="ENQ042",
        dataset="Feedback", intent="show_feedback", records=[...]
    )

    # Later, resolve "his phone number"
    if state.active_customer:
        # use state.active_customer to look up

    # After a document upload:
    state.set_document(doc)

    # Check if current query is document follow-up:
    if state.has_document and state.is_document_query(query):
        ...
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ─────────────────────────────────── typing helpers ──────────────────────────

PRONOUN_SET = {"he", "she", "they", "his", "her", "their", "him", "them",
               "it", "its", "this", "that", "these", "those"}

_DOC_SIGNALS = {
    "document", "file", "pdf", "docx", "csv", "sheet", "image", "photo",
    "picture", "upload", "uploaded", "attachment", "above", "this document",
    "the file", "read it", "summarize it", "summarise it", "explain this",
    "translate this",
}

_SALES_SIGNALS = {
    "enquiry", "enquiries", "appointment", "feedback", "customer", "vehicle",
    "rating", "status", "cancelled", "completed", "scheduled",
}


# ─────────────────────────────────── core state ──────────────────────────────

@dataclass
class SalesContext:
    customer: Optional[str] = None
    customer_id: Optional[str] = None
    enquiry_id: Optional[str] = None
    appointment_id: Optional[str] = None
    feedback_id: Optional[str] = None
    vehicle: Optional[str] = None
    city: Optional[str] = None
    dataset: Optional[str] = None      # "Enquiry" | "Appointment" | "Feedback"
    intent: Optional[str] = None
    last_result_records: List[Dict] = field(default_factory=list)
    last_query: str = ""
    last_facts: str = ""
    updated_at: float = field(default_factory=time.time)


@dataclass
class DocumentContext:
    """Per-session document context. Isolated from all other sessions."""
    document: Any = None         # ParsedDocument (or None)
    filename: str = ""
    summary: str = ""
    uploaded_at: float = 0.0
    turns_since_upload: int = 0
    max_turns: int = 10          # expire after 10 non-document turns

    @property
    def is_active(self) -> bool:
        return self.document is not None

    def tick(self) -> None:
        """Call when a non-document turn passes."""
        self.turns_since_upload += 1
        if self.turns_since_upload > self.max_turns:
            self.expire()

    def expire(self) -> None:
        self.document = None
        self.filename = ""
        self.summary = ""
        self.turns_since_upload = 0


@dataclass
class ConversationState:
    session_id: str = ""
    sales: SalesContext = field(default_factory=SalesContext)
    document: DocumentContext = field(default_factory=DocumentContext)
    history: List[Dict[str, str]] = field(default_factory=list)   # [{role, content}]
    last_intent: str = ""
    last_operation: str = ""
    created_at: float = field(default_factory=time.time)
    turn_count: int = 0

    # ── sales context setters ─────────────────────────────────────────────────
    def set_sales_context(
        self,
        *,
        customer: Optional[str] = None,
        enquiry_id: Optional[str] = None,
        vehicle: Optional[str] = None,
        city: Optional[str] = None,
        dataset: Optional[str] = None,
        intent: Optional[str] = None,
        records: Optional[List[Dict]] = None,
        facts: str = "",
        query: str = "",
    ) -> None:
        sc = self.sales
        if customer:
            sc.customer = customer
        if enquiry_id:
            sc.enquiry_id = enquiry_id
        if vehicle:
            sc.vehicle = vehicle
        if city:
            sc.city = city
        if dataset:
            sc.dataset = dataset
        if intent:
            sc.intent = intent
        if records is not None:
            sc.last_result_records = records[:50]    # cap to avoid huge state
        if facts:
            sc.last_facts = facts
        if query:
            sc.last_query = query
        sc.updated_at = time.time()

    # ── document context ──────────────────────────────────────────────────────
    def set_document(self, doc, filename: str = "", summary: str = "") -> None:
        self.document = DocumentContext(
            document=doc,
            filename=filename or getattr(doc, "filename", ""),
            summary=summary or getattr(doc, "summary", ""),
            uploaded_at=time.time(),
        )

    @property
    def has_document(self) -> bool:
        return self.document.is_active

    # ── pronoun / reference resolution ───────────────────────────────────────
    def resolve_references(self, query: str) -> str:
        """
        Replace ambiguous pronouns with the known entity from session state
        where unambiguous. Returns the (possibly enriched) query string.
        """
        q_low = query.lower()
        tokens = set(re.findall(r"[a-zA-Z]+", q_low))

        # If query contains only pronouns (no names/IDs), add the known context
        if tokens & PRONOUN_SET and self.sales.customer:
            if not re.search(r"\b[A-Z][a-z]{2,}\b", query):   # no capitalised word
                return query + f" [resolved: customer={self.sales.customer}]"
        return query

    # ── topic detection ───────────────────────────────────────────────────────
    def classify_query_topic(self, query: str) -> str:
        """Returns DOCUMENT | SALES | AMBIGUOUS | NEW_TOPIC"""
        q = query.lower()
        tokens = set(re.findall(r"[a-zA-Z]+", q))

        doc_score  = len(tokens & _DOC_SIGNALS)
        sales_score = len(tokens & _SALES_SIGNALS)

        # Explicit document signals
        for sig in ("this document", "the file", "the pdf", "the image",
                    "the photo", "read this", "explain this", "summarize"):
            if sig in q:
                doc_score += 3

        if doc_score > 0 and sales_score == 0:
            return "DOCUMENT"
        if sales_score > 0 and doc_score == 0:
            return "SALES"
        if doc_score > 0 and sales_score > 0:
            return "AMBIGUOUS"

        # Short pronoun-heavy messages → likely follow-up on last topic
        if len(tokens) <= 5 and (tokens & PRONOUN_SET):
            if self.last_intent.startswith("doc"):
                return "DOCUMENT"
            return "SALES"

        return "NEW_TOPIC"

    def is_document_query(self, query: str) -> bool:
        return self.classify_query_topic(query) in ("DOCUMENT", "AMBIGUOUS")

    # ── turn lifecycle ────────────────────────────────────────────────────────
    def record_turn(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        if len(self.history) > 40:    # keep last 40 turns
            self.history = self.history[-40:]
        self.turn_count += 1

    def on_non_document_turn(self) -> None:
        """Called after every turn that is not a document query."""
        self.document.tick()

    def to_debug_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "last_intent": self.last_intent,
            "last_operation": self.last_operation,
            "sales": {
                "customer": self.sales.customer,
                "enquiry_id": self.sales.enquiry_id,
                "dataset": self.sales.dataset,
                "vehicle": self.sales.vehicle,
                "city": self.sales.city,
            },
            "document": {
                "active": self.document.is_active,
                "filename": self.document.filename,
                "turns_since_upload": self.document.turns_since_upload,
            },
        }


# ── simple in-process session store (for development / single-process) ────────

class SessionStore:
    """
    Thread-safe in-memory session store.
    In production, back this with Redis or Django session storage.
    """

    def __init__(self):
        import threading
        self._lock = threading.Lock()
        self._sessions: Dict[str, ConversationState] = {}
        self._ttl = 1800     # 30-minute idle expiry

    def get(self, session_id: str) -> ConversationState:
        with self._lock:
            self._evict_stale()
            if session_id not in self._sessions:
                s = ConversationState(session_id=session_id)
                self._sessions[session_id] = s
            return self._sessions[session_id]

    def _evict_stale(self) -> None:
        now = time.time()
        stale = [sid for sid, s in self._sessions.items()
                 if (now - s.created_at) > self._ttl]
        for sid in stale:
            del self._sessions[sid]

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


SESSION_STORE = SessionStore()
