"""
domain_guard.py
═══════════════════════════════════════════════════════════════════════════════
Restricts the chatbot to sales / customer / appointment / feedback /
uploaded-document / uploaded-image / company-related questions.

ALL Qwen / transformers LLM classification removed.
The three-layer check is now:
  1. Exact keyword overlap     (fast)
  2. Fuzzy / stemmed overlap   (catches word-form variants)
  3. Entity registry match     (customer names / cities / vehicle models
                                 registered at startup by api.py)

No LLM fallback — the entity registry + fuzzy matching covers the cases
that the old LLM classifier was used for, without any model loading.
"""

from __future__ import annotations

import random
import re
from typing import Dict

# ── Sales-domain vocabulary ───────────────────────────────────────────────────
SALES_DOMAIN_WORDS = {
    "enquiry", "enquiries", "inquiry", "lead", "leads", "record", "records",
    "customer", "customers", "client", "clients", "profile", "case", "ticket",
    "person", "persons", "people", "individual", "individuals",
    "feedback", "review", "reviews", "rating", "ratings", "comment", "opinion",
    "experience", "satisfaction", "complaint", "complaints", "sentiment",
    "appointment", "appointments", "meeting", "visit", "booking", "booked",
    "schedule", "scheduled", "slot", "session", "confirmed", "cancelled",
    "cancel", "cancellation", "completed", "status", "state", "progress",
    "pending", "closed", "open", "active", "contact", "phone", "mobile",
    "email", "reach", "call", "vehicle", "vehicles", "car", "cars", "bike",
    "bikes", "model", "automobile", "product", "purchase", "buying",
    "payment", "paid", "pay", "loan", "cash", "emi", "finance", "amount",
    "test ride", "test drive", "trial", "demo", "city", "location", "region",
    "area", "new lead", "returning", "existing", "repeat", "sales", "sale",
    "dataset", "data", "database", "company", "showroom", "dealer",
    "dealership", "revenue", "target", "conversion", "quotation", "invoice",
    "enq", "eq",
}

# ── Document / image upload vocabulary ───────────────────────────────────────
DOCUMENT_DOMAIN_WORDS = {
    "document", "documents", "doc", "docs", "file", "files", "pdf", "docx",
    "excel", "xlsx", "csv", "sheet", "spreadsheet", "image", "images",
    "picture", "pictures", "photo", "photos", "png", "jpg", "jpeg", "webp",
    "upload", "uploaded", "attach", "attached", "attachment", "scan",
    "scanned", "ocr", "extract", "invoice", "chart", "graph", "table",
    "board", "screenshot", "read this", "describe this", "analyze this",
    "summarize this", "summarise this", "what does this say",
}

_GREETING_WORDS = {
    "hi", "hello", "hey", "hii", "hiii", "good morning", "good afternoon",
    "good evening", "thanks", "thank you", "ok", "okay", "bye", "goodbye",
}

REFUSAL_MESSAGES = [
    "I'm designed to answer only sales-related questions or analyze uploaded files. "
    "Please ask a relevant question.",
    "Please ask questions related to sales, customer data, appointments, feedback, "
    "or uploaded documents.",
    "I can only assist with sales information and uploaded files.",
]

# Track whether the most recent answered turn was a sales-domain answer.
# Keyed by session_id so one customer's/agent's conversation state can never
# leak into another's (the api.py process serves multiple sessions from a
# single global module, so a bare bool here would let session B's "short
# clarification" fall back onto session A's last sales turn).
_last_turn_sales: Dict[str, bool] = {}
_DEFAULT_SESSION = "default"

_HAS_DIGIT_RE = re.compile(r"\d")
_CAP_WORD_RE  = re.compile(r"\b[A-Z][a-z]+\b")

# ── Dynamic entity registry ───────────────────────────────────────────────────
_KNOWN_ENTITY_WORDS: set = set()

_STOPWORDS_FOR_ENTITIES = {
    "the", "and", "of", "for", "a", "an", "in", "on", "at", "to", "is",
}


def register_known_entities(*value_lists) -> None:
    """
    Register free-text values (names, city/state strings, vehicle models)
    so their individual words become recognised in-domain vocabulary.
    """
    global _KNOWN_ENTITY_WORDS
    for values in value_lists:
        if values is None:
            continue
        if isinstance(values, str):
            values = [values]
        for v in values:
            if v is None:
                continue
            sval = str(v).strip()
            if not sval or sval.lower() in ("nan", "none"):
                continue
            for w in re.findall(r"[a-zA-Z]+", sval.lower()):
                if len(w) >= 3 and w not in _STOPWORDS_FOR_ENTITIES:
                    _KNOWN_ENTITY_WORDS.add(w)


def mark_sales_turn(session_id: str = _DEFAULT_SESSION) -> None:
    _last_turn_sales[session_id] = True


def mark_other_turn(session_id: str = _DEFAULT_SESSION) -> None:
    _last_turn_sales[session_id] = False


def _normalize(text: str) -> tuple:
    text_low = text.lower()
    tokens   = set(re.findall(r"[a-zA-Z]+", text_low))
    return tokens, text_low


def _stem_prefix(word: str) -> str:
    """
    Dependency-free stem: strip a common suffix then take a short prefix.
    "enquired"/"enquiry"/"enquiries"/"enquiring" all → "enqui".
    """
    for suf in ("ations", "ation", "ing", "ies", "ied", "ers", "er",
                "es", "ed", "s"):
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            word = word[: -len(suf)]
            break
    return word[:5]


_SALES_PREFIXES = {
    _stem_prefix(w) for w in SALES_DOMAIN_WORDS if len(w) >= 5
}


def _fuzzy_sales_hit(tokens: set) -> bool:
    for tok in tokens:
        if len(tok) < 5:
            continue
        if _stem_prefix(tok) in _SALES_PREFIXES:
            return True
    return False


def has_sales_keywords(query: str) -> bool:
    """
    True if the query contains vocabulary clearly about the sales domain,
    including fuzzy word-form matches and known dataset entities.
    Exposed for use by attachment_handler.
    """
    tokens, text_low = _normalize(query)
    if tokens & SALES_DOMAIN_WORDS:
        return True
    for phrase in ("test ride", "test drive", "new lead"):
        if phrase in text_low:
            return True
    if _fuzzy_sales_hit(tokens):
        return True
    if _KNOWN_ENTITY_WORDS and (tokens & _KNOWN_ENTITY_WORDS):
        return True
    return False


def is_in_domain(query: str, has_attachment: bool = False,
                  session_id: str = _DEFAULT_SESSION) -> bool:
    """
    Decide whether `query` is allowed to reach the retrieval pipeline.

    Checks (cheapest first, stop at first hit):
      1. A file is attached to this request.
      2. Short greeting / pleasantry.
      3. Sales-domain vocabulary (exact, fuzzy, or known entity).
      4. Document / image vocabulary.
      5. Short clarification of the previous sales turn.

    No LLM fallback — entity registry + fuzzy matching covers ambiguous cases.
    """
    if has_attachment:
        return True

    tokens, text_low = _normalize(query)

    # Greeting check
    greeting_tokens = {w for phrase in _GREETING_WORDS for w in phrase.split()}
    if tokens & greeting_tokens and len(tokens) <= 3:
        return True

    if has_sales_keywords(query):
        return True

    # Document vocabulary
    doc_hit = bool(tokens & DOCUMENT_DOMAIN_WORDS)
    for phrase in ("read this", "describe this", "analyze this",
                   "analyse this", "summarize this", "summarise this",
                   "what does this say", "what does this document say"):
        if phrase in text_low:
            doc_hit = True
    if doc_hit:
        return True

    # Short clarification of a previous sales answer
    if _last_turn_sales.get(session_id, False) and tokens and (
        len(tokens) <= 4
        or _HAS_DIGIT_RE.search(query)
        or _CAP_WORD_RE.search(query)
    ):
        return True

    return False


def refusal_message() -> str:
    return random.choice(REFUSAL_MESSAGES)
