"""
bot_identity.py
────────────────────────────────────────────────────────────────────────────
Handles "who/what are you", "what's your name", "who made you" style
questions with a fixed, branded answer — instead of letting them fall
through to the retrieval + LLM pipeline.

Why this exists: those questions don't contain any sales vocabulary, so
they'd either get refused as out-of-domain, or (worse) slip through via
the "short clarification of the previous turn" fallback in domain_guard
and get treated as a real data query — the retriever then hands the LLM
a handful of random, unrelated customer rows and it hallucinates an
answer out of them. Intercepting identity questions here, first, avoids
both problems and lets the bot introduce itself consistently.

── Customize your branding here ────────────────────────────────────────────
Everything a user will actually see lives in the three constants below.
"""

import re

BOT_NAME    = "Platinum Sales Intelligence Assistant"
BOT_OWNER   = "Platinum Software"
BOT_TAGLINE = (
    "an AI-powered assistant for sales enquiries, appointments, customer "
    "feedback, and uploaded document/image analysis"
)

IDENTITY_RESPONSE = (
    f"I'm the {BOT_NAME}, built by {BOT_OWNER}. I'm {BOT_TAGLINE} — "
    "ask me things like \"who gave bad feedback\", \"show ENQ001\", "
    "\"cancelled appointments\", or upload a document/image and I'll "
    "read it for you."
)

_IDENTITY_PATTERNS = [
    r"\bwho are you\b",
    r"\bwhat are you\b",
    r"\bwhats? your name\b",
    r"\bwhat is your name\b",
    r"\bwho (made|built|created|owns|developed) you\b",
    r"\bwhos (your |is your )?(creator|owner|developer|maker)\b",
    r"\bare you (a )?(chatgpt|gpt|claude|bot|robot|ai|human)\b",
    r"\bintroduce yourself\b",
    r"\btell me about yourself\b",
    r"\bwhat (can|do) you do\b",
    r"\bwhich (company|platform|model) (are you|is this|powers? you)\b",
    r"\bpowered by\b",
]
_IDENTITY_RE = re.compile("|".join(_IDENTITY_PATTERNS), re.I)


def is_identity_question(query: str) -> bool:
    """True if `query` is asking who/what the bot is, rather than a
    sales/data/document question."""
    if not query:
        return False
    # Strip apostrophes so "what's your name" / "who's your creator" line
    # up with the same patterns as "whats your name" / "whos your creator".
    normalized = query.strip().replace("’", "'").replace("'", "")
    return bool(_IDENTITY_RE.search(normalized))


def identity_answer() -> str:
    return IDENTITY_RESPONSE
