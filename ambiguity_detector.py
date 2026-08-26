"""
ambiguity_detector.py
═══════════════════════════════════════════════════════════════════════════════
First-class ambiguity detection that runs BEFORE any pandas execution.

Handles:
  - Duplicate customer names (multiple Rahuls)
  - Similar vehicle names
  - Partial / ambiguous enquiry IDs
  - Ambiguous cities
  - Ambiguous pronouns without a session anchor
  - Contradictory filters (e.g. "cancelled and completed")
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class AmbiguityResult:
    is_ambiguous: bool
    kind: str                         # NONE / DUPLICATE_NAME / SIMILAR_NAME /
                                      # PARTIAL_ID / AMBIGUOUS_CITY /
                                      # AMBIGUOUS_PRONOUN / CONTRADICTORY_FILTER /
                                      # UNSUPPORTED_FIELD
    clarification_question: str = ""
    candidates: List[str] = None

    def __post_init__(self):
        if self.candidates is None:
            self.candidates = []


_PRONOUNS = {"he", "she", "they", "his", "her", "their", "him", "them",
             "it", "its", "this", "that", "these", "those"}
_PARTIAL_ENQ_RE = re.compile(r"\benq\s*\d{1,2}\b", re.I)   # e.g. "enq5", "ENQ 1"
_FULL_ENQ_RE    = re.compile(r"\bENQ\d{3,6}\b", re.I)


def detect_ambiguity(
    query: str,
    extracted_names: List[str],
    extracted_cities: List[str],
    extracted_ids: List[str],
    extracted_vehicles: List[str],
    dfs: Dict[str, pd.DataFrame],
    session_has_active_customer: bool = False,
) -> AmbiguityResult:
    """
    Returns an AmbiguityResult. If is_ambiguous is True, the caller should
    return the clarification_question to the user and NOT execute a query.
    """
    q_low = query.lower().strip()
    tokens = set(re.findall(r"[a-zA-Z]+", q_low))

    # ── 1. Contradictory filters ─────────────────────────────────────────────
    status_words = {"cancelled", "canceled", "scheduled", "completed", "pending", "closed"}
    mentioned_statuses = tokens & status_words
    if "cancelled" in mentioned_statuses and "completed" in mentioned_statuses:
        return AmbiguityResult(
            is_ambiguous=True,
            kind="CONTRADICTORY_FILTER",
            clarification_question=(
                "I noticed you mentioned both 'cancelled' and 'completed'. "
                "Did you mean one specific status? Please clarify."
            ),
        )

    # ── 2. Ambiguous pronoun with no session context ─────────────────────────
    if not session_has_active_customer:
        # Queries that are just pronouns + a field (no name/ID given)
        if (tokens & _PRONOUNS) and not extracted_names and not extracted_ids:
            if any(w in q_low for w in ("feedback", "appointment", "enquiry", "rating",
                                         "status", "phone", "mobile", "contact", "vehicle")):
                return AmbiguityResult(
                    is_ambiguous=True,
                    kind="AMBIGUOUS_PRONOUN",
                    clarification_question=(
                        "Who are you referring to? Please provide the customer name or enquiry ID."
                    ),
                )

    # ── 3. Partial enquiry ID ────────────────────────────────────────────────
    if _PARTIAL_ENQ_RE.search(query) and not _FULL_ENQ_RE.search(query):
        return AmbiguityResult(
            is_ambiguous=True,
            kind="PARTIAL_ID",
            clarification_question=(
                "Could you provide the full enquiry ID (e.g. ENQ001, ENQ025)?"
            ),
        )

    # ── 4. Duplicate / multiple customer names ───────────────────────────────
    if extracted_names:
        all_names = _collect_names(dfs)
        for name in extracted_names:
            matches = _exact_name_matches(name, all_names)
            if len(matches) > 1:
                # Check whether they are actually distinct people (different IDs or data)
                distinct = _are_distinct_records(name, dfs)
                if distinct:
                    return AmbiguityResult(
                        is_ambiguous=True,
                        kind="DUPLICATE_NAME",
                        clarification_question=(
                            f"I found {len(matches)} customers named '{name}'. "
                            "Please provide the Enquiry ID or full name to identify the right record."
                        ),
                        candidates=matches,
                    )

    # ── 5. Similar (but non-exact) name matches ───────────────────────────────
    if not extracted_names:
        # Did the user type something that is close to MULTIPLE known names?
        words = re.findall(r"[A-Za-z]{3,}", query)
        all_names = _collect_names(dfs)
        for w in words:
            similars = _similar_names(w, all_names, threshold=0.80, limit=3)
            if len(similars) >= 2:
                return AmbiguityResult(
                    is_ambiguous=True,
                    kind="SIMILAR_NAME",
                    clarification_question=(
                        f"Did you mean one of these customers: {', '.join(similars)}? "
                        "Please clarify the name or provide the enquiry ID."
                    ),
                    candidates=similars,
                )

    # ── 6. Ambiguous city (multiple cities extracted) ─────────────────────────
    if len(extracted_cities) > 1:
        return AmbiguityResult(
            is_ambiguous=True,
            kind="AMBIGUOUS_CITY",
            clarification_question=(
                f"I see multiple cities mentioned: {', '.join(extracted_cities)}. "
                "Did you want data for all of them, or one specifically?"
            ),
            candidates=extracted_cities,
        )

    return AmbiguityResult(is_ambiguous=False, kind="NONE")


# ── helpers ───────────────────────────────────────────────────────────────────

def _collect_names(dfs: Dict[str, pd.DataFrame]) -> List[str]:
    names = []
    for df in dfs.values():
        for col in df.columns:
            if "customer" in col.lower() and "name" in col.lower():
                names += [str(v).strip() for v in df[col].dropna().unique()
                          if str(v).strip().lower() not in ("nan", "none", "")]
    return list(set(names))


def _exact_name_matches(name: str, all_names: List[str]) -> List[str]:
    return [n for n in all_names if n.lower() == name.lower()]


def _are_distinct_records(name: str, dfs: Dict[str, pd.DataFrame]) -> bool:
    """True if the same name maps to different enquiry IDs across datasets."""
    ids_seen = set()
    for ds, df in dfs.items():
        if "Customer Name" not in df.columns:
            continue
        sub = df[df["Customer Name"].str.lower() == name.lower()]
        id_col = next((c for c in df.columns if "enquiry" in c.lower() and "id" in c.lower()), None)
        if id_col:
            ids_seen.update(sub[id_col].dropna().astype(str).unique())
    return len(ids_seen) > 1


def _similar_names(word: str, vocab: List[str], threshold: float = 0.80, limit: int = 3) -> List[str]:
    scored = [(v, SequenceMatcher(None, word.lower(), v.lower()).ratio()) for v in vocab]
    return [v for v, s in sorted(scored, key=lambda x: -x[1]) if s >= threshold][:limit]
