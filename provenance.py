"""
provenance.py
═══════════════════════════════════════════════════════════════════════════════
Fact provenance tracking and event-chain reasoning.

Every computed fact carries its origin (dataset, column, row index, record ID)
so downstream verification can confirm numbers without re-executing queries.

Event-chain reasoning represents a customer's journey as an ordered sequence:
  ENQUIRY → APPOINTMENT → TEST RIDE → PAYMENT → FEEDBACK

Supports questions like:
  "Who enquired but never had an appointment?"
  "Who had an appointment and later gave a rating below 3?"
  "Who cancelled after enquiring?"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


# ─────────────────────────────────── provenance dataclass ────────────────────

@dataclass
class Fact:
    """A single grounded fact with its data origin."""
    value: object
    dataset: str
    column: str
    record_id: Optional[str] = None   # Enquiry ID or similar
    row_index: Optional[int] = None
    document: Optional[str] = None
    chunk_id: Optional[str] = None

    def to_string(self) -> str:
        parts = [f"{self.value}"]
        parts.append(f"[{self.dataset}.{self.column}")
        if self.record_id:
            parts.append(f" id={self.record_id}")
        parts.append("]")
        return "".join(parts)


@dataclass
class FactSet:
    """Collection of facts with provenance, produced by one query execution."""
    facts: List[Fact] = field(default_factory=list)
    query: str = ""
    operation: str = ""

    def summary_string(self) -> str:
        return " | ".join(f.to_string() for f in self.facts[:10])

    def all_values(self) -> list:
        return [f.value for f in self.facts]

    def numbers(self) -> list:
        out = []
        for f in self.facts:
            try:
                out.append(float(str(f.value).replace(",", "")))
            except (ValueError, TypeError):
                pass
        return out


def extract_facts_from_df(
    df: pd.DataFrame,
    dataset: str,
    id_col: Optional[str],
    value_col: str,
) -> FactSet:
    """Build a FactSet from a filtered DataFrame."""
    fs = FactSet()
    for idx, row in df.iterrows():
        rec_id = str(row[id_col]) if id_col and id_col in row else None
        val = row[value_col] if value_col in row else None
        fs.facts.append(Fact(
            value=val,
            dataset=dataset,
            column=value_col,
            record_id=rec_id,
            row_index=int(idx),
        ))
    return fs


def document_fact(value: str, document: str, chunk_id: str = "") -> Fact:
    return Fact(value=value, dataset="document", column="text",
                document=document, chunk_id=chunk_id)


# ─────────────────────────────────── event-chain reasoning ───────────────────

@dataclass
class CustomerEvent:
    event_type: str           # ENQUIRY | APPOINTMENT | FEEDBACK
    customer_name: str
    enquiry_id: Optional[str]
    status: Optional[str]
    rating: Optional[float]
    dataset: str


@dataclass
class CustomerTimeline:
    customer_name: str
    events: List[CustomerEvent] = field(default_factory=list)

    @property
    def has_enquiry(self) -> bool:
        return any(e.event_type == "ENQUIRY" for e in self.events)

    @property
    def has_appointment(self) -> bool:
        return any(e.event_type == "APPOINTMENT" for e in self.events)

    @property
    def has_feedback(self) -> bool:
        return any(e.event_type == "FEEDBACK" for e in self.events)

    @property
    def appointment_cancelled(self) -> bool:
        return any(e.event_type == "APPOINTMENT" and
                   e.status and "cancel" in e.status.lower()
                   for e in self.events)

    @property
    def appointment_completed(self) -> bool:
        return any(e.event_type == "APPOINTMENT" and
                   e.status and "complet" in e.status.lower()
                   for e in self.events)

    @property
    def min_rating(self) -> Optional[float]:
        ratings = [e.rating for e in self.events if e.rating is not None]
        return min(ratings) if ratings else None


def build_timelines(dfs: Dict[str, pd.DataFrame]) -> Dict[str, CustomerTimeline]:
    """Build one CustomerTimeline per customer, keyed by lowercase name."""
    timelines: Dict[str, CustomerTimeline] = {}

    def _get(name: str) -> CustomerTimeline:
        k = name.lower().strip()
        if k not in timelines:
            timelines[k] = CustomerTimeline(customer_name=name)
        return timelines[k]

    # Enquiry events
    enq_df = dfs.get("Enquiry")
    if enq_df is not None:
        id_col = next((c for c in enq_df.columns if "enquiry" in c.lower() and "id" in c.lower()), None)
        status_col = next((c for c in enq_df.columns if "status" in c.lower()), None)
        for _, row in enq_df.iterrows():
            name = str(row.get("Customer Name", "")).strip()
            if not name or name.lower() in ("nan", "none"):
                continue
            t = _get(name)
            t.events.append(CustomerEvent(
                event_type="ENQUIRY",
                customer_name=name,
                enquiry_id=str(row[id_col]) if id_col else None,
                status=str(row[status_col]) if status_col else None,
                rating=None,
                dataset="Enquiry",
            ))

    # Appointment events
    apt_df = dfs.get("Appointment")
    if apt_df is not None:
        id_col = next((c for c in apt_df.columns if "enquiry" in c.lower() and "id" in c.lower()), None)
        for _, row in apt_df.iterrows():
            name = str(row.get("Customer Name", "")).strip()
            if not name or name.lower() in ("nan", "none"):
                continue
            t = _get(name)
            t.events.append(CustomerEvent(
                event_type="APPOINTMENT",
                customer_name=name,
                enquiry_id=str(row[id_col]) if id_col else None,
                status=str(row.get("Status", "")),
                rating=None,
                dataset="Appointment",
            ))

    # Feedback events
    fb_df = dfs.get("Feedback")
    if fb_df is not None:
        id_col = next((c for c in fb_df.columns if "enquiry" in c.lower() and "id" in c.lower()), None)
        for _, row in fb_df.iterrows():
            name = str(row.get("Customer Name", "")).strip()
            if not name or name.lower() in ("nan", "none"):
                continue
            rating = None
            try:
                rating = float(row.get("Rating", ""))
            except (ValueError, TypeError):
                pass
            t = _get(name)
            t.events.append(CustomerEvent(
                event_type="FEEDBACK",
                customer_name=name,
                enquiry_id=str(row[id_col]) if id_col else None,
                status=None,
                rating=rating,
                dataset="Feedback",
            ))

    return timelines


def query_event_chain(
    timelines: Dict[str, CustomerTimeline],
    query: str,
) -> Optional[str]:
    """
    Answer event-chain questions using the pre-built timelines.
    Returns a facts string, or None if not an event-chain question.
    """
    q = query.lower()

    # "enquired but never had an appointment"
    if ("enquir" in q) and ("never" in q or "no appointment" in q or "without appointment" in q) and "appoint" in q:
        matches = [t.customer_name for t in timelines.values()
                   if t.has_enquiry and not t.has_appointment]
        if matches:
            return (f"Computed event-chain fact: {len(matches)} customer(s) enquired but never "
                    f"had an appointment: {', '.join(sorted(matches)[:20])}.")
        return "Computed event-chain fact: All customers who enquired also had an appointment."

    # "cancelled after enquiring" / "who cancelled"
    if ("cancel" in q) and ("enquir" in q or "after" in q):
        matches = [t.customer_name for t in timelines.values()
                   if t.has_enquiry and t.appointment_cancelled]
        if matches:
            return (f"Computed event-chain fact: {len(matches)} customer(s) enquired then "
                    f"cancelled their appointment: {', '.join(sorted(matches)[:20])}.")
        return "Computed event-chain fact: No customers cancelled after enquiring."

    # "appointment and gave negative/low feedback"
    if "appoint" in q and ("negative" in q or "bad" in q or "below" in q or "low" in q or "rating" in q):
        threshold_m = re.search(r"below\s+(\d)", q) or re.search(r"less than\s+(\d)", q)
        threshold = float(threshold_m.group(1)) if threshold_m else 3.0
        matches = [t.customer_name for t in timelines.values()
                   if t.has_appointment and t.min_rating is not None and t.min_rating < threshold]
        if matches:
            return (f"Computed event-chain fact: {len(matches)} customer(s) had an appointment "
                    f"and gave a rating below {threshold}: {', '.join(sorted(matches)[:20])}.")
        return f"Computed event-chain fact: No customers had an appointment and rated below {threshold}."

    # "test ride but negative feedback" → appointment_completed + low rating
    if ("test ride" in q or "test drive" in q) and ("negative" in q or "bad" in q or "below" in q):
        matches = [t.customer_name for t in timelines.values()
                   if t.appointment_completed and t.min_rating is not None and t.min_rating <= 2]
        if matches:
            return (f"Computed event-chain fact: {len(matches)} customer(s) completed a test ride "
                    f"and gave negative feedback: {', '.join(sorted(matches)[:20])}.")
        return "Computed event-chain fact: No customers completed a test ride and gave negative feedback."

    return None
