"""
query_engine.py
════════════════════════════════════════════════════════════════════════════
Structured Query Understanding → Query Planning → Multi-hop Reasoning.

This module sits IN FRONT of the existing TF-IDF pipeline in test.py.
It only activates for queries that need real data OPERATIONS — counts,
averages, top-k, comparisons, or conditions that span more than one
dataset ("negative feedback AND cancelled appointment"). Those are exactly
the cases plain TF-IDF row-retrieval cannot do reliably: it can find rows
that *look* relevant, but it can't compute "127", and it can't intersect
two CSVs.

For everything else (single lookups, "status of ENQ001", "show feedback
for Rahul", simple filters) this engine deliberately returns None so the
existing, already-tuned detect_intent / IntelligentRetriever / build_answer
pipeline in test.py runs completely unchanged. Nothing that already works
is touched.

IMPORTANT DATA FACT (verified against the actual CSVs, not assumed):
  `Enquiry ID` is NOT a reliable key across the three files — only ~13%
  of rows agree on customer name for the same ID (there are only 8
  distinct customers, and IDs are assigned independently per file). So
  multi-hop joins here key on **Customer Name** (fuzzy-normalised),
  which is the only consistent relational key across the datasets.
  ID-based lookups still work fine for WITHIN-one-file questions.

Design goals (per the "no LLM invents facts" requirement):
  * All filtering / counting / averaging / joining is done with pandas.
  * The LLM (via test.py's _generate_phrase) is only ever used to phrase
    the already-computed `facts` string — never to produce the numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ═══════════════════════════════════════════════════════════ ENTITIES ═════

@dataclass
class Entities:
    enquiry_ids: List[str] = field(default_factory=list)
    names: List[str] = field(default_factory=list)
    cities: List[str] = field(default_factory=list)
    vehicles: List[str] = field(default_factory=list)
    statuses: List[str] = field(default_factory=list)
    sentiment: Optional[str] = None        # "negative" | "positive"
    rating_op: Optional[Tuple[str, float]] = None   # ("<", 3) / (">=", 4) ...
    numbers: List[float] = field(default_factory=list)
    limit: Optional[int] = None
    group_by: Optional[str] = None
    confidence: float = 1.0


@dataclass
class QueryPlan:
    operation: str                          # COUNT / AVERAGE / SUM / MIN / MAX /
                                             # TOP_K / BOTTOM_K / FILTER / LIST /
                                             # INTERSECTION / UNION / DIFFERENCE /
                                             # COMPARE / GROUP_BY / MULTI_HOP / NONE
    datasets: List[str] = field(default_factory=list)
    combine: str = "intersection"           # intersection | union | difference
    confidence: float = 0.0
    trace: List[str] = field(default_factory=list)


@dataclass
class EngineResult:
    plan: QueryPlan
    entities: Entities
    facts: str
    matched_names: List[str]
    row_count: int
    confidence: float


# ═══════════════════════════════════════════════════ VOCAB (built from CSVs)

_NEGATIVE_WORDS = {"bad", "poor", "negative", "unhappy", "dissatisfied",
                    "worst", "complaint", "complaints", "disappointed",
                    "terrible", "unsatisfied"}
_POSITIVE_WORDS = {"good", "great", "positive", "happy", "satisfied",
                    "best", "excellent", "pleased"}

_STATUS_VOCAB = {
    "cancelled": "Cancelled", "canceled": "Cancelled", "cancel": "Cancelled",
    "scheduled": "Scheduled", "schedule": "Scheduled",
    "completed": "Completed", "complete": "Completed",
    "booked": "Booked", "contacted": "Contacted",
    "pending": "Pending", "closed": "Closed",
}

_AGG_TRIGGERS = {
    "count":      {"how many", "count", "number of", "total number", "no. of", "no of"},
    "average":    {"average", "avg", "mean"},
    "sum":        {"sum", "total"},
    "max":        {"highest", "maximum", "most", "top rated", "best"},
    "min":        {"lowest", "minimum", "least", "worst"},
    "compare":    {"compare", " vs ", " versus ", "difference between"},
}

# Fields whose name literally contains the word "number" (phone / contact /
# mobile / account / registration number, etc.) — asking for THESE is a
# single-record lookup, not a COUNT aggregation, even though the phrase
# "... number of <name>" superficially matches the "number of" trigger.
_NUMBER_FIELD_RE = re.compile(
    r"\b(phone|contact|mobile|cell|whatsapp|account|registration|reg|"
    r"vehicle|license|licence|order|invoice|booking)\s+number\b", re.I)


def _has_count_trigger(q: str) -> bool:
    """True only if the query is really asking for a COUNT, not asking
    for a specific *_number field (phone number, contact number, ...)."""
    if _NUMBER_FIELD_RE.search(q):
        # e.g. "what is the phone number of Karthik" — strip that phrase
        # out before testing the generic triggers, so a query can't
        # accidentally still match via "count/how many" elsewhere.
        stripped = _NUMBER_FIELD_RE.sub(" ", q)
    else:
        stripped = q
    return any(t in stripped for t in _AGG_TRIGGERS["count"])
_TOPK_RE = re.compile(r"\btop\s+(\d+)\b", re.I)
_BOTTOMK_RE = re.compile(r"\bbottom\s+(\d+)\b", re.I)
_RATING_OP_RE = re.compile(
    r"\b(rating|rated|star)s?\s*(below|under|less than|<)\s*(\d+)", re.I)
_RATING_OP_RE2 = re.compile(
    r"\b(rating|rated|star)s?\s*(above|over|greater than|more than|>=|at least)\s*(\d+)", re.I)
_ENQ_ID_RE = re.compile(r"\bENQ\d{2,6}\b", re.I)

_MULTI_HOP_CONNECTORS_AND = {"and also", " and ", "as well as", "both"}
_MULTI_HOP_CONNECTORS_OR = {" or "}
_MULTI_HOP_CONNECTORS_DIFF = {"but not", "without", "never", "did not", "didn't",
                              "who wasn't", "who did not"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def _fuzzy_hit(token: str, vocab: List[str], threshold: float = 0.82) -> Optional[str]:
    best, best_score = None, 0.0
    tl = token.lower()
    for v in vocab:
        score = SequenceMatcher(None, tl, v.lower()).ratio()
        if score > best_score:
            best, best_score = v, score
    return best if best_score >= threshold else None


class KnownVocab:
    """Vocabulary of real values pulled from the actual CSVs, so entity
    matching is grounded in what the data actually contains (not guesses)."""

    def __init__(self, dfs: Dict[str, pd.DataFrame]):
        self.names: List[str] = []
        self.cities: List[str] = []
        self.vehicles: List[str] = []

        for src, df in dfs.items():
            for col in df.columns:
                cl = col.lower()
                if "customer" in cl and "name" in cl:
                    self.names += [str(v).strip() for v in df[col].dropna().unique()]
                if "city" in cl or "state" in cl:
                    for v in df[col].dropna().unique():
                        # "Hyderabad, TS" -> "Hyderabad"
                        city = str(v).split(",")[0].strip()
                        if city:
                            self.cities.append(city)
                if "vehicle" in cl or "model" in cl:
                    self.vehicles += [str(v).strip() for v in df[col].dropna().unique()]

        self.names    = sorted(set(n for n in self.names if n and n.lower() != "nan"))
        self.cities   = sorted(set(c for c in self.cities if c and c.lower() != "nan"))
        self.vehicles = sorted(set(v for v in self.vehicles if v and v.lower() != "nan"))


# ═══════════════════════════════════════════════ ENTITY / INTENT EXTRACTION

def extract_entities(query: str, vocab: KnownVocab) -> Entities:
    q = _norm(query)
    ent = Entities()

    ent.enquiry_ids = [m.upper() for m in _ENQ_ID_RE.findall(query)]

    # sentiment
    tokens = set(re.findall(r"[a-zA-Z]+", q))
    if tokens & _NEGATIVE_WORDS:
        ent.sentiment = "negative"
    elif tokens & _POSITIVE_WORDS:
        ent.sentiment = "positive"

    # rating comparisons ("below 3", "at least 4")
    m = _RATING_OP_RE.search(q)
    if m:
        ent.rating_op = ("<", float(m.group(3)))
    else:
        m2 = _RATING_OP_RE2.search(q)
        if m2:
            ent.rating_op = (">=", float(m2.group(3)))

    # statuses (appointment / enquiry)
    for word, canon in _STATUS_VOCAB.items():
        if word in q:
            ent.statuses.append(canon)
    ent.statuses = sorted(set(ent.statuses))

    # top/bottom K
    mt = _TOPK_RE.search(q)
    mb = _BOTTOMK_RE.search(q)
    if mt:
        ent.limit = int(mt.group(1))
    elif mb:
        ent.limit = int(mb.group(1))

    # cities / vehicles / names — exact substring first, then fuzzy per word
    words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", query)
    for city in vocab.cities:
        if city.lower() in q:
            ent.cities.append(city)
    for veh in vocab.vehicles:
        if veh.lower() in q:
            ent.vehicles.append(veh)
    for name in vocab.names:
        if name.lower() in q and len(name) > 2:
            ent.names.append(name)

    if not ent.cities:
        for w in words:
            hit = _fuzzy_hit(w, vocab.cities)
            if hit and hit not in ent.cities:
                ent.cities.append(hit)
    if not ent.vehicles:
        for w in words:
            hit = _fuzzy_hit(w, vocab.vehicles)
            if hit and hit not in ent.vehicles:
                ent.vehicles.append(hit)
    if not ent.names:
        for w in words:
            hit = _fuzzy_hit(w, vocab.names, threshold=0.8)
            if hit and hit not in ent.names:
                ent.names.append(hit)

    ent.cities = sorted(set(ent.cities))
    ent.vehicles = sorted(set(ent.vehicles))
    ent.names = sorted(set(ent.names))

    ent.numbers = [float(n) for n in re.findall(r"\b\d+(?:\.\d+)?\b", q)]

    if "by city" in q or "per city" in q or "each city" in q:
        ent.group_by = "city"
    elif "by vehicle" in q or "per vehicle" in q or "each vehicle" in q or "by model" in q:
        ent.group_by = "vehicle"
    elif "by status" in q or "per status" in q:
        ent.group_by = "status"

    return ent


def _which_datasets(q: str, ent: Entities) -> List[str]:
    ds = []
    if any(w in q for w in ("feedback", "rating", "review", "sentiment", "satisf")) or ent.sentiment or ent.rating_op:
        ds.append("Feedback")
    if any(w in q for w in ("appointment", "cancel", "schedul", "complet", "booking", "test ride", "test drive")):
        ds.append("Appointment")
    if any(w in q for w in ("enquiry", "enquiries", "vehicle", "city", "state", "payment", "lead",
                             "new lead", "returning", "car", "model", "status")) or ent.cities or ent.vehicles:
        ds.append("Enquiry")
    if not ds:
        ds = ["Enquiry"]
    # de-dup, preserve order
    seen, out = set(), []
    for d in ds:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def plan_query(query: str, ent: Entities) -> QueryPlan:
    q = _norm(query)
    trace = []
    datasets = _which_datasets(q, ent)
    trace.append(f"datasets_detected={datasets}")

    # Any query whose entities/keywords span more than one dataset is
    # treated as multi-hop by default (implicit AND / intersection) —
    # explicit connector words only refine the combine mode below.
    is_multi_hop = len(datasets) > 1

    combine = "intersection"
    if any(c in q for c in _MULTI_HOP_CONNECTORS_DIFF):
        combine = "difference"
    elif any(c in q for c in _MULTI_HOP_CONNECTORS_OR):
        combine = "union"

    if any(t in q for t in _AGG_TRIGGERS["compare"]):
        op = "COMPARE"
    elif ent.group_by:
        op = "GROUP_BY"
    elif ent.limit and ("top" in q or any(t in q for t in _AGG_TRIGGERS["max"])):
        op = "TOP_K"
    elif ent.limit and ("bottom" in q or any(t in q for t in _AGG_TRIGGERS["min"])):
        op = "BOTTOM_K"
    elif any(t in q for t in _AGG_TRIGGERS["average"]):
        op = "AVERAGE"
    elif any(t in q for t in _AGG_TRIGGERS["max"]):
        op = "MAX"
    elif any(t in q for t in _AGG_TRIGGERS["min"]):
        op = "MIN"
    elif is_multi_hop:
        op = "MULTI_HOP"
    elif _has_count_trigger(q):
        op = "COUNT"
    elif any(t in q for t in _AGG_TRIGGERS["sum"]):
        op = "SUM"
    else:
        op = "NONE"   # let the existing pipeline handle plain lookups/filters

    conf = 0.9 if op != "NONE" else 0.0
    if op == "MULTI_HOP" and len(datasets) < 2:
        conf = 0.3   # weak signal, not really cross-dataset
    trace.append(f"operation={op} combine={combine}")

    return QueryPlan(operation=op, datasets=datasets, combine=combine,
                      confidence=conf, trace=trace)


# ═══════════════════════════════════════════════════════════ FILTER HELPERS

def _filter_enquiry(df: pd.DataFrame, ent: Entities) -> pd.DataFrame:
    out = df
    if ent.enquiry_ids:
        out = out[out["ENQUIRY ID"].astype(str).str.upper().isin(ent.enquiry_ids)]
    if ent.cities:
        pattern = "|".join(re.escape(c) for c in ent.cities)
        out = out[out["City / State"].astype(str).str.contains(pattern, case=False, na=False)]
    if ent.vehicles:
        pattern = "|".join(re.escape(v) for v in ent.vehicles)
        out = out[out["Vehicle Name / Model"].astype(str).str.contains(pattern, case=False, na=False)]
    if ent.names:
        pattern = "|".join(re.escape(n) for n in ent.names)
        out = out[out["Customer Name"].astype(str).str.contains(pattern, case=False, na=False)]
    if ent.statuses:
        pattern = "|".join(re.escape(s) for s in ent.statuses)
        out = out[out["Status"].astype(str).str.contains(pattern, case=False, na=False)]
    return out


def _filter_appointment(df: pd.DataFrame, ent: Entities) -> pd.DataFrame:
    out = df
    if ent.enquiry_ids:
        out = out[out["Enquiry ID"].astype(str).str.upper().isin(ent.enquiry_ids)]
    if ent.names:
        pattern = "|".join(re.escape(n) for n in ent.names)
        out = out[out["Customer Name"].astype(str).str.contains(pattern, case=False, na=False)]
    if ent.statuses:
        pattern = "|".join(re.escape(s) for s in ent.statuses)
        out = out[out["Status"].astype(str).str.contains(pattern, case=False, na=False)]
    return out


def _filter_feedback(df: pd.DataFrame, ent: Entities) -> pd.DataFrame:
    out = df.copy()
    if ent.enquiry_ids:
        out = out[out["Enquiry ID"].astype(str).str.upper().isin(ent.enquiry_ids)]
    if ent.names:
        pattern = "|".join(re.escape(n) for n in ent.names)
        out = out[out["Customer Name"].astype(str).str.contains(pattern, case=False, na=False)]
    rating_num = pd.to_numeric(out["Rating"], errors="coerce")
    if ent.rating_op:
        op, val = ent.rating_op
        out = out[rating_num < val] if op == "<" else out[rating_num >= val]
    elif ent.sentiment == "negative":
        out = out[rating_num <= 2]
    elif ent.sentiment == "positive":
        out = out[rating_num >= 4]
    return out


_FILTERS = {"Enquiry": _filter_enquiry, "Appointment": _filter_appointment, "Feedback": _filter_feedback}
_NAME_COL = {"Enquiry": "Customer Name", "Appointment": "Customer Name", "Feedback": "Customer Name"}


# ══════════════════════════════════════════════════════════════ THE ENGINE

class QueryEngine:
    """
    Deterministic query engine: understanding → planning → pandas
    execution → verified facts string. Never invents numbers — every
    figure in `facts` is computed directly from the dataframes.
    """

    MIN_CONFIDENCE = 0.55

    def __init__(self, dfs: Dict[str, pd.DataFrame]):
        self.dfs = dfs
        self.vocab = KnownVocab(dfs)

    # -- public -------------------------------------------------------
    def try_handle(self, query: str) -> Optional[EngineResult]:
        ent = extract_entities(query, self.vocab)
        plan = plan_query(query, ent)
        if plan.operation == "NONE" or plan.confidence < self.MIN_CONFIDENCE:
            return None

        try:
            if plan.operation == "MULTI_HOP":
                return self._run_multi_hop(query, ent, plan)
            if plan.operation in ("COUNT", "SUM", "AVERAGE", "MAX", "MIN"):
                return self._run_aggregate(query, ent, plan)
            if plan.operation in ("TOP_K", "BOTTOM_K"):
                return self._run_topk(query, ent, plan)
            if plan.operation == "GROUP_BY":
                return self._run_group_by(query, ent, plan)
            if plan.operation == "COMPARE":
                return self._run_compare(query, ent, plan)
        except Exception as exc:  # never crash the API on a bad query
            plan.trace.append(f"error={exc}")
            return None
        return None

    # -- operations -----------------------------------------------------
    def _run_aggregate(self, query, ent, plan) -> Optional[EngineResult]:
        # pick the dataset most relevant to the aggregation target
        target = plan.datasets[0]
        df = self.dfs.get(target)
        if df is None:
            return None
        filtered = _FILTERS[target](df, ent)
        n = len(filtered)

        if plan.operation == "COUNT":
            facts = f"Computed fact: {n} matching record(s) in {target} data."
            conf = 1.0 if n >= 0 else 0.0
        elif plan.operation in ("AVERAGE", "SUM", "MAX", "MIN") and target == "Feedback":
            vals = pd.to_numeric(filtered["Rating"], errors="coerce").dropna()
            if vals.empty:
                facts = "No matching rows with a numeric rating were found."
            elif plan.operation == "AVERAGE":
                facts = f"Computed fact: the average rating is {round(vals.mean(), 2)} (from {len(vals)} rows)."
            elif plan.operation == "SUM":
                facts = f"Computed fact: the sum of ratings is {round(vals.sum(), 2)} (from {len(vals)} rows)."
            elif plan.operation == "MAX":
                facts = f"Computed fact: the highest rating is {vals.max()}."
            else:
                facts = f"Computed fact: the lowest rating is {vals.min()}."
            conf = 1.0 if not vals.empty else 0.3
        else:
            facts = f"Computed fact: {n} matching record(s) in {target} data."
            conf = 0.7

        names = sorted(set(filtered[_NAME_COL[target]].dropna().astype(str))) if n else []
        return EngineResult(plan=plan, entities=ent, facts=facts,
                             matched_names=names, row_count=n, confidence=conf)

    def _run_topk(self, query, ent, plan) -> Optional[EngineResult]:
        target = "Feedback" if "Feedback" in plan.datasets else plan.datasets[0]
        df = self.dfs.get(target)
        if df is None or "Rating" not in df.columns:
            return None
        filtered = _FILTERS[target](df, ent).copy()
        filtered["_r"] = pd.to_numeric(filtered["Rating"], errors="coerce")
        filtered = filtered.dropna(subset=["_r"])
        ascending = plan.operation == "BOTTOM_K"
        k = ent.limit or 5
        top = filtered.sort_values("_r", ascending=ascending).head(k)
        lines = [f"Computed fact: {'bottom' if ascending else 'top'} {len(top)} by rating:"]
        for _, r in top.iterrows():
            lines.append(f"  * {r.get('Customer Name','?')} — Rating {r.get('Rating','?')} "
                         f"(ID {r.get('Enquiry ID','?')})")
        facts = "\n".join(lines)
        names = list(top["Customer Name"].astype(str))
        return EngineResult(plan=plan, entities=ent, facts=facts,
                             matched_names=names, row_count=len(top),
                             confidence=1.0 if len(top) else 0.3)

    def _run_group_by(self, query, ent, plan) -> Optional[EngineResult]:
        target = plan.datasets[0]
        df = self.dfs.get(target)
        if df is None:
            return None
        filtered = _FILTERS[target](df, ent)
        col_map = {"city": "City / State", "vehicle": "Vehicle Name / Model", "status": "Status"}
        col = col_map.get(ent.group_by)
        if col is None or col not in filtered.columns:
            return None
        counts = filtered[col].astype(str).value_counts().head(15)
        lines = [f"Computed fact: {target} record counts grouped by {ent.group_by}:"]
        for k, v in counts.items():
            lines.append(f"  * {k}: {v}")
        facts = "\n".join(lines)
        return EngineResult(plan=plan, entities=ent, facts=facts,
                             matched_names=[], row_count=len(filtered),
                             confidence=1.0 if len(counts) else 0.3)

    def _run_compare(self, query, ent, plan) -> Optional[EngineResult]:
        # Compare two named entities (people, cities, or vehicles) on
        # whatever dataset is relevant.
        target = plan.datasets[0]
        df = self.dfs.get(target)
        if df is None:
            return None
        subjects = ent.names or ent.cities or ent.vehicles
        if len(subjects) < 2:
            return None
        col = "Customer Name" if ent.names else ("City / State" if ent.cities else "Vehicle Name / Model")
        lines = [f"Computed fact: comparison on {target} data —"]
        matched_names: List[str] = []
        for subj in subjects[:4]:
            sub_df = df[df[col].astype(str).str.contains(re.escape(subj), case=False, na=False)]
            if target == "Feedback" and "Rating" in sub_df.columns:
                vals = pd.to_numeric(sub_df["Rating"], errors="coerce").dropna()
                avg = round(vals.mean(), 2) if not vals.empty else "N/A"
                lines.append(f"  * {subj}: {len(sub_df)} record(s), average rating {avg}")
            else:
                lines.append(f"  * {subj}: {len(sub_df)} record(s)")
            if "Customer Name" in sub_df.columns:
                matched_names += list(sub_df["Customer Name"].astype(str))
        facts = "\n".join(lines)
        return EngineResult(plan=plan, entities=ent, facts=facts,
                             matched_names=sorted(set(matched_names)),
                             row_count=len(df), confidence=0.9)

    def _run_multi_hop(self, query, ent, plan) -> Optional[EngineResult]:
        """
        Real multi-hop reasoning: filter each relevant dataset
        independently, collect the set of customer names satisfying
        each dataset's condition, then combine (intersection / union /
        difference) per the query's connector words.

        Joined on Customer Name — verified to be the only consistent
        key across these three CSVs (see module docstring).
        """
        per_dataset_names: List[set] = []
        trace = []
        for ds in plan.datasets:
            df = self.dfs.get(ds)
            if df is None:
                continue
            filtered = _FILTERS[ds](df, ent)
            names = set(filtered[_NAME_COL[ds]].dropna().astype(str))
            per_dataset_names.append(names)
            trace.append(f"{ds}: {len(filtered)} rows -> {len(names)} distinct customers")

        if not per_dataset_names:
            return None

        if plan.combine == "intersection":
            combined = set.intersection(*per_dataset_names)
        elif plan.combine == "union":
            combined = set.union(*per_dataset_names)
        else:  # difference: first minus rest
            combined = per_dataset_names[0]
            for s in per_dataset_names[1:]:
                combined = combined - s

        combined = sorted(combined)
        lines = [f"Computed fact ({plan.combine} across {', '.join(plan.datasets)}): "
                 f"{len(combined)} customer(s) match all conditions."]
        for t in trace:
            lines.append(f"  - {t}")
        if combined:
            lines.append("Matching customers: " + ", ".join(combined[:25]))
            if len(combined) > 25:
                lines.append(f"  … and {len(combined) - 25} more.")

        facts = "\n".join(lines)
        conf = 0.9 if combined or len(plan.datasets) > 1 else 0.4
        return EngineResult(plan=plan, entities=ent, facts=facts,
                             matched_names=combined, row_count=len(combined),
                             confidence=conf)
