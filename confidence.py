"""
confidence.py
═══════════════════════════════════════════════════════════════════════════════
Multi-dimensional confidence calibration for the Platinum Sales chatbot.

Separates confidence into nine independent sub-scores, each covering a
distinct failure mode. The overall score is a weighted product — a single
catastrophically-low dimension can pull the whole answer down, which is
what we want: finding rows does NOT prove the interpretation was correct.

Thresholds
──────────
  HIGH   ≥ 0.75  → execute and answer
  MEDIUM ≥ 0.50  → answer only when ambiguity_confidence is also ≥ 0.65
  LOW    < 0.50  → ask clarification or abstain
  AMBIGUOUS      → forced to LOW regardless of other scores
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


# ─────────────────────────────────────── thresholds

HIGH_THRESHOLD   = 0.75
MEDIUM_THRESHOLD = 0.50
AMBIGUITY_FLOOR  = 0.65   # medium answers only when ambiguity is also clear


@dataclass
class ConfidenceProfile:
    intent_confidence:        float = 1.0   # did we understand what was asked?
    entity_confidence:        float = 1.0   # are extracted names/IDs unambiguous?
    ambiguity_confidence:     float = 1.0   # low → query is ambiguous (multiple matches, etc.)
    dataset_confidence:       float = 1.0   # are we querying the right dataset(s)?
    retrieval_confidence:     float = 1.0   # did retrieval find plausibly relevant rows?
    plan_confidence:          float = 1.0   # is the query-plan well-formed?
    execution_confidence:     float = 1.0   # did pandas execute cleanly?
    result_confidence:        float = 1.0   # is the result set non-empty / expected size?
    answer_grounding_confidence: float = 1.0  # are facts grounded in retrieved data?

    # human-readable notes set by the calibrator
    notes: list = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []

    @property
    def overall(self) -> float:
        """
        Weighted geometric-mean-ish product.
        Each sub-score is raised to a weight reflecting its importance.
        """
        weights = {
            "intent_confidence":           0.20,
            "entity_confidence":           0.18,
            "ambiguity_confidence":        0.15,
            "dataset_confidence":          0.12,
            "retrieval_confidence":        0.10,
            "plan_confidence":             0.10,
            "execution_confidence":        0.05,
            "result_confidence":           0.05,
            "answer_grounding_confidence": 0.05,
        }
        score = 1.0
        for attr, w in weights.items():
            val = max(0.0, min(1.0, getattr(self, attr)))
            # A zero sub-score collapses the overall score
            score *= val ** w
        return round(score, 4)

    @property
    def level(self) -> str:
        """HIGH / MEDIUM / LOW / AMBIGUOUS"""
        if self.ambiguity_confidence < 0.40:
            return "AMBIGUOUS"
        o = self.overall
        if o >= HIGH_THRESHOLD:
            return "HIGH"
        if o >= MEDIUM_THRESHOLD and self.ambiguity_confidence >= AMBIGUITY_FLOOR:
            return "MEDIUM"
        return "LOW"

    @property
    def should_answer(self) -> bool:
        return self.level in ("HIGH", "MEDIUM")

    @property
    def should_clarify(self) -> bool:
        return self.level in ("LOW", "AMBIGUOUS")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["overall"] = self.overall
        d["level"] = self.level
        return d


def calibrate(
    *,
    operation: str,
    entities_found: int,
    entity_ambiguous: bool,
    multiple_entity_matches: bool,
    datasets_matched: int,
    rows_found: int,
    expected_rows_min: int = 0,
    plan_well_formed: bool = True,
    execution_ok: bool = True,
    facts_grounded: bool = True,
    query_tokens: int = 5,
) -> ConfidenceProfile:
    """
    Build a ConfidenceProfile from observable signals.
    Every parameter has a measurable definition — no magic numbers.
    """
    p = ConfidenceProfile()

    # intent_confidence
    if operation == "NONE":
        p.intent_confidence = 0.3
        p.notes.append("Operation=NONE: intent unclear")
    elif operation in ("COUNT", "AVERAGE", "SUM", "MAX", "MIN",
                        "MULTI_HOP", "TOP_K", "GROUP_BY", "COMPARE"):
        p.intent_confidence = 0.95
    else:
        p.intent_confidence = 0.6

    # entity_confidence
    if entities_found == 0 and operation not in ("COUNT", "AVERAGE", "MAX", "MIN", "GROUP_BY"):
        p.entity_confidence = 0.5
        p.notes.append("No entities resolved")
    elif multiple_entity_matches:
        p.entity_confidence = 0.5
        p.notes.append("Multiple entity matches → lower entity confidence")
    else:
        p.entity_confidence = 1.0 if entities_found > 0 else 0.8

    # ambiguity_confidence
    if entity_ambiguous:
        p.ambiguity_confidence = 0.1   # AMBIGUOUS forced
        p.notes.append("Entity is ambiguous — must clarify before answering")
    elif multiple_entity_matches:
        p.ambiguity_confidence = 0.45
        p.notes.append("Multiple entities matched — possible ambiguity")
    elif query_tokens <= 2:
        p.ambiguity_confidence = 0.55
        p.notes.append("Very short query — possible underspecification")
    else:
        p.ambiguity_confidence = 1.0

    # dataset_confidence
    p.dataset_confidence = min(1.0, 0.6 + 0.2 * datasets_matched) if datasets_matched else 0.3

    # retrieval_confidence — rows found is necessary but NOT sufficient
    if rows_found == 0 and operation not in ("MULTI_HOP",):
        p.retrieval_confidence = 0.2
        p.notes.append("Zero rows returned")
    elif rows_found > 0:
        p.retrieval_confidence = 0.9
    else:
        p.retrieval_confidence = 0.6

    # plan_confidence
    p.plan_confidence = 1.0 if plan_well_formed else 0.2

    # execution_confidence
    p.execution_confidence = 1.0 if execution_ok else 0.0

    # result_confidence — rows found ≠ correct interpretation
    if rows_found < expected_rows_min and operation != "MULTI_HOP":
        p.result_confidence = 0.4
        p.notes.append(f"Fewer rows than expected (got {rows_found}, expected ≥{expected_rows_min})")
    elif rows_found > 0:
        p.result_confidence = 0.9
    else:
        p.result_confidence = 0.7

    # answer_grounding_confidence
    p.answer_grounding_confidence = 1.0 if facts_grounded else 0.3

    return p
