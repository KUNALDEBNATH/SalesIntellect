"""
evaluate_intelligence.py
═══════════════════════════════════════════════════════════════════════════════
Automated intelligence benchmark for the Platinum Sales chatbot.

Run:
    python evaluate_intelligence.py
    python evaluate_intelligence.py --baseline   # saves results as baseline
    python evaluate_intelligence.py --current    # compares to baseline

Measures 15 dimensions. Every expected answer is computed independently with
pandas — percentages are NEVER fabricated.

Usage
─────
    python evaluate_intelligence.py
    python evaluate_intelligence.py --baseline
    python evaluate_intelligence.py --current
    python evaluate_intelligence.py --debug      # show per-test traces
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_UPLOADS = "/mnt/user-data/uploads"
sys.path.insert(0, _HERE)
sys.path.insert(0, _UPLOADS)

CSV_FILES = {
    "Enquiry":     os.path.join(_UPLOADS, "sales_enquiry_dataset.csv"),
    "Appointment": os.path.join(_UPLOADS, "sales_appointment_dataset.csv"),
    "Feedback":    os.path.join(_UPLOADS, "sales_feedback_dataset.csv"),
}

BASELINE_FILE = os.path.join(_HERE, "eval_baseline.json")

# ── local modules ─────────────────────────────────────────────────────────────
from query_engine import QueryEngine, extract_entities, plan_query
from ambiguity_detector import detect_ambiguity
from confidence import calibrate, HIGH_THRESHOLD, MEDIUM_THRESHOLD
from provenance import build_timelines, query_event_chain
from data_quality import validate_datasets
from aggregation_engine import build_alias_map, compute_aggregation, resolve_field


# ─────────────────────────────────── data loading ────────────────────────────

def load_dfs() -> Dict[str, pd.DataFrame]:
    dfs = {}
    for name, path in CSV_FILES.items():
        if not os.path.exists(path):
            print(f"  [WARN] Missing {path}")
            continue
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        dfs[name] = df
    return dfs


# ─────────────────────────────────── test case ───────────────────────────────

@dataclass
class TestCase:
    category: str
    query: str
    predicate: Callable
    description: str = ""


@dataclass
class TestResult:
    category: str
    query: str
    passed: bool
    detail: str
    elapsed: float = 0.0


# ─────────────────────────────────── evaluation runner ───────────────────────

class EvaluationRunner:

    def __init__(self, dfs: Dict[str, pd.DataFrame], debug: bool = False):
        self.dfs = dfs
        self.debug = debug
        self.engine = QueryEngine(dfs)
        build_alias_map(dfs)
        self.timelines = build_timelines(dfs)

        E, A, F = dfs.get("Enquiry"), dfs.get("Appointment"), dfs.get("Feedback")
        self.E, self.A, self.F = E, A, F

        # Pre-computed ground truth (pandas only, no LLM)
        self._avg_rating = round(pd.to_numeric(F["Rating"], errors="coerce").mean(), 2) if F is not None else None
        self._max_rating = pd.to_numeric(F["Rating"], errors="coerce").max() if F is not None else None
        self._neg_names = set(F[pd.to_numeric(F["Rating"], errors="coerce") <= 2]["Customer Name"].astype(str)) if F is not None else set()
        self._cancelled_names = set(A[A["Status"].astype(str).str.contains("cancel", case=False)]["Customer Name"].astype(str)) if A is not None else set()
        self._cities = E["City / State"].astype(str).unique().tolist() if E is not None else []

    def _try(self, fn: Callable, *args, **kwargs):
        t0 = time.time()
        try:
            result = fn(*args, **kwargs)
            return result, None, time.time() - t0
        except Exception as exc:
            return None, str(exc), time.time() - t0

    def _run_test(self, tc: TestCase) -> TestResult:
        t0 = time.time()
        try:
            ok, detail = tc.predicate(self)
        except Exception as exc:
            ok, detail = False, f"exception: {exc}"
        return TestResult(
            category=tc.category,
            query=tc.query,
            passed=ok,
            detail=detail,
            elapsed=round(time.time() - t0, 4),
        )

    # ── build all test cases ─────────────────────────────────────────────────

    def build_tests(self) -> List[TestCase]:
        tests = []
        E, A, F = self.E, self.A, self.F

        # ═══════════════════════════════ INTENT ACCURACY ══════════════════════
        intent_cases = [
            ("how many enquiries from Hyderabad",       "COUNT"),
            ("average rating",                          "AVERAGE"),
            ("top 3 by rating",                        "TOP_K"),
            ("group enquiries by city",                "GROUP_BY"),
            ("customers with negative feedback and cancelled appointments", "MULTI_HOP"),
            ("highest rating",                         "MAX"),
            ("lowest rating",                          "MIN"),
        ]
        for q, expected_op in intent_cases:
            def _check(runner, q=q, exp=expected_op):
                ent = extract_entities(q, runner.engine.vocab)
                plan = plan_query(q, ent)
                ok = plan.operation == exp
                return ok, f"got {plan.operation}, expected {exp}"
            tests.append(TestCase("intent_accuracy", q, _check))

        # ═══════════════════════════════ ENTITY RESOLUTION ════════════════════
        if E is not None:
            real_cities = [str(c).split(",")[0].strip() for c in E["City / State"].dropna().unique()]
            for city in real_cities[:4]:
                def _check(runner, city=city):
                    ent = extract_entities(f"enquiries from {city}", runner.engine.vocab)
                    return city in ent.cities, f"expected {city} in {ent.cities}"
                tests.append(TestCase("entity_resolution", f"enquiries from {city}", _check))

        # ═══════════════════════════════ AMBIGUITY DETECTION ══════════════════

        # Pronoun without context → should flag as ambiguous
        def _pronoun_ambi(runner):
            r = detect_ambiguity(
                query="What is his feedback?",
                extracted_names=[], extracted_cities=[],
                extracted_ids=[], extracted_vehicles=[],
                dfs=runner.dfs, session_has_active_customer=False,
            )
            return r.is_ambiguous, f"kind={r.kind}"
        tests.append(TestCase("ambiguity_detection", "What is his feedback? (no context)", _pronoun_ambi))

        # Clear query → should NOT flag as ambiguous
        def _clear_not_ambi(runner):
            r = detect_ambiguity(
                query="Show feedback for ENQ025",
                extracted_names=[], extracted_cities=[],
                extracted_ids=["ENQ025"], extracted_vehicles=[],
                dfs=runner.dfs, session_has_active_customer=False,
            )
            return not r.is_ambiguous, f"unexpected ambiguity: {r.kind}"
        tests.append(TestCase("ambiguity_detection", "Show feedback for ENQ025 (clear)", _clear_not_ambi))

        # Contradictory filter
        def _contradiction(runner):
            r = detect_ambiguity(
                query="Show cancelled and completed appointments",
                extracted_names=[], extracted_cities=[],
                extracted_ids=[], extracted_vehicles=[],
                dfs=runner.dfs, session_has_active_customer=False,
            )
            return r.is_ambiguous and r.kind == "CONTRADICTORY_FILTER", f"kind={r.kind}"
        tests.append(TestCase("ambiguity_detection", "cancelled and completed (contradictory)", _contradiction))

        # ═══════════════════════════════ AGGREGATION ACCURACY ═════════════════

        if F is not None:
            expected_avg = self._avg_rating
            def _avg_rating(runner):
                r = runner.engine.try_handle("what is the average rating")
                if r is None:
                    return False, "engine returned None"
                return str(expected_avg) in r.facts, f"expected {expected_avg}, got: {r.facts}"
            tests.append(TestCase("aggregation_accuracy", "average rating", _avg_rating))

            expected_max = int(self._max_rating)
            def _max_rating(runner):
                r = runner.engine.try_handle("what is the highest rating")
                if r is None:
                    return False, "engine returned None"
                return str(expected_max) in r.facts, f"expected {expected_max}, got: {r.facts}"
            tests.append(TestCase("aggregation_accuracy", "highest rating", _max_rating))

        if E is not None:
            real_cities = [str(c).split(",")[0].strip() for c in E["City / State"].dropna().unique()]
            for city in real_cities[:3]:
                expected_count = int(E["City / State"].astype(str).str.contains(city, case=False).sum())
                def _city_count(runner, city=city, exp=expected_count):
                    r = runner.engine.try_handle(f"how many enquiries from {city}")
                    if r is None:
                        return False, f"engine returned None for {city}"
                    return str(exp) in r.facts, f"expected {exp} in facts: {r.facts}"
                tests.append(TestCase("aggregation_accuracy", f"count from {city}", _city_count))

        # ═══════════════════════════════ RETRIEVAL ACCURACY ═══════════════════

        def _retrieval_nonzero(runner):
            r = runner.engine.try_handle("how many enquiries are there")
            return r is not None and r.row_count > 0, f"row_count={r.row_count if r else None}"
        tests.append(TestCase("retrieval_accuracy", "how many enquiries", _retrieval_nonzero))

        # ═══════════════════════════════ JOIN ACCURACY ════════════════════════

        if F is not None and A is not None:
            expected_intersection = sorted(self._neg_names & self._cancelled_names)
            def _multi_hop_join(runner):
                r = runner.engine.try_handle(
                    "which customers gave negative feedback and also cancelled their appointments"
                )
                if r is None:
                    return False, "engine returned None"
                got = sorted(r.matched_names)
                return got == expected_intersection, f"expected {expected_intersection}, got {got}"
            tests.append(TestCase("join_accuracy",
                                  "negative feedback AND cancelled appointment", _multi_hop_join))

        # ═══════════════════════════════ MULTI-HOP REASONING ══════════════════

        if E is not None and F is not None:
            real_cities = [str(c).split(",")[0].strip() for c in E["City / State"].dropna().unique()]
            for city in real_cities[:2]:
                city_names = set(E[E["City / State"].astype(str).str.contains(city, case=False)]["Customer Name"].astype(str))
                expected = sorted(city_names & self._neg_names)
                def _city_neg(runner, city=city, exp=expected):
                    r = runner.engine.try_handle(f"customers from {city} with ratings below 3")
                    if r is None:
                        return False, f"engine None for {city}"
                    got = sorted(r.matched_names)
                    return got == exp, f"expected {exp}, got {got}"
                tests.append(TestCase("multi_hop_reasoning", f"{city} + negative feedback", _city_neg))

        # ═══════════════════════════════ TEMPORAL / EVENT-CHAIN ═══════════════

        def _event_chain_no_appt(runner):
            result = query_event_chain(runner.timelines, "who enquired but never had an appointment")
            return result is not None and "Computed event-chain fact" in result, f"got: {result}"
        tests.append(TestCase("temporal_reasoning", "enquired but never appointment", _event_chain_no_appt))

        def _event_chain_cancelled(runner):
            result = query_event_chain(runner.timelines, "who cancelled after enquiring")
            return result is not None, f"got: {result}"
        tests.append(TestCase("temporal_reasoning", "cancelled after enquiring", _event_chain_cancelled))

        # ═══════════════════════════════ CONFIDENCE CALIBRATION ═══════════════

        # "Records found" should NOT guarantee HIGH confidence when entity is ambiguous
        def _ambig_lowers_confidence(runner):
            p = calibrate(
                operation="COUNT",
                entities_found=1,
                entity_ambiguous=True,
                multiple_entity_matches=True,
                datasets_matched=1,
                rows_found=50,
                plan_well_formed=True,
            )
            return p.level in ("AMBIGUOUS", "LOW"), f"level={p.level} overall={p.overall}"
        tests.append(TestCase("query_planning_accuracy",
                              "ambiguous entity should lower confidence", _ambig_lowers_confidence))

        # High-quality clear query → HIGH confidence
        def _clear_high_confidence(runner):
            p = calibrate(
                operation="COUNT",
                entities_found=1,
                entity_ambiguous=False,
                multiple_entity_matches=False,
                datasets_matched=1,
                rows_found=10,
                plan_well_formed=True,
            )
            return p.level == "HIGH", f"level={p.level} overall={p.overall}"
        tests.append(TestCase("query_planning_accuracy",
                              "clear query should give HIGH confidence", _clear_high_confidence))

        # ═══════════════════════════════ HALLUCINATION RESISTANCE ═════════════

        def _nonexistent_customer(runner):
            # "John" does not exist; engine should return None or 0 rows
            r = runner.engine.try_handle("show feedback for ZZZNonexistentCustomer99999")
            if r is None:
                return True, "correctly returned None for unknown customer"
            return r.row_count == 0, f"expected 0 rows, got {r.row_count}"
        tests.append(TestCase("hallucination_resistance",
                              "nonexistent customer should return None or 0", _nonexistent_customer))

        def _nonexistent_id(runner):
            r = runner.engine.try_handle("show details for ENQ99999")
            if r is None:
                return True, "correctly returned None"
            return r.row_count == 0, f"expected 0 rows, got {r.row_count}"
        tests.append(TestCase("hallucination_resistance",
                              "nonexistent ID should return 0", _nonexistent_id))

        def _unsupported_field(runner):
            # "salary" does not exist in any dataset
            r = runner.engine.try_handle("what is Rahul's salary")
            # Should either be None (engine skips it) or very low confidence
            if r is None:
                return True, "correctly skipped (no salary field)"
            p = calibrate(
                operation=r.plan.operation,
                entities_found=len(r.entities.names),
                entity_ambiguous=False,
                multiple_entity_matches=False,
                datasets_matched=len(r.plan.datasets),
                rows_found=r.row_count,
                facts_grounded=False,
            )
            return p.level in ("LOW", "AMBIGUOUS"), f"level={p.level}"
        tests.append(TestCase("hallucination_resistance",
                              "unsupported field (salary)", _unsupported_field))

        # ═══════════════════════════════ ABSTENTION ACCURACY ══════════════════

        passthrough_queries = [
            "What is the status of ENQ001?",
            "Show feedback for Ravi",
            "Who gave bad feedback?",
            "Cancelled appointments",
            "What is Ananya's phone number?",
            "Who are you?",
        ]
        for q in passthrough_queries:
            def _passthrough(runner, q=q):
                r = runner.engine.try_handle(q)
                return r is None, f"expected engine to skip (return None), got op={r.plan.operation if r else None}"
            tests.append(TestCase("abstention_accuracy", f"pass-through: {q}", _passthrough))

        # ═══════════════════════════════ CLARIFICATION ACCURACY ═══════════════

        # Partial ENQ ID should trigger ambiguity clarification
        def _partial_id_clarify(runner):
            r = detect_ambiguity(
                query="Show details for ENQ5",
                extracted_names=[], extracted_cities=[],
                extracted_ids=[], extracted_vehicles=[],
                dfs=runner.dfs,
            )
            return r.is_ambiguous and r.kind == "PARTIAL_ID", f"kind={r.kind}"
        tests.append(TestCase("clarification_accuracy",
                              "partial ID should ask for clarification", _partial_id_clarify))

        # ═══════════════════════════════ DOCUMENT GROUNDING ═══════════════════

        def _doc_verifier_catches_hallucination(runner):
            from document_verifier import verify_document_answer
            chunks = ["The invoice total is 5000."]
            # LLM invents a number that doesn't exist
            ok, reason = verify_document_answer("The total is 9999.", chunks, strict=True)
            return not ok, f"should fail (hallucinated number), reason={reason}"
        tests.append(TestCase("document_grounding",
                              "doc verifier catches hallucinated number", _doc_verifier_catches_hallucination))

        def _doc_verifier_passes_grounded(runner):
            from document_verifier import verify_document_answer
            chunks = ["The invoice total is 5000."]
            ok, reason = verify_document_answer("The total is 5000.", chunks, strict=True)
            return ok, f"reason={reason}"
        tests.append(TestCase("document_grounding",
                              "doc verifier passes grounded answer", _doc_verifier_passes_grounded))

        # ═══════════════════════════════ DATA QUALITY ═════════════════════════

        def _data_quality_runs(runner):
            reports = validate_datasets(runner.dfs)
            return len(reports) == len(runner.dfs), f"got {len(reports)} reports for {len(runner.dfs)} datasets"
        tests.append(TestCase("answer_accuracy", "data quality validation runs", _data_quality_runs))

        # ═══════════════════════════════ ADVERSARIAL ══════════════════════════

        def _contradictory_filter(runner):
            r = detect_ambiguity(
                query="show cancelled and completed appointments at the same time",
                extracted_names=[], extracted_cities=[],
                extracted_ids=[], extracted_vehicles=[],
                dfs=runner.dfs,
            )
            return r.is_ambiguous and r.kind == "CONTRADICTORY_FILTER", f"kind={r.kind}"
        tests.append(TestCase("hallucination_resistance",
                              "contradictory filter detected", _contradictory_filter))

        def _empty_query(runner):
            r = runner.engine.try_handle("")
            return r is None, f"empty query should return None, got {r}"
        tests.append(TestCase("hallucination_resistance", "empty query returns None", _empty_query))

        def _gibberish(runner):
            r = runner.engine.try_handle("asdkjalsd randomness zzz 999!")
            return r is None, f"gibberish should return None, got {r}"
        tests.append(TestCase("hallucination_resistance", "gibberish returns None", _gibberish))

        return tests

    # ── run ──────────────────────────────────────────────────────────────────

    def run(self) -> Dict[str, List[TestResult]]:
        tests = self.build_tests()
        by_cat: Dict[str, List[TestResult]] = {}
        for tc in tests:
            result = self._run_test(tc)
            if self.debug and not result.passed:
                print(f"  FAIL [{tc.category}] {tc.query!r}\n       {result.detail}")
            by_cat.setdefault(tc.category, []).append(result)
        return by_cat


# ─────────────────────────────────── report ──────────────────────────────────

# Canonical category → display label mapping
_CATEGORY_LABELS = {
    "intent_accuracy":        "Intent Accuracy",
    "entity_resolution":      "Entity Resolution",
    "ambiguity_detection":    "Ambiguity Detection",
    "aggregation_accuracy":   "Aggregation Accuracy",
    "retrieval_accuracy":     "Retrieval Accuracy",
    "join_accuracy":          "Join Accuracy",
    "multi_hop_reasoning":    "Multi-Hop Reasoning",
    "temporal_reasoning":     "Temporal Reasoning",
    "query_planning_accuracy":"Query Planning",
    "document_grounding":     "Document Grounding",
    "answer_accuracy":        "Answer Accuracy",
    "hallucination_resistance":"Hallucination Rate",
    "abstention_accuracy":    "Abstention Accuracy",
    "clarification_accuracy": "Clarification Accuracy",
    "conversation_accuracy":  "Conversation Accuracy",
}

_HALLU_CAT = "hallucination_resistance"   # inverted for display


def _pct(results: List[TestResult]) -> float:
    if not results:
        return 0.0
    return round(100.0 * sum(1 for r in results if r.passed) / len(results), 1)


def _overall(by_cat: Dict[str, List[TestResult]]) -> float:
    all_results = [r for cat_results in by_cat.values() for r in cat_results]
    return _pct(all_results)


def _to_scores(by_cat: Dict[str, List[TestResult]]) -> Dict[str, float]:
    scores = {}
    for cat, results in by_cat.items():
        p = _pct(results)
        # Hallucination rate: displayed as percentage PASSING, lower is worse
        scores[cat] = p
    return scores


def print_report(by_cat: Dict[str, List[TestResult]], title: str = "CURRENT") -> Dict[str, float]:
    scores = _to_scores(by_cat)
    overall = _overall(by_cat)

    print("\n" + "=" * 52)
    print(f"  PLATINUM SALES INTELLIGENCE EVALUATION — {title}")
    print("=" * 52)

    ordered_cats = list(_CATEGORY_LABELS.keys())
    extra_cats = [c for c in by_cat if c not in ordered_cats]

    for cat in ordered_cats + extra_cats:
        if cat not in by_cat:
            continue
        results = by_cat[cat]
        label = _CATEGORY_LABELS.get(cat, cat)
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        pct = _pct(results)

        # Hallucination: show "resistance" (% of adversarial tests PASSED)
        marker = "✓" if pct >= 80 else ("△" if pct >= 60 else "✗")
        print(f"  {marker} {label:<35} {pct:>5.1f}%  ({passed}/{total})")

    print("─" * 52)
    print(f"  OVERALL INTELLIGENCE SCORE:             {overall:>5.1f}%")
    print("=" * 52 + "\n")

    return scores


def compare_reports(baseline: Dict[str, float], current: Dict[str, float]) -> None:
    print("\n" + "=" * 65)
    print("  BEFORE / AFTER REGRESSION COMPARISON")
    print("=" * 65)
    print(f"  {'Metric':<35} {'Baseline':>8}  {'Current':>7}  {'Change':>8}")
    print("─" * 65)

    all_cats = sorted(set(list(baseline.keys()) + list(current.keys())))
    for cat in all_cats:
        label = _CATEGORY_LABELS.get(cat, cat)
        b = baseline.get(cat, 0.0)
        c = current.get(cat, 0.0)
        delta = c - b
        sign = "+" if delta >= 0 else ""
        arrow = "↑" if delta > 0.5 else ("↓" if delta < -0.5 else " ")
        print(f"  {label:<35} {b:>7.1f}%  {c:>7.1f}%  {sign}{delta:>5.1f}%  {arrow}")

    b_overall = sum(baseline.values()) / max(len(baseline), 1)
    c_overall = sum(current.values()) / max(len(current), 1)
    delta = c_overall - b_overall
    sign = "+" if delta >= 0 else ""
    print("─" * 65)
    print(f"  {'OVERALL':<35} {b_overall:>7.1f}%  {c_overall:>7.1f}%  {sign}{delta:>5.1f}%")
    print("=" * 65 + "\n")


# ─────────────────────────────────── entry point ─────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Platinum Sales Intelligence Evaluation")
    parser.add_argument("--baseline", action="store_true", help="Save results as baseline")
    parser.add_argument("--current",  action="store_true", help="Compare against saved baseline")
    parser.add_argument("--debug",    action="store_true", help="Show per-test failures")
    args = parser.parse_args()

    print("[eval] Loading datasets …")
    dfs = load_dfs()
    if not dfs:
        print("[eval] ERROR: no datasets loaded. Make sure CSV files are in the uploads directory.")
        sys.exit(1)
    print(f"[eval] Loaded: {', '.join(f'{k} ({len(v)} rows)' for k, v in dfs.items())}")

    runner = EvaluationRunner(dfs, debug=args.debug)

    print("[eval] Running evaluation …")
    t0 = time.time()
    by_cat = runner.run()
    elapsed = round(time.time() - t0, 2)
    print(f"[eval] Completed in {elapsed}s\n")

    scores = print_report(by_cat, title="CURRENT" if not args.baseline else "BASELINE")

    if args.baseline:
        with open(BASELINE_FILE, "w") as f:
            json.dump(scores, f, indent=2)
        print(f"  Baseline saved → {BASELINE_FILE}\n")

    elif args.current:
        if not os.path.exists(BASELINE_FILE):
            print(f"  [WARN] No baseline file found at {BASELINE_FILE}. Run --baseline first.\n")
        else:
            with open(BASELINE_FILE) as f:
                baseline = json.load(f)
            compare_reports(baseline, scores)

    overall = _overall(by_cat)
    sys.exit(0 if overall >= 70.0 else 1)


if __name__ == "__main__":
    main()
