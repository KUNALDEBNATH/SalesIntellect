"""
regression_tests.py
────────────────────────────────────────────────────────────────────────────
Regression suite for query_engine.py (the new structured reasoning layer).

Every aggregation/multi-hop test's expected answer is computed
INDEPENDENTLY with plain pandas (not by calling the engine twice), so a
bug that produces a confident-but-wrong number is actually caught —
this is "answer accuracy", not "did it run without crashing".

Also includes a pass-through suite: single-dataset lookup/filter queries
that MUST return None from the engine (so the existing test.py pipeline
in test.py continues to handle them exactly as before — no regression).

Run:
    python regression_tests.py
"""
from __future__ import annotations

import sys
import pandas as pd

from query_engine import QueryEngine

CSV_FILES = {
    "Enquiry":     "sales_enquiry_dataset.csv",
    "Appointment": "sales_appointment_dataset.csv",
    "Feedback":    "sales_feedback_dataset.csv",
}


def load_dfs():
    dfs = {}
    for src, path in CSV_FILES.items():
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        dfs[src] = df
    return dfs


def run():
    dfs = load_dfs()
    eng = QueryEngine(dfs)
    E, A, F = dfs["Enquiry"], dfs["Appointment"], dfs["Feedback"]

    results = []  # (category, question, passed, detail)

    def check(category, question, predicate):
        r = eng.try_handle(question)
        try:
            ok, detail = predicate(r)
        except Exception as exc:
            ok, detail = False, f"exception: {exc}"
        results.append((category, question, ok, detail))

    # ── AGGREGATION: independently computed ground truth ──────────────
    for city in ["Hyderabad", "Chennai", "Bangalore", "Coimbatore"]:
        expected = int(E["City / State"].astype(str).str.contains(city, case=False).sum())
        check("aggregation", f"How many enquiries are from {city}?",
              lambda r, exp=expected: (r is not None and str(exp) in r.facts,
                                        f"expected {exp} in facts, got: {r.facts if r else None}"))

    expected_avg = round(pd.to_numeric(F["Rating"], errors="coerce").mean(), 2)
    check("aggregation", "What is the average rating?",
          lambda r: (r is not None and str(expected_avg) in r.facts,
                     f"expected avg {expected_avg}, got: {r.facts if r else None}"))

    expected_max = pd.to_numeric(F["Rating"], errors="coerce").max()
    check("aggregation", "What is the highest rating given?",
          lambda r: (r is not None and str(int(expected_max)) in r.facts,
                     f"expected max {expected_max}, got: {r.facts if r else None}"))

    # ── MULTI-HOP: independently computed intersection ────────────────
    neg_names = set(F[pd.to_numeric(F["Rating"], errors="coerce") <= 2]["Customer Name"].astype(str))
    cancelled_names = set(A[A["Status"].astype(str).str.contains("cancel", case=False)]["Customer Name"].astype(str))
    expected_intersection = sorted(neg_names & cancelled_names)
    check("multi_hop", "Which customers gave negative feedback and also cancelled their appointments?",
          lambda r, exp=expected_intersection: (
              r is not None and sorted(r.matched_names) == exp,
              f"expected {exp}, got {r.matched_names if r else None}"))

    chennai_names = set(E[E["City / State"].astype(str).str.contains("Chennai", case=False)]["Customer Name"].astype(str))
    expected_chennai_neg = sorted(chennai_names & neg_names)
    check("multi_hop", "Show customers from Chennai with ratings below 3",
          lambda r, exp=expected_chennai_neg: (
              r is not None and sorted(r.matched_names) == exp,
              f"expected {exp}, got {r.matched_names if r else None}"))

    # ── TOP_K correctness ───────────────────────────────────────────
    expected_top3 = F.assign(_r=pd.to_numeric(F["Rating"], errors="coerce")) \
                      .sort_values("_r", ascending=False).head(3)["Customer Name"].astype(str).tolist()
    check("ranking", "top 3 by rating",
          lambda r, exp=expected_top3: (
              r is not None and r.matched_names == exp,
              f"expected {exp}, got {r.matched_names if r else None}"))

    # ── GROUP_BY correctness ────────────────────────────────────────
    expected_top_city = E["City / State"].astype(str).value_counts().idxmax()
    check("group_by", "group enquiries by city",
          lambda r, exp=expected_top_city: (
              r is not None and exp.split(",")[0].strip() in r.facts,
              f"expected top city containing {exp}, got: {r.facts if r else None}"))

    # ── PASS-THROUGH: must NOT be handled by the engine (single lookups) ──
    passthrough_questions = [
        "What is the status of ENQ001?",
        "Show feedback for Ravi",
        "Who gave bad feedback?",
        "Cancelled appointments",
        "What is Ananya's phone number?",
        "Tell me about ENQ005",
        "New leads",
        "Who are you?",
        "what is the phone number of Karthik?",
        "what is the contact number of Ravi",
        "give me the mobile number of Ananya",
    ]
    for q in passthrough_questions:
        check("passthrough", q, lambda r: (r is None, f"expected engine to skip, got op={r.plan.operation if r else None}"))

    # ── OUT-OF-DOMAIN-ish / no crash guarantee ─────────────────────
    weird_questions = [
        "asdkjaslkdj random gibberish",
        "",
        "how many enquiries are from Atlantis?",
        "average rating of unicorns",
    ]
    for q in weird_questions:
        check("robustness", q, lambda r: (True, f"no crash, result={'None' if r is None else r.plan.operation}"))

    # ── REPORT ──────────────────────────────────────────────────────
    by_cat = {}
    for cat, q, ok, detail in results:
        by_cat.setdefault(cat, []).append((q, ok, detail))

    total, passed = len(results), sum(1 for *_, ok, _ in results if ok)
    print("=" * 78)
    print("QUERY ENGINE REGRESSION REPORT")
    print("=" * 78)
    for cat, rows in by_cat.items():
        cat_pass = sum(1 for _, ok, _ in rows if ok)
        print(f"\n[{cat}]  {cat_pass}/{len(rows)} passed")
        for q, ok, detail in rows:
            mark = "PASS" if ok else "FAIL"
            print(f"  {mark}  {q!r}")
            if not ok:
                print(f"        -> {detail}")
    print("\n" + "=" * 78)
    print(f"TOTAL: {passed}/{total} passed  ({round(100*passed/total,1)}%)")
    print("=" * 78)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(run())
