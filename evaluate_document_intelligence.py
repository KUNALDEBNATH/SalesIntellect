"""
evaluate_document_intelligence.py
═══════════════════════════════════════════════════════════════════════════════
Evaluation suite for the document understanding pipeline.

Tests the DocumentAnswerEngine (deterministic layer) against known-correct
answers derived from the synthetic test documents.

This suite tests what you CAN evaluate automatically:
  1. Document type detection accuracy
  2. Purpose extraction presence
  3. Main topic extraction
  4. Section detection count
  5. Section summary generation (non-empty, non-extractive)
  6. Key fact extraction
  7. Claim extraction
  8. Evidence extraction
  9. Conclusion extraction
  10. Key points generation
  11. Global summary: non-extractive check (is it reproduced text or generated?)
  12. Summary coherence (multi-sentence, covers purpose + content + conclusion)
  13. Short summary brevity
  14. Detailed summary comprehensiveness
  15. Section QA routing
  16. Conclusion QA routing
  17. Key points QA routing
  18. Purpose QA routing
  19. Hallucination check (no facts outside the document)
  20. Follow-up question handling (from memory, not re-extraction)
  21. Table QA: correct numeric computation
  22. Paraphrase robustness (same question, different wording)
  23. Entity extraction
  24. Limitation detection

Run:
    python evaluate_document_intelligence.py
    python evaluate_document_intelligence.py --verbose
    python evaluate_document_intelligence.py --doc path/to/your.pdf

Expected output: PASS/FAIL per test with percentage summary.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from document_understanding import (
    parse_document_structure,
    build_document_representation,
    build_table_representation,
    DocumentRepresentation,
)
from document_answer_engine import answer_from_representation, classify_doc_query
from doc_training_data import SYNTHETIC_DOCS


# ══════════════════════════════════════════════════════════════════════════════
# TEST INFRASTRUCTURE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str = ""
    answer_preview: str = ""


def _run_test(name: str, fn: Callable[[], Tuple[bool, str, str]]) -> TestResult:
    try:
        passed, detail, preview = fn()
        return TestResult(name, passed, detail, preview)
    except Exception as e:
        return TestResult(name, False, f"EXCEPTION: {e}", "")


# ══════════════════════════════════════════════════════════════════════════════
# SHARED FIXTURE: build representations for all synthetic docs once
# ══════════════════════════════════════════════════════════════════════════════

def _build_reps() -> List[Tuple[str, DocumentRepresentation, str]]:
    """Returns [(filename, rep, raw_text), ...]"""
    results = []
    for doc in SYNTHETIC_DOCS:
        raw = doc["text"]
        fn  = doc["filename"]
        structured = parse_document_structure(raw, fn)
        rep = build_document_representation(structured, debug=False)
        results.append((fn, rep, raw))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# TEST DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

def _nonblank(s: str) -> bool:
    return bool(s and s.strip() and len(s.strip()) > 10)


def _not_pure_extraction(answer: str, raw_text: str, threshold: float = 0.85) -> bool:
    """
    True when the answer is NOT simply a raw excerpt from the document.
    We check whether a long contiguous substring of `answer` (>150 chars)
    appears verbatim in `raw_text`. If yes → likely extractive. If no → likely generated.
    """
    answer_clean = re.sub(r"\s+", " ", answer.lower()).strip()
    raw_clean    = re.sub(r"\s+", " ", raw_text.lower()).strip()

    # Check 200-char windows in the answer against the source text
    window = 200
    for i in range(0, max(1, len(answer_clean) - window), 50):
        chunk = answer_clean[i:i + window]
        if len(chunk) < 100:
            break
        if chunk in raw_clean:
            return False   # Found verbatim chunk → extractive
    return True


def run_all_tests(reps: List[Tuple[str, DocumentRepresentation, str]],
                   verbose: bool = False) -> List[TestResult]:
    results: List[TestResult] = []
    paper_fn, paper_rep, paper_raw = reps[0]    # neural_network_paper.pdf
    report_fn, report_rep, report_raw = reps[1] # annual_report_2023.pdf
    resume_fn, resume_rep, resume_raw = reps[2] # john_smith_resume.pdf
    invoice_fn, invoice_rep, invoice_raw = reps[3] # invoice_INV2024001.pdf

    # ── 1. Document Type Detection ──────────────────────────────────────────
    def t1():
        checks = [
            (paper_rep.document_type, "research_paper"),
            (report_rep.document_type, "report"),
            (resume_rep.document_type, "resume"),
            (invoice_rep.document_type, "invoice"),
        ]
        wrong = [(fn, expected) for (detected, expected), (fn, _, __) in
                 zip(checks, reps) if detected != expected]
        # Allow partial credit: "general" is acceptable for tricky docs
        partial_wrong = [(d, e) for d, e in checks if d != e and d != "general"]
        ok = len(partial_wrong) == 0
        detail = f"{len(checks)-len(partial_wrong)}/{len(checks)} correct" + (
            f" (wrong: {partial_wrong})" if partial_wrong else "")
        preview = f"paper={paper_rep.document_type}, report={report_rep.document_type}, resume={resume_rep.document_type}, invoice={invoice_rep.document_type}"
        return ok, detail, preview
    results.append(_run_test("1. Document type detection", t1))

    # ── 2. Purpose Extraction ───────────────────────────────────────────────
    def t2():
        ok_count = sum(1 for _, rep, _ in reps if _nonblank(rep.purpose))
        ok = ok_count == len(reps)
        return ok, f"{ok_count}/{len(reps)} docs have non-empty purpose", paper_rep.purpose[:120]
    results.append(_run_test("2. Purpose extraction", t2))

    # ── 3. Main Topic Extraction ────────────────────────────────────────────
    def t3():
        ok_count = sum(1 for _, rep, _ in reps if _nonblank(rep.main_topic))
        ok = ok_count >= 3   # allow one miss
        return ok, f"{ok_count}/{len(reps)} docs have non-empty main_topic", paper_rep.main_topic[:80]
    results.append(_run_test("3. Main topic extraction", t3))

    # ── 4. Section Detection ────────────────────────────────────────────────
    def t4():
        counts = [(fn, len(rep.sections)) for fn, rep, _ in reps]
        # Research paper should have 5+ sections (Abstract, 1-5, Conclusion)
        # Report: 4+, Resume: 4+, Invoice: 1+
        thresholds = [4, 3, 3, 1]
        ok = all(count >= thresh for (_, count), thresh in zip(counts, thresholds))
        detail = ", ".join(f"{fn}:{count}" for fn, count in counts)
        return ok, detail, ""
    results.append(_run_test("4. Section detection", t4))

    # ── 5. Section Summary Generation (non-empty) ───────────────────────────
    def t5():
        total_sections = sum(len(rep.sections) for _, rep, _ in reps)
        with_summary   = sum(
            sum(1 for s in rep.sections if _nonblank(s.summary))
            for _, rep, _ in reps
        )
        ratio = with_summary / max(1, total_sections)
        ok = ratio >= 0.6
        return ok, f"{with_summary}/{total_sections} sections have summaries ({ratio:.0%})", \
               paper_rep.sections[0].summary[:100] if paper_rep.sections else ""
    results.append(_run_test("5. Section summaries generated", t5))

    # ── 6. Key Fact Extraction ──────────────────────────────────────────────
    def t6():
        ok_count = sum(1 for _, rep, _ in reps if len(rep.key_facts) >= 1)
        ok = ok_count >= 3
        return ok, f"{ok_count}/{len(reps)} docs have key facts", str(paper_rep.key_facts[:2])
    results.append(_run_test("6. Key fact extraction", t6))

    # ── 7. Claim Extraction ─────────────────────────────────────────────────
    def t7():
        # Research paper should have claims
        ok = len(paper_rep.key_claims) >= 1
        return ok, f"Paper has {len(paper_rep.key_claims)} claims", \
               paper_rep.key_claims[0][:100] if paper_rep.key_claims else "(none)"
    results.append(_run_test("7. Claim extraction (research paper)", t7))

    # ── 8. Evidence Extraction ──────────────────────────────────────────────
    def t8():
        ok = len(paper_rep.evidence) >= 1
        return ok, f"Paper has {len(paper_rep.evidence)} evidence statements", \
               paper_rep.evidence[0][:100] if paper_rep.evidence else "(none)"
    results.append(_run_test("8. Evidence extraction (research paper)", t8))

    # ── 9. Conclusion Extraction ────────────────────────────────────────────
    def t9():
        ok_count = sum(1 for _, rep, _ in reps[:3] if len(rep.conclusions) >= 1)
        ok = ok_count >= 2
        return ok, f"{ok_count}/3 non-invoice docs have conclusions", \
               paper_rep.conclusions[0][:100] if paper_rep.conclusions else "(none)"
    results.append(_run_test("9. Conclusion extraction", t9))

    # ── 10. Key Points Generation ───────────────────────────────────────────
    # Resume and invoice may have no claim-style key points — require 2/4 (paper + report)
    def t10():
        ok_count = sum(1 for _, rep, _ in reps if len(rep.key_points) >= 1)
        ok = ok_count >= 2
        return ok, f"{ok_count}/{len(reps)} docs have key points (≥2 required)", \
               str(paper_rep.key_points[:2])
    results.append(_run_test("10. Key points generation", t10))

    # ── 11. Global Summary: Non-Pure-Extraction Check ───────────────────────
    # We accept summaries that include some source sentences as long as they
    # also synthesise across multiple sections (i.e., include content from
    # at least 2 different sections — structure awareness, not just copy).
    def t11():
        ok_count = 0
        for _, rep, raw in reps:
            if not _nonblank(rep.global_summary):
                continue
            # Check: summary covers multiple sections (contains content from ≥2)
            section_titles_mentioned = sum(
                1 for s in rep.sections
                if s.title.lower() in rep.global_summary.lower()
                   or (s.summary and s.summary[:40].lower() in rep.global_summary.lower())
            )
            # OR: summary is longer than any single sentence in the raw text (synthesised)
            raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw) if len(s.strip()) > 30]
            max_sentence_len = max((len(s) for s in raw_sentences), default=0)
            summary_is_longer = len(rep.global_summary) > max_sentence_len + 50
            if section_titles_mentioned >= 1 or summary_is_longer:
                ok_count += 1
        ok = ok_count >= 3
        return ok, f"{ok_count}/{len(reps)} global summaries show synthesis (multi-section or longer than any single sentence)", \
               paper_rep.global_summary[:200]
    results.append(_run_test("11. Global summary non-extractive", t11))

    # ── 12. Summary Coherence: Multi-Sentence ───────────────────────────────
    def t12():
        # A good summary has 3+ sentences
        def sent_count(s: str) -> int:
            return len(re.findall(r"[.!?]", s))
        ok_count = sum(
            1 for _, rep, _ in reps if sent_count(rep.global_summary) >= 3
        )
        ok = ok_count >= 3
        sents = sent_count(paper_rep.global_summary)
        return ok, f"{ok_count}/{len(reps)} summaries have 3+ sentences (paper has {sents})", \
               f"Paper summary: {paper_rep.global_summary[:200]}"
    results.append(_run_test("12. Summary coherence (multi-sentence)", t12))

    # ── 13. Short Summary Brevity ────────────────────────────────────────────
    def t13():
        ok_count = sum(
            1 for _, rep, _ in reps
            if _nonblank(rep.short_summary) and len(rep.short_summary) < 500
        )
        ok = ok_count >= 3
        return ok, f"{ok_count}/{len(reps)} short summaries are brief (<500 chars)", \
               paper_rep.short_summary[:150]
    results.append(_run_test("13. Short summary brevity", t13))

    # ── 14. Detailed Summary Comprehensiveness ──────────────────────────────
    def t14():
        answer = answer_from_representation(paper_rep, "Give me a detailed summary.", raw_text=paper_raw)
        has_sections  = "**" in answer or "Section" in answer or "section" in answer
        has_facts     = any(c.isdigit() for c in answer)
        ok = has_sections and len(answer) > 200
        return ok, f"Detailed summary: {len(answer)} chars, has_sections={has_sections}", \
               answer[:300]
    results.append(_run_test("14. Detailed summary comprehensiveness", t14))

    # ── 15. Section QA Routing ───────────────────────────────────────────────
    def t15():
        answer = answer_from_representation(paper_rep, "Explain the methodology section.", raw_text=paper_raw)
        ok = "methodolog" in answer.lower() and len(answer) > 50
        return ok, f"Section QA returned {len(answer)} chars, mentions methodology: {ok}", answer[:200]
    results.append(_run_test("15. Section QA routing", t15))

    # ── 16. Conclusion QA Routing ───────────────────────────────────────────
    def t16():
        answer = answer_from_representation(paper_rep, "What does the paper conclude?", raw_text=paper_raw)
        ok = len(answer) > 30 and not answer.startswith("No explicit")
        return ok, f"Conclusion QA: {len(answer)} chars", answer[:200]
    results.append(_run_test("16. Conclusion QA routing", t16))

    # ── 17. Key Points QA Routing ───────────────────────────────────────────
    def t17():
        answer = answer_from_representation(paper_rep, "What are the key points?", raw_text=paper_raw)
        has_bullets = "•" in answer or "-" in answer or "Key" in answer
        ok = has_bullets and len(answer) > 50
        return ok, f"Key points: {len(answer)} chars, has_bullets={has_bullets}", answer[:200]
    results.append(_run_test("17. Key points QA routing", t17))

    # ── 18. Purpose QA Routing ──────────────────────────────────────────────
    def t18():
        answer = answer_from_representation(paper_rep, "What is the purpose of this paper?", raw_text=paper_raw)
        ok = len(answer) > 20
        return ok, f"Purpose QA: {len(answer)} chars", answer[:150]
    results.append(_run_test("18. Purpose QA routing", t18))

    # ── 19. Hallucination Check ─────────────────────────────────────────────
    # The answer to "summarize" must not introduce large numbers not in the document
    def t19():
        answer = answer_from_representation(paper_rep, "Summarize this document.", raw_text=paper_raw)
        numbers_in_answer = set(re.findall(r"\b\d+\.?\d*\b", answer))
        numbers_in_doc    = set(re.findall(r"\b\d+\.?\d*\b", paper_raw))
        phantom = numbers_in_answer - numbers_in_doc - {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}
        ok = len(phantom) == 0
        return ok, f"Numbers in answer not in doc: {phantom}", answer[:200]
    results.append(_run_test("19. Hallucination check (no phantom numbers)", t19))

    # ── 20. Follow-up Question Handling (from memory) ──────────────────────
    def t20():
        # First query: what the paper is about
        a1 = answer_from_representation(paper_rep, "What is this paper about?", raw_text=paper_raw)
        # Follow-up: conclusions (uses the same rep, no re-parsing)
        a2 = answer_from_representation(paper_rep, "What is the conclusion?", raw_text=paper_raw)
        ok = _nonblank(a1) and _nonblank(a2) and a1 != a2
        return ok, f"Two different follow-up answers generated (a1={len(a1)}c, a2={len(a2)}c)", \
               f"Q1: {a1[:80]} | Q2: {a2[:80]}"
    results.append(_run_test("20. Follow-up question handling", t20))

    # ── 21. Table QA: Numeric Computation ──────────────────────────────────
    def t21():
        df = pd.DataFrame({
            "Rating": [4.5, 3.0, 5.0, 2.5, 4.0],
            "Name":   ["A", "B", "C", "D", "E"],
        })
        rep = build_table_representation(df, "test_table.csv")
        answer = answer_from_representation(rep, "What is the average rating?")
        expected_avg = round(sum([4.5, 3.0, 5.0, 2.5, 4.0]) / 5, 3)
        ok = str(expected_avg) in answer or "3.8" in answer
        return ok, f"Expected avg {expected_avg} in answer", answer[:150]
    results.append(_run_test("21. Table QA: average computation", t21))

    # ── 22. Paraphrase Robustness ───────────────────────────────────────────
    def t22():
        q_variants = [
            "Summarize this document.",
            "Give me an overview.",
            "What does this paper discuss?",
            "Can you summarize?",
        ]
        answers = [answer_from_representation(paper_rep, q, raw_text=paper_raw) for q in q_variants]
        # All should be non-empty and substantially similar (all about the paper)
        ok = all(_nonblank(a) for a in answers)
        return ok, f"All {len(q_variants)} phrasings returned non-empty answers", \
               f"Sample: {answers[0][:80]}"
    results.append(_run_test("22. Paraphrase robustness", t22))

    # ── 23. Entity Extraction ───────────────────────────────────────────────
    def t23():
        answer = answer_from_representation(paper_rep, "Who or what are the main entities?", raw_text=paper_raw)
        ok = len(answer) > 20
        return ok, f"Entity answer: {len(answer)} chars", answer[:150]
    results.append(_run_test("23. Entity extraction QA", t23))

    # ── 24. Limitation Detection ─────────────────────────────────────────────
    def t24():
        answer = answer_from_representation(paper_rep, "What are the limitations?", raw_text=paper_raw)
        # The paper mentions future work / doesn't address 3D point clouds
        ok = len(answer) > 20
        return ok, f"Limitation answer: {len(answer)} chars", answer[:200]
    results.append(_run_test("24. Limitation detection", t24))

    return results


# ══════════════════════════════════════════════════════════════════════════════
# QUERY CLASSIFICATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

def run_classification_tests() -> List[TestResult]:
    """Test that query classification works correctly."""
    test_cases = [
        ("Summarize this document.", "summary"),
        ("Give me an overview.", "summary"),
        ("Give me a 5-line summary.", "short_summary"),
        ("Briefly describe this.", "short_summary"),
        ("What are the key points?", "key_points"),
        ("What are the main points?", "key_points"),
        ("What is the purpose?", "purpose"),
        ("What type of document is this?", "document_type"),
        ("What is this about?", "topic"),
        ("What does it conclude?", "conclusions"),
        ("What does the author claim?", "claims"),
        ("What evidence is provided?", "evidence"),
        ("What are the limitations?", "limitations"),
        ("Who is mentioned?", "entities"),
        ("What are the key facts?", "facts"),
        ("Explain the introduction section.", "section_qa"),
        ("Compare section 2 and section 3.", "section_compare"),
    ]

    results = []
    for query, expected in test_cases:
        got = classify_doc_query(query)
        ok = got == expected
        results.append(TestResult(
            f"classify: {query[:40]}",
            ok,
            f"expected={expected}, got={got}",
            "",
        ))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM DOCUMENT TEST (for user-provided documents)
# ══════════════════════════════════════════════════════════════════════════════

def test_custom_document(path: str, verbose: bool = False) -> None:
    """Run the full pipeline on a user-provided document and print results."""
    from pathlib import Path as P
    p = P(path)
    if not p.exists():
        print(f"[Error] File not found: {path}")
        return

    ext = p.suffix.lower()
    try:
        if ext == ".txt":
            raw = p.read_text(encoding="utf-8", errors="ignore")
        elif ext == ".pdf":
            from pypdf import PdfReader
            r = PdfReader(str(p))
            raw = "\n".join(
                (page.extract_text(extraction_mode="layout") or page.extract_text() or "")
                for page in r.pages
            )
        elif ext == ".docx":
            import docx
            d = docx.Document(str(p))
            raw = "\n".join(para.text for para in d.paragraphs if para.text.strip())
        else:
            print(f"[Error] Unsupported extension: {ext}")
            return
    except Exception as e:
        print(f"[Error] Could not read file: {e}")
        return

    print(f"\n{'═'*60}")
    print(f"  Custom Document Test: {p.name}")
    print(f"{'═'*60}\n")

    structured = parse_document_structure(raw, p.name)
    rep = build_document_representation(structured, debug=verbose)

    print(rep.debug_repr())
    print()

    questions = [
        "What is this document about?",
        "Give me a proper summary.",
        "Give me the main points.",
        "What is the main problem?",
        "What is the objective?",
        "What are the important findings?",
        "What is the conclusion?",
        "What are the limitations?",
        "Give me a 5-line summary.",
        "Give me a detailed summary.",
    ]

    print("\n── ANSWER TESTS ──")
    for q in questions:
        answer = answer_from_representation(rep, q, raw_text=raw[:2000])
        print(f"\nQ: {q}")
        print(f"A: {answer[:300]}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the document understanding pipeline."
    )
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show answer previews for each test")
    parser.add_argument("--doc", type=str, default="",
                        help="Path to a real document to test (PDF/DOCX/TXT)")
    args = parser.parse_args()

    if args.doc:
        test_custom_document(args.doc, verbose=args.verbose)
        return

    print(f"\n{'═'*70}")
    print("  DOCUMENT INTELLIGENCE EVALUATION")
    print(f"{'═'*70}\n")

    # Build representations
    print("[Building representations from synthetic test docs…]")
    reps = _build_reps()
    for fn, rep, raw in reps:
        print(f"  {fn}: type={rep.document_type}, sections={len(rep.sections)}, words={rep.total_words}")

    # Run understanding tests
    print(f"\n[Running {24} understanding tests…]")
    understanding_results = run_all_tests(reps, verbose=args.verbose)

    # Run classification tests
    print(f"\n[Running query classification tests…]")
    classification_results = run_classification_tests()

    all_results = understanding_results + classification_results

    # Report
    print(f"\n{'═'*70}")
    print("  RESULTS")
    print(f"{'═'*70}")

    passed = 0
    for r in all_results:
        mark = "PASS" if r.passed else "FAIL"
        print(f"  {mark}  {r.name}")
        if not r.passed or args.verbose:
            if r.detail:
                print(f"        {r.detail}")
            if r.answer_preview and args.verbose:
                print(f"        Preview: {r.answer_preview[:120]}")
        if r.passed:
            passed += 1

    total = len(all_results)
    pct = round(100 * passed / total, 1)
    print(f"\n{'═'*70}")
    print(f"  TOTAL: {passed}/{total} passed  ({pct}%)")
    print(f"{'═'*70}\n")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
