"""
document_answer_engine.py
═══════════════════════════════════════════════════════════════════════════════
The deterministic document answering layer.

This is the document equivalent of SmartFallbackEngine in scratch_llm.py.
SmartFallbackEngine guarantees correct answers from sales CSV data without
depending on the neural model. DocumentAnswerEngine guarantees grounded,
coherent answers from DocumentRepresentation without depending on the neural
model.

The neural model (SalesGPT) may optionally REPHRASE the deterministic answer
into more natural language if it has been trained on document tasks. It never
generates the factual content.

Supported query types:
  SUMMARY       — "summarize this", "overview", "what is this about"
  SHORT_SUMMARY — "5-line summary", "brief", "in short"
  KEY_POINTS    — "key points", "main points", "highlights", "imp points"
  SECTION_QA    — "what does section X say", "explain section X"
  DOCUMENT_TYPE — "what type of document is this"
  PURPOSE       — "what is the purpose", "what is this for", "why was this written"
  TOPIC         — "what is this about", "main topic", "subject"
  FACTS_QA      — "what are the key facts", "important findings"
  CONCLUSIONS   — "what are the conclusions", "what does it conclude"
  CLAIMS_QA     — "what does the author claim", "main argument"
  EVIDENCE_QA   — "what evidence is provided", "how is it supported"
  LIMITATIONS   — "what are the limitations"
  ENTITY_QA     — "who/what are the main entities/people/organisations"
  SECTION_COMPARE — "compare section X and section Y"
  GENERAL_QA    — TF-IDF sentence retrieval over full text
  TABLE_QA      — pandas computation for numeric questions on CSV/Excel
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from document_understanding import DocumentRepresentation, SectionRepresentation


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — QUERY CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

_QUERY_PATTERNS: Dict[str, List[str]] = {
    "summary": [
        r"\b(summar(?:y|ize|ise|ization|isation)|overview|"
        r"tell me about|about this (?:document|file|pdf)|"
        r"what is this (?:document|file|paper|report) about|"
        r"explain (?:this|the) (?:document|file|paper|report)|"
        r"what does this (?:document|say|contain|discuss|cover)|"
        r"give me a (?:complete|full|detailed) summary|"
        r"comprehensive summary)\b",
        r"\bexplain.{0,20}(?:in simple|simply|simply|in plain|to a layman|"
        r"easy to understand|plain language|simple language|simple terms|plain terms)\b",
        r"\bdescribe.{0,30}(?:document|file|paper|report|this)\b",
    ],
    "short_summary": [
        r"\b((?:5|five|3|three|2|two)[- ]line|brief(?:ly)?\s+(?:summar|descri|explain|overview)|"
        r"in short|tl;?dr|quick summary|short summary|in brief|gist|tldr|brief overview)\b",
        r"\bgive me a (?:5|five|3|three|2|two)[- ]line\b",
        r"\bbriefly (?:describe|summarize|summarise|explain)\b",
    ],
    "key_points": [
        r"\b((?:key|main|important|imp|core|critical|major|top)\s+"
        r"(?:points?|ideas?|takeaways?|highlights?|insights?|concepts?)|"
        r"highlights?|takeaways?|important points?|main ideas?|"
        r"what are the (?:key|main|important) (?:things?|points?|ideas?))\b"
    ],
    "purpose": [
        r"\b(purpose|why (?:was|is) (?:this|it) (?:written|created|made|published)|"
        r"what (?:is|was) the (?:goal|aim|objective|purpose|reason) of|"
        r"why does this (?:document|paper|report) exist)\b"
    ],
    "document_type": [
        r"\b(what (?:type|kind|sort) (?:is|of) (?:this|document)|"
        r"document type|is this a (?:report|paper|resume|invoice|article))\b"
    ],
    "topic": [
        r"\b(what (?:is (?:this|it) about|topic|subject|main (?:topic|theme|subject)|"
        r"is the (?:subject|topic))|main theme)\b"
    ],
    "conclusions": [
        r"\b(conclusion|what (?:does it|did (?:they|the (?:author|study|paper|report))) "
        r"conclude|what (?:is|are) the (?:conclusion|result|outcome|finding)|"
        r"final(?:ly)?|end result|overall finding)\b"
    ],
    "claims": [
        r"\b(what (?:does|did) the (?:author|paper|study|report|document) (?:claim|argue|say|state|assert)|"
        r"main (?:claim|argument|thesis|assertion|argument)|"
        r"what (?:is|are) the (?:claim|argument|thesis))\b"
    ],
    "evidence": [
        r"\b(evidence|how (?:is it|are (?:they|the claims?)) (?:supported|backed|proven|demonstrated)|"
        r"what (?:evidence|proof|data|support) (?:is|are) (?:provided|given|used|presented)|"
        r"support(?:ing|ed by))\b"
    ],
    "limitations": [
        r"\blimitation",
        r"\bweakness\b",
        r"\bdrawback\b",
        r"\bshortcoming\b",
        r"\bwhat (?:are|is|were) the (?:limit|weak|draw|short|constraint)",
        r"\bwhat (?:doesn'?t|does not|cannot|can'?t) (?:it|this|the (?:model|system|method|paper|document)) (?:do|cover|handle|address)",
    ],
    "entities": [
        r"\b(who (?:is|are) (?:mentioned|involved|the (?:author|person|people|key (?:person|people)))|"
        r"main (?:entities|people|organisations?|organizations?|companies)|"
        r"key (?:names?|entities?|stakeholders?))\b"
    ],
    "section_qa": [
        r"\b(section\s+\d+|(?:the\s+)?(?:abstract|introduction|methodology|methods?|"
        r"results?|conclusion|discussion|background|literature|related work|"
        r"education|experience|skills?|projects?|outlook|business review|"
        r"financial highlights?|challenges?)\s*(?:section|part)?|"
        r"explain\s+(?:the\s+)?(?:abstract|introduction|methodology|methods?|results?|"
        r"conclusion|background|literature|education|experience|skills))\b",
        r"\b(what (?:method(?:ology)?|approach|technique|algorithm|model|system|"
        r"framework|architecture) (?:is|was|are|were) (?:used?|proposed?|adopted?|employed?))\b",
        r"\b(what (?:is|was) the (?:main )?(?:problem|challenge|issue|objective|approach|"
        r"contribution|innovation|novelty))\b",
    ],
    "section_compare": [
        r"\b(compare|contrast)\b.{0,60}\b(section|part|chapter)\b",
        r"\bdifference between.{0,60}\b(section|part)\b",
        r"\b(section|part)\b.{0,40}\b(and|vs\.?|versus)\b.{0,40}\b(section|part)\b",
        r"\brelation(?:ship)? between.{0,60}\b(section|part)\b",
    ],
    "facts": [
        r"\b((?:key|main|important) (?:facts?|findings?|statistics?|numbers?|data|"
        r"figures?|results?)|what (?:facts?|findings?|results?) (?:does|did|were)|"
        r"numerical|numbers?|statistics?)\b"
    ],
}

_SUMMARY_MODES = {"summary", "short_summary", "key_points"}


_PRIORITY_ORDER = [
    "section_compare",   # must be before section_qa (more specific)
    "short_summary",     # must be before summary (more specific)
    "key_points",
    "purpose",
    "document_type",
    "topic",
    "conclusions",
    "claims",
    "evidence",
    "limitations",
    "entities",
    "facts",
    "section_qa",        # after section_compare
    "summary",           # after short_summary
]


def classify_doc_query(query: str) -> str:
    """
    Classify a query into one of the supported query types.
    Returns the type name string, or 'general_qa' if nothing specific matches.
    Checks in priority order so more-specific types win over general ones.
    """
    q = query.lower().strip()

    for qtype in _PRIORITY_ORDER:
        for pat in _QUERY_PATTERNS.get(qtype, []):
            if re.search(pat, q, re.I):
                return qtype

    return "general_qa"


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — SUMMARY MODE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _detect_summary_mode(query: str) -> str:
    """
    Returns: 'short' | 'detailed' | 'executive' | 'normal' | 'section_by_section'
    """
    q = query.lower()
    if re.search(r"\b(5|five|3|three|2|two)[- ]line|brief(?:ly)?|in short|tl;?dr|quick|short\b", q):
        return "short"
    if re.search(r"\b(detailed?|comprehensive|full|complete|thorough|in-depth|exhaustive)\b", q):
        return "detailed"
    if re.search(r"\bexecutive\b", q):
        return "executive"
    if re.search(r"\bsection.by.section\b", q):
        return "section_by_section"
    return "normal"


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — ANSWER GENERATORS (one per query type)
# ══════════════════════════════════════════════════════════════════════════════

def _answer_summary(rep: DocumentRepresentation, query: str) -> str:
    mode = _detect_summary_mode(query)

    if mode == "short":
        return _answer_short_summary(rep)
    if mode == "section_by_section":
        return _answer_section_by_section(rep)
    if mode == "executive":
        return _answer_executive_summary(rep)
    if mode == "detailed":
        return _answer_detailed_summary(rep)
    return _answer_normal_summary(rep)


def _answer_normal_summary(rep: DocumentRepresentation) -> str:
    """
    The main summary answer. Synthesised from the DocumentRepresentation.
    NOT extractive — builds narrative structure.
    """
    if not rep.global_summary:
        return _answer_short_summary(rep)

    parts = [f"**Summary of {rep.filename}**\n"]
    parts.append(rep.global_summary)

    import re as _re
    clean_kp = [_re.sub(r"\s*\n\s*", " ", pt).strip()
                for pt in rep.key_points if pt and len(pt.strip()) > 10]
    clean_kp = [pt for pt in clean_kp if len(pt) > 10]
    if clean_kp:
        parts.append("\n**Key points:**")
        for pt in clean_kp[:5]:
            parts.append(f"- {pt}")

    if rep.conclusions:
        parts.append(f"\n**Main conclusion:** {rep.conclusions[0]}")

    return "\n".join(parts)


def _answer_short_summary(rep: DocumentRepresentation) -> str:
    if rep.short_summary:
        return rep.short_summary
    if rep.global_summary:
        # Take first 2-3 sentences of global summary
        sentences = re.split(r"(?<=[.!?])\s+", rep.global_summary)
        return " ".join(sentences[:3])
    return rep.purpose or f"This document is about {rep.main_topic}."


def _answer_detailed_summary(rep: DocumentRepresentation) -> str:
    parts = [f"**Detailed Summary: {rep.filename}**\n"]

    # Document overview
    parts.append(f"**Document type:** {rep.document_type.replace('_', ' ').title()}")
    parts.append(f"**Purpose:** {rep.purpose}")
    parts.append(f"**Main topic:** {rep.main_topic}\n")

    # Section-by-section content
    if rep.sections:
        parts.append("**Section summaries:**")
        for sec in sorted(rep.sections, key=lambda s: s.order):
            if sec.importance < 0.1 or not sec.summary:
                continue
            parts.append(f"\n*{sec.title}*")
            parts.append(sec.summary)
            if sec.facts:
                parts.append("Key data points: " + "; ".join(sec.facts[:2]))

    # Entities
    if rep.entities:
        parts.append(f"\n**Key entities/topics:** {', '.join(rep.entities[:8])}")

    # Facts
    if rep.key_facts:
        parts.append("\n**Key facts:**")
        for f in rep.key_facts[:5]:
            parts.append(f"- {f}")

    # Conclusions
    if rep.conclusions:
        parts.append("\n**Conclusions:**")
        for c in rep.conclusions[:3]:
            parts.append(f"- {c}")

    return "\n".join(parts)


def _answer_executive_summary(rep: DocumentRepresentation) -> str:
    parts = ["**Executive Summary**\n"]
    if rep.purpose:
        parts.append(rep.purpose)
    if rep.key_points:
        parts.append("\nKey highlights:")
        for pt in rep.key_points[:4]:
            parts.append(f"• {pt}")
    if rep.conclusions:
        parts.append(f"\nConclusion: {rep.conclusions[0]}")
    return "\n".join(parts)


def _answer_section_by_section(rep: DocumentRepresentation) -> str:
    if not rep.sections:
        return _answer_normal_summary(rep)
    parts = [f"**Section-by-section summary of {rep.filename}:**\n"]
    for sec in sorted(rep.sections, key=lambda s: s.order):
        if not sec.summary and not sec.raw_text:
            continue
        parts.append(f"**{sec.title}**")
        if sec.summary:
            parts.append(sec.summary)
        elif sec.raw_text:
            # First 150 chars of the section
            parts.append(sec.raw_text[:150] + "...")
        parts.append("")
    return "\n".join(parts)


def _answer_key_points(rep: DocumentRepresentation) -> str:
    import re as _re
    clean_kp = [_re.sub(r"\s*\n\s*", " ", pt).strip()
                for pt in rep.key_points if pt and len(pt.strip()) > 10]
    clean_kp = [pt for pt in clean_kp if len(pt) > 10]
    if clean_kp:
        lines = ["**Key points:**"]
        for pt in clean_kp:
            lines.append(f"• {pt}")
        return "\n".join(lines)
    # Fall back to section summaries as key points
    lines = ["**Key points from each section:**"]
    for sec in sorted(rep.sections, key=lambda s: -s.importance)[:5]:
        if sec.summary:
            lines.append(f"• **{sec.title}**: {sec.summary.split('.')[0]}.")
    return "\n".join(lines) if len(lines) > 1 else _answer_short_summary(rep)


def _answer_purpose(rep: DocumentRepresentation) -> str:
    if rep.purpose:
        return f"**Purpose:** {rep.purpose}"
    return f"The purpose of this document is to {rep.document_type.replace('_', ' ')} {rep.main_topic}."


def _answer_document_type(rep: DocumentRepresentation) -> str:
    dtype = rep.document_type.replace("_", " ").title()
    return f"This is a **{dtype}** document about {rep.main_topic}."


def _answer_topic(rep: DocumentRepresentation) -> str:
    if rep.main_topic:
        return f"This document is about **{rep.main_topic}**. {rep.purpose}"
    return rep.purpose or rep.global_summary[:200]


def _answer_conclusions(rep: DocumentRepresentation) -> str:
    import re as _re
    clean_conc = [_re.sub(r"\s*\n\s*", " ", c).strip()
                  for c in rep.conclusions if c and len(c) > 15]
    clean_conc = [c for c in clean_conc if len(c) > 15]
    if clean_conc:
        lines = ["**Conclusions:**"]
        for c in clean_conc:
            lines.append(f"• {c}")
        return "\n".join(lines)
    # Look in conclusion/summary sections
    for sec in rep.sections:
        if any(kw in sec.title.lower() for kw in ("conclusion", "summary", "finding", "result")):
            if sec.summary:
                return f"**Conclusion:** {sec.summary}"
    return "No explicit conclusion section was detected. The document's main finding is: " + (rep.short_summary or rep.purpose)


def _answer_claims(rep: DocumentRepresentation) -> str:
    if rep.key_claims:
        lines = ["**Main claims/arguments:**"]
        for c in rep.key_claims[:5]:
            lines.append(f"• {c}")
        return "\n".join(lines)
    return f"The document's main argument is: {rep.purpose}"


def _answer_evidence(rep: DocumentRepresentation) -> str:
    if rep.evidence:
        lines = ["**Evidence/support provided:**"]
        for e in rep.evidence[:5]:
            lines.append(f"• {e}")
        return "\n".join(lines)
    if rep.key_facts:
        lines = ["**Supporting facts:**"]
        for f in rep.key_facts[:4]:
            lines.append(f"• {f}")
        return "\n".join(lines)
    return "The document does not include explicit evidence statements in a recognisable format, or they could not be automatically extracted."


def _answer_limitations(rep: DocumentRepresentation) -> str:
    # Search for limitation content in sections and section text
    limitation_texts = []
    for sec in rep.sections:
        text = sec.raw_text.lower()
        if "limitation" in sec.title.lower() or "limitation" in text[:200]:
            # Extract sentences mentioning limitations
            sentences = re.split(r"(?<=[.!?])\s+", sec.raw_text)
            for sent in sentences:
                if re.search(r"\b(limitation|limitation|drawback|shortcoming|weakness|however|"
                             r"challenge|constraint|future work|do not|cannot|can\'t|"
                             r"not (?:able to|capable of))\b", sent, re.I):
                    if len(sent) > 20:
                        limitation_texts.append(sent.strip())
        else:
            # Also scan any section for limitation-language
            sentences = re.split(r"(?<=[.!?])\s+", sec.raw_text)
            for sent in sentences:
                if re.search(r"\b(limitation|drawback|weakness|however we|one limitation|"
                             r"this (?:method|approach|system) (?:does not|cannot|fails?))\b",
                             sent, re.I):
                    if len(sent) > 20:
                        limitation_texts.append(sent.strip())

    limitation_texts = list(dict.fromkeys(limitation_texts))[:5]

    if limitation_texts:
        lines = ["**Limitations identified in the document:**"]
        for t in limitation_texts:
            lines.append(f"• {t}")
        return "\n".join(lines)
    return "No explicit limitations section was detected in this document, or limitations were not stated in a recognisable form."


def _answer_entities(rep: DocumentRepresentation) -> str:
    if rep.entities:
        return f"**Key entities/names mentioned:** {', '.join(rep.entities[:12])}"
    return "No specific named entities could be reliably extracted from this document."


def _answer_facts(rep: DocumentRepresentation) -> str:
    all_facts = rep.all_facts()
    if all_facts:
        lines = ["**Key facts from the document:**"]
        for f in all_facts[:8]:
            lines.append(f"• {f}")
        return "\n".join(lines)
    return f"The document covers {rep.main_topic}. {rep.short_summary}"


def _answer_section_qa(rep: DocumentRepresentation, query: str) -> str:
    """Answer a question about a specific named section or conceptual aspect."""
    q = query.lower()
    target_section: Optional[SectionRepresentation] = None

    # --- Direct section name match ---
    for sec in rep.sections:
        title_words = re.findall(r"[a-z]{4,}", sec.title.lower())
        if sec.title.lower() in q or any(w in q for w in title_words):
            target_section = sec
            break

    # --- Numbered section: "section 3" ---
    if not target_section:
        m = re.search(r"section\s+(\d+)", q, re.I)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(rep.sections):
                target_section = rep.sections[idx]

    # --- Semantic concept → section mapping ---
    if not target_section:
        concept_map = {
            ("problem", "challenge", "issue", "motivation"):
                ("introduction", "background", "problem", "motivation"),
            ("method", "methodology", "approach", "technique", "algorithm", "model",
             "architecture", "framework", "system", "proposed"):
                ("methodology", "method", "approach", "proposed", "system", "model"),
            ("result", "finding", "performance", "accuracy", "score", "benchmark", "evaluation"):
                ("result", "evaluation", "experiment", "finding", "performance"),
            ("contribution", "novelty", "innovation", "objective", "goal", "aim"):
                ("abstract", "introduction", "contribution", "objective"),
            ("future", "limitation", "drawback", "weakness"):
                ("conclusion", "future", "limitation", "discussion"),
        }
        for query_keywords, section_keywords in concept_map.items():
            if any(kw in q for kw in query_keywords):
                for sec in rep.sections:
                    if any(kw in sec.title.lower() for kw in section_keywords):
                        target_section = sec
                        break
                if target_section:
                    break

    if target_section and (target_section.summary or target_section.raw_text):
        parts = [f"**{target_section.title}**\n"]
        if target_section.summary:
            parts.append(target_section.summary)
        elif target_section.raw_text:
            # First 300 chars of section text as fallback
            parts.append(target_section.raw_text[:300].strip())
        if target_section.facts:
            parts.append("\nKey facts:")
            for f in target_section.facts[:3]:
                parts.append(f"• {f}")
        if target_section.conclusions:
            parts.append(f"\nConclusion: {target_section.conclusions[0]}")
        return "\n".join(parts)

    return ""    # caller uses general_qa fallback


def _answer_section_compare(rep: DocumentRepresentation, query: str) -> str:
    """Compare two named sections."""
    mentioned = []
    q = query.lower()

    # Build a scored list: how many words from each section title appear in query
    scored = []
    for sec in rep.sections:
        title_words = re.findall(r"[a-z]{4,}", sec.title.lower())
        score = sum(1 for w in title_words if w in q)
        if title_words and score > 0:
            scored.append((score, sec.order, sec))

    # Sort: highest match score first, then document order
    scored.sort(key=lambda x: (-x[0], x[1]))
    # Take top 2 distinct sections
    seen_orders = set()
    for score, order, sec in scored:
        if order not in seen_orders:
            mentioned.append(sec)
            seen_orders.add(order)
        if len(mentioned) >= 2:
            break

    if len(mentioned) < 2:
        return ""   # fallback to general_qa

    s1, s2 = mentioned[0], mentioned[1]
    lines = [f"**Comparison: {s1.title} vs {s2.title}**\n"]

    lines.append(f"**{s1.title}:**")
    lines.append(s1.summary or s1.raw_text[:200])
    if s1.facts:
        lines.append("Key data: " + "; ".join(s1.facts[:2]))

    lines.append(f"\n**{s2.title}:**")
    lines.append(s2.summary or s2.raw_text[:200])
    if s2.facts:
        lines.append("Key data: " + "; ".join(s2.facts[:2]))

    # Relationship commentary
    lines.append("\n**Relationship:**")
    if s1.order < s2.order:
        lines.append(f"{s1.title} appears earlier in the document and likely "
                     f"provides context or background for {s2.title}.")
    return "\n".join(lines)


def _answer_general_qa(rep: DocumentRepresentation, query: str,
                        raw_text: str = "") -> str:
    """
    TF-IDF sentence-level retrieval over the document's full text.
    This is the fallback when no structured answer is available.
    Used for specific questions not covered by the structured handlers.
    """
    # Collect all text
    text_sources = []
    if raw_text:
        text_sources.append(raw_text)
    for sec in rep.sections:
        if sec.raw_text:
            text_sources.append(sec.raw_text)

    combined_text = "\n".join(text_sources) if text_sources else ""

    if not combined_text.strip():
        return rep.short_summary or rep.purpose or "No content available."

    # Split into sentences
    sentences = []
    for line in combined_text.splitlines():
        line = line.strip()
        if not line:
            continue
        for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", line):
            sent = sent.strip()
            if len(sent) > 15:
                sentences.append(sent)

    # Deduplicate
    seen = set()
    uniq_sentences = []
    for s in sentences:
        key = re.sub(r"\s+", " ", s.lower())
        if key not in seen:
            seen.add(key)
            uniq_sentences.append(s)

    if not uniq_sentences:
        return rep.short_summary or rep.purpose or ""

    if len(uniq_sentences) < 3:
        return "\n".join(uniq_sentences)

    # TF-IDF ranking
    try:
        all_texts = uniq_sentences + [query]
        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
        matrix = vec.fit_transform(all_texts)
        query_vec = matrix[-1]
        doc_matrix = matrix[:-1]
        scores = cosine_similarity(query_vec, doc_matrix).flatten()
        ranked = np.argsort(-scores)[:5]
        # Filter by minimum score
        good = [i for i in ranked if scores[i] >= 0.05]
        if not good:
            good = ranked[:3].tolist()
        # Return in document order
        good_sorted = sorted(good)
        results = [uniq_sentences[i] for i in good_sorted]

        if len(results) == 1:
            return results[0]
        return "\n".join(f"- {s}" for s in results)

    except Exception:
        return "\n".join(uniq_sentences[:3])


def _answer_table_qa(rep: DocumentRepresentation, query: str) -> Optional[str]:
    """
    Answer numerical questions about an uploaded CSV/Excel using pandas.
    Returns None if not a table or if the question doesn't map to a computation.
    """
    if rep.dataframe is None:
        return None
    df = rep.dataframe
    q = query.lower()

    # Average / mean
    m = re.search(r"\b(average|mean|avg)\b.{0,40}?(\w[\w ]{1,30})\b", q)
    if m:
        target = m.group(2).strip()
        for col in df.columns:
            if target in col.lower() or col.lower() in target:
                if pd.api.types.is_numeric_dtype(df[col]):
                    val = round(float(df[col].dropna().mean()), 3)
                    return f"The average {col} is **{val}** (computed from {df[col].dropna().count()} rows)."

    # Count
    if re.search(r"\b(how many|count|number of|total)\b", q):
        return f"The dataset has **{len(df)}** rows and **{len(df.columns)}** columns."

    # Maximum
    m = re.search(r"\b(maximum|highest|max|largest)\b.{0,40}?(\w[\w ]{1,30})\b", q)
    if m:
        target = m.group(2).strip()
        for col in df.columns:
            if target in col.lower() or col.lower() in target:
                if pd.api.types.is_numeric_dtype(df[col]):
                    val = df[col].dropna().max()
                    return f"The maximum {col} is **{val}**."

    # Minimum
    m = re.search(r"\b(minimum|lowest|min|smallest)\b.{0,40}?(\w[\w ]{1,30})\b", q)
    if m:
        target = m.group(2).strip()
        for col in df.columns:
            if target in col.lower() or col.lower() in target:
                if pd.api.types.is_numeric_dtype(df[col]):
                    val = df[col].dropna().min()
                    return f"The minimum {col} is **{val}**."

    return None


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — MAIN ANSWER BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def answer_from_representation(
    rep: DocumentRepresentation,
    query: str,
    raw_text: str = "",
    debug: bool = False,
) -> str:
    """
    The main entry point for document Q&A.
    Takes a DocumentRepresentation and a query, returns a grounded answer.

    This is the document equivalent of SmartFallbackEngine.answer().
    It NEVER returns "I don't know" when the document has relevant content.
    It NEVER introduces facts not found in the DocumentRepresentation.

    Pipeline:
      1. Classify query type
      2. For table docs: try pandas computation first
      3. Route to the appropriate deterministic answer generator
      4. Fall back to general TF-IDF sentence retrieval

    debug=True prints the query classification and routing decision.
    """
    query_type = classify_doc_query(query)

    if debug:
        print(f"\n[DocumentAnswerEngine] Query: {query!r}")
        print(f"[DocumentAnswerEngine] Classified as: {query_type}")

    # --- Table-specific: pandas computation first ---
    if rep.dataframe is not None:
        table_answer = _answer_table_qa(rep, query)
        if table_answer:
            if debug:
                print("[DocumentAnswerEngine] → Table computation")
            return table_answer

    # --- Route by query type ---
    answer = None

    if query_type in ("summary", "short_summary"):
        answer = _answer_summary(rep, query)
    elif query_type == "key_points":
        answer = _answer_key_points(rep)
    elif query_type == "purpose":
        answer = _answer_purpose(rep)
    elif query_type == "document_type":
        answer = _answer_document_type(rep)
    elif query_type == "topic":
        answer = _answer_topic(rep)
    elif query_type == "conclusions":
        answer = _answer_conclusions(rep)
    elif query_type == "claims":
        answer = _answer_claims(rep)
    elif query_type == "evidence":
        answer = _answer_evidence(rep)
    elif query_type == "limitations":
        answer = _answer_limitations(rep)
    elif query_type == "entities":
        answer = _answer_entities(rep)
    elif query_type == "facts":
        answer = _answer_facts(rep)
    elif query_type == "section_qa":
        answer = _answer_section_qa(rep, query)
    elif query_type == "section_compare":
        answer = _answer_section_compare(rep, query)

    # If structured handler returned empty string or None → fall back to retrieval
    if not answer or len(answer.strip()) < 10:
        if debug:
            print(f"[DocumentAnswerEngine] Structured handler returned empty → general_qa")
        answer = _answer_general_qa(rep, query, raw_text)

    if debug:
        print(f"[DocumentAnswerEngine] Answer ({len(answer)} chars): {answer[:100]}...")

    return answer.strip() if answer else (rep.short_summary or rep.purpose or "No answer found.")
