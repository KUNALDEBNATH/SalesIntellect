"""
document_understanding.py
═══════════════════════════════════════════════════════════════════════════════
Semantic document understanding layer.

This module sits BETWEEN document_parser.py (structure extraction) and
attachment_handler.py (answer generation). Its job is identical to what
SmartFallbackEngine does for sales CSV data: produce correct, grounded
answers from the document's own content without depending on the neural
model's untrained ability to "understand" document text.

Architecture position:
  document_parser  →  StructuredDocument
  document_understanding  →  DocumentRepresentation  ←  (this file)
  DocumentAnswerEngine  →  final answer text

Key design principles
─────────────────────
1. DETERMINISTIC FIRST. Every answer is built from rule-based analysis of
   the document's own text. The neural model is never the primary source of
   facts.

2. NO CONFUSION OF EXTRACTION WITH UNDERSTANDING. We extract text in the
   parser. Here we ANALYSE the extracted text to build meaning structures:
   document type, purpose, sections, section summaries, entities, facts,
   claims, evidence, conclusions.

3. SECTION-LEVEL SUMMARIES are rule-based compressions, not TextRank.
   A section summary = topic sentence + key facts + conclusion sentence.
   This is deterministic and always grounded.

4. GLOBAL SUMMARY is synthesised from section summaries, not from ranking
   raw sentences. Document-level meaning comes from section-level meaning.

5. DOCUMENT MEMORY stores the DocumentRepresentation so follow-up questions
   don't re-parse. The representation is rich enough to answer any follow-up
   from memory without re-running the full pipeline.

6. The neural SalesGPT model (if trained on document tasks) may optionally
   REPHRASE the deterministic answer into more fluent natural language.
   It never generates the factual content — only the phrasing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — STRUCTURED DOCUMENT (output of the enhanced parser)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TableData:
    """A preserved table from the document."""
    caption: str = ""
    headers: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    source_section: str = ""

    def to_text(self) -> str:
        lines = []
        if self.caption:
            lines.append(f"Table: {self.caption}")
        if self.headers:
            lines.append(" | ".join(self.headers))
        for row in self.rows[:10]:
            lines.append(" | ".join(str(c) for c in row))
        return "\n".join(lines)

    def to_dataframe(self) -> Optional[pd.DataFrame]:
        if not self.rows:
            return None
        return pd.DataFrame(self.rows, columns=self.headers if self.headers else None)


@dataclass
class DocumentSection:
    """One logical section of the document with its content."""
    title: str
    level: int = 1           # 1=top-level, 2=subsection, 3=sub-subsection
    paragraphs: List[str] = field(default_factory=list)
    tables: List[TableData] = field(default_factory=list)
    lists: List[List[str]] = field(default_factory=list)
    order: int = 0           # position in document (0-indexed)

    @property
    def full_text(self) -> str:
        parts = []
        if self.title:
            parts.append(self.title)
        parts.extend(self.paragraphs)
        for t in self.tables:
            parts.append(t.to_text())
        for lst in self.lists:
            parts.extend(f"- {item}" for item in lst)
        return "\n".join(parts)

    @property
    def word_count(self) -> int:
        return len(self.full_text.split())


@dataclass
class StructuredDocument:
    """
    Output of the enhanced document parser.
    Structure is preserved; meaning is NOT yet computed here.
    """
    filename: str
    kind: str                              # "text" | "table" | "image"
    raw_text: str = ""
    pages: List[str] = field(default_factory=list)
    sections: List[DocumentSection] = field(default_factory=list)
    tables: List[TableData] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)
    dataframe: Optional[pd.DataFrame] = None   # for uploaded CSV/Excel

    @property
    def section_count(self) -> int:
        return len(self.sections)

    @property
    def total_words(self) -> int:
        return len(self.raw_text.split())

    def get_section(self, title_fragment: str) -> Optional[DocumentSection]:
        frag = title_fragment.lower()
        for s in self.sections:
            if frag in s.title.lower():
                return s
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — DOCUMENT REPRESENTATION (semantic output of this module)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SectionRepresentation:
    """Semantic representation of one document section."""
    title: str
    order: int
    raw_text: str
    summary: str = ""            # rule-based abstractive-ish summary
    topics: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    facts: List[str] = field(default_factory=list)
    claims: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    conclusions: List[str] = field(default_factory=list)
    importance: float = 0.5       # 0.0–1.0, higher = more central to document
    word_count: int = 0


@dataclass
class DocumentRepresentation:
    """
    Full semantic representation of an uploaded document.
    This is what gets stored in DocumentMemory and used to answer questions.
    """
    filename: str
    document_type: str = "unknown"    # research_paper | report | resume | invoice | article | general
    purpose: str = ""                  # "This document describes/proposes/reports/analyses..."
    main_topic: str = ""
    sections: List[SectionRepresentation] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    key_facts: List[str] = field(default_factory=list)
    key_claims: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    conclusions: List[str] = field(default_factory=list)
    global_summary: str = ""           # The multi-sentence coherent summary
    short_summary: str = ""            # 2–3 sentence version
    key_points: List[str] = field(default_factory=list)
    document_map: str = ""             # "Section 1: Introduction → Section 2: ..."
    total_words: int = 0
    # For CSV/table documents
    dataframe: Optional[pd.DataFrame] = None
    table_summaries: List[str] = field(default_factory=list)

    def get_section(self, title_fragment: str) -> Optional[SectionRepresentation]:
        frag = title_fragment.lower()
        for s in self.sections:
            if frag in s.title.lower():
                return s
        return None

    def all_facts(self) -> List[str]:
        facts = list(self.key_facts)
        for s in self.sections:
            facts.extend(s.facts)
        return list(dict.fromkeys(facts))  # deduplicate preserving order

    def debug_repr(self) -> str:
        lines = [
            f"=== DocumentRepresentation: {self.filename} ===",
            f"Type    : {self.document_type}",
            f"Topic   : {self.main_topic}",
            f"Purpose : {self.purpose[:120]}",
            f"Words   : {self.total_words}",
            f"Sections: {len(self.sections)}",
        ]
        for s in self.sections:
            lines.append(f"  [{s.order}] {s.title} ({s.word_count}w, importance={s.importance:.2f})")
            if s.summary:
                lines.append(f"      Summary: {s.summary[:100]}...")
            if s.facts:
                lines.append(f"      Facts  : {s.facts[:2]}")
        lines.append(f"Key points: {self.key_points[:3]}")
        lines.append(f"Global summary: {self.global_summary[:200]}...")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — DOCUMENT TYPE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

_DOCTYPE_PATTERNS = {
    "resume": [
        r"\b(curriculum vitae|cv\b|resume\b|work experience|education|skills|certifications?|employment history)\b",
        r"\b(bachelor|master|phd|degree|university|college|gpa|cgpa)\b",
    ],
    "research_paper": [
        r"\b(abstract|introduction|methodology|results|conclusion|references|literature review|related work)\b",
        r"\b(proposed|experiment|dataset|accuracy|precision|recall|baseline|state.of.the.art)\b",
    ],
    "invoice": [
        r"\b(invoice|bill to|ship to|total amount|subtotal|tax|due date|payment terms|item|quantity|unit price)\b",
        r"\b(invoice #|inv-|bill #|receipt)\b",
    ],
    "report": [
        r"\b(executive summary|findings|recommendations|analysis|quarterly|annual|fiscal|performance)\b",
        r"\b(key performance|kpi|dashboard|metrics|year.to.date|ytd)\b",
    ],
    "article": [
        r"\b(published|author|journal|doi|issn|volume|issue|pp\.)\b",
        r"\b(according to|the study|researchers|the paper|this article)\b",
    ],
    "legal": [
        r"\b(whereas|hereinafter|pursuant|notwithstanding|indemnification|jurisdiction|liability|clause)\b",
        r"\b(agreement|contract|terms and conditions|party|parties|effective date|termination)\b",
    ],
    "financial": [
        r"\b(balance sheet|profit.and.loss|income statement|cash flow|assets|liabilities|equity|revenue)\b",
        r"\b(fiscal year|quarterly|earnings per share|eps|ebitda|roi)\b",
    ],
}


def detect_document_type(text: str, filename: str) -> str:
    text_lower = text.lower()[:5000]   # check first 5000 chars for efficiency
    fname_lower = filename.lower()

    scores: Dict[str, float] = {dtype: 0.0 for dtype in _DOCTYPE_PATTERNS}

    # Filename signals (strong)
    for dtype in ("resume", "invoice", "report"):
        if dtype in fname_lower or (dtype == "resume" and "cv" in fname_lower):
            scores[dtype] += 3.0

    # Pattern matching
    for dtype, patterns in _DOCTYPE_PATTERNS.items():
        for pattern in patterns:
            matches = len(re.findall(pattern, text_lower, re.I))
            scores[dtype] += matches * 1.0

    best = max(scores, key=lambda k: scores[k])
    if scores[best] < 2.0:
        return "general"
    return best


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — DOCUMENT PURPOSE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

_PURPOSE_PATTERNS = [
    (r"this (?:paper|study|report|document|article|thesis|proposal|work)\s+(?:presents?|proposes?|describes?|analyses?|analyzes?|investigates?|examines?|explores?|introduces?|focuses?\s+on)", "presents/proposes"),
    (r"(?:the purpose|the aim|the goal|the objective)\s+of\s+this\s+\w+\s+is\s+to\s+(.{20,100})", "states purpose"),
    (r"we (?:propose|present|describe|introduce|investigate|analyse|analyze|examine)\s+(.{20,120})", "author-stated"),
    (r"this (?:report|document)\s+(?:provides?|outlines?|summarizes?)\s+(.{20,100})", "document-stated"),
]

_PURPOSE_VERBS = {
    "research_paper": "investigates and proposes",
    "resume": "presents the professional background of",
    "invoice": "records a financial transaction for",
    "report": "reports on",
    "legal": "establishes a legal agreement regarding",
    "financial": "presents financial data for",
    "article": "discusses",
    "general": "covers",
}


def extract_document_purpose(text: str, doc_type: str, main_topic: str, filename: str) -> str:
    """
    Attempt to extract or construct a purpose statement for the document.
    Returns a human-readable string like "This document investigates X...".
    """
    # 1. Try to find an explicit purpose statement in the text
    text_lower = text[:3000].lower()
    for pattern, kind in _PURPOSE_PATTERNS:
        m = re.search(pattern, text_lower, re.I)
        if m:
            start = m.start()
            ctx = text[start:start + 400].strip()
            ctx = re.sub(r"\s+", " ", ctx).strip(" .,;:")
            # Truncate at first sentence boundary after 30 chars
            sent_end = re.search(r"(?<=[.!?])\s", ctx[30:])
            if sent_end:
                ctx = ctx[:30 + sent_end.start()].strip()
            if len(ctx) > 20:
                return ctx[:300]

    # 2. Construct from document type + main topic
    verb = _PURPOSE_VERBS.get(doc_type, "covers")
    if main_topic:
        return f"This document {verb} {main_topic}."
    # 3. Fall back to filename-based purpose
    stem = re.sub(r"\.[a-zA-Z]+$", "", filename)
    stem = re.sub(r"[_\-]+", " ", stem).strip()
    return f"This document {verb} {stem}."


# ══════════════════════════════════════════════════════════════════════════════
# PART 5 — MAIN TOPIC EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_main_topic(text: str, doc_type: str) -> str:
    """
    Determine the main topic of the document.
    Uses: title-case first line, abstract first sentence, or TF-IDF top terms.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # 1. For research papers: first sentence of abstract
    if doc_type == "research_paper":
        abstract_start = -1
        for i, line in enumerate(lines[:50]):
            if re.match(r"^abstract\s*$", line, re.I):
                abstract_start = i + 1
                break
        if abstract_start > 0 and abstract_start < len(lines):
            # First non-empty line after "Abstract"
            for line in lines[abstract_start:abstract_start + 5]:
                if len(line) > 30:
                    return line[:200]

    # 2. Look for a "title-like" first line (short, title-case, no sentence end)
    for line in lines[:5]:
        if 5 < len(line) < 150 and not line.endswith((".","!","?")) and not re.search(r"@|\d{4}-\d{2}-\d{2}", line):
            words = line.split()
            if len(words) >= 2:
                return line

    # 3. TF-IDF key phrases from first 1000 chars
    snippet = " ".join(lines[:20])[:1000]
    if snippet:
        # Extract noun-phrase candidates: sequences of capitalized words or
        # common noun endings
        candidates = re.findall(
            r"\b([A-Z][a-z]+ (?:[A-Z][a-z]+ )*(?:System|Model|Method|Approach|Framework|Analysis|Study|Review|Report|Network|Algorithm))\b",
            snippet
        )
        if candidates:
            return candidates[0]

        # Fall back: most frequent meaningful bigrams in first 500 chars
        words = re.findall(r"[a-zA-Z]{4,}", snippet.lower())
        STOP = {"this","that","with","from","have","been","will","they","their","which","when","where","what","also","some","these","those","more","into","over","than","about","would","could","should","other","after","before"}
        words = [w for w in words if w not in STOP]
        if words:
            freq: Dict[str, int] = {}
            for w in words:
                freq[w] = freq.get(w, 0) + 1
            top = sorted(freq, key=lambda k: -freq[k])[:3]
            return " ".join(top)

    return "the subject matter described in the document"


# ══════════════════════════════════════════════════════════════════════════════
# PART 6 — STRUCTURED SECTION PARSING (enhanced version)
# ══════════════════════════════════════════════════════════════════════════════

# Expanded section headers for different document types
_SECTION_HEADER_PATTERNS = [
    # Numbered sections: "1. Introduction", "1.1 Background"
    re.compile(r"^(\d+\.(?:\d+\.?)*)\s+(.+)$"),
    # ALL CAPS sections: "INTRODUCTION", "METHODOLOGY"
    re.compile(r"^([A-Z][A-Z\s]{4,50})$"),
    # Title-case short lines: "Introduction", "Background and Related Work"
    re.compile(r"^([A-Z][a-zA-Z\s]{3,60})$"),
]

_KNOWN_SECTION_WORDS = {
    "abstract", "introduction", "background", "related work", "literature review",
    "methodology", "methods", "approach", "proposed method", "system design",
    "architecture", "implementation", "experiment", "experiments", "results",
    "evaluation", "discussion", "conclusion", "conclusions", "future work",
    "references", "bibliography", "acknowledgements", "appendix",
    # Resume sections
    "summary", "objective", "profile", "education", "experience", "employment",
    "work experience", "skills", "technical skills", "projects", "certifications",
    "achievements", "publications", "awards",
    # Report sections
    "executive summary", "findings", "recommendations", "analysis", "overview",
    # Invoice/financial sections
    "items", "services", "payment", "billing",
}


def _is_section_header(line: str, prev_line: str = "", next_line: str = "") -> Tuple[bool, int]:
    """
    Returns (is_header, level) where level 1=top, 2=sub, 3=sub-sub.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return False, 0

    # Numbered: "1. Introduction" → level 1, "1.1 Background" → level 2
    m = re.match(r"^(\d+)(\.(\d+))?(\.(\d+))?\s+(.+)$", stripped)
    if m:
        if m.group(5):
            return True, 3
        if m.group(3):
            return True, 2
        return True, 1

    stripped_lower = stripped.lower().rstrip(":").strip()

    # Known section words (exact or starts-with)
    for kw in _KNOWN_SECTION_WORDS:
        if stripped_lower == kw or stripped_lower.startswith(kw + " ") or stripped_lower.startswith(kw + ":"):
            return True, 1

    # Short ALL-CAPS line
    if stripped.isupper() and 4 <= len(stripped) <= 60:
        return True, 1

    # Short title-case line (no sentence-ending punctuation)
    if (not stripped.endswith((".", "!", "?", ",", ";"))
            and len(stripped) < 70
            and len(stripped.split()) <= 8
            and re.match(r"^[A-Z]", stripped)
            and not re.search(r"@|\d{10,}", stripped)):
        # Validate: surrounded by blank lines (strong signal)
        if not prev_line.strip() and not next_line.strip():
            return True, 1

    return False, 0


def parse_document_structure(raw_text: str, filename: str) -> StructuredDocument:
    """
    Parse raw text into a StructuredDocument with sections, paragraphs,
    and tables identified. This replaces the simple regex-based section
    detection in document_parser.py with a proper structure parser.
    """
    doc = StructuredDocument(filename=filename, kind="text", raw_text=raw_text)

    lines = raw_text.splitlines()
    n = len(lines)

    current_section = DocumentSection(title="Preamble", level=1, order=0)
    current_para_lines: List[str] = []
    sections: List[DocumentSection] = []

    def _flush_para():
        para = " ".join(current_para_lines).strip()
        if para and len(para) > 2:
            current_section.paragraphs.append(para)
        current_para_lines.clear()

    order = 0
    i = 0
    while i < n:
        line = lines[i]
        prev = lines[i - 1] if i > 0 else ""
        nxt  = lines[i + 1] if i < n - 1 else ""

        stripped = line.strip()

        if not stripped:
            # Blank line → paragraph break
            _flush_para()
            i += 1
            continue

        is_hdr, level = _is_section_header(stripped, prev, nxt)

        if is_hdr:
            _flush_para()
            # Save current section
            if current_section.paragraphs or current_section.title != "Preamble":
                sections.append(current_section)
            order += 1
            # Clean up numbered prefix from title
            title_clean = re.sub(r"^\d+\.(?:\d+\.?)* ", "", stripped).strip()
            current_section = DocumentSection(
                title=title_clean, level=level, order=order
            )
        else:
            # Skip lines that exactly duplicate the current section title
            # (happens when a heading is immediately followed by itself as body text)
            if stripped.lower().strip(":") == current_section.title.lower().strip(":"):
                pass
            else:
                current_para_lines.append(stripped)

        i += 1

    _flush_para()
    if current_section.paragraphs or current_section.tables:
        sections.append(current_section)

    # If we found zero sections or only one giant "Preamble" section,
    # try to split on numbered headings embedded within paragraphs.
    if len(sections) <= 1:
        sections = _split_flat_document(raw_text, filename)

    if not sections:
        sections = [DocumentSection(
            title="Document Content",
            level=1,
            order=0,
            paragraphs=[p for p in raw_text.split("\n\n") if p.strip()][:50]
        )]

    doc.sections = sections
    return doc


# ══════════════════════════════════════════════════════════════════════════════
# PART 7 — SECTION-LEVEL UNDERSTANDING (deterministic, rule-based)
# ══════════════════════════════════════════════════════════════════════════════

def _first_sentence(text: str) -> str:
    """Extract the first complete sentence."""
    text = text.strip()
    m = re.search(r"([^.!?]+[.!?])", text)
    if m:
        return m.group(1).strip()
    return text[:200].strip()


def _extract_facts_from_text(text: str) -> List[str]:
    """
    Extract factual statements from a paragraph/section.
    Looks for: numbers, percentages, specific claims, quantitative statements.
    Falls back to any sentence containing a number for non-quantitative documents.
    """
    facts = []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 15:
            continue
        # Primary: percentage / metric-style fact
        if re.search(r"\b\d+\.?\d*\s*(%|percent|accuracy|precision|recall|score|rate|times|fold|km|kg|mb|gb|years?|months?)\b", sent, re.I):
            facts.append(sent)
        # Comparative / achievement
        elif re.search(r"\b(outperform|surpass|exceed|improve|reduce|increase|decrease|achieve|reach|led|built|managed|developed|designed|implemented)\w*\b", sent, re.I):
            facts.append(sent)
        # Monetary amounts (invoices, reports)
        elif re.search(r"[₹$£€]\s*\d|(\d[\d,]+)\s*(million|billion|lakh|crore|thousand)", sent, re.I):
            facts.append(sent)
    # Secondary fallback: any sentence with a plain number (for resumes, invoices)
    if len(facts) < 2:
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 15 or sent in facts:
                continue
            if re.search(r"\b\d+\b", sent):
                facts.append(sent)
            if len(facts) >= 5:
                break
    return facts[:5]


def _extract_claims_from_text(text: str) -> List[str]:
    """
    Extract author claims / propositions.
    """
    claims = []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 20:
            continue
        if re.search(r"\b(we (propose|show|demonstrate|claim|argue|find|present)|"
                     r"this (paper|study|method|approach|work) (shows?|demonstrates?|"
                     r"proposes?|presents?|proves?)|our (method|approach|system) "
                     r"(outperforms?|achieves?|provides?))\b", sent, re.I):
            claims.append(sent)
    return claims[:3]


def _extract_evidence_from_text(text: str) -> List[str]:
    """Extract evidence/support statements."""
    evidence = []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 20:
            continue
        if re.search(r"\b(table|figure|fig\.|experiment|result|dataset|"
                     r"according to|as shown|as demonstrated|based on|"
                     r"with \d+|using \d+|trained on)\b", sent, re.I):
            evidence.append(sent)
    return evidence[:3]


def _extract_conclusions_from_text(text: str) -> List[str]:
    """Extract conclusion/implication statements."""
    conclusions = []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 20:
            continue
        if re.search(
            r"\b(therefore|thus|hence|in conclusion|we conclude|we have (presented|proposed|shown|demonstrated)|"
            r"this (paper|work|study) (has|presents?|shows?|demonstrates?|proposes?)|"
            r"this suggests?|this implies?|these results? (suggest|show|indicate|demonstrate)|"
            r"future work|in summary|to summarize|overall|in this (paper|work|study))\b",
            sent, re.I
        ):
            conclusions.append(sent)
    return conclusions[:3]


def _build_section_summary(section: DocumentSection, doc_type: str) -> str:
    """
    Build a rule-based section summary. NOT extractive in the naive
    sentence-selection sense. Instead:
    1. Topic sentence = the section's first meaningful sentence.
    2. Key content = key facts/numbers identified in the section.
    3. Closing = last sentence if it's a conclusion-style statement.

    This produces a summary that covers WHAT THE SECTION IS ABOUT, not
    just which sentences had high TF-IDF centrality.
    """
    full_text = section.full_text
    if not full_text.strip():
        return ""

    # Remove the section title if it appears as the very first line
    first_line = full_text.splitlines()[0].strip()
    if first_line.lower().rstrip(":") == section.title.lower().rstrip(":"):
        full_text = "\n".join(full_text.splitlines()[1:]).strip()
    if not full_text.strip():
        return ""

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", full_text) if len(s.strip()) > 15]
    if not sentences:
        return full_text[:200]
    if len(sentences) == 1:
        return sentences[0]

    # Topic sentence: first real sentence
    topic = sentences[0]

    # Key fact: find the most informative sentence (has numbers/claims)
    key_fact = ""
    for sent in sentences[1:]:
        if re.search(r"\b\d+\.?\d*\s*(%|accuracy|precision|percent|times)\b", sent, re.I):
            key_fact = sent
            break
        elif re.search(r"\b(propose|present|introduce|demonstrate|achieve|improve)\w*\b", sent, re.I):
            if not key_fact:
                key_fact = sent

    # Closing: last sentence if conclusion-like
    closing = ""
    last = sentences[-1]
    if re.search(r"\b(therefore|thus|hence|conclude|summary|result|finding)\b", last, re.I):
        closing = last

    parts = [topic]
    if key_fact and key_fact != topic:
        parts.append(key_fact)
    if closing and closing not in parts:
        parts.append(closing)

    summary = " ".join(parts)
    # Truncate to reasonable length
    return summary[:400] if len(summary) > 400 else summary


def _compute_section_importance(section: DocumentSection, total_words: int) -> float:
    """
    Compute importance score (0.0–1.0) for a section.
    Factors: word count proportion, section title signals, content quality.
    """
    if total_words == 0:
        return 0.5

    # Base: word count proportion (longer = more important, capped)
    word_ratio = min(section.word_count / max(total_words, 1), 0.4)
    score = word_ratio / 0.4 * 0.4   # normalise to 0–0.4

    # Title signals
    title_low = section.title.lower()
    HIGH_IMPORTANCE = {"abstract", "conclusion", "results", "findings",
                       "executive summary", "summary", "key findings", "methodology"}
    LOW_IMPORTANCE  = {"references", "bibliography", "acknowledgements",
                       "appendix", "table of contents", "index"}

    if any(kw in title_low for kw in HIGH_IMPORTANCE):
        score += 0.4
    elif any(kw in title_low for kw in LOW_IMPORTANCE):
        score = max(0.0, score - 0.2)
    else:
        score += 0.2

    # Content quality signals (numbers, claims)
    text = section.full_text
    if re.search(r"\b\d+\.?\d*\s*%\b", text):
        score += 0.1
    if re.search(r"\b(propose|conclude|demonstrate|show|find)\w*\b", text, re.I):
        score += 0.1

    return min(1.0, score)


# ══════════════════════════════════════════════════════════════════════════════
# PART 8 — GLOBAL DOCUMENT SUMMARY SYNTHESIS
# ══════════════════════════════════════════════════════════════════════════════

def _synthesize_global_summary(
    doc_type: str,
    purpose: str,
    main_topic: str,
    section_reprs: List[SectionRepresentation],
    filename: str,
) -> Tuple[str, str, List[str]]:
    """
    Build the global summary, short summary, and key points from
    section-level representations.

    This is the core of what makes this DIFFERENT from TextRank:
    - We don't select sentences from the document.
    - We SYNTHESISE a summary from the section summaries.
    - The global summary describes STRUCTURE + CONTENT + CONCLUSIONS.

    Returns: (global_summary, short_summary, key_points)
    """
    if not section_reprs:
        return purpose, purpose[:200], []

    # Sort sections by importance for summary construction
    by_importance = sorted(section_reprs, key=lambda s: -s.importance)

    # --- BUILD GLOBAL SUMMARY ---
    parts = []

    SKIP_TITLES = {"references", "bibliography", "acknowledgements", "appendix",
                   "preamble", "document content", "table of contents", "index"}

    # 1. Document purpose / overview sentence (use a clean single sentence)
    if purpose and len(purpose) > 10:
        # Take just the first sentence of the purpose
        first_sent = re.split(r"(?<=[.!?])\s", purpose)[0].strip()
        if not first_sent.lower().startswith("this"):
            first_sent = "This document " + first_sent
        if not first_sent.endswith("."):
            first_sent += "."
        parts.append(first_sent)

    # 2. Structure description (what major sections exist)
    meaningful_sections = [
        s for s in section_reprs
        if s.importance >= 0.15 and s.title.lower() not in SKIP_TITLES
        and s.word_count >= 10
    ]
    if len(meaningful_sections) >= 3:
        section_titles = [s.title for s in meaningful_sections[:6]]
        if len(section_titles) <= 4:
            last = section_titles[-1]
            rest = section_titles[:-1]
            parts.append(f"It covers {', '.join(rest)}, and {last}.")
        else:
            parts.append(
                f"The document is structured into {len(meaningful_sections)} sections, "
                f"covering {', '.join(section_titles[:3])}, and more."
            )

    # 3. Content summaries from high-importance sections (document order)
    top_sections = sorted(
        [s for s in section_reprs if s.importance >= 0.30 and s.title.lower() not in SKIP_TITLES],
        key=lambda s: s.order
    )
    for sec in top_sections[:3]:
        if not sec.summary or len(sec.summary) < 20:
            continue
        # Take only the first sentence of the section summary to avoid length explosion
        first_sec_sent = re.split(r"(?<=[.!?])\s", sec.summary)[0].strip()
        if not first_sec_sent.endswith("."):
            first_sec_sent += "."
        title_lc = sec.title.lower()
        if title_lc in ("abstract", "summary", "executive summary"):
            connector = f"The {title_lc} states:"
        else:
            connector = f"In {sec.title},"
        # Don't add if it duplicates purpose
        if first_sec_sent.lower()[:60] not in parts[0].lower()[:120] if parts else True:
            parts.append(f"{connector} {first_sec_sent[0].lower()}{first_sec_sent[1:]}")

    # 4. One conclusion sentence
    all_conclusions = []
    for s in section_reprs:
        all_conclusions.extend(s.conclusions)
    # Filter: skip if conclusion duplicates an existing part
    existing_text = " ".join(parts).lower()
    for conc in all_conclusions:
        conc_short = conc[:60].lower()
        if conc_short not in existing_text and len(conc) > 20:
            first_conc = re.split(r"(?<=[.!?])\s", conc)[0].strip()
            if not first_conc.endswith("."):
                first_conc += "."
            parts.append(first_conc)
            break

    global_summary = " ".join(parts)
    global_summary = re.sub(r"\s+", " ", global_summary).strip()

    # --- SHORT SUMMARY (2-3 sentences) ---
    short_parts = []
    if purpose:
        first_p = re.split(r"(?<=[.!?])\s", purpose)[0].strip()
        if not first_p.endswith("."):
            first_p += "."
        short_parts.append(first_p)
    # Add one sentence from the most important non-purpose section
    purpose_lower = short_parts[0].lower()[:80] if short_parts else ""
    for sec in by_importance[:3]:
        if sec.title.lower() in SKIP_TITLES:
            continue
        if not sec.summary or len(sec.summary) < 15:
            continue
        first_sec = re.split(r"(?<=[.!?])\s", sec.summary)[0].strip()
        if not first_sec.endswith("."):
            first_sec += "."
        # Avoid duplication with purpose sentence
        if first_sec.lower()[:60] not in purpose_lower:
            short_parts.append(first_sec)
            if len(short_parts) >= 3:
                break
    short_summary = " ".join(short_parts[:3])

    # --- KEY POINTS ---
    key_points = []
    # From section facts and claims — skip anything that looks like a section title
    for sec in sorted(section_reprs, key=lambda s: -s.importance)[:5]:
        for fact in sec.facts[:2]:
            if len(fact) > 20 and "\n" not in fact and fact not in key_points:
                key_points.append(fact)
        for claim in sec.claims[:1]:
            if len(claim) > 20 and "\n" not in claim and claim not in key_points:
                key_points.append(claim)
    key_points = _deduplicate_list(key_points)[:8]

    return global_summary, short_summary, key_points


def _deduplicate_list(items: List[str], threshold: float = 0.7) -> List[str]:
    """Remove near-duplicate strings from a list."""
    result = []
    norm = lambda s: re.sub(r"\s+", " ", s.lower())
    for item in items:
        ni = norm(item)
        is_dup = False
        for existing in result:
            ne = norm(existing)
            shorter = ni if len(ni) <= len(ne) else ne
            longer = ne if len(ni) <= len(ne) else ni
            if shorter and shorter in longer:
                is_dup = True
                break
            words_i = set(ni.split())
            words_e = set(ne.split())
            if words_i and words_e:
                overlap = len(words_i & words_e) / len(words_i | words_e)
                if overlap >= threshold:
                    is_dup = True
                    break
        if not is_dup:
            result.append(item)
    return result


def _split_flat_document(raw_text: str, filename: str) -> List[DocumentSection]:
    """
    Secondary section splitter for documents whose section headings are not
    on their own isolated lines (e.g. "1. Introduction\nText..." with no
    surrounding blank lines, or resume-style keyword headers embedded in flow).
    Tries three strategies in order:
      A) Numbered headings at line start: "1. Title" / "1.1 Title"
      B) Known keyword headers at line start (even without surrounding blanks)
      C) Paragraph-based split (every double newline = new section)
    """
    lines = raw_text.splitlines()

    # Strategy A: numbered headings
    sections: List[DocumentSection] = []
    current = DocumentSection(title="Preamble", level=1, order=0)
    order = 0
    current_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        # Numbered: "1. Introduction" or "2.3 Background"
        m = re.match(r"^(\d+)(\.(\d+))?\s+([A-Z].{2,60})$", stripped)
        if m:
            if current_lines:
                current.paragraphs.append(" ".join(current_lines))
                current_lines = []
            if current.paragraphs or order > 0:
                sections.append(current)
            order += 1
            title = m.group(4).strip()
            level = 2 if m.group(3) else 1
            current = DocumentSection(title=title, level=level, order=order)
        else:
            if stripped:
                current_lines.append(stripped)

    if current_lines:
        current.paragraphs.append(" ".join(current_lines))
    if current.paragraphs or current.title != "Preamble":
        sections.append(current)

    if len(sections) >= 3:
        return sections

    # Strategy B: keyword headers at line start
    sections = []
    current = DocumentSection(title="Preamble", level=1, order=0)
    order = 0
    current_lines = []

    for line in lines:
        stripped = line.strip()
        stripped_lower = stripped.lower().rstrip(":")
        is_kw = any(stripped_lower == kw or stripped_lower.startswith(kw + " ")
                    for kw in _KNOWN_SECTION_WORDS)
        if is_kw and len(stripped) <= 60:
            if current_lines:
                current.paragraphs.append(" ".join(current_lines))
                current_lines = []
            if current.paragraphs or order > 0:
                sections.append(current)
            order += 1
            current = DocumentSection(title=stripped.rstrip(":"), level=1, order=order)
        else:
            if stripped:
                current_lines.append(stripped)

    if current_lines:
        current.paragraphs.append(" ".join(current_lines))
    if current.paragraphs or current.title != "Preamble":
        sections.append(current)

    if len(sections) >= 3:
        return sections

    # Strategy C: paragraph split (double-newline)
    paras = [p.strip() for p in re.split(r"\n\s*\n", raw_text) if p.strip()]
    if len(paras) >= 2:
        sections = []
        for i, para in enumerate(paras[:20]):
            first_line = para.splitlines()[0].strip() if para.splitlines() else para[:50]
            title = first_line[:60] if len(first_line) <= 60 else f"Section {i+1}"
            rest_lines = para.splitlines()[1:] if len(para.splitlines()) > 1 else []
            body = " ".join(l.strip() for l in rest_lines if l.strip()) or para
            sections.append(DocumentSection(
                title=title, level=1, order=i,
                paragraphs=[body] if body else [],
            ))
        return sections

    return []


# ══════════════════════════════════════════════════════════════════════════════
# PART 9 — MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def build_document_representation(
    structured_doc: StructuredDocument,
    debug: bool = False,
) -> DocumentRepresentation:
    """
    Transform a StructuredDocument (structure only) into a
    DocumentRepresentation (semantic meaning).

    This is the main entry point for the document understanding layer.
    Called once per uploaded document; result is stored in DocumentMemory.

    Pipeline:
      1. Detect document type
      2. Extract main topic
      3. Extract purpose
      4. Build section-level representations
      5. Synthesise global summary from section summaries
      6. Collect entities, facts, claims, evidence, conclusions
    """
    raw_text = structured_doc.raw_text
    filename = structured_doc.filename

    if debug:
        print(f"\n[DocUnderstanding] Processing: {filename} ({len(raw_text)} chars, "
              f"{len(structured_doc.sections)} sections)")

    # --- 1. Document type ---
    doc_type = detect_document_type(raw_text, filename)
    if debug:
        print(f"[DocUnderstanding] Type: {doc_type}")

    # --- 2. Main topic ---
    main_topic = extract_main_topic(raw_text, doc_type)
    if debug:
        print(f"[DocUnderstanding] Topic: {main_topic}")

    # --- 3. Purpose ---
    purpose = extract_document_purpose(raw_text, doc_type, main_topic, filename)
    if debug:
        print(f"[DocUnderstanding] Purpose: {purpose[:100]}")

    # --- 4. Section representations ---
    total_words = structured_doc.total_words
    section_reprs: List[SectionRepresentation] = []

    for sec in structured_doc.sections:
        full_text = sec.full_text
        if len(full_text.strip()) < 10:
            continue

        sr = SectionRepresentation(
            title=sec.title,
            order=sec.order,
            raw_text=full_text,
            word_count=sec.word_count,
        )

        # Section-level analysis
        sr.summary   = _build_section_summary(sec, doc_type)
        sr.facts     = _extract_facts_from_text(full_text)
        sr.claims    = _extract_claims_from_text(full_text)
        sr.evidence  = _extract_evidence_from_text(full_text)
        sr.conclusions = _extract_conclusions_from_text(full_text)
        sr.importance  = _compute_section_importance(sec, total_words)

        # For sections with a "conclusion"-like title but no explicit signal words,
        # use the last meaningful sentence as the conclusion
        if not sr.conclusions and any(
            kw in sec.title.lower()
            for kw in ("conclusion", "finding", "result", "outlook", "recommendation",
                       "summary", "objective", "executive")
        ):
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", full_text) if len(s.strip()) > 20]
            if sentences:
                sr.conclusions = [sentences[-1]]

        # Named entity extraction (simple: capitalised noun phrases)
        sr.entities = _extract_entities(full_text)

        section_reprs.append(sr)

        if debug:
            print(f"  Section [{sec.order}] '{sec.title}': {sec.word_count}w, "
                  f"importance={sr.importance:.2f}, summary={sr.summary[:60]}...")

    # --- 5. Global synthesis ---
    global_summary, short_summary, key_points = _synthesize_global_summary(
        doc_type, purpose, main_topic, section_reprs, filename
    )

    # --- 6. Document-level aggregation ---
    all_entities: List[str] = []
    all_facts: List[str] = []
    all_claims: List[str] = []
    all_evidence: List[str] = []
    all_conclusions: List[str] = []

    for sr in section_reprs:
        all_entities.extend(sr.entities)
        all_facts.extend(sr.facts)
        all_claims.extend(sr.claims)
        all_evidence.extend(sr.evidence)
        all_conclusions.extend(sr.conclusions)

    def _clean_list(items: List[str]) -> List[str]:
        """
        Sanitise extracted text items:
        - Replace newlines with spaces (section titles embedded in text become inline)
        - Remove items that are very short after normalisation
        - Remove items that consist only of a section title (capitalised word, no verb)
        """
        result = []
        for s in items:
            if not s:
                continue
            # Normalise newlines → spaces
            s = re.sub(r"\s*\n\s*", " ", s).strip()
            if len(s) < 16:
                continue
            # Reject if it looks like a bare section title: e.g. "Abstract This paper"
            # i.e. first word is a known section keyword AND no verb follows within 5 words
            first_word = s.split()[0].lower().rstrip(":")
            if first_word in {"abstract", "introduction", "methodology", "results",
                              "conclusion", "discussion", "background", "summary",
                              "education", "experience", "skills", "projects"}:
                # Only reject if the NEXT part is also sentence-opening (capitalised)
                # meaning this is "SectionTitle\nFirstSentence" joined
                parts_check = s.split(None, 2)
                if len(parts_check) >= 2 and re.match(r"^[A-Z]", parts_check[1]):
                    # Strip the leading section-title word
                    s = " ".join(s.split()[1:]).strip()
                    if len(s) < 16:
                        continue
            result.append(s)
        return result

    all_entities    = _deduplicate_list(_clean_list(all_entities))[:20]
    all_facts       = _deduplicate_list(_clean_list(all_facts))[:15]
    all_claims      = _deduplicate_list(_clean_list(all_claims))[:10]
    all_evidence    = _deduplicate_list(_clean_list(all_evidence))[:10]
    all_conclusions = _deduplicate_list(_clean_list(all_conclusions))[:8]

    # --- 7. Document map ---
    document_map = _build_document_map(section_reprs)

    repr_ = DocumentRepresentation(
        filename=filename,
        document_type=doc_type,
        purpose=purpose,
        main_topic=main_topic,
        sections=section_reprs,
        entities=all_entities,
        key_facts=all_facts,
        key_claims=all_claims,
        evidence=all_evidence,
        conclusions=all_conclusions,
        global_summary=global_summary,
        short_summary=short_summary,
        key_points=key_points,
        document_map=document_map,
        total_words=total_words,
        dataframe=structured_doc.dataframe,
    )

    if debug:
        print("\n" + repr_.debug_repr())

    return repr_


def _extract_entities(text: str) -> List[str]:
    """
    Simple named entity extraction: capitalized noun phrases.
    Not a full NER — just finds likely proper nouns and technical terms.
    """
    phrases = re.findall(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})\b", text)
    stop_words = {
        "The", "This", "That", "These", "Those", "Here", "There",
        "In", "On", "At", "For", "By", "With", "From", "Of", "To",
        "A", "An", "And", "Or", "But", "Section", "Table", "Figure",
        "We", "Our", "They", "Their", "It", "Its", "All", "Each",
        "Some", "Many", "Most", "Both", "Such", "Which", "When",
        "Where", "How", "Why", "What", "Who", "However", "Therefore",
        "Thus", "Hence", "Finally", "First", "Second", "Third",
        "Also", "More", "Other", "New", "High", "Low", "Large", "Small",
        "Abstract", "Introduction", "Methodology", "Results", "Conclusion",
        "Discussion", "Background", "Related", "Work", "References",
        "Education", "Experience", "Skills", "Projects", "Summary",
        "Findings", "Overview", "Approach", "Method", "Model",
        "Training", "Existing", "Prior", "Future",
    }
    # Only keep proper-noun-style phrases: at least one word ≥4 chars,
    # not a known section heading, not a single common word
    result = []
    for p in phrases:
        if p in stop_words:
            continue
        words = p.split()
        # Skip single common words
        if len(words) == 1 and p.lower() in {
            "we", "our", "they", "he", "she", "it", "its", "this", "that",
            "these", "those", "no", "yes", "january", "february", "march",
            "april", "may", "june", "july", "august", "september", "october",
            "november", "december", "monday", "tuesday", "wednesday", "thursday",
            "friday", "saturday", "sunday",
        }:
            continue
        # Must contain at least one word that's ≥4 chars and not a stop word
        if any(len(w) >= 4 and w not in stop_words for w in words):
            result.append(p)

    freq: Dict[str, int] = {}
    for e in result:
        freq[e] = freq.get(e, 0) + 1
    sorted_entities = sorted(freq, key=lambda k: -freq[k])
    return sorted_entities[:15]


def _build_document_map(sections: List[SectionRepresentation]) -> str:
    """Build a concise document structure map."""
    if not sections:
        return ""
    parts = []
    for s in sections:
        if s.importance >= 0.15 and s.title.lower() not in ("preamble", "document content"):
            indent = "  " * (0)
            parts.append(f"{indent}• {s.title}")
    return "\n".join(parts[:15])


# ══════════════════════════════════════════════════════════════════════════════
# PART 10 — TABLE DOCUMENT UNDERSTANDING
# ══════════════════════════════════════════════════════════════════════════════

def build_table_representation(df: pd.DataFrame, filename: str) -> DocumentRepresentation:
    """
    Build a DocumentRepresentation for an uploaded CSV/Excel file.
    Uses pandas for all numerical reasoning.
    """
    cols = list(df.columns)
    n_rows = len(df)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    text_cols = [c for c in cols if c not in numeric_cols]

    # Compute statistics for all numeric columns
    stats_parts = []
    for col in numeric_cols[:8]:
        s = df[col].dropna()
        if len(s) > 0:
            stats_parts.append(
                f"{col}: mean={round(float(s.mean()), 2)}, "
                f"min={round(float(s.min()), 2)}, max={round(float(s.max()), 2)}, "
                f"count={len(s)}"
            )

    # Top values for categorical columns
    cat_parts = []
    for col in text_cols[:5]:
        try:
            vc = df[col].astype(str).value_counts().head(3)
            cat_parts.append(f"{col}: " + ", ".join(f"{k}({v})" for k, v in vc.items()))
        except Exception:
            pass

    purpose = (f"This is a spreadsheet file containing {n_rows} rows and "
               f"{len(cols)} columns ({', '.join(cols[:5])}"
               f"{', ...' if len(cols) > 5 else ''}).")

    summary_parts = [purpose]
    if stats_parts:
        summary_parts.append("Numerical columns: " + "; ".join(stats_parts))
    if cat_parts:
        summary_parts.append("Categorical columns: " + "; ".join(cat_parts))

    global_summary = " ".join(summary_parts)

    key_facts = stats_parts + cat_parts

    return DocumentRepresentation(
        filename=filename,
        document_type="spreadsheet",
        purpose=purpose,
        main_topic=f"data table with {n_rows} rows and {len(cols)} columns",
        global_summary=global_summary,
        short_summary=purpose,
        key_facts=key_facts[:10],
        key_points=key_facts[:5],
        total_words=n_rows * len(cols),
        dataframe=df,
    )
