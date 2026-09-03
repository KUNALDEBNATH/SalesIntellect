"""
document_parser.py
────────────────────────────────────────────────────────────────────────────
Extracts text / tabular data from uploaded documents and answers questions
about them using retrieval-augmented generation.

Supported formats: PDF, DOCX, TXT, CSV, XLSX.

Design notes
------------
* Large documents are NEVER passed to the LLM whole. Text is chunked and
  only the top-scoring chunks (via rag_utils.SimpleTfidfRetriever) are
  placed into the prompt.
* CSV / Excel files are summarised (shape, columns, quick stats) AND
  converted row-by-row into rich text so the same TF-IDF retrieval
  machinery can answer row-level questions ("who gave bad feedback?").
* Simple arithmetic questions (average / sum / count of a column) are
  additionally computed directly with pandas for accuracy, and the
  computed fact is injected into the context passed to the LLM.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag_utils import chunk_text, SimpleTfidfRetriever

MAX_TEXT_CHARS_FOR_SUMMARY = 4000  # cap used only for the "quick preview"


@dataclass
class ParsedDocument:
    """Normalised result of parsing any supported document type."""
    kind: str                       # "text" or "table"
    filename: str
    raw_text: str = ""              # full extracted text (text-type files)
    dataframe: Optional[pd.DataFrame] = None   # table-type files
    summary: str = ""                # human-readable summary
    retriever: Optional[SimpleTfidfRetriever] = field(default=None, repr=False)
    row_texts: List[str] = field(default_factory=list)

    def build_retriever(self):
        if self.kind == "text":
            chunks = chunk_text(self.raw_text)
            self.retriever = SimpleTfidfRetriever(chunks)
        elif self.kind == "table":
            self.retriever = SimpleTfidfRetriever(self.row_texts)


# ─────────────────────────────────────────────────────────── EXTRACTORS

def _extract_pdf(path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = ""
        # "layout" mode reconstructs text using each glyph's physical
        # position on the page, so multi-column resumes/brochures come
        # out with real spaces/line breaks between columns instead of
        # every text run glued together ("KunalDebnath...phone+91...").
        # Plain extract_text() has no column awareness and is what was
        # producing the unreadable, run-together document answers.
        try:
            text = page.extract_text(extraction_mode="layout") or ""
        except Exception:
            text = ""
        if not text.strip():
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
        pages.append(text)
    raw = "\n".join(pages).strip()
    return _strip_icon_glyphs(raw)


# Resume/brochure templates frequently embed icon webfonts (phone/email/
# link glyphs) that PDF text extraction turns into stray symbol
# characters (e.g. "☎", "✆", "†") sitting in the middle of words with no
# surrounding space. Stripping these ranges (Misc Symbols/Dingbats and
# the Private Use Area, where icon fonts commonly map their glyphs)
# removes the noise without touching normal text or punctuation.
_ICON_GLYPH_RE = re.compile(r"[\u2600-\u27BF\uE000-\uF8FF]")


def _strip_icon_glyphs(text: str) -> str:
    return _ICON_GLYPH_RE.sub(" ", text)


# ── Generic "describe/summarize the whole document" detection ────────────────
# These queries want a coherent overview, not whatever happens to score
# highest in TF-IDF retrieval. Retrieval chunks overlap by design (150
# chars of shared text between adjacent chunks), so stitching several of
# them together — as the old fallback did — visibly repeats the same
# sentence. For these broad requests we instead hand back a single clean
# slice of the document's own text (see attachment_handler._build_doc_answer).
_OVERVIEW_RE = re.compile(
    r"\b(describe|summar(?:y|ize|ise|ization)|overview|"
    r"tell me about|about this (?:document|file|pdf)|"
    r"(?:important|imp|key|main|core)\s+points?|highlights?|"
    r"in short|briefly|tl;?dr|gist|in brief|quick summary)\b",
    re.I,
)


def is_generic_overview_request(query: str) -> bool:
    return bool(_OVERVIEW_RE.search(query or ""))


def _extract_docx(path: str) -> str:
    import docx
    document = docx.Document(path)
    parts = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts).strip()


def _extract_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


def _row_to_text(row: dict, row_num: int) -> str:
    parts = [f"Row {row_num}:"]
    for col, val in row.items():
        sval = str(val).strip()
        if sval and sval.lower() not in ("nan", "none", ""):
            parts.append(f"{col}: {sval}")
    return "  ".join(parts)


def _dataframe_summary(df: pd.DataFrame, filename: str) -> str:
    lines = [
        f"File: {filename}",
        f"Rows: {len(df)}, Columns: {len(df.columns)}",
        f"Column names: {', '.join(str(c) for c in df.columns)}",
    ]
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        lines.append("Numeric column statistics:")
        stats = df[numeric_cols].describe().round(2)
        for col in numeric_cols:
            lines.append(
                f"  {col}: mean={stats.loc['mean', col]}, "
                f"min={stats.loc['min', col]}, max={stats.loc['max', col]}"
            )
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    for col in cat_cols[:6]:
        try:
            top_vals = df[col].astype(str).value_counts().head(3)
            preview = ", ".join(f"{k} ({v})" for k, v in top_vals.items())
            lines.append(f"  {col} top values: {preview}")
        except Exception:
            continue
    return "\n".join(lines)


def _load_table(path: str, ext: str) -> pd.DataFrame:
    if ext == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


# ─────────────────────────────────────────────────────────── PUBLIC API

TEXT_EXTS = {".pdf", ".docx", ".txt"}
TABLE_EXTS = {".csv", ".xlsx", ".xls"}


def parse_document(path: str, filename: str) -> ParsedDocument:
    """
    Parse an uploaded document from disk and return a ParsedDocument ready
    for retrieval. Raises ValueError for unsupported / corrupt files.
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        text = _extract_pdf(path)
        doc = ParsedDocument(kind="text", filename=filename, raw_text=text)
        doc.summary = (
            f"PDF document '{filename}' — extracted {len(text)} characters "
            f"of text."
        )

    elif ext == ".docx":
        text = _extract_docx(path)
        doc = ParsedDocument(kind="text", filename=filename, raw_text=text)
        doc.summary = (
            f"Word document '{filename}' — extracted {len(text)} characters "
            f"of text."
        )

    elif ext == ".txt":
        text = _extract_txt(path)
        doc = ParsedDocument(kind="text", filename=filename, raw_text=text)
        doc.summary = f"Text file '{filename}' — {len(text)} characters."

    elif ext in TABLE_EXTS:
        df = _load_table(path, ext)
        df.columns = [str(c).strip() for c in df.columns]
        row_texts = [
            _row_to_text(row.to_dict(), i + 1) for i, row in df.iterrows()
        ]
        summary = _dataframe_summary(df, filename)
        doc = ParsedDocument(
            kind="table", filename=filename, dataframe=df,
            summary=summary, row_texts=row_texts,
        )

    else:
        raise ValueError(f"Unsupported document type: {ext}")

    if not doc.raw_text and doc.kind == "text":
        raise ValueError(
            f"Could not extract any readable text from '{filename}'. "
            "The file may be empty, scanned, or corrupted."
        )

    doc.build_retriever()
    return doc


# ─────────────────────────────────────────────────────────── TABULAR Q&A

_AVG_PATTERN = re.compile(
    r"\b(average|mean|avg)\b.{0,40}?\b([a-zA-Z][a-zA-Z0-9 _]{1,30})\b", re.I
)
_COUNT_PATTERN = re.compile(r"\b(how many|count|number of|total)\b", re.I)


def compute_quick_stat(doc: ParsedDocument, query: str) -> Optional[str]:
    """
    Try to directly compute a simple statistic (average / count / filter)
    from the uploaded table using pandas, so numeric answers are accurate
    rather than left entirely to the LLM's judgement.

    Returns a fact string to inject into the LLM context, or None if no
    direct computation could be confidently made (falls back to pure RAG).
    """
    if doc.kind != "table" or doc.dataframe is None:
        return None
    df = doc.dataframe
    q_low = query.lower()

    # ── Average / mean of a numeric column ───────────────────────────────
    m = _AVG_PATTERN.search(q_low)
    if m:
        target = m.group(2).strip()
        for col in df.columns:
            if target in col.lower() or col.lower() in target:
                if pd.api.types.is_numeric_dtype(df[col]):
                    val = df[col].mean()
                    return f"Computed statistic: the average {col} is {round(val, 2)}."

    # ── Count / how many, optionally with a filter keyword ────────────────
    if _COUNT_PATTERN.search(q_low):
        filter_words = [w for w in re.findall(r"[a-zA-Z]+", q_low)
                         if len(w) > 3]
        best_col, best_mask, best_count = None, None, -1
        for col in df.columns:
            col_str = df[col].astype(str).str.lower()
            for w in filter_words:
                mask = col_str.str.contains(re.escape(w), na=False)
                cnt = int(mask.sum())
                if 0 < cnt < len(df) and cnt > best_count:
                    best_col, best_mask, best_count = col, mask, cnt
        if best_mask is not None:
            return (f"Computed statistic: {best_count} row(s) match, "
                    f"based on column '{best_col}'.")
        return f"Computed statistic: the file has {len(df)} total rows."

    return None


def retrieve_relevant_rows_or_chunks(doc: ParsedDocument, query: str,
                                       top_k: int = 5) -> List[str]:
    """Retrieve the most relevant chunks (text docs) or rows (tables)."""
    if doc.retriever is None:
        return []
    results = doc.retriever.retrieve(query, top_k=top_k)
    return dedupe_near_duplicates([text for text, _score in results])


# ═══════════════════════════════════════════════════════════════════════════
# DOCUMENT INTELLIGENCE ENGINE  (sentence-level extractive QA + summarization)
# ─────────────────────────────────────────────────────────────────────────
# No pretrained model and no API call is involved anywhere below — every
# score is a TF-IDF cosine similarity or a graph centrality computed on the
# document's own text, the same "from scratch" philosophy scratch_llm.py
# uses for the sales CSVs. This is what lets the bot answer document
# questions the hand-written regex facts (name/email/phone/…) don't cover,
# e.g. "what's his CGPA", "which companies has he worked at", "what's the
# vision statement" — WITHOUT depending on the tiny neural net (which was
# never trained on document text and reliably fails at that job) and
# without dumping the whole chunk as an unreadable wall of text.
# ═══════════════════════════════════════════════════════════════════════════

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9•\-])")
_BULLET_RE = re.compile(r"^[•\-\*\u2022\u25CF]\s*")


def split_sentences(text: str) -> List[str]:
    """
    Dependency-free sentence splitter tuned for résumé/report style text:
    bullet lines are already atomic "sentences"; normal prose is split on
    terminal punctuation followed by a capital/digit/bullet.
    """
    out: List[str] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if _BULLET_RE.match(line) or len(line) < 120:
            out.append(_BULLET_RE.sub("", line).strip())
            continue
        for s in _SENT_SPLIT_RE.split(line):
            s = s.strip()
            if s:
                out.append(s)
    # de-duplicate while preserving order (headers/footers often repeat)
    seen, uniq = set(), []
    for s in out:
        key = re.sub(r"\s+", " ", s.lower())
        if key not in seen and len(s) > 1:
            seen.add(key)
            uniq.append(s)
    return uniq


def dedupe_near_duplicates(texts: List[str], threshold: float = 0.85) -> List[str]:
    """
    Remove near-duplicate strings. Retrieval chunks overlap by design
    (rag_utils.chunk_text shares ~150 chars between neighbours), so naively
    joining top-k chunks visibly repeats the same sentence(s) — this is
    what was happening in the raw document dump. Two chunks are treated as
    duplicates when one is (almost) fully contained in the other, judged by
    normalised-character overlap rather than exact string equality so
    partial/rephrased overlaps are still caught.
    """
    kept: List[str] = []
    norm = lambda t: re.sub(r"\s+", " ", (t or "").strip().lower())
    for t in texts:
        nt = norm(t)
        if not nt:
            continue
        is_dupe = False
        for k in kept:
            nk = norm(k)
            shorter, longer = (nt, nk) if len(nt) <= len(nk) else (nk, nt)
            if not shorter:
                continue
            if shorter in longer:
                is_dupe = True
                break
            # fuzzy overlap for near-but-not-exact repeats
            overlap = len(set(shorter.split()) & set(longer.split()))
            if overlap / max(1, len(set(shorter.split()))) >= threshold:
                is_dupe = True
                break
        if not is_dupe:
            kept.append(t)
    return kept


def _tfidf_matrix(texts: List[str]):
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
    try:
        return vec, vec.fit_transform(texts)
    except ValueError:
        return None, None


def extractive_qa(doc: "ParsedDocument", query: str, top_k: int = 3,
                   min_score: float = 0.08) -> List[str]:
    """
    Sentence-level extractive question answering: ranks every sentence in
    the document against the query by TF-IDF cosine similarity and returns
    the best few, in original document order. This is finer-grained than
    chunk retrieval (which returns whole ~900-char blocks) so the answer to
    a specific question is a couple of precise sentences instead of a
    paragraph the user has to read through.
    """
    if doc.kind != "text" or not doc.raw_text:
        return []
    sentences = split_sentences(doc.raw_text)
    if len(sentences) < 2:
        return []
    vec, matrix = _tfidf_matrix(sentences + [query])
    if matrix is None:
        return []
    sims = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    ranked_idx = [i for i in np.argsort(-sims) if sims[i] >= min_score][:top_k]
    ranked_idx.sort()  # restore document order for a coherent answer
    return [sentences[i] for i in ranked_idx]


def summarize_text(text: str, max_sentences: int = 6) -> str:
    """
    TextRank-style extractive summary: builds a sentence-similarity graph
    from TF-IDF cosine similarity, scores each sentence by centrality via
    power-iteration (the same idea PageRank uses), and returns the
    highest-scoring sentences back in their original order.

    Pure TF-IDF + linear algebra — no pretrained embeddings, no API call —
    so a "describe this document" request gets a genuine, coherent,
    non-repeating summary instead of a raw truncated excerpt.
    """
    sentences = split_sentences(text)
    if not sentences:
        return ""
    if len(sentences) <= max_sentences:
        return "\n".join(sentences)

    vec, matrix = _tfidf_matrix(sentences)
    if matrix is None:
        return "\n".join(sentences[:max_sentences])

    sim = cosine_similarity(matrix)
    np.fill_diagonal(sim, 0.0)
    row_sums = sim.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transition = sim / row_sums

    scores = np.ones(len(sentences)) / len(sentences)
    damping = 0.85
    for _ in range(40):
        new_scores = (1 - damping) / len(sentences) + damping * transition.T.dot(scores)
        if np.abs(new_scores - scores).sum() < 1e-5:
            scores = new_scores
            break
        scores = new_scores

    top_idx = sorted(np.argsort(-scores)[:max_sentences])
    return "\n".join(sentences[i] for i in top_idx)


# ═══════════════════════════════════════════════════════════════════════════
# STRUCTURED DOCUMENT FACTS ENGINE
# ─────────────────────────────────────────────────────────────────────────
# The scratch LLM (see scratch_llm.py) is an ~8M-parameter transformer
# trained exclusively on the three sales CSVs' Q&A pairs. It has never
# seen a resume, invoice, or any other uploaded-document text, so asking
# it to "describe" or answer facts about one is not a language problem it
# can solve — it will reliably produce nothing usable, exactly like the
# CSV pipeline's neural model can't be trusted for list/aggregation
# queries either. scratch_llm.py's own answer for that is
# `SmartFallbackEngine`: pure pattern-matching + retrieval that
# *guarantees* a correct answer, with the neural net only allowed to
# polish phrasing on top. This section is that same idea applied to
# uploaded documents — deterministic extraction so specific questions
# ("what is his name", "what's the email") and overview requests get a
# real, correct answer regardless of whether the tiny LLM can help.
# ═══════════════════════════════════════════════════════════════════════════

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?"
    r"(?:\d{10}\b|\d{5}[-.\s]\d{5}\b|\d{3}[-.\s]\d{3}[-.\s]\d{4}\b)"
)
_LINK_RE  = re.compile(
    r"(?:https?://\S+|(?:www\.)?(?:linkedin\.com|github\.com)\S*)", re.I
)

# Common icon-webfont glyphs get mis-extracted as their literal name
# glued to the following word ("graduati—n-capEducation",
# "laptopCore Technical Skills"). Stripping a known set of these name
# fragments when they sit directly in front of a capital letter cleans
# this up without touching genuine words.
_ICON_NAME_RE = re.compile(
    r"\b(?:graduat\w*[-\u2010-\u2015]?n?[-\u2010-\u2015]?cap|briefcase|laptop|"
    r"lightbulb|envelope|user)(?=[A-Z])", re.I
)

_SECTION_KEYWORDS = (
    "summary", "objective", "profile", "education", "experience",
    "employment", "work experience", "skills", "technical skills",
    "projects", "selected projects", "certifications", "achievements",
    "publications", "awards", "core competencies",
)
_SECTION_HEADER_RE = re.compile(
    "(" + "|".join(re.escape(k) for k in _SECTION_KEYWORDS) + ")", re.I
)

_NAME_STOPWORDS = {
    "summary", "education", "experience", "skills", "projects", "resume",
    "curriculum", "vitae", "objective", "profile", "contact", "details",
}


def clean_extracted_text(text: str) -> str:
    """Remove icon-font glyph noise so extracted text reads as prose."""
    return _ICON_NAME_RE.sub("", text or "")


def extract_key_facts(raw_text: str) -> dict:
    """Deterministic, regex-based extraction of contact facts + name."""
    text = clean_extracted_text(raw_text)
    email = _EMAIL_RE.search(text)
    phone = _PHONE_RE.search(text)
    links = _LINK_RE.findall(text)

    # 1. Look inside the document's own "header" block first — the text
    #    before any recognised section keyword (Summary/Education/…) is
    #    where a name genuinely belongs. This avoids a false positive from
    #    label:value lines deeper in the document (e.g. "Deployment:
    #    Render, Vercel" under a Tools section) that a blind line-window
    #    scan could otherwise mistake for a name.
    name = _find_name_in_header_section(text)
    # 2. Fall back to a tight scan of just the first few lines.
    if not name:
        name = _find_name_line(text, window=8)
    if not name:
        # Fallback 1: the document filename itself is often "Firstname
        # Lastname.pdf" / "Firstname_Lastname_Resume.pdf" — a genuine
        # signal, not a guess, when the body text yields nothing.
        name = None  # filled in by extract_key_facts_with_filename() below
    if not name:
        # Fallback 2: candidate lines anywhere in the first third of the
        # document that are title-case and hug an email/phone line (a
        # name commonly sits directly above or below its own contact
        # block even when it fails the strict "first 30 lines" scan,
        # e.g. right-aligned headers PDF layout mode reorders).
        name = _find_name_near_contact(text, email, phone)

    return {
        "name": name,
        "email": email.group(0) if email else None,
        "phone": phone.group(0) if phone else None,
        "links": links[:5],
    }


def _looks_like_name_line(ln: str) -> Optional[str]:
    """Return the cleaned name if `ln` plausibly IS a person's name, else None."""
    if _EMAIL_RE.search(ln) or _PHONE_RE.search(ln) or _LINK_RE.search(ln):
        return None
    # A real person's name is never written as "Label: value, value" or a
    # comma-separated list — that pattern belongs to key:value fields
    # (e.g. "Deployment: Render, Vercel", "Skills: Python, SQL") which a
    # bare "2-4 title-case words" check would otherwise wrongly accept.
    if ":" in ln or "," in ln or ";" in ln or "|" in ln:
        return None
    alpha_len = len(re.sub(r"[^A-Za-z]", "", ln))
    if not alpha_len:
        return None
    header_match = _SECTION_HEADER_RE.search(ln)
    if header_match and len(header_match.group(1)) >= 0.4 * alpha_len:
        return None
    words = ln.split()
    # Real words only — a stray icon glyph rendered as punctuation
    # (e.g. "/Summary", "•Education") must not silently count as a
    # capitalised "word" just because it starts with a non-letter.
    alpha_words = [w for w in words if w[:1].isalpha()]
    if not (2 <= len(alpha_words) <= 4):
        return None
    if alpha_len < 0.7 * len(ln.replace(" ", "")):
        return None
    if any(ch.isdigit() for ch in ln):
        return None
    if ln.lower() in _NAME_STOPWORDS:
        return None
    if not all(w[:1].isupper() for w in alpha_words):
        return None
    # Reject known non-name technical/label vocabulary that can otherwise
    # slip through as "2-4 title-case words" (tool/platform names, resume
    # field labels, etc.).
    if any(w.lower().rstrip(".,") in _NON_NAME_WORDS for w in alpha_words):
        return None
    return " ".join(alpha_words)


_NON_NAME_WORDS = {
    "deployment", "render", "vercel", "heroku", "netlify", "aws", "gcp",
    "azure", "docker", "kubernetes", "cloud", "database", "databases",
    "frameworks", "libraries", "tools", "platforms", "stack", "languages",
    "programming", "skills", "technical", "core", "concepts", "soft",
    "contact", "email", "phone", "mobile", "address", "linkedin", "github",
    "portfolio", "website", "objective", "summary", "profile", "resume",
    "curriculum", "vitae", "references", "declaration",
}


def _find_name_in_header_section(text: str) -> Optional[str]:
    """
    Search only within the document's own "header" block — everything
    before the first recognised section keyword (Summary/Education/
    Experience/…). This is structurally where a name belongs, and keeps
    the search from wandering into an unrelated line deep in the document
    that happens to look title-case.
    """
    sections = extract_document_sections(text)
    header_lines = sections.get("header") or []
    for ln in header_lines[:10]:
        name = _looks_like_name_line(ln)
        if name:
            return name
    return None


def _find_name_line(text: str, window: int = 8) -> Optional[str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for ln in lines[:window]:
        name = _looks_like_name_line(ln)
        if name:
            return name
    return None


def _find_name_near_contact(text: str, email_match, phone_match) -> Optional[str]:
    """
    If the strict top-of-document scan found nothing (common when a resume
    layout places a photo/sidebar first and PDF layout-mode extraction
    reorders the visible header), look at the few lines immediately
    surrounding the email/phone match instead — a name is almost always
    adjacent to its own contact details.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    anchor_text = (email_match.group(0) if email_match else
                   phone_match.group(0) if phone_match else None)
    if not anchor_text:
        return None
    anchor_idx = next((i for i, ln in enumerate(lines) if anchor_text in ln), None)
    if anchor_idx is None:
        return None
    for i in list(range(max(0, anchor_idx - 3), anchor_idx)) + \
             list(range(anchor_idx + 1, min(len(lines), anchor_idx + 4))):
        name = _looks_like_name_line(lines[i])
        if name:
            return name
    return None


def _name_from_filename(filename: str) -> Optional[str]:
    """Last-resort name guess from the uploaded filename itself."""
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", filename or "")
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = re.sub(r"\b(resume|cv|profile|final|updated|v\d+|copy)\b", "", stem, flags=re.I)
    words = [w for w in stem.split() if w.isalpha()]
    if 2 <= len(words) <= 4 and all(w[:1].isalpha() for w in words):
        return " ".join(w.capitalize() for w in words)
    return None


def extract_document_sections(raw_text: str) -> Dict[str, List[str]]:
    """
    Best-effort split of a resume/profile-style document into named
    sections (summary, education, experience, skills, projects, …), so
    section-specific questions and overviews can be answered directly
    from the document's own structure instead of raw TF-IDF chunks.
    """
    text = clean_extracted_text(raw_text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    sections: Dict[str, List[str]] = {"header": []}
    current = "header"

    for ln in lines:
        m = _SECTION_HEADER_RE.search(ln)
        alpha_len = len(re.sub(r"[^A-Za-z]", "", ln)) or 1
        # Only treat this as a section boundary if the keyword makes up
        # most of the line's letters — a real heading like "Education",
        # not an incidental mention inside a longer sentence.
        if m and len(m.group(1)) >= 0.4 * alpha_len:
            current = m.group(1).lower()
            sections.setdefault(current, [])
            trailing = ln[m.end():].strip(" :-—")
            if trailing:
                sections[current].append(trailing)
            continue
        sections.setdefault(current, []).append(ln)

    return sections


def answer_structured_document_query(doc: "ParsedDocument", query: str) -> Optional[str]:
    """
    Try to answer a specific factual question about a text document
    directly and correctly, without any LLM involved. Returns None if the
    question doesn't map to anything extractable, so the caller can fall
    back to retrieval/LLM.
    """
    if doc.kind != "text" or not doc.raw_text:
        return None

    q = query.lower()
    facts = extract_key_facts(doc.raw_text)
    if not facts["name"]:
        facts["name"] = _name_from_filename(doc.filename)
    sections = extract_document_sections(doc.raw_text)

    def _section_text(*keys, limit: int = 6) -> Optional[str]:
        for k in keys:
            lines = sections.get(k)
            if lines:
                return "\n".join(f"- {ln}" for ln in lines[:limit] if ln)
        return None

    if re.search(r"\b(his|her|their|the (?:person|candidate)'?s?)?\s*name\b", q) or \
            q.strip() in ("who is this", "who is he", "who is she", "whose resume is this"):
        if facts["name"]:
            return f"The name on the document is **{facts['name']}**."

    if any(w in q for w in ("email", "e-mail", "mail id", "mail address")):
        if facts["email"]:
            return f"Email: {facts['email']}"

    if any(w in q for w in ("phone", "mobile", "contact number", "number", "call")):
        if facts["phone"]:
            return f"Phone number: {facts['phone']}"

    if any(w in q for w in ("linkedin", "github", "portfolio", "link")):
        if facts["links"]:
            return "Link(s) found in the document:\n" + "\n".join(facts["links"])

    if any(w in q for w in ("education", "college", "degree", "university", "school", "cgpa")):
        sec = _section_text("education")
        if sec:
            return f"Education:\n{sec}"

    if any(w in q for w in ("experience", "internship", "employment", " job", "work history")):
        sec = _section_text("experience", "employment", "work experience")
        if sec:
            return f"Experience:\n{sec}"

    if any(w in q for w in ("skill", "technolog", "tech stack", "programming language")):
        sec = _section_text("skills", "technical skills")
        if sec:
            return f"Skills:\n{sec}"

    if "project" in q:
        sec = _section_text("projects", "selected projects")
        if sec:
            return f"Projects:\n{sec}"

    return None


def _truncate_at_word(text: str, limit: int) -> str:
    """Trim to `limit` chars without cutting a word in half."""
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(".,;: ")
    return cut + "…"


def build_document_overview(doc: "ParsedDocument") -> Optional[str]:
    """
    A real, structured overview (name / contact / summary / education /
    experience / skills) built from the document's own sections — used
    for broad "describe / summarize / key points" requests instead of a
    raw, possibly-duplicated chunk dump.
    """
    if doc.kind != "text" or not doc.raw_text:
        return None

    facts = extract_key_facts(doc.raw_text)
    if not facts["name"]:
        facts["name"] = _name_from_filename(doc.filename)
    sections = extract_document_sections(doc.raw_text)
    parts: List[str] = []

    if facts["name"]:
        parts.append(f"**Name:** {facts['name']}")
    contact_bits = [b for b in (facts["email"], facts["phone"]) if b]
    if contact_bits:
        parts.append(f"**Contact:** {' | '.join(contact_bits)}")

    summary_lines = sections.get("summary") or sections.get("objective") or sections.get("profile")
    if summary_lines:
        parts.append("**Summary:** " + _truncate_at_word(" ".join(summary_lines), 500))

    for label, keys in (
        ("Education", ("education",)),
        ("Experience", ("experience", "employment", "work experience")),
        ("Skills", ("skills", "technical skills")),
        ("Projects", ("projects", "selected projects")),
    ):
        for k in keys:
            lines = sections.get(k)
            if lines:
                joined = "\n".join(f"- {ln}" for ln in lines[:5] if ln)
                if joined:
                    parts.append(f"**{label}:**\n{joined}")
                break

    if not parts:
        return None
    return "\n\n".join(parts)
