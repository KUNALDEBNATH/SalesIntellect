"""
rag_utils.py
────────────────────────────────────────────────────────────────────────────
Small, generic RAG helpers reused by document_parser.py / attachment_handler.py
for retrieval over *uploaded* files.

This intentionally mirrors the architecture already used in test.py
(IntelligentRetriever): TF-IDF vectorisation + cosine similarity. We do not
duplicate that class directly because it is tightly coupled to the sales
CSV schema — instead we reuse the same *technique* for arbitrary uploaded
text so the whole project follows one consistent retrieval philosophy.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """
    Split `text` into overlapping chunks so that large documents are never
    sent to the LLM in one shot.

    chunk_size / overlap are measured in characters, which keeps this
    dependency-free (no tokenizer needed) while still producing chunks that
    comfortably fit inside the LLM's context window.
    """
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        # try to break on a sentence/paragraph boundary near `end`
        boundary = text.rfind(". ", start, end)
        if boundary != -1 and boundary > start + chunk_size * 0.5:
            end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


class SimpleTfidfRetriever:
    """
    Lightweight, ephemeral (per-request) TF-IDF retriever used for RAG over
    an uploaded document's chunks or an uploaded spreadsheet's rows.

    Not persisted to disk — built fresh for every uploaded file, since these
    are user-specific, transient documents (unlike the sales dataset index,
    which IS cached to rag_index.pkl by test.py).
    """

    def __init__(self, texts: List[str]):
        self.texts = texts or []
        if not self.texts:
            self.vec = None
            self.matrix = None
            return
        self.vec = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20_000,
            sublinear_tf=True,
            min_df=1,
            stop_words="english",
        )
        self.matrix = self.vec.fit_transform(self.texts)

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Return [(chunk_text, score), …] sorted by relevance, best first."""
        if not self.texts or self.vec is None:
            return []
        qvec = self.vec.transform([query])
        scores = cosine_similarity(qvec, self.matrix).flatten()
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for i in ranked[:top_k]:
            if scores[i] <= 0:
                continue
            results.append((self.texts[i], float(scores[i])))
        # If nothing scored above zero (e.g. very short/odd query), fall back
        # to the first few chunks so the LLM still has *something* to work with.
        if not results:
            results = [(t, 0.0) for t in self.texts[:top_k]]
        return results
