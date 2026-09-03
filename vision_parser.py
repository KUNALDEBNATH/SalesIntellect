"""
vision_parser.py
═══════════════════════════════════════════════════════════════════════════════
Handles image uploads (PNG / JPG / JPEG / WEBP).

ALL Qwen VLM references removed.

Strategy (in order):
  1. Tesseract OCR  — extract visible text from the image.
  2. ScratchLLM     — if OCR finds text, pass it to the scratch LLM to
                       answer the user's question about the image content.
  3. Pixel-level heuristics — if OCR finds nothing, return a structured
                       description (dimensions, dominant colours, brightness).

No pretrained vision model is loaded at any point.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from PIL import Image

# ── Scratch LLM (text, for answering questions about OCR-extracted content) ──
# Reuse the single instance test.py loads at startup instead of loading a
# second copy of the same weights — see attachment_handler.py for the full
# rationale.
try:
    from test import _scratch_llm, _load_llm as _ensure_llm
except ImportError:
    from scratch_llm import ScratchLLM as _ScratchLLM

    _scratch_llm  = _ScratchLLM()
    _llm_lock     = threading.Lock()
    _llm_loaded   = False

    def _ensure_llm() -> bool:
        global _llm_loaded
        if _llm_loaded:
            return True
        with _llm_lock:
            if not _llm_loaded:
                _llm_loaded = _scratch_llm.load()
        return _llm_loaded

# ── Anti-hallucination verification, same guarantee attachment_handler.py
# already applies to document answers — OCR'd images had no such check. ──
from document_verifier import ground_answer_or_fallback

# Reuse the same from-scratch extractive-QA engine document_parser.py uses
# for uploaded text documents, so a specific question about an OCR'd image
# ("what's the total on this invoice?", "what's the phone number here?")
# gets a precise, grounded sentence pulled straight from the OCR text
# instead of depending on the tiny neural net, which was never trained on
# this kind of content and is unreliable for it.
from document_parser import ParsedDocument, extractive_qa, is_generic_overview_request, summarize_text


# ════════════════════════════════════ OCR ═════════════════════════════════════

def _ocr(image_path: str) -> str:
    """Extract visible text from an image using Tesseract OCR."""
    try:
        import pytesseract
        img  = Image.open(image_path).convert("RGB")
        text = pytesseract.image_to_string(img, config="--psm 6").strip()
        return text
    except ImportError:
        return ""
    except Exception as e:
        print(f"  [OCR] Error: {e}")
        return ""


# ════════════════════════════════ PIXEL HEURISTICS ════════════════════════════

def _pixel_description(image_path: str) -> str:
    """
    Fallback when OCR finds no text: return a structured description built
    from image metadata and basic pixel statistics — no ML model needed.
    """
    try:
        img  = Image.open(image_path).convert("RGB")
        w, h = img.size
        fmt  = Path(image_path).suffix.upper().lstrip(".")

        import numpy as np
        arr  = np.array(img, dtype=float)
        mean = arr.mean(axis=(0, 1))           # [R, G, B] mean
        brightness = float(mean.mean())

        r, g, b = mean
        if brightness > 200:
            tone = "light / white-dominant"
        elif brightness < 60:
            tone = "dark / black-dominant"
        elif r > g and r > b:
            tone = "red-dominant"
        elif g > r and g > b:
            tone = "green-dominant"
        elif b > r and b > g:
            tone = "blue-dominant"
        else:
            tone = "neutral / mixed colours"

        return (
            f"Image analysis (pixel heuristics — no OCR text detected):\n"
            f"  Format     : {fmt}\n"
            f"  Dimensions : {w} × {h} pixels\n"
            f"  Brightness : {brightness:.1f}/255  ({tone})\n"
            f"  Avg colour : R={r:.0f}  G={g:.0f}  B={b:.0f}\n\n"
            f"No readable text was detected in this image via OCR. "
            f"If this is a chart or diagram, please describe what you need "
            f"and I will try to help based on any text labels visible."
        )
    except Exception as e:
        return (
            f"Could not analyse the image: {e}. "
            "Please check that the file is a valid PNG, JPG, JPEG, or WEBP."
        )


# ════════════════════════════════ PUBLIC API ══════════════════════════════════

def analyze_image(image_path: str, query: str) -> str:
    """
    Answer a question about an uploaded image.

    Pipeline:
      1. OCR  → extract text
      2. If text found: ScratchLLM answers the question using the OCR text
      3. If no text   : pixel heuristics description
    """
    ocr_text = _ocr(image_path)

    if ocr_text and len(ocr_text.strip()) >= 10:
        pseudo_doc = ParsedDocument(kind="text", filename=image_path, raw_text=ocr_text)

        # ── Step 0: deterministic extractive QA over the OCR'd text ────────
        # Skipped for broad "describe this image" requests, which want a
        # summary rather than the single sentence closest to the (generic)
        # question.
        if not is_generic_overview_request(query):
            qa_hits = extractive_qa(pseudo_doc, query, top_k=3)
            if qa_hits:
                answer = "\n".join(f"- {s}" for s in qa_hits) if len(qa_hits) > 1 else qa_hits[0]
                return (
                    f"[Image analysed via OCR]\n\n{answer}\n\n"
                    f"--- Extracted text ---\n{ocr_text[:800]}"
                    + (" …" if len(ocr_text) > 800 else "")
                )

        # ── Try scratch LLM first ─────────────────────────────────────────────
        if _ensure_llm():
            prompt = (
                f"The following text was extracted from an uploaded image:\n\n"
                f"{ocr_text[:1500]}\n\n"
                f"Question: {query}"
            )
            answer = _scratch_llm.generate_for_document(prompt, max_new=120, temperature=0.5, top_p=0.9)
            if answer and len(answer.strip()) >= 15:
                BAD = ["i don't know", "i cannot", "as an ai", "i was trained",
                       "no information", "not able to"]
                if not any(b in answer.lower() for b in BAD):
                    grounded = ground_answer_or_fallback(
                        answer.strip(), [ocr_text], document_filename=image_path,
                        fallback_context=ocr_text,
                    )
                    return (
                        f"[Image analysed via OCR]\n\n"
                        f"{grounded}\n\n"
                        f"--- Extracted text ---\n{ocr_text[:800]}"
                        + (" …" if len(ocr_text) > 800 else "")
                    )

        # ── Scratch LLM unavailable or returned nothing: return raw OCR ───────
        return (
            f"[OCR text extracted from image]\n\n"
            f"{ocr_text[:2000]}"
            + (" …" if len(ocr_text) > 2000 else "")
        )

    # ── No readable text in the image ─────────────────────────────────────────
    return _pixel_description(image_path)


def vlm_status() -> dict:
    """
    Returns status dict compatible with api.py's HealthView.
    Since there is no VLM, vlm_ready is always False and we report
    OCR availability instead.
    """
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        ocr_available = True
    except Exception:
        ocr_available = False

    return {
        "vlm_ready":      False,
        "vlm_attempted":  False,
        "ocr_available":  ocr_available,
        "image_backend":  "OCR + ScratchLLM (no pretrained vision model)",
    }
