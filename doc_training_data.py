"""
doc_training_data.py
═══════════════════════════════════════════════════════════════════════════════
Document training data generator for the ScratchLLM.

Generates (prompt, response) pairs that teach the model:
  - Document purpose identification
  - Main topic extraction
  - Document type classification
  - Section summarisation
  - Key fact extraction
  - Claim/evidence/conclusion extraction
  - Document Q&A
  - Cross-section reasoning
  - Summary modes (short, detailed, key-points)

ALL training targets are generated from DETERMINISTIC RULES applied to
real document structure — not from another LLM or external API. The same
document_understanding.py analysis that produces runtime answers also
produces training targets.

Task conditioning: every prompt starts with a <TASK=...> token so the model
learns different output styles for different task types. This is the same
idea as instruction-tuning but built from scratch with no pretrained weights.

Usage:
    from doc_training_data import generate_doc_training_pairs
    pairs = generate_doc_training_pairs(doc_dir="./documents")
    # returns List[Tuple[str, str]] ready for TextDataset

The training pairs use the same (prompt, response) format as the sales CSV
pairs so they can be MIXED into one training batch, teaching the model
both document and sales tasks simultaneously.
"""

from __future__ import annotations

import os
import re
import random
from pathlib import Path
from typing import List, Optional, Tuple

from document_understanding import (
    DocumentRepresentation,
    SectionRepresentation,
    parse_document_structure,
    build_document_representation,
    build_table_representation,
)
from document_answer_engine import (
    _answer_normal_summary,
    _answer_short_summary,
    _answer_detailed_summary,
    _answer_key_points,
    _answer_purpose,
    _answer_document_type,
    _answer_topic,
    _answer_conclusions,
    _answer_claims,
    _answer_evidence,
    _answer_limitations,
    _answer_facts,
    _answer_section_by_section,
)

# ── Synthetic document bank (used when no real documents are available) ───────
# These cover the most important document types and teaching tasks.
# They are NOT hallucinated facts — each is a minimal but structurally
# complete example that teaches the model WHAT KIND OF ANSWER to produce
# for each task type.

SYNTHETIC_DOCS = [
    {
        "filename": "neural_network_paper.pdf",
        "text": """Abstract
This paper proposes a novel attention-based neural network for image classification.
We demonstrate that our approach achieves 94.3% accuracy on the CIFAR-10 benchmark,
outperforming the previous state of the art by 2.1%.

1. Introduction
Image classification is a fundamental problem in computer vision. Existing methods,
including convolutional neural networks and vision transformers, have achieved strong
results. However, these methods require significant computational resources.
We propose a lightweight attention mechanism that reduces computation by 40% while
maintaining competitive accuracy.

2. Related Work
Prior work in this area includes ResNet (He et al., 2016) and Vision Transformer
(Dosovitskiy et al., 2021). Our approach builds on these methods but introduces
a sparse attention pattern that reduces the quadratic attention cost to linear.

3. Methodology
Our model consists of three components: a convolutional backbone, a sparse attention
module, and a classification head. The backbone extracts local features from input
images of size 32×32. The attention module identifies global relationships between
these features. Training uses the AdamW optimizer with a learning rate of 0.001
for 100 epochs on a dataset of 50,000 training images.

4. Results
We evaluate our model on CIFAR-10 and CIFAR-100 benchmarks. On CIFAR-10, we achieve
94.3% top-1 accuracy. On CIFAR-100, we achieve 76.8% top-1 accuracy. These results
surpass the baseline ResNet-50 by 2.1% and 1.9% respectively. Training time is
reduced by 35% compared to the standard vision transformer.

5. Conclusion
We have presented a lightweight attention-based neural network for image classification.
Our model achieves state-of-the-art results while requiring significantly less
computation. Future work will explore applications to video understanding and
3D point cloud classification.
""",
    },
    {
        "filename": "annual_report_2023.pdf",
        "text": """Executive Summary
This report presents the annual performance of Acme Corporation for fiscal year 2023.
Total revenue reached $45.2 million, representing a 12% increase over the prior year.
Operating profit margin improved to 18.3% from 15.1%.

Financial Highlights
Revenue: $45.2M (+12% YoY)
Operating Income: $8.3M (+28% YoY)
Net Profit: $6.1M (+31% YoY)
Total Customers: 1,240 (+8% YoY)
Customer Retention Rate: 89%

Business Review
The company expanded into three new markets in 2023: Southeast Asia, Eastern Europe,
and South America. The Asia-Pacific region showed the strongest growth at 34%.
Product line B accounted for 42% of total revenue. Investment in R&D increased to
$3.2M, up 15% from the prior year.

Challenges and Risks
Supply chain disruptions affected Q2 performance, resulting in a 3% revenue shortfall.
Currency fluctuations reduced international revenue by approximately $0.8M.
Competitive pressure in the North American market led to a 5% decline in that segment.

Outlook
Management projects revenue of $50–52M for fiscal year 2024, assuming stable
macroeconomic conditions. Three new product launches are planned for H1 2024.
The board has approved a capital expenditure budget of $5M for facility expansion.
""",
    },
    {
        "filename": "john_smith_resume.pdf",
        "text": """John Smith
john.smith@email.com | +91 9876543210 | linkedin.com/in/johnsmith

Summary
Software Engineer with 5 years of experience in backend development, cloud architecture,
and machine learning. Passionate about building scalable systems and solving complex
technical problems.

Education
B.Tech in Computer Science — IIT Madras, 2018
CGPA: 8.7/10

Work Experience
Senior Software Engineer — TechCorp Pvt Ltd (2021–Present)
- Led development of microservices architecture serving 2M daily active users
- Reduced API latency by 45% through caching optimisation
- Mentored team of 4 junior developers

Software Engineer — StartupXYZ (2018–2021)
- Built data pipeline processing 500K records per day using Apache Kafka
- Implemented recommendation system improving CTR by 23%

Technical Skills
Programming: Python, Java, Go, SQL
Frameworks: Django, Spring Boot, FastAPI
Cloud: AWS (EC2, S3, Lambda), GCP, Docker, Kubernetes
ML: TensorFlow, PyTorch, scikit-learn

Projects
Distributed Cache System — Built from scratch in Go, handling 100K requests/second
NLP Chatbot — Transformer-based model fine-tuned on customer support data

Certifications
AWS Certified Solutions Architect, Google Cloud Professional Data Engineer
""",
    },
    {
        "filename": "invoice_INV2024001.pdf",
        "text": """INVOICE

Invoice Number: INV-2024-001
Date: January 15, 2024
Due Date: February 14, 2024

From:
Acme Services Pvt Ltd
123 Business Park, Mumbai 400001
GST: 27AAACM1234A1Z5

Bill To:
Global Tech Solutions
456 Tech Hub, Bangalore 560001

Services:
Item 1: Web Development Services (40 hours × ₹2,500/hr) — ₹1,00,000
Item 2: UI/UX Design (15 hours × ₹2,000/hr) — ₹30,000
Item 3: Database Setup and Configuration — ₹15,000
Item 4: Project Management (10 hours × ₹1,500/hr) — ₹15,000

Subtotal: ₹1,60,000
GST (18%): ₹28,800
Total Amount Due: ₹1,88,800

Payment Terms: Net 30 days
Bank: State Bank of India
Account: 1234567890
IFSC: SBIN0001234

Notes: Payment after due date attracts 1.5% monthly interest.
""",
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING PAIR GENERATORS (one per task type)
# ══════════════════════════════════════════════════════════════════════════════

def _task(task_name: str, context: str, question: str, answer: str) -> Tuple[str, str]:
    """Format a training pair with task conditioning."""
    prompt = f"<TASK={task_name}>\n{context}\n\nQuestion: {question}"
    return (prompt, answer.strip())


def _pairs_from_representation(rep: DocumentRepresentation, raw_text: str = "") -> List[Tuple[str, str]]:
    """
    Generate all training pairs for one DocumentRepresentation.
    Each pair covers a different task type so the model learns all tasks.
    """
    pairs: List[Tuple[str, str]] = []
    doc_ctx = f"Document: {rep.filename}\nType: {rep.document_type}\nContent: {raw_text[:600] if raw_text else '(see document)'}"

    # --- TASK 1: DOCUMENT TYPE ---
    pairs.append(_task(
        "DOCUMENT_TYPE", doc_ctx,
        "What type of document is this?",
        _answer_document_type(rep),
    ))

    # --- TASK 2: DOCUMENT PURPOSE ---
    if rep.purpose:
        pairs.append(_task(
            "DOCUMENT_PURPOSE", doc_ctx,
            "What is the purpose of this document?",
            _answer_purpose(rep),
        ))

    # --- TASK 3: MAIN TOPIC ---
    if rep.main_topic:
        pairs.append(_task(
            "DOCUMENT_TOPIC", doc_ctx,
            "What is the main topic of this document?",
            _answer_topic(rep),
        ))
        pairs.append(_task(
            "DOCUMENT_TOPIC", doc_ctx,
            "What is this document about?",
            rep.purpose or f"This document is about {rep.main_topic}.",
        ))

    # --- TASK 4: NORMAL SUMMARY ---
    summary = _answer_normal_summary(rep)
    if summary and len(summary) > 30:
        for q in ["Summarize this document.", "Give me an overview of this document.",
                  "What does this document discuss?"]:
            pairs.append(_task("DOCUMENT_SUMMARY", doc_ctx, q, summary))

    # --- TASK 5: SHORT SUMMARY ---
    short = _answer_short_summary(rep)
    if short and len(short) > 15:
        for q in ["Give me a 5-line summary.", "Briefly describe this document.",
                  "What is the TL;DR of this document?", "In short, what does this say?"]:
            pairs.append(_task("SHORT_SUMMARY", doc_ctx, q, short))

    # --- TASK 6: KEY POINTS ---
    kp = _answer_key_points(rep)
    if kp:
        for q in ["What are the key points?", "What are the main points?",
                  "List the important points.", "What are the highlights?"]:
            pairs.append(_task("KEY_POINTS", doc_ctx, q, kp))

    # --- TASK 7: FACTS ---
    facts_ans = _answer_facts(rep)
    if facts_ans and rep.key_facts:
        pairs.append(_task(
            "FACTS_EXTRACTION", doc_ctx,
            "What are the key facts or findings?",
            facts_ans,
        ))

    # --- TASK 8: CONCLUSIONS ---
    conc_ans = _answer_conclusions(rep)
    if conc_ans and rep.conclusions:
        for q in ["What are the conclusions?", "What does the document conclude?",
                  "What is the main conclusion?"]:
            pairs.append(_task("CONCLUSIONS", doc_ctx, q, conc_ans))

    # --- TASK 9: CLAIMS ---
    if rep.key_claims:
        claims_ans = _answer_claims(rep)
        pairs.append(_task(
            "CLAIMS_EXTRACTION", doc_ctx,
            "What does the author claim in this document?",
            claims_ans,
        ))

    # --- TASK 10: EVIDENCE ---
    if rep.evidence:
        evid_ans = _answer_evidence(rep)
        pairs.append(_task(
            "EVIDENCE_EXTRACTION", doc_ctx,
            "What evidence is provided in this document?",
            evid_ans,
        ))

    # --- TASK 11: SECTION SUMMARIES ---
    for sec in rep.sections:
        if not sec.summary or len(sec.summary) < 20:
            continue
        sec_ctx = f"Document: {rep.filename}\nSection: {sec.title}\nSection content:\n{sec.raw_text[:400]}"
        pairs.append(_task(
            "SECTION_SUMMARY", sec_ctx,
            f"Summarize the {sec.title} section.",
            sec.summary,
        ))
        pairs.append(_task(
            "SECTION_UNDERSTANDING", sec_ctx,
            f"What does the {sec.title} section discuss?",
            sec.summary,
        ))
        if sec.facts:
            pairs.append(_task(
                "SECTION_FACTS", sec_ctx,
                f"What are the key facts in the {sec.title} section?",
                "Key facts from " + sec.title + ":\n" + "\n".join(f"• {f}" for f in sec.facts[:3]),
            ))

    # --- TASK 12: CROSS-SECTION REASONING ---
    if len(rep.sections) >= 2:
        s1 = rep.sections[0]
        for s2 in rep.sections[1:]:
            if s1.summary and s2.summary and s1.importance >= 0.2 and s2.importance >= 0.2:
                cross_ctx = (f"Document: {rep.filename}\n"
                             f"Section A ({s1.title}): {s1.summary[:200]}\n"
                             f"Section B ({s2.title}): {s2.summary[:200]}")
                pairs.append(_task(
                    "CROSS_SECTION_REASONING", cross_ctx,
                    f"How do the {s1.title} and {s2.title} sections relate to each other?",
                    f"The {s1.title} section {_section_relation_verb(s1, s2)} the {s2.title} section. "
                    f"{s1.title} {_section_relation_desc(s1, s2, rep)}",
                ))
                break

    # --- TASK 13: DOCUMENT QA ---
    _add_qa_pairs(pairs, rep, doc_ctx, raw_text)

    # --- TASK 14: ENTITIES ---
    if rep.entities:
        pairs.append(_task(
            "ENTITY_EXTRACTION", doc_ctx,
            "Who or what are the main entities mentioned in this document?",
            f"Key entities mentioned: {', '.join(rep.entities[:8])}.",
        ))

    # --- TASK 15: DETAILED SUMMARY ---
    detailed = _answer_detailed_summary(rep)
    if detailed and len(detailed) > 100:
        pairs.append(_task(
            "DETAILED_SUMMARY", doc_ctx,
            "Give me a detailed summary of this document.",
            detailed[:800],   # cap to avoid excessively long training targets
        ))

    return pairs


def _section_relation_verb(s1: SectionRepresentation, s2: SectionRepresentation) -> str:
    s2_low = s2.title.lower()
    if "result" in s2_low or "finding" in s2_low or "evaluation" in s2_low:
        return "provides the experimental setup and motivation for"
    if "conclusion" in s2_low:
        return "provides the context and evidence leading to"
    if "methodology" in s2_low or "method" in s2_low:
        return "motivates and contextualises"
    return "provides context for"


def _section_relation_desc(s1: SectionRepresentation, s2: SectionRepresentation,
                             rep: DocumentRepresentation) -> str:
    if s1.summary:
        return f"covers {s1.summary.split('.')[0].lower()}, while {s2.title} {s2.summary.split('.')[0].lower() if s2.summary else 'builds on this'}."
    return f"and {s2.title} together form the core of the document."


def _add_qa_pairs(pairs: List, rep: DocumentRepresentation,
                   doc_ctx: str, raw_text: str) -> None:
    """Add document-specific Q&A pairs based on the document type."""
    dtype = rep.document_type

    if dtype == "research_paper":
        if rep.key_claims:
            pairs.append(_task(
                "DOCUMENT_QA", doc_ctx,
                "What is the main contribution of this paper?",
                rep.key_claims[0] if rep.key_claims else rep.purpose,
            ))
        if rep.evidence:
            pairs.append(_task(
                "DOCUMENT_QA", doc_ctx,
                "What experiments were conducted?",
                rep.evidence[0] if rep.evidence else "Experimental details could not be automatically extracted.",
            ))

    elif dtype == "resume":
        pairs.append(_task(
            "DOCUMENT_QA", doc_ctx,
            "Who is this resume for?",
            rep.purpose.replace("This document presents the professional background of", "The resume is for").strip(),
        ))
        pairs.append(_task(
            "DOCUMENT_QA", doc_ctx,
            "What are the person's qualifications?",
            _answer_short_summary(rep),
        ))

    elif dtype == "invoice":
        if rep.key_facts:
            pairs.append(_task(
                "DOCUMENT_QA", doc_ctx,
                "What is the total amount?",
                "Based on the invoice: " + " ".join(f for f in rep.key_facts if any(
                    w in f.lower() for w in ("total", "amount", "₹", "$", "£", "eur")
                ))[:200] or "Total amount information: " + rep.key_facts[0],
            ))

    elif dtype == "report":
        if rep.conclusions:
            pairs.append(_task(
                "DOCUMENT_QA", doc_ctx,
                "What are the main findings?",
                rep.conclusions[0],
            ))

    # Universal QA pair
    pairs.append(_task(
        "DOCUMENT_QA", doc_ctx,
        "What is the most important information in this document?",
        rep.short_summary or rep.purpose,
    ))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DATA GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_from_synthetic_docs() -> List[Tuple[str, str]]:
    """
    Generate training pairs from the synthetic document bank.
    Returns a list of (prompt, response) pairs.
    """
    all_pairs: List[Tuple[str, str]] = []

    for doc_spec in SYNTHETIC_DOCS:
        filename = doc_spec["filename"]
        raw_text = doc_spec["text"]

        # Build structure
        structured = parse_document_structure(raw_text, filename)
        # Build representation
        rep = build_document_representation(structured, debug=False)

        # Generate pairs
        pairs = _pairs_from_representation(rep, raw_text)
        all_pairs.extend(pairs)

    return all_pairs


def generate_from_real_documents(doc_dir: str = ".") -> List[Tuple[str, str]]:
    """
    Generate training pairs from real documents found in `doc_dir`.
    Supports PDF, DOCX, TXT files.
    """
    all_pairs: List[Tuple[str, str]] = []
    doc_path = Path(doc_dir)

    supported = {".pdf", ".docx", ".txt"}
    found = list(doc_path.glob("**/*"))
    doc_files = [f for f in found if f.suffix.lower() in supported]

    if not doc_files:
        return all_pairs

    for doc_file in doc_files[:20]:   # cap at 20 documents to avoid huge datasets
        try:
            # Extract text using the appropriate extractor
            raw_text = _extract_file_text(str(doc_file))
            if not raw_text or len(raw_text) < 100:
                continue

            structured = parse_document_structure(raw_text, doc_file.name)
            rep = build_document_representation(structured, debug=False)

            pairs = _pairs_from_representation(rep, raw_text[:1000])
            all_pairs.extend(pairs)
            print(f"  [DocTraining] {doc_file.name}: {len(pairs)} pairs generated")

        except Exception as e:
            print(f"  [DocTraining] Skipped {doc_file.name}: {e}")

    return all_pairs


def _extract_file_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".txt":
            return open(path, encoding="utf-8", errors="ignore").read()
        elif ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(path)
            return "\n".join(
                (page.extract_text(extraction_mode="layout") or page.extract_text() or "")
                for page in reader.pages
            ).strip()
        elif ext == ".docx":
            import docx
            document = docx.Document(path)
            return "\n".join(p.text for p in document.paragraphs if p.text.strip())
    except Exception:
        return ""
    return ""


def generate_doc_training_pairs(
    doc_dir: str = ".",
    include_synthetic: bool = True,
    augment: bool = True,
) -> List[Tuple[str, str]]:
    """
    Main entry point. Generates all document training pairs.

    Called from train.py to add document understanding capability to SalesGPT.

    Args:
        doc_dir: Directory to scan for real documents (PDF/DOCX/TXT).
        include_synthetic: Include the built-in synthetic document examples.
        augment: Lightly augment prompts for diversity.

    Returns:
        List[Tuple[str, str]] — (prompt, response) pairs in the same format
        as load_training_pairs() in scratch_llm.py.
    """
    all_pairs: List[Tuple[str, str]] = []

    if include_synthetic:
        syn_pairs = generate_from_synthetic_docs()
        all_pairs.extend(syn_pairs)
        print(f"  [DocTraining] Synthetic documents: {len(syn_pairs)} pairs")

    real_pairs = generate_from_real_documents(doc_dir)
    all_pairs.extend(real_pairs)
    print(f"  [DocTraining] Real documents: {len(real_pairs)} pairs")

    if augment and all_pairs:
        all_pairs = _augment_doc_pairs(all_pairs)

    print(f"  [DocTraining] Total document pairs: {len(all_pairs)}")
    return all_pairs


_QUESTION_VARIANTS = {
    "Summarize this document.": [
        "Give me a summary of this document.",
        "Can you summarize this?",
        "What is the summary?",
        "Please summarize.",
    ],
    "What is this document about?": [
        "What does this document cover?",
        "Tell me about this document.",
        "What is the subject of this document?",
    ],
    "What are the key points?": [
        "What are the main points?",
        "List the important points.",
        "What are the highlights?",
        "Give me the key takeaways.",
    ],
    "Give me a 5-line summary.": [
        "Brief summary please.",
        "In short, what does this say?",
        "TL;DR?",
    ],
    "What is the purpose of this document?": [
        "Why was this written?",
        "What is the goal of this document?",
        "What is the objective?",
    ],
}


def _augment_doc_pairs(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Add question-variant forms for common question types."""
    augmented = list(pairs)
    for prompt, response in pairs:
        # Extract the question part (after "Question: ")
        m = re.search(r"Question: (.+)$", prompt, re.M)
        if not m:
            continue
        question = m.group(1).strip()
        if question in _QUESTION_VARIANTS:
            for variant in _QUESTION_VARIANTS[question]:
                new_prompt = prompt.replace(f"Question: {question}", f"Question: {variant}")
                augmented.append((new_prompt, response))
    return augmented
