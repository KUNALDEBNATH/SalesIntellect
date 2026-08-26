# Sales LLM — Built From Scratch

## What "from scratch" means here

Every parameter in this model starts as a **random number**.  
No pretrained weights from Qwen, GPT, LLaMA, or any other model are used.  
The model learns **only** from your three sales CSV files.

---

## Files

| File | Purpose |
|---|---|
| `scratch_llm.py` | Everything: custom tokenizer, transformer architecture, training loop, inference API |
| `patch_test_py.py` | One-time script to swap Qwen out of `test.py` and plug the scratch LLM in |

---

## Architecture (all hand-written in PyTorch)

```
Input tokens
    │
    ▼
Token Embedding (learned, random init)
  + Positional Embedding (learned, random init)
    │
    ▼
┌─────────────────────────────┐  ×4 layers
│  RMSNorm                    │
│  Multi-Head Causal Attention│  (4 heads, d=128)
│  Residual add               │
│  RMSNorm                    │
│  Feed-Forward (GELU, d=512) │
│  Residual add               │
└─────────────────────────────┘
    │
    ▼
RMSNorm → Linear (d=128 → vocab)
    │
    ▼
Next-token logits
```

- **Parameters**: ~4 million (intentionally tiny — your dataset has ~100K tokens)
- **Tokenizer**: Character-level BPE, trained on your corpus, 4 096-token vocabulary
- **Training objective**: Causal language modelling (predict next token)
- **Loss only on response tokens** — the prompt is masked with `-100`

---

## Quick start

### 1. Train the model

```bash
# Run from the directory that contains your 3 CSVs
cd /path/to/your/project
python scratch_llm.py train
```

Training takes **5–30 minutes** on CPU, **1–5 minutes** on GPU.  
Output: `scratch_model/model.pt` and `scratch_model/vocab.json`.

### 2. Test generation

```bash
python scratch_llm.py sample "What is the status of enquiry ENQ001?"
```

### 3. Plug into your existing chatbot

```bash
# Backup test.py, then apply the patch
python patch_test_py.py
```

The patch makes two targeted edits to `test.py`:
- Replaces `from transformers import ...` + `_load_llm()` with `ScratchLLM`
- Replaces the Qwen generation call inside `_generate_phrase()` with `ScratchLLM.generate()`

Everything else (`IntelligentRetriever`, `build_answer`, `detect_intent`, all the
rule-based logic) is untouched — it already works without any LLM.

### 4. Start the server

```bash
python api.py
```

---

## How it actually works in the chatbot

The scratch LLM is **only used for response phrasing**, not for retrieval
or intent detection.  The data pipeline is:

```
User query
    │
    ▼
domain_guard.py  ──── out-of-domain? → refuse
    │
    ▼
detect_intent()  ──── TF-IDF intent classifier (no LLM)
    │
    ▼
IntelligentRetriever.retrieve()  ──── TF-IDF search over CSVs
    │
    ▼
build_answer()   ──── deterministic template answer  ← correct answer lives here
    │
    ▼
ScratchLLM.generate()  ──── optional phrasing polish  ← scratch model here
    │
    ▼
Response to user
```

If the scratch LLM produces garbage (it will during early epochs), `_generate_phrase()`
falls back to the template answer automatically — the chatbot stays correct.

---

## Honest limitations

| Limitation | Explanation |
|---|---|
| The model will sound repetitive | It only learned from ~3 000 pairs. That's normal. |
| It can't generalise beyond your CSVs | It has never read anything else. |
| It won't improve with RLHF / instruction tuning | That requires human labellers and massive compute. |
| It may occasionally repeat or truncate | BPE artefacts on rare names/IDs. |

**The retrieval + template pipeline (`build_answer`) is what makes the chatbot accurate.**  
The scratch LLM only polishes the phrasing of already-correct answers.

---

## Hyperparameters (in `scratch_llm.py`)

| Parameter | Default | Change to… |
|---|---|---|
| `D_MODEL` | 128 | 256 for slightly better quality (4× slower) |
| `N_LAYERS` | 4 | 6 for depth (slower) |
| `N_HEADS` | 4 | 8 (needs `D_MODEL` divisible by 8) |
| `VOCAB_SIZE` | 4096 | 2048 for faster training on small data |
| `EPOCHS` | 30 | 50 if loss hasn't converged |
| `LR` | 3e-4 | 1e-4 if loss spikes |
