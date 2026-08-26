"""
patch_test_py.py
═══════════════════════════════════════════════════════════════════════════════
Replace the Qwen-based LLM section in test.py with the custom scratch LLM.

Apply this patch by running:
    python patch_test_py.py

It reads your existing test.py, applies the changes in-place, and writes
the patched version back.  A backup is saved to test.py.bak first.

What changes
────────────
OLD (lines 1-23 of test.py):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    ...
    CHAT_MODELS = ["sales_llm_model", "Qwen/Qwen2.5-1.5B-Instruct", ...]
    _llm_tok = None
    _llm_mdl = None
    _llm_ready = False
    def _load_llm() -> bool: ...   (loads Qwen via transformers)

NEW:
    from scratch_llm import ScratchLLM as _ScratchLLM
    _scratch_llm = _ScratchLLM()
    _llm_ready   = False

    def _load_llm() -> bool:
        global _llm_ready
        _llm_ready = _scratch_llm.load()
        return _llm_ready

    # _generate_phrase is also replaced — same signature, same output contract.
"""

import re
import shutil
import sys
from pathlib import Path

TEST_PY = Path("test.py")

# ── Old block to replace ──────────────────────────────────────────────────────
OLD_IMPORTS = """\
from transformers import AutoTokenizer, AutoModelForCausalLM

LLM_MODEL_DIR   = "sales_llm_model"
LLM_DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
LLM_MAX_TOKENS  = 80
LLM_TEMPERATURE = 0.3
LLM_TOP_P       = 0.9

CHAT_MODELS = [
    LLM_MODEL_DIR,                   # your fine-tuned model (train.py output), tried first
    "Qwen/Qwen2.5-1.5B-Instruct",    # fallback if sales_llm_model/ is missing or fails to load
    "Qwen/Qwen2.5-0.5B-Instruct",
]

_llm_tok   = None
_llm_mdl   = None
_llm_ready = False


def _load_llm() -> bool:
    \\"\\"\\"
    global _llm_tok, _llm_mdl, _llm_ready
    if _llm_ready:
        return True
    for model_path in CHAT_MODELS:
        if model_path == LLM_MODEL_DIR and not os.path.isdir(model_path):
            print(f"  [LLM] Local fine-tuned model '{model_path}' not found — skipping.")
            continue
        try:
            print(f"  [LLM] Loading '{model_path}' on {LLM_DEVICE} …")
            tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            dtype = torch.float16 if LLM_DEVICE == "cuda" else torch.float32
            mdl   = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=LLM_DEVICE,
                trust_remote_code=True,
            )
            mdl.eval()
            _llm_tok, _llm_mdl, _llm_ready = tok, mdl, True
            print(f"  [LLM] Ready: {model_path}")
            return True
        except Exception as e:
            print(f"  [LLM] '{model_path}' failed: {e}")
    print("  [LLM] No model available — using structured answers only.")
    return False"""

NEW_IMPORTS = """\
# ── Scratch-built LLM (no pretrained weights) ────────────────────────────────
from scratch_llm import ScratchLLM as _ScratchLLM

LLM_DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
LLM_MAX_TOKENS  = 80
LLM_TEMPERATURE = 0.7
LLM_TOP_P       = 0.9

_scratch_llm = _ScratchLLM()
_llm_ready   = False


def _load_llm() -> bool:
    \\"\\"\\"Load the scratch-trained SalesGPT model.\\"\\"\\"
    global _llm_ready
    _llm_ready = _scratch_llm.load()
    return _llm_ready"""

# ── Old _generate_phrase call inside the function ─────────────────────────────
# We don't change _generate_phrase's SIGNATURE — only its internals — so
# the rest of test.py (which calls _generate_phrase) requires zero edits.

OLD_GENERATE_INNER = """\
    try:
        try:
            prompt = _llm_tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = (
                f"<s>[INST] <<SYS>>\\n{_SALES_SYSTEM_PROMPT}\\n<</SYS>>\\n\\n"
                f"{user_content} [/INST]"
            )

        inputs = _llm_tok(
            prompt, return_tensors="pt",
            truncation=True, max_length=1024
        ).to(LLM_DEVICE)

        if hasattr(_llm_mdl, "generation_config"):
            _llm_mdl.generation_config.max_length = None

        with torch.no_grad():
            output = _llm_mdl.generate(
                **inputs,
                max_new_tokens=LLM_MAX_TOKENS,
                do_sample=True,
                temperature=LLM_TEMPERATURE,
                top_p=LLM_TOP_P,
                repetition_penalty=1.2,
                pad_token_id=_llm_tok.pad_token_id,
                eos_token_id=_llm_tok.eos_token_id,
            )

        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        raw = _llm_tok.decode(new_tokens, skip_special_tokens=True).strip()"""

NEW_GENERATE_INNER = """\
    try:
        # Build a compact prompt string for the scratch LLM
        prompt_for_llm = f"{user_content}"
        raw = _scratch_llm.generate(
            prompt_for_llm,
            max_new     = LLM_MAX_TOKENS,
            temperature = LLM_TEMPERATURE,
            top_p       = LLM_TOP_P,
        )
        if not raw:
            return None"""


def patch() -> None:
    if not TEST_PY.exists():
        sys.exit(f"[Error] {TEST_PY} not found. Run from the project directory.")

    src = TEST_PY.read_text(encoding="utf-8")

    # Backup
    bak = TEST_PY.with_suffix(".py.bak")
    shutil.copy(TEST_PY, bak)
    print(f"  Backup saved → {bak}")

    # ── Patch 1: replace imports + _load_llm ─────────────────────────────────
    if "from transformers import AutoTokenizer, AutoModelForCausalLM" in src:
        # Find the block up to the end of _load_llm by searching line-by-line
        lines = src.splitlines(keepends=True)
        start_idx = end_idx = None
        in_load_llm = False
        for i, line in enumerate(lines):
            if "from transformers import AutoTokenizer, AutoModelForCausalLM" in line:
                start_idx = i
            if start_idx is not None and "def _load_llm" in line:
                in_load_llm = True
            if in_load_llm and i > start_idx and line.strip() == "":
                # First blank line AFTER the function body
                # Check the next non-blank line doesn't look like a continuation
                rest = "".join(lines[i+1:i+4]).strip()
                if rest and not rest.startswith(" ") and not rest.startswith("\t"):
                    end_idx = i
                    break

        if start_idx is not None and end_idx is not None:
            replacement = NEW_IMPORTS + "\n\n"
            new_lines   = lines[:start_idx] + [replacement] + lines[end_idx:]
            src         = "".join(new_lines)
            print("  ✓ Replaced import block + _load_llm")
        else:
            print("  [WARNING] Could not locate exact import block; skipping patch 1.")
    else:
        print("  [INFO] Import block already patched or not found.")

    # ── Patch 2: replace interior of _generate_phrase's try block ────────────
    # We use a simple string find; the inner try is unique enough.
    INNER_MARKER = "prompt = _llm_tok.apply_chat_template("
    if INNER_MARKER in src:
        # Find the outer try block start and swap to the scratch version
        # Strategy: replace from the "try:" that contains INNER_MARKER
        # up to (not including) the post-generation "for art in _LLM_ARTIFACTS" line
        ART_MARKER = "        for art in _LLM_ARTIFACTS:"
        i_start = src.find("    try:\n        try:\n            prompt = _llm_tok")
        i_end   = src.find(ART_MARKER)
        if i_start != -1 and i_end != -1:
            src = src[:i_start] + NEW_GENERATE_INNER + "\n\n" + src[i_end:]
            print("  ✓ Replaced _generate_phrase inner try block")
        else:
            print("  [WARNING] Could not pinpoint try-block boundaries; skipping patch 2.")
    else:
        print("  [INFO] _generate_phrase already patched or not found.")

    # ── Remove the _LLM_ARTIFACTS cleanup (no longer needed) ─────────────────
    # Those artifact strings were Qwen-specific chat-template leftovers.
    # The scratch model doesn't produce them.
    ARTIFACT_BLOCK_START = "_LLM_ARTIFACTS = ["
    ARTIFACT_BLOCK_END   = "]\n\n"
    if ARTIFACT_BLOCK_START in src:
        i1 = src.find(ARTIFACT_BLOCK_START)
        i2 = src.find(ARTIFACT_BLOCK_END, i1)
        if i1 != -1 and i2 != -1:
            # Keep the closing bracket + newlines so subsequent code is intact
            # Just comment the block out rather than deleting it.
            block = src[i1: i2 + len(ARTIFACT_BLOCK_END)]
            commented = "\n".join("# " + l for l in block.splitlines()) + "\n\n"
            src = src[:i1] + commented + src[i2 + len(ARTIFACT_BLOCK_END):]
            print("  ✓ Commented out _LLM_ARTIFACTS (not needed for scratch model)")

    TEST_PY.write_text(src, encoding="utf-8")
    print(f"\n  ✓ Patched test.py written successfully.\n")
    print("  Next steps:")
    print("    1.  python scratch_llm.py train     # ~5–30 min depending on hardware")
    print("    2.  python test.py                  # chatbot starts with scratch LLM")
    print("    3.  python api.py                   # or run the Django API\n")


if __name__ == "__main__":
    patch()
