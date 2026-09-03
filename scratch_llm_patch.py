"""
scratch_llm_patch.py
═══════════════════════════════════════════════════════════════════════════════
PATCH INSTRUCTIONS FOR scratch_llm.py
────────────────────────────────────────
This file documents and implements the changes needed in scratch_llm.py to
support document understanding tasks.

There are THREE changes:

CHANGE 1 — Add task-conditioning tokens to the special token set.
  In scratch_llm.py, update the special tokens section (near top of file)
  by adding TASK_TOKENS and including them in the tokenizer's special set.

CHANGE 2 — Add generate_for_document_task() method to ScratchLLM.
  This is task-conditioned generation with a proper task prefix, instead
  of the generic generate_for_document() which uses no task token.

CHANGE 3 — load_training_pairs() optionally includes document pairs.
  train.py calls this function; it now merges document Q&A pairs in.

HOW TO APPLY:
  Run:  python scratch_llm_patch.py
  This patches scratch_llm.py in-place (saves scratch_llm.py.bak first).

ALTERNATIVELY, manually apply each PATCH below.
"""

import shutil
import sys
from pathlib import Path

SCRATCH_LLM = Path("scratch_llm.py")

# ═══════════════════════════════════════════════════════════════════════════
# PATCH 1 — Task conditioning tokens
# ═══════════════════════════════════════════════════════════════════════════
#
# In scratch_llm.py, find the line:
#   UNK = "<unk>"; PAD = "<pad>"; BOS = "<bos>"; EOS = "<eos>"; SEP = "<sep>"
#
# Replace with:
PATCH1_OLD = 'UNK = "<unk>"; PAD = "<pad>"; BOS = "<bos>"; EOS = "<eos>"; SEP = "<sep>"'

PATCH1_NEW = '''\
UNK = "<unk>"; PAD = "<pad>"; BOS = "<bos>"; EOS = "<eos>"; SEP = "<sep>"

# Task-conditioning tokens (added for document understanding support).
# Including these in the special set means the tokenizer treats them as
# single atoms — the model sees <TASK=DOCUMENT_SUMMARY> as ONE token,
# not character-split fragments, so it reliably learns to condition on it.
TASK_TOKENS = [
    "<TASK=SALES_QA>",
    "<TASK=DOCUMENT_SUMMARY>",
    "<TASK=SHORT_SUMMARY>",
    "<TASK=KEY_POINTS>",
    "<TASK=DETAILED_SUMMARY>",
    "<TASK=DOCUMENT_TYPE>",
    "<TASK=DOCUMENT_PURPOSE>",
    "<TASK=DOCUMENT_TOPIC>",
    "<TASK=SECTION_SUMMARY>",
    "<TASK=SECTION_UNDERSTANDING>",
    "<TASK=SECTION_FACTS>",
    "<TASK=FACTS_EXTRACTION>",
    "<TASK=CLAIMS_EXTRACTION>",
    "<TASK=EVIDENCE_EXTRACTION>",
    "<TASK=CONCLUSIONS>",
    "<TASK=ENTITY_EXTRACTION>",
    "<TASK=CROSS_SECTION_REASONING>",
    "<TASK=DOCUMENT_QA>",
    "<TASK=DOCUMENT_ANSWER>",
]'''

# ═══════════════════════════════════════════════════════════════════════════
# PATCH 2 — Include TASK_TOKENS in the tokenizer's special set
# ═══════════════════════════════════════════════════════════════════════════
#
# In CharBPETokenizer.train(), find the line:
#   special = {PAD, UNK, BOS, EOS, SEP}
#
# Replace with:
PATCH2_OLD = "        special = {PAD, UNK, BOS, EOS, SEP}"
PATCH2_NEW = "        special = {PAD, UNK, BOS, EOS, SEP} | set(TASK_TOKENS)"

# Also update the all_toks list to include TASK_TOKENS:
# Find:
#   all_toks = [PAD, UNK, BOS, EOS, SEP] + sorted(s for s in all_syms if s not in special)
# Replace with:
PATCH3_OLD = "        all_toks = [PAD, UNK, BOS, EOS, SEP] + sorted(s for s in all_syms if s not in special)"
PATCH3_NEW = "        all_toks = [PAD, UNK, BOS, EOS, SEP] + TASK_TOKENS + sorted(s for s in all_syms if s not in special)"

# ═══════════════════════════════════════════════════════════════════════════
# PATCH 3 — Task-conditioned generation method on ScratchLLM
# ═══════════════════════════════════════════════════════════════════════════
#
# In the ScratchLLM class, after generate_for_document(), add:

TASK_GENERATE_METHOD = '''
    def generate_for_document_task(
        self,
        task: str,
        context: str,
        question: str,
        max_new: int = 180,
        temperature: float = 0.45,
        top_p: float = 0.9,
        top_k: int = 50,
        rep_penalty: float = 1.2,
    ) -> "Optional[str]":
        """
        Task-conditioned document generation.

        Formats the prompt with a <TASK=...> prefix token so the model
        uses its document-task-specific weights (if trained on doc data)
        rather than its default sales Q&A behaviour.

        This replaces the bare generate_for_document() call for all
        document tasks. generate_for_document() remains for backwards
        compatibility with vision_parser.py which does not use task tokens.

        Args:
            task    : Task token name, e.g. "DOCUMENT_SUMMARY" (without <TASK=>)
            context : Document context (filename, type, content excerpt)
            question: The user's question
        """
        if not self.ready or not self._model or not self._tokenizer:
            return None
        prompt = f"<TASK={task}>\\n{context}\\n\\nQuestion: {question}"
        return self._neural_generate(prompt, max_new, temperature, top_p, top_k, rep_penalty)
'''

# The insertion point: find the end of generate_for_document(), just before _neural_generate
PATCH4_OLD = "    def _neural_generate(self, prompt: str, max_new: int, temperature: float,"
PATCH4_NEW = TASK_GENERATE_METHOD + "\n    def _neural_generate(self, prompt: str, max_new: int, temperature: float,"


def apply_patch() -> bool:
    """Apply all patches to scratch_llm.py."""
    if not SCRATCH_LLM.exists():
        print(f"[Error] {SCRATCH_LLM} not found. Run from the project directory.")
        return False

    src = SCRATCH_LLM.read_text(encoding="utf-8")
    bak = SCRATCH_LLM.with_suffix(".py.bak2")
    shutil.copy(SCRATCH_LLM, bak)
    print(f"  Backup → {bak}")

    changed = 0

    if PATCH1_OLD in src:
        src = src.replace(PATCH1_OLD, PATCH1_NEW, 1)
        print("  ✓ Patch 1: Task conditioning tokens added")
        changed += 1
    else:
        print("  [INFO] Patch 1 already applied or not found")

    if PATCH2_OLD in src:
        src = src.replace(PATCH2_OLD, PATCH2_NEW, 1)
        print("  ✓ Patch 2: Tokenizer special set updated")
        changed += 1

    if PATCH3_OLD in src:
        src = src.replace(PATCH3_OLD, PATCH3_NEW, 1)
        print("  ✓ Patch 3: TASK_TOKENS added to vocab list")
        changed += 1

    if PATCH4_OLD in src and "generate_for_document_task" not in src:
        src = src.replace(PATCH4_OLD, PATCH4_NEW, 1)
        print("  ✓ Patch 4: generate_for_document_task() method added")
        changed += 1
    else:
        print("  [INFO] Patch 4 already applied")

    SCRATCH_LLM.write_text(src, encoding="utf-8")
    print(f"\n  ✓ scratch_llm.py patched ({changed} change(s) applied)")
    return True


if __name__ == "__main__":
    success = apply_patch()
    sys.exit(0 if success else 1)
