"""
train_with_documents.py
═══════════════════════════════════════════════════════════════════════════════
Trains SalesGPT on BOTH sales CSV Q&A AND document understanding tasks.

GPU confirmed working on RTX 4060 — this version fixes:
  1. _save_model() now passes all 7 required args (model, tokenizer,
     d_model, n_heads, n_kv_heads, n_layers, d_ff)
  2. torch.amp.autocast('cuda', ...) replaces deprecated
     torch.cuda.amp.autocast() for PyTorch 2.6+

Run:
    python train_with_documents.py --epochs 80
    python train_with_documents.py --epochs 80 --batch-mult 8
    python train_with_documents.py --fp32        # disable mixed precision
    python train_with_documents.py --cpu         # force CPU (debug only)
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from scratch_llm import (
    CharBPETokenizer,
    SalesGPT,
    TextDataset,
    ScratchLLM,
    SAVE_DIR,
    MODEL_PATH,
    VOCAB_PATH,
    VOCAB_SIZE,
    D_MODEL,
    N_HEADS,
    N_KV_HEADS,
    N_LAYERS,
    D_FF,
    MAX_SEQ_LEN,
    DROPOUT,
    BATCH_SIZE,
    EPOCHS,
    LR,
    GRAD_CLIP,
    EVAL_EVERY,
    SEED,
    CSV_DIR,
    _save_model,
    load_training_pairs,
)

from doc_training_data import generate_doc_training_pairs


# ══════════════════════════════════════════════════════════════════════════════
# GPU ENVIRONMENT CHECK
# ══════════════════════════════════════════════════════════════════════════════

def check_gpu_environment() -> tuple:
    """
    Detect whether a CUDA-capable GPU is usable.
    Returns (device_str, cuda_ok).
    """
    cuda_ok = torch.cuda.is_available()

    print(f"\n{'─'*64}")
    print(f"  PyTorch version : {torch.__version__}")
    print(f"  CUDA built-in   : {torch.version.cuda or 'None  ← CPU-only build'}")

    if cuda_ok:
        device_name = torch.cuda.get_device_name(0)
        vram        = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU             : {device_name}")
        print(f"  VRAM            : {vram:.1f} GB")
        print(f"  CUDA runtime    : {torch.version.cuda}")
        print(f"  cuDNN           : {torch.backends.cudnn.version()}")
        device = "cuda"
    else:
        print(f"  GPU             : NOT available to PyTorch")
        print()
        if torch.version.cuda is None:
            print("  ╔══════════════════════════════════════════════════════╗")
            print("  ║  CPU-ONLY PYTORCH — reinstall with CUDA support:    ║")
            print("  ╠══════════════════════════════════════════════════════╣")
            print("  ║  pip uninstall torch torchvision torchaudio -y      ║")
            print("  ║  pip install torch torchvision torchaudio           ║")
            print("  ║      --index-url https://download.pytorch.org/      ║")
            print("  ║                         whl/cu124                   ║")
            print("  ╚══════════════════════════════════════════════════════╝")
            print()
            ans = input("  Continue on CPU anyway? [y/N]: ").strip().lower()
            if ans != "y":
                print("  Exiting. Reinstall PyTorch with CUDA support first.")
                sys.exit(1)
        device = "cpu"

    print(f"{'─'*64}\n")
    return device, cuda_ok


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _cosine_lr(step: int, warmup: int, total: int, lr: float) -> float:
    if step < warmup:
        return lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return lr * 0.5 * (1 + math.cos(math.pi * progress))


def _make_dataloader(pairs, tokenizer, max_seq, batch_size,
                     shuffle, pin_memory, num_workers) -> DataLoader:
    ds = TextDataset(pairs, tokenizer, max_seq)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )


def _log_gpu() -> str:
    if not torch.cuda.is_available():
        return ""
    alloc = torch.cuda.memory_allocated(0) / 1024**2
    return f" | GPU {alloc:.0f}MB"


def _save(model, tokenizer) -> None:
    """
    Call _save_model with all 7 required positional arguments.
    Matches the signature in scratch_llm.py:
        _save_model(model, tokenizer, d_model, n_heads, n_kv_heads, n_layers, d_ff)
    """
    _save_model(model, tokenizer, D_MODEL, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF)


def _eval_loss(model, val_dl, device, use_amp: bool) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for ids, lbls in val_dl:
            ids  = ids.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            if use_amp:
                # PyTorch 2.6+: use torch.amp.autocast instead of
                # the deprecated torch.cuda.amp.autocast
                with torch.amp.autocast("cuda"):
                    _, loss = model(ids, lbls)
            else:
                _, loss = model(ids, lbls)
            total += loss.item()
    model.train()
    return total / max(len(val_dl), 1)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def _run_phase(
    label: str,
    model,
    pairs: list,
    tokenizer,
    optimizer,
    scaler,
    device: str,
    use_amp: bool,
    n_epochs: int,
    base_lr: float,
    effective_batch: int,
    pin_memory: bool,
    num_workers: int,
    global_step_start: int,
    best_val: float,
) -> tuple:
    """
    Generic training phase loop.
    Returns (global_step_end, best_val_updated).
    """
    random.shuffle(pairs)
    split  = max(1, int(0.9 * len(pairs)))
    tr_dl  = _make_dataloader(pairs[:split], tokenizer, MAX_SEQ_LEN,
                               effective_batch, True,  pin_memory, num_workers)
    va_dl  = _make_dataloader(pairs[split:],  tokenizer, MAX_SEQ_LEN,
                               effective_batch, False, pin_memory, num_workers)

    total_steps = n_epochs * len(tr_dl)
    warmup      = min(200, total_steps // 10)
    step        = 0
    global_step = global_step_start

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for ids, lbls in tr_dl:
            ids  = ids.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)

            new_lr = _cosine_lr(step, warmup, total_steps, base_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast("cuda"):          # ← fixed API
                    _, loss = model(ids, lbls)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                _, loss = model(ids, lbls)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            epoch_loss += loss.item()
            step        += 1
            global_step += 1

            if step % EVAL_EVERY == 0:
                val_loss = _eval_loss(model, va_dl, device, use_amp)
                print(f"  [{label}] step {step:>5} | val_loss {val_loss:.4f}"
                      f" | lr {new_lr:.2e}{_log_gpu()}")
                if val_loss < best_val:
                    best_val = val_loss
                    _save(model, tokenizer)
                    print(f"  ✓ New best saved  (val={best_val:.4f})")

        avg     = epoch_loss / max(len(tr_dl), 1)
        elapsed = time.time() - t0
        speed   = len(tr_dl) * effective_batch / elapsed
        print(f"  {label} Epoch {epoch:>2}/{n_epochs}  "
              f"train_loss={avg:.4f}  {elapsed:.1f}s  ({speed:.0f} samples/s)")

    return global_step, best_val


def train_with_curriculum(
    csv_dir: Path = CSV_DIR,
    doc_dir: str = ".",
    epochs: int = EPOCHS,
    sales_only_fraction: float = 0.5,
    augment: bool = True,
    batch_mult: int = 4,
    use_fp16: bool = True,
    force_cpu: bool = False,
) -> None:
    torch.manual_seed(SEED)

    device, cuda_ok = ("cpu", False) if force_cpu else check_gpu_environment()

    if cuda_ok:
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed_all(SEED)
        effective_batch = BATCH_SIZE * batch_mult
        pin_memory      = True
        num_workers     = 2
        use_amp         = use_fp16
        # GradScaler: new API in PyTorch 2.6 uses torch.amp.GradScaler("cuda")
        # Fall back to old API for older builds
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        except TypeError:
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    else:
        effective_batch = BATCH_SIZE
        pin_memory      = False
        num_workers     = 0
        use_amp         = False
        scaler          = None

    print(f"\n{'═'*64}")
    print(f"  SalesGPT — Document Understanding Training  [{device.upper()}]")
    print(f"  Epochs total  : {epochs}")
    print(f"  Phase 1       : {int(epochs * sales_only_fraction)} epochs  (sales only)")
    print(f"  Phase 2       : {epochs - int(epochs * sales_only_fraction)} epochs  (sales + documents)")
    print(f"  Batch size    : {effective_batch}" +
          (f"  (base {BATCH_SIZE} × {batch_mult})" if cuda_ok else ""))
    print(f"  Mixed prec.   : {'FP16 / AMP' if use_amp else 'FP32'}")
    print(f"  pin_memory    : {pin_memory}   num_workers : {num_workers}")
    print(f"{'═'*64}\n")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────
    print("[1/5] Loading sales training data …")
    sales_pairs = load_training_pairs(csv_dir)
    print(f"  Sales pairs: {len(sales_pairs)}")

    print("\n[2/5] Generating document training data …")
    doc_pairs = generate_doc_training_pairs(
        doc_dir=doc_dir, include_synthetic=True, augment=augment
    )
    print(f"  Document pairs: {len(doc_pairs)}")

    mixed_pairs = sales_pairs + doc_pairs
    random.shuffle(mixed_pairs)
    all_corpus = [text for p, r in mixed_pairs for text in (p, r)]

    # ── Tokenizer ──────────────────────────────────────────────────────────
    print("\n[3/5] Training BPE tokenizer on combined corpus …")
    tokenizer = CharBPETokenizer()
    tokenizer.train(all_corpus, vocab_size=VOCAB_SIZE)
    tokenizer.save(VOCAB_PATH)
    print(f"  Vocabulary size: {len(tokenizer.vocab)}")

    # ── Model ──────────────────────────────────────────────────────────────
    print("\n[4/5] Initialising SalesGPT …")
    model = SalesGPT(
        vocab_size=len(tokenizer.vocab),
        d_model=D_MODEL, n_heads=N_HEADS, n_kv_heads=N_KV_HEADS,
        n_layers=N_LAYERS, d_ff=D_FF,
        max_seq=MAX_SEQ_LEN, dropout=DROPOUT,
    ).to(device)
    print(f"  Parameters: {model.n_params:,}  (~{model.n_params/1e6:.2f}M)")
    if cuda_ok:
        alloc = torch.cuda.memory_allocated(0) / 1024**2
        resv  = torch.cuda.memory_reserved(0)  / 1024**2
        print(f"  After model load — GPU mem: {alloc:.0f} MB alloc / {resv:.0f} MB reserved")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95)
    )

    # ── Train ──────────────────────────────────────────────────────────────
    print(f"\n[5/5] Training …")
    phase1_epochs = max(1, int(epochs * sales_only_fraction))
    phase2_epochs = epochs - phase1_epochs
    best_val      = float("inf")
    global_step   = 0

    if phase1_epochs > 0:
        print(f"\n  ── Phase 1: Sales data only ({phase1_epochs} epochs) ──")
        global_step, best_val = _run_phase(
            "P1", model, list(sales_pairs), tokenizer, optimizer, scaler,
            device, use_amp, phase1_epochs, LR,
            effective_batch, pin_memory, num_workers,
            global_step, best_val,
        )

    if phase2_epochs > 0:
        print(f"\n  ── Phase 2: Sales + documents ({phase2_epochs} epochs) ──")
        global_step, best_val = _run_phase(
            "P2", model, list(mixed_pairs), tokenizer, optimizer, scaler,
            device, use_amp, phase2_epochs, LR * 0.5,   # half LR for phase 2
            effective_batch, pin_memory, num_workers,
            global_step, best_val,
        )

    # ── Final save ─────────────────────────────────────────────────────────
    _save(model, tokenizer)
    print(f"\n{'═'*64}")
    print(f"  Training complete.  Best val_loss : {best_val:.4f}")
    print(f"  Model → {MODEL_PATH}")
    print(f"  Vocab → {VOCAB_PATH}")
    if cuda_ok:
        alloc = torch.cuda.memory_allocated(0) / 1024**2
        print(f"  Final GPU mem: {alloc:.0f} MB")
    print(f"{'═'*64}\n")

    _post_training_eval()


def _post_training_eval() -> None:
    print("\n── Post-training sanity check ──")
    llm = ScratchLLM()
    if not llm.load():
        print("  [WARN] Could not load model.")
        return
    tests = [
        ("Sales Q&A",        "What is the status of enquiry ENQ001?"),
        ("Doc Summary",      "<TASK=DOCUMENT_SUMMARY>\nDocument: paper.pdf\nType: research_paper\n\nQuestion: Summarize this document."),
        ("Short Summary",    "<TASK=SHORT_SUMMARY>\nDocument: report.pdf\n\nQuestion: Give me a 5-line summary."),
        ("Document Type",    "<TASK=DOCUMENT_TYPE>\nDocument: invoice.pdf\n\nQuestion: What type of document is this?"),
    ]
    for label, prompt in tests:
        out = llm.generate_for_document(prompt, max_new=80, temperature=0.5)
        print(f"\n  [{label}]")
        print(f"  Prompt : {prompt[:80]}…")
        print(f"  Output : {out or '(none — DocumentAnswerEngine handles this)'}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train SalesGPT with GPU + document understanding."
    )
    parser.add_argument("--epochs",      type=int,   default=EPOCHS)
    parser.add_argument("--csv-dir",     type=str,   default=str(CSV_DIR))
    parser.add_argument("--doc-dir",     type=str,   default=".")
    parser.add_argument("--phase1-frac", type=float, default=0.5,
                        help="Fraction of epochs for Phase 1 / sales only")
    parser.add_argument("--batch-mult",  type=int,   default=4,
                        help="GPU batch multiplier: effective = BATCH_SIZE × N (default 4)")
    parser.add_argument("--fp32",        action="store_true",
                        help="Disable FP16 mixed precision")
    parser.add_argument("--cpu",         action="store_true",
                        help="Force CPU (debug only)")
    parser.add_argument("--no-aug",      action="store_true",
                        help="Disable data augmentation")
    args = parser.parse_args()

    train_with_curriculum(
        csv_dir=Path(args.csv_dir),
        doc_dir=args.doc_dir,
        epochs=args.epochs,
        sales_only_fraction=args.phase1_frac,
        augment=not args.no_aug,
        batch_mult=args.batch_mult,
        use_fp16=not args.fp32,
        force_cpu=args.cpu,
    )


if __name__ == "__main__":
    main()
