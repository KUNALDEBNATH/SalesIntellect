from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
import pandas as pd

# ── Import everything from scratch_llm ──────────────────────────────────────
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
)
from torch.utils.data import DataLoader


# ════════════════════════════════ DATA AUGMENTATION ═══════════════════════════

def _jitter_query(prompt: str) -> str:
    """
    Randomly rephrase a prompt so the model sees multiple ways of asking
    the same thing — helps generalisation with such a small corpus.
    """
    rewrites = [
        ("What is the status of", "Tell me the status of"),
        ("What is the status of", "Status for"),
        ("Show details for",      "Give me the profile of"),
        ("Show details for",      "Full details on"),
        ("Has",                   "Did"),
        ("What vehicle did",      "Which car did"),
        ("What vehicle did",      "What model did"),
        ("What is",               "What's"),
        ("What feedback did",     "What review did"),
        ("What feedback did",     "What did"),
        ("What is the rating for","Rating for"),
        ("How satisfied was",     "Was"),
    ]
    for src, dst in rewrites:
        if prompt.startswith(src):
            return dst + prompt[len(src):]
    return prompt


def _jitter_response(response: str) -> str:
    """Minor word-level synonym swap to diversify response phrasing."""
    swaps = [
        ("is interested in",  "is enquiring about"),
        ("is enquiring about","enquired about"),
        ("prefers",           "has chosen"),
        ("has taken",         "completed"),
        ("has not taken",     "has not completed"),
        ("gave",              "submitted"),
        ("received",          "got"),
        ("on",                "dated"),
    ]
    for src, dst in swaps:
        if src in response:
            return response.replace(src, dst, 1)
    return response


def augment_pairs(pairs: List[Tuple[str, str]],
                  factor: int = 2) -> List[Tuple[str, str]]:
    """Return pairs + `factor` augmented variants of each pair."""
    augmented = list(pairs)
    for prompt, response in pairs:
        for _ in range(factor):
            p = _jitter_query(prompt)
            r = _jitter_response(response)
            # Only add if something actually changed
            if p != prompt or r != response:
                augmented.append((p, r))
    random.shuffle(augmented)
    return augmented


# ════════════════════════════════ EXTENDED DATA LOADING ═══════════════════════

def _safe_read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  [Data] NOT FOUND: {path}")
        return None
    df = pd.read_csv(path).fillna("N/A")
    df.columns = [c.strip() for c in df.columns]
    return df


def load_training_pairs_extended(csv_dir: Path = CSV_DIR,
                                  augment: bool = True) -> List[Tuple[str, str]]:
    """
    Builds a richer set of (prompt, response) pairs than scratch_llm.py's
    basic version — adds cross-table JOIN-style pairs (e.g. "show everything
    about ENQ001" pulling from all three CSVs), list-type queries ("who gave
    bad feedback?"), and boolean questions ("has Arjun taken a test ride?").
    """
    pairs: List[Tuple[str, str]] = []

    enq_df  = _safe_read(csv_dir / "sales_enquiry_dataset.csv")
    appt_df = _safe_read(csv_dir / "sales_appointment_dataset.csv")
    fb_df   = _safe_read(csv_dir / "sales_feedback_dataset.csv")

    # ── Per-row pairs ─────────────────────────────────────────────────────────
    if enq_df is not None:
        for _, r in enq_df.iterrows():
            eid    = r.get("ENQUIRY ID", "?")
            name   = r.get("Customer Name", "?")
            phone  = r.get("Phone Number", "?")
            email  = r.get("Email", "?")
            gender = r.get("Gender", "?")
            veh    = r.get("Vehicle Name / Model", "?")
            src    = r.get("Enquiry Source", "?")
            edate  = r.get("Enquiry Date", "?")
            adate  = r.get("Appointment Date", "?")
            city   = r.get("City / State", "?")
            ctype  = r.get("Customer Type", "?")
            pay    = r.get("Payment Type", "?")
            ride   = r.get("Test Ride Taken", "?")
            status = r.get("Status", "?")
            ride_s = "has taken" if str(ride).strip() == "Yes" else "has not taken"

            pairs += [
                (f"What is the status of enquiry {eid}?",
                 f"Enquiry {eid} for {name} ({veh}) has status '{status}'. "
                 f"Received {edate} via {src}. Appointment: {adate}. Payment: {pay}."),

                (f"Show full details for {name}.",
                 f"{name} ({gender}, {ctype}) from {city} enquired about the {veh} "
                 f"on {edate} via {src}. Phone: {phone}. Email: {email}. "
                 f"Payment: {pay}. Status: {status}."),

                (f"Has {name} taken a test ride?",
                 f"{name} {ride_s} a test ride for the {veh} (enquiry {eid})."),

                (f"What vehicle is {name} interested in?",
                 f"{name} (ID: {eid}) is interested in the {veh}."),

                (f"What is {name}'s preferred payment method?",
                 f"{name} prefers {pay} payment for their {veh} enquiry."),

                (f"What is the contact information for {name}?",
                 f"{name}'s phone number is {phone} and email is {email}."),

                (f"Is {name} a new or returning customer?",
                 f"{name} is a {ctype} customer from {city}."),

                (f"When did {name} enquire?",
                 f"{name} submitted their enquiry on {edate} via {src}."),
            ]

    if appt_df is not None:
        for _, r in appt_df.iterrows():
            eid    = r.get("Enquiry ID", "?")
            name   = r.get("Customer Name", "?")
            adate  = r.get("Appointment Date", "?")
            atime  = r.get("Time", "?")
            veh    = r.get("Vehicle", "?")
            status = r.get("Status", "?")
            followup = {
                "Scheduled": "Please arrive 10 minutes early.",
                "Completed": "The appointment was completed successfully.",
                "Cancelled": "The appointment was cancelled. Please call to reschedule.",
            }.get(str(status), "Contact the dealership for details.")

            pairs += [
                (f"What is the appointment status for {name}?",
                 f"{name}'s appointment (Enquiry {eid}) for the {veh} is {status}. "
                 f"Date: {adate} at {atime}. {followup}"),

                (f"When is {name}'s next appointment?",
                 f"{name}'s appointment is on {adate} at {atime} for the {veh}. "
                 f"Current status: {status}."),

                (f"Is {name}'s appointment confirmed?",
                 f"{name}'s appointment status is '{status}' — scheduled for {adate} "
                 f"at {atime}. Vehicle: {veh}. {followup}"),

                (f"What vehicle is booked for {name}'s appointment?",
                 f"{name}'s appointment is for the {veh} on {adate} at {atime}. "
                 f"Status: {status}."),
            ]

    if fb_df is not None:
        for _, r in fb_df.iterrows():
            eid      = r.get("Enquiry ID", "?")
            name     = r.get("Customer Name", "?")
            feedback = r.get("Feedback", "?")
            rating   = r.get("Rating", "?")
            date     = r.get("Date", "?")
            try:
                ri   = int(float(rating))
                sent = ("very positive" if ri >= 4 else
                        "neutral"       if ri == 3 else
                        "negative")
            except Exception:
                sent = "unknown"

            pairs += [
                (f"What feedback did {name} give?",
                 f"{name} (Enquiry {eid}) rated the service {rating}/5 on {date}. "
                 f"Comment: \"{feedback}\". Overall sentiment: {sent}."),

                (f"What is the rating for enquiry {eid}?",
                 f"Enquiry {eid} received a rating of {rating}/5. "
                 f"{name} commented: \"{feedback}\" on {date}."),

                (f"How satisfied was {name}?",
                 f"{name} gave {rating}/5 ({sent}) and wrote: \"{feedback}\"."),

                (f"Was {name} happy with the service?",
                 f"{name} rated the experience {rating}/5 — {sent}. "
                 f"Their comment was: \"{feedback}\"."),
            ]

    # ── Aggregate / list-type pairs ───────────────────────────────────────────
    if fb_df is not None:
        bad_names  = [str(r.get("Customer Name","?")) for _, r in fb_df.iterrows()
                      if _try_int(r.get("Rating")) is not None
                      and _try_int(r.get("Rating")) <= 2]
        good_names = [str(r.get("Customer Name","?")) for _, r in fb_df.iterrows()
                      if _try_int(r.get("Rating")) is not None
                      and _try_int(r.get("Rating")) >= 4]

        if bad_names:
            sample = ", ".join(bad_names[:3])
            total  = len(bad_names)
            pairs.append((
                "Who gave bad feedback?",
                f"{total} customer(s) gave a rating of 2 or below. "
                f"Examples: {sample}{'.' if total <= 3 else ', and more.'}"
            ))
            pairs.append((
                "Show customers with low ratings.",
                f"There are {total} customer(s) with low ratings (≤2/5): "
                f"{', '.join(bad_names[:5])}."
            ))

        if good_names:
            sample = ", ".join(good_names[:3])
            total  = len(good_names)
            pairs.append((
                "Who gave excellent feedback?",
                f"{total} customer(s) gave a rating of 4 or above. "
                f"Examples: {sample}."
            ))

    if appt_df is not None:
        cancelled = [str(r.get("Customer Name","?")) for _, r in appt_df.iterrows()
                     if "cancel" in str(r.get("Status","")).lower()]
        completed = [str(r.get("Customer Name","?")) for _, r in appt_df.iterrows()
                     if "complet" in str(r.get("Status","")).lower()]

        if cancelled:
            pairs.append((
                "Show all cancelled appointments.",
                f"There are {len(cancelled)} cancelled appointment(s): "
                f"{', '.join(cancelled[:5])}."
            ))
        if completed:
            pairs.append((
                "Show all completed appointments.",
                f"There are {len(completed)} completed appointment(s): "
                f"{', '.join(completed[:5])}."
            ))

    if enq_df is not None:
        cities = enq_df.get("City / State", pd.Series(dtype=str)).dropna().unique()
        for city in cities:
            city_customers = [
                str(r.get("Customer Name","?"))
                for _, r in enq_df.iterrows()
                if str(r.get("City / State","")).strip() == city
            ]
            if len(city_customers) >= 1:
                pairs.append((
                    f"Show customers from {city}.",
                    f"There are {len(city_customers)} customer(s) from {city}: "
                    f"{', '.join(city_customers[:4])}."
                ))

    print(f"  [Data] Base pairs: {len(pairs)}")

    if augment:
        pairs = augment_pairs(pairs, factor=2)
        print(f"  [Data] After augmentation: {len(pairs)} pairs")

    return pairs


def _try_int(val) -> int | None:
    try:
        return int(float(val))
    except Exception:
        return None


# ══════════════════════════════════════ EVALUATION ════════════════════════════

def evaluate_model(llm: ScratchLLM, sample_prompts: List[str]) -> None:
    """Run a few sample prompts and print the model's responses."""
    print("\n" + "═"*60)
    print("  SAMPLE GENERATIONS (scratch model)")
    print("═"*60)
    for prompt in sample_prompts:
        response = llm.generate(prompt, max_new=60, temperature=0.7, top_p=0.9)
        print(f"\nQ: {prompt}")
        print(f"A: {response or '(no output)'}")
    print("\n" + "═"*60 + "\n")


# ══════════════════════════════════════ MAIN TRAINING ═════════════════════════

def train(csv_dir: Path = CSV_DIR,
          epochs:  int  = EPOCHS,
          augment: bool = True) -> None:

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    else:
        gpu_name = None

    print(f"\n{'═'*60}")
    print(f"  SalesGPT — Training from scratch (zero pretrained weights)")
    print(f"  Device : {device.upper()}" + (f"  ({gpu_name}, {total_mem:.1f} GB)" if gpu_name else ""))
    print(f"  Epochs : {epochs}")
    print(f"{'═'*60}\n")

    if device == "cpu":
        print("  [WARN] CUDA not detected — training will run on CPU and be slow.")
        print("  [WARN] If you have an NVIDIA GPU, reinstall PyTorch with CUDA support:")
        print("  [WARN]   pip uninstall torch torchvision torchaudio -y")
        print("  [WARN]   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124\n")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("[1/4] Loading & augmenting training data …")
    pairs  = load_training_pairs_extended(csv_dir, augment=augment)
    corpus = [p for pair in pairs for p in pair]   # flatten for tokenizer training

    # ── 2. Tokenizer ──────────────────────────────────────────────────────────
    print("\n[2/4] Training BPE tokenizer on corpus …")
    tokenizer = CharBPETokenizer()
    tokenizer.train(corpus, vocab_size=VOCAB_SIZE)
    tokenizer.save(VOCAB_PATH)

    # ── 3. Model ──────────────────────────────────────────────────────────────
    print("\n[3/4] Initialising SalesGPT (all weights from random init) …")
    model = SalesGPT(
        vocab_size = len(tokenizer.vocab),
        d_model    = D_MODEL,
        n_heads    = N_HEADS,
        n_layers   = N_LAYERS,
        d_ff       = D_FF,
        max_seq    = MAX_SEQ_LEN,
        dropout    = DROPOUT,
    ).to(device)
    print(f"  Parameters : {model.n_params:,}  (~{model.n_params/1e6:.2f} M)")
    print(f"  Vocab size : {len(tokenizer.vocab)}")
    print(f"  Max seq    : {MAX_SEQ_LEN} tokens")
    print(f"  Architecture: {N_LAYERS} × [{N_HEADS}-head attention | FFN d={D_FF}]")

    # ── 4. Training loop ──────────────────────────────────────────────────────
    print(f"\n[4/4] Training ({epochs} epochs, batch={BATCH_SIZE}, lr={LR}) …\n")

    random.shuffle(pairs)
    split  = max(1, int(0.9 * len(pairs)))
    tr_ds  = TextDataset(pairs[:split], tokenizer, MAX_SEQ_LEN)
    va_ds  = TextDataset(pairs[split:], tokenizer, MAX_SEQ_LEN)
    tr_dl  = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    va_dl  = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    print(f"  Train samples : {len(tr_ds)}")
    print(f"  Val   samples : {len(va_ds)}\n")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95)
    )
    total_steps = epochs * len(tr_dl)
    warmup      = min(200, total_steps // 10)

    def lr_lambda(step: int) -> float:
        """Linear warmup then cosine decay."""
        import math
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler    = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_val     = float("inf")
    global_step  = 0
    train_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0         = time.time()

        for batch_ids, batch_lbls in tr_dl:
            batch_ids  = batch_ids.to(device)
            batch_lbls = batch_lbls.to(device)

            optimizer.zero_grad()
            _, loss = model(batch_ids, batch_lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            epoch_loss  += loss.item()
            global_step += 1

            if global_step % EVAL_EVERY == 0:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for v_ids, v_lbls in va_dl:
                        _, vl = model(v_ids.to(device), v_lbls.to(device))
                        val_loss += vl.item()
                val_loss /= max(len(va_dl), 1)
                cur_lr = scheduler.get_last_lr()[0] * LR
                print(f"  step {global_step:>5} | val_loss {val_loss:.4f} | "
                      f"lr {cur_lr:.2e}")
                if val_loss < best_val:
                    best_val = val_loss
                    _save_model(model, tokenizer)
                    print(f"  ✓ New best model saved (val_loss={best_val:.4f})")
                model.train()

        avg = epoch_loss / max(len(tr_dl), 1)
        train_losses.append(avg)
        print(f"  Epoch {epoch:>3}/{epochs}  "
              f"train_loss={avg:.4f}  "
              f"time={time.time()-t0:.1f}s")

    # ── Final save if val-loop didn't get the last epoch ──────────────────────
    _save_model(model, tokenizer)

    print(f"\n{'═'*60}")
    print(f"  Training complete!")
    print(f"  Best val_loss : {best_val:.4f}")
    print(f"  Model saved   : {MODEL_PATH}")
    print(f"  Vocab saved   : {VOCAB_PATH}")
    print(f"{'═'*60}\n")


# ════════════════════════════════════════ CLI ══════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train SalesGPT from scratch — no pretrained weights."
    )
    parser.add_argument("--epochs",  type=int,  default=EPOCHS,
                        help=f"Training epochs (default {EPOCHS})")
    parser.add_argument("--no-aug",  action="store_true",
                        help="Disable data augmentation")
    parser.add_argument("--eval",    action="store_true",
                        help="Run sample evaluation after training")
    parser.add_argument("--csv-dir", type=str, default=str(CSV_DIR),
                        help="Directory containing the sales CSV files")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    train(csv_dir=csv_dir, epochs=args.epochs, augment=not args.no_aug)

    if args.eval:
        print("Running post-training evaluation …")
        llm = ScratchLLM()
        if llm.load():
            evaluate_model(llm, [
                "What is the status of enquiry ENQ001?",
                "Who gave bad feedback?",
                "Show all cancelled appointments.",
                "Has the customer taken a test ride?",
                "What vehicle is Arjun interested in?",
            ])
        else:
            print("  [WARN] Could not load model for evaluation.")


if __name__ == "__main__":
    main()
