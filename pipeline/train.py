"""
pipeline/train.py — Fine-tune T5-small for complaint classification

WHAT THIS TRAINS:
  Input : "Classify complaint: <complaint text>"
  Output: one of: fraud, account, payment, credit_card, loan, other

  This is a text-to-text classification task using T5's seq2seq architecture.
  T5 learns to map the complaint prefix → the correct label token.

PROMPT ENGINEERING FOR TRAINING:
  The training input format is FIXED as "Classify complaint: <text>"
  This exact prefix must also be used at inference time (predict.py).
  Any change to this prefix = model sees out-of-distribution input = wrong predictions.

OUTPUT LABEL FORMAT:
  Labels are lowercase with underscore: credit_card (not "Credit Card")
  predict.py normalizes these back to Title Case via T5_LABEL_MAP.
  Do NOT change the label format in training data — it must match T5_LABEL_MAP keys.

Usage:
    python pipeline/train.py
    python pipeline/train.py --epochs 5
    python pipeline/train.py --data data/training_dataset.csv --epochs 3
"""

import argparse, os, torch
from datasets import load_dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
)

# ── Config ────────────────────────────────────────────────────────────
MODEL_NAME   = "t5-small"
OUTPUT_DIR   = "./models/support_llm"
DEFAULT_DATA = "./data/training_dataset.csv"

MAX_INPUT_LEN  = 128   # avg training input = ~58 tokens; 128 gives safe headroom
MAX_LABEL_LEN  = 8     # longest label "credit_card" = ~4 tokens; 8 is safe


def tokenize_batch(examples, tokenizer):
    """
    Tokenize input/output pairs for seq2seq training.

    KEY DECISIONS:
    1. padding="max_length" for consistent batch shapes
    2. Labels use -100 for padding (standard seq2seq — ignored in loss)
    3. Input truncated at MAX_INPUT_LEN (matches training data distribution)
    4. Label truncated at MAX_LABEL_LEN (labels are short, 4-8 tokens max)
    """
    model_inputs = tokenizer(
        examples["input"],
        max_length=MAX_INPUT_LEN,
        padding="max_length",
        truncation=True,
    )

    # FIX: as_target_tokenizer() was removed in transformers >= 4.30
    # Use text_target= parameter directly instead — identical behaviour
    labels = tokenizer(
        text_target=examples["output"],
        max_length=MAX_LABEL_LEN,
        padding="max_length",
        truncation=True,
    )

    # Replace padding token id with -100 so loss ignores padding positions
    label_ids = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels["input_ids"]
    ]

    model_inputs["labels"] = label_ids
    return model_inputs


def main(data_path: str, epochs: int):

    print(f"\n{'='*60}")
    print(f"T5-small Fine-tuning for Complaint Classification")
    print(f"{'='*60}")
    print(f"  Base model  : {MODEL_NAME}")
    print(f"  Data        : {data_path}")
    print(f"  Epochs      : {epochs}")
    print(f"  Output      : {OUTPUT_DIR}")
    print(f"  Device      : {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*60}\n")

    # ── Load model ────────────────────────────────────────────────────
    print("Loading T5-small...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model     = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    # ── Load and split dataset ────────────────────────────────────────
    print(f"Loading dataset: {data_path}")
    raw = load_dataset("csv", data_files={"train": data_path})

    # 90/10 train/val split — stratification not needed (balanced dataset)
    split    = raw["train"].train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    val_ds   = split["test"]

    print(f"  Train: {len(train_ds)} rows")
    print(f"  Val  : {len(val_ds)} rows")
    print(f"  Label distribution (train):")
    from collections import Counter
    labels = [r["output"] for r in train_ds]
    for lbl, cnt in sorted(Counter(labels).items()):
        print(f"    {lbl:<15} {cnt}")

    # ── Tokenize ──────────────────────────────────────────────────────
    print("\nTokenizing...")
    tok_fn = lambda ex: tokenize_batch(ex, tokenizer)
    cols   = train_ds.column_names

    train_tok = train_ds.map(tok_fn, batched=True, batch_size=64,
                              remove_columns=cols)
    val_tok   = val_ds.map(tok_fn,   batched=True, batch_size=64,
                              remove_columns=cols)

    # ── Training arguments ────────────────────────────────────────────
    use_fp16 = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        # Batch + compute
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        fp16=use_fp16,
        dataloader_num_workers=2,

        # Learning schedule
        num_train_epochs=epochs,
        learning_rate=5e-4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="cosine",

        # Evaluation + checkpointing
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Logging
        logging_steps=50,
        report_to="none",
    )

    # ── Data collator ─────────────────────────────────────────────────
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        pad_to_multiple_of=8 if use_fp16 else None,
    )

    # ── Trainer ───────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ── Train ─────────────────────────────────────────────────────────
    print(f"\nStarting training on {'GPU' if use_fp16 else 'CPU'}...")
    print("EarlyStopping active — stops if val_loss does not improve for 2 epochs\n")

    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\n{'='*60}")
    print(f"✅ Training complete")
    print(f"   Model saved → {OUTPUT_DIR}")
    print(f"   Restart Streamlit to load the new model")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fine-tune T5-small for complaint classification")
    ap.add_argument("--data",   default=DEFAULT_DATA, help="Path to training CSV")
    ap.add_argument("--epochs", type=int, default=3,  help="Training epochs (default: 3)")
    a = ap.parse_args()
    main(a.data, a.epochs)
