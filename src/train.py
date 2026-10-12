"""Train and evaluate one (input_type, model_type) configuration.

Usage examples:
  python src/train.py --input output --model cnn
  python src/train.py --input input  --model lstm
  python src/train.py --input combined --model cnn
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from torch.utils.data import DataLoader

from models import build_model
from textdata import LLMDataset, build_vocab, collate

PROC_DIR = "data/processed"
RESULT_DIR = "results"
SEED = 448

MAX_LEN = {"input": 128, "output": 300, "combined": 320}


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def evaluate(model, loader):
    model.eval()
    preds, labels, probs = [], [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, lengths, y in loader:
            logits = model(x, lengths)
            total_loss += criterion(logits, y).item() * len(y)
            probs.append(torch.softmax(logits, dim=1).numpy())
            preds.extend(logits.argmax(dim=1).tolist())
            labels.extend(y.tolist())
    probs = np.vstack(probs)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return acc, f1, total_loss / len(loader.dataset), preds, labels, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", choices=["input", "output", "combined"], required=True)
    ap.add_argument("--model", choices=["cnn", "lstm"], required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_len", type=int, default=None)
    ap.add_argument("--data_dir", default=PROC_DIR)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    set_seed(SEED)
    max_len = args.max_len or MAX_LEN[args.input]
    os.makedirs(RESULT_DIR, exist_ok=True)
    tag = args.tag or f"{args.input}_{args.model}"

    train = pd.read_parquet(f"{args.data_dir}/train.parquet")
    val = pd.read_parquet(f"{args.data_dir}/val.parquet")
    test = pd.read_parquet(f"{args.data_dir}/test.parquet")

    labels = sorted(train["label"].unique())
    label_to_id = {lab: i for i, lab in enumerate(labels)}

    train_texts = (train["prompt"] + " \n " + train["response"]).tolist()
    vocab = build_vocab(train_texts)
    print(f"vocab size: {len(vocab)}, max_len: {max_len}, labels: {labels}")

    loaders = {}
    for name, df in [("train", train), ("val", val), ("test", test)]:
        ds = LLMDataset(df, args.input, vocab, max_len, label_to_id)
        loaders[name] = DataLoader(
            ds, batch_size=args.batch_size, shuffle=(name == "train"),
            collate_fn=collate,
        )

    model = build_model(args.model, len(vocab), len(labels))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {args.model}, parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_f1, best_epoch, best_state = -1.0, -1, None
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, lengths, y in loaders["train"]:
            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * len(y)
        train_loss = running / len(loaders["train"].dataset)
        val_acc, val_f1, val_loss, _, _, _ = evaluate(model, loaders["val"])
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
        })
        print(f"epoch {epoch}: train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= args.patience:
            print(f"early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    acc, f1, loss, preds, gold, probs = evaluate(model, loaders["test"])
    report = classification_report(gold, preds, target_names=labels, digits=4)
    cm = confusion_matrix(gold, preds).tolist()
    print(f"\ntest acc={acc:.4f} macro_f1={f1:.4f} (best epoch {best_epoch})\n")
    print(report)

    pred_df = test.reset_index(drop=True).copy()
    pred_df["pred_id"] = preds
    pred_df["pred_label"] = [labels[p] for p in preds]
    for i, lab in enumerate(labels):
        pred_df[f"prob_{lab}"] = probs[:, i]
    pred_df.to_parquet(f"{RESULT_DIR}/preds_{tag}.parquet", index=False)

    torch.save(best_state, f"{RESULT_DIR}/model_{tag}.pt")
    result = {
        "tag": tag,
        "input_type": args.input,
        "model_type": args.model,
        "n_params": n_params,
        "vocab_size": len(vocab),
        "max_len": max_len,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": SEED,
        "labels": labels,
        "best_epoch": best_epoch,
        "history": history,
        "test": {
            "accuracy": acc,
            "macro_f1": f1,
            "loss": loss,
            "confusion_matrix": cm,
            "report": report,
        },
        "wall_time_sec": round(time.time() - start, 1),
    }
    with open(f"{RESULT_DIR}/result_{tag}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved {RESULT_DIR}/result_{tag}.json")


if __name__ == "__main__":
    main()
