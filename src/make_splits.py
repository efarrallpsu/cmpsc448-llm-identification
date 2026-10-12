"""Select models, balance classes, and build train/val/test splits.

Labels are LLM families, one representative model each:
  GPT     -> gpt-4.1-2025-04-14
  Claude  -> claude-sonnet-4-20250514
  Gemini  -> gemini-2.5-pro
  Qwen    -> qwen3-235b-a22b-no-thinking
  DeepSeek-> deepseek-v3-0324
  Llama   -> llama-4-maverick-03-26-experimental

Splits are grouped by normalized prompt (md5 hash) so that the same prompt
never appears in two different splits (no leakage, important for RQ2).
"""

import hashlib
import json
import re

import pandas as pd

IN_PATH = "data/processed/samples_all.parquet"
OUT_DIR = "data/processed"
SEED = 448

MODEL_TO_LABEL = {
    "gpt-4.1-2025-04-14": "GPT",
    "claude-sonnet-4-20250514": "Claude",
    "gemini-2.5-pro": "Gemini",
    "qwen3-235b-a22b-no-thinking": "Qwen",
    "deepseek-v3-0324": "DeepSeek",
    "llama-4-maverick-03-26-experimental": "Llama",
}

CAP_TRAIN = 1050
CAP_VAL = 225
CAP_TEST = 225

FEATURE_COLS = [
    "response_tokens", "prompt_tokens", "headers", "list_ordered",
    "list_unordered", "bold_marks", "turns",
]


def normalize(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def prompt_hash(text):
    return hashlib.md5(normalize(text).encode("utf-8")).hexdigest()


def assign_group_split(h, ratios=(0.70, 0.85)):
    r = int(h[:8], 16) / 0xFFFFFFFF
    if r < ratios[0]:
        return "train"
    if r < ratios[1]:
        return "val"
    return "test"


def main():
    df = pd.read_parquet(IN_PATH)
    n0 = len(df)

    df = df[df["model"].isin(MODEL_TO_LABEL)].copy()
    df["label"] = df["model"].map(MODEL_TO_LABEL)
    print(f"kept {len(df)} of {n0} samples for {df['model'].nunique()} models")

    df = df[df["prompt"].str.len() >= 5]
    df = df[df["response"].str.len() >= 100]
    print(f"after length filters: {len(df)}")

    df = df.drop_duplicates(subset=["model", "prompt", "response"])
    print(f"after exact-duplicate removal: {len(df)}")

    df["prompt_hash"] = df["prompt"].map(prompt_hash)
    df["split"] = df["prompt_hash"].map(assign_group_split)

    caps = {"train": CAP_TRAIN, "val": CAP_VAL, "test": CAP_TEST}
    parts = []
    for split, cap in caps.items():
        part = df[df["split"] == split]
        part = part.sample(frac=1.0, random_state=SEED).groupby("label").head(cap)
        parts.append(part)
        print(f"{split}: {len(part)} samples")

    out = pd.concat(parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    out["prompt_chars"] = out["prompt"].str.len()
    out["response_chars"] = out["response"].str.len()

    splits = {s: set(g["prompt_hash"]) for s, g in out.groupby("split")}
    assert not (splits["train"] & splits["val"]), "prompt leakage train/val"
    assert not (splits["train"] & splits["test"]), "prompt leakage train/test"
    assert not (splits["val"] & splits["test"]), "prompt leakage val/test"

    print("\nclass counts per split:")
    print(pd.crosstab(out["label"], out["split"]).to_string())

    out.to_parquet(f"{OUT_DIR}/dataset.parquet", index=False)
    for split in caps:
        out[out["split"] == split].to_parquet(f"{OUT_DIR}/{split}.parquet", index=False)

    counts = {
        str(label): {str(split): int(n) for split, n in row.items()}
        for label, row in pd.crosstab(out["label"], out["split"]).to_dict("index").items()
    }
    stats = {
        "n_samples": len(out),
        "labels": sorted(out["label"].unique().tolist()),
        "counts": counts,
        "median_response_chars": float(out["response_chars"].median()),
        "median_prompt_chars": float(out["prompt_chars"].median()),
    }
    with open(f"{OUT_DIR}/dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nsaved dataset.parquet ({len(out)} rows) and split files to {OUT_DIR}")


if __name__ == "__main__":
    main()
