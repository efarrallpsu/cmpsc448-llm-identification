"""Extract (model, prompt, response) samples from the LMArena 140k dataset.

Source: lmarena-ai/arena-human-preference-140k (CC-BY-4.0)
Each row of the raw data is a "battle": one prompt answered by two models.
We split it into individual samples: one per (model, response) pair.
Only English, text-only, first-turn conversations are kept.
"""

import json
import os
import urllib.request

import pandas as pd
import pyarrow.parquet as pq

RAW_DIR = "data/raw"
OUT_PATH = "data/processed/samples_all.parquet"
BASE_URL = "https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k/resolve/main/data"
SHARDS = [f"train-{i:05d}-of-00007.parquet" for i in range(7)]

COLUMNS = [
    "id", "model_a", "model_b", "conversation_a", "conversation_b",
    "conv_metadata", "category_tag", "language", "is_code", "timestamp",
]


def download_shards():
    os.makedirs(RAW_DIR, exist_ok=True)
    for shard in SHARDS:
        path = os.path.join(RAW_DIR, shard)
        if os.path.exists(path):
            continue
        print(f"downloading {shard} ...")
        urllib.request.urlretrieve(f"{BASE_URL}/{shard}", path)


def first_message_text(conversation, role):
    for msg in conversation:
        if msg["role"] != role:
            continue
        parts = []
        for block in msg.get("content", []):
            if block.get("image") is not None:
                return None
            if block.get("type") == "text" and block.get("text"):
                parts.append(block["text"])
        text = "\n".join(parts).strip()
        return text if text else None
    return None


def side_features(meta, side):
    headers = meta.get(f"header_count_{side}", {})
    lists = meta.get(f"list_count_{side}", {})
    bold = meta.get(f"bold_count_{side}", {})
    return {
        "response_tokens": meta.get(f"sum_assistant_{side}_tokens"),
        "prompt_tokens": meta.get(f"context_{side}_tokens"),
        "headers": sum(headers.values()) if headers else 0,
        "list_ordered": lists.get("ordered", 0) if lists else 0,
        "list_unordered": lists.get("unordered", 0) if lists else 0,
        "bold_marks": sum(bold.values()) if bold else 0,
        "turns": meta.get("turns"),
    }


def category_flags(cat):
    cw = cat.get("creative_writing_v0.1", {})
    math = cat.get("math_v0.1", {})
    instr = cat.get("if_v0.1", {})
    criteria = cat.get("criteria_v0.1", {})
    return {
        "creative_writing": bool(cw.get("creative_writing")),
        "math": bool(math.get("math")),
        "instruction_following": bool(instr.get("if")),
        "complexity": bool(criteria.get("complexity")),
        "creativity": bool(criteria.get("creativity")),
        "domain_knowledge": bool(criteria.get("domain_knowledge")),
        "problem_solving": bool(criteria.get("problem_solving")),
        "real_world": bool(criteria.get("real_world")),
        "specificity": bool(criteria.get("specificity")),
        "technical_accuracy": bool(criteria.get("technical_accuracy")),
    }


def main():
    download_shards()
    rows = []
    for shard in SHARDS:
        pf = pq.ParquetFile(os.path.join(RAW_DIR, shard))
        for batch in pf.iter_batches(batch_size=2048, columns=COLUMNS):
            data = batch.to_pylist()
            for r in data:
                if r["language"] != "en":
                    continue
                cat = category_flags(r["category_tag"] or {})
                for side in ("a", "b"):
                    model = r[f"model_{side}"]
                    conv = r[f"conversation_{side}"]
                    prompt = first_message_text(conv, "user")
                    response = first_message_text(conv, "assistant")
                    if not prompt or not response:
                        continue
                    row = {
                        "battle_id": r["id"],
                        "side": side,
                        "model": model,
                        "prompt": prompt,
                        "response": response,
                        "is_code": bool(r["is_code"]),
                        "timestamp": r["timestamp"],
                        **cat,
                        **side_features(r["conv_metadata"] or {}, side),
                    }
                    rows.append(row)
        print(f"{shard}: {len(rows)} samples so far")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nsaved {len(df)} samples to {OUT_PATH}")
    print(f"distinct models: {df['model'].nunique()}")
    print(df["model"].value_counts().head(20).to_string())


if __name__ == "__main__":
    main()
