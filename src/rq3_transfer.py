"""RQ3: build domain-transfer subsets from the full sample pool.

Domains: code (is_code), math, creative writing, general (none of those).
All subsets reuse ONE global prompt-group split, so no prompt is ever
shared between a training and a test set, within or across domains.
"""

import os

import pandas as pd

from make_splits import MODEL_TO_LABEL, prompt_hash, assign_group_split

IN_PATH = "data/processed/samples_all.parquet"
OUT_ROOT = "data/processed/rq3"
SEED = 448

CAP_TRAIN = 400
CAP_VAL = 75
CAP_TEST = 60


def load_pool():
    df = pd.read_parquet(IN_PATH)
    df = df[df["model"].isin(MODEL_TO_LABEL)].copy()
    df["label"] = df["model"].map(MODEL_TO_LABEL)
    df = df[df["prompt"].str.len() >= 5]
    df = df[df["response"].str.len() >= 100]
    df = df.drop_duplicates(subset=["model", "prompt", "response"])
    df["prompt_hash"] = df["prompt"].map(prompt_hash)
    df["split"] = df["prompt_hash"].map(assign_group_split)
    return df


def domain_mask(df, name):
    if name == "code":
        return df["is_code"]
    if name == "math":
        return df["math"]
    if name == "creative":
        return df["creative_writing"]
    if name == "general":
        return ~df["code_domain"] & ~df["math"] & ~df["creative_writing"]
    if name == "noncreative":
        return ~df["creative_writing"]
    raise ValueError(name)


def capped(df, split, cap):
    part = df[df["split"] == split]
    part = part.sample(frac=1.0, random_state=SEED).groupby("label").head(cap)
    return part.reset_index(drop=True)


def main():
    df = load_pool()
    df["code_domain"] = df["is_code"]

    domains = {name: df[domain_mask(df, name)] for name in
               ["code", "math", "creative", "general", "noncreative"]}
    for name, sub in domains.items():
        print(f"{name}: {len(sub)} samples")

    splits = {name: {s: capped(sub, s, c) for s, c in
                     [("train", CAP_TRAIN), ("val", CAP_VAL), ("test", CAP_TEST)]}
              for name, sub in domains.items()}
    for name, sl in splits.items():
        print(f"{name}: train={len(sl['train'])} val={len(sl['val'])} test={len(sl['test'])}")

    experiments = {
        "in_code": ("code", "code"),
        "in_math": ("math", "math"),
        "in_creative": ("creative", "creative"),
        "in_general": ("general", "general"),
        "noncreative_to_creative": ("noncreative", "creative"),
        "code_to_math": ("code", "math"),
        "general_to_code": ("general", "code"),
    }
    for exp, (train_dom, test_dom) in experiments.items():
        d = os.path.join(OUT_ROOT, exp)
        os.makedirs(d, exist_ok=True)
        splits[train_dom]["train"].to_parquet(f"{d}/train.parquet", index=False)
        splits[train_dom]["val"].to_parquet(f"{d}/val.parquet", index=False)
        splits[test_dom]["test"].to_parquet(f"{d}/test.parquet", index=False)
        print(f"wrote {d}: train={train_dom} test={test_dom}")

    hashes = {s: set(df[df["split"] == s]["prompt_hash"]) for s in ("train", "val", "test")}
    assert not (hashes["train"] & hashes["val"])
    assert not (hashes["train"] & hashes["test"])
    assert not (hashes["val"] & hashes["test"])
    print("global prompt groups are disjoint across splits (no leakage)")


if __name__ == "__main__":
    main()
