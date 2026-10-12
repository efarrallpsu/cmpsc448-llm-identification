"""RQ4: what distinguishes the LLMs? Structural / lexical feature analysis.

1. Per-class feature summary table (for the report).
2. Classifiers on non-neural features:
   - length only
   - structural formatting features (from conv_metadata)
   - structural + lexical features
   These quantify how far simple signals get and which signals matter.
"""

import json
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler

WORD_RE = re.compile(r"[a-z0-9']+")


def lexical_features(text, max_tokens=200):
    words = WORD_RE.findall(text.lower())
    window = words[:max_tokens]
    ttr = len(set(window)) / max(len(window), 1)
    avg_len = np.mean([len(w) for w in words]) if words else 0.0
    chars = max(len(text), 1)
    return {
        "ttr_200": ttr,
        "avg_word_len": float(avg_len),
        "newlines_per_1k": text.count("\n") / chars * 1000,
        "exclam_per_1k": text.count("!") / chars * 1000,
        "questions_per_1k": text.count("?") / chars * 1000,
        "uppercase_per_1k": sum(c.isupper() for c in text) / chars * 1000,
    }


def load_with_features():
    df = pd.read_parquet("data/processed/dataset.parquet").reset_index(drop=True)
    df["response_words"] = df["response"].str.split().str.len()
    df["log_response_words"] = np.log1p(df["response_words"])
    lex = pd.DataFrame([lexical_features(t) for t in df["response"]])
    return pd.concat([df, lex], axis=1)


def fit_eval(df, features, name, results):
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(train[features].fillna(0))
    Xte = scaler.transform(test[features].fillna(0))
    clf = LogisticRegression(max_iter=2000, random_state=448)
    clf.fit(Xtr, train["label"])
    pred = clf.predict(Xte)
    acc = accuracy_score(test["label"], pred)
    f1 = f1_score(test["label"], pred, average="macro")
    report = classification_report(test["label"], pred, digits=4)
    print(f"\n== {name} ==\nfeatures: {features}\nacc={acc:.4f} macro_f1={f1:.4f}")
    print(report)
    results[name] = {
        "features": features,
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "report": report,
    }


def main():
    df = load_with_features()
    train = df[df["split"] == "train"]

    key_features = [
        "response_words", "response_tokens", "headers", "list_ordered",
        "list_unordered", "bold_marks", "ttr_200", "avg_word_len",
        "newlines_per_1k", "exclam_per_1k", "questions_per_1k",
        "uppercase_per_1k",
    ]
    table = train.groupby("label")[key_features].median().round(2)
    print("== median feature values (train) ==")
    print(table.to_string())
    table.to_csv("results/rq4_feature_table.csv")

    results = {}
    fit_eval(df, ["log_response_words"], "length_only", results)
    structural = ["log_response_words", "response_tokens", "headers",
                  "list_ordered", "list_unordered", "bold_marks"]
    fit_eval(df, structural, "structural", results)
    fit_eval(df, structural + ["ttr_200", "avg_word_len", "newlines_per_1k",
                               "exclam_per_1k", "questions_per_1k",
                               "uppercase_per_1k"], "structural_lexical", results)

    with open("results/rq4_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved results/rq4_results.json and results/rq4_feature_table.csv")


if __name__ == "__main__":
    main()
