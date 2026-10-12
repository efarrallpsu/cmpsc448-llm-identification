"""Classical baseline: TF-IDF (word 1-2 grams) + logistic regression.

Runs on the three input types using the exact same split files, giving a
non-neural reference point for RQ1/RQ2.
"""

import json
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

PROC_DIR = "data/processed"
RESULT_DIR = "results"


def combine_text(df):
    return df["prompt"] + " \n " + df["response"]


def run(input_type):
    train = pd.read_parquet(f"{PROC_DIR}/train.parquet")
    val = pd.read_parquet(f"{PROC_DIR}/val.parquet")
    test = pd.read_parquet(f"{PROC_DIR}/test.parquet")
    if input_type == "input":
        tr, va, te = train["prompt"], val["prompt"], test["prompt"]
    elif input_type == "output":
        tr, va, te = train["response"], val["response"], test["response"]
    else:
        tr, va, te = combine_text(train), combine_text(val), combine_text(test)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                 max_features=50000, sublinear_tf=True)
    Xtr = vectorizer.fit_transform(tr)
    Xva = vectorizer.transform(va)
    Xte = vectorizer.transform(te)

    clf = LogisticRegression(max_iter=3000, C=4.0, random_state=448)
    clf.fit(Xtr, train["label"])

    val_pred = clf.predict(Xva)
    val_acc = accuracy_score(val["label"], val_pred)
    val_f1 = f1_score(val["label"], val_pred, average="macro")

    pred = clf.predict(Xte)
    acc = accuracy_score(test["label"], pred)
    f1 = f1_score(test["label"], pred, average="macro")
    report = classification_report(test["label"], pred, digits=4)
    labels = sorted(train["label"].unique())
    cm = confusion_matrix(test["label"], pred, labels=labels).tolist()
    print(f"== tfidf_{input_type} == val_acc={val_acc:.4f} "
          f"test_acc={acc:.4f} macro_f1={f1:.4f}")
    print(report)

    tag = f"tfidf_{input_type}"
    result = {
        "tag": tag,
        "input_type": input_type,
        "model_type": "tfidf+logreg",
        "vocab_size": len(vectorizer.vocabulary_),
        "labels": labels,
        "test": {"accuracy": acc, "macro_f1": f1,
                 "confusion_matrix": cm, "report": report},
    }
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(f"{RESULT_DIR}/result_{tag}.json", "w") as f:
        json.dump(result, f, indent=2)


def main():
    for input_type in ["input", "output", "combined"]:
        run(input_type)


if __name__ == "__main__":
    main()
