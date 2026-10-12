"""Assemble markdown tables for the report from saved results."""

import glob
import json
import os

import pandas as pd

RESULT_DIR = "results"
OUT_DIR = "report/tables"
os.makedirs(OUT_DIR, exist_ok=True)


def load(tag):
    with open(f"{RESULT_DIR}/result_{tag}.json") as f:
        return json.load(f)


def main():
    main_tags = ["input_cnn", "input_lstm", "output_cnn", "output_lstm",
                 "combined_cnn", "combined_lstm",
                 "tfidf_input", "tfidf_output", "tfidf_combined"]
    rows = []
    for tag in main_tags:
        path = f"{RESULT_DIR}/result_{tag}.json"
        if not os.path.exists(path):
            continue
        r = load(tag)
        rows.append({
            "Input": r["input_type"], "Model": r["model_type"].upper(),
            "Accuracy": round(r["test"]["accuracy"], 4),
            "Macro-F1": round(r["test"]["macro_f1"], 4),
            "Best epoch": r.get("best_epoch", "n/a"),
            "Params": r.get("n_params", "n/a"),
        })
    table = pd.DataFrame(rows)
    table.to_markdown(f"{OUT_DIR}/main_results.md", index=False)
    print("== main results ==")
    print(table.to_string(index=False))

    best_tag = max(
        (t for t in main_tags if os.path.exists(f"{RESULT_DIR}/result_{t}.json")),
        key=lambda t: load(t)["test"]["accuracy"],
    )
    best = load(best_tag)
    print(f"\nbest config: {best_tag} ({best['test']['accuracy']:.4f})")

    rq3_rows = []
    for exp in ["in_code", "in_math", "in_creative", "in_general",
                "noncreative_to_creative", "code_to_math", "general_to_code"]:
        path = f"{RESULT_DIR}/result_rq3_{exp}.json"
        if not os.path.exists(path):
            continue
        r = load(f"rq3_{exp}")
        rq3_rows.append({
            "Experiment": exp,
            "Accuracy": round(r["test"]["accuracy"], 4),
            "Macro-F1": round(r["test"]["macro_f1"], 4),
        })
    if rq3_rows:
        rq3 = pd.DataFrame(rq3_rows)
        rq3.to_markdown(f"{OUT_DIR}/rq3_results.md", index=False)
        print("\n== RQ3 ==")
        print(rq3.to_string(index=False))

    path = f"{RESULT_DIR}/result_rq4_len128_output_cnn.json"
    if os.path.exists(path):
        r = load("rq4_len128_output_cnn")
        base = load("output_cnn")
        print(f"\nRQ4 length ablation: max_len 128 -> acc {r['test']['accuracy']:.4f} "
              f"(vs {base['test']['accuracy']:.4f} full length)")
        with open(f"{OUT_DIR}/rq4_ablation.md", "w") as f:
            f.write("| Condition | Accuracy | Macro-F1 |\n|---|---|---|\n")
            f.write(f"| Full length (300 tokens) | {base['test']['accuracy']:.4f} "
                    f"| {base['test']['macro_f1']:.4f} |\n")
            f.write(f"| Truncated (128 tokens) | {r['test']['accuracy']:.4f} "
                    f"| {r['test']['macro_f1']:.4f} |\n")


if __name__ == "__main__":
    main()
