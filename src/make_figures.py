"""Generate all figures for the report from saved result JSONs."""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULT_DIR = "results"
FIG_DIR = "report/figures"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 10})


def load_results(pattern="result_*.json"):
    out = {}
    for path in glob.glob(f"{RESULT_DIR}/{pattern}"):
        r = json.load(open(path))
        out[r["tag"]] = r
    return out


def fig_rq2(results):
    inputs = ["input", "output", "combined"]
    series = [("CNN", lambda i: results.get(f"{i}_cnn")),
              ("LSTM", lambda i: results.get(f"{i}_lstm")),
              ("TF-IDF + LR", lambda i: results.get(f"tfidf_{i}"))]
    series = [(name, fn) for name, fn in series
              if all(fn(i) is not None for i in inputs)]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(inputs))
    width = 0.8 / len(series)
    for i, (name, fn) in enumerate(series):
        accs = [fn(inp)["test"]["accuracy"] for inp in inputs]
        offset = (i - (len(series) - 1) / 2) * width
        bars = ax.bar(x + offset, accs, width, label=name)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)
    ax.axhline(1 / 6, color="gray", ls="--", lw=1)
    ax.text(2.42, 1 / 6 + 0.005, "chance", color="gray", fontsize=8)
    ax.set_xticks(x, ["Prompt only", "Response only", "Prompt + response"])
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 0.9)
    ax.set_title("RQ2: effect of input type on LLM identification")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/rq2_input_type.png")
    plt.close(fig)


def fig_confusion(tag, results):
    r = results[tag]
    cm = np.array(r["test"]["confusion_matrix"])
    labels = r["labels"]
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalized")
    ax.set_title(f"Confusion matrix: {tag}")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/confusion_{tag}.png")
    plt.close(fig)


def fig_curves(results, tags):
    pretty = {
        "output_cnn": "Response CNN",
        "output_lstm": "Response BiLSTM",
        "combined_cnn": "Prompt+response CNN",
        "combined_lstm": "Prompt+response BiLSTM",
        "input_cnn": "Prompt CNN",
        "input_lstm": "Prompt BiLSTM",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
    for tag in tags:
        hist = pd.DataFrame(results[tag]["history"])
        label = pretty.get(tag, tag)
        axes[0].plot(hist["epoch"], hist["val_acc"], marker="o", ms=3,
                     label=label)
        axes[1].plot(hist["epoch"], hist["val_f1"], marker="o", ms=3,
                     label=label)
    for ax, name in zip(axes, ["val accuracy", "val macro-F1"]):
        ax.set_xlabel("epoch")
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
    axes[0].axhline(1 / 6, color="gray", ls="--", lw=1)
    axes[0].text(0.5, 1 / 6 + 0.012, "chance", color="gray", fontsize=8)
    axes[0].legend(fontsize=7, loc="lower right")
    axes[1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Validation curves (response-based models)", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(f"{FIG_DIR}/training_curves.png")
    plt.close(fig)


def fig_rq4(results):
    rq4 = json.load(open(f"{RESULT_DIR}/rq4_results.json"))
    if "output_cnn" in results:
        rq4["CNN (response)"] = {
            "accuracy": results["output_cnn"]["test"]["accuracy"]}
    names = ["length_only", "structural", "structural_lexical"]
    pretty = ["Length only", "Structural", "Structural + lexical"]
    accs = [rq4[n]["accuracy"] for n in names]
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    bars = ax.bar(pretty, accs, color=["#bbb", "#88a", "#557"])
    ax.bar_label(bars, fmt="%.3f")
    ax.axhline(1 / 6, color="gray", ls="--", lw=1)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Test accuracy")
    ax.set_title("RQ4: accuracy from non-neural features")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/rq4_features.png")
    plt.close(fig)

    df = pd.read_parquet("data/processed/dataset.parquet")
    df["words"] = df["response"].str.split().str.len()
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    order = df.groupby("label")["words"].median().sort_values().index
    data = [df[df["label"] == lab]["words"] for lab in order]
    bp = ax.boxplot(data, tick_labels=order, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#88a5cc")
    ax.set_ylabel("response length (words)")
    ax.set_title("RQ4: response length distribution by model")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/rq4_length.png")
    plt.close(fig)


def fig_rq3():
    tags = {
        "in_code": "Code (in-domain)",
        "in_math": "Math (in-domain)",
        "in_creative": "Creative writing (in-domain)",
        "in_general": "General (in-domain)",
        "noncreative_to_creative": "Train non-creative -> test creative",
        "code_to_math": "Train code -> test math",
        "general_to_code": "Train general -> test code",
    }
    names, accs, colors = [], [], []
    for tag, pretty in tags.items():
        path = f"{RESULT_DIR}/result_rq3_{tag}.json"
        if not os.path.exists(path):
            continue
        r = json.load(open(path))
        names.append(pretty)
        accs.append(r["test"]["accuracy"])
        colors.append("#4c72b0" if tag.startswith("in_") else "#c44e52")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.barh(names, accs, color=colors)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.axvline(1 / 6, color="gray", ls="--", lw=1)
    ax.text(1 / 6 + 0.005, 0.25, "chance", color="gray", fontsize=8)
    ax.set_xlim(0, 0.92)
    ax.invert_yaxis()
    ax.set_xlabel("Test accuracy")
    ax.set_title("RQ3: in-domain vs cross-domain transfer\n"
                 "(CNN, response-only)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/rq3_transfer.png")
    plt.close(fig)


def main():
    results = load_results()
    print("found:", sorted(results))
    if all(f"{i}_{m}" in results for i in ("input", "output", "combined") for m in ("cnn", "lstm")):
        fig_rq2(results)
    if "output_cnn" in results:
        fig_confusion("output_cnn", results)
    curves = [t for t in ("output_cnn", "output_lstm", "combined_cnn") if t in results]
    if curves:
        fig_curves(results, curves)
    if os.path.exists(f"{RESULT_DIR}/rq4_results.json"):
        fig_rq4(results)
    if os.path.exists(f"{RESULT_DIR}/result_rq3_in_code.json"):
        fig_rq3()
    print("figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
