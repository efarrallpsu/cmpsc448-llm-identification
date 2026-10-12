"""Tokenization, vocabulary, and dataset construction shared by all models."""

import re
from collections import Counter

import torch
from torch.utils.data import Dataset

TOKEN_RE = re.compile(r"[a-z0-9']+|[^\sa-z0-9]")
PAD, UNK, SEP = 0, 1, 2


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def build_vocab(texts, min_freq=2, max_size=30000):
    counts = Counter()
    for text in texts:
        counts.update(tokenize(text))
    words = [w for w, c in counts.most_common() if c >= min_freq][: max_size - 3]
    vocab = {"<pad>": PAD, "<unk>": UNK, "<sep>": SEP}
    for w in words:
        vocab[w] = len(vocab)
    return vocab


def encode(tokens, vocab, max_len):
    ids = [vocab.get(t, UNK) for t in tokens][:max_len]
    return ids, len(ids)


def model_input(row, input_type, vocab, max_len):
    prompt = tokenize(row["prompt"])
    response = tokenize(row["response"])
    if input_type == "input":
        tokens = prompt
    elif input_type == "output":
        tokens = response
    else:
        prompt = prompt[: max_len // 4]
        tokens = prompt + [SEP_TOKEN] + response
    return encode(tokens, vocab, max_len)


SEP_TOKEN = "<sep>"


class LLMDataset(Dataset):
    def __init__(self, df, input_type, vocab, max_len, label_to_id):
        self.samples = []
        for _, row in df.iterrows():
            ids, length = model_input(row, input_type, vocab, max_len)
            self.samples.append((ids, length, label_to_id[row["label"]]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, length, label = self.samples[idx]
        return torch.tensor(ids, dtype=torch.long), length, label


def collate(batch):
    ids, lengths, labels = zip(*batch)
    max_len = max(lengths)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, seq in enumerate(ids):
        padded[i, : len(seq)] = seq
    return padded, torch.tensor(lengths), torch.tensor(labels)
