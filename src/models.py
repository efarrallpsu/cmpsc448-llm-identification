"""Text CNN and BiLSTM classifiers."""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class TextCNN(nn.Module):
    def __init__(self, vocab_size, n_classes, emb_dim=64, channels=128,
                 kernel_sizes=(3, 4, 5), dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            nn.Conv1d(emb_dim, channels, k, padding=k // 2) for k in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels * len(kernel_sizes), n_classes)

    def forward(self, x, lengths=None):
        emb = self.embedding(x).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            h = torch.relu(conv(emb))
            pooled.append(torch.amax(h, dim=2))
        h = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(h)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, n_classes, emb_dim=64, hidden=128, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            emb_dim, hidden, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(self.dropout(h))


def build_model(name, vocab_size, n_classes):
    if name == "cnn":
        return TextCNN(vocab_size, n_classes)
    if name == "lstm":
        return LSTMClassifier(vocab_size, n_classes)
    raise ValueError(name)
