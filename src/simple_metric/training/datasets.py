"""Torch datasets and collators for preference pairs and regression examples."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


class PairDataset(Dataset):
    """Tokenises ``(source, chosen)`` and ``(source, rejected)`` for each preference pair."""

    def __init__(self, records: list[dict], tokenizer, max_length: int = 512) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        pos = self.tokenizer(r["source"], r["chosen"], max_length=self.max_length, truncation=True)
        neg = self.tokenizer(r["source"], r["rejected"], max_length=self.max_length, truncation=True)
        return {
            "pos": pos["input_ids"],
            "neg": neg["input_ids"],
            "weight": float(r.get("weight", 1.0)),
            "margin": float(r.get("margin", 0.0)),
        }


class RegressionDataset(Dataset):
    """Tokenises ``(source, candidate)`` with a target on (-1, 1)."""

    def __init__(self, records: list[dict], tokenizer, max_length: int = 512) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        enc = self.tokenizer(r["source"], r["candidate"], max_length=self.max_length,
                             truncation=True)
        return {"ids": enc["input_ids"], "target": float(r["target"])}


def pad(sequences: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad token id lists; return ``(input_ids, attention_mask)``."""
    length = max(len(s) for s in sequences)
    ids = torch.full((len(sequences), length), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), length), dtype=torch.long)
    for i, s in enumerate(sequences):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = 1
    return ids, mask


@dataclass
class PairCollator:
    pad_id: int

    def __call__(self, features: list[dict]) -> dict:
        return {
            "pos": [f["pos"] for f in features],
            "neg": [f["neg"] for f in features],
            "weight": torch.tensor([f["weight"] for f in features], dtype=torch.float),
            "margin": torch.tensor([f["margin"] for f in features], dtype=torch.float),
        }


@dataclass
class RegressionCollator:
    pad_id: int

    def __call__(self, features: list[dict]) -> dict:
        return {
            "ids": [f["ids"] for f in features],
            "target": torch.tensor([f["target"] for f in features], dtype=torch.float),
        }
