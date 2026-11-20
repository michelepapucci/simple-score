"""High-level scoring API.

>>> from simple_score import SIMPLE
>>> metric = SIMPLE.from_pretrained("large")
>>> out = metric.score(["The cat perched upon the mat."], ["The cat sat on the mat."])
>>> out.mu, out.sigma
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from .hub import DEFAULT_MODEL
from .losses import preference_probability
from .modeling import SimpleModel, load_tokenizer, resolve_checkpoint_dir


@dataclass
class ScoreOutput:
    """Per-example SIMPLE outputs.

    Attributes:
        mu: quality score in (-1, 1). +1 = very successful simplification, 0 = verbatim
            copy of the source, -1 = unrelated or degenerate text. Negative values flag
            unsuccessful simplifications (including complexifications).
        sigma: the model's uncertainty about ``mu`` (> 0.05). Low sigma = confident.
            It is a confidence signal, not an estimate of human disagreement.
    """

    mu: list[float]
    sigma: list[float]

    @property
    def mean(self) -> float:
        return float(np.mean(self.mu)) if self.mu else float("nan")

    def __len__(self) -> int:
        return len(self.mu)

    def to_dict(self) -> dict:
        return {"mu": self.mu, "sigma": self.sigma, "mean": self.mean}


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SIMPLE:
    """Reference-free text simplification metric with per-example uncertainty.

    Args:
        model: a :class:`SimpleModel`.
        tokenizer: the matching tokenizer.
        device: where to run; defaults to CUDA, then MPS, then CPU.
        batch_size: default batch size for :meth:`score`.
    """

    def __init__(self, model: SimpleModel, tokenizer, device=None, batch_size: int = 32) -> None:
        self.device = torch.device(device) if device is not None else _default_device()
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = model.config.max_length

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str | os.PathLike = DEFAULT_MODEL,
        *,
        device=None,
        batch_size: int = 32,
        revision: str | None = None,
        cache_dir: str | os.PathLike | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
    ) -> SIMPLE:
        """Load a released model (``"small"``, ``"base"``, ``"large"``), a Hub repo id,
        or a local checkpoint directory written by training."""
        directory = resolve_checkpoint_dir(
            name_or_path,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        model = SimpleModel.from_pretrained(directory)
        tokenizer = load_tokenizer(directory)
        return cls(model, tokenizer, device=device, batch_size=batch_size)

    # ------------------------------------------------------------------ scoring

    @torch.no_grad()
    def score(
        self,
        sources: str | Sequence[str],
        candidates: str | Sequence[str],
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> ScoreOutput:
        """Score each candidate simplification against its source sentence."""
        if isinstance(sources, str):
            sources = [sources]
        if isinstance(candidates, str):
            candidates = [candidates]
        sources, candidates = list(sources), list(candidates)
        if len(sources) != len(candidates):
            raise ValueError(
                f"Got {len(sources)} sources but {len(candidates)} candidates; "
                "they must be aligned one-to-one."
            )
        n = len(sources)
        if n == 0:
            return ScoreOutput(mu=[], sigma=[])
        batch_size = batch_size or self.batch_size

        # Sort by length so each batch pads to a similar length; restore order afterwards.
        order = sorted(range(n), key=lambda i: len(sources[i]) + len(candidates[i]), reverse=True)
        mu = np.empty(n, dtype=np.float64)
        sigma = np.empty(n, dtype=np.float64)

        batches: Iterable[int] = range(0, n, batch_size)
        if show_progress:
            from tqdm.auto import tqdm

            batches = tqdm(batches, total=(n + batch_size - 1) // batch_size, desc="SIMPLE")
        for start in batches:
            idx = order[start : start + batch_size]
            enc = self.tokenizer(
                [sources[i] for i in idx],
                [candidates[i] for i in idx],
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            m, s = self.model(
                enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)
            )
            mu[idx] = m.clamp(-1.0, 1.0).double().cpu().numpy()
            sigma[idx] = s.double().cpu().numpy()
        return ScoreOutput(mu=mu.tolist(), sigma=sigma.tolist())

    __call__ = score

    def preference_prob(
        self,
        sources: str | Sequence[str],
        candidates_a: str | Sequence[str],
        candidates_b: str | Sequence[str],
        batch_size: int | None = None,
    ) -> list[float]:
        """Probability that ``candidates_a[i]`` is a better simplification than
        ``candidates_b[i]`` of ``sources[i]``, from Thurstone's model (paper Eq. 3)."""
        if isinstance(sources, str):
            sources, candidates_a, candidates_b = [sources], [candidates_a], [candidates_b]
        a = self.score(sources, candidates_a, batch_size=batch_size)
        b = self.score(sources, candidates_b, batch_size=batch_size)
        p = preference_probability(a.mu, a.sigma, b.mu, b.sigma)
        return np.atleast_1d(p).tolist()


def load(name_or_path: str | os.PathLike = DEFAULT_MODEL, **kwargs) -> SIMPLE:
    """Shortcut for :meth:`SIMPLE.from_pretrained`."""
    return SIMPLE.from_pretrained(name_or_path, **kwargs)
