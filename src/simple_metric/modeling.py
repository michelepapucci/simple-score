"""The SIMPLE model: a DeBERTa-v3 cross-encoder with a mean head and an uncertainty head.

Given ``[CLS] source [SEP] candidate [SEP]``, the [CLS] representation ``h`` is projected
to a quality score and its uncertainty::

    mu    = tanh(W_mu h)                    in (-1, 1)
    sigma = softplus(W_sigma h) + floor     > floor   (floor = 0.05)

A checkpoint directory holds ``config.json`` (this module's config, with the backbone
config nested under ``"encoder"``), ``model.safetensors`` (all weights) and the
tokenizer in ``tokenizer/``. Loading never downloads the original backbone.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._version import __version__

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"
# The tokenizer lives in a subfolder so that transformers never mistakes this package's
# config.json for a transformers model config when loading it.
TOKENIZER_SUBFOLDER = "tokenizer"
FORMAT_VERSION = 1

# Keys of a transformers config dict that only describe where/how it was saved.
_VOLATILE_ENCODER_KEYS = ("_name_or_path", "transformers_version", "_commit_hash")


@dataclass
class SimpleConfig:
    """Configuration of a SIMPLE model.

    Args:
        encoder: the backbone's transformers config as a dict (must contain ``model_type``).
        sigma_floor: constant added after the softplus so that ``sigma > sigma_floor``.
        max_length: maximum tokenized length of ``source + candidate``.
    """

    encoder: dict[str, Any] = field(default_factory=dict)
    sigma_floor: float = 0.05
    max_length: int = 512
    model_type: str = "simple"
    format_version: int = FORMAT_VERSION
    simple_metric_version: str = __version__

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimpleConfig:
        known = {f for f in cls.__dataclass_fields__}
        cfg = cls(**{k: v for k, v in data.items() if k in known})
        if cfg.format_version > FORMAT_VERSION:
            raise ValueError(
                f"This checkpoint uses format version {cfg.format_version}, but this "
                f"simple-metric ({__version__}) reads up to {FORMAT_VERSION}. "
                "Upgrade with `pip install -U simple-metric`."
            )
        return cfg

    def save(self, directory: str | os.PathLike) -> None:
        path = Path(directory) / CONFIG_NAME
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, directory: str | os.PathLike) -> SimpleConfig:
        data = json.loads((Path(directory) / CONFIG_NAME).read_text())
        if data.get("model_type") != "simple":
            raise ValueError(
                f"{directory} does not look like a SIMPLE checkpoint "
                f"(config.json has model_type={data.get('model_type')!r})."
            )
        return cls.from_dict(data)


def _encoder_config_from_dict(data: dict[str, Any]):
    from transformers import AutoConfig

    data = {k: v for k, v in data.items() if k not in _VOLATILE_ENCODER_KEYS}
    model_type = data.pop("model_type")
    return AutoConfig.for_model(model_type, **data)


def _encoder_config_to_dict(config) -> dict[str, Any]:
    data = config.to_dict()
    for k in _VOLATILE_ENCODER_KEYS:
        data.pop(k, None)
    return data


def _load_backbone(name_or_path: str, **kwargs):
    from transformers import AutoModel

    # Prefer safetensors: transformers refuses .bin files on torch < 2.6.
    try:
        encoder = AutoModel.from_pretrained(name_or_path, use_safetensors=True, **kwargs)
    except (OSError, ValueError):
        encoder = AutoModel.from_pretrained(name_or_path, **kwargs)
    # Always train and score in float32 (bf16 backbone files can produce NaNs here).
    return encoder.to(torch.float32)


class SimpleModel(nn.Module):
    """DeBERTa-v3 cross-encoder returning ``(mu, sigma)`` for each (source, candidate) pair."""

    def __init__(self, config: SimpleConfig, encoder: nn.Module | None = None) -> None:
        super().__init__()
        self.config = config
        if encoder is None:
            from transformers import AutoModel

            encoder = AutoModel.from_config(_encoder_config_from_dict(config.encoder))
            encoder = encoder.to(torch.float32)
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size
        self.mu_head = nn.Linear(hidden_size, 1)
        self.sigma_head = nn.Linear(hidden_size, 1)

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_backbone(
        cls,
        backbone: str = "microsoft/deberta-v3-base",
        sigma_floor: float = 0.05,
        max_length: int = 512,
        **kwargs: Any,
    ) -> SimpleModel:
        """A new, untrained SIMPLE model on top of a pretrained encoder (for training)."""
        encoder = _load_backbone(backbone, **kwargs)
        config = SimpleConfig(
            encoder=_encoder_config_to_dict(encoder.config),
            sigma_floor=sigma_floor,
            max_length=max_length,
        )
        return cls(config, encoder=encoder)

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str | os.PathLike,
        *,
        revision: str | None = None,
        cache_dir: str | os.PathLike | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
    ) -> SimpleModel:
        """Load a trained SIMPLE model from a local directory or a Hub repo id."""
        directory = resolve_checkpoint_dir(
            name_or_path,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        from safetensors.torch import load_model

        model = cls(SimpleConfig.load(directory))
        load_model(model, str(Path(directory) / WEIGHTS_NAME), strict=True, device="cpu")
        return model.eval()

    def save_pretrained(self, directory: str | os.PathLike, tokenizer=None) -> None:
        """Write ``config.json``, ``model.safetensors`` and (optionally) the tokenizer."""
        from safetensors.torch import save_model

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.config.simple_metric_version = __version__
        self.config.save(directory)
        save_model(self, str(directory / WEIGHTS_NAME), metadata={"format": "pt"})
        if tokenizer is not None:
            tokenizer.save_pretrained(directory / TOKENIZER_SUBFOLDER)

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``mu`` in (-1, 1) and ``sigma`` > ``sigma_floor``, each of shape (batch,)."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = outputs.last_hidden_state[:, 0, :].float()
        mu = torch.tanh(self.mu_head(cls_repr).squeeze(-1))
        sigma = F.softplus(self.sigma_head(cls_repr).squeeze(-1)) + self.config.sigma_floor
        return mu, sigma


def resolve_checkpoint_dir(
    name_or_path: str | os.PathLike,
    *,
    revision: str | None = None,
    cache_dir: str | os.PathLike | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> Path:
    """Return a local directory for a checkpoint: the path itself, or a Hub snapshot."""
    from .hub import resolve_model_name

    path = Path(name_or_path)
    if path.is_dir():
        return path
    repo_id = resolve_model_name(str(name_or_path))
    if Path(repo_id).is_dir():
        return Path(repo_id)
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer/*"],
        )
    )


def load_tokenizer(directory: str | os.PathLike):
    """Load the tokenizer stored with a checkpoint (``tokenizer/``, or the folder itself)."""
    from transformers import AutoTokenizer

    directory = Path(directory)
    sub = directory / TOKENIZER_SUBFOLDER
    return AutoTokenizer.from_pretrained(str(sub if sub.is_dir() else directory))
