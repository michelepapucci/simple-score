"""Training configuration: dataclasses plus YAML loading with ``key=value`` overrides."""

from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from ..data.io import resolve_path


@dataclass
class ModelConfig:
    backbone: str = "microsoft/deberta-v3-base"
    max_length: int = 512
    sigma_floor: float = 0.05
    gradient_checkpointing: bool = False


@dataclass
class TrainingConfig:
    epochs: int = 5
    #: preference pairs per device per micro-step
    batch_size: int = 16
    #: pairs per optimizer step across devices and accumulation; None = batch_size x devices
    global_batch_size: int | None = None
    #: regression examples per device per micro-step
    regression_batch_size: int = 32
    lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    grad_clip: float = 1.0
    #: weight of the regression loss (paper: lambda = 0.7)
    lambda_reg: float = 0.7
    pairwise_loss: str = "thurstone"  # thurstone | bt | none
    regression_loss: str = "nll"  # nll | mse | none
    #: fraction of the training pairs (and regression set) drawn at random each epoch
    epoch_fraction: float = 1.0
    seed: int = 42
    precision: str = "fp32"  # fp32 | bf16
    num_workers: int = 2
    #: checkpoint selection: val_inter | val_all | monitor[:name] | last
    select_on: str = "val_inter"
    save_every_epoch: bool = True
    eval_batch_size: int = 64
    #: score at most this many validation pairs per epoch (None = all)
    max_val_pairs: int | None = None
    log_every: int = 100
    resume: bool = False


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    #: the data spec (see simple_metric.data.builders) or a path to a YAML file with it
    data: Any = field(default_factory=dict)
    #: held-out sets reported every epoch: [{path, kind: pairwise|ratings, name, scale}]
    monitor: list = field(default_factory=list)
    output_dir: str = "runs/simple"
    #: folder that relative paths in the config are resolved against (the config's folder)
    base_dir: str | None = None
    #: folder that relative paths in the data spec are resolved against
    data_base_dir: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        raise ImportError("YAML configs need `pip install simple-metric[train]`.") from e
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_value(text: str) -> Any:
    import yaml

    return yaml.safe_load(text)


def apply_overrides(raw: dict, overrides: list[str]) -> dict:
    """Apply ``section.key=value`` overrides (values parsed as YAML)."""
    raw = copy.deepcopy(raw)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override {item!r} must look like section.key=value")
        key, value = item.split("=", 1)
        node = raw
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(value)
    return raw


def _build(cls, data: dict, where: str):
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"Unknown {where} option(s): {sorted(unknown)}. Valid: {sorted(names)}")
    return cls(**data)


def config_from_dict(raw: dict, base_dir: str | os.PathLike | None = None) -> Config:
    raw = dict(raw)
    base = Path(base_dir) if base_dir is not None else None
    data = raw.get("data", {})
    data_base = base
    if isinstance(data, (str, os.PathLike)):
        data_path = resolve_path(data, base)
        data = _load_yaml(data_path)
        data_base = data_path.parent
    cfg = Config(
        model=_build(ModelConfig, raw.get("model") or {}, "model"),
        training=_build(TrainingConfig, raw.get("training") or {}, "training"),
        data=data,
        monitor=list(raw.get("monitor") or []),
        output_dir=str(raw.get("output_dir", "runs/simple")),
        base_dir=str(base) if base is not None else None,
        data_base_dir=str(data_base) if data_base is not None else None,
    )
    unknown = set(raw) - {"model", "training", "data", "monitor", "output_dir", "base_dir",
                          "data_base_dir"}
    if unknown:
        raise ValueError(f"Unknown top-level config key(s): {sorted(unknown)}")
    return cfg


def load_config(path: str | os.PathLike, overrides: list[str] | None = None) -> Config:
    path = Path(path)
    raw = apply_overrides(_load_yaml(path), overrides or [])
    return config_from_dict(raw, base_dir=path.parent)
