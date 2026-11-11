"""Turn plain data files into SIMPLE's training signal: preference pairs plus regression targets.

You describe *what kind* of data each file holds; SIMPLE derives the rest (paper §4.2).

=============  ===============================================  ====================================
kind           columns                                          what is derived
=============  ===============================================  ====================================
parallel       source, simplification                           original-simplification pairs
                                                                ``s > o`` given ``o`` (+ mirror and
                                                                random-sentence pairs, see below)
graded         source, simplification, level [, group]          as ``parallel``, plus inter-
                                                                simplification pairs: for the same
                                                                source, a higher level (= simpler)
                                                                beats a lower one by >= min_level_gap
ratings        source, candidate, rating                        regression targets on (-1, 1);
                                                                human-preference pairs with margins;
                                                                copy/flipped/cross-source anchors
preferences    source, chosen, rejected [, margin, weight,      used as given
               type]
=============  ===============================================  ====================================

Derived from every original-simplification pair ``(o: s > o)``:

* mirror pair ``(s: s > o)``: given the simple sentence, copying it beats complexifying it;
* random-sentence pair ``(o: s > r)`` with ``r`` an unrelated source sentence.

Regression anchors (copies and degenerate rewrites, see :mod:`simple_metric.data.augment`) are
drawn from the training sources after the train/validation split.

Pair types in the output: ``original``, ``mirror``, ``random_sentence``, ``inter`` (graded),
``rating_pair`` (from ratings), ``preference`` (or the ``type`` given in the file).
"""

from __future__ import annotations

import logging
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import augment
from .io import DataFormatError, describe, read_records

logger = logging.getLogger(__name__)

KINDS = ("parallel", "graded", "ratings", "preferences")

#: Pair types that compare two simplifications; used for checkpoint selection ("val_inter").
INTER_TYPES = ("inter", "rating_pair", "preference")

DEFAULT_TYPE_WEIGHTS = {"rating_pair": 3.0}


@dataclass
class AugmentationOptions:
    mirror: bool = True
    random_sentence: bool = True
    copy_anchors: int = 5000
    adversarial_anchors: int = 5000
    rating_anchors: bool = True


@dataclass
class ValidationOptions:
    fraction: float = 0.1
    split: str = "grouped"  # "grouped" (by source sentence, no leakage) or "random" (paper)
    path: Any = None  # explicit validation preferences (same format as `preferences`)


@dataclass
class DataSpec:
    parallel: list = field(default_factory=list)
    graded: list = field(default_factory=list)
    ratings: list = field(default_factory=list)
    preferences: list = field(default_factory=list)
    augmentation: AugmentationOptions = field(default_factory=AugmentationOptions)
    validation: ValidationOptions = field(default_factory=ValidationOptions)
    type_weights: dict = field(default_factory=lambda: dict(DEFAULT_TYPE_WEIGHTS))
    seed: int = 42
    base_dir: Path | None = None

    @classmethod
    def from_dict(cls, data: dict, base_dir: Path | None = None) -> DataSpec:
        data = dict(data or {})
        unknown = set(data) - {*KINDS, "augmentation", "validation", "type_weights", "seed"}
        if unknown:
            raise DataFormatError(
                f"Unknown key(s) in data spec: {sorted(unknown)}. Expected some of {KINDS} "
                "plus augmentation / validation / type_weights / seed."
            )
        spec = cls(base_dir=base_dir)
        for kind in KINDS:
            entries = data.get(kind) or []
            if isinstance(entries, (str, dict)):
                entries = [entries]
            setattr(spec, kind, list(entries))
        spec.augmentation = AugmentationOptions(**(data.get("augmentation") or {}))
        spec.validation = ValidationOptions(**(data.get("validation") or {}))
        spec.type_weights = {**DEFAULT_TYPE_WEIGHTS, **(data.get("type_weights") or {})}
        spec.seed = int(data.get("seed", 42))
        if not any(getattr(spec, k) for k in KINDS):
            raise DataFormatError(f"The data spec lists no data: add at least one of {KINDS}.")
        return spec


@dataclass
class TrainingData:
    train: list[dict]
    val: list[dict]
    regression: list[dict]

    def stats(self) -> dict:
        return {
            "train_pairs": len(self.train),
            "val_pairs": len(self.val),
            "regression_examples": len(self.regression),
            "train_pairs_by_type": dict(sorted(Counter(r["type"] for r in self.train).items())),
            "val_pairs_by_type": dict(sorted(Counter(r["type"] for r in self.val).items())),
            "regression_by_type": dict(sorted(Counter(r["type"] for r in self.regression).items())),
        }


# ---------------------------------------------------------------------------- helpers


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _number(value: Any, what: str, where: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise DataFormatError(f"{where}: {what} must be numeric, got {value!r}") from None


def _norm(s: str) -> str:
    return " ".join(s.split())


def _pair(source, chosen, rejected, type_, weight=None, margin=0.0, group=None) -> dict:
    return {
        "source": source,
        "chosen": chosen,
        "rejected": rejected,
        "type": type_,
        "margin": float(margin),
        "weight": weight,
        "group": group if group is not None else source,
    }


def _entry_options(entry: Any) -> dict:
    return entry if isinstance(entry, dict) else {}


# ---------------------------------------------------------------------------- per kind


def pairs_from_parallel(rows: list[dict], weight: float | None = None) -> list[dict]:
    """``(o, s)`` rows -> ``(o: s > o)`` original-simplification pairs."""
    out = []
    for r in rows:
        src, simp = _text(r["source"]), _text(r["simplification"])
        if src and simp and simp != src:
            out.append(_pair(src, simp, src, "original", weight))
    return out


def pairs_from_graded(
    rows: list[dict], min_level_gap: float = 1.0, weight: float | None = None,
    original_pairs: bool = True, where: str = "graded",
) -> list[dict]:
    """Rows of ``(source, simplification, level[, group])`` where a higher level means a
    simpler text. Emits an original-simplification pair for every row and, for each
    (group, source), an inter-simplification pair for every two simplifications whose
    levels differ by at least ``min_level_gap`` (each distinct text keeps its first level)."""
    out = []
    targets: dict[tuple, dict[str, float]] = defaultdict(dict)
    for r in rows:
        src, simp = _text(r["source"]), _text(r["simplification"])
        if not src or not simp or simp == src:
            continue
        level = _number(r["level"], "level", where)
        if original_pairs:
            out.append(_pair(src, simp, src, "original", weight))
        key = (_text(r.get("group")), src)
        targets[key].setdefault(simp, level)
    for (_, src), simp_to_level in targets.items():
        items = list(simp_to_level.items())
        for hi_text, hi_level in items:
            for lo_text, lo_level in items:
                if hi_level - lo_level >= min_level_gap:
                    out.append(_pair(src, hi_text, lo_text, "inter", weight))
    return out


def rating_targets(rows: list[dict], scale: tuple[float, float], where: str) -> list[dict]:
    """Map ratings on ``[lo, hi]`` linearly to targets on ``[-1, 1]`` (paper: y = 2r/100 - 1)."""
    lo, hi = float(scale[0]), float(scale[1])
    if hi <= lo:
        raise DataFormatError(f"{where}: rating scale must be [low, high], got {scale}")
    out = []
    for r in rows:
        src, cand = _text(r["source"]), _text(r["candidate"])
        if not src or not cand:
            continue
        rating = _number(r["rating"], "rating", where)
        target = round(2.0 * (rating - lo) / (hi - lo) - 1.0, 4)
        out.append({"source": src, "candidate": cand, "target": target, "type": "human"})
    return out


def pairs_from_ratings(
    rated: list[dict], min_gap: float = 0.15, max_margin: float = 0.9,
    vs_copy: bool = True, weight: float | None = None,
) -> list[dict]:
    """Human-preference pairs from rated candidates of the same source (paper §4.2 (iii)).

    Any two candidates whose targets differ by at least ``min_gap`` give a pair with margin
    ``min(gap, max_margin)``. With ``vs_copy``, every candidate with ``|target| >= min_gap``
    is also compared with a verbatim copy of the source (target 0), anchoring the zero point.
    """
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in rated:
        by_source[r["source"]].append(r)
    out = []
    for src, group in by_source.items():
        for a in group:
            for b in group:
                if a is b:
                    continue
                gap = a["target"] - b["target"]
                if gap >= min_gap and a["candidate"] != b["candidate"]:
                    out.append(_pair(src, a["candidate"], b["candidate"], "rating_pair",
                                     weight, round(min(gap, max_margin), 4)))
        if vs_copy:
            for a in group:
                t = a["target"]
                if abs(t) < min_gap:
                    continue
                if t > 0:
                    out.append(_pair(src, a["candidate"], src, "rating_pair", weight,
                                     round(min(t, max_margin), 4)))
                else:
                    out.append(_pair(src, src, a["candidate"], "rating_pair", weight,
                                     round(min(-t, max_margin), 4)))
    return out


def pairs_from_preferences(rows: list[dict], default_type: str = "preference",
                           weight: float | None = None, where: str = "preferences") -> list[dict]:
    out = []
    for r in rows:
        src, ch, rj = _text(r["source"]), _text(r["chosen"]), _text(r["rejected"])
        if not src or not ch or not rj:
            continue
        margin = r.get("margin")
        margin = 0.0 if margin in (None, "") else _number(margin, "margin", where)
        w = r.get("weight")
        w = weight if w in (None, "") else _number(w, "weight", where)
        out.append(_pair(src, ch, rj, _text(r.get("type")) or default_type, w, margin))
    return out


# ---------------------------------------------------------------------------- assembly


def _derive(records: list[dict], opts: AugmentationOptions, seed: int) -> list[dict]:
    originals = [r for r in records if r["type"] == "original"]
    derived = []
    if opts.random_sentence and originals:
        partner = augment.derangement_partner(sorted({r["source"] for r in records}),
                                              random.Random(seed + 1))
        for r in originals:
            other = partner[r["source"]]
            if other != r["source"]:
                derived.append(_pair(r["source"], r["chosen"], other, "random_sentence",
                                     r["weight"], group=r["group"]))
    if opts.mirror:
        derived.extend(_pair(r["chosen"], r["chosen"], r["rejected"], "mirror", r["weight"],
                             group=r["group"]) for r in originals)
    return derived


def _split(records: list[dict], opts: ValidationOptions, seed: int):
    if opts.fraction <= 0:
        return records, []
    rng = random.Random(seed)
    if opts.split == "random":
        shuffled = records[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * opts.fraction))
        return shuffled[n_val:], shuffled[:n_val]
    if opts.split != "grouped":
        raise DataFormatError(f"validation.split must be 'grouped' or 'random', not {opts.split!r}")
    groups = sorted({r["group"] for r in records})
    rng.shuffle(groups)
    held_out = set(groups[: max(1, int(len(groups) * opts.fraction))])
    train = [r for r in records if r["group"] not in held_out]
    val = [r for r in records if r["group"] in held_out]
    return train, val


def build_training_data(spec: DataSpec | dict, base_dir: Path | None = None) -> TrainingData:
    """Read every source in ``spec`` and build train/validation pairs and regression data."""
    if isinstance(spec, dict):
        spec = DataSpec.from_dict(spec, base_dir=base_dir)
    base = spec.base_dir
    seed = spec.seed
    records: list[dict] = []
    rated: list[dict] = []

    for entry in spec.parallel:
        o = _entry_options(entry)
        rows = read_records(entry, ["source", "simplification"], base_dir=base)
        new = pairs_from_parallel(rows, o.get("weight"))
        logger.info("parallel   %-40s %8d rows -> %8d pairs", describe(entry), len(rows), len(new))
        records += new

    for entry in spec.graded:
        o = _entry_options(entry)
        rows = read_records(entry, ["source", "simplification", "level"], ["group"], base_dir=base)
        new = pairs_from_graded(rows, float(o.get("min_level_gap", 1)), o.get("weight"),
                                bool(o.get("original_pairs", True)), describe(entry))
        logger.info("graded     %-40s %8d rows -> %8d pairs", describe(entry), len(rows), len(new))
        records += new

    for entry in spec.preferences:
        o = _entry_options(entry)
        rows = read_records(entry, ["source", "chosen", "rejected"], ["margin", "weight", "type"],
                            base_dir=base)
        new = pairs_from_preferences(rows, o.get("type", "preference"), o.get("weight"),
                                     describe(entry))
        logger.info("preference %-40s %8d rows -> %8d pairs", describe(entry), len(rows), len(new))
        records += new

    for entry in spec.ratings:
        o = _entry_options(entry)
        if "scale" not in o:
            raise DataFormatError(
                f"ratings {describe(entry)}: give the rating scale, e.g. scale: [0, 100]")
        rows = read_records(entry, ["source", "candidate", "rating"], base_dir=base)
        targets = rating_targets(rows, o["scale"], describe(entry))
        n_pairs = 0
        if o.get("pairs", True):
            new = pairs_from_ratings(targets, float(o.get("min_gap", 0.15)),
                                     float(o.get("max_margin", 0.9)), bool(o.get("vs_copy", True)),
                                     o.get("weight"))
            records += new
            n_pairs = len(new)
        if o.get("regression", True):
            rated += targets
        logger.info("ratings    %-40s %8d rows -> %8d pairs", describe(entry), len(rows), n_pairs)

    records = [r for r in records if _norm(r["chosen"]) != _norm(r["rejected"])]
    records += _derive(records, spec.augmentation, seed)

    # Fill per-type default weights where the source did not set one.
    for r in records:
        if r["weight"] is None:
            r["weight"] = float(spec.type_weights.get(r["type"], 1.0))

    if spec.validation.path:
        rows = read_records(spec.validation.path, ["source", "chosen", "rejected"],
                            ["margin", "weight", "type"], base_dir=base)
        val = pairs_from_preferences(rows, "preference", 1.0, "validation")
        train = records
    else:
        train, val = _split(records, spec.validation, seed)

    regression = list(rated)
    opts = spec.augmentation
    if rated and opts.rating_anchors:
        regression += augment.rating_anchors(rated, seed)
    train_sources = [r["source"] for r in train]
    if opts.copy_anchors > 0:
        regression += augment.copy_anchors(train_sources, opts.copy_anchors, seed + 99)
    if opts.adversarial_anchors > 0:
        regression += augment.adversarial_anchors(train_sources, opts.adversarial_anchors,
                                                  seed + 777)

    data = TrainingData(train=train, val=val, regression=regression)
    if not data.train:
        raise DataFormatError("No training pairs were produced; check the data spec.")
    return data
