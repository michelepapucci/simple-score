"""Read tabular training data from common formats, with flexible column names.

A *source spec* is either a path string or a dict:

    {"path": "ratings.csv", "columns": {"source": "original", "rating": "score"}}
    {"files": {"source": "train.complex.txt", "simplification": "train.simple.txt"}}
    {"hf": "user/dataset", "split": "train", "config": None}

Supported files: ``.jsonl``/``.ndjson``, ``.json`` (a list of objects), ``.csv``,
``.tsv``/``.tab`` (no quoting, so sentences may contain quotes), ``.parquet``, each
optionally gzipped (``.gz``). ``files`` reads line-aligned plain-text files, one
column per file. ``hf`` loads a Hugging Face Hub dataset (needs ``datasets``).

``columns`` maps the canonical column name to the name used in the file. Without it,
common aliases are tried (e.g. ``original``/``complex`` for ``source``).
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# canonical name -> accepted aliases (first match wins, case-insensitive)
ALIASES: dict[str, tuple[str, ...]] = {
    "source": ("source", "src", "original", "orig", "complex", "complex_sentence", "input"),
    "simplification": (
        "simplification", "simple", "simplified", "simp", "target", "tgt", "reference",
        "candidate", "output",
    ),
    "candidate": (
        "candidate", "simplification", "generation", "output", "prediction", "hypothesis",
        "system_output", "simp",
    ),
    "chosen": ("chosen", "positive", "pos", "preferred", "better", "winner"),
    "rejected": ("rejected", "negative", "neg", "dispreferred", "worse", "loser"),
    "rating": ("rating", "score", "human_score", "composite", "composite_rating", "label"),
    "level": ("level", "grade", "reading_level", "simplification_level", "version"),
    "group": ("group", "group_id", "doc_id", "cluster"),
    "margin": ("margin",),
    "weight": ("weight",),
    "type": ("type", "pair_type"),
}

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


class DataFormatError(ValueError):
    """Raised when an input file cannot be mapped onto the expected columns."""


def _open_text(path: Path):
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return path.open(encoding="utf-8", newline="")


def _base_suffix(path: Path) -> str:
    suffixes = [s.lower() for s in path.suffixes]
    if suffixes and suffixes[-1] == ".gz":
        suffixes = suffixes[:-1]
    return suffixes[-1] if suffixes else ""


def _read_file(path: Path) -> list[dict[str, Any]]:
    suffix = _base_suffix(path)
    if suffix in (".jsonl", ".ndjson"):
        with _open_text(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    if suffix == ".json":
        with _open_text(path) as f:
            data = json.load(f)
        if isinstance(data, dict):  # {"data": [...]} or column-oriented
            if "data" in data and isinstance(data["data"], list):
                data = data["data"]
            else:
                keys = list(data)
                data = [dict(zip(keys, row)) for row in zip(*(data[k] for k in keys))]
        return list(data)
    if suffix == ".csv":
        with _open_text(path) as f:
            return list(csv.DictReader(f))
    if suffix in (".tsv", ".tab"):
        with _open_text(path) as f:
            return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as e:  # pragma: no cover
            raise ImportError("Reading Parquet needs `pip install simple-metric[data]`.") from e
        return pq.read_table(path).to_pylist()
    raise DataFormatError(
        f"Don't know how to read {path} (suffix {suffix!r}). Use .jsonl, .json, .csv, "
        ".tsv, .parquet (optionally .gz), or line-aligned text files via `files:`."
    )


def _read_aligned_files(files: dict[str, str | os.PathLike], base_dir: Path | None) -> list[dict]:
    columns = {}
    for name, p in files.items():
        with _open_text(resolve_path(p, base_dir)) as f:
            columns[name] = [line.rstrip("\n").rstrip("\r") for line in f]
    lengths = {k: len(v) for k, v in columns.items()}
    if len(set(lengths.values())) != 1:
        raise DataFormatError(f"Line-aligned files have different lengths: {lengths}")
    names = list(columns)
    return [dict(zip(names, row)) for row in zip(*(columns[k] for k in names))]


def _read_hf(spec: dict) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise ImportError("Loading Hub datasets needs `pip install simple-metric[data]`.") from e
    ds = load_dataset(spec["hf"], spec.get("config"), split=spec.get("split", "train"))
    return ds.to_list()


def resolve_path(p: str | os.PathLike, base_dir: Path | None = None) -> Path:
    """Resolve ``p`` against ``base_dir`` (e.g. the YAML file's folder) if it exists there,
    else against the current working directory."""
    path = Path(os.path.expanduser(str(p)))
    if not path.is_absolute() and base_dir is not None and (base_dir / path).exists():
        return base_dir / path
    return path


def describe(spec: Any) -> str:
    if isinstance(spec, dict):
        for key in ("name", "path", "hf"):
            if key in spec:
                return str(spec[key])
        if "files" in spec:
            return " + ".join(str(v) for v in spec["files"].values())
    return str(spec)


def read_records(
    spec: str | os.PathLike | dict,
    required: Iterable[str],
    optional: Iterable[str] = (),
    base_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Read a source spec and return rows keyed by canonical column names.

    Only the ``required`` and ``optional`` canonical columns are kept. Raises
    :class:`DataFormatError` with the available columns if a required one is missing.
    """
    if not isinstance(spec, dict):
        spec = {"path": spec}
    if "files" in spec:
        rows = _read_aligned_files(spec["files"], base_dir)
    elif "hf" in spec:
        rows = _read_hf(spec)
    elif "path" in spec:
        rows = _read_file(resolve_path(spec["path"], base_dir))
    else:
        raise DataFormatError(f"A data source needs `path`, `files` or `hf`; got {spec}")
    if not rows:
        return []

    header = list(rows[0].keys())
    lower = {h.lower(): h for h in header}
    explicit = spec.get("columns") or {}
    mapping: dict[str, str] = {}
    for canonical in list(required) + list(optional):
        if canonical in explicit:
            name = explicit[canonical]
            if name not in rows[0]:
                raise DataFormatError(
                    f"{describe(spec)}: column {name!r} (for {canonical!r}) not found. "
                    f"Available columns: {header}"
                )
            mapping[canonical] = name
            continue
        for alias in ALIASES.get(canonical, (canonical,)):
            if alias in lower and lower[alias] not in mapping.values():
                mapping[canonical] = lower[alias]
                break
    missing = [c for c in required if c not in mapping]
    if missing:
        raise DataFormatError(
            f"{describe(spec)}: could not find column(s) {missing}. Available columns: "
            f"{header}. Map them explicitly, e.g. columns: {{{missing[0]}: <your column>}}"
        )
    return [{c: row.get(name) for c, name in mapping.items()} for row in rows]


def write_jsonl(records: Iterable[dict], path: str | os.PathLike) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n
