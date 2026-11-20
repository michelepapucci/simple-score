"""Command line interface: ``simple-score {score,train,build-data}``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _read_lines(path: str) -> list[str]:
    return [line.rstrip("\n") for line in Path(path).read_text(encoding="utf-8").splitlines()]


# ----------------------------------------------------------------------------- score


def _cmd_score(args: argparse.Namespace) -> int:
    from .scorer import SIMPLE

    if args.input:
        from .data.io import read_records

        rows = read_records(args.input, ["source", "candidate"])
        sources = [r["source"] for r in rows]
        candidates = [r["candidate"] for r in rows]
    else:
        if not (args.sources and args.candidates):
            print("Give --input FILE, or both --sources and --candidates.", file=sys.stderr)
            return 2
        sources, candidates = _read_lines(args.sources), _read_lines(args.candidates)

    metric = SIMPLE.from_pretrained(args.model, device=args.device, batch_size=args.batch_size)
    out = metric.score(sources, candidates, show_progress=not args.quiet)
    sink = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    try:
        for s, c, m, sg in zip(sources, candidates, out.mu, out.sigma):
            sink.write(json.dumps({"source": s, "candidate": c, "mu": round(m, 6),
                                   "sigma": round(sg, 6)}, ensure_ascii=False) + "\n")
    finally:
        if args.output:
            sink.close()
    print(f"SIMPLE mean score over {len(out)} examples: {out.mean:.4f}", file=sys.stderr)
    return 0


# ----------------------------------------------------------------------------- train / build-data


def _config_from_args(args: argparse.Namespace):
    from .training.config import apply_overrides, config_from_dict, load_config

    if args.config:
        cfg = load_config(args.config, args.set)
    else:
        data: dict = {}
        for kind in ("parallel", "graded", "preferences"):
            data[kind] = list(getattr(args, kind) or [])
        if args.graded and args.min_level_gap is not None:
            data["graded"] = [{"path": p, "min_level_gap": args.min_level_gap} for p in args.graded]
        if args.ratings:
            if not args.rating_scale:
                raise SystemExit("--ratings needs --rating-scale LOW HIGH (e.g. 0 100)")
            data["ratings"] = [{"path": p, "scale": args.rating_scale} for p in args.ratings]
        raw = apply_overrides({"data": data}, args.set)
        cfg = config_from_dict(raw, base_dir=Path.cwd())
    if getattr(args, "output_dir", None):
        cfg.output_dir = args.output_dir
    if getattr(args, "backbone", None):
        cfg.model.backbone = args.backbone
    return cfg


def _cmd_build_data(args: argparse.Namespace) -> int:
    from .data.builders import DataSpec, build_training_data
    from .training.trainer import export_training_data

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = _config_from_args(args)
    spec = DataSpec.from_dict(cfg.data, base_dir=Path(cfg.data_base_dir) if cfg.data_base_dir else None)
    data = build_training_data(spec)
    stats = data.stats()
    if not args.dry_run:
        export_training_data(data, args.output_dir or Path(cfg.output_dir) / "data")
    print(json.dumps(stats, indent=2))
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from .training.trainer import train

    cfg = _config_from_args(args)
    train(cfg)
    return 0


def _add_data_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="YAML config (see configs/). Other data flags are ignored.")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="Override a config value, e.g. --set training.lr=2e-5 (repeatable).")
    g = p.add_argument_group("quick data flags (no config file)")
    g.add_argument("--parallel", action="append", metavar="FILE",
                   help="(source, simplification) pairs.")
    g.add_argument("--graded", action="append", metavar="FILE",
                   help="(source, simplification, level) rows; higher level = simpler.")
    g.add_argument("--min-level-gap", type=float, default=None,
                   help="Minimum level difference for inter-simplification pairs (default 1).")
    g.add_argument("--ratings", action="append", metavar="FILE",
                   help="(source, candidate, rating) rows; needs --rating-scale.")
    g.add_argument("--rating-scale", type=float, nargs=2, metavar=("LOW", "HIGH"))
    g.add_argument("--preferences", action="append", metavar="FILE",
                   help="(source, chosen, rejected[, margin, weight]) rows.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simple-score",
        description="SIMPLE Score: reference-free text simplification metric with uncertainty.")
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("score", help="Score simplifications against their sources.")
    s.add_argument("--model", default="large",
                   help="small | base | large, a Hub repo id, or a local checkpoint dir.")
    s.add_argument("--input", help="File with source and candidate columns (jsonl/csv/tsv).")
    s.add_argument("--sources", help="Plain-text file, one source sentence per line.")
    s.add_argument("--candidates", help="Plain-text file, one simplification per line.")
    s.add_argument("--output", help="Write JSONL results here (default: stdout).")
    s.add_argument("--batch-size", type=int, default=32)
    s.add_argument("--device", default=None)
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=_cmd_score)

    t = sub.add_parser("train", help="Train a SIMPLE model on your data.")
    _add_data_flags(t)
    t.add_argument("--output-dir")
    t.add_argument("--backbone", help="e.g. microsoft/deberta-v3-base")
    t.set_defaults(func=_cmd_train)

    b = sub.add_parser("build-data",
                       help="Build (and optionally export) the training pairs without training.")
    _add_data_flags(b)
    b.add_argument("--output-dir", help="Where to write the JSONL files.")
    b.add_argument("--dry-run", action="store_true", help="Only print counts.")
    b.set_defaults(func=_cmd_build_data)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
