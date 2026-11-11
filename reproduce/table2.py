#!/usr/bin/env python3
"""Reproduce SIMPLE's rows of paper Table 2 (and the SIMPLE rows of Table 5).

Data is downloaded from the original repositories (nothing is redistributed here):
  * SimpEval 2022 (Maddela et al., 2023): github.com/Yao-Dou/LENS
  * ASSET human ratings (Alva-Manchego et al., 2020): github.com/facebookresearch/asset
    (only used so the adversarial candidates are generated in the same order as in the
    paper's evaluation; the reported numbers are on SimpEval 2022 only)

Reports, on SimpEval 2022:
  * pairwise accuracy and Kendall tau_all = 2 * acc - 1 over the 484 pairs whose mean
    ratings differ by >= 10 points
  * Pearson / Spearman between mu and the mean human rating (n = 360)
  * adversarial beat-rate: fraction of sources where the genuine simplification outscores
    a degenerate candidate (copy, truncation, shuffle, random, over-deletion), averaged
  * score distributions: genuine, reversed (complexification) and adversarial mu

    python reproduce/table2.py --model large            # released model from the Hub
    python reproduce/table2.py --model hub_export/simple-metric-large
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

from simple_metric import SIMPLE
from simple_metric.training.trainer import pearson, spearman

SIMPEVAL_URL = "https://raw.githubusercontent.com/Yao-Dou/LENS/master/data/simpeval_2022.csv"
ASSET_RATINGS_URL = ("https://raw.githubusercontent.com/facebookresearch/asset/main/"
                     "human_ratings/human_ratings.csv")
PAPER = {  # Table 2: acc, tau_all, pearson, spearman, beat-rate
    "small": (0.738, 0.475, 0.414, 0.407, 0.994),
    "base": (0.781, 0.562, 0.427, 0.448, 0.989),
    "large": (0.814, 0.628, 0.391, 0.454, 0.989),
}


def download(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def simpeval(cache: Path):
    rows = read_csv(download(SIMPEVAL_URL, cache / "simpeval_2022.csv"))
    for r in rows:
        r["composite"] = np.mean([float(r[f"rating_{i}"]) for i in (1, 2, 3)])
    pairs = []
    by_id = defaultdict(list)
    for r in rows:
        by_id[r["original_id"]].append(r)
    for key in sorted(by_id, key=lambda k: (float(k) if k.replace(".", "").isdigit() else k)):
        group = by_id[key]
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if abs(a["composite"] - b["composite"]) >= 10:
                    pairs.append((a["original"], a["generation"], b["generation"],
                                  a["composite"] > b["composite"]))
    return rows, pairs


def asset_pointwise(cache: Path) -> list[tuple[str, str]]:
    rows = read_csv(download(ASSET_RATINGS_URL, cache / "asset_human_ratings.csv"))
    return sorted({(r["original"], r["simplification"]) for r in rows})


def adversarials(sources: list[str], seed: int = 42) -> dict[str, list[str]]:
    """Same generator, seed and call order as the paper's evaluation."""
    rng = random.Random(seed)

    def truncate(text, ratio=0.5):
        words = text.split()
        return " ".join(words[: max(1, int(len(words) * ratio))])

    def shuffle(text):
        words = text.split()
        rng.shuffle(words)
        return " ".join(words)

    def over_delete(text, keep_ratio=0.35):
        words = text.split()
        content = [w for w in words if len(w) > 3]
        filler = [w for w in words if len(w) <= 3]
        kept = filler + rng.sample(content, min(max(1, int(len(content) * keep_ratio)), len(content)))
        order = {w: i for i, w in enumerate(words)}
        kept.sort(key=lambda w: order.get(w, 0))
        return " ".join(kept) if kept else words[0]

    shuffled = sources[:]
    rng.shuffle(shuffled)
    random_sents = [s if s != src else sources[(i + 1) % len(sources)]
                    for i, (s, src) in enumerate(zip(shuffled, sources))]
    return {
        "copy": list(sources),
        "truncation": [truncate(s) for s in sources],
        "shuffle": [shuffle(s) for s in sources],
        "random": random_sents,
        "over-delete": [over_delete(s) for s in sources],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", default="large", help="small | base | large | repo id | local dir")
    ap.add_argument("--cache", type=Path, default=Path("data/benchmarks"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--output", type=Path, help="Write results JSON here.")
    args = ap.parse_args()

    metric = SIMPLE.from_pretrained(args.model, device=args.device, batch_size=args.batch_size)
    rows, pairs = simpeval(args.cache)

    # --- pairwise ranking
    src = [p[0] for p in pairs]
    a = np.asarray(metric.score(src, [p[1] for p in pairs]).mu)
    b = np.asarray(metric.score(src, [p[2] for p in pairs]).mu)
    pref_a = np.asarray([p[3] for p in pairs])
    acc = float(((a > b) == pref_a).mean())

    # --- pointwise correlation
    sources = [r["original"] for r in rows]
    genuine_c = [r["generation"] for r in rows]
    human = [r["composite"] for r in rows]
    genuine = metric.score(sources, genuine_c)
    mu = np.asarray(genuine.mu)

    # --- adversarial beat-rate (generated over ASSET + SimpEval, reported on SimpEval)
    asset = asset_pointwise(args.cache)
    all_sources = [s for s, _ in asset] + sources
    n_asset = len(asset)
    adv = adversarials(all_sources)
    beat, adv_mu = {}, []
    for name, cands in adv.items():
        s = np.asarray(metric.score(sources, cands[n_asset:]).mu)
        beat[name] = float((s < mu).mean())
        adv_mu.append(s)
    adv_mu = np.concatenate(adv_mu)

    # --- complexification: swap source and simplification
    reversed_mu = np.asarray(metric.score(genuine_c, sources).mu)

    res = {
        "model": str(args.model),
        "n_pairs": len(pairs),
        "pairwise_accuracy": round(acc, 3),
        "tau_all": round(2 * acc - 1, 3),
        "n_pointwise": len(rows),
        "pearson": round(pearson(mu, human), 3),
        "spearman": round(spearman(mu, human), 3),
        "beat_rate": round(float(np.mean(list(beat.values()))), 3),
        "beat_rate_by_type": {k: round(v, 3) for k, v in beat.items()},
        # Genuine candidates identical to their source: for these the "copy" adversarial is
        # the same input, so the comparison is decided by float noise (~1e-6). This makes
        # the beat-rate vary by up to 0.002 across hardware/batching (paper: .989 for
        # base/large; this script typically gives .989-.991).
        "copy_ties": sum(s.strip() == c.strip() for s, c in zip(sources, genuine_c)),
        "distribution": {
            "genuine": [round(float(mu.mean()), 2), round(float(mu.std(ddof=1)), 2)],
            "complexification": [round(float(reversed_mu.mean()), 2),
                                 round(float(reversed_mu.std(ddof=1)), 2)],
            "adversarial": [round(float(adv_mu.mean()), 2), round(float(adv_mu.std(ddof=1)), 2)],
        },
        "sigma": {"min": round(float(np.min(genuine.sigma)), 3),
                  "median": round(float(np.median(genuine.sigma)), 3),
                  "max": round(float(np.max(genuine.sigma)), 3)},
    }
    print(json.dumps(res, indent=2))
    size = next((k for k in PAPER if k in str(args.model).lower()), None)
    if size:
        got = (res["pairwise_accuracy"], res["tau_all"], res["pearson"], res["spearman"],
               res["beat_rate"])
        print(f"paper  (SIMPLE-{size}): acc/tau/pearson/spearman/beat = {PAPER[size]}")
        print(f"this run           : acc/tau/pearson/spearman/beat = {got}")
    if args.output:
        args.output.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
