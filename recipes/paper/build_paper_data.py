#!/usr/bin/env python3
"""Rebuild the training data of the paper's SIMPLE models in simple-score's generic format.

Public sources are downloaded from their original repositories; Newsela is read from
your own licensed copy (``--newsela-dir``; it cannot be redistributed). The script writes
plain TSV files plus ``data.yaml`` (the data spec) into ``--output-dir``:

  asset_valid_parallel.tsv       ASSET validation: 2,000 sources x 10 crowd simplifications
  asset_pairwise.tsv             ASSET human pairwise comparisons (majority of 3 aspects)
  onestop_parallel.tsv           OneStopEnglish aligned ADV->ELE, ADV->INT, INT->ELE pairs
  onestop_cross_level.tsv        OneStopEnglish (ADV: ELE > INT) inter-simplification pairs
  newsela_graded.tsv             Newsela aligned sentences with reading level (V0..V4)
  simpeval_past_ratings.tsv      SimpEval PAST: 100 sources x 24 systems, mean of 5 ratings
  monitor/simpeval2022_*.tsv     SimpEval 2022 test pairs / ratings (reported each epoch)

Then train with, e.g.:

    simple-score train --config configs/paper/simple-base.yaml

Sources:
  ASSET (Alva-Manchego et al., 2020)        github.com/facebookresearch/asset
  OneStopEnglish (Vajjala and Lucic, 2018)  github.com/nishkalavallabhi/OneStopEnglishCorpus
  SimpEval PAST / 2022 (Maddela et al., 2023) github.com/Yao-Dou/LENS
  Newsela (Xu et al., 2015)                 newsela.com/data (license required)
"""

from __future__ import annotations

import argparse
import csv
import urllib.request
from collections import defaultdict
from pathlib import Path

ASSET = "https://raw.githubusercontent.com/facebookresearch/asset/main"
ONESTOP = ("https://raw.githubusercontent.com/nishkalavallabhi/OneStopEnglishCorpus/master/"
           "Sentence-Aligned")
LENS = "https://raw.githubusercontent.com/Yao-Dou/LENS/master/data"
NEWSELA_ALIGNED = "newsela_articles_20150302.aligned.sents.txt"


def download(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


def clean(text) -> str:
    return " ".join(str(text).replace("\t", " ").splitlines()).strip()


def write_tsv(path: Path, header: list[str], rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(clean(v) for v in row) + "\n")
    print(f"  wrote {len(rows):>8} rows -> {path}")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ----------------------------------------------------------------------------- ASSET


def asset_parallel(raw: Path) -> list[tuple]:
    base = f"{ASSET}/dataset"
    sources = download(f"{base}/asset.valid.orig", raw / "asset.valid.orig").read_text(
        encoding="utf-8").splitlines()
    refs = [download(f"{base}/asset.valid.simp.{k}", raw / f"asset.valid.simp.{k}").read_text(
        encoding="utf-8").splitlines() for k in range(10)]
    rows = []
    for i, src in enumerate(sources):
        src = src.strip()
        if not src:
            continue
        for k in range(10):
            simp = refs[k][i].strip()
            if simp and simp != src:
                rows.append((src, simp))
    return rows


def asset_pairwise(raw: Path, min_agreement: int = 2) -> list[tuple]:
    name = "asset_pairwise_comparisons_with_turkcorpus_and_hsplit.csv"
    rows = read_csv(download(f"{ASSET}/human_ratings/{name}", raw / name))
    votes: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        key = (r["source"], r["asset_simplification"], r["other_simplification"])
        winner = r["winner"]
        votes[key]["other" if winner in ("turkcorpus", "hsplit") else winner] += 1
    out = []
    for (src, asset_s, other_s), v in votes.items():
        if " ".join(asset_s.split()) == " ".join(other_s.split()):
            continue
        n_a, n_o = v.get("asset", 0), v.get("other", 0)
        if n_a >= min_agreement and n_a > n_o:
            out.append((src, asset_s, other_s, "inter"))
        elif n_o >= min_agreement and n_o > n_a:
            out.append((src, other_s, asset_s, "inter"))
    return out


# ----------------------------------------------------------------------------- OneStopEnglish


def _onestop_pairs(path: Path) -> list[tuple[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    pairs, i = [], 0
    while i < len(lines):
        if not lines[i].strip() or lines[i].strip().startswith("*******"):
            i += 1
            continue
        a = lines[i].strip()
        i += 1
        if i >= len(lines):
            break
        b = lines[i].strip()
        i += 1
        if a and b:
            pairs.append((a, b))
        if i < len(lines) and lines[i].strip().startswith("*******"):
            i += 1
    return pairs


def onestop(raw: Path) -> tuple[list[tuple], list[tuple]]:
    files = {n: download(f"{ONESTOP}/{n}.txt", raw / f"{n}.txt") for n in ("ADV-ELE", "ADV-INT", "ELE-INT")}
    adv_ele, adv_int, ele_int = (_onestop_pairs(files[n]) for n in ("ADV-ELE", "ADV-INT", "ELE-INT"))
    parallel = [(adv, ele) for adv, ele in adv_ele]
    parallel += [(adv, int_) for adv, int_ in adv_int]
    parallel += [(int_, ele) for ele, int_ in ele_int]
    # Cross-level: ADV sentences aligned in both files -> (ADV: ELE > INT)
    to_ele = {a.lower().strip(): b for a, b in adv_ele}
    to_int = {a.lower().strip(): b for a, b in adv_int}
    original_case = {a.lower().strip(): a for a, _ in adv_ele}
    cross = []
    for key in sorted(set(to_ele) & set(to_int)):
        ele, int_ = to_ele[key], to_int[key]
        if " ".join(ele.split()) != " ".join(int_.split()):
            cross.append((original_case[key], ele, int_, "inter"))
    return parallel, cross


# ----------------------------------------------------------------------------- Newsela


def newsela(newsela_dir: Path) -> list[tuple]:
    candidates = [newsela_dir / NEWSELA_ALIGNED, *(c / NEWSELA_ALIGNED for c in newsela_dir.iterdir()
                                                    if c.is_dir())]
    path = next((c for c in candidates if c.exists()), None)
    if path is None:
        raise SystemExit(f"Could not find {NEWSELA_ALIGNED} under {newsela_dir}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) != 5:
            continue
        _doc, vc, vs, comp, simp = parts
        comp, simp = comp.strip(), simp.strip()
        if comp and simp and int(vc[1]) < int(vs[1]) and comp != simp:
            # group = source version: the same sentence text at two source versions is
            # treated as two sources, as in the paper.
            rows.append((vc, comp, simp, int(vs[1])))
    return rows


# ----------------------------------------------------------------------------- SimpEval


def simpeval_past(raw: Path) -> list[tuple]:
    rows = read_csv(download(f"{LENS}/simpeval_past.csv", raw / "simpeval_past.csv"))
    out = []
    for r in rows:
        src, cand = r["original"].strip(), r["generation"].strip()
        if src and cand:
            mean = sum(float(r[f"rating_{i}"]) for i in range(1, 6)) / 5.0
            out.append((src, cand, f"{mean:.6g}"))
    return out


def simpeval_2022(raw: Path) -> tuple[list[tuple], list[tuple]]:
    rows = read_csv(download(f"{LENS}/simpeval_2022.csv", raw / "simpeval_2022.csv"))
    by_id: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        r["composite"] = sum(float(r[f"rating_{i}"]) for i in (1, 2, 3)) / 3.0
        by_id[r["original_id"]].append(r)
    pairs = []
    for group in by_id.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if abs(a["composite"] - b["composite"]) >= 10:
                    hi, lo = (a, b) if a["composite"] > b["composite"] else (b, a)
                    pairs.append((a["original"], hi["generation"], lo["generation"]))
    ratings = [(r["original"], r["generation"], f"{r['composite']:.6g}") for r in rows]
    return pairs, ratings


DATA_YAML = """\
# Training data of the paper's SIMPLE models, in simple-score's generic format.
# Written by recipes/paper/build_paper_data.py. Paths are relative to this file.
parallel:
  - {{path: asset_valid_parallel.tsv, name: asset-valid}}
  - {{path: onestop_parallel.tsv, name: onestop-direct}}
graded:
{newsela}
preferences:
  - {{path: asset_pairwise.tsv, name: asset-pairwise}}
  - {{path: onestop_cross_level.tsv, name: onestop-cross-level}}
ratings:
  - {{path: simpeval_past_ratings.tsv, name: simpeval-past, scale: [0, 100],
      min_gap: 0.15, max_margin: 0.9}}
type_weights: {{rating_pair: 3.0}}
augmentation:
  mirror: true
  random_sentence: true
  copy_anchors: 5000
  adversarial_anchors: 5000
  rating_anchors: true
validation:
  fraction: 0.1
  # The paper split pairs at random. "grouped" (the default) holds out whole source
  # sentences, which avoids train/validation leakage.
  split: random
seed: 42
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=Path("data/paper"))
    ap.add_argument("--raw-dir", type=Path, default=None,
                    help="Where downloads are cached (default: <output-dir>/raw).")
    ap.add_argument("--newsela-dir", type=Path, default=None,
                    help="Extracted Newsela release (newsela_data_share-20150302).")
    args = ap.parse_args()
    out = args.output_dir
    raw = args.raw_dir or out / "raw"

    print("ASSET")
    write_tsv(out / "asset_valid_parallel.tsv", ["source", "simplification"],
              asset_parallel(raw / "asset"))
    write_tsv(out / "asset_pairwise.tsv", ["source", "chosen", "rejected", "type"],
              asset_pairwise(raw / "asset"))
    print("OneStopEnglish")
    parallel, cross = onestop(raw / "onestop")
    write_tsv(out / "onestop_parallel.tsv", ["source", "simplification"], parallel)
    write_tsv(out / "onestop_cross_level.tsv", ["source", "chosen", "rejected", "type"], cross)
    print("SimpEval")
    write_tsv(out / "simpeval_past_ratings.tsv", ["source", "candidate", "rating"],
              simpeval_past(raw / "simpeval"))
    pairs, ratings = simpeval_2022(raw / "simpeval")
    write_tsv(out / "monitor" / "simpeval2022_pairs.tsv", ["source", "chosen", "rejected"], pairs)
    write_tsv(out / "monitor" / "simpeval2022_ratings.tsv", ["source", "candidate", "rating"],
              ratings)

    if args.newsela_dir:
        print("Newsela")
        write_tsv(out / "newsela_graded.tsv", ["group", "source", "simplification", "level"],
                  newsela(args.newsela_dir))
        newsela_entry = "  - {path: newsela_graded.tsv, name: newsela, min_level_gap: 2}"
    else:
        print("Newsela: skipped (no --newsela-dir); the paper models were trained with it.")
        newsela_entry = "  []"
    (out / "data.yaml").write_text(DATA_YAML.format(newsela=newsela_entry))
    print(f"Data spec: {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
