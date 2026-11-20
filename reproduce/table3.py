#!/usr/bin/env python3
"""Reproduce paper Table 3 (Pearson) and Table 9 (Kendall tau_b): correlation of each metric with
human ratings on Simplicity-DA (Alva-Manchego et al., 2021) and Structural Simplicity (Sulem et al., 2018).

Inputs
  reproduce/data/table3_scores.csv
      One row per rated pair (dataset, sent_id, sys_name) with the per-pair scores computed for
      the paper: SIMPLE-small/base/large, ΔSLE, LENS-SALSA, and the length/readability gains.
  github.com/feralvam/metaeval-simplification  (downloaded at run time; CC BY-NC-SA 4.0, not
      redistributed here): the human ratings, the sentences, and the reference-based metric scores
      (BLEU, SARI, BERTScore F1, iBLEU, SAMSA, word length), averaged over the reference sets.

Usage
  python reproduce/table3.py                    # print both tables, check them against the paper
  python reproduce/table3.py --json out.json    # also write every correlation with p-values
  python reproduce/table3.py --rescore-simple   # recompute the SIMPLE and gain columns (needs GPU/CPU + Hub access)
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr

HERE = Path(__file__).resolve().parent
SCORES = HERE / "data" / "table3_scores.csv"
BASE_URL = "https://raw.githubusercontent.com/feralvam/metaeval-simplification/main/data"
RATINGS = {
    "simplicity_DA": ["fluency", "meaning", "simplicity"],
    "structural_simplicity": ["grammaticality", "meaning", "simplicity", "structural_simplicity"],
}
# Row order and sources as in the paper. "metaeval:<col>" = precomputed by the benchmark authors
# (mean over reference sets); other names = columns of table3_scores.csv.
ROWS = [
    ("BLEU", "reference-based", {"*": "metaeval:bleu"}),
    ("SARI", "reference-based", {"*": "metaeval:sari"}),
    ("BERTScore", "reference-based", {"*": "metaeval:bertscore_F1"}),
    ("iBLEU", "reference-based", {"*": "metaeval:ibleu"}),
    ("SAMSA", "reference-free", {"*": "metaeval:samsa"}),
    ("ΔSLE", "reference-free", {"*": "delta_sle"}),
    ("LENS-SALSA", "reference-free", {"*": "lens_salsa"}),
    # As published: Simplicity-DA uses the word-length gain l(o) - l(s); Structural Simplicity
    # uses the benchmark's precomputed `word_length` column.
    ("Word length", "reference-free", {"simplicity_DA": "word_length_gain",
                                       "structural_simplicity": "metaeval:word_length"}),
    ("Sent. length gain", "reference-free", {"*": "sent_length_gain"}),
    ("FKGL", "reference-free", {"*": "fkgl_gain"}),
    ("FRE", "reference-free", {"*": "fre_gain"}),
    ("SIMPLE-small", "this work", {"*": "simple_small"}),
    ("SIMPLE-base", "this work", {"*": "simple_base"}),
    ("SIMPLE-large", "this work", {"*": "simple_large"}),
]
# Published values, for the check. Table 3 = Pearson, Table 9 = Kendall tau_b.
PAPER_T3 = {
    "simplicity_DA": {
        "BLEU": [.478, .653, .456], "SARI": [.379, .599, .341], "BERTScore": [.610, .764, .534],
        "iBLEU": [.482, .650, .470], "SAMSA": [.106, .148, .056], "ΔSLE": [-.111, -.311, -.011],
        "LENS-SALSA": [.657, .567, .573], "Word length": [-.013, -.084, -.016],
        "Sent. length gain": [-.182, -.380, -.038], "FKGL": [-.147, -.251, -.033], "FRE": [-.070, -.116, .012],
        "SIMPLE-small": [.576, .501, .606], "SIMPLE-base": [.624, .584, .680], "SIMPLE-large": [.665, .588, .695]},
    "structural_simplicity": {
        "BLEU": [.415, .796, -.363, -.365], "SARI": [.011, .406, -.252, -.301],
        "BERTScore": [.495, .830, -.306, -.325], "iBLEU": [.413, .783, -.341, -.345],
        "SAMSA": [.183, .460, -.321, -.284], "ΔSLE": [-.205, -.680, .595, .577],
        "LENS-SALSA": [.678, .403, .198, .170], "Word length": [-.232, .214, -.432, -.456],
        "Sent. length gain": [-.007, -.621, .679, .726], "FKGL": [-.292, -.562, .412, .407],
        "FRE": [-.251, -.475, .298, .261], "SIMPLE-small": [.296, -.142, .537, .467],
        "SIMPLE-base": [.439, .009, .483, .396], "SIMPLE-large": [.374, -.069, .531, .461]},
}
PAPER_T9 = {
    "simplicity_DA": {
        "BLEU": [.294, .444, .304], "SARI": [.201, .399, .207], "BERTScore": [.400, .586, .376],
        "iBLEU": [.301, .441, .319], "SAMSA": [.073, .126, .048], "ΔSLE": [-.066, -.206, .017],
        "LENS-SALSA": [.499, .404, .493], "Word length": [-.045, -.029, -.007],
        "Sent. length gain": [-.051, -.237, .009], "FKGL": [-.088, -.195, -.011], "FRE": [-.082, -.106, -.005],
        "SIMPLE-small": [.339, .292, .427], "SIMPLE-base": [.392, .367, .479], "SIMPLE-large": [.428, .370, .522]},
    "structural_simplicity": {
        "BLEU": [.329, .617, -.199, -.177], "SARI": [-.029, .217, -.145, -.186],
        "BERTScore": [.387, .667, -.169, -.160], "iBLEU": [.325, .601, -.182, -.162],
        "SAMSA": [.131, .369, -.245, -.206], "ΔSLE": [-.177, -.494, .386, .350],
        "LENS-SALSA": [.474, .246, .215, .201], "Word length": [-.158, .163, -.321, -.349],
        "Sent. length gain": [.005, -.457, .486, .530], "FKGL": [-.269, -.495, .296, .274],
        "FRE": [-.224, -.398, .194, .150], "SIMPLE-small": [.119, -.194, .410, .340],
        "SIMPLE-base": [.291, .007, .361, .281], "SIMPLE-large": [.189, -.131, .398, .327]},
}


def download(cache: Path) -> None:
    for name in RATINGS:
        for rel in (f"{name}.csv", f"metrics_per_sentence/metrics_{name}.csv"):
            dest = cache / Path(rel).name
            if not dest.exists():
                cache.mkdir(parents=True, exist_ok=True)
                print(f"downloading {BASE_URL}/{rel}")
                urllib.request.urlretrieve(f"{BASE_URL}/{rel}", dest)


def load(name: str, cache: Path, scores: pd.DataFrame | None) -> pd.DataFrame:
    """Ratings + reference-set-averaged metaeval metrics (+ our per-pair scores) for one benchmark."""
    rat = pd.read_csv(cache / f"{name}.csv")
    met = pd.read_csv(cache / f"metrics_{name}.csv")
    cols = [c for c in met.columns if c not in {"sent_id", "sys_name", "test_set", "sys_type"}]
    met = met.groupby(["sent_id", "sys_name"])[cols].mean().add_prefix("metaeval:").reset_index()
    df = rat.merge(met, on=["sent_id", "sys_name"], how="left")
    if scores is not None:
        ours = scores[scores.dataset == name].drop(columns="dataset")
        df = df.merge(ours, on=["sent_id", "sys_name"], how="left", validate="one_to_one")
    return df


def correlations(df: pd.DataFrame, name: str) -> dict:
    out = {}
    for row, kind, src in ROWS:
        col = src.get(name, src.get("*"))
        if col not in df or df[col].isna().all():
            continue
        x = df[col].to_numpy(float)
        out[row] = {"type": kind, "score_column": col, "ratings": {}}
        for rc in RATINGS[name]:
            y = df[rc].to_numpy(float)
            m = ~(np.isnan(x) | np.isnan(y))
            r, p = pearsonr(x[m], y[m])
            tau, tp = kendalltau(x[m], y[m])  # tau_b (handles ties in the 1-5 ratings)
            out[row]["ratings"][rc] = {"pearson": round(float(r), 4), "pearson_p": float(p),
                                       "kendall_tau_b": round(float(tau), 4), "kendall_p": float(tp),
                                       "n": int(m.sum())}
    return out


def print_and_check(results: dict) -> int:
    mismatches = 0
    for table, key, paper in (("Table 3 (Pearson)", "pearson", PAPER_T3),
                              ("Table 9 (Kendall tau_b)", "kendall_tau_b", PAPER_T9)):
        print(f"\n{table}   (* = p > 0.05;  ! = differs from the paper by more than 0.0015)")
        for name, rows in results.items():
            print(f"  {name}: " + "  ".join(f"{rc[:6]:>7s}" for rc in RATINGS[name]))
            for row, res in rows.items():
                cells = []
                for k, rc in enumerate(RATINGS[name]):
                    c = res["ratings"][rc]
                    pval = c["pearson_p" if key == "pearson" else "kendall_p"]
                    flag = "*" if pval > 0.05 else " "
                    ok = abs(c[key] - paper[name][row][k]) <= 0.0015
                    mismatches += not ok
                    cells.append(f"{c[key]:+.3f}{flag}{'' if ok else '!'}")
                print(f"    {row:18s} " + "  ".join(f"{s:>7s}" for s in cells))
    return mismatches


def rescore_simple(scores_path: Path, cache: Path) -> None:
    """Recompute the SIMPLE and length/readability-gain columns of the scores file."""
    import textstat

    from simple_score import SIMPLE

    def awl(t):
        w = [x for x in t.split() if x.isalpha()]
        return sum(map(len, w)) / len(w) if w else 0.0

    frames = []
    for name in RATINGS:
        df = pd.read_csv(cache / f"{name}.csv")
        src, cand = df.orig_sent.tolist(), df.simp_sent.tolist()
        f = pd.DataFrame({"dataset": name, "sent_id": df.sent_id, "sys_name": df.sys_name})
        f["word_length_gain"] = [awl(o) - awl(s) for o, s in zip(src, cand)]
        f["sent_length_gain"] = [len(o.split()) - len(s.split()) for o, s in zip(src, cand)]
        f["fkgl_gain"] = [(textstat.flesch_kincaid_grade(o) - textstat.flesch_kincaid_grade(s))
                          / max(abs(textstat.flesch_kincaid_grade(o)), 1.0) for o, s in zip(src, cand)]
        f["fre_gain"] = [textstat.flesch_reading_ease(s) - textstat.flesch_reading_ease(o)
                         for o, s in zip(src, cand)]
        frames.append(f)
    new = pd.concat(frames, ignore_index=True)
    for size in ("small", "base", "large"):
        metric = SIMPLE.from_pretrained(size)
        vals = []
        for name in RATINGS:
            df = pd.read_csv(cache / f"{name}.csv")
            vals += metric.score(df.orig_sent.tolist(), df.simp_sent.tolist()).mu
        new[f"simple_{size}"] = vals
    if scores_path.exists():  # keep ΔSLE / LENS-SALSA, which need their own packages
        old = pd.read_csv(scores_path)
        keep = [c for c in old.columns if c not in new.columns or c in ("dataset", "sent_id", "sys_name")]
        new = new.merge(old[keep], on=["dataset", "sent_id", "sys_name"], how="left")
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(scores_path, index=False, float_format="%.6g")
    print(f"wrote {scores_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scores", type=Path, default=SCORES)
    ap.add_argument("--cache", type=Path, default=Path("data/benchmarks/metaeval"))
    ap.add_argument("--json", type=Path, help="Write all correlations (with p-values) here.")
    ap.add_argument("--rescore-simple", action="store_true")
    args = ap.parse_args()
    download(args.cache)
    if args.rescore_simple:
        rescore_simple(args.scores, args.cache)
    scores = pd.read_csv(args.scores) if args.scores.exists() else None
    results = {name: correlations(load(name, args.cache, scores), name) for name in RATINGS}
    bad = print_and_check(results)
    print(f"\n{'all cells match the paper' if bad == 0 else f'{bad} cell(s) differ from the paper'}"
          " (tolerance 0.0015)")
    missing = [row for row, _, _ in ROWS if any(row not in results[name] for name in RATINGS)]
    if missing:
        print(f"rows not reproduced (no scores in {args.scores.name}): {', '.join(missing)}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
