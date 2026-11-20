# SIMPLE Score: SImplicity Metric via Preference LEarning

SIMPLE Score is a **reference-free** learned metric for sentence simplification. For each
source sentence and candidate simplification it returns:

- **`mu`**, a quality score in (-1, 1):
  - **+1**: a highly successful simplification
  - **0**: a verbatim copy of the source
  - **-1**: unrelated or degenerate text

  Negative scores flag unsuccessful simplifications, including complexifications.
- **`sigma`**, the model's uncertainty about that score (> 0.05). Low-sigma scores agree
  with human judgements more often, so sigma tells you which scores to double-check.

SIMPLE is a DeBERTa-v3 cross-encoder trained on preferences between simplifications
(Thurstone's model) and calibrated on human quality ratings. This repository has:

- the `simple-score` Python package and command line tool;
- the full training code, which reads plain data files (no dataset-specific code needed);
- the recipe and configs used to train the paper's models;
- scripts to reproduce the paper's results.

| Model | Hub | SimpEval 2022 pairwise acc. | Pearson | Spearman |
|---|---|---|---|---|
| SIMPLE-small | `mpapucci/simple-score-small` | 0.738 | 0.414 | 0.407 |
| SIMPLE-base | `mpapucci/simple-score-base` | 0.781 | 0.427 | 0.448 |
| SIMPLE-large | `mpapucci/simple-score-large` | 0.814 | 0.391 | 0.454 |

**Try it without installing anything:** the [SIMPLE Score Space](https://huggingface.co/spaces/mpapucci/simple-score)
runs SIMPLE-small in your browser. The model is downloaded only when you start the demo.

## Installation

```bash
pip install simple-score            # scoring
pip install "simple-score[train]"   # + training
```

Python ≥ 3.10 is required, and the package works with transformers 4.x and 5.x.

## Scoring

```python
from simple_score import SIMPLE

metric = SIMPLE.from_pretrained("large")   # "small", "base", "large", a Hub id or a local dir
out = metric.score(
    sources=["The feline perched upon the mat in a state of repose."],
    candidates=["The cat sat on the mat."],
)
out.mu       # [0.82...]   quality score per example
out.sigma    # [0.05...]   uncertainty per example
out.mean     # corpus-level score

# Probability that candidate A is a better simplification than candidate B (Thurstone)
metric.preference_prob(source, candidate_a, candidate_b)
```

The model downloads on first use and runs on GPU when one is available
(`device="cpu"` forces CPU).

From the command line:

```bash
simple-score score --model large --sources src.txt --candidates system.txt --output scores.jsonl
simple-score score --model base --input outputs.csv   # any file with source + candidate columns
```

With [🤗 Evaluate](https://huggingface.co/docs/evaluate):

```python
import evaluate
simple_score = evaluate.load("mpapucci/simple-score", "large")
simple_score.compute(sources=sources, predictions=predictions)   # {"simple_score", "scores", "sigmas"}
```

### Reading the scores

- **Comparing systems:** compare mean `mu`. Inspect the examples with negative `mu`; they
  are copies, complexifications or broken outputs.
- **`sigma` is model confidence**, not a measure of human disagreement. Most predictions
  sit near the 0.05 floor. Examples in the high-`sigma` tail deserve a human look.
- **Scope:** SIMPLE is trained on English, sentence-level news and Wikipedia
  simplification. It is less reliable on rewrites that mostly add or elaborate content.

## Training on your own data

The data can't be redistributed, so the training code makes no assumptions about where it
came from. You list your files by **what kind of data they hold**, and SIMPLE derives the
training signal described in the paper:

| kind | columns | what SIMPLE derives |
|---|---|---|
| `parallel` | `source`, `simplification` | simplification ≻ source; a mirror pair (copying the simple sentence ≻ complexifying it); simplification ≻ unrelated sentence |
| `graded` | `source`, `simplification`, `level` (higher = simpler), optional `group` | as `parallel`, plus more-simplified ≻ less-simplified for the same source |
| `ratings` | `source`, `candidate`, `rating` (+ the rating scale) | regression targets on (-1, 1); human-preference pairs with rating margins; copy, flipped and unrelated anchors |
| `preferences` | `source`, `chosen`, `rejected`, optional `margin`, `weight`, `type` | used as given |

**Supported files**
- Formats: `.jsonl`, `.json`, `.csv`, `.tsv` (quotes allowed in sentences), `.parquet`, gzipped
  versions of these, line-aligned `.txt` files, or a Hugging Face Hub dataset.
- Column names: common aliases are recognised (`original`, `complex`, `simple`, `score`,
  `positive`/`negative`, …). Anything else can be mapped explicitly.

In addition, synthetic regression anchors teach the bottom half of the scale. They come
from your training sentences: copies score 0, shuffled/truncated/word-dropped sentences
score between −0.8 and −0.4, and unrelated sentences score −1.

**Which kinds do you need?** Any one is enough to start.
- A parallel corpus alone gives a metric that knows the direction of simplification and
  rejects degenerate outputs.
- Human ratings calibrate the scale and give the best agreement with human judgements.

### Quick start, no config file

```bash
simple-score train \
    --parallel my_corpus.tsv \
    --ratings my_ratings.csv --rating-scale 1 5 \
    --backbone microsoft/deberta-v3-base \
    --output-dir runs/my-simple
```

### With a config file

```yaml
# my_config.yaml
model:
  backbone: microsoft/deberta-v3-base
training:
  epochs: 5
  batch_size: 16
  lr: 1.0e-5
data:
  parallel:
    - {path: data/wiki_pairs.tsv, columns: {source: complex, simplification: simple}}
    - files: {source: data/train.src, simplification: data/train.tgt}   # line-aligned text
  graded:
    - {path: data/levels.csv, min_level_gap: 2}
  ratings:
    - {path: data/human_scores.jsonl, scale: [0, 100]}
  preferences:
    - {hf: some-user/simplification-preferences, split: train}
  validation: {fraction: 0.1}       # held out by source sentence; or {path: my_val.tsv}
monitor:                            # extra held-out sets reported every epoch
  - {path: data/test_pairs.tsv, kind: pairwise, name: test}
output_dir: runs/my-simple
```

```bash
simple-score build-data --config my_config.yaml --dry-run   # counts of every derived pair type
simple-score train --config my_config.yaml
simple-score train --config my_config.yaml --set training.lr=2e-5   # override any value
torchrun --nproc_per_node=4 -m simple_score.cli train --config my_config.yaml   # multi-GPU
```

`build-data` without `--dry-run` writes the derived pairs as JSONL, so you can check exactly
what the model will see. A run's output directory holds:
- `best/`, the checkpoint selected on validation accuracy between simplifications. It
  loads with `SIMPLE.from_pretrained("runs/my-simple/best")`.
- `final/` and `epoch_N/`.
- `metrics.jsonl`, `data_stats.json` and the resolved `config.json`.

`examples/` holds a runnable toy configuration. All options, with defaults, are in
[`src/simple_score/training/config.py`](src/simple_score/training/config.py) and
[`src/simple_score/data/builders.py`](src/simple_score/data/builders.py).

The same steps from Python:

```python
from simple_score.training import config_from_dict, train
cfg = config_from_dict({"model": {"backbone": "microsoft/deberta-v3-base"},
                        "data": {"parallel": ["pairs.tsv"]}, "output_dir": "runs/x"})
train(cfg)
```

## Reproducing the paper

```bash
# 1. Rebuild the training data from the original sources (Newsela needs your own license)
python recipes/paper/build_paper_data.py --newsela-dir /path/to/newsela_data_share-20150302

# 2. Train (see the notes in each config)
simple-score train --config configs/paper/simple-base.yaml
simple-score train --config configs/ablations/bt-mse.yaml    # Appendix A ablations

# 3. Evaluate: SIMPLE rows of Table 2 (and Table 5)
python reproduce/table2.py --model large
```

The recipe rebuilds the paper's training pairs; the counts match the original build. See
[`recipes/paper/README.md`](recipes/paper/README.md) for the details and for how the
paper's runs map to the configs.

## Citation

Citation will be added upon publication.

## License

The code and models are released under the MIT license. The training data is not
distributed here; each source dataset keeps its own license.
