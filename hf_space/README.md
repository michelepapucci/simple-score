---
title: SIMPLE Score
emoji: 🧩
colorFrom: blue
colorTo: green
sdk: static
app_file: index.html
pinned: false
license: mit
tags:
  - evaluate
  - metric
  - text-simplification
description: >-
  SIMPLE Score: a reference-free learned metric for text simplification that also reports its
  own uncertainty.
---

# Metric Card for SIMPLE Score

## Metric description

SIMPLE Score (SImplicity Metric via Preference LEarning) scores a candidate simplification
against its source sentence, without references. It returns:

- **score** `mu` in (-1, 1): +1 = highly successful simplification, 0 = verbatim copy of
  the source, -1 = unrelated or degenerate text. Negative scores flag unsuccessful
  simplifications, including complexifications.
- **uncertainty** `sigma` (> 0.05): the model's confidence in that score; lower = more
  confident. Predictions with low sigma agree more often with human judgements.

SIMPLE Score is a DeBERTa-v3 cross-encoder trained with a Thurstone pairwise preference loss and
calibrated on human quality ratings with a Gaussian likelihood. Three sizes are available.

This Space's page runs an in-browser demo (SIMPLE-small with 8-bit weights, via ONNX Runtime
Web); the model is downloaded only when a visitor starts the demo.

## How to use

```python
import evaluate

simple_score = evaluate.load("mpapucci/simple-score")            # SIMPLE-large (default)
# simple_score = evaluate.load("mpapucci/simple-score", "base")  # or "small"
out = simple_score.compute(
    sources=["The feline perched upon the mat in a state of repose."],
    predictions=["The cat sat on the mat."],
)
print(out["simple_score"], out["scores"], out["sigmas"])
```

This needs `pip install simple-score`. The package can also be used directly, which avoids
the `evaluate` dependency:

```python
from simple_score import SIMPLE
metric = SIMPLE.from_pretrained("large")
metric.score(sources, predictions)
```

### Inputs

- `sources` (list of str): original sentences.
- `predictions` (list of str): candidate simplifications, aligned with `sources`.
- `batch_size` (int, default 32), `device` (str, default CUDA when available).

### Output values

- `simple_score` (float): mean score.
- `scores` (list of float): per-example score `mu`.
- `sigmas` (list of float): per-example uncertainty `sigma`.

## Results

SimpEval 2022 (Maddela et al., 2023): pairwise accuracy on 484 human preference pairs, and
Pearson/Spearman correlation with 360 human ratings.

| Model | Pairwise acc. | Kendall tau | Pearson | Spearman |
|---|---|---|---|---|
| SIMPLE-small | 0.738 | 0.475 | 0.414 | 0.407 |
| SIMPLE-base | 0.781 | 0.562 | 0.427 | 0.448 |
| SIMPLE-large | 0.814 | 0.628 | 0.391 | 0.454 |

## Limitations and bias

- Sentence-level and English only; trained mostly on news and Wikipedia simplifications.
- It inherits the notion of simplification quality of its training data (ASSET,
  OneStopEnglish, Newsela, SimpEval PAST).
- Insertion/elaboration-heavy rewrites are rare in the training data; both the score and
  sigma are less reliable on them.
- Sigma is a model-confidence signal, not an estimate of human disagreement.

## Citation

Citation will be added upon publication.
