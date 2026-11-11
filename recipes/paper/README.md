# Paper recipe

How the released SIMPLE models were trained, and how to rerun that with this package.

## 1. Data

```bash
python recipes/paper/build_paper_data.py \
    --newsela-dir /path/to/newsela_data_share-20150302 \
    --output-dir data/paper
```

The script downloads ASSET, OneStopEnglish and SimpEval (PAST and 2022) from their original
repositories and reads Newsela from your licensed copy. It writes plain TSV files and
`data/paper/data.yaml`, which describes them with the generic data kinds:

| file | kind | content |
|---|---|---|
| `asset_valid_parallel.tsv` | parallel | ASSET validation, 2,000 sources × 10 simplifications (19,875 rows) |
| `onestop_parallel.tsv` | parallel | OneStopEnglish ADV→ELE, ADV→INT, INT→ELE aligned sentences |
| `newsela_graded.tsv` | graded (min level gap 2) | Newsela aligned sentences V0–V4 (141,582 rows) |
| `asset_pairwise.tsv` | preferences | ASSET human pairwise comparisons, ≥ 2 of 3 aspects agree (476) |
| `onestop_cross_level.tsv` | preferences | OneStopEnglish ADV source: ELE ≻ INT (361) |
| `simpeval_past_ratings.tsv` | ratings (0–100) | SimpEval PAST, mean of 5 ratings per output |
| `monitor/simpeval2022_*.tsv` | monitor | SimpEval 2022 pairs / ratings (the Table 2 test set) |

`simple-metric build-data --config configs/paper/simple-base.yaml --dry-run` reproduces the
original build's pair counts by type: 167,469 original-simplification, mirror and
random-sentence pairs each, and 41,713 inter-simplification pairs. Human-preference pairs
are 20,233, against 20,257 in the original build. One SimpEval PAST output is empty; the
original pipeline read it as the string `"nan"` and paired it with other outputs, while
this recipe drops it. The regression set has ≈ 29.6k examples. The exact number depends on
which 5,000 sentences are sampled for the synthetic anchors.

## 2. Training

| config | backbone | batch | lr | data per epoch |
|---|---|---|---|---|
| `configs/paper/simple-small.yaml` | deberta-v3-small | 32 | 2e-5 | all |
| `configs/paper/simple-base.yaml` | deberta-v3-base | 16 | 1e-5 | random half |
| `configs/paper/simple-large.yaml` | deberta-v3-large | 8 | 1e-5 | random quarter |

All three configs share the rest: 5 epochs, λ = 0.7, human-preference pair weight 3, max
length 512, AdamW with 6% linear warm-up.

**Why "half" and "quarter".** The paper's base and large runs were launched on 2 and 4 GPUs,
but the original training script never synchronised gradients across GPUs. The released
models are therefore what a single GPU learned: batch 16 (base) or 8 (large), seeing a
random 1/2 or 1/4 of the pairs each epoch. `epoch_fraction` reproduces that on one GPU.
The Appendix A ablations ran the same way on 2 GPUs, with max length 256 and regression
batch 16; see `configs/ablations/`.

This package's trainer synchronises gradients correctly, so running a config with
`torchrun --nproc_per_node=N` trains with batch `batch_size × N`.

**Checkpoint selection.** The released checkpoints were selected by pairwise accuracy on
SimpEval 2022, which is also the Table 2 evaluation set. The paper configs keep that
(`select_on: monitor:simpeval2022`) so that reruns behave like the released models. For new
models use the default, `select_on: val_inter`: accuracy on held-out pairs that compare two
simplifications.

## 3. Evaluation

```bash
python reproduce/table2.py --model large        # or small / base / a local checkpoint dir
```

This reports pairwise accuracy, Kendall τ, Pearson/Spearman, the adversarial beat-rate, and
the Table 5 score distributions. With the released models it reproduces Table 2 to the
third decimal. The one exception is that the beat-rate of base and large can come out as
.991 instead of .989. Four SimpEval 2022 "simplifications" are identical to their source;
for those the copy adversarial is the same input, and the comparison is decided by
floating-point noise.
