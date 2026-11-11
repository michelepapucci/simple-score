#!/usr/bin/env python3
"""Write model cards, and upload the models and the `evaluate` Space to the Hub.

Every repository this script creates is PRIVATE, and it refuses to upload to a repository
that is public. Making anything public is a manual step on the Hub (Settings -> Change
visibility) once the release is approved. Needs `huggingface-cli login` with a write token.

    # write README.md model cards into hub_export/simple-metric-{small,base,large}
    python scripts/prepare_hub_release.py --cards

    # upload the converted models (writes the cards first)
    python scripts/prepare_hub_release.py --upload-models

    # upload the Space: a static page with the in-browser demo, plus simple.py for
    # `evaluate.load("mpapucci/simple")`. Export the browser model first:
    #   python scripts/export_onnx.py --model small --output-dir hub_export/onnx-small
    python scripts/prepare_hub_release.py --upload-space

The Space is static (free to host). Its demo downloads SIMPLE-small (int8 ONNX) into the
visitor's browser only when they press "Load the demo".
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAMESPACE = "mpapucci"
SPACE_ID = f"{NAMESPACE}/simple"

MODELS = {
    "small": dict(backbone="microsoft/deberta-v3-small", lr="2e-5", batch=32, fraction="all",
                  t2=(0.738, 0.475, 0.414, 0.407, 0.994),
                  t3=(".576", ".501", ".606", ".296", "-.142", ".537", ".467"),
                  t5=("+.58 (.21)", "-.64 (.20)", "-.56 (.35)")),
    "base": dict(backbone="microsoft/deberta-v3-base", lr="1e-5", batch=16, fraction="a random half",
                 t2=(0.781, 0.562, 0.427, 0.448, 0.989),
                 t3=(".624", ".584", ".680", ".439", ".009", ".483", ".396"),
                 t5=("+.64 (.26)", "-.67 (.20)", "-.54 (.36)")),
    "large": dict(backbone="microsoft/deberta-v3-large", lr="1e-5", batch=8,
                  fraction="a random quarter",
                  t2=(0.814, 0.628, 0.391, 0.454, 0.989),
                  t3=(".665", ".588", ".695", ".374", "-.069", ".531", ".461"),
                  t5=("+.66 (.25)", "-.68 (.19)", "-.56 (.34)")),
}

CARD = """\
---
license: mit
language:
  - en
library_name: simple-metric
base_model: {backbone}
tags:
  - text-simplification
  - evaluation
  - metric
  - reference-free
---

# SIMPLE-{size}

**SIMPLE** (SImplicity Metric via Preference LEarning) is a reference-free learned metric
for sentence simplification. For a source sentence and a candidate simplification it returns

- a **score** `mu` in (-1, 1): +1 = highly successful simplification, 0 = verbatim copy of
  the source, -1 = unrelated or degenerate text (negative = unsuccessful, including
  complexifications);
- an **uncertainty** `sigma` (> 0.05): the model's confidence in that score. Lower sigma
  means more reliable; it is a confidence signal, not an estimate of human disagreement.

This is the `{size}` model ({backbone} cross-encoder). Sizes: small, base, large.

## Usage

```bash
pip install simple-metric
```

```python
from simple_metric import SIMPLE

metric = SIMPLE.from_pretrained("{size}")          # = "{repo}"
out = metric.score(
    ["The feline perched upon the mat in a state of repose."],
    ["The cat sat on the mat."],
)
print(out.mu, out.sigma)
```

From the command line: `simple-metric score --model {size} --sources src.txt --candidates sys.txt`.
With Hugging Face `evaluate`: `evaluate.load("{space}", "{size}")`.

## Results

SimpEval 2022 (Maddela et al., 2023): pairwise accuracy and Kendall tau over 484 human
preference pairs; Pearson/Spearman with 360 human ratings; adversarial beat-rate (genuine
simplification scored above a copy / truncation / shuffle / random sentence /
over-deletion of the source, averaged).

| Pairwise acc. | Kendall tau | Pearson | Spearman | Beat-rate |
|---|---|---|---|---|
| {t2[0]:.3f} | {t2[1]:.3f} | {t2[2]:.3f} | {t2[3]:.3f} | {t2[4]:.3f} |

Pearson correlation with human ratings on Simplicity-DA (Alva-Manchego et al., 2021) and
Structural Simplicity (Sulem et al., 2018):

| SDA fluency | SDA meaning | SDA simplicity | SS grammaticality | SS meaning | SS simplicity | SS structural |
|---|---|---|---|---|---|---|
| {t3[0]} | {t3[1]} | {t3[2]} | {t3[3]} | {t3[4]} | {t3[5]} | {t3[6]} |

Mean (std) score on SimpEval 2022: genuine simplifications {t5[0]}, complexifications
(source and simplification swapped) {t5[1]}, adversarial candidates {t5[2]}.

The checkpoint was selected by pairwise accuracy on SimpEval 2022, the same data as the
first table, so those numbers are optimistic; the other benchmarks were not used for
selection.

## Training

A cross-encoder over `[CLS] source [SEP] candidate [SEP]` with two linear heads,
`mu = tanh(W h)` and `sigma = softplus(W' h) + 0.05`, trained with

- a Thurstone (Case III) pairwise loss, `-log Phi((mu_pos - mu_neg - m) / sqrt(sigma_pos^2 + sigma_neg^2))`,
  on 507,940 preference pairs built from ASSET, OneStopEnglish, Newsela and SimpEval PAST
  (original vs. simplification, more vs. less simplified, human-preferred vs. dispreferred
  with rating margin `m`, simplification vs. unrelated sentence);
- a Gaussian negative log-likelihood (weight 0.7) on 2,400 SimpEval PAST human ratings plus
  synthetic anchors (copies = 0, degenerate rewrites < 0, unrelated text = -1).

Hyperparameters: AdamW, learning rate {lr}, batch {batch}, 5 epochs over {fraction} of the
pairs per epoch, linear warmup over 6% of steps, max length 512. The training code and a
recipe to rebuild the data from the original sources are in the
[simple-metric repository](https://github.com/michelepapucci/simple-metric).

## Limitations

- Sentence-level, English, mostly news and Wikipedia; behaviour may differ in other
  domains or for other reader populations.
- Insertion/elaboration-heavy simplifications are rare in the training data; both `mu` and
  `sigma` are less reliable on them.
- `sigma` is reasonably but not perfectly calibrated (temperature scaling helps).

## Citation

Citation will be added upon publication.
"""


def _ensure_private(api, repo_id: str, repo_type: str = "model", **create_kwargs) -> None:
    api.create_repo(repo_id, repo_type=repo_type, private=True, exist_ok=True, **create_kwargs)
    info = api.repo_info(repo_id, repo_type=repo_type)
    if not info.private:
        raise SystemExit(f"{repo_id} exists and is PUBLIC; refusing to upload.")


def write_cards(export_dir: Path) -> None:
    for size, info in MODELS.items():
        folder = export_dir / f"simple-metric-{size}"
        if not (folder / "model.safetensors").exists():
            raise SystemExit(f"Missing {folder}; run scripts/convert_checkpoint.py first.")
        repo = f"{NAMESPACE}/simple-metric-{size}"
        (folder / "README.md").write_text(CARD.format(size=size, repo=repo, space=SPACE_ID, **info))
        print(f"wrote {folder / 'README.md'}")


SPACE_FILES = ("README.md", "index.html", "demo.js", "simple.py")
# Size of onnxruntime-web's WebAssembly runtime (ort-wasm-simd-threaded.wasm, 1.31.1), which
# the browser downloads from the CDN; shown to visitors as part of the demo's download size.
ORT_WASM_BYTES = 14_848_852


def stage_space(staging: Path, onnx_dir: Path) -> Path:
    """Copy the static Space files plus the browser model (with a manifest) into ``staging``."""
    import hashlib

    staging.mkdir(parents=True, exist_ok=True)
    for name in SPACE_FILES:
        shutil.copy(ROOT / "hf_space" / name, staging / name)
    model_dir = staging / "model"
    model_dir.mkdir()
    files = {"model": "model_quantized.onnx", "tokenizer": "tokenizer.json",
             "tokenizer_config": "tokenizer_config.json"}
    manifest = {}
    digest = hashlib.sha256()
    for key, name in files.items():
        src = onnx_dir / name
        if not src.exists():
            raise SystemExit(f"Missing {src}; run scripts/export_onnx.py first.")
        shutil.copy(src, model_dir / name)
        manifest[key] = {"path": f"model/{name}", "bytes": src.stat().st_size}
        with src.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
    # The version keys the browser cache, so a new model replaces the old one on visitors' devices.
    manifest["version"] = "small-int8-" + digest.hexdigest()[:12]
    manifest["runtime_bytes"] = ORT_WASM_BYTES
    report = onnx_dir / "export_report.json"
    if report.exists():
        manifest["export_report"] = json.loads(report.read_text())
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return staging


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--export-dir", type=Path, default=ROOT / "hub_export")
    ap.add_argument("--cards", action="store_true", help="Write the model cards only.")
    ap.add_argument("--upload-models", action="store_true",
                    help="Upload the converted models to PRIVATE Hub repos.")
    ap.add_argument("--upload-space", action="store_true",
                    help="Upload the static demo + evaluate module to a PRIVATE Space.")
    ap.add_argument("--onnx-dir", type=Path, default=ROOT / "hub_export" / "onnx-small",
                    help="Output of scripts/export_onnx.py for the browser demo.")
    ap.add_argument("--stage-only", type=Path, default=None,
                    help="Write the Space files to this folder instead of uploading (for testing).")
    args = ap.parse_args()
    if args.stage_only:
        print(f"staged Space in {stage_space(args.stage_only, args.onnx_dir)}")
        return
    if not (args.cards or args.upload_models or args.upload_space):
        ap.error("nothing to do: pass --cards, --upload-models, --upload-space or --stage-only")

    if args.cards or args.upload_models:
        write_cards(args.export_dir)
    if not (args.upload_models or args.upload_space):
        return
    from huggingface_hub import HfApi

    api = HfApi()
    if args.upload_models:
        for size in MODELS:
            repo = f"{NAMESPACE}/simple-metric-{size}"
            _ensure_private(api, repo)
            api.upload_folder(repo_id=repo, folder_path=args.export_dir / f"simple-metric-{size}",
                              commit_message="Upload SIMPLE checkpoint")
            print(f"uploaded {repo} (private)")
    if args.upload_space:
        _ensure_private(api, SPACE_ID, repo_type="space", space_sdk="static")
        with tempfile.TemporaryDirectory() as tmp:
            folder = stage_space(Path(tmp) / "space", args.onnx_dir)
            api.upload_folder(repo_id=SPACE_ID, repo_type="space", folder_path=folder,
                              commit_message="Upload SIMPLE static Space (in-browser demo)",
                              delete_patterns=["*.py", "*.js", "*.html", "model/**"])
        print(f"uploaded {SPACE_ID} (private static Space)")


if __name__ == "__main__":
    main()
