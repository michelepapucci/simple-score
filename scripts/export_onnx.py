#!/usr/bin/env python3
"""Export a SIMPLE model to ONNX for in-browser inference, and measure what quantization costs.

Writes to ``--output-dir``:
  model.onnx            float32 graph: (input_ids, attention_mask) [batch, seq] -> mu, sigma [batch]
  model_quantized.onnx  int8 weights (per-channel, weight-only; including the embedding
                        matrix) with float32 arithmetic, for the web demo
  tokenizer.json, tokenizer_config.json, special_tokens_map.json
  export_report.json    agreement with PyTorch and SimpEval 2022 results for each variant

    python scripts/export_onnx.py --model small --output-dir hub_export/onnx-small

Needs `pip install onnx onnxruntime` (numpy < 2 if your pandas needs it). The SimpEval 2022 check downloads the benchmark from
github.com/Yao-Dou/LENS (as reproduce/table2.py does).
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from simple_metric.modeling import SimpleModel, load_tokenizer, resolve_checkpoint_dir
from simple_metric.training.trainer import pearson, spearman

ROOT = Path(__file__).resolve().parent.parent


class _Graph(torch.nn.Module):
    def __init__(self, model: SimpleModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)


def export(model: SimpleModel, tokenizer, path: Path, opset: int) -> None:
    enc = tokenizer(["The feline perched upon the mat in a state of repose.", "It rained."],
                    ["The cat sat on the mat.", "It rained all day long."],
                    padding=True, return_tensors="pt")
    kwargs = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False  # the TorchScript exporter handles DeBERTa's relative attention
    torch.onnx.export(
        _Graph(model).eval(), (enc["input_ids"], enc["attention_mask"]), str(path),
        input_names=["input_ids", "attention_mask"], output_names=["mu", "sigma"],
        dynamic_axes={"input_ids": {0: "batch", 1: "sequence"},
                      "attention_mask": {0: "batch", 1: "sequence"},
                      "mu": {0: "batch"}, "sigma": {0: "batch"}},
        opset_version=opset, do_constant_folding=True, **kwargs)


def quantize(src: Path, dst: Path, min_elements: int = 65536) -> dict:
    """Weight-only int8 quantization with one scale per output channel.

    Every large 2-D float weight read by a MatMul (one scale per column) or a Gather (the
    embedding matrix; one scale per row) is stored as int8 and dequantized inside the graph
    by a DequantizeLinear node, so all arithmetic stays in float32. Unlike dynamic
    quantization (8-bit activations too), this leaves DeBERTa's outputs nearly unchanged.
    """
    from collections import defaultdict

    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(str(src))
    graph = model.graph
    uses = defaultdict(set)
    for node in graph.node:
        for i, name in enumerate(node.input):
            uses[name].add((node.op_type, i))
    keep, added, dequant = [], [], []
    n_quantized = 0
    for init in graph.initializer:
        kinds = uses.get(init.name, set())
        if (init.data_type != TensorProto.FLOAT or len(init.dims) != 2
                or int(np.prod(init.dims)) < min_elements or not kinds):
            keep.append(init)
            continue
        if kinds <= {("Gather", 0)}:
            axis = 0  # embedding matrix: one scale per row (token)
        elif kinds <= {("MatMul", 1)}:
            axis = 1  # linear layer weight [in, out]: one scale per output column
        else:
            keep.append(init)
            continue
        w = numpy_helper.to_array(init).astype(np.float32)
        scale = np.abs(w).max(axis=1 - axis) / 127.0
        scale[scale == 0] = 1.0
        q = np.clip(np.round(w / np.expand_dims(scale, 1 - axis)), -127, 127).astype(np.int8)
        names = (f"{init.name}_int8", f"{init.name}_scale", f"{init.name}_zero")
        added += [numpy_helper.from_array(q, names[0]),
                  numpy_helper.from_array(scale.astype(np.float32), names[1]),
                  numpy_helper.from_array(np.zeros(scale.shape, dtype=np.int8), names[2])]
        dequant.append(helper.make_node("DequantizeLinear", list(names), [init.name],
                                        axis=axis, name=f"{init.name}_dequantize"))
        n_quantized += 1
    del graph.initializer[:]
    graph.initializer.extend(keep + added)
    nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(dequant + nodes)
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))
    return {"quantized_weights": n_quantized}


class _OnnxScorer:
    def __init__(self, path: Path, tokenizer, max_length: int) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.tokenizer, self.max_length = tokenizer, max_length

    def score(self, sources, candidates, batch_size=32):
        mu, sigma = [], []
        for i in range(0, len(sources), batch_size):
            enc = self.tokenizer(sources[i:i + batch_size], candidates[i:i + batch_size],
                                 max_length=self.max_length, truncation=True, padding=True,
                                 return_tensors="np")
            m, s = self.session.run(None, {"input_ids": enc["input_ids"].astype(np.int64),
                                           "attention_mask": enc["attention_mask"].astype(np.int64)})
            mu += np.clip(m, -1, 1).tolist()
            sigma += s.tolist()
        return np.asarray(mu), np.asarray(sigma)


class _TorchScorer:
    def __init__(self, model, tokenizer, max_length):
        self.model, self.tokenizer, self.max_length = model.eval(), tokenizer, max_length

    @torch.no_grad()
    def score(self, sources, candidates, batch_size=32):
        mu, sigma = [], []
        for i in range(0, len(sources), batch_size):
            enc = self.tokenizer(sources[i:i + batch_size], candidates[i:i + batch_size],
                                 max_length=self.max_length, truncation=True, padding=True,
                                 return_tensors="pt")
            m, s = self.model(enc["input_ids"], enc["attention_mask"])
            mu += m.clamp(-1, 1).tolist()
            sigma += s.tolist()
        return np.asarray(mu), np.asarray(sigma)


def _simpeval(cache: Path):
    spec = importlib.util.spec_from_file_location("table2", ROOT / "reproduce" / "table2.py")
    table2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(table2)
    return table2.simpeval(cache)


def evaluate(scorer, rows, pairs) -> dict:
    src = [p[0] for p in pairs]
    a, _ = scorer.score(src, [p[1] for p in pairs])
    b, _ = scorer.score(src, [p[2] for p in pairs])
    acc = float(((a > b) == np.asarray([p[3] for p in pairs])).mean())
    mu, sigma = scorer.score([r["original"] for r in rows], [r["generation"] for r in rows])
    human = [r["composite"] for r in rows]
    return {"pairwise_accuracy": round(acc, 3), "pearson": round(pearson(mu, human), 3),
            "spearman": round(spearman(mu, human), 3), "_mu": mu, "_sigma": sigma}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", default="small", help="small | base | large | repo id | local dir")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "benchmarks")
    ap.add_argument("--skip-check", action="store_true")
    args = ap.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    directory = resolve_checkpoint_dir(args.model)
    model = SimpleModel.from_pretrained(directory).eval()
    tokenizer = load_tokenizer(directory)
    max_length = model.config.max_length

    export(model, tokenizer, out / "model.onnx", args.opset)
    quantize(out / "model.onnx", out / "model_quantized.onnx")
    tok_dir = directory / "tokenizer" if (directory / "tokenizer").is_dir() else directory
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        if (tok_dir / name).exists():
            shutil.copy(tok_dir / name, out / name)
    sizes = {p.name: round(p.stat().st_size / 2**20, 1) for p in sorted(out.glob("*.onnx"))}
    print("sizes (MB):", sizes)
    if args.skip_check:
        return

    rows, pairs = _simpeval(args.cache)
    report = {"model": str(args.model), "sizes_mb": sizes, "max_length": max_length}
    ref = evaluate(_TorchScorer(model, tokenizer, max_length), rows, pairs)
    for name, scorer in [("pytorch", None), ("onnx_fp32", "model.onnx"),
                         ("onnx_int8", "model_quantized.onnx")]:
        res = ref if scorer is None else evaluate(
            _OnnxScorer(out / scorer, tokenizer, max_length), rows, pairs)
        entry = {k: v for k, v in res.items() if not k.startswith("_")}
        entry["max_abs_diff_mu_vs_pytorch"] = float(np.max(np.abs(res["_mu"] - ref["_mu"])))
        entry["mean_abs_diff_mu_vs_pytorch"] = float(np.mean(np.abs(res["_mu"] - ref["_mu"])))
        entry["max_abs_diff_sigma_vs_pytorch"] = float(np.max(np.abs(res["_sigma"] - ref["_sigma"])))
        report[name] = entry
        print(name, json.dumps(entry))
    (out / "export_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
