"""Training loop for SIMPLE: ``L = L_pairwise + lambda * L_regression`` (paper Eq. 6).

Runs on one device, or on several with ``torchrun --nproc_per_node=N`` (DDP). Each step
scores the chosen and rejected candidates of a batch of preference pairs and a batch of
regression examples in a single forward pass, so gradients are synchronised correctly
under DDP and gradient accumulation (``no_sync``) works as expected.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import random
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from ..data.builders import INTER_TYPES, DataSpec, TrainingData, build_training_data
from ..data.io import read_records, write_jsonl
from ..losses import PAIRWISE_LOSSES, REGRESSION_LOSSES
from ..modeling import SimpleModel, load_tokenizer
from ..scorer import SIMPLE
from .config import Config
from .datasets import PairCollator, PairDataset, RegressionCollator, RegressionDataset, pad

logger = logging.getLogger("simple_score.training")


# ----------------------------------------------------------------------------- utilities


class _Dist:
    def __init__(self) -> None:
        self.enabled = "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ
        if self.enabled:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            # Long timeout: rank 0 evaluates alone at the end of each epoch.
            dist.init_process_group(backend=backend, timeout=timedelta(hours=3))
            self.rank = dist.get_rank()
            self.world = dist.get_world_size()
            self.local_rank = int(os.environ["LOCAL_RANK"])
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device("cuda", self.local_rank)
            else:
                self.device = torch.device("cpu")
        else:
            self.rank, self.world, self.local_rank = 0, 1, 0
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def close(self) -> None:
        if self.enabled:
            dist.destroy_process_group()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _epoch_indices(n: int, fraction: float, seed: int, epoch: int, rank: int, world: int) -> list[int]:
    """Shuffled indices for one epoch: a random ``fraction`` of ``n``, sharded across ranks
    (padded so every rank gets the same number)."""
    g = torch.Generator()
    g.manual_seed(seed * 1_000_003 + epoch)
    perm = torch.randperm(n, generator=g).tolist()
    perm = perm[: max(1, math.ceil(n * fraction))] if fraction < 1.0 else perm
    per_rank = math.ceil(len(perm) / world)
    perm = (perm * (1 + (per_rank * world) // max(1, len(perm))))[: per_rank * world]
    return perm[rank::world]


class _RegressionStream:
    """Endless regression batches; reshuffles after each pass over its per-epoch subset."""

    def __init__(self, dataset, batch_size, collate, fraction, seed, epoch, rank, world, workers):
        self.args = (dataset, batch_size, collate, workers)
        self.indices = _epoch_indices(len(dataset), fraction, seed + 17, epoch, rank, world)
        self.seed, self.epoch, self.passes = seed, epoch, 0
        self._it = None

    def _new_loader(self):
        dataset, batch_size, collate, workers = self.args
        rng = random.Random(self.seed * 7919 + self.epoch * 101 + self.passes)
        order = self.indices[:]
        rng.shuffle(order)
        self.passes += 1
        return iter(DataLoader(dataset, batch_size=batch_size, sampler=order, collate_fn=collate,
                               num_workers=workers))

    def next(self) -> dict:
        if self._it is None:
            self._it = self._new_loader()
        try:
            return next(self._it)
        except StopIteration:
            self._it = self._new_loader()
            return next(self._it)


def _optimizer_and_scheduler(model, cfg, num_steps: int):
    from transformers import get_linear_schedule_with_warmup

    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(num_steps * cfg.warmup_ratio), num_steps)
    return optimizer, scheduler


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y) -> float:
    return pearson(_rank(np.asarray(x, float)), _rank(np.asarray(y, float)))


# ----------------------------------------------------------------------------- evaluation


def _pair_accuracy(scorer: SIMPLE, pairs: list[dict]) -> np.ndarray:
    src = [p["source"] for p in pairs]
    a = scorer.score(src, [p["chosen"] for p in pairs]).mu
    b = scorer.score(src, [p["rejected"] for p in pairs]).mu
    return np.asarray(a) > np.asarray(b)


def evaluate(scorer: SIMPLE, val: list[dict], monitors: list[dict]) -> dict:
    """Validation accuracy (overall, per pair type, and on inter-simplification types)
    plus every monitor set's metrics."""
    out: dict = {}
    if val:
        correct = _pair_accuracy(scorer, val)
        types = np.asarray([p["type"] for p in val])
        out["val_all"] = float(correct.mean())
        inter = np.isin(types, INTER_TYPES)
        if inter.any():
            out["val_inter"] = float(correct[inter].mean())
        for t in sorted(set(types.tolist())):
            out[f"val_type/{t}"] = float(correct[types == t].mean())
    for m in monitors:
        name = m["name"]
        if m["kind"] == "pairwise":
            correct = _pair_accuracy(scorer, m["records"])
            out[f"monitor/{name}/accuracy"] = float(correct.mean())
        else:
            rec = m["records"]
            mu = scorer.score([r["source"] for r in rec], [r["candidate"] for r in rec]).mu
            ratings = [float(r["rating"]) for r in rec]
            out[f"monitor/{name}/pearson"] = pearson(mu, ratings)
            out[f"monitor/{name}/spearman"] = spearman(mu, ratings)
    return out


def _load_monitors(cfg: Config) -> list[dict]:
    base = Path(cfg.base_dir) if cfg.base_dir else None
    monitors = []
    for i, m in enumerate(cfg.monitor):
        kind = m.get("kind", "pairwise")
        if kind == "pairwise":
            recs = read_records(m, ["source", "chosen", "rejected"], base_dir=base)
        elif kind == "ratings":
            recs = read_records(m, ["source", "candidate", "rating"], base_dir=base)
        else:
            raise ValueError(f"monitor kind must be 'pairwise' or 'ratings', not {kind!r}")
        monitors.append({"name": m.get("name", f"monitor{i}"), "kind": kind, "records": recs})
    return monitors


def _selection_value(select_on: str, metrics: dict, monitors: list[dict]) -> float | None:
    if select_on == "last":
        return None
    if select_on == "val_inter":
        return metrics.get("val_inter", metrics.get("val_all"))
    if select_on == "val_all":
        return metrics.get("val_all")
    if select_on.startswith("monitor"):
        name = select_on.split(":", 1)[1] if ":" in select_on else monitors[0]["name"]
        for key in (f"monitor/{name}/accuracy", f"monitor/{name}/pearson"):
            if key in metrics:
                return metrics[key]
        raise ValueError(f"select_on={select_on!r}: no monitor named {name!r}")
    raise ValueError(f"Unknown select_on {select_on!r}")


# ----------------------------------------------------------------------------- training


def _latest_epoch_dir(output_dir: Path) -> Path | None:
    dirs = [d for d in output_dir.glob("epoch_*") if (d / "training_state.pt").exists()]
    return max(dirs, key=lambda d: int(d.name.split("_")[1])) if dirs else None


def train(cfg: Config, data: TrainingData | None = None) -> Path:
    """Train a SIMPLE model as configured; returns the output directory.

    ``output_dir`` receives ``best/`` (selected checkpoint), ``final/``, optionally
    ``epoch_N/``, plus ``metrics.jsonl``, ``data_stats.json`` and ``config.json``.
    """
    from transformers import AutoTokenizer

    tc, mc = cfg.training, cfg.model
    if tc.pairwise_loss not in (*PAIRWISE_LOSSES, "none"):
        raise ValueError(f"pairwise_loss must be one of {list(PAIRWISE_LOSSES)} or 'none'")
    if tc.regression_loss not in (*REGRESSION_LOSSES, "none"):
        raise ValueError(f"regression_loss must be one of {list(REGRESSION_LOSSES)} or 'none'")
    use_reg = tc.regression_loss != "none" and tc.lambda_reg > 0
    use_pairs = tc.pairwise_loss != "none"
    if not (use_pairs or use_reg):
        raise ValueError("Both losses are disabled; nothing to train.")

    ctx = _Dist()
    _set_seed(tc.seed)
    out = Path(cfg.output_dir)
    if ctx.is_main:
        out.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    ctx.barrier()

    # ---------------- data (deterministic, so every rank builds the same thing)
    if data is None:
        spec = DataSpec.from_dict(cfg.data, base_dir=Path(cfg.data_base_dir) if cfg.data_base_dir else None)
        data = build_training_data(spec)
    monitors = _load_monitors(cfg)
    if ctx.is_main:
        (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str))
        (out / "data_stats.json").write_text(json.dumps(data.stats(), indent=2))
        logger.info("Data: %s", json.dumps(data.stats()))
    if use_reg and not data.regression:
        raise ValueError("regression loss is on but there is no regression data "
                         "(add `ratings`, or keep copy/adversarial anchors on).")

    # ---------------- model
    resume_dir = _latest_epoch_dir(out) if tc.resume else None
    if resume_dir is not None:
        model = SimpleModel.from_pretrained(resume_dir)
        tokenizer = load_tokenizer(resume_dir)
        start_epoch = int(resume_dir.name.split("_")[1]) + 1
        if ctx.is_main:
            logger.info("Resuming from %s (epoch %d)", resume_dir, start_epoch)
    else:
        model = SimpleModel.from_backbone(mc.backbone, sigma_floor=mc.sigma_floor,
                                          max_length=mc.max_length)
        tokenizer = AutoTokenizer.from_pretrained(mc.backbone)
        start_epoch = 1
    model.to(ctx.device)
    if mc.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    wrapped = model
    if ctx.enabled:
        # sigma gets no gradient when neither loss uses it (the BT + MSE ablation).
        uses_sigma = tc.pairwise_loss == "thurstone" or (use_reg and tc.regression_loss == "nll")
        wrapped = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[ctx.local_rank] if ctx.device.type == "cuda" else None,
            find_unused_parameters=not uses_sigma)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    pair_ds = PairDataset(data.train, tokenizer, mc.max_length)
    reg_ds = RegressionDataset(data.regression, tokenizer, mc.max_length) if use_reg else None
    main_ds, main_bs = (pair_ds, tc.batch_size) if use_pairs else (reg_ds, tc.regression_batch_size)
    main_collate = PairCollator(pad_id) if use_pairs else RegressionCollator(pad_id)

    per_step = main_bs * ctx.world
    if tc.global_batch_size and use_pairs:
        if tc.global_batch_size % per_step:
            raise ValueError(f"global_batch_size={tc.global_batch_size} is not a multiple of "
                             f"batch_size x devices = {per_step}")
        accum = tc.global_batch_size // per_step
    else:
        accum = 1
    micro_per_epoch = math.ceil(
        len(_epoch_indices(len(main_ds), tc.epoch_fraction, tc.seed, 1, ctx.rank, ctx.world))
        / main_bs)
    steps_per_epoch = math.ceil(micro_per_epoch / accum)
    total_steps = steps_per_epoch * tc.epochs
    optimizer, scheduler = _optimizer_and_scheduler(model, tc, total_steps)

    best_value = -float("inf")
    if resume_dir is not None:
        state = torch.load(resume_dir / "training_state.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best_value = state.get("best_value", best_value)

    if ctx.is_main:
        logger.info(
            "Training: %d pairs, %d regression examples | %d device(s) x %d x accum %d "
            "= %d pairs/step | %d steps/epoch x %d epochs | losses %s + %.2f x %s",
            len(data.train), len(data.regression), ctx.world, tc.batch_size, accum,
            tc.batch_size * ctx.world * accum, steps_per_epoch, tc.epochs,
            tc.pairwise_loss, tc.lambda_reg if use_reg else 0.0, tc.regression_loss)

    pair_loss_fn = PAIRWISE_LOSSES.get(tc.pairwise_loss)
    reg_loss_fn = REGRESSION_LOSSES.get(tc.regression_loss)
    autocast = (torch.autocast(device_type=ctx.device.type, dtype=torch.bfloat16)
                if tc.precision == "bf16" else contextlib.nullcontext())
    val = data.val
    if tc.max_val_pairs and len(val) > tc.max_val_pairs:
        val = random.Random(tc.seed).sample(val, tc.max_val_pairs)

    for epoch in range(start_epoch, tc.epochs + 1):
        wrapped.train()
        indices = _epoch_indices(len(main_ds), tc.epoch_fraction, tc.seed, epoch, ctx.rank,
                                 ctx.world)
        loader = DataLoader(main_ds, batch_size=main_bs, sampler=indices, collate_fn=main_collate,
                            num_workers=tc.num_workers, pin_memory=ctx.device.type == "cuda")
        reg_stream = None
        if use_reg and use_pairs:
            reg_stream = _RegressionStream(reg_ds, tc.regression_batch_size, RegressionCollator(pad_id),
                                           tc.epoch_fraction, tc.seed, epoch, ctx.rank, ctx.world,
                                           tc.num_workers)
        sums = {"loss": 0.0, "pairwise": 0.0, "regression": 0.0}
        t0 = time.time()
        n_micro = len(loader)
        for micro, batch in enumerate(loader, 1):
            sync = micro % accum == 0 or micro == n_micro
            # L = L_pairwise + lambda * L_regression. The two terms get their own forward and
            # backward (gradients add up), so only one batch's activations live at a time.
            parts = []
            if use_pairs:
                parts.append(("pairwise", batch))
                if reg_stream is not None:
                    parts.append(("regression", reg_stream.next()))
            else:
                parts.append(("regression", batch))
            # Under DDP the last backward of a step does the all-reduce, so it must reach every
            # parameter that gets a gradient: put the part that uses sigma last.
            if len(parts) == 2 and tc.pairwise_loss == "thurstone" and tc.regression_loss != "nll":
                parts.reverse()
            for k, (kind, b) in enumerate(parts):
                reduce_now = sync and k == len(parts) - 1
                no_sync = (wrapped.no_sync() if ctx.enabled and not reduce_now
                           else contextlib.nullcontext())
                with no_sync:
                    seqs = b["pos"] + b["neg"] if kind == "pairwise" else b["ids"]
                    ids, mask = pad(seqs, pad_id)
                    with autocast:
                        mu, sigma = wrapped(ids.to(ctx.device), mask.to(ctx.device))
                    mu, sigma = mu.float(), sigma.float()
                    if kind == "pairwise":
                        n = len(b["pos"])
                        raw = pair_loss_fn(mu[:n], mu[n:], sigma[:n], sigma[n:],
                                           b["weight"].to(ctx.device), b["margin"].to(ctx.device))
                        loss = raw
                    else:
                        raw = reg_loss_fn(mu, sigma, b["target"].to(ctx.device))
                        loss = tc.lambda_reg * raw
                    sums[kind] += raw.item()
                    sums["loss"] += loss.item()
                    (loss / accum).backward()
            if sync:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if ctx.is_main and micro % (tc.log_every * accum) == 0:
                logger.info("epoch %d  step %d/%d  loss %.4f  pairwise %.4f  regression %.4f  "
                            "(%.1f min)", epoch, micro // accum, steps_per_epoch,
                            sums["loss"] / micro, sums["pairwise"] / micro,
                            sums["regression"] / micro, (time.time() - t0) / 60)

        # ---------------- end of epoch: evaluate, select, save (main process only)
        if ctx.is_main:
            scorer = SIMPLE(model, tokenizer, device=ctx.device, batch_size=tc.eval_batch_size)
            metrics = evaluate(scorer, val, monitors)
            model.train()
            value = _selection_value(tc.select_on, metrics, monitors)
            is_best = value is None or value > best_value
            if is_best and value is not None:
                best_value = value
            record = {"epoch": epoch, **{f"train/{k}": v / max(1, n_micro) for k, v in sums.items()},
                      **metrics, "selection": value, "is_best": is_best}
            with (out / "metrics.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            logger.info("epoch %d done: %s", epoch,
                        "  ".join(f"{k}={v:.4f}" for k, v in record.items()
                                  if isinstance(v, float)))
            if is_best:
                model.save_pretrained(out / "best", tokenizer)
                logger.info("  -> saved best/ (%s=%s)", tc.select_on, value)
            if tc.save_every_epoch or tc.resume:
                ep_dir = out / f"epoch_{epoch}"
                model.save_pretrained(ep_dir, tokenizer)
                torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                            "best_value": best_value}, ep_dir / "training_state.pt")
        ctx.barrier()

    if ctx.is_main:
        model.save_pretrained(out / "final", tokenizer)
        logger.info("Done. Selected checkpoint: %s", out / "best")
    ctx.barrier()
    ctx.close()
    return out


def export_training_data(data: TrainingData, output_dir: str | os.PathLike) -> dict:
    """Write the derived training data as JSONL (useful to inspect what SIMPLE will see)."""
    output_dir = Path(output_dir)
    write_jsonl(data.train, output_dir / "train_pairs.jsonl")
    write_jsonl(data.val, output_dir / "val_pairs.jsonl")
    write_jsonl(data.regression, output_dir / "regression.jsonl")
    stats = data.stats()
    (output_dir / "data_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


__all__ = ["Config", "evaluate", "export_training_data", "pearson", "spearman", "train"]
