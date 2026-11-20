#!/usr/bin/env python3
"""Convert a research checkpoint into the released SIMPLE format.

Research checkpoints (``train_simrmsigma.py``) are directories holding the encoder
(``config.json`` + ``model.safetensors`` saved by transformers) plus ``mu_head.pt`` and
``sigma_head.pt``. The released format is one ``model.safetensors`` with every weight,
a SIMPLE ``config.json`` and the tokenizer files.

With ``--check FILE`` (JSONL with ``source`` and ``candidate``) the converted model is
compared with the original loader; conversion fails if any |delta mu| or |delta sigma|
exceeds ``--tolerance``.

    python scripts/convert_checkpoint.py \
        --legacy /path/to/large_exp15/best \
        --tokenizer /path/to/tokenizer_dir \
        --output hub_export/simple-score-large \
        --check checks.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from simple_score import SIMPLE, SimpleConfig, SimpleModel
from simple_score.modeling import _encoder_config_to_dict

LEGACY_SIGMA_FLOOR = 0.05


def load_legacy(legacy: Path) -> tuple[torch.nn.Module, torch.nn.Linear, torch.nn.Linear]:
    try:
        encoder = AutoModel.from_pretrained(str(legacy), use_safetensors=True)
    except (OSError, ValueError):
        encoder = AutoModel.from_pretrained(str(legacy))
    encoder = encoder.to(torch.float32).eval()
    hidden = encoder.config.hidden_size
    heads = []
    for name in ("mu_head.pt", "sigma_head.pt"):
        head = torch.nn.Linear(hidden, 1)
        head.load_state_dict(torch.load(legacy / name, map_location="cpu", weights_only=True))
        heads.append(head.eval())
    return encoder, heads[0], heads[1]


@torch.no_grad()
def legacy_scores(encoder, mu_head, sigma_head, tokenizer, pairs, device, bs=32):
    encoder, mu_head, sigma_head = encoder.to(device), mu_head.to(device), sigma_head.to(device)
    mus, sigmas = [], []
    for i in range(0, len(pairs), bs):
        s, c = zip(*pairs[i : i + bs])
        enc = tokenizer(list(s), list(c), max_length=512, truncation=True, padding=True,
                        return_tensors="pt").to(device)
        h = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        cls = h.last_hidden_state[:, 0, :].float()
        mus += torch.tanh(mu_head(cls).squeeze(-1)).clamp(-1, 1).tolist()
        sigmas += (F.softplus(sigma_head(cls).squeeze(-1)) + LEGACY_SIGMA_FLOOR).tolist()
    return mus, sigmas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--legacy", required=True, type=Path)
    ap.add_argument("--tokenizer", type=Path, help="Tokenizer dir (default: --legacy).")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--check", type=Path, help="JSONL with source/candidate to compare outputs.")
    ap.add_argument("--tolerance", type=float, default=1e-5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer or args.legacy))
    encoder, mu_head, sigma_head = load_legacy(args.legacy)
    config = SimpleConfig(encoder=_encoder_config_to_dict(encoder.config),
                          sigma_floor=LEGACY_SIGMA_FLOOR, max_length=512)
    model = SimpleModel(config, encoder=encoder)
    model.mu_head.load_state_dict(mu_head.state_dict())
    model.sigma_head.load_state_dict(sigma_head.state_dict())
    model.save_pretrained(args.output, tokenizer)
    print(f"Saved {args.output}")

    if args.check:
        rows = [json.loads(x) for x in args.check.read_text().splitlines() if x.strip()]
        pairs = [(r["source"], r["candidate"]) for r in rows]
        ref_mu, ref_sigma = legacy_scores(*load_legacy(args.legacy), tokenizer, pairs, args.device)
        new = SIMPLE.from_pretrained(args.output, device=args.device).score(
            [p[0] for p in pairs], [p[1] for p in pairs])
        d_mu = max(abs(a - b) for a, b in zip(ref_mu, new.mu))
        d_sigma = max(abs(a - b) for a, b in zip(ref_sigma, new.sigma))
        print(f"Checked {len(pairs)} pairs: max |d mu| = {d_mu:.2e}, max |d sigma| = {d_sigma:.2e}")
        if max(d_mu, d_sigma) > args.tolerance:
            raise SystemExit(f"Outputs differ by more than {args.tolerance}")


if __name__ == "__main__":
    main()
