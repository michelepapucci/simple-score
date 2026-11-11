import torch

from simple_metric import SIMPLE, SimpleModel
from simple_metric.hub import resolve_model_name


def test_output_ranges(tiny_model, tokenizer):
    enc = tokenizer(["the cat sat on the mat"], ["the cat sat"], return_tensors="pt")
    mu, sigma = tiny_model(enc["input_ids"], enc["attention_mask"])
    assert mu.shape == sigma.shape == (1,)
    assert -1 < mu.item() < 1
    assert sigma.item() > tiny_model.config.sigma_floor


def test_save_load_roundtrip(tmp_path, tiny_model, tokenizer):
    tiny_model.save_pretrained(tmp_path / "ckpt", tokenizer)
    assert (tmp_path / "ckpt" / "model.safetensors").exists()
    loaded = SimpleModel.from_pretrained(tmp_path / "ckpt")
    for (k, a), (_, b) in zip(tiny_model.state_dict().items(), loaded.state_dict().items()):
        assert torch.equal(a, b), k


def test_scorer_order_and_batching(tmp_path, tiny_model, tokenizer):
    tiny_model.save_pretrained(tmp_path / "ckpt", tokenizer)
    metric = SIMPLE.from_pretrained(tmp_path / "ckpt", device="cpu")
    src = ["the old man went to the store and bought some bread", "it rained all day", "the dog ran"]
    cand = ["the man bought bread", "it rained", "the dog ran over the bridge quickly"]
    full = metric.score(src, cand, batch_size=3)
    one_by_one = [metric.score(s, c).mu[0] for s, c in zip(src, cand)]
    assert len(full) == 3
    for a, b in zip(full.mu, one_by_one):
        assert abs(a - b) < 1e-5
    assert all(s > 0.05 for s in full.sigma)
    p = metric.preference_prob(src[0], cand[0], src[0])
    assert 0 < p[0] < 1


def test_aliases():
    assert resolve_model_name("large") == "mpapucci/simple-metric-large"
    assert resolve_model_name("SIMPLE-base") == "mpapucci/simple-metric-base"
    assert resolve_model_name("someone/other") == "someone/other"
