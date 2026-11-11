import json

import pytest

from simple_metric import SIMPLE
from simple_metric.cli import main as cli_main
from simple_metric.training import config_from_dict, train

SENTS = [
    ("the old man went to the store and bought some bread", "the man bought bread"),
    ("it rained all day long and people were sad", "it rained all day"),
    ("the children played outside in the city", "children played outside"),
    ("the big dog ran quickly over the old bridge", "the dog ran over the bridge"),
    ("the woman was very happy and went to the river", "the woman was happy"),
    ("people in the city were sad it rained all day", "people were sad"),
]


@pytest.fixture()
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "parallel.tsv").write_text(
        "source\tsimplification\n" + "".join(f"{s}\t{t}\n" for s, t in SENTS))
    (d / "ratings.csv").write_text(
        "source,candidate,rating\n" + "".join(f"{s},{t},{50 + 7 * i}\n{s},{s},50\n"
                                             for i, (s, t) in enumerate(SENTS)))
    (d / "monitor.jsonl").write_text("".join(
        json.dumps({"source": s, "chosen": t, "rejected": s}) + "\n" for s, t in SENTS))
    return d


def _cfg(data_dir, backbone, out, **training):
    return config_from_dict({
        "model": {"backbone": str(backbone), "max_length": 64},
        "data": {
            "parallel": ["parallel.tsv"],
            "ratings": [{"path": "ratings.csv", "scale": [0, 100]}],
            "augmentation": {"copy_anchors": 4, "adversarial_anchors": 4},
            "validation": {"fraction": 0.34},
        },
        "monitor": [{"path": "monitor.jsonl", "kind": "pairwise", "name": "toy"}],
        "training": {"epochs": 2, "batch_size": 2, "regression_batch_size": 4, "lr": 1e-3,
                     "num_workers": 0, "log_every": 1, **training},
        "output_dir": str(out),
    }, base_dir=data_dir)


def test_train_end_to_end(data_dir, tiny_backbone, tmp_path):
    out = train(_cfg(data_dir, tiny_backbone, tmp_path / "run", global_batch_size=4))
    for name in ("best", "final", "epoch_1", "epoch_2"):
        assert (out / name / "tokenizer" / "tokenizer_config.json").exists(), name
        assert (out / name / "model.safetensors").exists(), name
    lines = [json.loads(x) for x in (out / "metrics.jsonl").read_text().splitlines()]
    assert len(lines) == 2 and "val_all" in lines[0] and "monitor/toy/accuracy" in lines[0]
    metric = SIMPLE.from_pretrained(out / "best", device="cpu")
    res = metric.score([SENTS[0][0]], [SENTS[0][1]])
    assert len(res) == 1


@pytest.mark.parametrize("pair,reg", [("bt", "mse"), ("none", "nll"), ("thurstone", "none")])
def test_ablation_losses(data_dir, tiny_backbone, tmp_path, pair, reg):
    out = train(_cfg(data_dir, tiny_backbone, tmp_path / f"{pair}_{reg}", epochs=1,
                     pairwise_loss=pair, regression_loss=reg, epoch_fraction=0.5,
                     save_every_epoch=False, select_on="last"))
    assert (out / "best" / "model.safetensors").exists()


def test_resume(data_dir, tiny_backbone, tmp_path):
    cfg = _cfg(data_dir, tiny_backbone, tmp_path / "run", epochs=1)
    train(cfg)
    cfg.training.epochs, cfg.training.resume = 2, True
    out = train(cfg)
    lines = (out / "metrics.jsonl").read_text().splitlines()
    assert [json.loads(x)["epoch"] for x in lines] == [1, 2]


def test_cli_build_data(data_dir, tmp_path, capsys):
    rc = cli_main(["build-data", "--parallel", str(data_dir / "parallel.tsv"),
                   "--ratings", str(data_dir / "ratings.csv"), "--rating-scale", "0", "100",
                   "--output-dir", str(tmp_path / "built")])
    assert rc == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["train_pairs"] > 0
    assert (tmp_path / "built" / "train_pairs.jsonl").exists()
