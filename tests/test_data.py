import json

import pytest

from simple_score.data import DataFormatError, build_training_data, read_records
from simple_score.data.builders import pairs_from_graded, pairs_from_ratings, rating_targets

PARALLEL = [
    ("the old man went to the store and bought some bread", "the man bought bread"),
    ("it rained all day long and people were sad", "it rained all day"),
    ("the children played outside in the city", "children played outside"),
    ("the big dog ran quickly over the old bridge", "the dog ran over the bridge"),
]


def _write_tsv(path, header, rows):
    path.write_text("\t".join(header) + "\n" + "".join("\t".join(map(str, r)) + "\n" for r in rows))
    return path


def test_read_formats_and_aliases(tmp_path):
    _write_tsv(tmp_path / "p.tsv", ["complex", "simple"], PARALLEL)
    rows = read_records(tmp_path / "p.tsv", ["source", "simplification"])
    assert rows[0] == {"source": PARALLEL[0][0], "simplification": PARALLEL[0][1]}

    (tmp_path / "p.jsonl").write_text(
        "".join(json.dumps({"orig": s, "simp_text": t}) + "\n" for s, t in PARALLEL))
    rows = read_records({"path": tmp_path / "p.jsonl", "columns": {"simplification": "simp_text"}},
                        ["source", "simplification"])
    assert len(rows) == 4 and rows[1]["simplification"] == PARALLEL[1][1]

    (tmp_path / "a.txt").write_text("\n".join(s for s, _ in PARALLEL) + "\n")
    (tmp_path / "b.txt").write_text("\n".join(t for _, t in PARALLEL) + "\n")
    rows = read_records({"files": {"source": "a.txt", "simplification": "b.txt"}},
                        ["source", "simplification"], base_dir=tmp_path)
    assert rows[3] == {"source": PARALLEL[3][0], "simplification": PARALLEL[3][1]}


def test_tsv_allows_quotes(tmp_path):
    _write_tsv(tmp_path / "q.tsv", ["source", "simplification"],
               [('he said " now we go', 'he said "go"')])
    assert read_records(tmp_path / "q.tsv", ["source", "simplification"])[0]["source"].count('"') == 1


def test_missing_column_error_lists_columns(tmp_path):
    _write_tsv(tmp_path / "bad.tsv", ["foo", "bar"], [("a", "b")])
    with pytest.raises(DataFormatError, match="Available columns"):
        read_records(tmp_path / "bad.tsv", ["source", "simplification"])


def test_parallel_derivations_and_grouped_split(tmp_path):
    _write_tsv(tmp_path / "p.tsv", ["source", "simplification"], PARALLEL)
    data = build_training_data({
        "parallel": ["p.tsv"],
        "augmentation": {"copy_anchors": 3, "adversarial_anchors": 2},
        "validation": {"fraction": 0.25},
    }, base_dir=tmp_path)
    all_pairs = data.train + data.val
    types = {t: sum(r["type"] == t for r in all_pairs) for t in ("original", "mirror", "random_sentence")}
    assert types == {"original": 4, "mirror": 4, "random_sentence": 4}
    mirror = next(r for r in all_pairs if r["type"] == "mirror")
    assert mirror["source"] == mirror["chosen"] and mirror["rejected"] == mirror["group"]
    # grouped split: no source sentence group appears on both sides
    assert not ({r["group"] for r in data.train} & {r["group"] for r in data.val})
    assert len(data.val) == 3  # one group of three pairs
    reg_types = {r["type"] for r in data.regression}
    assert "copy" in reg_types and "random_other" in reg_types
    assert all(-1 <= r["target"] <= 1 for r in data.regression)


def test_graded_inter_pairs():
    rows = [
        {"source": "S", "simplification": "s1", "level": 1},
        {"source": "S", "simplification": "s2", "level": 2},
        {"source": "S", "simplification": "s4", "level": 4},
    ]
    pairs = pairs_from_graded(rows, min_level_gap=2)
    inter = [(p["chosen"], p["rejected"]) for p in pairs if p["type"] == "inter"]
    assert sorted(inter) == [("s4", "s1"), ("s4", "s2")]
    assert sum(p["type"] == "original" for p in pairs) == 3


def test_ratings_targets_pairs_and_margins():
    rows = [
        {"source": "S", "candidate": "good", "rating": 90},
        {"source": "S", "candidate": "ok", "rating": 60},
        {"source": "S", "candidate": "bad", "rating": 10},
    ]
    targets = rating_targets(rows, (0, 100), "t")
    assert [t["target"] for t in targets] == [0.8, 0.2, -0.8]
    pairs = pairs_from_ratings(targets, min_gap=0.15, max_margin=0.9)
    margins = {(p["chosen"], p["rejected"]): p["margin"] for p in pairs}
    assert margins[("good", "bad")] == 0.9  # capped
    assert margins[("good", "ok")] == 0.6
    assert margins[("S", "bad")] == 0.8  # copy beats a bad simplification
    assert margins[("good", "S")] == 0.8


def test_ratings_need_scale(tmp_path):
    _write_tsv(tmp_path / "r.tsv", ["source", "candidate", "rating"], [("a b", "a", 50)])
    with pytest.raises(DataFormatError, match="scale"):
        build_training_data({"ratings": ["r.tsv"]}, base_dir=tmp_path)


def test_default_weights_and_explicit_preferences(tmp_path):
    _write_tsv(tmp_path / "r.tsv", ["source", "candidate", "rating"],
               [("the old man went home", "the man went home", 90),
                ("the old man went home", "man home", 20)])
    _write_tsv(tmp_path / "pref.tsv", ["source", "chosen", "rejected", "margin"],
               [("it rained all day long", "it rained", "rained day all it", 0.3)])
    data = build_training_data({
        "ratings": [{"path": "r.tsv", "scale": [0, 100]}],
        "preferences": [{"path": "pref.tsv", "type": "inter"}],
        "validation": {"fraction": 0},
    }, base_dir=tmp_path)
    by_type = {r["type"]: r for r in data.train}
    assert by_type["rating_pair"]["weight"] == 3.0
    assert by_type["inter"]["weight"] == 1.0 and by_type["inter"]["margin"] == 0.3
    assert any(r["type"] == "flipped" for r in data.regression)
