"""Offline fixtures: a tiny DeBERTa-v2 encoder and a word-level tokenizer built in memory."""

from __future__ import annotations

import pytest
import torch

WORDS = (
    "the a cat dog sat on mat big small house was is very old man woman went to "
    "store bought some bread it rained all day long people were happy sad and of "
    "in quickly slowly city river ran over bridge children played outside"
).split()


def make_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocab = {t: i for i, t in enumerate(["[PAD]", "[CLS]", "[SEP]", "[UNK]", *WORDS])}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
    )
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", cls_token="[CLS]",
                                   sep_token="[SEP]", unk_token="[UNK]")


def make_encoder_config(vocab_size: int):
    from transformers import DebertaV2Config

    return DebertaV2Config(
        vocab_size=vocab_size, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=64, max_position_embeddings=128, relative_attention=True,
        pos_att_type=["p2c", "c2p"], position_buckets=16, max_relative_positions=-1,
        type_vocab_size=0, position_biased_input=False, pad_token_id=0,
    )


@pytest.fixture(scope="session")
def tokenizer():
    return make_tokenizer()


@pytest.fixture()
def tiny_model(tokenizer):
    from simple_metric.modeling import SimpleConfig, SimpleModel, _encoder_config_to_dict

    torch.manual_seed(0)
    cfg = SimpleConfig(encoder=_encoder_config_to_dict(make_encoder_config(len(tokenizer))),
                       max_length=64)
    return SimpleModel(cfg).eval()


@pytest.fixture()
def tiny_backbone(tmp_path, tokenizer):
    """A directory holding a tiny pretrained-style backbone + tokenizer (for training)."""
    from transformers import AutoModel

    torch.manual_seed(0)
    enc = AutoModel.from_config(make_encoder_config(len(tokenizer)))
    path = tmp_path / "backbone"
    enc.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path
