"""Tests für den BPE-Tokenizer."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tokenizer.bpe_tokenizer import BPETokenizer, SpecialTokens

CORPUS = [
    "das modell lernt sprache",
    "das modell lernt schnell",
    "sprache besteht aus wörtern",
    "wörter bestehen aus zeichen",
] * 20


def train_small() -> BPETokenizer:
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=400, min_frequency=1)
    return tok


def test_special_token_ids():
    tok = train_small()
    assert tok.pad_id == 0
    assert tok.bos_id == 1
    assert tok.eos_id == 2


def test_roundtrip_preserves_text():
    tok = train_small()
    text = "das modell lernt sprache"
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_roundtrip_unicode():
    """Byte-Level-BPE muss beliebigen Unicode verlustfrei rekonstruieren."""
    tok = train_small()
    text = "wörter mit ümlauten und emoji 🚀"
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_bos_eos_insertion():
    tok = train_small()
    ids = tok.encode("das modell", add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id


def test_vocab_size_respected():
    tok = train_small()
    assert tok.vocab_size <= 400
    assert tok.vocab_size >= 260  # Sondertokens + 256 Bytes


def test_min_vocab_size_error():
    tok = BPETokenizer()
    with pytest.raises(ValueError):
        tok.train(CORPUS, vocab_size=100)  # kleiner als Basis


def test_save_load_roundtrip():
    tok = train_small()
    text = "das modell lernt sprache 🚀"
    ids = tok.encode(text)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tok.json"
        tok.save(path)
        loaded = BPETokenizer.load(path)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.encode(text) == ids
    assert loaded.decode(ids) == text


def test_custom_special_tokens():
    tok = BPETokenizer(SpecialTokens(pad="<p>", bos="<s>", eos="</s>", unk="<u>"))
    tok.train(CORPUS, vocab_size=300, min_frequency=1)
    assert tok.decode(tok.encode("das modell")) == "das modell"
