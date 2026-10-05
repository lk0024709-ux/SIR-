"""Tokenizer API + bake-off machinery. These run on a tiny in-memory corpus, no GPU, no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tokenizer.api import SPECIALS, SirTokenizer, TokenizerSpec
from tokenizer.evaluate_tokenizer import build_slices, slice_stats
from tokenizer.train_tokenizer import read_texts, sha256_file


def test_roundtrip_is_exact_for_byte_level_bpe(tiny_tokenizer):
    for text in [
        "भारत का मौसम अच्छा है।",
        "The weather in the city was quite pleasant all week.",
        "aaj office nahi jaunga, meeting cancel ho gayi thi.",
        "मैं अभी report finalize कर रहा हूँ, दस मिनट बाद भेजता हूँ।",
    ]:
        ids = tiny_tokenizer.encode(text)
        assert ids, "empty encoding"
        assert all(0 <= i < tiny_tokenizer.vocab_size for i in ids)
        assert tiny_tokenizer.decode(ids) == text, "byte-level BPE must round-trip exactly"


def test_unicode_edge_cases_survive_encoding(tiny_tokenizer):
    tricky = [
        "क्ष त्र ज्ञ",  # conjunct clusters
        "का कि की कु कू के कै को",  # matras
        "अनुस्वारः विराम। ॥",  # danda + anusvara
        "एक — दो – तीन ‘चौ’ “पाँच”",  # dashes and smart quotes
        "emoji 🟢 mixed with हिंदी",  # astral plane
        "नंबर 1,23,456 और 40%",  # digits/percent in Indic context
    ]
    for t in tricky:
        assert tiny_tokenizer.decode(tiny_tokenizer.encode(t)) == t


def test_no_unknown_tokens_for_byte_bpe(tiny_tokenizer):
    m = tiny_tokenizer.measure(["ज़बर्दस्त जगह ज़रूरत ज़्यादा"])
    assert m["unk_rate"] == 0.0


@pytest.mark.parametrize("kind", ["sp_bpe", "sp_unigram"])
def test_sentencepiece_candidates_load_and_encode(tiny_tokenizers, kind):
    tok = SirTokenizer.load(tiny_tokenizers[kind])
    ids = tok.encode("नगर निगम ने आदेश दिया है।")
    assert ids and all(isinstance(i, int) for i in ids)
    out = tok.decode(ids)
    assert "निगम" in out.replace("\u2581", " ")


@pytest.mark.parametrize("kind", ["byte_bpe", "sp_bpe", "sp_unigram"])
def test_measure_reports_every_bakeoff_metric(tiny_tokenizers, kind):
    tok = SirTokenizer.load(tiny_tokenizers[kind])
    texts = ["भारत का मौसम अच्छा है।", "The lab recorded sensor drift over six weeks."]
    m = tok.measure(texts)
    for key in (
        "tokens",
        "chars",
        "words",
        "chars_per_token",
        "tokens_per_word",
        "bytes_per_token",
        "unk_rate",
        "roundtrip_exact_rate",
        "encode_chars_per_sec",
        "encode_ms_per_doc",
    ):
        assert key in m and m[key] is not None, f"{key} missing"
    assert m["chars"] == sum(len(t) for t in texts)
    assert m["tokens"] >= len(texts)
    # Devanagari costs at least one token per character for a byte-level scheme, so chars/token
    # must stay positive and finite for every candidate — a NaN here means the counter broke
    assert 0 < m["chars_per_token"] < 50


def test_special_tokens_are_ids_not_text(tiny_tokenizers):
    spec = json.loads(Path(tiny_tokenizers["byte_bpe"]).read_text(encoding="utf-8"))
    assert [spec["pad_id"], spec["unk_id"], spec["bos_id"], spec["eos_id"]] == [0, 1, 2, 3]
    assert spec["special_tokens"] == SPECIALS


def test_encode_for_model_adds_bos_eos(tiny_tokenizer):
    plain = tiny_tokenizer.encode("भारत")
    with_sp = tiny_tokenizer.encode("भारत", specials=True)
    assert with_sp[0] == tiny_tokenizer.spec.bos_id
    assert with_sp[-1] == tiny_tokenizer.spec.eos_id
    assert with_sp[1:-1] == plain


def test_context_capacity_is_reported(tiny_tokenizer):
    c = tiny_tokenizer.context_tokens(512)
    assert c["probe_tokens"] > 0
    assert c["context_capacity_chars"] == int(512 * c["chars_per_token_probe"]) or c["context_capacity_chars"] > 0


def test_vocab_size_is_actual_not_requested(tiny_tokenizers):
    for kind, spec_path in tiny_tokenizers.items():
        tok = SirTokenizer.load(spec_path)
        requested = json.loads(Path(spec_path).read_text(encoding="utf-8"))["vocab_size"]
        assert tok.vocab_size > 0
        # byte-level BPE initial alphabet is 256 bytes + 4 specials = 260, so a 256-vocab request floors at 260
        assert tok.vocab_size <= requested + 4
        assert tok.vocab_size >= requested or tok.vocab_size >= 256  # at least covers the byte alphabet


def test_missing_artifact_raises_clear_error(tmp_path):
    p = tmp_path / "spec.json"
    p.write_text(json.dumps({"kind": "byte_bpe", "path": "tokenizer.json", "vocab_size": 10, "pad_id": 0, "unk_id": 1, "bos_id": 2, "eos_id": 3}), encoding="utf-8")
    tok = SirTokenizer.load(p) if (tmp_path / "tokenizer.json").exists() else None
    assert tok is None
    with pytest.raises(Exception, match="artifact missing"):
        SirTokenizer.load(p)


def test_read_texts_caps_and_skips_blanks(tmp_path):
    (tmp_path / "train.jsonl").write_text(
        json.dumps({"text": "एक दो तीन चार पाँच छह सात आठ नौ दस"}) + "\n\n" + json.dumps({"text": ""}) + "\n" + json.dumps({"text": "और भी वाक्य यहाँ हैं"}) + "\n",
        encoding="utf-8",
    )
    out = read_texts(tmp_path / "train.jsonl", max_chars=0)
    assert len(out) == 2  # the empty one is skipped
    capped = read_texts(tmp_path / "train.jsonl", max_chars=5)
    assert len(capped) == 1


def test_sha256_file_matches_hashlib(tmp_path):
    import hashlib

    f = tmp_path / "x.bin"
    f.write_bytes(b"hello")
    assert sha256_file(f) == hashlib.sha256(b"hello").hexdigest()


def test_build_slices_covers_all_languages_and_marks_constructed(tmp_path, mini_docs):
    p = tmp_path / "val.jsonl"
    p.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in mini_docs), encoding="utf-8")
    docs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    slices = build_slices(docs, min_chars=1)
    assert {"hi", "en", "hinc-latn", "hinc-deva"} <= set(slices)
    assert "constructed_codeswitch" in slices
    assert all((" " in s) for s in slices["constructed_codeswitch"])


def test_slice_stats_are_provisional_below_threshold(mini_texts, tiny_tokenizer):
    st = slice_stats(tiny_tokenizer, mini_texts[:2], seq_len=32)
    assert st["provisional_slice"] is True
    assert st["tokens"] > 0 and st["with_special_tokens"] > st["tokens"]
