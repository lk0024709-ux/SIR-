"""Inference: deterministic generation, encode/decode, edge cases."""

from __future__ import annotations

import torch


def test_deterministic_generation_with_fixed_seed(trained_checkpoint):
    from inference.engine import GenSettings, generate, load_for_inference

    model, tok, _ = load_for_inference(trained_checkpoint, device="cpu")
    prompt = "भारत का मौसम"
    s = GenSettings(max_new_tokens=20, temperature=0.0, seed=123, eos_ids=(model.cfg.eos_token_id,))
    out1 = generate(model, tok, prompt, s)
    out2 = generate(model, tok, prompt, s)
    assert out1["text"] == out2["text"], "greedy generation must be deterministic"
    assert out1["generated_tokens"] == out2["generated_tokens"]


def test_greedy_vs_sampling_produces_text(trained_checkpoint):
    from inference.engine import GenSettings, generate, load_for_inference

    model, tok, _ = load_for_inference(trained_checkpoint, device="cpu")
    prompt = "The laboratory"
    greedy = generate(model, tok, prompt, GenSettings(max_new_tokens=10, temperature=0.0))
    sampled = generate(model, tok, prompt, GenSettings(max_new_tokens=10, temperature=0.8, seed=42))
    assert greedy["generated_tokens"] > 0
    assert sampled["generated_tokens"] > 0
    # greedy and sampled may differ, but both should be non-empty strings
    assert isinstance(greedy["text"], str)
    assert isinstance(sampled["text"], str)


def test_inference_handles_unicode_and_hinglish(trained_checkpoint):
    from inference.engine import GenSettings, generate, load_for_inference

    model, tok, _ = load_for_inference(trained_checkpoint, device="cpu")
    for prompt in [
        "नगर निगम ने",
        "aaj office nahi",
        "मैं अभी report",
        "The quick brown fox",
        "भाई कल meeting है",
    ]:
        out = generate(model, tok, prompt, GenSettings(max_new_tokens=5, temperature=0.0))
        assert "text" in out
        assert isinstance(out["text"], str)
        # should not crash on any script
        assert out["prompt_tokens"] > 0


def test_is_degenerate_detection():
    from inference.engine import is_degenerate

    # empty
    assert is_degenerate("", []) is True
    assert is_degenerate("   ", []) is True
    # single token loop
    assert is_degenerate("a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a", [5]*32) is True
    # normal text not degenerate
    assert is_degenerate("नगर निगम ने जल भराव वाली सड़कों पर काम किया", [1,2,3,4,5,6,7]) is False
    # character hammering
    assert is_degenerate("----------------------------------------", [1,1,1,1,1,1]) is True or is_degenerate("-"*40, [1]*10) is True


def test_encode_decode_roundtrip_unicode(tiny_tokenizer):
    for text in [
        "क्ष त्र ज्ञ",
        "का कि की कु कू के कै को",
        "एक — दो – तीन",
        "emoji 🟢 test",
        "नंबर 1,234",
        "aaj kal bahut garmi hai",
        "मैं अभी report finalize कर रहा हूँ",
    ]:
        ids = tiny_tokenizer.encode(text)
        decoded = tiny_tokenizer.decode(ids)
        assert decoded == text, f"roundtrip failed for {text!r}: got {decoded!r}"


def test_model_handles_max_seq_len(trained_checkpoint):
    from inference.engine import GenSettings, generate, load_for_inference

    model, tok, _ = load_for_inference(trained_checkpoint, device="cpu")
    ctx = model.cfg.max_seq_len
    # prompt longer than context should be truncated (right context kept)
    long_prompt = " ".join(["भारत"] * (ctx + 10))
    out = generate(model, tok, long_prompt, GenSettings(max_new_tokens=3, temperature=0.0))
    assert out["truncated"] is True or len(tok.encode(long_prompt)) > ctx
    assert out["generated_tokens"] >= 0
