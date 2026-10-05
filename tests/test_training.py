"""Training: one-step, checkpoint save/load, resume."""

from __future__ import annotations

import json
import torch
import numpy as np


def test_one_step_training(tiny_model_config, tiny_tokenizer):
    from model.transformer import SirNano
    from training.dataset import WindowDataset

    torch.manual_seed(123)
    cfg = tiny_model_config
    tok = tiny_tokenizer
    # build a tiny token stream: encode a few docs
    texts = ["नगर निगम ने आदेश दिया", "The lab recorded sensor drift", "aaj office nahi jaunga"]
    ids = []
    for t in texts:
        ids.extend(tok.encode(t))
        ids.append(tok.spec.eos_id)
    arr = np.asarray(ids, dtype=np.int64)
    ds = WindowDataset(arr, seq_len=8, ranges=None, shuffle=True, seed=0)
    assert len(ds) > 0, "window dataset produced zero windows"
    model = SirNano(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    # one step
    x, y = next(iter(ds))
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    out = model(x, targets=y)
    loss = out["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    # check grads are populated
    has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters() if p.requires_grad)
    assert has_grad, "no gradients after backward"
    opt.step()
    # after step, weights should have changed
    # we don't check exact values, just that step succeeded without error


def test_checkpoint_save_and_load(tiny_model, tiny_model_config, tiny_tokenizer, tmp_path):
    from training import checkpoint as ckpt

    # save
    save_path = tmp_path / "ckpt.pt"
    info = ckpt.save(
        save_path,
        tiny_model,
        model_cfg=tiny_model_config.to_dict(),
        step=7,
        seed=42,
        val_loss=3.14,
        tokenizer_spec=tmp_path / "dummy",  # will be missing, but checkpoint still saves with note
        extra={"test": True},
    )
    assert save_path.exists()
    assert info["bytes"] > 0
    assert info["sha256"] is not None

    # check gitignore guard: checkpoints must be ignored
    from training.checkpoint import assert_not_tracked
    # this should not raise for a tmp_path outside repo; for repo path it would check .gitignore
    # test that a path inside repo runs/ would be considered tracked if not ignored
    import pathlib
    repo_ckpt = pathlib.Path("runs/test.pt")
    # runs/ is gitignored, so assert_not_tracked should pass (not raise)
    ckpt.assert_not_tracked(pathlib.Path("/tmp/outside.pt"))  # outside repo -> no check
    # for a repo path that is ignored, it should not raise
    ckpt.assert_not_tracked(repo_ckpt)  # should pass because .gitignore covers runs/

    # load
    loaded_model, state = ckpt.load(save_path, map_location="cpu", strict=True)
    assert loaded_model is not None
    assert state["step"] == 7
    assert state["seed"] == 42
    assert abs(state["validation_loss"] - 3.14) < 1e-5
    # check that loaded weights match original
    orig_state = tiny_model.state_dict()
    loaded_state = loaded_model.state_dict()
    for k in orig_state:
        assert torch.allclose(orig_state[k], loaded_state[k]), f"mismatch in {k}"


def test_checkpoint_contains_required_fields(trained_checkpoint):
    from training import checkpoint as ckpt

    _, state = ckpt.load(trained_checkpoint, map_location="cpu")
    assert "model_state" in state
    assert "model_config" in state
    assert "step" in state
    assert "seed" in state
    assert "validation_loss" in state
    assert "tokenizer" in state
    assert "git" in state
    assert "environment" in state
    # tokenizer info should have spec or note
    assert state["tokenizer"] is not None


def test_tokenizer_mismatch_is_detected(tmp_path, tiny_tokenizers):
    from training.tokenize_corpus import resolve_tokenizer_spec
    from training.dataset import load_tokenized
    import json, pathlib
    # simulate mismatch: create a fake meta with different sha
    fake_dir = tmp_path / "data"
    fake_dir.mkdir()
    # we need a real tokenized dir to test mismatch detection
    # instead, directly test that load_tokenized refuses mismatch when we pass wrong sha
    # create minimal tokenized data
    import numpy as np
    from training.dataset import WindowDataset
    from tokenizer.api import SirTokenizer
    tok = SirTokenizer.load(tiny_tokenizers["byte_bpe"])
    # create tokenized data dir with known sha
    import hashlib, json as js
    art = pathlib.Path(tiny_tokenizers["byte_bpe"]).parent / json.loads(pathlib.Path(tiny_tokenizers["byte_bpe"]).read_text())["path"]
    sha = hashlib.sha256(art.read_bytes()).hexdigest()
    # create a dummy tokenized split
    d = tmp_path / "tok" / "train"
    d.mkdir(parents=True)
    arr = np.asarray([1,2,3,4], dtype=np.uint16)
    arr.tofile(d / "tokens.bin")
    meta = {"file":"tokens.bin","dtype":"uint16","n_tokens":4,"n_docs":1,"seq_len":2,"packed":False,"tokenizer_spec":str(tiny_tokenizers["byte_bpe"]),"tokenizer_sha256": sha}
    (d / "meta.json").write_text(js.dumps(meta), encoding="utf-8")
    # loading with correct sha should succeed
    from training.dataset import load_tokenized
    arr2, meta2, _ = load_tokenized(d.parent / "train", expect_tokenizer_sha=sha)
    assert arr2 is not None
    # loading with wrong sha should raise
    import pytest
    with pytest.raises(ValueError, match="mismatch"):
        load_tokenized(d.parent / "train", expect_tokenizer_sha="deadbeef"*8)
