"""Shared fixtures. Tests are hermetic: they build a tiny corpus and tokenizers in a tmp dir and
never touch data/processed or runs/. Nothing here needs a GPU, network, or the real corpus."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HI = [
    "नगर निगम ने जल भराव वाली सड़कों पर पंप लगाने का आदेश दिया है।",
    "विक्रम साराभाई ने अंतरिक्ष कार्यक्रम को विकास से जोड़कर देखा था।",
    "जिले के पंचायत स्कूलों में गणित की कक्षाएँ सुबह सात बजे से शुरू होती हैं।",
    "किसान ने कहा कि सिंचाई की नई नहर से फसल का समय बदल गया है।",
    "मौसम विभाग ने अगले चार दिनों तक भारी बारिश की चेतावनी जारी की है।",
    "प्रयोगशाला में तापमान नियंत्रण बिगड़ने से पूरा नमूना नष्ट हो गया।",
    "पुस्तकालय में हिंदी, अंग्रेज़ी और संस्कृत की पांडुलिपियाँ रखी हैं।",
    "आरटीआई के तहत माँगी गई जानकारी तीस दिन में देना अनिवार्य है।",
]
EN = [
    "The laboratory recorded sensor drift over six weeks before reporting the result.",
    "A negative finding published honestly is worth more than an unreproducible positive one.",
    "The municipal corporation published the tender documents in a machine-readable format.",
    "Disk usage climbed every night because the log rotation policy had never been enabled.",
    "Teachers reported that explaining a concept in the students' home language improved retention.",
    "The migration script was idempotent, which is why re-running it caused no harm.",
]
HINGLAT = [
    "aaj office nahi jaunga, meeting cancel ho gayi thi subah hi.",
    "model train karte waqt loss NaN aa gaya, shayad learning rate zyada hai.",
    "git push karne se pehle tests chalana chahiye tha, ab CI fail ho raha hai.",
    "UPI payment fail hua lekin paisa kat gaya, bank me complaint dalni padegi.",
    "server pe OOM kill aa raha hai, memory leak check karo bhai.",
]
HINGDEV = [
    "मैं अभी report finalize कर रहा हूँ, दस मिनट बाद भेजता हूँ।",
    "server restart करते ही training दोबारा शुरू हो गई, लेकिन checkpoint बचा नहीं था।",
    "यह bug सिर्फ़ Android 13 पर reproduce होता है, emulator पर नहीं।",
    "tokenizer की vocab size बढ़ाने से Hindi tokens 18% कम हो गए, यह number परीक्षण में आया।",
]
BY_LANG = {"hi": HI, "en": EN, "hinc-latn": HINGLAT, "hinc-deva": HINGDEV}


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def mini_texts() -> list[str]:
    return [t for bucket in BY_LANG.values() for t in bucket] * 3


@pytest.fixture(scope="session")
def mini_docs() -> list[dict]:
    """Each doc is ONE sentence so that leakage-safe grouping keeps docs separate.

    The previous sliding-window construction made every doc share sentences with its neighbours,
    which collapsed the entire language into ONE content component and made a deterministic
    leakage-safe split impossible — not a bug in the splitter, but an unrealistic test fixture.
    Real corpora have overlapping sentences, but not *every* document overlapping every other.
    """
    docs = []
    i = 0
    for lang, bucket in BY_LANG.items():
        for j, t in enumerate(bucket):
            docs.append(
                {
                    "id": f"doc{i:04d}",
                    "source_id": "test_source",
                    "language": lang,
                    "topic": f"t{j % 3}",
                    "text": t,
                    "chars": len(t),
                }
            )
            i += 1
    return docs


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory, mini_docs) -> Path:
    d = tmp_path_factory.mktemp("corpus")
    (d / "raw.json").write_text(json.dumps(mini_docs, ensure_ascii=False, indent=1), encoding="utf-8")
    return d


@pytest.fixture(scope="session")
def tiny_tokenizers(tmp_path_factory, mini_texts) -> dict[str, Path]:
    """Train one of each candidate on the mini corpus, once per session (~2s total)."""
    from tokenizer.train_tokenizer import train_one

    out = tmp_path_factory.mktemp("tokenizers")
    specs: dict[str, Path] = {}
    for kind in ("byte_bpe", "sp_bpe", "sp_unigram"):
        rec = train_one(kind, 256, mini_texts, out / kind, sentence_limit=10_000)
        specs[kind] = Path(rec["spec"])
    return specs


@pytest.fixture(scope="session")
def tiny_tokenizer(tiny_tokenizers):
    from tokenizer.api import SirTokenizer

    return SirTokenizer.load(tiny_tokenizers["byte_bpe"])


@pytest.fixture(scope="session")
def tiny_model_config(tiny_tokenizers):
    from model.config import SirModelConfig
    from tokenizer.api import SirTokenizer

    # use the actual tokenizer vocab so ids are always in range; byte-level BPE floors at 260
    tok = SirTokenizer.load(tiny_tokenizers["byte_bpe"])
    return SirModelConfig(vocab_size=tok.vocab_size, d_model=64, n_layers=2, n_heads=4, d_ff=128, max_seq_len=32, dropout=0.0, target_params=None)


@pytest.fixture(scope="session")
def tiny_model(tiny_model_config):
    import torch

    from model.transformer import SirNano

    torch.manual_seed(0)
    return SirNano(tiny_model_config)


@pytest.fixture(scope="session")
def trained_checkpoint(tmp_path_factory, tiny_tokenizers, tiny_model_config, mini_docs) -> Path:
    """A real, tiny training run that checkpoint/inference tests consume (CPU, a couple of seconds)."""
    import numpy as np
    import torch

    from model.transformer import SirNano
    from tokenizer.api import SirTokenizer
    from training import checkpoint as ckpt
    from training.dataset import WindowDataset

    run = tmp_path_factory.mktemp("trained")
    tok = SirTokenizer.load(tiny_tokenizers["byte_bpe"])
    ids: list[int] = []
    for d in mini_docs:
        ids.extend(tok.encode(d["text"]))
        ids.append(tok.spec.eos_id)
    arr = np.asarray(ids, dtype=np.uint16)
    ds = WindowDataset(arr, seq_len=16, ranges=None, shuffle=True, seed=7)

    torch.manual_seed(1)
    model = SirNano(tiny_model_config)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    steps, last = 0, float("nan")
    for x, y in ds:
        opt.zero_grad()
        out = model(x.unsqueeze(0), targets=y.unsqueeze(0))
        out["loss"].backward()
        opt.step()
        last = float(out["loss"].item())
        steps += 1
        if steps >= 12:
            break
    ckpt.save(
        run / "final.pt",
        model,
        model_cfg=tiny_model_config.to_dict(),
        step=steps,
        seed=1,
        val_loss=last,
        tokenizer_spec=tiny_tokenizers["byte_bpe"],
        extra={"provisional": True, "counts": {"params": model.num_parameters()}},
    )
    return run / "final.pt"
