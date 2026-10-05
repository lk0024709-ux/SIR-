"""Common API over the three tokenizer families SIR compares.

Why a wrapper: a tokenizer bake-off is only meaningful if all candidates are asked to do the same
job — same text in, same counting rules, same special tokens, same fidelity measurement. This
module hides the library differences (HuggingFace `tokenizers` vs `sentencepiece`) behind one
small surface:

    tok = SirTokenizer.load("tokenizer/artifacts/smoke/byte_bpe-2048/spec.json")
    ids = tok.encode("भारत")
    text = tok.decode(ids)

Counting conventions used everywhere in SIR (so numbers can be compared across reports):
  * `encode(text, specials=False)` returns ONLY text tokens. Efficiency metrics use this.
  * `encode_for_model(text)` adds <s>/</s> and is what the LM trains on. Both counts are reported.
  * Unknown-token rate counts pieces that decode to the unk piece — for byte-level BPE this must
    be 0 by construction (byte fallback), which is exactly the property worth measuring.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

SPECIALS = ["<pad>", "<unk>", "<s>", "</s>"]
KINDS = ("byte_bpe", "sp_bpe", "sp_unigram")


class TokenizerError(RuntimeError):
    pass


@dataclass
class TokenizerSpec:
    kind: str
    path: str
    vocab_size: int
    special_tokens: list[str] = field(default_factory=lambda: list(SPECIALS))
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "path"} | {"path": str(self.path), "kind": self.kind}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "TokenizerSpec":
        return TokenizerSpec(**{k: v for k, v in d.items() if k in TokenizerSpec.__dataclass_fields__})


# --------------------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------------------
class _HFBpeBackend:
    kind = "byte_bpe"

    def __init__(self, tok: Any):
        self.tok = tok

    def encode_ids(self, text: str) -> list[int]:
        return list(self.tok.encode(text, add_special_tokens=False).ids)

    def decode_ids(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)

    def pieces(self, text: str) -> list[str]:
        return list(self.tok.encode(text, add_special_tokens=False).tokens)

    @property
    def vocab(self) -> int:
        return self.tok.get_vocab_size()


class _SPBackend:
    def __init__(self, sp: Any, kind: str):
        self.sp = sp
        self.kind = kind

    def encode_ids(self, text: str) -> list[int]:
        # sentencepiece 0.2 rejects out_type=list; out_type=int already yields list[int] for a str input
        out = self.sp.Encode(text, out_type=int)
        return list(out) if isinstance(out, (list, tuple)) else [int(out)]

    def decode_ids(self, ids: list[int]) -> str:
        return self.sp.Decode([int(i) for i in ids])

    def pieces(self, text: str) -> list[str]:
        out = self.sp.Encode(text, out_type=str)
        return list(out) if isinstance(out, (list, tuple)) else [str(out)]

    @property
    def vocab(self) -> int:
        return self.sp.GetPieceSize()


class SirTokenizer:
    """Uniform tokenizer object. Construct via `SirTokenizer.load` or `SirTokenizer.from_backend`."""

    def __init__(self, backend: Any, spec: TokenizerSpec):
        self.backend = backend
        self.spec = spec

    # -- constructors ------------------------------------------------------------------
    @classmethod
    def load(cls, spec_path: Path | str) -> "SirTokenizer":
        spec_path = Path(spec_path)
        if spec_path.is_dir():
            spec_path = spec_path / "spec.json"
        if not spec_path.exists():
            raise TokenizerError(f"tokenizer spec not found: {spec_path}")
        spec = TokenizerSpec.from_dict(json.loads(spec_path.read_text(encoding="utf-8")))
        # spec.path stores the artifact file name; it always lives beside spec.json
        art = spec_path.parent / Path(spec.path).name
        if spec.kind == "byte_bpe":
            from tokenizers import Tokenizer

            target = art if art.exists() else spec_path.parent / "tokenizer.json"
            if not target.exists():
                raise TokenizerError(f"byte_bpe artifact missing: {target}")
            return cls(_HFBpeBackend(Tokenizer.from_str(target.read_text(encoding="utf-8"))), spec)
        if spec.kind in ("sp_bpe", "sp_unigram"):
            import sentencepiece as spm

            target = art if art.exists() else spec_path.parent / "tokenizer.model"
            if not target.exists():
                raise TokenizerError(f"sentencepiece artifact missing: {target}")
            sp = spm.SentencePieceProcessor()
            sp.Load(str(target))
            # No encode/decode extra options are set: add_dummy_prefix=false and
            # remove_extra_whitespaces=false were fixed at TRAINING time (see train_tokenizer),
            # so the model already encodes raw text the same way the BPE baseline does.
            if not hasattr(sp, "Encode"):
                raise TokenizerError("sentencepiece processor lacks Encode; unsupported version")
            return cls(_SPBackend(sp, spec.kind), spec)
        raise TokenizerError(f"unknown tokenizer kind {spec.kind!r}")

    # -- core ops ----------------------------------------------------------------------
    @property
    def kind(self) -> str:
        return self.spec.kind

    @property
    def vocab_size(self) -> int:
        return int(self.backend.vocab)

    def encode(self, text: str, specials: bool = False) -> list[int]:
        ids = self.backend.encode_ids(text)
        if specials:
            ids = [self.spec.bos_id, *ids, self.spec.eos_id]
        return ids

    def encode_many(self, texts: Iterable[str], specials: bool = False) -> list[list[int]]:
        return [self.encode(t, specials) for t in texts]

    def decode(self, ids: list[int]) -> str:
        return self.backend.decode_ids(list(ids))

    def pieces(self, text: str) -> list[str]:
        return self.backend.pieces(text)

    def is_unk_piece(self, piece: str) -> bool:
        return piece in ("<unk>", "[UNK]", "⁇")

    # -- facts a report needs ----------------------------------------------------------
    def measure(self, texts: list[str], unk_piece_ids: set[int] | None = None) -> dict[str, Any]:
        """Tokens, character counts, unk rate, round-trip fidelity, encode speed.

        Single implementation used by both the bake-off and the LM evaluation so that the same
        text cannot produce two different token counts in two different reports.
        """
        unk_ids = unk_piece_ids or {self.spec.unk_id}
        total_tokens = 0
        total_chars = 0
        total_words = 0
        unk_tokens = 0
        exact_roundtrip = 0
        bytes_enc = 0
        t0 = time.perf_counter()
        for text in texts:
            ids = self.encode(text)
            toks = len(ids)
            total_tokens += toks
            total_chars += len(text)
            total_words += max(1, len(text.split()))
            bytes_enc += len(text.encode("utf-8"))
            unk_tokens += sum(1 for i in ids if i in unk_ids)
            if toks:
                rt = self.decode(ids)
                if rt == text:
                    exact_roundtrip += 1
        dt = max(1e-9, time.perf_counter() - t0)
        n = max(1, len(texts))
        return {
            "docs": len(texts),
            "tokens": total_tokens,
            "chars": total_chars,
            "words": total_words,
            "utf8_bytes": bytes_enc,
            "chars_per_token": round(total_chars / max(1, total_tokens), 4),
            "tokens_per_char": round(total_tokens / max(1, total_chars), 5),
            "tokens_per_word": round(total_tokens / max(1, total_words), 4),
            "bytes_per_token": round(bytes_enc / max(1, total_tokens), 4),
            "unk_rate": round(unk_tokens / max(1, total_tokens), 8),
            "roundtrip_exact_rate": round(exact_roundtrip / n, 6),
            "encode_chars_per_sec": round(total_chars / dt),
            "encode_ms_per_doc": round(1000 * dt / n, 4),
        }

    def context_tokens(self, seq_len: int) -> dict[str, Any]:
        """A cheap, comparable 'how much text fits in a window' figure.

        chars_per_token alone is misleading: a tokenizer can win it by mapping whole words to
        rare ids and then blow up on unseen text. Capacity-in-window is what actually drives
        training cost and context budget, so it is measured on a fixed probe string for all
        candidates (same string, same rule) rather than inherited from the corpus average.
        """
        probe = "\u092d\u093e\u0930\u0924 \u0915\u093e \u092e\u094c\u0938\u092e \u0905\u091a\u094d\u091b\u093e \u0939\u0948\u0964 The quick brown fox jumps over the lazy dog. " * 20
        ids = self.encode(probe)
        if not ids:
            return {"probe_tokens": 0, "probe_chars": len(probe), "chars_per_token_probe": 0.0, "context_capacity_chars": 0}
        cpt = len(probe) / len(ids)
        return {
            "probe_tokens": len(ids),
            "probe_chars": len(probe),
            "chars_per_token_probe": round(cpt, 4),
            "context_capacity_chars": int(seq_len * cpt),
        }

    def save_spec(self, out_dir: Path, artifact_name: str, extra: dict[str, Any] | None = None) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        spec = dict(self.spec.to_dict())
        spec["path"] = artifact_name
        spec["meta"] = {**spec.get("meta", {}), **(extra or {}), "artifact_sha256": _sha256(out_dir / artifact_name)}
        p = out_dir / "spec.json"
        p.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        return p


def _sha256(path: Path) -> str:
    import hashlib

    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()
