"""Token datasets and batching.

The corpus is tokenized ONCE into a flat `uint16`/`uint32` memmap, then training reads fixed-size
windows from it. Two reasons this shape matters more than it looks:

  * the tokenizer is pinned by sha256 in `meta.json`; loading a token stream beside a different
    tokenizer is a silent-corruption class of bug, so it is refused by default (`--allow-mismatch`
    exists for forensics, not convenience);
  * windows never cross document boundaries unless `pack_across_docs: true`, because packing
    unrelated documents together teaches the model a nonexistent "topic jump" and changes loss.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset  # noqa: F401  (DataLoader re-exported for callers)


@dataclass
class DatasetMeta:
    path: Path
    tokenizer_spec: Path
    tokenizer_sha256: str
    dtype: str
    n_tokens: int
    n_docs: int
    seq_len: int
    packed: bool
    source_id_counts: dict[str, int]
    language_counts: dict[str, int]
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(self.__dict__ | {"path": str(self.path), "tokenizer_spec": str(self.tokenizer_spec)}, indent=2)


def load_tokenized(data_dir: Path, expect_tokenizer_sha: str | None = None, allow_mismatch: bool = False) -> tuple[np.memmap, DatasetMeta, np.ndarray | None]:
    """Memory-map a tokenized split and verify it matches the tokenizer it claims to come from."""
    data_dir = Path(data_dir)
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} missing — run python -m training.tokenize_corpus first")
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    bin_path = data_dir / raw["file"]
    if not bin_path.exists():
        raise FileNotFoundError(f"token stream {bin_path} missing — re-run tokenize_corpus")
    if expect_tokenizer_sha and not allow_mismatch:
        recorded = raw.get("tokenizer_sha256")
        if recorded != expect_tokenizer_sha:
            raise ValueError(
                f"tokenizer mismatch: tokenized data used {recorded!r}, the tokenizer you passed is "
                f"{expect_tokenizer_sha!r}. Retokenize instead of trusting the ids."
            )
    dtype = np.uint16 if raw["dtype"] == "uint16" else np.uint32
    arr = np.memmap(bin_path, dtype=dtype, mode="r")
    ranges_path = data_dir / "ranges.npy"
    ranges = np.load(ranges_path) if ranges_path.exists() else None
    meta = DatasetMeta(
        path=bin_path,
        tokenizer_spec=Path(raw.get("tokenizer_spec", "")),
        tokenizer_sha256=raw.get("tokenizer_sha256", "unknown"),
        dtype=raw["dtype"],
        n_tokens=int(raw["n_tokens"]),
        n_docs=int(raw.get("n_docs", 0)),
        seq_len=int(raw.get("seq_len", 0)),
        packed=bool(raw.get("packed", False)),
        source_id_counts=raw.get("source_id_counts", {}),
        language_counts=raw.get("language_counts", {}),
        notes=raw.get("notes", ""),
    )
    return arr, meta, ranges


class WindowDataset(torch.utils.data.IterableDataset):
    """Yields (inputs, targets) windows of `seq_len + 1` tokens from a flat token stream.

    `ranges` is an (n_docs, 2) int64 array of [start, end) spans, one per document, as written by
    training/tokenize_corpus.py. When present, windows are generated strictly INSIDE a document:
    no window ever straddles two unrelated documents. That costs some data efficiency and buys a
    loss number that means what it says.

    Window order is driven by a torch Generator seeded from the run seed, so a rerun with the same
    seed visits the same windows in the same order — the reproducibility criterion depends on it.
    """

    def __init__(
        self,
        tokens,
        seq_len: int,
        ranges: np.ndarray | None = None,
        stride: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        max_windows: int | None = None,
    ):
        self.tokens = np.asarray(tokens, dtype=np.uint32)
        self.seq_len = int(seq_len)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.stride = int(stride or self.seq_len + 1)
        self.starts, self.docs_total, self.docs_too_short = self._build_starts(
            np.asarray(ranges, dtype=np.int64) if ranges is not None else None
        )
        if max_windows:
            self.starts = self.starts[: int(max_windows)]

    def _build_starts(self, ranges: np.ndarray | None) -> tuple[np.ndarray, int, int]:
        need = self.seq_len + 1
        if ranges is None or len(ranges) == 0:
            n = len(self.tokens)
            if n <= need:
                return np.array([], dtype=np.int64), 1, 1
            return np.arange(0, n - need + 1, self.stride, dtype=np.int64), 1, 0
        out: list[int] = []
        short = 0
        for start, end in ranges:
            span = int(end) - int(start)
            if span < need:
                short += 1  # a document shorter than one window cannot produce a clean window
                continue
            out.extend(range(int(start), int(end) - need + 1, self.stride))
        return np.asarray(out, dtype=np.int64), len(ranges), short

    def starvation_warning(self) -> str | None:
        """Tell the operator when the context window is eating the corpus."""
        if not self.docs_total or not self.docs_too_short:
            return None
        frac = self.docs_too_short / self.docs_total
        if frac < 0.2:
            return None
        return (
            f"{self.docs_too_short}/{self.docs_total} documents ({frac:.0%}) are shorter than one "
            f"window (seq_len={self.seq_len}); their text is not trained on at all. Lower seq_len, "
            "merge short documents into a single manifest group at tokenization time, or use a longer corpus."
        )

    def __len__(self) -> int:
        return int(len(self.starts))

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed)
        order = torch.randperm(len(self.starts), generator=g).tolist() if self.shuffle else list(range(len(self.starts)))
        for i in order:
            s = int(self.starts[i])
            chunk = self.tokens[s : s + self.seq_len + 1].astype(np.int64)
            if len(chunk) < self.seq_len + 1:
                continue
            yield torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:])


def collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    xs = torch.stack([b[0] for b in batch])
    ys = torch.stack([b[1] for b in batch])
    return xs, ys


def windows_needed(n_tokens: int, seq_len: int, batch_size: int) -> dict[str, float]:
    """Tokens/step math, exposed so configs can be checked rather than eyeballed."""
    per_step = batch_size * seq_len
    return {
        "tokens_per_step": per_step,
        "windows_available": max(0, (n_tokens - seq_len) // seq_len) if n_tokens > seq_len else 0,
        "steps_per_epoch": math.ceil(max(0, n_tokens - seq_len) / per_step) if n_tokens > seq_len else 0,
        "epochs_for_1_step": round(per_step / max(1, n_tokens), 4),
    }
