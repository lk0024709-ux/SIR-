"""Train every tokenizer candidate on identical text. Never on validation data.

Fairness rules, because a bake-off is only as good as its controls:
  1. All candidates train on exactly the same file (train split only; val is never tokenizer
     training text — the bake-off is scored on val, so training on it would be leakage).
  2. All candidates target the same vocab size and the same special-token ids
     (pad=0, unk=1, bos=2, eos=3).
  3. SentencePiece runs with `normalization_rule_name=identity` and
     `split_by_whitespace=false`, otherwise SP would silently win or lose on normalization and
     space-crossing behaviour instead of subword modelling. Those switches are recorded in the
     manifest so the comparison is auditable.
  4. `character_coverage=1.0` so no Devanagari character is dropped to <unk> by coverage.

    python -m tokenizer.train_tokenizer --config configs/sir_nano_smoke.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tokenizer.api import SPECIALS, SirTokenizer, TokenizerSpec  # noqa: E402

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


@dataclass
class TrainArgs:
    train_path: Path
    out_dir: Path
    candidates: tuple[str, ...]
    vocab_sizes: tuple[int, ...]
    max_chars: int
    sp_sentence_limit: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_path": str(self.train_path),
            "out_dir": str(self.out_dir),
            "candidates": list(self.candidates),
            "vocab_sizes": list(self.vocab_sizes),
            "max_chars": self.max_chars,
            "sp_sentence_limit": self.sp_sentence_limit,
            "seed": self.seed,
        }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_texts(path: Path, max_chars: int) -> list[str]:
    out, total = [], 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            text = json.loads(line).get("text", "")
            if not text:
                continue
            out.append(text)
            total += len(text)
            if max_chars and total >= max_chars:
                break
    return out


def _train_hf_bpe(texts: list[str], vocab: int, out_dir: Path) -> Path:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    # ByteLevel decoder is mandatory: without it decode() returns the intermediate byte-alphabet
    # string ("à¤®à¥‡") instead of text, which would wreck every round-trip number we report.
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab,
        special_tokens=list(SPECIALS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
        limit_alphabet=vocab,
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    target = out_dir / "tokenizer.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(target))
    return target


def _sp_train(texts: list[str], vocab: int, model_type: str, out_dir: Path, sentence_limit: int) -> Path:
    import sentencepiece as spm

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="sir-sp-"))
    corpus = tmp / "train_input.txt"
    corpus.write_text("\n".join(texts) + "\n", encoding="utf-8")
    prefix = out_dir / "tokenizer"
    args = " ".join(
        [
            f"--input={corpus.resolve()}",
            f"--model_prefix={prefix.resolve()}",
            f"--vocab_size={vocab}",
            f"--model_type={model_type}",
            "--character_coverage=1.0",
            "--normalization_rule_name=identity",
            "--split_by_whitespace=false",
            "--add_dummy_prefix=false",
            "--remove_extra_whitespaces=false",
            f"--input_sentence_size={sentence_limit}",
            "--shuffle_input_sentence=false",
            "--minloglevel=1",
            f"--unk_id={UNK_ID}",
            f"--bos_id={BOS_ID}",
            f"--eos_id={EOS_ID}",
            f"--pad_id={PAD_ID}",
            "--unk_piece=<unk>",
        ]
    )
    # SP refuses unknown flags loudly; keep the arg string in the manifest for reproducibility
    (out_dir / "sp_args.txt").write_text(args + "\n", encoding="utf-8")
    try:
        spm.SentencePieceTrainer.Train(args)
    finally:
        corpus.unlink(missing_ok=True)
        try:
            tmp.rmdir()
        except OSError:
            pass
    return prefix.with_suffix(".model")


def train_one(kind: str, vocab: int, texts: list[str], out_dir: Path, sentence_limit: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    clamped_from: int | None = None
    if kind == "byte_bpe":
        artifact = _train_hf_bpe(texts, vocab, out_dir)
    elif kind in ("sp_bpe", "sp_unigram"):
        mt = "bpe" if kind == "sp_bpe" else "unigram"
        try:
            artifact = _sp_train(texts, vocab, mt, out_dir, sentence_limit)
        except Exception as e:
            m = re.search(r"value <= (\d+)", str(e))
            if not m:
                raise
            # SentencePiece cannot always reach the requested vocab on a small corpus. We honour
            # the ceiling, report the clamp in the manifest, and let the evaluator penalise the
            # comparison (a smaller vocab is not an equal-vocab win). Never silently "fix" this.
            clamped_from = vocab
            vocab = int(m.group(1))
            print(f"    note: {kind} vocab clamped {clamped_from} -> {vocab} (corpus too small)")
            artifact = _sp_train(texts, vocab, mt, out_dir, sentence_limit)
    else:  # pragma: no cover - guarded by CLI choices
        raise ValueError(f"unknown tokenizer kind {kind!r}")

    spec = TokenizerSpec(
        kind=kind,
        path=artifact.name,
        vocab_size=vocab,
        special_tokens=list(SPECIALS),
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        meta={
            "trained_on_chars": sum(len(t) for t in texts),
            "docs": len(texts),
            "vocab_clamped_from": clamped_from,
            "equal_vocab": clamped_from is None,
        },
    )
    tok = SirTokenizer.load(_load_stub(out_dir, spec))
    dt = time.perf_counter() - t0
    spec_path = tok.save_spec(out_dir, artifact.name, extra={"train_seconds": round(dt, 3), "vocab_requested": vocab})
    actual = tok.vocab_size
    return {
        "kind": kind,
        "vocab_requested_clamped_from": clamped_from,
        "vocab_requested": vocab,
        "vocab_actual": actual,
        "out_dir": str(out_dir),
        "spec": str(spec_path),
        "train_seconds": round(dt, 3),
        "artifact_bytes": artifact.stat().st_size,
        "artifact_sha256": sha256_file(artifact),
    }


def _load_stub(out_dir: Path, spec: TokenizerSpec) -> Path:
    """Write a provisional spec.json so SirTokenizer.load can find the artifact, then re-save."""
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "spec.json"
    p.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train SIR tokenizer candidates on identical text.")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "sir_nano_smoke.yaml"))
    ap.add_argument("--train", help="override: train jsonl path")
    ap.add_argument("--outdir", help="override: artifact output dir")
    ap.add_argument("--candidates", default=None, help="comma list, default from config")
    ap.add_argument("--vocab-sizes", default=None, help="comma list, default from config")
    ap.add_argument("--max-chars", type=int, default=2_000_000, help="cap training text (0 = no cap)")
    ap.add_argument("--sp-sentence-limit", type=int, default=1_000_000)
    ap.add_argument("--results", default=str(REPO_ROOT / "tokenizer" / "artifacts" / "bakeoff_manifest.json"))
    args = ap.parse_args(argv)

    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    tcfg = cfg.get("tokenizer") or {}
    paths = cfg.get("paths") or {}
    train_path = Path(args.train) if args.train else REPO_ROOT / paths.get("processed_dir", "data/processed/latest") / "train.jsonl"
    out_root = Path(args.outdir) if args.outdir else REPO_ROOT / paths.get("tokenizer_dir", "tokenizer/artifacts/latest")
    if not train_path.exists():
        print(f"train file not found: {train_path}\n  run: python -m training.data.pipeline --source ... --config {args.config}", file=sys.stderr)
        return 2

    candidates = tuple((args.candidates or ",".join(tcfg.get("candidates") or ["byte_bpe", "sp_bpe", "sp_unigram"])).split(","))
    vocab_sizes = tuple(int(v) for v in (args.vocab_sizes or ",".join(str(v) for v in (tcfg.get("vocab_sizes") or [4096]))).split(","))
    for c in candidates:
        if c not in ("byte_bpe", "sp_bpe", "sp_unigram"):
            print(f"unknown candidate {c!r}; choose from byte_bpe|sp_bpe|sp_unigram", file=sys.stderr)
            return 2

    texts = read_texts(train_path, args.max_chars)
    if not texts:
        print("train split contained no text", file=sys.stderr)
        return 2
    print(f"training text: {len(texts)} docs, {sum(len(t) for t in texts)} chars (train split only)")

    out_root.mkdir(parents=True, exist_ok=True)
    records = []
    for vocab in vocab_sizes:
        for kind in candidates:
            d = out_root / f"{kind}-{vocab}"
            print(f"  training {kind} @ {vocab} -> {d}")
            try:
                rec = train_one(kind, vocab, texts, d, args.sp_sentence_limit)
            except Exception as e:  # record the failure instead of pretending the bake-off happened
                rec = {"kind": kind, "vocab_requested": vocab, "error": f"{type(e).__name__}: {e}", "out_dir": str(d)}
                print(f"    FAILED {type(e).__name__}: {e}", file=sys.stderr)
            records.append(rec)

    manifest = {
        "tool": "tokenizer/train_tokenizer.py",
        "config": args.config,
        "args": {
            "train_path": str(train_path),
            "candidates": list(candidates),
            "vocab_sizes": list(vocab_sizes),
            "max_chars": args.max_chars,
            "out_dir": str(out_root),
        },
        "train_file": {"path": str(train_path), "sha256": sha256_file(train_path), "docs": len(texts), "chars": sum(len(t) for t in texts)},
        "git_commit": _git_commit(),
        "rules": {
            "special_token_ids": {"pad": PAD_ID, "unk": UNK_ID, "bos": BOS_ID, "eos": EOS_ID},
            "sentencepiece_flags": ["normalization_rule_name=identity", "split_by_whitespace=false", "character_coverage=1.0", "add_dummy_prefix=false"],
            "trained_on": "train split only; val is reserved for scoring the bake-off",
        },
        "candidates": records,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    outp = Path(args.results)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    ok = [r for r in records if "error" not in r]
    print(f"trained {len(ok)}/{len(records)} candidates -> {outp}")
    for r in records:
        if "error" in r:
            print(f"  ! {r['kind']}@{r['vocab_requested']}: {r['error']}")
    return 0 if ok else 1


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
