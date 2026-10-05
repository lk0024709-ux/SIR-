"""Tokenize the processed splits into flat token streams, measuring (not estimating) every count.

    python -m training.tokenize_corpus --config configs/sir_nano_smoke.yaml

Guards implemented here, because they are cheaper than a wasted training run:
  * every document is separated by the tokenizer's EOS id and boundary offsets are written beside
    the stream, so `WindowDataset` can avoid packing across documents;
  * the artifact sha256 of the tokenizer is recorded, and `dataset.py` refuses a mismatch;
  * the multilingual probe set must not appear in the training stream (checked by n-gram overlap)
    — training on the generation test would make the test meaningless;
  * token totals are written into data/manifests/acquisition.json. That file is the only place
    corpus sizes may be recorded, and only from a measurement.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sir_paths import rel, resolve  # noqa: E402
from tokenizer.api import SirTokenizer  # noqa: E402
from training.data.manifest import DEFAULT_ACQUISITION  # noqa: E402

SENT = re.compile(r"(?<=[.!?।])\s+")


def resolve_tokenizer_spec(cfg: dict[str, Any], override: str | None) -> Path:
    if override:
        p = resolve(override)
        return p if p.is_file() else p / "spec.json"
    paths = cfg.get("paths") or {}
    tdir = resolve(paths.get("tokenizer_dir", "tokenizer/artifacts/latest"))
    chosen = tdir / "chosen.json"
    if chosen.exists():
        spec = json.loads(chosen.read_text(encoding="utf-8")).get("spec")
        if spec:
            p = resolve(spec)
            if p.exists():
                return p
    raise SystemExit(
        f"no tokenizer selected: {rel(chosen)} is missing. Run the bake-off first — SIR does not "
        "silently fall back to a default tokenizer, because the tokenizer defines the experiment."
    )


def probe_guard(tokenizer: SirTokenizer, train_text: list[str], probe_path: Path, ngram: int = 8) -> dict[str, Any]:
    """Fail if the generation probe set overlaps the training text."""
    if not probe_path.exists():
        return {"checked": False, "reason": f"probe file missing: {rel(probe_path)}"}
    probes = [json.loads(l) for l in probe_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    prompt_texts = [p.get("prompt", "") for p in probes if p.get("prompt")]

    def grams(texts: list[str]) -> set[str]:
        out: set[str] = set()
        for t in texts:
            toks = re.sub(r"\s+", " ", t).strip().lower().split()
            out |= {" ".join(toks[i : i + ngram]) for i in range(0, max(0, len(toks) - ngram + 1))}
        return out

    tg = {tuple(sorted(g)) for g in [grams(train_text)]}
    train_grams = grams(train_text)
    shared: list[dict[str, str]] = []
    for pr, t in zip(probes, prompt_texts):
        for g in grams([t]):
            if g in train_grams:
                shared.append({"probe_id": str(pr.get("id")), "ngram": g})
                break
    return {
        "checked": True,
        "probes": len(probes),
        "ngram": ngram,
        "overlapping_probes": shared,
        "clean": not shared,
    }


def tokenize_file(texts: list[str], tokenizer: SirTokenizer, seq_len: int, eos_id: int, pad_id: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return (flat tokens, doc ranges, stats). Ranges let training keep windows inside documents."""
    chunks: list[np.ndarray] = []
    boundaries: list[int] = []
    per_lang: Counter[str] = Counter()
    pos = 0
    for t in texts:
        ids = tokenizer.encode(t)  # text tokens only; specials are added explicitly below
        arr = np.asarray(ids, dtype=np.int64)
        if arr.size == 0:
            continue
        chunks.append(arr)
        chunks.append(np.asarray([eos_id], dtype=np.int64))
        pos += int(arr.size) + 1
        boundaries.append(pos)
    # (n_docs, 2) [start, end) spans including each document's trailing EOS
    starts_arr = np.asarray([0] + list(boundaries[:-1]), dtype=np.int64) if boundaries else np.asarray([], dtype=np.int64)
    ends_arr = np.asarray(boundaries, dtype=np.int64)
    ranges = np.stack([starts_arr, ends_arr], axis=1) if boundaries else np.zeros((0, 2), dtype=np.int64)
    flat = np.concatenate(chunks) if chunks else np.asarray([], dtype=np.int64)
    dtype = np.uint16 if int(flat.max(initial=0)) < 65535 else np.uint32
    stats = {
        "docs": len([t for t in texts if t]),
        "text_tokens": int(len(flat) - len(boundaries)),  # minus separators
        "total_tokens_including_eos": int(len(flat)),
        "eos_id": int(eos_id),
        "pad_id": int(pad_id),
    }
    return flat.astype(dtype), ranges, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Tokenize processed splits for SIR-Nano training.")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "sir_nano_smoke.yaml"))
    ap.add_argument("--tokenizer-spec", help="explicit spec.json; default: the bake-off's chosen.json")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--allow-mismatch", action="store_true", help="skip the probe-leak guard (forensics only)")
    ap.add_argument("--out", help="override output dir (default: paths.processed_dir from config)")
    args = ap.parse_args(argv)

    from sir_paths import load_config

    cfg = load_config(args.config)
    paths = cfg.get("paths") or {}
    processed = resolve(paths.get("processed_dir", "data/processed/latest"))
    out_dir = resolve(args.out) if args.out else processed / "tokenized"
    seq_len = int((cfg.get("model") or {}).get("max_seq_len", 128))
    eval_cfg = cfg.get("evaluation") or {}
    probes_path = resolve(eval_cfg.get("probes", "evaluation/probes/m1_probe_v0.jsonl"))

    spec_path = resolve_tokenizer_spec(cfg, args.tokenizer_spec)
    tok = SirTokenizer.load(spec_path)
    tok_meta = json.loads(spec_path.read_text(encoding="utf-8"))
    import hashlib

    artifact = spec_path.parent / Path(tok_meta["path"]).name
    tok_sha = hashlib.sha256(artifact.read_bytes()).hexdigest() if artifact.exists() else "missing"

    out_dir.mkdir(parents=True, exist_ok=True)
    ds_report_path = processed / "dataset_report.json"
    ds_report = json.loads(ds_report_path.read_text(encoding="utf-8")) if ds_report_path.exists() else {}

    summary: dict[str, Any] = {
        "tool": "training/tokenize_corpus.py",
        "config": rel(resolve(args.config)),
        "tokenizer": {
            "spec": rel(spec_path),
            "kind": tok.kind,
            "vocab_size": tok.vocab_size,
            "artifact": rel(artifact) if artifact.exists() else "missing",
            "artifact_sha256": tok_sha,
        },
        "seq_len": seq_len,
        "splits": {},
        "corpus_provisional": ds_report.get("provisional", None),
        "notes": "text_tokens counts text pieces only; the EOS separator is counted separately",
    }

    # probe guard runs against the concatenated TRAIN text
    train_path = processed / "train.jsonl"
    if train_path.exists() and not args.allow_mismatch:
        tr_texts = [json.loads(l)["text"] for l in train_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        guard = probe_guard(tok, tr_texts, probes_path, ngram=int(eval_cfg.get("leak_ngram", 8)))
        summary["probe_guard"] = guard
        if not guard.get("clean", True):
            print(
                f"REFUSING to produce a training stream: {len(guard['overlapping_probes'])} probe prompts "
                "overlap the train split. The generation test must stay unseen. Fix the corpus, not the test.",
                file=sys.stderr,
            )
            for g in guard["overlapping_probes"][:5]:
                print(f"  {g['probe_id']}: {g['ngram'][:90]}", file=sys.stderr)
            return 3
    else:
        summary["probe_guard"] = {"checked": False, "reason": "skipped (--allow-mismatch) or no train split"}

    total_tokens = 0
    lang_tokens: Counter[str] = Counter()
    source_ids: set[str] = set()
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        src = processed / f"{split}.jsonl"
        if not src.exists():
            print(f"skip {split}: {rel(src)} not found", file=sys.stderr)
            continue
        docs = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
        flat, bounds, stats = tokenize_file([d["text"] for d in docs], tok, seq_len, tok.spec.eos_id, tok.spec.pad_id)
        sdir = out_dir / split
        sdir.mkdir(parents=True, exist_ok=True)
        flat.tofile(sdir / "tokens.bin")
        np.save(sdir / "ranges.npy", bounds)
        per_lang: Counter[str] = Counter()
        for d in docs:
            per_lang[str(d.get("language", "unknown"))] += len(tok.encode(d["text"]))
            source_ids.add(str(d.get("source_id")))
        lang_tokens.update(per_lang)
        total_tokens += stats["total_tokens_including_eos"]
        meta = {
            "file": "tokens.bin",
            "dtype": "uint16" if flat.dtype == np.uint16 else "uint32",
            "n_tokens": int(flat.size),
            "n_docs": stats["docs"],
            "seq_len": seq_len,
            "packed": False,
            "doc_ranges": int(len(bounds)),
            "tokenizer_spec": rel(spec_path),
            "tokenizer_sha256": tok_sha,
            "tokenizer_kind": tok.kind,
            "source_id_counts": {s: stats["docs"] for s in sorted(source_ids)},
            "language_counts": dict(sorted(per_lang.items())),
            "token_stats": stats,
            "notes": summary["notes"],
        }
        (sdir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        summary["splits"][split] = {k: v for k, v in meta.items() if k not in ("notes",)}
        print(f"{split}: {int(flat.size):,} tokens, {stats['docs']} docs -> {rel(sdir / 'tokens.bin')}")

    # measured corpus size goes to acquisition.json — this is the ONLY sanctioned write of a size
    if total_tokens and source_ids:
        try:
            from training.data.manifest import record_acquisition  # noqa: F401  (kept for clarity)

            acq_path = DEFAULT_ACQUISITION
            doc = json.loads(acq_path.read_text(encoding="utf-8")) if acq_path.exists() else {"records": {}}
            for sid in sorted(source_ids):
                rec = doc["records"].setdefault(sid, {})
                rec["tokens_measured"] = int(total_tokens)
                rec["tokens_measured_by"] = "training/tokenize_corpus.py"
                rec["tokens_measured_at_utc"] = __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime())
            doc["policy_note"] = "token counts here are measured by tokenize_corpus.py; sources.yaml keeps size_tokens null"
            acq_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            summary["acquisition_updated"] = rel(acq_path)
        except Exception as e:  # never block a run on bookkeeping, but say so loudly
            summary["acquisition_error"] = f"{type(e).__name__}: {e}"

    summary["total_tokens"] = total_tokens
    summary["language_token_counts"] = dict(sorted(lang_tokens.items()))
    (out_dir / "tokenization_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"total tokens: {total_tokens:,} | per language: {dict(sorted(lang_tokens.items()))}")
    print(f"summary: {rel(out_dir / 'tokenization_summary.json')}")
    if summary.get("corpus_provisional"):
        print("NOTE: corpus is provisional; these token counts are pipeline-validation numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
