"""Experiment tracking: an experiment without a record is an anecdote.

Every training run that could be cited in a report must be reconstructable from a committed record:
which git commit, which dataset (and its content hash), which tokenizer, which model config, how many
parameters, how many steps and tokens, which seed, on what hardware, with what software versions, what
losses, what evaluation results and — the field most often missing — **what failed**.

The record is produced from artifacts that already exist (`runs/*/train_log.json`, the tokenized
dataset's `meta.json`, the tokenizer spec), so filling it in is mechanical rather than a chore that
gets skipped. Writing a record is not the same as claiming a result: `status` and `known_failures`
are part of the record, and a run whose evaluation is missing says so.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sir_paths import REPO_ROOT, rel, resolve, write_json

RECORD_STATUSES = ("planned", "running", "completed", "failed", "aborted", "superseded")


def file_sha256(path: Path | str, chunk: int = 1 << 20) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def tree_fingerprint(paths: list[Path | str]) -> dict[str, Any]:
    """Hash a set of files (or a directory) without assuming a layout."""
    entries: list[tuple[str, str]] = []
    total_bytes = 0
    for raw in paths:
        p = resolve(raw)
        if p.is_dir():
            files = sorted(f for f in p.rglob("*") if f.is_file())
        elif p.is_file():
            files = [p]
        else:
            continue
        for f in files:
            entries.append((rel(f), file_sha256(f)))
            total_bytes += f.stat().st_size
    blob = json.dumps(entries, sort_keys=True).encode("utf-8")
    return {
        "files": len(entries),
        "bytes": total_bytes,
        "tree_sha256": hashlib.sha256(blob).hexdigest(),
        "detail": entries[:50],
    }


def git_info(repo: Path | str = REPO_ROOT) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=str(repo), capture_output=True, text=True, timeout=20, check=False
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit or "unknown",
        "branch": run("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "dirty": bool(status),
        "dirty_files": len(status.splitlines()) if status else 0,
    }


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }
    try:  # memory is optional; a missing number must not break a run
        import resource

        info["rss_limit_kb"] = resource.getrlimit(resource.RLIMIT_AS)[0]
    except Exception:  # pragma: no cover
        pass
    try:  # torch is heavy; only report CUDA when torch is importable
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        info["torch"] = "not importable in this environment"
        info["cuda_available"] = None
    return info


def software_versions() -> dict[str, str]:
    out: dict[str, str] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("torch", "numpy", "yaml", "sentencepiece", "tokenizers"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            out[mod] = "not installed"
    return out


def dataset_fingerprint(processed_dir: Path | str) -> dict[str, Any]:
    """Reads the tokenized dataset's meta.json and hashes the token streams it points at."""
    d = resolve(processed_dir)
    meta_path = d / "meta.json"
    out: dict[str, Any] = {"processed_dir": rel(d), "present": meta_path.exists()}
    if not meta_path.exists():
        return out
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    out.update(
        {
            "file": meta.get("file"),
            "dtype": meta.get("dtype"),
            "n_tokens": meta.get("n_tokens"),
            "n_docs": meta.get("n_docs"),
            "seq_len": meta.get("seq_len"),
            "packed": meta.get("packed"),
            "tokenizer_sha256": meta.get("tokenizer_sha256"),
        }
    )
    files = [d / meta["file"]] if meta.get("file") else []
    files += [d / "ranges.npy"] if (d / "ranges.npy").exists() else []
    out["hash"] = tree_fingerprint(files)
    return out


@dataclass
class ExperimentRecord:
    experiment_id: str
    created_utc: str
    status: str
    git: dict[str, Any]
    dataset: dict[str, Any]
    tokenizer: dict[str, Any]
    model: dict[str, Any]
    training: dict[str, Any]
    evaluation: dict[str, Any] = field(default_factory=dict)
    known_failures: list[str] = field(default_factory=list)
    notes: str = ""
    config_path: str = ""
    artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "created_utc": self.created_utc,
            "status": self.status,
            "git": self.git,
            "dataset": self.dataset,
            "tokenizer": self.tokenizer,
            "model": self.model,
            "training": self.training,
            "evaluation": self.evaluation,
            "known_failures": list(self.known_failures),
            "notes": self.notes,
            "config_path": self.config_path,
            "artifacts": list(self.artifacts),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentRecord":
        return cls(
            experiment_id=str(d.get("experiment_id", "")),
            created_utc=str(d.get("created_utc", "")),
            status=str(d.get("status", "")),
            git=dict(d.get("git") or {}),
            dataset=dict(d.get("dataset") or {}),
            tokenizer=dict(d.get("tokenizer") or {}),
            model=dict(d.get("model") or {}),
            training=dict(d.get("training") or {}),
            evaluation=dict(d.get("evaluation") or {}),
            known_failures=[str(x) for x in (d.get("known_failures") or [])],
            notes=str(d.get("notes", "")),
            config_path=str(d.get("config_path", "")),
            artifacts=[str(x) for x in (d.get("artifacts") or [])],
        )

    def validate(self) -> list[str]:
        errs: list[str] = []
        if not self.experiment_id:
            errs.append("experiment_id is required")
        if self.status not in RECORD_STATUSES:
            errs.append(f"status {self.status!r} not in {RECORD_STATUSES}")
        if not self.git.get("commit"):
            errs.append("git.commit is required (an experiment with no commit cannot be reproduced)")
        if not self.dataset.get("hash", {}).get("tree_sha256"):
            errs.append("dataset.hash.tree_sha256 is required (unhashed data cannot be re-identified)")
        if not self.tokenizer.get("sha256") and not self.tokenizer.get("spec"):
            errs.append("tokenizer identity is required (spec path and/or sha256)")
        if self.model.get("parameter_count") in (None, 0):
            errs.append("model.parameter_count is required (the measured count, not the target)")
        for key in ("steps", "tokens_seen", "seed"):
            if self.training.get(key) is None:
                errs.append(f"training.{key} is required")
        if self.status == "completed" and not self.evaluation:
            errs.append(
                "status=completed with no evaluation results: record the evaluation, or mark the run "
                "aborted/failed — a completed training run with no evaluation cannot support any claim"
            )
        if self.status in ("failed", "aborted") and not self.known_failures:
            errs.append(f"status={self.status} requires known_failures (what went wrong is the point of the record)")
        return errs


def write_record(record: ExperimentRecord, directory: Path | str = "experiments/records") -> Path:
    errs = record.validate()
    if errs:
        raise ValueError("refusing to write an invalid experiment record:\n  - " + "\n  - ".join(errs))
    d = resolve(directory)
    d.mkdir(parents=True, exist_ok=True)
    return write_json(d / f"{record.experiment_id}.json", record.to_dict())


def load_records(directory: Path | str = "experiments/records") -> list[ExperimentRecord]:
    d = resolve(directory)
    if not d.exists():
        return []
    return [ExperimentRecord.from_dict(json.loads(p.read_text(encoding="utf-8"))) for p in sorted(d.glob("*.json"))]


def from_train_log(
    train_log: Path | str,
    *,
    processed_dir: Path | str | None = None,
    tokenizer_spec: Path | str | None = None,
    evaluation_results: dict[str, Any] | None = None,
    known_failures: list[str] | None = None,
    status: str = "completed",
    created_utc: str = "",
    notes: str = "",
) -> ExperimentRecord:
    """Build a record from a real training run's `train_log.json` (plus its dataset directory)."""
    p = resolve(train_log)
    if not p.exists():
        raise FileNotFoundError(f"train log not found: {p}")
    log = json.loads(p.read_text(encoding="utf-8"))
    data = log.get("data") or {}
    counts = log.get("counts") or {}
    model_cfg = log.get("model_config") or {}

    processed = processed_dir
    if processed is None:
        recorded = (log.get("data") or {}).get("processed_dir")
        processed = recorded or (Path(p).parent.parent / "data" / "processed" / "smoke")
    ds = dataset_fingerprint(processed) if processed else {"present": False, "processed_dir": None}

    spec = tokenizer_spec or (log.get("tokenizer") or {}).get("tokenizer_spec") or log.get("tokenizer")
    tok: dict[str, Any] = {"spec": str(spec) if spec else ""}
    if spec and resolve(spec).exists():
        tok["sha256"] = file_sha256(resolve(spec))

    seed = log.get("seed")
    config_sha = str(log.get("config_sha256", ""))
    exp_name = str(log.get("experiment", "experiment"))
    experiment_id = f"{exp_name}-s{seed}-c{config_sha[:8] or 'noconfig'}" if seed is not None else exp_name

    training = {
        "steps": log.get("tokens_seen") is not None and (log.get("best") or {}).get("step") or None,
        "tokens_seen": log.get("tokens_seen"),
        "seed": seed,
        "batch": log.get("batch"),
        "optimizer": log.get("optimizer"),
        "schedule": log.get("schedule"),
        "wallclock_seconds": log.get("wallclock_seconds"),
        "tokens_per_sec": log.get("mean_train_tokens_per_sec"),
        "peak_rss_mb": log.get("peak_rss_mb"),
    }
    steps = max((entry.get("step", 0) for entry in (log.get("log") or [])), default=0)
    training["steps"] = steps or None

    final = log.get("final_eval") or {}
    evaluation = {
        "trainer_val_loss": final.get("val_loss"),
        "trainer_val_perplexity": final.get("val_perplexity"),
        "tokens_scored": final.get("tokens_scored"),
        "external_results": evaluation_results or {},
        "train_tokens": data.get("train_tokens"),
        "val_tokens": data.get("val_tokens"),
        "language_token_counts": data.get("language_token_counts"),
        "provisional_corpus": data.get("provisional"),
    }

    return ExperimentRecord(
        experiment_id=experiment_id,
        created_utc=created_utc,
        status=status,
        git=log.get("git") or git_info(),
        dataset=ds,
        tokenizer=tok,
        model={
            "config": model_cfg,
            "config_sha256": config_sha,
            "parameter_count": counts.get("params", {}).get("total"),
            "parameter_breakdown": counts.get("params"),
        },
        training=training,
        evaluation=evaluation,
        known_failures=list(known_failures or []),
        notes=notes or str(log.get("status_note", "")),
        config_path=str(log.get("config", "")),
        artifacts=[rel(p)],
    )


def index(records: list[ExperimentRecord]) -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": r.experiment_id,
            "created_utc": r.created_utc,
            "status": r.status,
            "commit": r.git.get("commit", "")[:12],
            "dataset_sha256": (r.dataset.get("hash") or {}).get("tree_sha256", "")[:12],
            "params": r.model.get("parameter_count"),
            "steps": r.training.get("steps"),
            "tokens_seen": r.training.get("tokens_seen"),
            "seed": r.training.get("seed"),
            "val_loss": (r.evaluation or {}).get("trainer_val_loss"),
            "known_failures": len(r.known_failures),
        }
        for r in sorted(records, key=lambda r: r.experiment_id)
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Create/inspect SIR experiment records.")
    ap.add_argument("--from-train-log", help="path to runs/<run>/train_log.json")
    ap.add_argument("--processed-dir", help="tokenized dataset directory (default: inferred from the log)")
    ap.add_argument("--tokenizer-spec", help="tokenizer spec path")
    ap.add_argument("--status", default="completed", choices=RECORD_STATUSES)
    ap.add_argument("--failure", action="append", default=[], help="known failure note (repeatable)")
    ap.add_argument("--notes", default="")
    ap.add_argument("--out", default="experiments/records")
    ap.add_argument("--index", action="store_true", help="print the index of existing records")
    args = ap.parse_args(argv)

    if args.index or not args.from_train_log:
        records = load_records(args.out)
        print(json.dumps(index(records), indent=2))
        return 0

    rec = from_train_log(
        args.from_train_log,
        processed_dir=args.processed_dir,
        tokenizer_spec=args.tokenizer_spec,
        known_failures=args.failure,
        status=args.status,
        notes=args.notes,
    )
    errs = rec.validate()
    if errs:
        print("record is incomplete; nothing was written:")
        for e in errs:
            print(f"  - {e}")
        return 1
    path = write_record(rec, args.out)
    print(f"record: {rel(path)} (experiment_id={rec.experiment_id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
