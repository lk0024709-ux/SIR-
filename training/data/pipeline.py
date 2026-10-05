"""The whole data pipeline in one command: manifest gate -> clean -> dedup -> split -> validate.

    python -m training.data.pipeline --source sir_fixture_v0 --config configs/sir_nano_smoke.yaml

Design decisions that matter:
  * inputs come from the *manifest*, so a source with an unverified license or a missing
    acquisition record aborts the run instead of training on it;
  * every stage writes a report and the run writes `provenance.json` (git commit, script
    sha256s, config, counts at each stage). A result whose provenance cannot be regenerated is
    not a result;
  * stage implementations are imported, never copied, so the CLI and the pipeline cannot drift;
  * outputs land under data/{cleaned,processed}, which .gitignore keeps out of the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training.data.clean import CleanConfig, clean_stream, load_documents  # noqa: E402
from training.data.deduplicate import DedupConfig, deduplicate  # noqa: E402
from training.data.manifest import DEFAULT_ACQUISITION, DEFAULT_MANIFEST, ManifestError, load_acquisition, load_manifest, record_acquisition, validate_manifest  # noqa: E402
from training.data.split import SplitConfig, split, write_jsonl  # noqa: E402
from training.data.validate import validate_processed  # noqa: E402

STAGE_FILES = [
    "training/data/manifest.py",
    "training/data/clean.py",
    "training/data/deduplicate.py",
    "training/data/split.py",
    "training/data/validate.py",
    "scripts/build_fixture_corpus.py",
]
# a corpus this small cannot support release-grade metrics; the threshold is a disclosure
# device, not a quality gate — see docs/dataset-policy.md
PROVISIONAL_BELOW_CHARS = 1_000_000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def raw_paths_for(source_id: str, manifest_dir: Path) -> list[Path]:
    """Where a source's raw text lives. Fixture sources are committed; corpora are under data/raw/<id>/."""
    candidates = [
        REPO_ROOT / "data" / "fixtures" / source_id / "docs.json",
        REPO_ROOT / "data" / "raw" / source_id,
        REPO_ROOT / "data" / "raw" / f"{source_id}.jsonl",
        manifest_dir / source_id,
    ]
    for c in candidates:
        if c.is_file():
            return [c]
        if c.is_dir():
            files = sorted(f for f in c.rglob("*") if f.suffix in {".json", ".jsonl", ".txt"})
            if files:
                return files
    return []


def run_pipeline(
    source_ids: list[str],
    out_dir: Path,
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    acquisition_path: Path = DEFAULT_ACQUISITION,
    clean_cfg: CleanConfig | None = None,
    dedup_cfg: DedupConfig | None = None,
    split_cfg: SplitConfig | None = None,
    allow_warnings: bool = True,
    write_provenance: bool = True,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    issues, usable = validate_manifest(manifest, load_acquisition(acquisition_path))
    errors = [i for i in issues if i.level == "error"]
    warnings = [i for i in issues if i.level == "warning"]
    for i in errors + warnings:
        print("  " + i.format(), file=sys.stderr)
    if errors and not allow_warnings:
        raise ManifestError(f"manifest has {len(errors)} error(s); refusing to build a dataset")

    selected = [s for s in source_ids]
    for sid in selected:
        if sid not in usable:
            raise ManifestError(
                f"source {sid!r} is not usable: it is not (status: available AND licensed AND "
                "acquisition-recorded). Declare and record it first — do not bypass this gate."
            )

    clean_cfg = clean_cfg or CleanConfig()
    dedup_cfg = dedup_cfg or DedupConfig()
    split_cfg = split_cfg or SplitConfig()
    out_dir.mkdir(parents=True, exist_ok=True)

    stage: dict[str, Any] = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    all_docs: list[dict[str, Any]] = []
    per_source: dict[str, Any] = {}
    for sid in selected:
        paths = raw_paths_for(sid, (manifest_path.parent / "raw"))
        if not paths:
            raise ManifestError(f"{sid}: declared available but no raw files found under data/raw/{sid} or data/fixtures/{sid}")
        docs: list[dict[str, Any]] = []
        for p in paths:
            docs.extend(load_documents(p))
        kept, rep, rejects = clean_stream(docs, clean_cfg, sid)
        deduped, drep = deduplicate(kept, dedup_cfg)
        (out_dir / f"cleaned_{sid}.jsonl").write_text(
            "\n".join(json.dumps(d, ensure_ascii=False) for d in kept) + "\n", encoding="utf-8"
        )
        per_source[sid] = {
            "clean": rep.as_dict(),
            "dedup": drep.as_dict(),
            "reject_log_entries": len(rejects),
        }
        (out_dir / f"rejects_{sid}.json").write_text(json.dumps(rejects, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        for d in deduped:
            d["source_id"] = sid
        all_docs.extend(deduped)
        stage["raw_files_" + sid] = [str(p.relative_to(REPO_ROOT) if p.is_relative_to(REPO_ROOT) else p) for p in paths]

    if not all_docs:
        raise ManifestError("pipeline produced zero documents — nothing to train on (and zero is a real answer, not a warning)")

    res = split(all_docs, split_cfg)
    write_jsonl(out_dir / "train.jsonl", res.train)
    write_jsonl(out_dir / "val.jsonl", res.val)
    if res.test:
        write_jsonl(out_dir / "test.jsonl", res.test)

    v_issues, v_summary = validate_processed(out_dir / "train.jsonl", out_dir / "val.jsonl")
    v_errors = [i for i in v_issues if i.level == "error"]
    for i in v_issues:
        print("  " + i.format(), file=sys.stderr)

    chars = sum(len(d["text"]) for d in all_docs)
    result = {
        "sources": selected,
        "docs_after_clean": sum(v["clean"]["kept"] for v in per_source.values()),
        "docs_after_dedup": len(all_docs),
        "chars_after_dedup": chars,
        "split": res.as_dict(),
        "validation": {"issues": [i.__dict__ for i in v_issues], **{k: v_summary[k] for k in ("leakage_8gram",)}},
        "per_source": per_source,
        "provisional": chars < PROVISIONAL_BELOW_CHARS,
        "provisional_reason": (
            f"corpus is {chars} chars (< {PROVISIONAL_BELOW_CHARS}); metrics are pipeline-validation only"
            if chars < PROVISIONAL_BELOW_CHARS
            else ""
        ),
        "gate_status": "fail" if v_errors else "pass",
    }

    if write_provenance:
        prov = {
            "tool": "training/data/pipeline.py",
            "git_commit": git_commit(),
            "config": {
                "clean": clean_cfg.__dict__,
                "dedup": dedup_cfg.__dict__,
                "split": split_cfg.__dict__,
            },
            "script_sha256": {f: sha256(REPO_ROOT / f)[:16] for f in STAGE_FILES if (REPO_ROOT / f).exists()},
            "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)[:16], "updated": manifest.updated},
            "manifest_errors": [i.__dict__ for i in errors],
            "manifest_warnings": [i.__dict__ for i in warnings],
            **result,
        }
        prov.update(stage)
        (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")
        (out_dir / "dataset_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _cfg_section(cfg_path: Path | None, name: str) -> dict[str, Any]:
    if not cfg_path or not Path(cfg_path).exists():
        return {}
    import yaml

    doc = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8")) or {}
    sec = doc.get(name) or {}
    return sec if isinstance(sec, dict) else {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the SIR data pipeline for one or more manifest sources.")
    ap.add_argument("--source", action="append", required=True, help="manifest source id (repeatable)")
    ap.add_argument("--config", help="YAML config; sections: dataset.{clean,dedup,split}")
    ap.add_argument("--outdir", default=str(REPO_ROOT / "data" / "processed" / "latest"))
    ap.add_argument("--record", action="store_true", help="measure the input into acquisition.json first")
    ap.add_argument("--strict", action="store_true", help="abort on manifest errors instead of reporting them")
    args = ap.parse_args(argv)

    cfg_path = Path(args.config) if args.config else None
    ds = _cfg_section(cfg_path, "dataset")
    clean_cfg = CleanConfig(**{k: v for k, v in (ds.get("clean") or {}).items() if k in CleanConfig.__dataclass_fields__})
    dedup_cfg = DedupConfig(**{k: v for k, v in (ds.get("dedup") or {}).items() if k in DedupConfig.__dataclass_fields__})
    sp = ds.get("split") or {}
    split_cfg = SplitConfig(
        val_frac=float(sp.get("val_frac", 0.05)),
        test_frac=float(sp.get("test_frac", 0.0)),
        seed=int(sp.get("seed", 20261005)),
        group_by=str(sp.get("group_by", "auto")),
        leak_ngram=int(sp.get("leak_ngram", 8)),
    )

    if args.record:
        for sid in args.source:
            paths = raw_paths_for(sid, (DEFAULT_MANIFEST.parent / "raw"))
            if paths:
                rec = record_acquisition(sid, paths)
                print(f"recorded {sid}: {rec['bytes']} bytes, {rec['files']} files")

    result = run_pipeline(
        args.source,
        Path(args.outdir),
        clean_cfg=clean_cfg,
        dedup_cfg=dedup_cfg,
        split_cfg=split_cfg,
        allow_warnings=not args.strict,
    )
    print(json.dumps({k: result[k] for k in ("sources", "docs_after_clean", "docs_after_dedup", "chars_after_dedup", "provisional", "gate_status")}, indent=2))
    print(f"outputs: {args.outdir}")
    return 0 if result["gate_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
