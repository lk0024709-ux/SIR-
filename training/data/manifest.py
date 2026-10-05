"""SIR dataset manifest: the single gate between "some text exists" and "SIR may train on it".

Why this module exists
----------------------
An AI project's credibility usually dies here, not in the model: a corpus quietly scraped,
a dataset size typed into a README from a guess, a research dataset used without reading its
license. So the pipeline does not accept a list of paths; it accepts a *manifest* and refuses
to proceed on anything it cannot account for.

Enforced gates (see data/manifests/sources.yaml for the human-readable statement)
    G1  every source id used by the pipeline must be declared in the manifest
    G2  license must be a real string, not "UNVERIFIED"/"" -- else the source is unusable
    G3  size_tokens must be null unless status == available AND an acquisition record exists
        (corpus sizes are measured by the fetcher, never typed by a human)
    G4  status == blocked requires blocked_reason; status == rejected requires rejected_reason
    G5  outputs derived from redistribution_allowed: false sources must land in a git-ignored
        directory, so corpus text cannot be committed by accident

Nothing in this module downloads anything. Fetching is a separate, explicit step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifests" / "sources.yaml"
DEFAULT_ACQUISITION = REPO_ROOT / "data" / "manifests" / "acquisition.json"

VALID_STATUSES = ("available", "planned", "blocked", "needs_review", "rejected")
UNUSABLE_LICENSES = {"", "unverified", "unknown", "none", "tbd", "various-per-page", "n/a"}
REQUIRED_FIELDS = (
    "id",
    "name",
    "kind",
    "url",
    "license",
    "languages",
    "natural",
    "size_tokens",
    "redistribution_allowed",
    "status",
    "notes",
)
# Fields a source must declare; optional ones carry defaults below.
OPTIONAL_DEFAULTS: dict[str, Any] = {
    "local_path": None,
    "license_notes": "",
    "download_instructions": "",
    "blocked_reason": "",
    "rejected_reason": "",
    "redistribution_notes": "",
    "derived_from": None,
    "natural_notes": "",
    "size_tokens_note": "",
    "used_for": [],
    "expected_minimum_useful_chars": 0,
    # --- license-record fields added in schema_version 2 -------------------------------
    "provider": "",
    "license_url": None,
    "license_verified": False,
    "license_evidence": "",
    "domain": "",
    "synthetic": False,
    "synthetic_notes": "",
    "acquisition": {},
    "download_method": "",
    "commercial_use": "unknown",
    "derivatives": "unknown",
    "model_training_allowed": False,
    "attribution_required": True,
    "share_alike": "unknown",
    "provenance": "",
    "reason": "",
    "needs_review_reason": "",
    "review_action": "",
}

ERROR = "error"
WARN = "warning"


@dataclass
class Issue:
    level: str
    code: str
    message: str
    source_id: str | None = None

    def format(self) -> str:
        loc = f"[{self.source_id}] " if self.source_id else ""
        return f"{self.level.upper():7s} {loc}{self.code}: {self.message}"


@dataclass
class Source:
    id: str
    raw: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, item: str) -> Any:  # convenience accessor
        raw = self.__dict__.get("raw", {})
        if item in raw:
            return raw[item]
        if item in OPTIONAL_DEFAULTS:
            return OPTIONAL_DEFAULTS[item]
        raise AttributeError(item)

    @property
    def license_ok(self) -> bool:
        lic = str(self.raw.get("license") or "").strip()
        return lic.lower() not in UNUSABLE_LICENSES

    @property
    def usable_for_training(self) -> bool:
        return self.raw.get("status") == "available" and self.license_ok

    @property
    def redistributable(self) -> bool:
        return bool(self.raw.get("redistribution_allowed"))

    @property
    def is_natural(self) -> bool:
        return bool(self.raw.get("natural"))

    def local_dir(self) -> Path | None:
        p = self.raw.get("local_path")
        return (REPO_ROOT / p).resolve() if p else None


@dataclass
class Manifest:
    sources: dict[str, Source]
    policy: dict[str, Any]
    schema_version: int
    updated: str
    path: Path

    def get(self, source_id: str) -> Source | None:
        return self.sources.get(source_id)

    def by_status(self, *statuses: str) -> list[Source]:
        return [s for s in self.sources.values() if s.raw.get("status") in statuses]

    def training_usable(self) -> list[Source]:
        return [s for s in self.sources.values() if s.usable_for_training]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "updated": self.updated,
            "policy": self.policy,
            "sources": [s.raw for s in self.sources.values()],
        }


class ManifestError(RuntimeError):
    """Raised when a hard gate fails. Deliberate: silent fallback would defeat the purpose."""


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------
def load_manifest(path: Path | str = DEFAULT_MANIFEST) -> Manifest:
    path = Path(path).resolve()
    if not path.exists():
        raise ManifestError(f"manifest not found: {path} — SIR refuses to train without one")
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or "sources" not in doc:
        raise ManifestError(f"{path}: top-level 'sources' list is required")
    sources: dict[str, Source] = {}
    for entry in doc["sources"]:
        if not isinstance(entry, dict) or "id" not in entry:
            raise ManifestError(f"{path}: every source entry needs an 'id'")
        sid = str(entry["id"])
        if sid in sources:
            raise ManifestError(f"{path}: duplicate source id {sid!r}")
        for k, v in OPTIONAL_DEFAULTS.items():
            entry.setdefault(k, list(v) if isinstance(v, list) else v)
        sources[sid] = Source(id=sid, raw=entry)
    return Manifest(
        sources=sources,
        policy=doc.get("policy") or {},
        schema_version=int(doc.get("schema_version", 1)),
        updated=str(doc.get("updated", "")),
        path=path,
    )


def load_acquisition(path: Path | str = DEFAULT_ACQUISITION) -> dict[str, Any]:
    path = Path(path).resolve().resolve()
    if not path.exists():
        return {"records": {}}
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------------------
def validate_manifest(
    manifest: Manifest, acquisition: dict[str, Any] | None = None
) -> tuple[list[Issue], dict[str, Source]]:
    """Return (issues, usable_sources). Errors mean "do not train"; warnings mean "disclose"."""
    acquisition = acquisition or {"records": {}}
    records: dict[str, Any] = acquisition.get("records", {})
    issues: list[Issue] = []
    usable: dict[str, Source] = {}

    def err(code: str, msg: str, sid: str | None = None) -> None:
        issues.append(Issue(ERROR, code, msg, sid))

    def warn(code: str, msg: str, sid: str | None = None) -> None:
        issues.append(Issue(WARN, code, msg, sid))

    for sid, src in manifest.sources.items():
        raw = src.raw
        for fld in REQUIRED_FIELDS:
            if fld not in raw:
                err("G0-missing-field", f"required field {fld!r} absent", sid)

        status = str(raw.get("status", ""))
        if status not in VALID_STATUSES:
            err("G4-bad-status", f"status {status!r} not in {VALID_STATUSES}", sid)

        langs = raw.get("languages")
        if not isinstance(langs, list) or not all(isinstance(x, str) and x for x in langs):
            err("G0-bad-languages", "languages must be a non-empty list of BCP-47-ish tags", sid)

        if status == "blocked" and not str(raw.get("blocked_reason") or raw.get("reason") or "").strip():
            err("G4-blocked-needs-reason", "status: blocked requires blocked_reason", sid)
        if status == "rejected" and not str(raw.get("rejected_reason") or raw.get("reason") or "").strip():
            err("G4-rejected-needs-reason", "status: rejected requires rejected_reason", sid)
        if status == "needs_review" and not str(raw.get("needs_review_reason") or raw.get("reason") or "").strip():
            err(
                "G4-needs-review-needs-reason",
                "status: needs_review requires needs_review_reason (what exactly is unverified)",
                sid,
            )

        # G1 — training-permission gate (schema_version 2). "Downloadable" is not permission.
        # A source may only be *available* if the license was read (license_verified) AND that
        # license permits model training (model_training_allowed).
        require_permission = bool((manifest.policy or {}).get("require_training_permission", True))
        if require_permission and status == "available":
            if not bool(raw.get("license_verified")):
                err(
                    "G1-license-not-verified",
                    "status: available requires license_verified: true — quote the license text in "
                    "license_evidence; an unread license is not a license",
                    sid,
                )
            if not bool(raw.get("model_training_allowed")):
                err(
                    "G1-no-training-permission",
                    "status: available requires model_training_allowed: true — publicly accessible "
                    "is not the same as licensed for model training",
                    sid,
                )
            if not str(raw.get("provenance") or "").strip():
                err("G2-no-provenance", "status: available requires a provenance string", sid)
            if not str(raw.get("license_evidence") or "").strip():
                warn(
                    "G1-no-license-evidence",
                    "license_verified: true but license_evidence is empty — say where it was read",
                    sid,
                )

        # G2 — license gate
        if not src.license_ok:
            msg = f"license {raw.get('license')!r} is not a usable license record"
            if status in ("available", "planned"):
                err("G2-unverified-license", msg + f"; status {status!r} cannot be used for training", sid)
            else:
                warn("G2-unverified-license", msg, sid)

        # G3 — measured-sizes gate
        declared_size = raw.get("size_tokens")
        rec = records.get(sid)
        if declared_size is not None:
            err(
                "G3-hand-typed-size",
                "size_tokens must be null in sources.yaml; sizes are measured by the fetcher "
                "into acquisition.json. A typed size is treated as fabricated.",
                sid,
            )
        if status == "available" and rec is None:
            err(
                "G3-missing-acquisition",
                "status: available requires an acquisition record (run the fetcher or "
                "--record-path). Claiming availability without a hash is not accepted.",
                sid,
            )
        if rec is not None and status != "available":
            warn("G3-record-without-availability", "acquisition record exists but status != available", sid)
        if rec is not None:
            for fld in ("sha256", "bytes", "docs", "files"):
                if fld not in rec:
                    err("G3-incomplete-acquisition", f"acquisition record missing {fld!r}", sid)

        # G1 usability rollup
        if status == "available":
            if not src.license_ok:
                err("G2-available-but-unlicensed", "an available source must have a license", sid)
            elif rec is None:
                pass  # already errored above
            else:
                usable[sid] = src

        # G5 non-redistributable text must live in an ignored path
        if not src.redistributable and raw.get("local_path"):
            lp = str(raw["local_path"])
            if not _git_ignored(REPO_ROOT / lp):
                err(
                    "G5-tracked-corpus-path",
                    f"local_path {lp!r} is redistribution-restricted but not git-ignored; "
                    "move it under data/raw|cleaned|processed",
                    sid,
                )

    # cross-cutting warnings worth surfacing every run
    natural_available = [s for s in usable.values() if s.is_natural]
    if not natural_available:
        warn(
            "Q0-no-natural-corpus",
            "no natural (non-fixture) corpus is available, so every metric produced from "
            "this data is pipeline-validation only and must be labelled provisional",
        )
    fixture_only = [s for s in usable.values() if not s.is_natural]
    for s in fixture_only:
        warn(
            "Q1-fixture-only-source",
            f"{s.id} is a composed fixture: acceptable for tests, NOT a release corpus",
            s.id,
        )
    if not str(manifest.updated or ""):
        warn("Q2-no-updated-date", "manifest 'updated' should be set so staleness is visible")

    return issues, usable


def _git_ignored(path: Path) -> bool:
    try:
        out = subprocess.run(
            ["git", "check-ignore", "-q", "--", str(path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return out.returncode == 0
    except FileNotFoundError:  # git unavailable: fail closed
        return False


# --------------------------------------------------------------------------------------
# acquisition records (measured, never typed)
# --------------------------------------------------------------------------------------
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def count_jsonl(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def _git_commit() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
            )
            .stdout.strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def verify_acquisition_integrity(
    only: set[str] | None = None,
    *,
    artifacts_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Re-hash every artifact recorded in the per-source ledgers and compare with the recorded SHA-256.

    This is gate G3: it answers "are the bytes still the bytes that were licensed and measured?".
    A missing file is *not* silently skipped — it is a failure, because a corpus built from files
    that no longer exist cannot be reproduced.
    """
    ledger_dir = artifacts_dir or (REPO_ROOT / "data" / "manifests" / "artifacts")
    out: list[dict[str, Any]] = []
    for ledger in sorted(ledger_dir.glob("*.jsonl")):
        sid = ledger.stem
        if only is not None and sid not in only:
            continue
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("skipped"):
                continue
            path = Path(rec["path"])
            full = path if path.is_absolute() else REPO_ROOT / path
            if not full.exists():
                status, got = "missing", ""
            else:
                got = sha256_file(full)
                status = "ok" if got == rec.get("sha256") else "sha256-mismatch"
            out.append(
                {
                    "source_id": sid,
                    "path": str(path),
                    "recorded": rec.get("sha256"),
                    "recomputed": got,
                    "bytes": rec.get("bytes"),
                    "status": status,
                }
            )
    return out


def record_acquisition(
    source_id: str,
    paths: Iterable[Path | str],
    acquisition_path: Path | str = DEFAULT_ACQUISITION,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure what is actually on disk and write it into acquisition.json.

    Docs/bytes/sha256 are computed here so no human ever has to claim a number.
    """
    paths = [Path(p) for p in paths]
    if not paths:
        raise ManifestError(f"{source_id}: no files to measure")
    total_bytes = 0
    total_docs = 0
    files = 0
    h = hashlib.sha256()
    for p in paths:
        if not p.exists():
            raise ManifestError(f"{source_id}: missing file {p}")
        total_bytes += p.stat().st_size
        h.update(p.name.encode("utf-8"))
        h.update(sha256_file(p).encode("ascii"))
        if p.suffix == ".jsonl":
            total_docs += count_jsonl(p)
        else:
            total_docs += 1
        files += 1
    acq_path = Path(acquisition_path)
    doc = json.loads(acq_path.read_text(encoding="utf-8")) if acq_path.exists() else {"records": {}}
    doc.setdefault("records", {})
    rec: dict[str, Any] = {
        "sha256": h.hexdigest(),
        "bytes": total_bytes,
        "docs": total_docs,
        "files": files,
        "measured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "measured_by_git_commit": _git_commit(),
        "file_list": sorted(str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p) for p in paths),
    }
    rec.update(extra or {})
    doc["records"][source_id] = rec
    doc["policy_note"] = (
        "Generated by training/data/manifest.py. Numbers here are measured, not declared. "
        "token_count is appended by training/data/pipeline.py after tokenization."
    )
    acq_path.parent.mkdir(parents=True, exist_ok=True)
    acq_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rec


def measured_tokens(acquisition_path: Path | str = DEFAULT_ACQUISITION, source_id: str = "") -> None:
    """Called by the pipeline once tokenization has actually counted tokens."""
    acq_path = Path(acquisition_path)
    doc = json.loads(acq_path.read_text(encoding="utf-8")) if acq_path.exists() else {"records": {}}
    doc["records"].setdefault(source_id, {})["tokens_measured"] = True
    acq_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate the SIR dataset manifest (policy gates G1-G5).")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--acquisition", default=str(DEFAULT_ACQUISITION))
    ap.add_argument("--record", nargs=2, metavar=("SOURCE_ID", "PATH..."), action="store", help="measure files into acquisition.json")
    ap.add_argument("--as-json", action="store_true", help="machine-readable report")
    ap.add_argument("--strict-warnings", action="store_true", help="treat warnings as failures")
    args = ap.parse_args(argv)

    if args.record:
        sid, paths = args.record
        paths = paths.split(",")
        rec = record_acquisition(sid, paths, args.acquisition)
        print(f"recorded {sid}: {rec['bytes']} bytes, {rec['docs']} docs, {rec['files']} files")
        print(f"  sha256 {rec['sha256'][:16]}…")
        print("  (now flip status: available in sources.yaml; size_tokens must stay null)")

    manifest = load_manifest(args.manifest)
    issues, usable = validate_manifest(manifest, load_acquisition(args.acquisition))
    errors = [i for i in issues if i.level == ERROR]
    warnings = [i for i in issues if i.level == WARN]

    if args.as_json:
        print(
            json.dumps(
                {
                    "manifest": str(manifest.path),
                    "updated": manifest.updated,
                    "sources": len(manifest.sources),
                    "usable_for_training": sorted(usable),
                    "errors": [asdict(i) for i in errors],
                    "warnings": [asdict(i) for i in warnings],
                },
                indent=2,
            )
        )
    else:
        try:
            shown = manifest.path.relative_to(REPO_ROOT)
        except ValueError:
            shown = manifest.path
        print(f"manifest: {shown} (updated {manifest.updated})")
        print(f"sources: {len(manifest.sources)} | usable for training: {sorted(usable) or 'none'}")
        print(f"by status: " + ", ".join(f"{s}={len(manifest.by_status(s))}" for s in VALID_STATUSES))
        for i in errors + warnings:
            print("  " + i.format())
        if not errors:
            print("GATES: pass — but read the warnings; they are the limits of any result below.")
    if errors or (args.strict_warnings and warnings):
        print(f"RESULT: FAIL ({len(errors)} errors, {len(warnings)} warnings)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
