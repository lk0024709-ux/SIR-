"""Hard quality gates for the M1 corpus: G1–G7, plus the M1 verdict.

A gate is a *measurement plus a threshold*, and a failing gate must be able to stop a claim, not
just print a warning. Every gate below reads numbers that the build produced and compares them to
a policy constant defined here, once, next to its justification.

  G1 licence      every document's source is `available` with verified training permission
  G2 provenance   every document has doc_id/source_id/artifact; every source has a pinned revision
  G3 integrity    every acquired artifact re-hashes to the SHA-256 recorded at acquisition
  G4 leakage      zero shared 8-grams between train and validation, zero cross-split near-dupes
  G5 measured     token counts exist, come from tokenization, and carry the tokenizer hash
  G6 language     unknown-language and script-mismatch shares stay under the documented threshold
  G7 reproducible pinned revisions + recorded config/script hashes + normalized files re-hash on disk

The M1 verdict never rounds up: below 10M measured tokens nothing that calls itself a model may be
trained; between 10M and 200M a data-scaled experiment is allowed but M1 is *not* complete.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# --------------------------------------------------------------------------------------
# documented thresholds (policy, not tuning knobs)
# --------------------------------------------------------------------------------------
G6_UNKNOWN_MAX_SHARE = 0.02          # ≤2% of documents may end up with no language label
G6_SCRIPT_MISMATCH_MAX_SHARE = 0.02  # ≤2% may contradict the script their label implies
G4_SHARED_8GRAMS_MAX = 0             # exact: zero
G4_CROSS_SPLIT_NEAR_DUPS_MAX = 0     # exact: zero
M1_TARGET_TOKENS = 200_000_000
M1_MIN_TOKENS_FOR_TRAINING = 10_000_000


@dataclass
class GateResult:
    id: str
    name: str
    passed: bool
    requirement: str
    detail: str
    measured: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _share(n: int, d: int) -> float:
    return (n / d) if d else 0.0


# --------------------------------------------------------------------------------------
# G1 licence
# --------------------------------------------------------------------------------------
def gate_license(manifest: Any, stats: dict[str, Any]) -> GateResult:
    offenders: dict[str, str] = {}
    checked = 0
    for sid, info in sorted((stats.get("per_source") or {}).items()):
        src = manifest.get(sid)
        if src is None:
            offenders[sid] = "not declared in the manifest"
            continue
        raw = src.raw
        checked += 1
        if raw.get("status") != "available":
            offenders[sid] = f"status={raw.get('status')}"
        elif not raw.get("license_verified"):
            offenders[sid] = "license_verified is not true"
        elif not raw.get("model_training_allowed"):
            offenders[sid] = "model_training_allowed is not true"
        elif not str(raw.get("license_evidence") or "").strip():
            offenders[sid] = "no licence evidence recorded"
    passed = not offenders and checked > 0
    return GateResult(
        "G1",
        "licence",
        passed,
        "every training source is available with a verified, recorded training permission",
        "all sources carry a verified training permission" if passed else f"unusable sources: {offenders}",
        {"sources_checked": checked, "offenders": offenders},
    )


# --------------------------------------------------------------------------------------
# G2 provenance
# --------------------------------------------------------------------------------------
def _pinned_revision(rec: dict[str, Any]) -> bool:
    rev = str(rec.get("revision") or "")
    if rev == "local":
        return True
    if rec.get("resolved_commits"):
        return True
    return len(rev) == 40 and all(c in "0123456789abcdef" for c in rev)


def gate_provenance(stats: dict[str, Any]) -> GateResult:
    gaps = stats.get("provenance_gaps") or {}
    unpinned = sorted(sid for sid, rec in (stats.get("acquisition") or {}).items() if not _pinned_revision(rec))
    missing = {k: int(v) for k, v in gaps.items() if int(v)}
    passed = not missing and not unpinned
    return GateResult(
        "G2",
        "provenance",
        passed,
        "every document carries doc_id + source_id + artifact provenance; every source has a pinned revision",
        "every document traces to a pinned artifact"
        if passed
        else f"provenance gaps: documents={missing or 'none'}, sources without a pinned revision={unpinned or 'none'}",
        {"document_gaps": missing, "unpinned_sources": unpinned},
    )


# --------------------------------------------------------------------------------------
# G3 integrity
# --------------------------------------------------------------------------------------
def gate_integrity(verify: list[dict[str, Any]] | None) -> GateResult:
    if not verify:
        return GateResult("G3", "integrity", False, "every acquired artifact re-hashes to its recorded SHA-256", "no artifacts were verified", {"checked": 0})
    bad = [v for v in verify if v.get("status") != "ok"]
    return GateResult(
        "G3",
        "integrity",
        not bad,
        "every acquired artifact re-hashes to its recorded SHA-256",
        f"{len(verify):,} artifacts re-hashed, {len(bad)} mismatch(es)" if not bad else f"{len(bad)} of {len(verify):,} artifacts failed re-hashing: {bad[:3]}",
        {"checked": len(verify), "failures": bad[:20], "failure_count": len(bad)},
    )


# --------------------------------------------------------------------------------------
# G4 leakage
# --------------------------------------------------------------------------------------
def gate_leakage(stats: dict[str, Any]) -> GateResult:
    val = stats.get("validation") or {}
    leak = val.get("leakage_8gram") or {}
    near = val.get("cross_split_near_dups") or {}
    uniq = int(leak.get("unique_overlapping_ngrams", -1))
    occurrences = int(leak.get("overlapping_ngram_occurrences_train", -1))
    pairs = int(near.get("pairs", -1))
    passed = uniq == 0 and occurrences == 0 and pairs == 0
    return GateResult(
        "G4",
        "leakage",
        passed,
        "zero shared 8-grams between train and validation AND zero cross-split near-duplicates",
        f"shared 8-grams: {uniq} unique / {occurrences} occurrences; cross-split near-duplicate pairs: {pairs}",
        {
            "unique_overlapping_ngrams": uniq,
            "overlapping_ngram_occurrences_train": occurrences,
            "cross_split_near_duplicate_pairs": pairs,
            "overlap_rate": leak.get("overlap_rate"),
            "val_ngrams": leak.get("val_ngrams"),
            "train_ngrams": leak.get("train_ngrams"),
        },
    )


# --------------------------------------------------------------------------------------
# G5 measured size
# --------------------------------------------------------------------------------------
def gate_measured_size(stats: dict[str, Any]) -> GateResult:
    tok = stats.get("tokens") or {}
    total = int(tok.get("total") or 0)
    measured = bool(tok.get("measured"))
    has_hash = bool(tok.get("tokenizer_sha256"))
    has_tool = bool(tok.get("tool"))
    passed = measured and has_hash and has_tool and total > 0
    detail = (
        f"{total:,} tokens measured by {tok.get('tool')} with tokenizer {str(tok.get('tokenizer_sha256'))[:12]}…"
        if passed
        else "token counts are missing or were not produced by tokenization"
    )
    return GateResult(
        "G5",
        "measured size",
        passed,
        "token counts were produced by tokenizing the corpus with a recorded tokenizer",
        detail,
        {
            "measured": measured,
            "tool": tok.get("tool"),
            "tokenizer_sha256": tok.get("tokenizer_sha256"),
            "total": total,
            "train": tok.get("train"),
            "validation": tok.get("validation"),
        },
    )


# --------------------------------------------------------------------------------------
# G6 language integrity
# --------------------------------------------------------------------------------------
def gate_language(stats: dict[str, Any]) -> GateResult:
    lang = stats.get("language_quality") or {}
    docs = int(lang.get("documents") or 0)
    unknown = int(lang.get("unknown") or 0)
    mismatch = int(lang.get("script_mismatch") or 0)
    # unknown-language documents are a property of the corpus that was kept; the script check is a
    # cleaning-stage decision, so its denominator is the number of documents the cleaner examined.
    mismatch_base = int(lang.get("documents_seen_cleaning") or docs)
    unknown_share = _share(unknown, docs)
    mismatch_share = _share(mismatch, mismatch_base)
    enough = docs > 0
    passed = enough and unknown_share <= G6_UNKNOWN_MAX_SHARE and mismatch_share <= G6_SCRIPT_MISMATCH_MAX_SHARE
    return GateResult(
        "G6",
        "language integrity",
        passed,
        f"unknown-language share ≤ {G6_UNKNOWN_MAX_SHARE:.0%} and script-mismatch share ≤ {G6_SCRIPT_MISMATCH_MAX_SHARE:.0%}",
        (
            f"unknown {unknown_share:.4%} ({unknown:,}/{docs:,}), "
            f"script mismatch {mismatch_share:.4%} ({mismatch:,}/{mismatch_base:,} documents the cleaner examined)"
            if enough
            else "no documents were measured"
        ),
        {
            "documents": docs,
            "unknown": unknown,
            "unknown_share": round(unknown_share, 6),
            "script_mismatch": mismatch,
            "script_mismatch_share": round(mismatch_share, 6),
            "script_mismatch_base": mismatch_base,
            "unknown_threshold": G6_UNKNOWN_MAX_SHARE,
            "script_mismatch_threshold": G6_SCRIPT_MISMATCH_MAX_SHARE,
        },
    )


# --------------------------------------------------------------------------------------
# G7 reproducibility
# --------------------------------------------------------------------------------------
def gate_reproducibility(stats: dict[str, Any]) -> GateResult:
    rep = stats.get("reproducibility") or {}
    required = ("git_commit", "config_sha256", "seed", "script_sha256")
    missing = [k for k in required if not rep.get(k)]
    unverified = sorted(
        sid for sid, rec in (rep.get("normalized_files") or {}).items() if rec.get("recorded") != rec.get("recomputed")
    )
    passed = not missing and not unverified and bool(rep.get("normalized_files"))
    return GateResult(
        "G7",
        "reproducibility",
        passed,
        "config/script hashes, seed and pinned revisions are recorded; normalized files re-hash on disk",
        "reproducibility record complete and re-verified"
        if passed
        else f"missing={missing or 'none'}, normalized files that no longer match their recorded hash={unverified or 'none'}",
        {
            "missing_fields": missing,
            "unverified_normalized_files": unverified,
            "seed": rep.get("seed"),
            "sources_pinned": len(rep.get("source_revisions") or {}),
        },
    )


def run_gates(manifest: Any, stats: dict[str, Any], verify: list[dict[str, Any]] | None) -> list[GateResult]:
    return [
        gate_license(manifest, stats),
        gate_provenance(stats),
        gate_integrity(verify),
        gate_leakage(stats),
        gate_measured_size(stats),
        gate_language(stats),
        gate_reproducibility(stats),
    ]


# --------------------------------------------------------------------------------------
# verdict
# --------------------------------------------------------------------------------------
def m1_status(stats: dict[str, Any], gates: list[GateResult], *, target_tokens: int = M1_TARGET_TOKENS) -> dict[str, Any]:
    """Map measurements to one verdict. The mapping is fixed policy, so it lives here.

      measured <  10M  -> BLOCKED  (below the floor where anything may be trained)
      measured < 200M  -> PARTIAL  (corpus is real but M1 is not complete)
      measured ≥ 200M  -> PASS or FAIL depending on the gates
    """
    total = int((stats.get("tokens") or {}).get("total") or 0)
    failed = [g.id for g in gates if not g.passed]
    if total < M1_MIN_TOKENS_FOR_TRAINING:
        status = "BLOCKED"
        reason = (
            f"{total:,} measured tokens is below the {M1_MIN_TOKENS_FOR_TRAINING:,} floor; "
            "no training run may be started or reported as a model"
        )
    elif total < target_tokens:
        status = "PARTIAL"
        reason = f"{total:,} measured tokens is below the {target_tokens:,} M1 target; M1 is not complete"
    elif failed:
        status = "FAIL"
        reason = f"{total:,} measured tokens meets the target but gates failed: {failed}"
    else:
        status = "PASS"
        reason = f"{total:,} measured tokens meets the {target_tokens:,} target and all gates pass"
    gates_ok = not failed
    if status == "PASS":
        readiness = {
            "state": "READY FOR M1 TRAINING",
            "ready": True,
            "reason": "measured corpus meets the M1 target and every gate passes",
        }
    elif total >= M1_MIN_TOKENS_FOR_TRAINING and gates_ok:
        readiness = {
            "state": "READY FOR A DATA-SCALED EXPERIMENT (NOT M1)" if status == "PARTIAL" else "BLOCKED BY GATES",
            "ready": status == "PARTIAL",
            "reason": "corpus is above the 10M floor but below the 200M M1 target; only a provisional, data-scaled run is allowed",
        }
    else:
        readiness = {
            "state": "NOT READY",
            "ready": False,
            "reason": (
                f"{total:,} measured tokens is below the 10M floor"
                if total < M1_MIN_TOKENS_FOR_TRAINING
                else f"gates failing: {failed}"
            ),
        }
    return {
        "status": status,
        "reason": reason,
        "measured_tokens": total,
        "target_tokens": target_tokens,
        "min_tokens_for_any_training": M1_MIN_TOKENS_FOR_TRAINING,
        "failing_gates": failed,
        "training_readiness": readiness,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate gates G1–G7 and the M1 verdict against a build's stats.json.")
    ap.add_argument("--stats", required=True)
    ap.add_argument("--manifest", default=str(REPO_ROOT / "data" / "manifests" / "sources.yaml"))
    ap.add_argument("--target-tokens", type=int, default=M1_TARGET_TOKENS)
    ap.add_argument("--as-json", action="store_true")
    args = ap.parse_args(argv)

    from training.data.manifest import load_manifest

    stats = json.loads(Path(args.stats).read_text(encoding="utf-8"))
    manifest = load_manifest(args.manifest)
    verify = (stats.get("integrity") or {}).get("results")
    gates = run_gates(manifest, stats, verify)
    verdict = m1_status(stats, gates, target_tokens=args.target_tokens)
    if args.as_json:
        print(json.dumps({"gates": [g.as_dict() for g in gates], "m1": verdict}, indent=2, ensure_ascii=False))
    else:
        for g in gates:
            print(f"{'PASS' if g.passed else 'FAIL'}  {g.id}  {g.name}: {g.detail}")
        print(f"\nM1: {verdict['status']} — {verdict['reason']}")
        print(f"training readiness: {verdict['training_readiness']['state']} — {verdict['training_readiness']['reason']}")
    return 0 if all(g.passed for g in gates) else 1


if __name__ == "__main__":
    raise SystemExit(main())
