"""Generation promotion gates: PROMOTE only with evidence, otherwise HOLD.

Promotion is the single decision that most easily becomes dishonest, so the gate is built to fail
closed. A generation moves forward only when **every** requirement in its contract is backed by a
result file that exists, covers enough cases, and clears the threshold:

1. every required evaluation axis is present with ≥ `min_cases_per_axis` scored cases;
2. each axis score is ≥ its `min_axis_scores` threshold;
3. the evidence is *verified* evidence (deterministic checks or human review), not model
   self-assessment (`require_verified_evidence`);
4. the leakage report passes (`require_no_leakage`);
5. the reproducibility harness passes (`require_reproducibility`);
6. no axis regressed beyond tolerance against the previous generation (`require_no_axis_regression`).

Anything missing is a `MISSING` criterion, not a pass. The decision object records the exact
thresholds, observed values, case counts and evidence paths, so a promotion claim can be re-audited
without rerunning the argument — and "we think it is much better now" cannot be expressed here.

Parameter count and training loss are deliberately not inputs: they are engineering variables, not
capability evidence.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from development.generations import GenerationContract, load_contracts
from development.regression import RegressionReport, compare_runs, load_run
from sir_paths import rel, resolve, write_json

DECISIONS = ("PROMOTE", "HOLD")
CRITERION_STATUSES = ("MET", "NOT_MET", "MISSING")


@dataclass
class AxisEvidence:
    axis: str
    score: float | None
    cases: int
    source: str = ""
    suite_hash: str = ""
    checkpoint: str = ""
    verified: bool = False
    evaluated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "score": self.score,
            "cases": self.cases,
            "source": self.source,
            "suite_hash": self.suite_hash,
            "checkpoint": self.checkpoint,
            "verified": self.verified,
            "evaluated_at": self.evaluated_at,
        }


@dataclass
class Criterion:
    name: str
    status: str
    detail: str
    required: Any = None
    observed: Any = None
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion": self.name,
            "status": self.status,
            "detail": self.detail,
            "required": self.required,
            "observed": self.observed,
            "evidence": list(self.evidence),
        }


@dataclass
class PromotionDecision:
    generation_id: str
    decision: str
    criteria: list[Criterion] = field(default_factory=list)
    blocking: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    regression: dict[str, Any] | None = None
    decided_utc: str = ""

    @property
    def promoted(self) -> bool:
        return self.decision == "PROMOTE"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": "development/promotion.py",
            "generation_id": self.generation_id,
            "decision": self.decision,
            "decided_utc": self.decided_utc,
            "blocking": self.blocking,
            "warnings": self.warnings,
            "criteria": [c.to_dict() for c in self.criteria],
            "regression": self.regression,
            "note": (
                "PROMOTE means every contract requirement was backed by evidence at the recorded "
                "thresholds. It is not a statement about parameter count, and it does not by itself "
                "verify any capability outside the listed axes."
            ),
        }


def load_axis_evidence(path: Path | str) -> tuple[dict[str, AxisEvidence], dict[str, Any]]:
    """Read a suite result (evaluation/suite.py) into per-axis evidence."""
    p = resolve(path)
    if not p.exists():
        raise FileNotFoundError(f"evidence file not found: {p}")
    doc = json.loads(p.read_text(encoding="utf-8"))
    meta = {
        "path": rel(p),
        "suite_hash": str(doc.get("suite_hash", "")),
        "checkpoint": str(doc.get("checkpoint", "") or ""),
        "verified_evidence": bool(doc.get("verified_evidence", False)),
        "evaluated_at": str(doc.get("created_utc", "") or doc.get("evaluated_at", "")),
    }
    out: dict[str, AxisEvidence] = {}
    for axis, payload in (doc.get("axes") or {}).items():
        if isinstance(payload, dict):
            cases = int(payload.get("scored", payload.get("cases", 0)) or 0)
            score = payload.get("score")
            out[str(axis)] = AxisEvidence(
                axis=str(axis),
                score=None if score is None else float(score),
                cases=cases,
                source=meta["path"],
                suite_hash=meta["suite_hash"],
                checkpoint=meta["checkpoint"],
                verified=meta["verified_evidence"],
                evaluated_at=meta["evaluated_at"],
            )
        elif isinstance(payload, (int, float)):
            out[str(axis)] = AxisEvidence(axis=str(axis), score=float(payload), cases=0, source=meta["path"])
    return out, meta


def _load_flag_report(path: Path | str | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = resolve(path)
    if not p.exists():
        return {"path": rel(p), "present": False, "pass": None}
    doc = json.loads(p.read_text(encoding="utf-8"))
    return {"path": rel(p), "present": True, "pass": doc.get("pass"), "doc": doc}


def evaluate_promotion(
    contract: GenerationContract,
    evidence: dict[str, AxisEvidence],
    *,
    evidence_meta: dict[str, Any] | None = None,
    regression: RegressionReport | None = None,
    leakage: dict[str, Any] | None = None,
    reproducibility: dict[str, Any] | None = None,
    decided_utc: str = "",
) -> PromotionDecision:
    req = contract.promotion_requirements or {}
    min_scores: dict[str, float] = {k: float(v) for k, v in (req.get("min_axis_scores") or {}).items()}
    min_cases = int(req.get("min_cases_per_axis", 1))
    decision = PromotionDecision(generation_id=contract.generation_id, decision="PROMOTE", decided_utc=decided_utc)

    for axis in contract.required_evaluations:
        ev = evidence.get(axis)
        threshold = min_scores.get(axis)
        if ev is None:
            decision.criteria.append(
                Criterion(
                    f"axis:{axis}",
                    "MISSING",
                    f"no evaluation evidence for required axis {axis!r}",
                    required={"min_score": threshold, "min_cases": min_cases},
                    observed=None,
                )
            )
            continue
        if ev.cases < min_cases:
            decision.criteria.append(
                Criterion(
                    f"axis:{axis}",
                    "MISSING",
                    f"{ev.cases} scored case(s) < min_cases_per_axis={min_cases}: too few cases to support a "
                    "promotion claim",
                    required={"min_cases": min_cases},
                    observed={"cases": ev.cases, "score": ev.score},
                    evidence=[ev.source],
                )
            )
            continue
        if ev.score is None:
            decision.criteria.append(
                Criterion(f"axis:{axis}", "MISSING", "cases were scored but no score was recorded", evidence=[ev.source])
            )
            continue
        ok = threshold is None or ev.score >= threshold - 1e-9
        decision.criteria.append(
            Criterion(
                f"axis:{axis}",
                "MET" if ok else "NOT_MET",
                f"score {ev.score:.3f} vs threshold {threshold}" if threshold is not None else f"score {ev.score:.3f}",
                required={"min_score": threshold},
                observed={"score": ev.score, "cases": ev.cases},
                evidence=[ev.source],
            )
        )

    if req.get("require_verified_evidence"):
        unverified = sorted(a for a, e in evidence.items() if a in contract.required_evaluations and not e.verified)
        verified_flag = bool((evidence_meta or {}).get("verified_evidence"))
        ok = verified_flag and not unverified
        decision.criteria.append(
            Criterion(
                "verified_evidence",
                "MET" if ok else "MISSING",
                "evidence declares verified scoring"
                if ok
                else f"evidence is not independently verified (suite verified_evidence={verified_flag}, "
                f"unverified axes={unverified})",
                required=True,
                observed=verified_flag,
                evidence=sorted({e.source for e in evidence.values() if e.source}),
            )
        )

    if req.get("require_no_leakage"):
        if leakage is None or not leakage.get("present"):
            decision.criteria.append(
                Criterion("no_leakage", "MISSING", "no leakage report supplied", required=True, observed=None)
            )
        else:
            passed = leakage.get("pass") is True
            decision.criteria.append(
                Criterion(
                    "no_leakage",
                    "MET" if passed else "NOT_MET",
                    "leakage report passes" if passed else f"leakage report does not pass (pass={leakage.get('pass')})",
                    required=True,
                    observed=leakage.get("pass"),
                    evidence=[leakage.get("path", "")],
                )
            )

    if req.get("require_reproducibility"):
        if reproducibility is None or not reproducibility.get("present"):
            decision.criteria.append(
                Criterion("reproducibility", "MISSING", "no reproducibility report supplied", required=True, observed=None)
            )
        else:
            passed = reproducibility.get("pass") is True
            decision.criteria.append(
                Criterion(
                    "reproducibility",
                    "MET" if passed else "NOT_MET",
                    "reproducibility harness passes" if passed else "reproducibility harness does not pass",
                    required=True,
                    observed=reproducibility.get("pass"),
                    evidence=[reproducibility.get("path", "")],
                )
            )

    if req.get("require_no_axis_regression"):
        if regression is None:
            decision.criteria.append(
                Criterion("no_axis_regression", "MISSING", "no baseline comparison supplied", required=True, observed=None)
            )
        else:
            decision.regression = regression.to_dict()
            ok = regression.verdict == "PASS"
            decision.criteria.append(
                Criterion(
                    "no_axis_regression",
                    "MET" if ok else "NOT_MET",
                    f"regression verdict {regression.verdict}"
                    + (f"; regressed axes: {regression.regressed_axes}" if regression.regressed_axes else ""),
                    required="PASS",
                    observed=regression.verdict,
                )
            )
            if regression.verdict == "INVALID":
                decision.warnings.append(
                    "baseline comparison was INVALID (suite mismatch); fix the baseline before re-running the gate"
                )

    blocking = [c.criterion_name if hasattr(c, "criterion_name") else c.name for c in decision.criteria if c.status != "MET"]
    decision.blocking = [f"{c.name}: {c.detail}" for c in decision.criteria if c.status != "MET"]
    if decision.blocking:
        decision.decision = "HOLD"
    return decision


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate a generation promotion gate.")
    ap.add_argument("--generation", required=True)
    ap.add_argument("--evidence", required=True, help="suite result JSON from evaluation/suite.py")
    ap.add_argument("--baseline", help="previous generation's suite result JSON (for regression check)")
    ap.add_argument("--leakage", help="leakage report JSON (evaluation/results/leakage.json)")
    ap.add_argument("--reproducibility", help="reproducibility report JSON")
    ap.add_argument("--tolerance", type=float, default=None, help="override regression tolerance")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    contracts, _axes = load_contracts()
    if args.generation not in contracts:
        print(f"unknown generation {args.generation!r}; known: {sorted(contracts)}")
        return 2
    contract = contracts[args.generation]
    evidence, meta = load_axis_evidence(args.evidence)
    regression = None
    if args.baseline:
        tol = args.tolerance if args.tolerance is not None else float(
            (contract.promotion_requirements or {}).get("regression_tolerance", 0.02)
        )
        regression = compare_runs(load_run(args.baseline), load_run(args.evidence), tolerance=tol)
    decision = evaluate_promotion(
        contract,
        evidence,
        evidence_meta=meta,
        regression=regression,
        leakage=_load_flag_report(args.leakage),
        reproducibility=_load_flag_report(args.reproducibility),
    )
    print(f"generation: {contract.name} ({contract.generation_id})")
    for c in decision.criteria:
        print(f"  [{c.status:7s}] {c.name}: {c.detail}")
    print(f"decision: {decision.decision}")
    if args.out:
        print(f"written: {rel(write_json(resolve(args.out), decision.to_dict()))}")
    return 0 if decision.promoted else 1


if __name__ == "__main__":
    raise SystemExit(main())
