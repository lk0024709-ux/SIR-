"""Regression protection: a new checkpoint may not trade an old capability for a new one.

The rule this module enforces is uncomfortable but necessary: **an aggregate improvement does not
excuse a per-axis regression.** A model that gains 10 points on mathematics while losing 15 on Hindi
has not improved as an Indian AI, and a promotion gate that only looks at the mean will happily ship
it.

Two further guards:

* **Comparisons must be like-for-like.** If the two runs were scored on different suites (different
  `suite_hash`), the comparison is reported as `INVALID` rather than computed — a score change caused
  by swapping the test set is not a capability change.
* **A missing axis is a regression.** If the candidate never evaluated coding, "no coding result" is
  treated as a loss of the coding evidence, not as a neutral unknown.

Tolerances are explicit and per-axis possible: noise differs between exact-match arithmetic and
heuristic brainstorming scoring, so one global tolerance would be wrong by construction.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sir_paths import rel, resolve, write_json

VERDICTS = ("PASS", "FAIL", "INVALID")


@dataclass
class AxisDelta:
    axis: str
    baseline: float | None
    candidate: float | None
    delta: float | None
    tolerance: float
    verdict: str  # improved | stable | regressed | missing_in_candidate | missing_in_baseline

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "baseline": self.baseline,
            "candidate": self.candidate,
            "delta": self.delta,
            "tolerance": self.tolerance,
            "verdict": self.verdict,
        }


@dataclass
class RegressionReport:
    verdict: str
    axes: list[AxisDelta] = field(default_factory=list)
    suite_hash_baseline: str = ""
    suite_hash_candidate: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def regressed_axes(self) -> list[str]:
        return [a.axis for a in self.axes if a.verdict in ("regressed", "missing_in_candidate")]

    @property
    def improved_axes(self) -> list[str]:
        return [a.axis for a in self.axes if a.verdict == "improved"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": "development/regression.py",
            "verdict": self.verdict,
            "suite_hash_baseline": self.suite_hash_baseline,
            "suite_hash_candidate": self.suite_hash_candidate,
            "improved_axes": self.improved_axes,
            "regressed_axes": self.regressed_axes,
            "notes": self.notes,
            "axes": [a.to_dict() for a in self.axes],
        }


def _extract_axes(doc: dict[str, Any]) -> tuple[dict[str, float], str]:
    """Accept a suite result (evaluation/suite.py) or a flat {axis: score} mapping."""
    if isinstance(doc.get("axes"), dict):
        out: dict[str, float] = {}
        for axis, payload in doc["axes"].items():
            if isinstance(payload, dict) and "score" in payload:
                if payload["score"] is None:
                    continue  # an axis with zero scored cases carries no evidence; it is reported as missing
                out[str(axis)] = float(payload["score"])
            elif isinstance(payload, (int, float)):
                out[str(axis)] = float(payload)
        return out, str(doc.get("suite_hash", ""))
    if isinstance(doc.get("scores"), dict):
        return {str(k): float(v) for k, v in doc["scores"].items()}, str(doc.get("suite_hash", ""))
    flat = {str(k): float(v) for k, v in doc.items() if isinstance(v, (int, float)) and not k.startswith("_")}
    return flat, str(doc.get("suite_hash", ""))


def load_run(path: Path | str) -> tuple[dict[str, float], str]:
    p = resolve(path)
    if not p.exists():
        raise FileNotFoundError(f"evaluation result not found: {p}")
    doc = json.loads(p.read_text(encoding="utf-8"))
    return _extract_axes(doc)


def compare_runs(
    baseline: tuple[dict[str, float], str] | dict[str, Any],
    candidate: tuple[dict[str, float], str] | dict[str, Any],
    *,
    tolerance: float = 0.02,
    per_axis_tolerance: dict[str, float] | None = None,
    require_same_suite: bool = True,
) -> RegressionReport:
    base_axes, base_hash = baseline if isinstance(baseline, tuple) else _extract_axes(baseline)
    cand_axes, cand_hash = candidate if isinstance(candidate, tuple) else _extract_axes(candidate)
    report = RegressionReport(
        verdict="PASS",
        suite_hash_baseline=base_hash,
        suite_hash_candidate=cand_hash,
    )

    if require_same_suite and base_hash and cand_hash and base_hash != cand_hash:
        report.verdict = "INVALID"
        report.notes.append(
            f"suite hashes differ ({base_hash[:12]} vs {cand_hash[:12]}): the runs were scored on different "
            "test sets, so score differences cannot be attributed to the model"
        )
        report.axes = [
            AxisDelta(axis, base_axes.get(axis), cand_axes.get(axis), None, tolerance, "incomparable")
            for axis in sorted(set(base_axes) | set(cand_axes))
        ]
        return report
    if require_same_suite and not (base_hash and cand_hash):
        report.notes.append("one or both runs carry no suite_hash: like-for-like cannot be proven (comparison allowed)")

    per_axis_tolerance = per_axis_tolerance or {}
    for axis in sorted(set(base_axes) | set(cand_axes)):
        tol = float(per_axis_tolerance.get(axis, tolerance))
        b, c = base_axes.get(axis), cand_axes.get(axis)
        if b is None:
            report.axes.append(AxisDelta(axis, None, c, None, tol, "missing_in_baseline"))
            continue
        if c is None:
            report.axes.append(AxisDelta(axis, b, None, None, tol, "missing_in_candidate"))
            continue
        delta = c - b
        if delta < -tol:
            verdict = "regressed"
        elif delta > tol:
            verdict = "improved"
        else:
            verdict = "stable"
        report.axes.append(AxisDelta(axis, b, c, round(delta, 4), tol, verdict))

    regressed = report.regressed_axes
    if regressed:
        report.verdict = "FAIL"
        report.notes.append(
            f"regressed axes: {regressed}. A candidate that improves elsewhere does not compensate: "
            "promotion is blocked until these are explained or fixed."
        )
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare a candidate evaluation run against a baseline.")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--tolerance", type=float, default=0.02)
    ap.add_argument("--ignore-suite-hash", action="store_true", help="compare even if the suites differ (recorded in notes)")
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    report = compare_runs(
        load_run(args.baseline),
        load_run(args.candidate),
        tolerance=args.tolerance,
        require_same_suite=not args.ignore_suite_hash,
    )
    for axis in report.axes:
        print(f"  {axis.axis:14s} {axis.verdict:22s} baseline={axis.baseline} candidate={axis.candidate} delta={axis.delta}")
    print(f"verdict: {report.verdict}")
    for note in report.notes:
        print(f"  note: {note}")
    if args.out:
        print(f"report: {rel(write_json(resolve(args.out), report.to_dict()))}")
    return 0 if report.verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
