"""Curriculum coverage: turn *assessment records* into a measurable node state.

The rule this module exists to enforce: **the curriculum graph can never claim learning.** Only
records produced by an actual evaluation, with a real denominator, move a node forward. If nobody
assessed a node, its state is NOT_STARTED no matter how much material was generated for it.

States and their evidence requirements:

| State | Meaning | Requires |
|-------|---------|----------|
| `NOT_STARTED` | no assessment record at all | — |
| `LEARNING` | material seen, too little evidence to judge | ≥1 record |
| `PRACTICE` | attempted, not yet at mastery accuracy/volume | accuracy < mastery or attempts < mastery |
| `ASSESSED` | volume and accuracy met, but mastery evidence incomplete | missing difficulty, transfer, verification or integrity |
| `MASTERED` | all mastery evidence present | difficulty floor, transfer, verification, accuracy |
| `REVIEW_REQUIRED` | was mastered, spaced-review schedule says it is due | mastered + due in review schedule |

Every non-mastered state carries the blocking reasons, so a coverage report answers "what is
missing?" rather than printing a label.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from curriculum.graph import DEFAULT_GRAPH, CurriculumGraph
from sir_paths import REPO_ROOT, load_config, rel, resolve, write_json

DIFFICULTY_LADDER = ("beginner", "elementary", "intermediate", "advanced", "expert", "olympiad")
DEFAULT_THRESHOLDS = REPO_ROOT / "configs" / "curriculum_sampling.yaml"


class CoverageState(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    LEARNING = "LEARNING"
    PRACTICE = "PRACTICE"
    ASSESSED = "ASSESSED"
    MASTERED = "MASTERED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


@dataclass
class AssessmentRecord:
    node_id: str
    kind: str
    correct: int
    total: int
    difficulty: str = "intermediate"
    verified: bool = False
    language: str = "multi"
    source: str = ""
    assessed_utc: str = ""
    notes: str = ""

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AssessmentRecord":
        return cls(
            node_id=str(d.get("node_id", "")),
            kind=str(d.get("kind", "")),
            correct=int(d.get("correct", 0)),
            total=int(d.get("total", 0)),
            difficulty=str(d.get("difficulty", "intermediate")),
            verified=bool(d.get("verified", False)),
            language=str(d.get("language", "multi")),
            source=str(d.get("source", "")),
            assessed_utc=str(d.get("assessed_utc", "")),
            notes=str(d.get("notes", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "correct": self.correct,
            "total": self.total,
            "difficulty": self.difficulty,
            "verified": self.verified,
            "language": self.language,
            "source": self.source,
            "assessed_utc": self.assessed_utc,
            "notes": self.notes,
        }


def load_assessments(path: Path | str) -> list[AssessmentRecord]:
    p = resolve(path)
    if not p.exists():
        raise FileNotFoundError(f"assessment file not found: {p}")
    out: list[AssessmentRecord] = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(AssessmentRecord.from_dict(json.loads(line)))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{rel(p)}:{i}: not valid JSON ({exc})") from exc
    return out


def validate_assessments(records: Iterable[AssessmentRecord], graph: CurriculumGraph | None = None) -> list[str]:
    errs: list[str] = []
    for rec in records:
        if graph is not None and rec.node_id not in graph:
            errs.append(f"{rec.node_id}: assessment refers to a node that is not in the graph")
        if rec.total <= 0:
            errs.append(f"{rec.node_id}: total must be > 0 (a zero-denominator accuracy is not evidence)")
        if rec.correct < 0 or rec.correct > rec.total:
            errs.append(f"{rec.node_id}: correct={rec.correct} outside 0..total={rec.total}")
        if rec.difficulty not in DIFFICULTY_LADDER:
            errs.append(f"{rec.node_id}: difficulty {rec.difficulty!r} not in {DIFFICULTY_LADDER}")
        if not rec.source:
            errs.append(f"{rec.node_id}: assessment needs a source (which suite/harness produced it)")
    return errs


@dataclass
class CoverageThresholds:
    practice_min_attempts: int = 3
    mastery_min_attempts: int = 8
    mastery_accuracy: float = 0.8
    min_difficulty: str = "intermediate"
    require_transfer: bool = True
    require_verification: bool = True
    weak_accuracy: float = 0.6

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "CoverageThresholds":
        cov = (cfg or {}).get("coverage") or {}
        kwargs = {k: v for k, v in cov.items() if k in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        th = cls(**kwargs)
        th.check()
        return th

    def check(self) -> None:
        if not 0.0 <= self.mastery_accuracy <= 1.0:
            raise ValueError("coverage.mastery_accuracy must be within [0, 1]")
        if self.min_difficulty not in DIFFICULTY_LADDER:
            raise ValueError(f"coverage.min_difficulty {self.min_difficulty!r} not in {DIFFICULTY_LADDER}")
        if self.mastery_min_attempts < self.practice_min_attempts:
            raise ValueError("coverage.mastery_min_attempts must be >= practice_min_attempts")


@dataclass
class NodeCoverage:
    node_id: str
    state: CoverageState
    attempts: int = 0
    correct: int = 0
    total: int = 0
    difficulty_reached: str = "none"
    kinds: dict[str, int] = field(default_factory=dict)
    languages: dict[str, int] = field(default_factory=dict)
    verified_evidence: int = 0
    has_transfer: bool = False
    has_verification: bool = False
    blockers: list[str] = field(default_factory=list)
    evidence_sources: list[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float | None:
        return round(self.correct / self.total, 4) if self.total else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "state": self.state.value,
            "attempts": self.attempts,
            "items": self.total,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "difficulty_reached": self.difficulty_reached,
            "kinds": dict(sorted(self.kinds.items())),
            "languages": dict(sorted(self.languages.items())),
            "verified_evidence": self.verified_evidence,
            "has_transfer": self.has_transfer,
            "has_verification": self.has_verification,
            "blockers": self.blockers,
            "evidence_sources": sorted(set(self.evidence_sources)),
        }


def _difficulty_rank(d: str) -> int:
    return DIFFICULTY_LADDER.index(d) if d in DIFFICULTY_LADDER else -1


def compute_node_coverage(
    node_id: str,
    records: list[AssessmentRecord],
    thresholds: CoverageThresholds,
    *,
    review_due: bool = False,
) -> NodeCoverage:
    cov = NodeCoverage(node_id=node_id, state=CoverageState.NOT_STARTED)
    if not records:
        cov.blockers.append("no assessment evidence: state is NOT_STARTED regardless of generated material")
        return cov

    cov.attempts = len(records)
    cov.total = sum(r.total for r in records)
    cov.correct = sum(r.correct for r in records)
    cov.difficulty_reached = max((r.difficulty for r in records), key=_difficulty_rank, default="none")
    cov.verified_evidence = sum(1 for r in records if r.verified)
    for r in records:
        cov.kinds[r.kind] = cov.kinds.get(r.kind, 0) + 1
        cov.languages[r.language] = cov.languages.get(r.language, 0) + 1
        if r.source:
            cov.evidence_sources.append(r.source)
    cov.has_transfer = cov.kinds.get("transfer", 0) > 0
    cov.has_verification = cov.kinds.get("verification", 0) > 0

    acc = cov.accuracy or 0.0
    if cov.attempts < thresholds.practice_min_attempts:
        cov.state = CoverageState.LEARNING
        cov.blockers.append(
            f"{cov.attempts} assessment(s) < practice_min_attempts={thresholds.practice_min_attempts}: "
            "too little evidence to judge"
        )
        return cov

    if cov.attempts < thresholds.mastery_min_attempts:
        cov.blockers.append(
            f"{cov.attempts} assessment(s) < mastery_min_attempts={thresholds.mastery_min_attempts}"
        )
    if acc < thresholds.mastery_accuracy:
        cov.blockers.append(f"accuracy {acc:.2f} < mastery_accuracy {thresholds.mastery_accuracy:.2f}")
    if _difficulty_rank(cov.difficulty_reached) < _difficulty_rank(thresholds.min_difficulty):
        cov.blockers.append(
            f"difficulty_reached {cov.difficulty_reached!r} below floor {thresholds.min_difficulty!r}"
        )
    if thresholds.require_transfer and not cov.has_transfer:
        cov.blockers.append("no transfer assessment recorded (memorisation must not read as mastery)")
    if thresholds.require_verification and not cov.has_verification:
        cov.blockers.append("no verification assessment recorded")
    if cov.verified_evidence == 0:
        cov.blockers.append("no independently verified evidence (all records self-reported)")

    if not cov.blockers:
        cov.state = CoverageState.REVIEW_REQUIRED if review_due else CoverageState.MASTERED
        if review_due:
            cov.blockers.append("spaced-review schedule marks this node due for revisiting")
        return cov

    # enough volume to be past "learning"; distinguish "practising" from "assessed but incomplete"
    if cov.attempts >= thresholds.mastery_min_attempts and acc >= thresholds.mastery_accuracy:
        cov.state = CoverageState.ASSESSED
    else:
        cov.state = CoverageState.PRACTICE
    return cov


@dataclass
class CoverageReport:
    graph_id: str
    thresholds: CoverageThresholds
    nodes: dict[str, NodeCoverage]
    ready_to_learn: list[str] = field(default_factory=list)
    graph_path: str = ""
    method_note: str = (
        "States are computed only from assessment records. MASTERED requires volume, accuracy, a "
        "difficulty floor, transfer evidence, verification evidence and at least one independently "
        "verified record. Nothing here is inferred from generated material."
    )

    @property
    def counts(self) -> dict[str, int]:
        out = {s.value: 0 for s in CoverageState}
        for cov in self.nodes.values():
            out[cov.state.value] += 1
        return out

    def by_subject(self, graph: CurriculumGraph) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for nid, cov in self.nodes.items():
            subject = graph.get(nid).subject
            bucket = out.setdefault(subject, {s.value: 0 for s in CoverageState})
            bucket[cov.state.value] += 1
        return dict(sorted(out.items()))

    def to_dict(self, graph: CurriculumGraph | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "tool": "curriculum/coverage.py",
            "graph": self.graph_path,
            "graph_id": self.graph_id,
            "thresholds": {
                k: getattr(self.thresholds, k) for k in self.thresholds.__dataclass_fields__  # type: ignore[attr-defined]
            },
            "counts": self.counts,
            "ready_to_learn": self.ready_to_learn,
            "nodes": {nid: cov.to_dict() for nid, cov in sorted(self.nodes.items())},
            "method_note": self.method_note,
        }
        if graph is not None:
            payload["by_subject"] = self.by_subject(graph)
        return payload


def build_report(
    graph: CurriculumGraph,
    assessments: Iterable[AssessmentRecord],
    thresholds: CoverageThresholds | None = None,
    *,
    review_due: Iterable[str] = (),
    graph_path: str = "",
) -> CoverageReport:
    thresholds = thresholds or CoverageThresholds()
    thresholds.check()
    grouped: dict[str, list[AssessmentRecord]] = {}
    for rec in assessments:
        grouped.setdefault(rec.node_id, []).append(rec)
    due = set(review_due)
    nodes: dict[str, NodeCoverage] = {}
    for node_id in graph.order():
        nodes[node_id] = compute_node_coverage(
            node_id, grouped.get(node_id, []), thresholds, review_due=node_id in due
        )
    mastered = {nid for nid, cov in nodes.items() if cov.state in (CoverageState.MASTERED, CoverageState.REVIEW_REQUIRED)}
    ready = [
        nid
        for nid in graph.order()
        if nodes[nid].state == CoverageState.NOT_STARTED
        and all(dep in mastered or nodes[dep].state != CoverageState.NOT_STARTED for dep in graph.prerequisites(nid))
    ]
    return CoverageReport(
        graph_id=graph.graph_id,
        thresholds=thresholds,
        nodes=nodes,
        ready_to_learn=ready,
        graph_path=graph_path,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compute curriculum coverage from assessment records.")
    ap.add_argument("--assessments", required=True, help="JSONL of assessment records")
    ap.add_argument("--graph", default=str(DEFAULT_GRAPH))
    ap.add_argument("--config", default=str(DEFAULT_THRESHOLDS), help="config with a `coverage:` section")
    ap.add_argument("--out", default="evaluation/results/curriculum_coverage.json")
    args = ap.parse_args(argv)

    graph = CurriculumGraph.load(resolve(args.graph))
    cfg_path = resolve(args.config)
    thresholds = CoverageThresholds.from_config(load_config(cfg_path)) if cfg_path.exists() else CoverageThresholds()
    assessments = load_assessments(args.assessments)
    errs = validate_assessments(assessments, graph)
    if errs:
        print(f"assessment file has {len(errs)} problem(s):", file=sys.stderr)
        for e in errs[:20]:
            print(f"  - {e}", file=sys.stderr)
        return 2

    from curriculum.review import SpacedReviewScheduler

    review_path = resolve("data/curriculum/review_state.jsonl")
    due: list[str] = []
    if review_path.exists():
        scheduler = SpacedReviewScheduler()
        items = scheduler.load(review_path)
        day = scheduler.latest_day(items)
        due = [i.node_id for i in scheduler.due(items, day)]
    report = build_report(graph, assessments, thresholds, review_due=due, graph_path=rel(resolve(args.graph)))
    out = write_json(resolve(args.out), report.to_dict(graph))
    print(f"coverage: {report.counts}")
    print(f"assessments: {len(assessments)} | ready_to_learn: {len(report.ready_to_learn)}")
    print(f"report: {rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
