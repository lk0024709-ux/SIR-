"""Check a curriculum graph against the written requirement contract.

`configs/curriculum_requirements.yaml` states what the curriculum must contain (topics, grade
coverage, ordered sequences, the learning loop, generation order). This module answers one
question: *does the graph actually deliver it?*

The separation is deliberate. A requirement file that nobody checks is documentation, and a graph
that nobody checks drifts. `python -m curriculum.requirements` is the check, and
`tests/test_curriculum_requirements.py` runs it in CI, so a topic can only be dropped by editing
the contract in the same commit — which makes the change visible in review.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from curriculum.graph import DEFAULT_GRAPH, CurriculumGraph
from sir_paths import REPO_ROOT, load_config, rel, resolve, write_json

DEFAULT_REQUIREMENTS = REPO_ROOT / "configs" / "curriculum_requirements.yaml"


@dataclass
class RequirementReport:
    requirements_path: str
    graph_path: str
    errors: list[str] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)
    ok: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": "curriculum/requirements.py",
            "ok": self.ok,
            "requirements": self.requirements_path,
            "graph": self.graph_path,
            "checks": self.checks,
            "errors": self.errors,
        }


def check_requirements(graph: CurriculumGraph, req: dict[str, Any]) -> RequirementReport:
    report = RequirementReport(
        requirements_path=str(DEFAULT_REQUIREMENTS),
        graph_path=str(graph.source_path or DEFAULT_GRAPH),
    )
    declared_subjects = set(graph.subjects)

    # 1. required topics per subject ------------------------------------------------
    for subject, topics in sorted((req.get("required_topics") or {}).items()):
        if subject not in declared_subjects:
            report.errors.append(f"required subject {subject!r} is not declared in the graph")
            continue
        available = graph.topics(subject)
        missing = sorted(set(topics) - available)
        report.checks.append(
            {
                "check": "required_topics",
                "subject": subject,
                "required": len(topics),
                "missing": missing,
                "status": "pass" if not missing else "fail",
            }
        )
        if missing:
            report.errors.append(
                f"{subject}: graph exposes no node topic for {missing} "
                "(add a topic tag to the node that covers it, or the requirement is unmet)"
            )

    # 2. cross-cutting capability topics --------------------------------------------
    cap_topics = set(req.get("required_capability_topics") or [])
    if cap_topics:
        all_topics = graph.topics()
        missing = sorted(cap_topics - all_topics)
        report.checks.append(
            {
                "check": "required_capability_topics",
                "required": len(cap_topics),
                "missing": missing,
                "status": "pass" if not missing else "fail",
            }
        )
        if missing:
            report.errors.append(f"cross-cutting capability topics missing from the graph: {missing}")

    # 3. grade coverage -------------------------------------------------------------
    for subject, span in sorted((req.get("required_grade_coverage") or {}).items()):
        lo, hi = int(span[0]), int(span[1])
        have = sorted(
            int(n.grade)
            for n in graph.nodes.values()
            if n.subject == subject and n.track == "foundation" and isinstance(n.grade, int)
        )
        missing = [g for g in range(lo, hi + 1) if g not in have]
        report.checks.append(
            {
                "check": "grade_coverage",
                "subject": subject,
                "span": [lo, hi],
                "missing_grades": missing,
                "status": "pass" if not missing else "fail",
            }
        )
        if missing:
            report.errors.append(f"{subject}: no foundation node for grade(s) {missing}")

    # 4. ordered sequences must be real dependency chains ---------------------------
    for name, chain in sorted((req.get("required_sequences") or {}).items()):
        problems: list[str] = []
        for i, node_id in enumerate(chain):
            if node_id not in graph:
                problems.append(f"missing node {node_id}")
                continue
            if i and node_id in graph and chain[i - 1] not in graph.get(node_id).depends_on:
                problems.append(f"{node_id} does not depend on {chain[i - 1]}")
        report.checks.append(
            {
                "check": "required_sequence",
                "name": name,
                "length": len(chain),
                "problems": problems,
                "status": "pass" if not problems else "fail",
            }
        )
        if problems:
            report.errors.append(f"sequence {name!r} is not a real dependency chain: {problems}")

    # 5. learning loop stages must be usable as record/assessment kinds -------------
    loop = list(req.get("learning_loop") or [])
    if loop:
        try:
            from training.data.records import RECORD_KINDS

            kinds = set(RECORD_KINDS)
        except Exception as exc:  # pragma: no cover - import error is itself the finding
            kinds = set()
            report.errors.append(f"cannot import training.data.records to check the learning loop: {exc}")
        missing = [k for k in loop if k not in kinds]
        report.checks.append(
            {
                "check": "learning_loop_kinds",
                "stages": len(loop),
                "missing": missing,
                "status": "pass" if not missing else "fail",
            }
        )
        if missing:
            report.errors.append(f"learning-loop stages not accepted as record kinds: {missing}")

    # 6. generation order must match the development program ------------------------
    gen_order = list(req.get("generation_order") or [])
    if gen_order:
        program = load_config(REPO_ROOT / "configs" / "sir_human_development.yaml")
        declared = [g["id"] for g in program.get("generations") or []]
        mismatch = "match" if declared == gen_order else f"program declares {declared}"
        report.checks.append(
            {
                "check": "generation_order",
                "declared": gen_order,
                "status": "pass" if declared == gen_order else "fail",
                "detail": mismatch,
            }
        )
        if declared != gen_order:
            report.errors.append(
                "generation order in the requirements file does not match "
                f"configs/sir_human_development.yaml: requirements={gen_order}, program={declared}"
            )

    # 7. required tracks non-empty --------------------------------------------------
    for track in req.get("required_tracks") or []:
        ids = graph.by_track(track)
        report.checks.append(
            {"check": "required_track", "track": track, "nodes": len(ids), "status": "pass" if ids else "fail"}
        )
        if not ids:
            report.errors.append(f"track {track!r} has no nodes")

    report.ok = not report.errors
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check the curriculum graph against the requirement contract.")
    ap.add_argument("--graph", default=str(DEFAULT_GRAPH))
    ap.add_argument("--requirements", default=str(DEFAULT_REQUIREMENTS))
    ap.add_argument("--out", help="write the report JSON here")
    args = ap.parse_args(argv)

    graph_path = resolve(args.graph)
    req_path = resolve(args.requirements)
    try:
        graph = CurriculumGraph.load(graph_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"graph load failed: {exc}", file=sys.stderr)
        return 2
    req = load_config(req_path)
    report = check_requirements(graph, req)
    report.requirements_path = rel(req_path)
    report.graph_path = rel(graph_path)
    print(f"graph: {report.graph_path} ({len(graph)} nodes)")
    for check in report.checks:
        mark = "PASS" if check["status"] == "pass" else "FAIL"
        detail = {k: v for k, v in check.items() if k not in ("check", "status")}
        print(f"  [{mark}] {check['check']} {detail}")
    if report.errors:
        print(f"\n{len(report.errors)} requirement violation(s):")
        for e in report.errors:
            print(f"  - {e}")
    else:
        print("\nall curriculum requirements satisfied")
    if args.out:
        out = write_json(resolve(args.out), report.to_dict())
        print(f"report: {rel(out)}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
