"""Curriculum graph: the prerequisite structure, and the rules that keep it honest.

The graph is the *dependency structure* of SIR's development program, not a claim of knowledge.
Nothing here can mark a node as learned — that is `curriculum/coverage.py`, and it only reads
assessment records.

Design notes that matter:

* **Validation returns errors, it does not raise on the first one.** A curriculum file with six
  problems should report six problems; stopping at the first makes fixing the file a guessing game.
* **The grade ladder is checked, not assumed.** For every subject, Class *n+1* must depend on
  Class *n* of the same subject. A silent gap in the ladder would let a "Class 10" claim rest on
  material that was never scheduled.
* **Foundation nodes may not depend on a higher grade.** Cross-subject prerequisites are allowed
  (Hinglish needs Hindi and English), but a Class 4 node cannot require Class 8 material.
* **Node ids are stable and dotted** so they can be referenced from dataset records, evaluation
  cases, coverage reports and contracts without a translation table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from curriculum.schema import BANDS, ID_PATTERN, TRACKS, CurriculumNode  # noqa: F401
from sir_paths import REPO_ROOT, resolve

#: Canonical location of the committed curriculum graph (committed data, not a derived artifact).
DEFAULT_GRAPH_PATH = REPO_ROOT / "data" / "curriculum" / "curriculum_graph.yaml"
DEFAULT_GRAPH = DEFAULT_GRAPH_PATH

@dataclass
class CurriculumGraph:
    nodes: dict[str, CurriculumNode]
    subjects: dict[str, dict[str, Any]]
    graph_id: str = "unnamed"
    card: str = ""
    schema_version: int = 1
    source_path: Path | None = None

    # ---- construction ----------------------------------------------------------------
    @classmethod
    def from_dict(cls, doc: dict[str, Any], source_path: Path | None = None) -> "CurriculumGraph":
        subjects = {str(s["id"]): dict(s) for s in (doc.get("subjects") or [])}
        nodes: dict[str, CurriculumNode] = {}
        for raw in doc.get("nodes") or []:
            node = CurriculumNode.from_dict(raw)
            nodes[node.id] = node
        return cls(
            nodes=nodes,
            subjects=subjects,
            graph_id=str(doc.get("graph_id", "unnamed")),
            card=str(doc.get("card", "")),
            schema_version=int(doc.get("schema_version", 1)),
            source_path=source_path,
        )

    @classmethod
    def load(cls, path: Path | str = DEFAULT_GRAPH_PATH, *, validate: bool = True) -> "CurriculumGraph":
        import yaml

        p = resolve(path)
        if not p.exists():
            raise FileNotFoundError(f"curriculum graph not found: {p}")
        doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        graph = cls.from_dict(doc, source_path=p)
        if validate:
            errors = graph.validate()
            if errors:
                raise ValueError(
                    f"curriculum graph {p} failed validation with {len(errors)} problem(s):\n  - "
                    + "\n  - ".join(errors)
                )
        return graph

    # ---- validation ------------------------------------------------------------------
    def validate(self) -> list[str]:
        errs: list[str] = []
        if not self.nodes:
            errs.append("graph contains no nodes")
        declared = set(self.subjects)
        for node in self.nodes.values():
            errs += node.validate(declared)
        for node in self.nodes.values():
            for dep in node.depends_on:
                if dep not in self.nodes:
                    errs.append(f"{node.id}: depends on unknown node {dep!r}")
                elif self.nodes[dep].id == node.id:
                    errs.append(f"{node.id}: depends on itself")
        errs += self._check_cycles()
        errs += self._check_grade_ladders()
        errs += self._check_forward_dependencies()
        errs += self._check_advanced_orphans()
        errs += self._check_epistemic_notes()
        return errs

    def _check_cycles(self) -> list[str]:
        errs: list[str] = []
        state: dict[str, int] = {}
        stack: list[str] = []

        def visit(nid: str) -> None:
            if state.get(nid) == 2:
                return
            if state.get(nid) == 1:
                cycle = stack[stack.index(nid) :] + [nid]
                errs.append("dependency cycle: " + " -> ".join(cycle))
                return
            state[nid] = 1
            stack.append(nid)
            for dep in self.nodes[nid].depends_on:
                if dep in self.nodes:
                    visit(dep)
            stack.pop()
            state[nid] = 2

        for nid in sorted(self.nodes):
            visit(nid)
        return errs

    def _check_grade_ladders(self) -> list[str]:
        errs: list[str] = []
        by_subject: dict[str, dict[int, str]] = {}
        for node in self.nodes.values():
            if node.track == "foundation" and isinstance(node.grade, int):
                by_subject.setdefault(node.subject, {})[int(node.grade)] = node.id
        for subject, ladder in sorted(by_subject.items()):
            grades = sorted(ladder)
            if grades != list(range(grades[0], grades[-1] + 1)):
                missing = sorted(set(range(grades[0], grades[-1] + 1)) - set(grades))
                errs.append(f"{subject}: grade ladder has gaps at {missing}")
            for grade in grades[1:]:
                node = self.nodes[ladder[grade]]
                prev = ladder.get(grade - 1)
                if prev is None:
                    continue  # the gap itself is already reported above
                if prev not in node.depends_on:
                    errs.append(
                        f"{node.id}: Class {grade} node must depend on the Class {grade - 1} node {prev!r}"
                    )
        return errs

    def _check_forward_dependencies(self) -> list[str]:
        errs: list[str] = []
        for node in self.nodes.values():
            if node.track != "foundation" or not isinstance(node.grade, int):
                continue
            for dep in node.depends_on:
                other = self.nodes.get(dep)
                if other is None:
                    continue
                if isinstance(other.grade, int) and int(other.grade) > int(node.grade):
                    errs.append(
                        f"{node.id} (Class {node.grade}) depends on higher-grade material {dep} "
                        f"(Class {other.grade})"
                    )
        return errs

    def _check_advanced_orphans(self) -> list[str]:
        errs: list[str] = []
        for node in self.nodes.values():
            if node.track in ("advanced", "bridge") and not node.depends_on:
                errs.append(f"{node.id}: {node.track} node has no prerequisites (unreachable from the ladder)")
            if node.track == "meta" and not node.depends_on and not node.available_from:
                errs.append(f"{node.id}: meta node needs either prerequisites or an available_from marker")
        return errs

    def _check_epistemic_notes(self) -> list[str]:
        errs: list[str] = []
        for node in self.nodes.values():
            if node.subject == "indian_knowledge" and len(node.epistemic_note.strip()) < 40:
                errs.append(
                    f"{node.id}: indian_knowledge nodes must carry an epistemic_note explaining how "
                    "historical record, tradition and interpretation are separated"
                )
        return errs

    # ---- queries ---------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.nodes)

    def __contains__(self, node_id: object) -> bool:
        return node_id in self.nodes

    def __iter__(self) -> Iterator[CurriculumNode]:
        for nid in self.order():
            yield self.nodes[nid]

    def get(self, node_id: str) -> CurriculumNode:
        if node_id not in self.nodes:
            raise KeyError(f"unknown curriculum node {node_id!r}")
        return self.nodes[node_id]

    def prerequisites(self, node_id: str) -> tuple[str, ...]:
        return self.get(node_id).depends_on

    def children(self, node_id: str) -> tuple[str, ...]:
        return tuple(sorted(n.id for n in self.nodes.values() if node_id in n.depends_on))

    def ancestors(self, node_id: str) -> set[str]:
        seen: set[str] = set()
        stack = list(self.prerequisites(node_id))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(self.prerequisites(cur))
        return seen

    def descendants(self, node_id: str) -> set[str]:
        out: set[str] = set()
        frontier = [node_id]
        while frontier:
            cur = frontier.pop()
            for child in self.children(cur):
                if child not in out:
                    out.add(child)
                    frontier.append(child)
        return out

    def order(self) -> list[str]:
        """Deterministic topological order (prerequisites first, ties broken lexically)."""
        import heapq

        indeg = {nid: 0 for nid in self.nodes}
        for node in self.nodes.values():
            for dep in node.depends_on:
                if dep in self.nodes:
                    indeg[node.id] += 1
        ready = [nid for nid, d in indeg.items() if d == 0]
        heapq.heapify(ready)
        out: list[str] = []
        while ready:
            nid = heapq.heappop(ready)
            out.append(nid)
            for child in self.children(nid):
                indeg[child] -= 1
                if indeg[child] == 0:
                    heapq.heappush(ready, child)
        if len(out) != len(self.nodes):
            raise ValueError("graph contains a cycle; call validate() for details")
        return out

    def by_track(self, track: str) -> tuple[str, ...]:
        return tuple(nid for nid in self.order() if self.nodes[nid].track == track)

    def by_subject(self, subject: str) -> tuple[str, ...]:
        return tuple(nid for nid in self.order() if self.nodes[nid].subject == subject)

    def by_grade(self, grade: int) -> tuple[str, ...]:
        return tuple(nid for nid in self.order() if self.nodes[nid].grade == grade)

    def topics(self, subject: str | None = None) -> set[str]:
        return {
            t
            for n in self.nodes.values()
            if subject is None or n.subject == subject
            for t in n.topics
        }

    def summary(self) -> dict[str, Any]:
        tracks: dict[str, int] = {}
        subjects: dict[str, int] = {}
        for node in self.nodes.values():
            tracks[node.track] = tracks.get(node.track, 0) + 1
            subjects[node.subject] = subjects.get(node.subject, 0) + 1
        edg = sum(len(n.depends_on) for n in self.nodes.values())
        return {
            "graph_id": self.graph_id,
            "nodes": len(self.nodes),
            "edges": edg,
            "subjects": dict(sorted(subjects.items())),
            "tracks": dict(sorted(tracks.items())),
            "foundation_grades": sorted({n.grade for n in self.nodes.values() if n.grade}),
            "topics": len(self.topics()),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "graph_id": self.graph_id,
            "card": self.card,
            "subjects": list(self.subjects.values()),
            "nodes": [self.nodes[nid].to_dict() for nid in self.order()],
        }

    def to_json(self, path: Path | str | None = None) -> str:
        payload = json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n"
        if path is not None:
            p = resolve(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(payload, encoding="utf-8")
        return payload


def load_default_graph(*, validate: bool = True) -> CurriculumGraph:
    return CurriculumGraph.load(DEFAULT_GRAPH_PATH, validate=validate)
