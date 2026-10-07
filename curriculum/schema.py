"""Curriculum node schema: the shape of one learning node, and its validation rules.

A node is a *statement about what must be learned*, never a claim that it has been learned.
Nothing in this file can mark a node as mastered: that is `curriculum/coverage.py`, and it only
reads assessment records.

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
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from sir_paths import REPO_ROOT  # noqa: F401  (kept for callers that resolve graph paths)

ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)+$")
TRACKS = ("foundation", "bridge", "advanced", "meta")
BANDS = (
    "primary",
    "middle",
    "secondary",
    "higher_secondary",
    "undergraduate",
    "advanced",
    "research",
    "cross_cutting",
)


@dataclass(frozen=True)
class CurriculumNode:
    id: str
    title: str
    subject: str
    track: str
    band: str
    depends_on: tuple[str, ...] = ()
    grade: int | None = None
    topics: tuple[str, ...] = ()
    skills: tuple[str, ...] = ()
    language: str = "multi"
    provenance: str = "self_authored"
    epistemic_note: str = ""
    available_from: str | None = None
    extra: dict[str, Any] = field(default_factory=dict, compare=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CurriculumNode":
        known = {
            "id",
            "title",
            "subject",
            "track",
            "band",
            "depends_on",
            "grade",
            "topics",
            "skills",
            "language",
            "provenance",
            "epistemic_note",
            "available_from",
        }
        unknown = {k: v for k, v in d.items() if k not in known}
        return cls(
            id=str(d.get("id", "")).strip(),
            title=str(d.get("title", "")).strip(),
            subject=str(d.get("subject", "")).strip(),
            track=str(d.get("track", "")).strip(),
            band=str(d.get("band", "")).strip(),
            depends_on=tuple(str(x) for x in (d.get("depends_on") or ())),
            grade=d.get("grade"),
            topics=tuple(str(x) for x in (d.get("topics") or ())),
            skills=tuple(str(x) for x in (d.get("skills") or ())),
            language=str(d.get("language", "multi")),
            provenance=str(d.get("provenance", "self_authored")),
            epistemic_note=str(d.get("epistemic_note", "") or ""),
            available_from=d.get("available_from"),
            extra=unknown,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "title": self.title,
            "subject": self.subject,
            "track": self.track,
            "band": self.band,
            "depends_on": list(self.depends_on),
            "grade": self.grade,
            "topics": list(self.topics),
            "skills": list(self.skills),
            "language": self.language,
            "provenance": self.provenance,
        }
        if self.epistemic_note:
            out["epistemic_note"] = self.epistemic_note
        if self.available_from:
            out["available_from"] = self.available_from
        out.update(self.extra)
        return out

    def validate(self, subjects: Iterable[str] = ()) -> list[str]:
        """Return schema problems for this node. Empty list means valid."""
        errs: list[str] = []
        if not ID_PATTERN.match(self.id):
            errs.append(f"node id {self.id!r} must be dotted lowercase, e.g. 'math.g07'")
        if not self.title:
            errs.append(f"{self.id}: title is empty")
        subject_set = set(subjects)
        if not self.subject:
            errs.append(f"{self.id}: subject is empty")
        elif subject_set and self.subject not in subject_set:
            errs.append(f"{self.id}: subject {self.subject!r} is not declared in subjects:")
        if self.track not in TRACKS:
            errs.append(f"{self.id}: track {self.track!r} not in {TRACKS}")
        if self.band not in BANDS:
            errs.append(f"{self.id}: band {self.band!r} not in {BANDS}")
        if self.track == "foundation":
            if not isinstance(self.grade, int) or not 1 <= int(self.grade) <= 12:
                errs.append(f"{self.id}: foundation nodes need an integer grade in 1..12")
        elif self.grade is not None:
            errs.append(f"{self.id}: non-foundation node must not set grade (got {self.grade!r})")
        if self.id in self.depends_on:
            errs.append(f"{self.id}: depends on itself")
        if len(set(self.depends_on)) != len(self.depends_on):
            errs.append(f"{self.id}: duplicate entries in depends_on")
        if self.provenance != "self_authored":
            errs.append(
                f"{self.id}: provenance {self.provenance!r} is not allowed in the curriculum graph; "
                "third-party curriculum content must be added through a dataset with its own card"
            )
        if not self.topics:
            errs.append(f"{self.id}: at least one topic tag is required (requirements are checked against them)")
        return errs




def load_graph_document(path: Path) -> dict:
    """Read a graph YAML/JSON document. Kept here so schema users do not import the graph class."""
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
