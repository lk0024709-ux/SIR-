"""Training-record schema, provenance policy and leakage-safe record splitting.

This is the schema every generated lesson, exercise, solution, correction and transfer task must
satisfy before it can enter a dataset. It exists because the failure mode of synthetic curricula is
not "the model learns nothing" — it is "nobody can tell afterwards where a claim came from".

Three rules are enforced in code rather than documented and hoped for:

1. **Provenance is mandatory.** Every record names its `provenance` class and, when it is
   synthetic, the teacher models that produced it. `partition_records` quarantines `uncertain`
   material and rejects `rejected` material — an unlabelled or doubtfully-licensed record cannot
   reach a training split by simply not mentioning the problem.
2. **Verification status is explicit.** A record is `unverified` until something verifies it;
   `multi_teacher_agreement` and `deterministically_verified` are distinct claims, and the
   consensus pipeline (teachers/consensus.py) decides which one applies.
3. **Reasoning representation is chosen, not implied.** `rationale_style` records whether a record
   carries no rationale, a concise one, structured steps, or a verifier-compatible trace. SIR must
   not depend on hidden chain-of-thought at runtime, so "train the model to emit long private
   reasoning" is a deliberate, visible choice rather than an accident of data generation.

Splitting is by *content hash*, not by row, so two records with identical content can never land on
opposite sides of a train/validation boundary — the cheapest possible contamination bug.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from sir_paths import rel, resolve

# --------------------------------------------------------------------------------------
# controlled vocabularies (kept in code: a typo here must fail loudly, not silently)
# --------------------------------------------------------------------------------------
RECORD_KINDS = (
    "learn",
    "practice",
    "error_diagnosis",
    "correction",
    "application",
    "explanation",
    "verification",
    "transfer",
    "review",
    "progression",
)
DIFFICULTIES = ("beginner", "elementary", "intermediate", "advanced", "expert", "olympiad")
RATIONAL_STYLES = ("none", "concise", "structured_steps", "verifier_trace")
PROVENANCE_CLASSES = ("self_authored", "licensed", "synthetic", "public_domain", "uncertain", "rejected")
LICENSE_STATUSES = ("cleared", "restricted_no_redistribution", "unknown_quarantine", "rejected")
VERIFICATION_STATUSES = (
    "unverified",
    "single_teacher",
    "critic_reviewed",
    "multi_teacher_agreement",
    "deterministically_verified",
    "human_reviewed",
    "rejected",
)
LANGUAGES = (
    "multi",
    "en",
    "hi",
    "hinc-latn",
    "hinc-deva",
    "sa",
    "bn",
    "gu",
    "kn",
    "ml",
    "mr",
    "or",
    "pa",
    "ta",
    "te",
    "as",
    "ur",
)
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.:-]*$")

# Fields that participate in the content hash. `record_id` and `created_utc` are provenance, not
# content: two records with identical content must hash identically so deduplication and
# train/val grouping work.
_HASH_FIELDS = (
    "kind",
    "curriculum_node",
    "domain",
    "subject",
    "grade_level",
    "difficulty",
    "language",
    "task",
    "response",
    "rationale",
    "rationale_style",
    "verification_status",
    "teacher_models",
    "error_tags",
)


@dataclass
class TrainingRecord:
    record_id: str
    kind: str
    task: str
    response: str
    curriculum_node: str
    domain: str
    subject: str
    language: str
    difficulty: str
    provenance: str
    verification_status: str
    license_status: str
    grade_level: int | str | None = None
    source: str = ""
    rationale: str = ""
    rationale_style: str = "none"
    teacher_models: list[str] = field(default_factory=list)
    error_tags: list[str] = field(default_factory=list)
    notes: str = ""
    created_utc: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    # ---- serialisation --------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainingRecord":
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}  # type: ignore[attr-defined]
        unknown = {k: v for k, v in d.items() if k not in known}
        return cls(
            record_id=str(d.get("record_id", "")),
            kind=str(d.get("kind", "")),
            task=str(d.get("task", "")),
            response=str(d.get("response", "")),
            curriculum_node=str(d.get("curriculum_node", "")),
            domain=str(d.get("domain", "")),
            subject=str(d.get("subject", "")),
            language=str(d.get("language", "")),
            difficulty=str(d.get("difficulty", "")),
            provenance=str(d.get("provenance", "")),
            verification_status=str(d.get("verification_status", "")),
            license_status=str(d.get("license_status", "")),
            grade_level=d.get("grade_level"),
            source=str(d.get("source", "")),
            rationale=str(d.get("rationale", "")),
            rationale_style=str(d.get("rationale_style", "none")),
            teacher_models=[str(x) for x in (d.get("teacher_models") or [])],
            error_tags=[str(x) for x in (d.get("error_tags") or [])],
            notes=str(d.get("notes", "")),
            created_utc=str(d.get("created_utc", "")),
            extra=unknown,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "record_id": self.record_id,
            "kind": self.kind,
            "curriculum_node": self.curriculum_node,
            "domain": self.domain,
            "subject": self.subject,
            "grade_level": self.grade_level,
            "language": self.language,
            "difficulty": self.difficulty,
            "provenance": self.provenance,
            "license_status": self.license_status,
            "verification_status": self.verification_status,
            "teacher_models": list(self.teacher_models),
            "error_tags": list(self.error_tags),
            "source": self.source,
            "task": self.task,
            "response": self.response,
            "rationale": self.rationale,
            "rationale_style": self.rationale_style,
            "notes": self.notes,
        }
        if self.created_utc:
            out["created_utc"] = self.created_utc
        out.update(self.extra)
        return out

    @property
    def content_hash(self) -> str:
        payload = {k: getattr(self, k) for k in _HASH_FIELDS}
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @property
    def trainable(self) -> bool:
        """May this record enter a training split?

        Verification is part of the answer, not a separate concern: an `unverified` record has been
        checked by nobody, and a corpus of unchecked text is a hypothesis, not training material. Such
        records stay in quarantine until a teacher, critic, verifier or reviewer signs them.
        """
        return (
            self.provenance in ("self_authored", "licensed", "synthetic", "public_domain")
            and self.license_status in ("cleared", "restricted_no_redistribution")
            and self.verification_status not in ("unverified", "rejected")
        )

    # ---- validation -----------------------------------------------------------------
    def validate(self) -> list[str]:
        errs: list[str] = []
        if not self.record_id or not ID_PATTERN.match(self.record_id):
            errs.append(f"record_id {self.record_id!r} must be a lowercase dotted identifier")
        if self.kind not in RECORD_KINDS:
            errs.append(f"{self.record_id}: kind {self.kind!r} not in {RECORD_KINDS}")
        if not self.task.strip():
            errs.append(f"{self.record_id}: task is empty")
        if not self.response.strip():
            errs.append(f"{self.record_id}: response is empty")
        if not self.curriculum_node.strip():
            errs.append(f"{self.record_id}: curriculum_node is required (records must attach to the graph)")
        if self.difficulty not in DIFFICULTIES:
            errs.append(f"{self.record_id}: difficulty {self.difficulty!r} not in {DIFFICULTIES}")
        if self.language not in LANGUAGES:
            errs.append(f"{self.record_id}: language {self.language!r} not in {LANGUAGES}")
        if not self.domain.strip():
            errs.append(f"{self.record_id}: domain is required")
        if not self.subject.strip():
            errs.append(f"{self.record_id}: subject is required")
        if self.provenance not in PROVENANCE_CLASSES:
            errs.append(f"{self.record_id}: provenance {self.provenance!r} not in {PROVENANCE_CLASSES}")
        if self.license_status not in LICENSE_STATUSES:
            errs.append(f"{self.record_id}: license_status {self.license_status!r} not in {LICENSE_STATUSES}")
        if self.verification_status not in VERIFICATION_STATUSES:
            errs.append(
                f"{self.record_id}: verification_status {self.verification_status!r} not in {VERIFICATION_STATUSES}"
            )
        if self.rationale_style not in RATIONAL_STYLES:
            errs.append(f"{self.record_id}: rationale_style {self.rationale_style!r} not in {RATIONAL_STYLES}")
        if self.rationale_style != "none" and not self.rationale.strip():
            errs.append(f"{self.record_id}: rationale_style={self.rationale_style} but rationale is empty")
        if self.provenance == "synthetic" and not self.teacher_models:
            errs.append(
                f"{self.record_id}: synthetic records must name their teacher_models; a synthetic record "
                "with no teacher provenance is an unattributable claim"
            )
        if self.kind in ("error_diagnosis", "correction") and not self.error_tags:
            errs.append(f"{self.record_id}: {self.kind} records require at least one error_tag")
        if not self.source.strip():
            errs.append(
                f"{self.record_id}: source is required — 'teacher pipeline v0' is a valid value, an empty "
                "string is not"
            )
        return errs


# --------------------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------------------
def write_records(path: Path | str, records: Iterable[TrainingRecord]) -> Path:
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
    return p


def load_records(path: Path | str) -> list[TrainingRecord]:
    p = resolve(path)
    if not p.exists():
        raise FileNotFoundError(f"record file not found: {p}")
    out: list[TrainingRecord] = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(TrainingRecord.from_dict(json.loads(line)))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{rel(p)}:{i}: not valid JSON ({exc})") from exc
    return out


def validate_records(records: Iterable[TrainingRecord]) -> list[str]:
    errs: list[str] = []
    for rec in records:
        errs += rec.validate()
    return errs


# --------------------------------------------------------------------------------------
# provenance policy
# --------------------------------------------------------------------------------------
def partition_records(
    records: Iterable[TrainingRecord],
) -> dict[str, list[TrainingRecord]]:
    """Split records into `trainable`, `quarantine` and `rejected`.

    Quarantine is not a bin: it is a holding area for records that may become usable once their
    licence or provenance is resolved. Nothing reaches `trainable` by default — it has to qualify.
    """
    out: dict[str, list[TrainingRecord]] = {"trainable": [], "quarantine": [], "rejected": []}
    for rec in records:
        if rec.provenance == "rejected" or rec.license_status == "rejected" or rec.verification_status == "rejected":
            out["rejected"].append(rec)
        elif rec.provenance == "uncertain" or rec.license_status == "unknown_quarantine":
            out["quarantine"].append(rec)
        elif rec.trainable:
            out["trainable"].append(rec)
        else:
            out["quarantine"].append(rec)
    return out


def assert_trainable(records: Iterable[TrainingRecord]) -> None:
    """Raise if any record in a set that is about to become a training split is not trainable."""
    bad = [r.record_id for r in records if not r.trainable]
    if bad:
        raise ValueError(
            f"{len(bad)} record(s) are not trainable and must not enter a training split: "
            f"{bad[:10]}{' …' if len(bad) > 10 else ''}"
        )


# --------------------------------------------------------------------------------------
# duplication and splitting
# --------------------------------------------------------------------------------------
def find_duplicates(records: Iterable[TrainingRecord]) -> dict[str, list[str]]:
    """Group record ids by content hash. Any group with >1 id is an exact duplicate."""
    by_hash: dict[str, list[str]] = {}
    for rec in records:
        by_hash.setdefault(rec.content_hash, []).append(rec.record_id)
    return {h: ids for h, ids in by_hash.items() if len(ids) > 1}


def dedupe(records: Iterable[TrainingRecord], *, keep: str = "first") -> list[TrainingRecord]:
    seen: dict[str, TrainingRecord] = {}
    for rec in records:
        h = rec.content_hash
        if h not in seen or keep == "last":
            seen[h] = rec
    return list(seen.values())


def split_records(
    records: Iterable[TrainingRecord], *, val_fraction: float = 0.1, seed: int = 0
) -> dict[str, list[TrainingRecord]]:
    """Deterministic train/val split grouping *identical content* onto the same side.

    The bucket is a hash of (seed, content_hash), so the split does not depend on input order and
    reruns are reproducible without storing an index file.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    threshold = int(val_fraction * 10_000)
    out: dict[str, list[TrainingRecord]] = {"train": [], "val": []}
    for rec in records:
        key = f"{seed}:{rec.content_hash}"
        bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 10_000
        out["val" if bucket < threshold else "train"].append(rec)
    return out


# --------------------------------------------------------------------------------------
# cross-checks and reporting
# --------------------------------------------------------------------------------------
def check_against_graph(records: Iterable[TrainingRecord], graph: Any) -> list[str]:
    """Cross-check record metadata against the curriculum graph (unknown nodes, subject/grade drift)."""
    errs: list[str] = []
    for rec in records:
        if rec.curriculum_node not in graph:
            errs.append(f"{rec.record_id}: curriculum_node {rec.curriculum_node!r} is not in the graph")
            continue
        node = graph.get(rec.curriculum_node)
        if rec.subject and rec.subject != node.subject:
            errs.append(
                f"{rec.record_id}: subject {rec.subject!r} disagrees with node {node.id} subject {node.subject!r}"
            )
        if node.grade is not None and rec.grade_level is not None and str(rec.grade_level) != str(node.grade):
            errs.append(
                f"{rec.record_id}: grade_level {rec.grade_level!r} disagrees with node {node.id} grade {node.grade}"
            )
    return errs


def stats(records: Iterable[TrainingRecord]) -> dict[str, Any]:
    """Measured counts only. These are the numbers a report is allowed to quote."""
    recs = list(records)
    def count(key: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in recs:
            value = getattr(r, key)
            if isinstance(value, list):
                for v in value or ["(none)"]:
                    out[str(v)] = out.get(str(v), 0) + 1
            else:
                out[str(value)] = out.get(str(value), 0) + 1
        return dict(sorted(out.items()))

    tasks = sum(len(r.task) for r in recs)
    responses = sum(len(r.response) for r in recs)
    parts = partition_records(recs)
    return {
        "records": len(recs),
        "unique_content": len({r.content_hash for r in recs}),
        "by_kind": count("kind"),
        "by_domain": count("domain"),
        "by_subject": count("subject"),
        "by_language": count("language"),
        "by_difficulty": count("difficulty"),
        "by_provenance": count("provenance"),
        "by_verification_status": count("verification_status"),
        "by_teacher_model": count("teacher_models"),
        "by_curriculum_node": count("curriculum_node"),
        "characters": {"task": tasks, "response": responses, "total": tasks + responses},
        "policy": {
            "trainable": len(parts["trainable"]),
            "quarantine": len(parts["quarantine"]),
            "rejected": len(parts["rejected"]),
        },
        "duplicate_content_groups": len(find_duplicates(recs)),
    }
