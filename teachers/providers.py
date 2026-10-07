"""Teacher providers: where generated material comes from, and what is recorded about it.

**External models are teachers, not SIR's runtime.** Nothing in this package can call a model at
inference time, and nothing here ships an API client. The contract is the opposite direction:

* a *request* describes what the curriculum needs (a lesson, exercises, a transfer problem…);
* a *provider* returns drafts that were produced by something — a recorded teacher transcript, a
  deterministic template, or (in future) a live adapter run by a human/agent outside the repository;
* the pipeline stores the drafts, their teacher identity, and the terms under which the output may
  be reused, and refuses to treat anything unverified as trusted training data.

That means a run is reproducible from committed artifacts: the requests are deterministic functions
of the graph, and the drafts are files. Re-running the consensus/verification stages never depends on
a third-party service being up, or on a model behaving the same way twice.

`output_reuse_permitted` is deliberately a per-teacher flag rather than an assumption: a model whose
terms do not clearly permit using its outputs for training cannot silently become a data source. Its
records are written to quarantine with `license_status: unknown_quarantine`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

from sir_paths import rel, resolve

DRAFT_FIELDS = ("task", "response", "answer", "rationale")


@dataclass(frozen=True)
class TeacherSpec:
    model_id: str
    role: str = "teacher"  # teacher | critic | verifier | difficulty | adversarial | synthesizer
    vendor: str = ""
    version: str = ""
    terms_note: str = ""
    output_reuse_permitted: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TeacherSpec":
        return cls(
            model_id=str(d.get("model_id", "")),
            role=str(d.get("role", "teacher")),
            vendor=str(d.get("vendor", "")),
            version=str(d.get("version", "")),
            terms_note=str(d.get("terms_note", "")),
            output_reuse_permitted=bool(d.get("output_reuse_permitted", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "role": self.role,
            "vendor": self.vendor,
            "version": self.version,
            "terms_note": self.terms_note,
            "output_reuse_permitted": self.output_reuse_permitted,
        }

    def validate(self) -> list[str]:
        errs: list[str] = []
        if not self.model_id:
            errs.append("teacher spec needs a model_id")
        if self.role not in ("teacher", "critic", "verifier", "difficulty", "adversarial", "synthesizer"):
            errs.append(f"unknown teacher role {self.role!r}")
        if not self.terms_note:
            errs.append(
                f"{self.model_id}: terms_note is required — a teacher whose reuse terms are unstated must not "
                "be treated as a permitted data source (output_reuse_permitted stays false)"
            )
        return errs


@dataclass
class GenerationRequest:
    request_id: str
    node_id: str
    kind: str
    language: str
    difficulty: str
    prompt: str
    required_fields: tuple[str, ...] = DRAFT_FIELDS
    seed: int = 0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "node_id": self.node_id,
            "kind": self.kind,
            "language": self.language,
            "difficulty": self.difficulty,
            "required_fields": list(self.required_fields),
            "seed": self.seed,
            "prompt": self.prompt,
            "notes": self.notes,
        }


@dataclass
class TeacherDraft:
    request_id: str
    teacher: TeacherSpec
    payload: dict[str, Any]
    created_utc: str = ""
    notes: str = ""

    @property
    def answer(self) -> str:
        return str(self.payload.get("answer") or self.payload.get("response") or "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "teacher": self.teacher.to_dict(),
            "created_utc": self.created_utc,
            "notes": self.notes,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TeacherDraft":
        return cls(
            request_id=str(d.get("request_id", "")),
            teacher=TeacherSpec.from_dict(d.get("teacher") or {}),
            payload=dict(d.get("payload") or {}),
            created_utc=str(d.get("created_utc", "")),
            notes=str(d.get("notes", "")),
        )


class TeacherProvider(Protocol):
    """Minimal provider contract. Returning an empty list is always allowed and is not an error."""

    name: str
    spec: TeacherSpec

    def generate(self, request: GenerationRequest) -> list[TeacherDraft]:  # pragma: no cover - protocol
        ...


# --------------------------------------------------------------------------------------
# concrete providers
# --------------------------------------------------------------------------------------
@dataclass
class StaticProvider:
    """In-memory drafts keyed by request id. Used by tests and by small hand-curated batches."""

    name: str
    spec: TeacherSpec
    drafts: dict[str, list[TeacherDraft]] = field(default_factory=dict)

    def add(self, request_id: str, payload: dict[str, Any]) -> None:
        self.drafts.setdefault(request_id, []).append(TeacherDraft(request_id=request_id, teacher=self.spec, payload=payload))

    def generate(self, request: GenerationRequest) -> list[TeacherDraft]:
        return list(self.drafts.get(request.request_id, []))


@dataclass
class RecordedProvider:
    """Drafts loaded from a committed JSONL transcript (one draft per line).

    A transcript is written by whoever ran the teacher (agent, human, batch job). It carries the
    teacher spec with it, so provenance cannot be lost by copying drafts between files.
    """

    name: str
    drafts: list[TeacherDraft] = field(default_factory=list)
    source_path: str = ""

    @classmethod
    def from_jsonl(cls, path: Path | str, name: str = "") -> "RecordedProvider":
        p = resolve(path)
        if not p.exists():
            raise FileNotFoundError(f"teacher transcript not found: {p}")
        drafts: list[TeacherDraft] = []
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                drafts.append(TeacherDraft.from_dict(json.loads(line)))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{rel(p)}:{i}: not valid JSON ({exc})") from exc
        return cls(name=name or p.stem, drafts=drafts, source_path=rel(p))

    @property
    def spec(self) -> TeacherSpec:
        if self.drafts:
            return self.drafts[0].teacher
        return TeacherSpec(model_id=self.name or "recorded", role="teacher", terms_note="transcript teacher")

    def generate(self, request: GenerationRequest) -> list[TeacherDraft]:
        return [d for d in self.drafts if d.request_id == request.request_id]


@dataclass
class TemplateProvider:
    """Deterministic, offline drafts built from the request text.

    This exists so the pipeline, critics and consensus logic can be tested and smoke-run end to end
    without a network or a model. Its output is honest about what it is: `synthetic`, `unverified`,
    `template: true`, and never used as a source of facts.
    """

    name: str = "template_v0"
    spec: TeacherSpec = field(
        default_factory=lambda: TeacherSpec(
            model_id="template_v0",
            role="teacher",
            vendor="sir-repo",
            version="0",
            terms_note="deterministic local template; output carries no factual authority",
            output_reuse_permitted=True,
        )
    )
    seed: int = 0

    def generate(self, request: GenerationRequest) -> list[TeacherDraft]:
        topic = request.notes or request.node_id
        task = f"[{request.kind}] {request.language} {request.difficulty}: {topic}"
        payload = {
            "task": task,
            "response": f"{topic}: a {request.difficulty} {request.kind} item for {request.node_id}",
            "answer": f"{request.node_id}.{request.kind}.{request.language}",
            "rationale": f"Placeholder rationale for a {request.kind} task; no factual claim is made.",
            "template": True,
        }
        return [TeacherDraft(request_id=request.request_id, teacher=self.spec, payload=payload, notes="template draft")]


class ProviderRegistry:
    def __init__(self, providers: Iterable[Any] = ()):
        self._providers: dict[str, Any] = {}
        for p in providers:
            self.add(p)

    def add(self, provider: Any) -> None:
        name = getattr(provider, "name", "")
        if not name:
            raise ValueError("provider needs a name")
        self._providers[name] = provider

    def get(self, name: str) -> Any:
        return self._providers[name]

    @property
    def names(self) -> list[str]:
        return sorted(self._providers)

    def specs(self) -> list[dict[str, Any]]:
        out = []
        for name in self.names:
            spec = getattr(self._providers[name], "spec", None)
            entry = {"provider": name}
            if isinstance(spec, TeacherSpec):
                entry.update(spec.to_dict())
            out.append(entry)
        return out

    def generate(self, request: GenerationRequest) -> list[TeacherDraft]:
        drafts: list[TeacherDraft] = []
        for name in self.names:
            drafts.extend(self._providers[name].generate(request))
        return drafts
