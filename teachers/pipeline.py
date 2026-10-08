"""The curriculum data-generation pipeline, end to end.

    curriculum node
        → request plan (deterministic, one request per kind × language × difficulty)
        → teacher drafts (recorded, template or live adapters — all provenance-stamped)
        → critics (mechanical checks: fields, placeholders, language, degeneracy, arithmetic)
        → consensus (multi-teacher agreement or deterministic verification; disagreement is flagged)
        → routing (accepted / quarantined / rejected) with a QC report

Design rules, each of which exists because the opposite is the usual failure:

* **Requests are a pure function of the graph + config + seed.** So a batch can be regenerated and
  compared, and "which prompt produced this record" is answerable months later.
* **No drafts, no records.** If providers return nothing the run reports a shortfall; it never
  fabricates filler to fill the quota.
* **Quarantine is a first-class outcome.** Material from a teacher whose reuse terms are unstated, or
  whose kind requires verification it has not received, lands in quarantine with the reason — visible,
  counted, and excluded from training.
* **The rationale policy is explicit.** Records carry `rationale_style: concise` (or `structured_steps`
  when the payload provides them), and the request template never asks a teacher to dump private
  chain-of-thought. SIR does not train on hidden reasoning it will not have at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from curriculum.graph import DEFAULT_GRAPH, CurriculumGraph
from sir_paths import REPO_ROOT, load_config, rel, resolve, write_json
from teachers.consensus import ConsensusOutcome, ConsensusStatus, reach_consensus
from teachers.critics import Critique, critique_draft, summarise as summarise_critiques, worst_verdict
from teachers.providers import (
    GenerationRequest,
    ProviderRegistry,
    TeacherDraft,
    TemplateProvider,
    TeacherSpec,
)
from training.data.records import RECORD_KINDS, TrainingRecord, stats as record_stats

DEFAULT_CONFIG = REPO_ROOT / "configs" / "curriculum_sampling.yaml"
PIPELINE_VERSION = "teacher-pipeline-v0"

PROMPT_TEMPLATE = """\
Produce one {kind} item for SIR's curriculum (an Indian AI learning systematically, in stages).

Curriculum node : {node_id} — {title}
Subject / band  : {subject} / {band}{grade_line}
Language        : {language}
Difficulty      : {difficulty}
Topics required : {topics}
Stage in the learning loop: {kind}

Return a JSON object with exactly these fields:
  task      : the question, exercise or instruction shown to the learner
  response  : the teaching material or worked answer
  answer    : the short, checkable final answer (one line)
  rationale : a CONCISE justification (2-4 sentences). Do not dump private step-by-step
              reasoning; a short explanation that a verifier can follow is what is needed.

Rules: no placeholder text; if arithmetic appears, write it as an explicit statement
(e.g. "17% of 240 = 40.8") so it can be checked mechanically; stay inside the stated language and
difficulty; do not invent sources or citations."""


def build_request(
    graph: CurriculumGraph,
    node_id: str,
    kind: str,
    language: str,
    difficulty: str,
    *,
    index: int = 0,
    seed: int = 0,
) -> GenerationRequest:
    node = graph.get(node_id)
    if kind not in RECORD_KINDS:
        raise ValueError(f"kind {kind!r} is not a learning-loop stage ({RECORD_KINDS})")
    grade_line = f" | grade {node.grade}" if node.grade is not None else ""
    prompt = PROMPT_TEMPLATE.format(
        kind=kind,
        node_id=node.id,
        title=node.title,
        subject=node.subject,
        band=node.band,
        grade_line=grade_line,
        language=language,
        difficulty=difficulty,
        topics=", ".join(node.topics),
    )
    return GenerationRequest(
        request_id=f"{node.id}.{kind}.{language}.{difficulty}.{index:02d}",
        node_id=node.id,
        kind=kind,
        language=language,
        difficulty=difficulty,
        prompt=prompt,
        seed=seed,
        notes=f"{node.title} ({node.subject}, {node.band})",
    )


@dataclass
class GenerationConfig:
    kinds: tuple[str, ...] = ("learn", "practice", "application", "verification", "transfer")
    languages: tuple[str, ...] = ("hi", "en", "hinc-latn")
    difficulties: tuple[str, ...] = ("beginner", "elementary", "intermediate", "advanced")
    min_teachers: int = 2
    allow_single_teacher: bool = False
    require_verification_for: tuple[str, ...] = ("verification", "transfer", "application")
    max_records_per_node: int = 60
    seed: int = 0

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "GenerationConfig":
        g = (cfg or {}).get("generation") or {}
        out = cls(
            kinds=tuple(g.get("kinds_per_node") or cls().kinds),
            languages=tuple(g.get("languages") or cls().languages),
            difficulties=tuple(g.get("difficulties") or cls().difficulties),
            min_teachers=int(g.get("min_teachers", 2)),
            allow_single_teacher=bool(g.get("allow_single_teacher", False)),
            require_verification_for=tuple(g.get("require_verification_for") or ()),
            max_records_per_node=int(g.get("max_records_per_node", 60)),
            seed=int(g.get("seed", 0)) if g.get("seed") is not None else int((cfg or {}).get("sampler", {}).get("seed", 0)),
        )
        out.check()
        return out

    def check(self) -> None:
        bad = [k for k in self.kinds if k not in RECORD_KINDS]
        if bad:
            raise ValueError(f"generation.kinds_per_node contains non loop-stage kinds: {bad}")
        unknown = [k for k in self.require_verification_for if k not in RECORD_KINDS]
        if unknown:
            raise ValueError(f"generation.require_verification_for contains unknown kinds: {unknown}")
        if self.min_teachers < 1:
            raise ValueError("generation.min_teachers must be >= 1")
        if self.max_records_per_node < 1:
            raise ValueError("generation.max_records_per_node must be >= 1")

    def fingerprint(self) -> str:
        blob = json.dumps(
            {
                "kinds": self.kinds,
                "languages": self.languages,
                "difficulties": self.difficulties,
                "min_teachers": self.min_teachers,
                "allow_single_teacher": self.allow_single_teacher,
                "require_verification_for": self.require_verification_for,
                "max_records_per_node": self.max_records_per_node,
                "seed": self.seed,
            },
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class DraftOutcome:
    request: GenerationRequest
    draft: TeacherDraft
    critiques: list[Critique]
    verdict: str


@dataclass
class PipelineResult:
    requests: list[GenerationRequest] = field(default_factory=list)
    outcomes: list[ConsensusOutcome] = field(default_factory=list)
    accepted: list[TrainingRecord] = field(default_factory=list)
    quarantined: list[tuple[TrainingRecord, str]] = field(default_factory=list)
    rejected: list[tuple[TrainingRecord, str]] = field(default_factory=list)
    critiques: list[Critique] = field(default_factory=list)
    draft_count: int = 0
    providers: list[dict[str, Any]] = field(default_factory=list)
    config_fingerprint: str = ""
    seed: int = 0
    notes: list[str] = field(default_factory=list)
    created_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        by_status: dict[str, int] = {}
        for o in self.outcomes:
            by_status[o.status.value] = by_status.get(o.status.value, 0) + 1
        by_kind: dict[str, int] = {}
        for r in self.accepted:
            by_kind[r.kind] = by_kind.get(r.kind, 0) + 1
        by_language: dict[str, int] = {}
        for r in self.accepted:
            by_language[r.language] = by_language.get(r.language, 0) + 1
        return {
            "tool": "teachers/pipeline.py",
            "pipeline_version": PIPELINE_VERSION,
            "created_utc": self.created_utc,
            "seed": self.seed,
            "config_fingerprint": self.config_fingerprint,
            "providers": self.providers,
            "requests": len(self.requests),
            "drafts": self.draft_count,
            "consensus": dict(sorted(by_status.items())),
            "accepted": len(self.accepted),
            "quarantined": len(self.quarantined),
            "rejected": len(self.rejected),
            "accepted_by_kind": dict(sorted(by_kind.items())),
            "accepted_by_language": dict(sorted(by_language.items())),
            "critiques": summarise_critiques(self.critiques),
            "record_stats": record_stats(self.accepted) if self.accepted else {"records": 0},
            "notes": self.notes,
            "request_plan": [r.to_dict() for r in self.requests],
            "outcomes": [o.to_dict() for o in self.outcomes],
            "quarantine_reasons": [
                {"record_id": r.record_id, "reason": reason, "verification_status": r.verification_status}
                for r, reason in self.quarantined
            ],
            "rejected_reasons": [{"record_id": r.record_id, "reason": reason} for r, reason in self.rejected],
        }

    def write(self, out_dir: Path | str) -> dict[str, Path]:
        from training.data.records import write_records

        d = resolve(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        paths = {
            "records": write_records(d / "records.jsonl", self.accepted),
            "quarantine": write_records(d / "quarantine.jsonl", [r for r, _ in self.quarantined]),
            "rejected": write_records(d / "rejected.jsonl", [r for r, _ in self.rejected]),
            "qc_report": write_json(d / "qc_report.json", self.to_dict()),
        }
        return paths


class TeacherPipeline:
    def __init__(self, graph: CurriculumGraph, registry: ProviderRegistry, config: GenerationConfig):
        self.graph = graph
        self.registry = registry
        self.config = config

    def plan_requests(self, nodes: Sequence[str] | None = None) -> list[GenerationRequest]:
        node_ids = list(nodes) if nodes else self.graph.order()
        requests: list[GenerationRequest] = []
        for node_id in node_ids:
            per_node = 0
            for kind in self.config.kinds:
                for language in self.config.languages:
                    for difficulty in self.config.difficulties:
                        if per_node >= self.config.max_records_per_node:
                            break
                        requests.append(
                            build_request(
                                self.graph,
                                node_id,
                                kind,
                                language,
                                difficulty,
                                index=per_node,
                                seed=self.config.seed,
                            )
                        )
                        per_node += 1
        return requests

    def run(self, nodes: Sequence[str] | None = None, *, created_utc: str = "") -> PipelineResult:
        result = PipelineResult(
            providers=self.registry.specs(),
            config_fingerprint=self.config.fingerprint(),
            seed=self.config.seed,
            created_utc=created_utc,
        )
        for request in self.plan_requests(nodes):
            result.requests.append(request)
            drafts = self.registry.generate(request)
            result.draft_count += len(drafts)

            survivors: list[TeacherDraft] = []
            for draft in drafts:
                critiques = critique_draft(
                    draft,
                    language=request.language,
                    required_fields=("task", "response", "answer"),
                    prompt_text=request.prompt,
                )
                result.critiques.extend(critiques)
                verdict = worst_verdict(critiques)
                if verdict == "reject":
                    result.notes.append(
                        f"{request.request_id}: draft from {draft.teacher.model_id} rejected by critics "
                        f"({[c for c in critiques if c.verdict == 'reject'][0].detail})"
                    )
                    continue
                survivors.append(draft)

            outcome = reach_consensus(
                survivors,
                request_id=request.request_id,
                mode=_mode_for_payload(survivors),
                expected=None,
                min_teachers=self.config.min_teachers,
                allow_single_teacher=self.config.allow_single_teacher,
            )
            result.outcomes.append(outcome)
            if not survivors:
                continue

            record = self._record_from(request, survivors, outcome, created_utc=created_utc)
            reason = self._route_reason(record, outcome)
            if reason is None:
                result.accepted.append(record)
            elif outcome.status is ConsensusStatus.REJECTED:
                result.rejected.append((record, reason))
            else:
                result.quarantined.append((record, reason))

        if not result.accepted:
            result.notes.append(
                "no records were accepted: providers produced no drafts, or none survived critics, "
                "consensus and the licence/verification policy. Nothing was fabricated to fill the gap."
            )
        return result

    # ---- helpers ---------------------------------------------------------------------
    def _record_from(
        self,
        request: GenerationRequest,
        survivors: list[TeacherDraft],
        outcome: ConsensusOutcome,
        *,
        created_utc: str,
    ) -> TrainingRecord:
        node = self.graph.get(request.node_id)
        # canonical draft: prefer the one whose answer matches the consensus answer
        canonical = next((d for d in survivors if d.answer == (outcome.canonical_answer or "")), survivors[0])
        payload = canonical.payload or {}
        rationale = str(payload.get("rationale", "")).strip()
        rationale_style = "none"
        if rationale:
            rationale_style = "structured_steps" if "\n" in rationale and any(ch.isdigit() for ch in rationale[:40]) else "concise"
        reuse_ok = all(d.teacher.output_reuse_permitted for d in survivors)
        error_tags = [str(t) for t in (payload.get("error_tags") or [])]
        if request.kind in ("error_diagnosis", "correction") and not error_tags:
            error_tags = sorted({c for cr in survivors for ct in (cr.payload.get("_critic_codes") or []) for c in ct}) or [
                "UNCLASSIFIED"
            ]
        return TrainingRecord(
            record_id=request.request_id,
            kind=request.kind,
            task=str(payload.get("task", "")),
            response=str(payload.get("response", "")),
            curriculum_node=node.id,
            domain=node.track,
            subject=node.subject,
            grade_level=node.grade,
            language=request.language,
            difficulty=request.difficulty,
            provenance="synthetic",
            verification_status=outcome.verification_status_for_record(),
            license_status="cleared" if reuse_ok else "unknown_quarantine",
            source=f"{PIPELINE_VERSION}:{'/'.join(outcome.teachers) or 'unknown'}",
            rationale=rationale,
            rationale_style=rationale_style,
            teacher_models=outcome.teachers,
            error_tags=error_tags,
            notes=f"consensus={outcome.status.value}; " + "; ".join(outcome.notes),
            created_utc=created_utc,
        )

    def _route_reason(self, record: TrainingRecord, outcome: ConsensusOutcome) -> str | None:
        if outcome.status is ConsensusStatus.REJECTED:
            return "consensus REJECTED: no teacher answer passed verification"
        if not outcome.usable:
            return f"consensus {outcome.status.value}: {outcome.notes[-1] if outcome.notes else 'not usable'}"
        if record.license_status != "cleared":
            return "teacher output reuse terms unstated (output_reuse_permitted=false): quarantined by policy"
        if record.kind in self.config.require_verification_for and record.verification_status not in (
            "deterministically_verified",
            "human_reviewed",
        ):
            return (
                f"kind {record.kind!r} requires verification; consensus gave "
                f"{record.verification_status!r}"
            )
        errs = record.validate()
        if errs:
            return f"record failed schema validation: {errs[0]}"
        return None


def _mode_for_payload(drafts: Iterable[TeacherDraft]) -> str:
    """Drafts may declare how they can be checked; anything undeclared stays a manual review."""
    for d in drafts:
        declared = str((d.payload or {}).get("verification_mode", "")).strip()
        if declared:
            return declared
    return "manual"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the curriculum data-generation pipeline.")
    ap.add_argument("--graph", default=str(DEFAULT_GRAPH))
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--nodes", nargs="*", help="curriculum node ids (default: whole graph)")
    ap.add_argument("--transcript", action="append", default=[], help="teacher transcript JSONL (repeatable)")
    ap.add_argument("--template", action="store_true", help="use the offline template provider (no facts)")
    ap.add_argument("--out", default="data/generated/curriculum_v0")
    args = ap.parse_args(argv)

    from teachers.providers import RecordedProvider

    graph = CurriculumGraph.load(resolve(args.graph))
    cfg = GenerationConfig.from_config(load_config(resolve(args.config)))
    registry = ProviderRegistry()
    for path in args.transcript:
        provider = RecordedProvider.from_jsonl(path)
        registry.add(provider)
    if args.template or not args.transcript:
        registry.add(TemplateProvider())
        print(
            "NOTE: template provider output is synthetic scaffolding with no factual authority; it "
            "exercises the pipeline, it does not teach anything."
        )
    pipeline = TeacherPipeline(graph, registry, cfg)
    result = pipeline.run(args.nodes)
    paths = result.write(args.out)
    print(f"requests={len(result.requests)} drafts={result.draft_count} accepted={len(result.accepted)} "
          f"quarantined={len(result.quarantined)} rejected={len(result.rejected)}")
    for name, p in paths.items():
        print(f"  {name}: {rel(p)}")
    for note in result.notes[:10]:
        print(f"  note: {note}")
    return 0 if result.accepted else 1




    def _collect_teacher_responses(
        self, request: GenerationRequest, drafts: list[TeacherDraft]
    ) -> list[TeacherResponseRecord]:
        """Collect structured teacher responses with full provenance."""
        records = []
        for draft in drafts:
            tr = TeacherResponseRecord(
                task_id=request.request_id,
                teacher_id=draft.teacher.model_id,
                teacher_provider=draft.teacher.vendor or "unknown",
                model_identifier=draft.teacher.model_id,
                response=draft.payload.get("response", ""),
                task_domain=request.kind,
                curriculum_node=request.node_id,
                language=request.language,
                difficulty=request.difficulty,
                generation_timestamp=time.time(),
                provenance=f"synthetic:{draft.teacher.model_id}",
                verification_status="single_teacher",
                critic_status=worst_verdict(
                    critique_draft(draft, language=request.language, required_fields=("task", "response", "answer"), prompt_text=request.prompt)
                ),
                payload=draft.payload,
            )
            records.append(tr)
        return records

    def _perform_synthesis(
        self,
        request: GenerationRequest,
        result: PipelineResult,
        teacher_responses: list[TeacherResponseRecord],
        outcome: ConsensusOutcome,
    ) -> None:
        """Perform multi-model synthesis of teacher responses.

        Identifies common points, disagreements, unique insights,
        and produces a synthesized answer with provenance.
        """
        from teachers.consensus import normalise

        tr_ids = [t.task_id for t in teacher_responses]
        # Normalise answers for comparison
        answer_groups: dict[str, list[TeacherResponseRecord]] = {}
        for t in teacher_responses:
            key = normalise(t.response)
            answer_groups.setdefault(key, []).append(t)

        common_points: list[str] = []
        disagreements: list[str] = []
        unique_insights: list[str] = []
        rejected_claims: list[str] = []

        if len(outcome.groups) == 1:
            # All agree
            group_key = next(iter(outcome.groups))
            common_points.append(f"All {len(teacher_responses)} teachers agree on answer")
            selected_reasoning = outcome.canonical_answer or teacher_responses[0].response
            synthesized = outcome.canonical_answer or teacher_responses[0].response
            synthesis_method = SynthesisMethod.CONSENSUS_VERIFIED
            confidence = "high"
        elif outcome.usable:
            # Some resolution occurred
            if outcome.status == ConsensusStatus.RESOLVED_BY_VERIFICATION:
                # One answer verified, others not
                verified_answer = outcome.canonical_answer
                verified_record = next(
                    (t for t in teacher_responses if normalise(t.response) == normalise(verified_answer)), None
                )
                selected_reasoning = verified_answer or (verified_record.response if verified_record else "")
                synthesized = verified_answer or (verified_record.response if verified_record else "")
                synthesis_method = SynthesisMethod.CONSENSUS_VERIFIED
                confidence = "high"
            else:
                # AGREED but with verification path
                selected_reasoning = outcome.canonical_answer or teacher_responses[0].response
                synthesized = outcome.canonical_answer or teacher_responses[0].response
                synthesis_method = SynthesisMethod.CONSENSUS_VERIFIED
                confidence = "medium"
        else:
            # Disagreement - collect and flag
            for key, group in answer_groups.items():
                if len(group) == 1:
                    # Unique answer - could be an insight
                    unique_insights.append(f"Teacher {group[0].teacher_id}: {group[0].response[:80]}...")
                else:
                    # Multiple teachers with same answer but no consensus - flag
                    disagreements.append(f"Multiple teachers agree on: {group[0].response[:80]}... (no consensus reached)")
            # Check for unsupported claims
            for t in teacher_responses:
                rejected_claims.extend(self._detect_unsupported_claims(t.response, request.kind))
            selected_reasoning = "; ".join(disagreements[:3]) if disagreements else "No consensus reached"
            synthesized = "DISAGREEMENT: teachers could not reach agreement; see rejection/quarantine rationale"
            synthesis_method = SynthesisMethod.MAJORITY_VOTE
            confidence = "low"

        # Build the synthesis record
        provenance = f"synthetic:{'/'.join(outcome.teachers) or 'unknown'}"
        synthesis_record = ArenaSynthesisRecord(
            task_id=request.request_id,
            teacher_response_ids=tr_ids,
            common_points=common_points,
            disagreements=disagreements,
            unique_insights=unique_insights,
            rejected_claims=rejected_claims,
            selected_reasoning=selected_reasoning,
            synthesized_response=synthesized,
            verification_results={},
            synthesis_method=synthesis_method,
            confidence_status=confidence,
            provenance=provenance,
        )
        result.synthesis_record = synthesis_record

        # Attach synthesis info to the record's notes
        if synthesis_record:
            result.notes.append(
                f"synthesis={synthesis_record.synthesis_method.value}; "
                f"confidence={synthesis_record.confidence_status}; "
                f"common_points={len(synthesis_record.common_points)}; "
                f"disagreements={len(synthesis_record.disagreements)}; "
                f"unique_insights={len(synthesis_record.unique_insights)}; "
                f"rejected_claims={len(synthesis_record.rejected_claims)}"
            )

    @staticmethod
    def _detect_unsupported_claims(response: str, task_domain: str) -> list[str]:
        """Detect claims in the response that are not supported by evidence."""
        claims: list[str] = []
        # Check for arithmetic claims that weren't verified
        import re
        arithmetic_pattern = r"\d+[\/\*\-+]\d+[\/\*\-+]\d+"
        if re.search(arithmetic_pattern, response):
            claims.append("potential unsupported arithmetic claim")
        return claims
if __name__ == "__main__":
    raise SystemExit(main())
