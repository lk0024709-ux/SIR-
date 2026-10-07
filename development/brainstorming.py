"""Brainstorming: the workflow, the rubric, and an honest mechanical score.

SIR's target is structured brainstorming — decompose → generate → connect → compare → challenge →
verify → synthesise — not "produce many sentences". This module holds three things:

1. `WORKFLOW` — the canonical stage list, so an attempt can be checked for *coverage of the process*,
   not only for output volume.
2. `BrainstormAttempt` — the artifact an evaluation or a teacher produces (problem, constraints, ideas,
   alternatives, trade-offs, challenges, verification steps, synthesis).
3. `score_attempt` — deterministic, evidence-carrying scoring per rubric dimension
   (`evaluation/rubrics/brainstorming_v0.yaml`).

What the score is and is not:

* It **is** reproducible: the same attempt always scores the same, every fired signal is reported, and
  no model is consulted.
* It is **not** a judgement of quality. The signals are lexical proxies: an idea containing "because"
  scores higher on reasoning quality than one that does not, which is a heuristic, not understanding.
* When a dimension cannot be measured (novelty without a reference set, too few items to compare), the
  dimension is reported as `UNSCORED` and excluded from the weighted mean — silently treating it as
  zero would understate the attempt, and silently treating it as one would overstate it.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from sir_paths import REPO_ROOT, load_config, rel, resolve, write_json

DEFAULT_RUBRIC = REPO_ROOT / "evaluation" / "rubrics" / "brainstorming_v0.yaml"

WORKFLOW = (
    "problem",
    "clarify_objective",
    "identify_constraints",
    "decompose",
    "retrieve_knowledge",
    "generate_hypotheses",
    "cross_domain_connect",
    "alternative_approaches",
    "compare_tradeoffs",
    "challenge_assumptions",
    "adversarial_critique",
    "verify",
    "synthesize",
    "actionable_answer",
)

CAUSAL_CONNECTIVES = (
    "because", "therefore", "so that", "hence", "leads to", "results in", "implies", "since",
    "इसलिए", "क्योंकि", "जिससे", "नतीजा",
)
VERIFICATION_VERBS = (
    "verify", "check", "test", "measure", "estimate", "validate", "compare with", "calibrate",
    "confirm", "जाँच", "जांच", "माप", "सत्यापित", "अनुमान",
)
COMPARISON_WORDS = ("better", "worse", "trade-off", "tradeoff", "versus", "vs", "instead", "prefer", "however", "बेहतर", "मुकाबले")
RECOMMENDATION_WORDS = ("recommend", "prefer", "choose", "best option", "should", "सुझाव", "चुनना", "बेहतर होगा")
CONSTRAINT_VIOLATION_WORDS = ("unlimited budget", "ignore the", "disregard", "no limit", "any cost", "जरूरत नहीं")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|kg|km|g|m|s|hour|hours|day|days|rupees|rs\.?|₹|watts|w|litre|litres|mb|gb|ms)?", re.I)
RESOURCE_WORDS = ("budget", "cost", "rs", "₹", "rupees", "staff", "time", "hours", "days", "energy", "material", "tools", "hardware", "compute", "पैसा", "समय", "लागत")
STEP_WORDS = ("first", "then", "next", "finally", "step", "phase", "stage", "पहले", "फिर", "अंत में")
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are", "be", "as",
    "by", "that", "this", "it", "its", "at", "from", "we", "can", "will", "would", "should", "how",
    "what", "why", "when", "which", "की", "का", "के", "और", "है", "हैं", "को", "में", "से", "पर",
}


@dataclass
class BrainstormAttempt:
    attempt_id: str
    problem: str
    ideas: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    tradeoffs: list[str] = field(default_factory=list)
    challenges: list[str] = field(default_factory=list)
    verification_steps: list[str] = field(default_factory=list)
    cross_domain_links: list[str] = field(default_factory=list)
    synthesis: str = ""
    actionable_answer: str = ""
    subject: str = ""
    language: str = "en"
    provenance: str = "self_authored"
    source: str = ""
    references: list[str] = field(default_factory=list)  # for the novelty dimension

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BrainstormAttempt":
        def lst(key: str) -> list[str]:
            v = d.get(key) or []
            if isinstance(v, str):
                return [v]
            return [str(x) for x in v]

        return cls(
            attempt_id=str(d.get("attempt_id", "")),
            problem=str(d.get("problem", "")),
            ideas=lst("ideas"),
            constraints=lst("constraints"),
            alternatives=lst("alternatives"),
            tradeoffs=lst("tradeoffs"),
            challenges=lst("challenges"),
            verification_steps=lst("verification_steps"),
            cross_domain_links=lst("cross_domain_links"),
            synthesis=str(d.get("synthesis", "")),
            actionable_answer=str(d.get("actionable_answer", "")),
            subject=str(d.get("subject", "")),
            language=str(d.get("language", "en")),
            provenance=str(d.get("provenance", "self_authored")),
            source=str(d.get("source", "")),
            references=lst("references"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}  # type: ignore[attr-defined]


# --------------------------------------------------------------------------------------
# lexical helpers
# --------------------------------------------------------------------------------------
def _tokens(text: str) -> set[str]:
    words = re.findall(r"[\w\u0900-\u097F]+", text.casefold())
    return {w for w in words if len(w) > 2 and w not in STOPWORDS}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _contains_any(text: str, needles: Iterable[str]) -> list[str]:
    low = text.casefold()
    return [n for n in needles if n.casefold() in low]


def distinct_idea_ratio(ideas: Sequence[str], *, similarity_threshold: float = 0.6) -> tuple[float, list[str]]:
    """Fraction of ideas that are not near-duplicates of an earlier idea."""
    distinct: list[str] = []
    token_sets: list[set[str]] = []
    for idea in ideas:
        toks = _tokens(idea)
        if any(_jaccard(toks, prev) >= similarity_threshold for prev in token_sets):
            continue
        distinct.append(idea)
        token_sets.append(toks)
    return (len(distinct) / len(ideas) if ideas else 0.0), distinct


# --------------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------------
@dataclass
class DimensionScore:
    name: str
    score: float | None
    weight: float
    evidence: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def scored(self) -> bool:
        return self.score is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.name,
            "score": None if self.score is None else round(self.score, 4),
            "weight": self.weight,
            "scored": self.scored,
            "evidence": self.evidence,
            "notes": self.notes,
        }


@dataclass
class BrainstormScore:
    attempt_id: str
    dimensions: list[DimensionScore]
    total: float | None
    workflow_coverage: dict[str, bool]
    rubric_id: str = "brainstorming_v0"
    notes: list[str] = field(default_factory=list)

    @property
    def unscored_dimensions(self) -> list[str]:
        return [d.name for d in self.dimensions if not d.scored]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": "development/brainstorming.py",
            "attempt_id": self.attempt_id,
            "rubric_id": self.rubric_id,
            "total": None if self.total is None else round(self.total, 4),
            "dimensions": [d.to_dict() for d in self.dimensions],
            "unscored_dimensions": self.unscored_dimensions,
            "workflow_coverage": self.workflow_coverage,
            "workflow_stages_present": sum(1 for v in self.workflow_coverage.values() if v),
            "notes": self.notes,
        }


def _dimension_scores(attempt: BrainstormAttempt, rubric: dict[str, Any]) -> tuple[list[DimensionScore], list[str]]:
    dims = (rubric.get("dimensions") or {})
    notes: list[str] = []
    idea_text = " ".join(attempt.ideas)
    all_text = " ".join(
        [idea_text, " ".join(attempt.alternatives), " ".join(attempt.tradeoffs), attempt.synthesis, attempt.actionable_answer]
    )
    out: list[DimensionScore] = []

    def weight(name: str, default: float) -> float:
        return float((dims.get(name) or {}).get("weight", default))

    # diversity
    ratio, distinct = distinct_idea_ratio(attempt.ideas)
    out.append(
        DimensionScore(
            "diversity",
            round(ratio, 4),
            weight("diversity", 0.15),
            evidence=[f"{len(distinct)}/{len(attempt.ideas)} ideas are not near-duplicates"],
            notes="lexical distinctness; two genuinely different ideas sharing vocabulary count as one",
        )
    )

    # relevance
    problem_tokens = _tokens(attempt.problem)
    if not problem_tokens:
        out.append(DimensionScore("relevance", None, weight("relevance", 0.15), [], "no problem statement tokens to compare"))
    else:
        hits = [i for i, idea in enumerate(attempt.ideas) if _tokens(idea) & problem_tokens]
        out.append(
            DimensionScore(
                "relevance",
                round(len(hits) / len(attempt.ideas), 4) if attempt.ideas else 0.0,
                weight("relevance", 0.15),
                evidence=[f"{len(hits)}/{len(attempt.ideas)} ideas share vocabulary with the problem"],
                notes="vocabulary overlap is a proxy; topical-but-irrelevant ideas can still pass",
            )
        )

    # novelty (needs a reference set)
    if not attempt.references:
        out.append(
            DimensionScore(
                "novelty",
                None,
                weight("novelty", 0.10),
                [],
                "UNSCORED: no reference set supplied, so 'new relative to what' is unanswerable",
            )
        )
    else:
        ref_sets = [_tokens(r) for r in attempt.references]
        scores = []
        for idea in attempt.ideas:
            toks = _tokens(idea)
            scores.append(1.0 - max((_jaccard(toks, r) for r in ref_sets), default=0.0))
        out.append(
            DimensionScore(
                "novelty",
                round(sum(scores) / len(scores), 4) if scores else None,
                weight("novelty", 0.10),
                evidence=[f"compared against {len(ref_sets)} reference item(s)"],
                notes="lexical novelty only: semantic novelty needs review",
            )
        )

    # feasibility
    feats: list[str] = []
    for idea in attempt.ideas:
        if NUMBER_RE.search(idea):
            feats.append("quantity")
        if _contains_any(idea, RESOURCE_WORDS):
            feats.append("resource")
        if _contains_any(idea, STEP_WORDS):
            feats.append("steps")
    cov = len({f for f in feats})
    feas = (len([f for f in feats if f == "quantity"]) / len(attempt.ideas) if attempt.ideas else 0.0)
    score = min(1.0, 0.5 * feas + 0.2 * cov + (0.3 if attempt.tradeoffs else 0.0))
    out.append(
        DimensionScore(
            "feasibility",
            round(score, 4),
            weight("feasibility", 0.15),
            evidence=[f"signal kinds present: {sorted(set(feats))}", f"ideas with concrete quantities: {feats.count('quantity')}"],
            notes="concreteness proxy; a detailed but infeasible plan can score highly",
        )
    )

    # reasoning quality
    causal = _contains_any(all_text, CAUSAL_CONNECTIVES)
    assumptions = len(attempt.challenges) + (1 if _contains_any(all_text, ("assume", "assumption", "मान लें")) else 0)
    score = min(1.0, 0.6 * (1.0 if causal else 0.0) + 0.4 * min(1.0, assumptions / 2))
    out.append(
        DimensionScore(
            "reasoning_quality",
            round(score, 4),
            weight("reasoning_quality", 0.15),
            evidence=[f"causal connectives: {causal}", f"challenge/assumption statements: {assumptions}"],
            notes="presence of explanation markers, not a check that the reasoning is valid",
        )
    )

    # constraint adherence
    if not attempt.constraints:
        out.append(DimensionScore("constraint_adherence", None, weight("constraint_adherence", 0.10), [], "no constraints declared for this problem"))
    else:
        covered = []
        for c in attempt.constraints:
            ctoks = _tokens(c)
            if ctoks and (_tokens(all_text) & ctoks):
                covered.append(c)
        violations = _contains_any(all_text, CONSTRAINT_VIOLATION_WORDS)
        score = len(covered) / len(attempt.constraints)
        if violations:
            score = max(0.0, score - 0.5)
        out.append(
            DimensionScore(
                "constraint_adherence",
                round(score, 4),
                weight("constraint_adherence", 0.10),
                evidence=[f"{len(covered)}/{len(attempt.constraints)} constraints referenced", f"violation phrases: {violations}"],
                notes="reference detection is lexical; a violated constraint that is never mentioned is not caught",
            )
        )

    # verification
    verbs = _contains_any(all_text, VERIFICATION_VERBS)
    explicit = len(attempt.verification_steps)
    units = len(NUMBER_RE.findall(all_text)) > 0
    score = min(1.0, 0.5 * (1.0 if verbs else 0.0) + 0.3 * min(1.0, explicit / 2) + 0.2 * (1.0 if units else 0.0))
    out.append(
        DimensionScore(
            "verification",
            round(score, 4),
            weight("verification", 0.10),
            evidence=[f"verification verbs: {verbs}", f"explicit verification steps: {explicit}", f"quantities present: {units}"],
            notes="claims made without any check are the failure mode this dimension is meant to catch",
        )
    )

    # synthesis
    rec = _contains_any(attempt.synthesis + " " + attempt.actionable_answer, RECOMMENDATION_WORDS)
    comp = _contains_any(attempt.synthesis + " " + " ".join(attempt.tradeoffs), COMPARISON_WORDS)
    refs = len(re.findall(r"\b(?:idea|option|approach)\s*\d+", attempt.synthesis, re.I))
    score = min(1.0, 0.4 * (1.0 if rec else 0.0) + 0.3 * (1.0 if comp else 0.0) + 0.3 * min(1.0, refs / 2))
    out.append(
        DimensionScore(
            "synthesis",
            round(score, 4),
            weight("synthesis", 0.10),
            evidence=[f"recommendation language: {rec}", f"comparison language: {comp}", f"explicit option references: {refs}"],
            notes="a synthesis that names no option and recommends nothing scores 0 here",
        )
    )
    return out, notes


def workflow_coverage(attempt: BrainstormAttempt) -> dict[str, bool]:
    low = " ".join(
        [attempt.problem, " ".join(attempt.ideas), " ".join(attempt.alternatives), " ".join(attempt.tradeoffs),
         " ".join(attempt.challenges), " ".join(attempt.verification_steps), " ".join(attempt.cross_domain_links),
         attempt.synthesis, attempt.actionable_answer]
    ).casefold()
    return {
        "problem": bool(attempt.problem.strip()),
        "clarify_objective": bool(_contains_any(low, ("objective", "goal", "aim", "उद्देश्य", "लक्ष्य")) or attempt.actionable_answer.strip()),
        "identify_constraints": bool(attempt.constraints),
        "decompose": bool(_contains_any(low, ("component", "part", "sub-problem", "decompose", "break down", "भाग", "चरण"))),
        "retrieve_knowledge": bool(_contains_any(low, ("literature", "known", "existing", "research shows", "prior work", "अध्ययन"))),
        "generate_hypotheses": bool(attempt.ideas),
        "cross_domain_connect": bool(attempt.cross_domain_links),
        "alternative_approaches": bool(attempt.alternatives) or len(attempt.ideas) > 1,
        "compare_tradeoffs": bool(attempt.tradeoffs) or bool(_contains_any(low, COMPARISON_WORDS)),
        "challenge_assumptions": bool(attempt.challenges),
        "adversarial_critique": bool(_contains_any(low, ("weakness", "fails", "counter", "risk", "limitation", "कमज़ोर", "जोखिम"))),
        "verify": bool(attempt.verification_steps) or bool(_contains_any(low, VERIFICATION_VERBS)),
        "synthesize": bool(attempt.synthesis.strip()),
        "actionable_answer": bool(attempt.actionable_answer.strip()) or bool(_contains_any(low, RECOMMENDATION_WORDS)),
    }


def score_attempt(
    attempt: BrainstormAttempt,
    rubric: dict[str, Any] | None = None,
    *,
    rubric_id: str = "brainstorming_v0",
) -> BrainstormScore:
    rubric = rubric or {}
    dims, notes = _dimension_scores(attempt, rubric)
    min_items = int((rubric.get("reporting") or {}).get("min_items_for_scoring", 2))
    if len(attempt.ideas) < min_items:
        for d in dims:
            if d.name in ("diversity", "relevance", "novelty", "feasibility"):
                d.notes = (d.notes + " " if d.notes else "") + f"UNSCORED: fewer than {min_items} ideas supplied"
                d.score = None
        notes.append(f"attempt has {len(attempt.ideas)} idea(s); dimensions requiring comparison are UNSCORED")
    scored = [d for d in dims if d.scored]
    total = None
    if scored:
        wsum = sum(d.weight for d in scored)
        if wsum > 0:
            total = sum((d.score or 0.0) * d.weight for d in scored) / wsum
    else:
        notes.append("no dimension could be scored mechanically")
    notes.append(
        "mechanical rubric only: novelty and feasibility in particular require human/model review before "
        "any capability claim"
    )
    return BrainstormScore(
        attempt_id=attempt.attempt_id,
        dimensions=dims,
        total=total,
        workflow_coverage=workflow_coverage(attempt),
        rubric_id=rubric_id,
        notes=notes,
    )


def summarise(scores: Sequence[BrainstormScore]) -> dict[str, Any]:
    totals = [s.total for s in scores if s.total is not None]
    stage_counts: dict[str, int] = {stage: 0 for stage in WORKFLOW}
    for s in scores:
        for stage, present in s.workflow_coverage.items():
            stage_counts[stage] += 1 if present else 0
    return {
        "attempts": len(scores),
        "scored_attempts": len(totals),
        "mean_total": round(sum(totals) / len(totals), 4) if totals else None,
        "unscored_dimensions": sorted({d for s in scores for d in s.unscored_dimensions}),
        "workflow_stage_coverage": {k: f"{v}/{len(scores)}" for k, v in stage_counts.items()} if scores else {},
        "note": "workflow coverage counts attempts whose text shows the stage; it is not a quality score",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score brainstorming attempts against the v0 rubric.")
    ap.add_argument("--attempts", required=True, help="JSONL of brainstorm attempts")
    ap.add_argument("--rubric", default=str(DEFAULT_RUBRIC))
    ap.add_argument("--out", default="evaluation/results/brainstorming_score.json")
    args = ap.parse_args(argv)

    p = resolve(args.attempts)
    rubric = load_config(resolve(args.rubric))
    attempts = [BrainstormAttempt.from_dict(json.loads(line)) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    scores = [score_attempt(a, rubric, rubric_id=str(rubric.get("rubric_id", "brainstorming_v0"))) for a in attempts]
    payload = {"tool": "development/brainstorming.py", "summary": summarise(scores), "scores": [s.to_dict() for s in scores]}
    out = write_json(resolve(args.out), payload)
    for s in scores:
        print(f"{s.attempt_id}: total={s.total} stages={sum(1 for v in s.workflow_coverage.values() if v)}/{len(WORKFLOW)}"
              + (f" unscored={s.unscored_dimensions}" if s.unscored_dimensions else ""))
    print(f"summary: {json.dumps(summarise(scores))}")
    print(f"written: {rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
