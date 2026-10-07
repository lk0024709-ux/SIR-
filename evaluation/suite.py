"""Curated evaluation suites: cases, scoring, and a result file promotion can audit.

The existing evaluation scripts measure a *language model* (perplexity, degeneration, latency). This
module measures *task behaviour* on curated cases across the axes the promotion gate requires:
knowledge, mathematics, science, language, reasoning, coding, application, transfer, verification,
brainstorming and reliability.

Scoring is where honesty is hardest, so it is explicit:

* every case declares its `verification` mode;
* deterministic modes (`exact`, `numeric`, `set`, `arithmetic_claims`) are scored by
  `teachers/verify.py`, offline and reproducibly;
* `manual` and `none` cases are **UNSCORED** — counted, reported, and excluded from the axis score.
  They are not silently treated as passes, and they are not quietly dropped either;
* the run records `verified_evidence: true` only when every scored case used a deterministic mode, so
  a promotion gate can tell "checked by execution" apart from "graded by vibes".

Scoring a model's own output with the same model is deliberately not an option here: an
`Answerer` is either static (recorded answers) or a local checkpoint via `inference/`. There is no
network path in this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence

from sir_paths import REPO_ROOT, rel, resolve, write_json
from teachers.verify import CheckResult, CheckStatus, VerificationMode, verify_answer

AXES = (
    "knowledge",
    "mathematics",
    "science",
    "language",
    "reasoning",
    "coding",
    "application",
    "transfer",
    "verification",
    "brainstorming",
    "reliability",
)
STATUSES = ("pass", "fail", "unscored", "error")
DETERMINISTIC_MODES = {
    VerificationMode.EXACT.value,
    VerificationMode.NUMERIC.value,
    VerificationMode.SET.value,
    VerificationMode.ARITHMETIC_CLAIMS.value,
    VerificationMode.INTEGRITY_ONLY.value,
}
DEFAULT_SUITE_DIR = REPO_ROOT / "evaluation" / "suites"


@dataclass
class EvalCase:
    case_id: str
    axis: str
    prompt: str
    expected: str | None = None
    verification: str = VerificationMode.EXACT.value
    curriculum_node: str = ""
    subject: str = ""
    language: str = "en"
    difficulty: str = "beginner"
    accepted: list[str] = field(default_factory=list)
    tolerance: float = 0.0
    tags: list[str] = field(default_factory=list)
    source: str = ""
    provenance: str = "self_authored"
    notes: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvalCase":
        return cls(
            case_id=str(d.get("case_id", "")),
            axis=str(d.get("axis", "")),
            prompt=str(d.get("prompt", "")),
            expected=None if d.get("expected") is None else str(d.get("expected")),
            verification=str(d.get("verification", VerificationMode.EXACT.value)),
            curriculum_node=str(d.get("curriculum_node", "")),
            subject=str(d.get("subject", "")),
            language=str(d.get("language", "en")),
            difficulty=str(d.get("difficulty", "beginner")),
            accepted=[str(a) for a in (d.get("accepted") or [])],
            tolerance=float(d.get("tolerance", 0.0) or 0.0),
            tags=[str(t) for t in (d.get("tags") or [])],
            source=str(d.get("source", "")),
            provenance=str(d.get("provenance", "self_authored")),
            notes=str(d.get("notes", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "axis": self.axis,
            "prompt": self.prompt,
            "expected": self.expected,
            "verification": self.verification,
            "curriculum_node": self.curriculum_node,
            "subject": self.subject,
            "language": self.language,
            "difficulty": self.difficulty,
            "accepted": list(self.accepted),
            "tolerance": self.tolerance,
            "tags": list(self.tags),
            "source": self.source,
            "provenance": self.provenance,
            "notes": self.notes,
        }

    def validate(self) -> list[str]:
        errs: list[str] = []
        if not self.case_id:
            errs.append("case_id is required")
        if self.axis not in AXES:
            errs.append(f"{self.case_id}: axis {self.axis!r} not in {AXES}")
        if not self.prompt.strip():
            errs.append(f"{self.case_id}: prompt is empty")
        if self.verification not in {m.value for m in VerificationMode}:
            errs.append(f"{self.case_id}: verification mode {self.verification!r} is not recognised")
        if self.verification in DETERMINISTIC_MODES and self.verification != VerificationMode.INTEGRITY_ONLY.value:
            if self.expected is None or not str(self.expected).strip():
                errs.append(f"{self.case_id}: mode {self.verification!r} needs an expected answer to score against")
        if self.accepted and self.verification in (VerificationMode.MANUAL.value, VerificationMode.NONE.value):
            errs.append(
                f"{self.case_id}: 'accepted' alternatives only make sense for scored modes; a manual case is "
                "reviewer-scored and must not carry machine-checked alternatives"
            )
        if not self.source.strip():
            errs.append(f"{self.case_id}: source is required (which suite/authoring pass produced this case)")
        return errs


def load_suite(path: Path | str) -> list[EvalCase]:
    p = resolve(path)
    if not p.exists():
        raise FileNotFoundError(f"evaluation suite not found: {p}")
    cases: list[EvalCase] = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            case = EvalCase.from_dict(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{rel(p)}:{i}: not valid JSON ({exc})") from exc
        errs = case.validate()
        if errs:
            raise ValueError(f"{rel(p)}:{i}: " + "; ".join(errs))
        cases.append(case)
    return cases


def load_suites(paths: Iterable[str] | None = None, directory: Path | str = DEFAULT_SUITE_DIR) -> list[EvalCase]:
    out: list[EvalCase] = []
    if paths:
        for p in paths:
            out.extend(load_suite(p))
        return out
    d = resolve(directory)
    for p in sorted(d.glob("*.jsonl")):
        out.extend(load_suite(p))
    seen: dict[str, str] = {}
    for case in out:
        if case.case_id in seen:
            raise ValueError(f"duplicate case_id {case.case_id!r}: two suite files would be scored as one set")
        seen[case.case_id] = case.axis
    return out


def suite_hash(cases: Sequence[EvalCase]) -> str:
    """Stable identity of a case set: two runs are comparable only if this matches."""
    blob = json.dumps([c.to_dict() for c in sorted(cases, key=lambda c: c.case_id)], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------
# answerers
# --------------------------------------------------------------------------------------
class Answerer(Protocol):
    name: str

    def answer(self, case: EvalCase) -> str:  # pragma: no cover - protocol
        ...


@dataclass
class StaticAnswerer:
    """Recorded answers keyed by case id (a transcript, a reviewer's answer sheet, or a baseline)."""

    answers: dict[str, str | Callable[[EvalCase], str]]
    name: str = "static"

    def answer(self, case: EvalCase) -> str:
        value = self.answers.get(case.case_id, "")
        if callable(value):
            return str(value(case))
        return str(value)


@dataclass
class ModelAnswerer:
    """Answers from a local SIR checkpoint through `inference/`. No network, no external model."""

    checkpoint: Path | str
    max_new_tokens: int = 64
    temperature: float = 0.0
    name: str = "sir-checkpoint"

    def __post_init__(self) -> None:
        from inference.engine import SirEngine  # imported lazily: evaluation without a model must not need torch

        self._engine = SirEngine.from_checkpoint(self.checkpoint)

    def answer(self, case: EvalCase) -> str:
        return self._engine.generate(case.prompt, max_new_tokens=self.max_new_tokens, temperature=self.temperature)


# --------------------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------------------
@dataclass
class CaseResult:
    case_id: str
    axis: str
    status: str
    answer: str
    expected: str | None
    verification: str
    error_codes: list[str] = field(default_factory=list)
    detail: str = ""
    curriculum_node: str = ""
    language: str = "en"
    difficulty: str = "beginner"
    checks: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "axis": self.axis,
            "status": self.status,
            "answer": self.answer,
            "expected": self.expected,
            "verification": self.verification,
            "error_codes": list(self.error_codes),
            "detail": self.detail,
            "curriculum_node": self.curriculum_node,
            "language": self.language,
            "difficulty": self.difficulty,
            "checks": self.checks,
        }


def score_case(case: EvalCase, answer: str) -> CaseResult:
    if case.verification in (VerificationMode.MANUAL.value, VerificationMode.NONE.value):
        return CaseResult(
            case_id=case.case_id,
            axis=case.axis,
            status="unscored",
            answer=answer,
            expected=case.expected,
            verification=case.verification,
            detail="declared manual/none: nothing was scored automatically (counted, not counted as a pass)",
            curriculum_node=case.curriculum_node,
            language=case.language,
            difficulty=case.difficulty,
        )
    candidates = [case.expected] + [a for a in case.accepted if a != case.expected]
    attempts: list[CheckResult] = []
    result: CheckResult | None = None
    for candidate in candidates:
        check = verify_answer(answer, candidate, case.verification, tolerance=case.tolerance)
        attempts.append(check)
        result = check
        if check.status == CheckStatus.PASS.value:
            break
    assert result is not None  # at least `expected` is in candidates
    status = {"pass": "pass", "fail": "fail", "unverifiable": "unscored"}[result.status]
    detail = result.detail if len(attempts) == 1 else f"{result.detail} (checked {len(attempts)} accepted answer forms)"
    return CaseResult(
        case_id=case.case_id,
        axis=case.axis,
        status=status,
        answer=answer,
        expected=case.expected,
        verification=case.verification,
        error_codes=list(result.error_codes),
        detail=detail,
        curriculum_node=case.curriculum_node,
        language=case.language,
        difficulty=case.difficulty,
        checks=[a.to_dict() for a in attempts],
    )


def run_suite(
    cases: Sequence[EvalCase],
    answerer: Answerer,
    *,
    suite_paths: Iterable[str] | None = None,
    checkpoint: str = "",
    created_utc: str = "",
    provisional: bool = False,
) -> dict[str, Any]:
    results: list[CaseResult] = []
    for case in sorted(cases, key=lambda c: c.case_id):
        try:
            answer = answerer.answer(case)
        except Exception as exc:  # a crashing answerer must be visible, not counted as a wrong answer
            results.append(
                CaseResult(
                    case_id=case.case_id,
                    axis=case.axis,
                    status="error",
                    answer="",
                    expected=case.expected,
                    verification=case.verification,
                    detail=f"{type(exc).__name__}: {exc}",
                    curriculum_node=case.curriculum_node,
                    language=case.language,
                    difficulty=case.difficulty,
                )
            )
            continue
        results.append(score_case(case, answer))

    axes = aggregate(results)
    scored = [r for r in results if r.status in ("pass", "fail")]
    verified = bool(scored) and all(r.verification in DETERMINISTIC_MODES for r in scored)
    payload: dict[str, Any] = {
        "tool": "evaluation/suite.py",
        "created_utc": created_utc,
        "answerer": getattr(answerer, "name", "unknown"),
        "checkpoint": checkpoint,
        "suites": [rel(resolve(p)) for p in (suite_paths or [])],
        "suite_hash": suite_hash(cases),
        "cases_total": len(cases),
        "cases_scored": len(scored),
        "cases_unscored": sum(1 for r in results if r.status == "unscored"),
        "cases_error": sum(1 for r in results if r.status == "error"),
        "verified_evidence": verified,
        "verified_evidence_note": (
            "true only when every scored case used a deterministic verification mode "
            f"({sorted(DETERMINISTIC_MODES)}); manual cases are unscored and excluded"
        ),
        "provisional": provisional,
        "axes": axes,
        "cases": [r.to_dict() for r in results],
    }
    return payload


def aggregate(results: Sequence[CaseResult]) -> dict[str, dict[str, Any]]:
    axes: dict[str, dict[str, Any]] = {}
    for r in results:
        bucket = axes.setdefault(
            r.axis, {"cases": 0, "scored": 0, "passed": 0, "failed": 0, "unscored": 0, "error": 0, "score": None, "error_codes": {}}
        )
        bucket["cases"] += 1
        if r.status in ("pass", "fail"):
            bucket["scored"] += 1
        if r.status == "pass":
            bucket["passed"] += 1
        elif r.status == "fail":
            bucket["failed"] += 1
            for code in r.error_codes:
                bucket["error_codes"][code] = bucket["error_codes"].get(code, 0) + 1
        elif r.status == "unscored":
            bucket["unscored"] += 1
        else:
            bucket["error"] += 1
    for axis, bucket in axes.items():
        bucket["score"] = round(bucket["passed"] / bucket["scored"], 4) if bucket["scored"] else None
        bucket["error_codes"] = dict(sorted(bucket["error_codes"].items()))
    return dict(sorted(axes.items()))


def write_result(payload: dict[str, Any], path: Path | str) -> Path:
    return write_json(resolve(path), payload)


def summarise(payload: dict[str, Any]) -> str:
    lines = [
        f"suite hash {payload['suite_hash'][:12]} | answerer={payload['answerer']} | "
        f"scored={payload['cases_scored']}/{payload['cases_total']} unscored={payload['cases_unscored']} "
        f"errors={payload['cases_error']}",
        f"verified_evidence={payload['verified_evidence']}",
    ]
    for axis, bucket in payload["axes"].items():
        score = "unscored" if bucket["score"] is None else f"{bucket['score']:.2f}"
        lines.append(
            f"  {axis:14s} {score:>8s}  (passed {bucket['passed']} / scored {bucket['scored']}, "
            f"unscored {bucket['unscored']}, errors {bucket['error']})"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run curated evaluation suites and score them deterministically.")
    ap.add_argument("--suite", action="append", default=[], help="suite JSONL (repeatable; default: all in evaluation/suites)")
    ap.add_argument("--answers", help="JSONL of {case_id, answer} recorded answers")
    ap.add_argument("--checkpoint", help="SIR checkpoint to answer with (mutually exclusive with --answers)")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--out", default="evaluation/results/suite_v0.json")
    ap.add_argument("--created-utc", default="")
    ap.add_argument("--provisional", action="store_true")
    args = ap.parse_args(argv)

    cases = load_suites(args.suite)
    if not cases:
        print("no evaluation cases found; nothing to run")
        return 2
    errs = [e for c in cases for e in c.validate()]
    if errs:
        print(f"{len(errs)} invalid case(s):")
        for e in errs[:10]:
            print(f"  - {e}")
        return 2

    if args.answers:
        recorded: dict[str, str] = {}
        for line in resolve(args.answers).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                recorded[str(row["case_id"])] = str(row.get("answer", ""))
        answerer: Answerer = StaticAnswerer(recorded, name=f"recorded:{rel(resolve(args.answers))}")
    elif args.checkpoint:
        answerer = ModelAnswerer(args.checkpoint, max_new_tokens=args.max_new_tokens)
    else:
        print("supply --answers or --checkpoint: there is no default answerer (an evaluation without a model is meaningless)")
        return 2

    payload = run_suite(cases, answerer, suite_paths=args.suite, checkpoint=args.checkpoint or "", created_utc=args.created_utc, provisional=args.provisional)
    print(summarise(payload))
    print(f"written: {rel(write_result(payload, args.out))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
