"""Error-based learning: the taxonomy, the case format, and the conversion to training records.

A model that only ever sees correct answers learns to produce confident text, not to notice when it is
wrong. This module carries the other half: every detected mistake becomes a case that walks the same
path a careful tutor would —

    wrong answer → error classification → why it is wrong → correct principle →
    corrected answer → new similar problem → transfer problem

Two honest constraints:

* **Classification must be justified.** A category is assigned either from a deterministic error code
  produced by a verifier/critic (`CALCULATION_ERROR` → `CALCULATION_ERROR`) or by a reviewer, who is
  named in `classified_by`. There is no keyword-based guesser pretending to diagnose reasoning.
* **A case is evidence, not a template.** Every field is required, including the verification status of
  the corrected answer. A half-filled case is a validation error, because "the model got this wrong
  (we think)" is exactly the kind of unverifiable claim the pipeline is built to reject.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from sir_paths import rel, resolve
from training.data.records import TrainingRecord, write_records

DEFAULT_CASES = "data/errors/error_cases.jsonl"


class ErrorCategory(str, Enum):
    FACTUAL_ERROR = "FACTUAL_ERROR"
    LOGICAL_ERROR = "LOGICAL_ERROR"
    CALCULATION_ERROR = "CALCULATION_ERROR"
    LANGUAGE_ERROR = "LANGUAGE_ERROR"
    MISINTERPRETATION = "MISINTERPRETATION"
    HALLUCINATION = "HALLUCINATION"
    MISSING_CONTEXT = "MISSING_CONTEXT"
    BAD_ASSUMPTION = "BAD_ASSUMPTION"
    INCOMPLETE_REASONING = "INCOMPLETE_REASONING"
    OVERGENERALIZATION = "OVERGENERALIZATION"
    SOURCE_ERROR = "SOURCE_ERROR"
    CODE_ERROR = "CODE_ERROR"


#: Deterministic codes (from teachers/verify.py and teachers/critics.py) mapped onto the taxonomy.
#: Codes that are *not* errors (e.g. REQUIRES_HUMAN_REVIEW) are deliberately absent: they mean
#: "nobody checked yet", which is a different thing from "this is wrong".
CATEGORY_BY_CODE: dict[str, ErrorCategory] = {
    "CALCULATION_ERROR": ErrorCategory.CALCULATION_ERROR,
    "ANSWER_MISMATCH": ErrorCategory.FACTUAL_ERROR,
    "FACT_FORMAT_ERROR": ErrorCategory.FACTUAL_ERROR,
    "LANGUAGE_SCRIPT_MISMATCH": ErrorCategory.LANGUAGE_ERROR,
    "FORMAT_MISMATCH": ErrorCategory.MISINTERPRETATION,
    "MISSING_FIELD": ErrorCategory.INCOMPLETE_REASONING,
    "TOO_SHORT": ErrorCategory.INCOMPLETE_REASONING,
    "DEGENERATE_REPETITION": ErrorCategory.INCOMPLETE_REASONING,
    "ANSWER_ECHOES_PROMPT": ErrorCategory.MISINTERPRETATION,
    "PLACEHOLDER_TEXT": ErrorCategory.HALLUCINATION,
    "MISSING_CITATION": ErrorCategory.SOURCE_ERROR,
    "CONTRADICTS_SOURCE": ErrorCategory.SOURCE_ERROR,
    "UNSUPPORTED_GENERALISATION": ErrorCategory.OVERGENERALIZATION,
    "CODE_TEST_FAILURE": ErrorCategory.CODE_ERROR,
    "MISSING_ASSUMPTION": ErrorCategory.BAD_ASSUMPTION,
    "MISSING_CONTEXT": ErrorCategory.MISSING_CONTEXT,
    "LOGICAL_INVALIDITY": ErrorCategory.LOGICAL_ERROR,
}


def categories_for_codes(codes: Iterable[str]) -> list[str]:
    out: list[str] = []
    for code in codes:
        cat = CATEGORY_BY_CODE.get(code.upper())
        if cat and cat.value not in out:
            out.append(cat.value)
    return out


#: The stub -> explanation -> corrected -> new problem -> transfer chain, expressed as the
#: dataclass fields that must be non-empty for a case to become training material.
LEARNING_SEQUENCE = (
    "wrong_answer",
    "category",
    "why_wrong",
    "correct_principle",
    "corrected_answer",
    "new_similar_problem",
    "transfer_problem",
)


@dataclass
class ErrorCase:
    case_id: str
    curriculum_node: str
    category: str
    wrong_answer: str
    why_wrong: str
    correct_principle: str
    corrected_answer: str
    new_similar_problem: str
    transfer_problem: str
    subject: str = ""
    language: str = "en"
    difficulty: str = "intermediate"
    classified_by: str = ""
    verification_status: str = "unverified"
    provenance: str = "self_authored"
    source: str = ""
    created_utc: str = ""
    notes: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ErrorCase":
        return cls(
            case_id=str(d.get("case_id", "")),
            curriculum_node=str(d.get("curriculum_node", "")),
            category=str(d.get("category", "")),
            wrong_answer=str(d.get("wrong_answer", "")),
            why_wrong=str(d.get("why_wrong", "")),
            correct_principle=str(d.get("correct_principle", "")),
            corrected_answer=str(d.get("corrected_answer", "")),
            new_similar_problem=str(d.get("new_similar_problem", "")),
            transfer_problem=str(d.get("transfer_problem", "")),
            subject=str(d.get("subject", "")),
            language=str(d.get("language", "en")),
            difficulty=str(d.get("difficulty", "intermediate")),
            classified_by=str(d.get("classified_by", "")),
            verification_status=str(d.get("verification_status", "unverified")),
            provenance=str(d.get("provenance", "self_authored")),
            source=str(d.get("source", "")),
            created_utc=str(d.get("created_utc", "")),
            notes=str(d.get("notes", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}  # type: ignore[attr-defined]

    def validate(self) -> list[str]:
        errs: list[str] = []
        if not self.case_id:
            errs.append("case_id is required")
        if not self.curriculum_node:
            errs.append(f"{self.case_id}: curriculum_node is required")
        if self.category not in {c.value for c in ErrorCategory}:
            errs.append(f"{self.case_id}: category {self.category!r} is not in the taxonomy")
        for name in LEARNING_SEQUENCE:
            if not str(getattr(self, name)).strip():
                errs.append(f"{self.case_id}: '{name}' is empty — the learning sequence must be complete")
        if not self.classified_by.strip():
            errs.append(
                f"{self.case_id}: classified_by is required — an error category must be attributable to a "
                "verifier, critic or reviewer, not asserted"
            )
        if self.provenance not in ("self_authored", "synthetic", "licensed", "public_domain", "uncertain"):
            errs.append(f"{self.case_id}: provenance {self.provenance!r} not recognised")
        if not self.source.strip():
            errs.append(f"{self.case_id}: source is required (which evaluation run produced this mistake)")
        return errs

    def learning_sequence(self) -> list[tuple[str, str]]:
        return [
            ("wrong answer", self.wrong_answer),
            ("error classification", self.category),
            ("why it is wrong", self.why_wrong),
            ("correct principle", self.correct_principle),
            ("corrected answer", self.corrected_answer),
            ("new similar problem", self.new_similar_problem),
            ("transfer problem", self.transfer_problem),
        ]

    def render(self) -> str:
        lines = [f"[{self.case_id}] {self.category} on {self.curriculum_node}"]
        for label, value in self.learning_sequence():
            lines.append(f"  {label}: {value}")
        return "\n".join(lines)

    # ---- conversion to training material ------------------------------------------------
    def to_training_records(self, *, domain: str = "meta") -> list[TrainingRecord]:
        """Emit the three loop stages this case can teach: diagnosis, correction and transfer."""
        errs = self.validate()
        if errs:
            raise ValueError("cannot convert an invalid error case into training records:\n  - " + "\n  - ".join(errs))
        common: dict[str, Any] = {
            "curriculum_node": self.curriculum_node,
            "domain": domain,
            "subject": self.subject or "reasoning",
            "grade_level": None,
            "language": self.language,
            "difficulty": self.difficulty,
            "provenance": self.provenance,
            "license_status": "cleared" if self.provenance in ("self_authored", "public_domain", "licensed") else "unknown_quarantine",
            "source": self.source,
            "teacher_models": [self.classified_by] if self.classified_by else [],
            "error_tags": [self.category],
        }
        return [
            TrainingRecord(
                record_id=f"{self.case_id}.diag",
                kind="error_diagnosis",
                task=(
                    f"A learner answered the following incorrectly on '{self.curriculum_node}'.\n"
                    f"Question context: {self.correct_principle}\n"
                    f"Their answer: {self.wrong_answer}\n"
                    "Identify the error category and explain what went wrong."
                ),
                response=f"{self.category}: {self.why_wrong}",
                rationale=self.correct_principle,
                rationale_style="concise",
                verification_status=self.verification_status,
                notes="error-learning chain: diagnosis step",
                **common,
            ),
            TrainingRecord(
                record_id=f"{self.case_id}.corr",
                kind="correction",
                task=f"Correct this answer on '{self.curriculum_node}': {self.wrong_answer}",
                response=self.corrected_answer,
                rationale=f"{self.why_wrong} → {self.correct_principle}",
                rationale_style="concise",
                verification_status=self.verification_status,
                notes="error-learning chain: correction step",
                **common,
            ),
            TrainingRecord(
                record_id=f"{self.case_id}.transfer",
                kind="transfer",
                task=self.transfer_problem,
                response=self.corrected_answer,
                rationale=f"Same principle ({self.correct_principle}) applied in a different context.",
                rationale_style="concise",
                verification_status=self.verification_status,
                notes="error-learning chain: transfer step",
                **common,
            ),
        ]


def load_cases(path: Path | str = DEFAULT_CASES) -> list[ErrorCase]:
    p = resolve(path)
    if not p.exists():
        return []
    out: list[ErrorCase] = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(ErrorCase.from_dict(json.loads(line)))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{rel(p)}:{i}: not valid JSON ({exc})") from exc
    return out


def cases_from_evaluation(
    results_path: Path | str,
    *,
    verified_by: str = "evaluation-suite",
    source: str = "",
) -> list[dict[str, Any]]:
    """Turn failed evaluation cases into *drafts* of error cases (human/reviewer fills the prose).

    This deliberately does not fabricate `why_wrong` or a corrected answer: it produces the skeleton
    with the wrong answer, the deterministic error codes and the taxonomy mapping, so a reviewer-
    or teacher-model step has to supply the teaching content. A skeleton that is never filled in stays
    invalid and cannot become training data.
    """
    p = resolve(results_path)
    if not p.exists():
        raise FileNotFoundError(f"suite results not found: {p}")
    doc = json.loads(p.read_text(encoding="utf-8"))
    drafts: list[dict[str, Any]] = []
    for case in doc.get("cases", []):
        if case.get("status") != "fail":
            continue
        codes = case.get("error_codes") or []
        drafts.append(
            {
                "case_id": f"err.{case.get('case_id', 'unknown')}",
                "curriculum_node": case.get("curriculum_node", ""),
                "category": (categories_for_codes(codes) or ["UNCURATED"])[0],
                "wrong_answer": case.get("answer", ""),
                "why_wrong": "",
                "correct_principle": "",
                "corrected_answer": "",
                "new_similar_problem": "",
                "transfer_problem": "",
                "language": case.get("language", "en"),
                "difficulty": case.get("difficulty", "intermediate"),
                "classified_by": verified_by,
                "verification_status": "unverified",
                "provenance": "self_authored",
                "source": source or rel(p),
                "notes": "SKELETON: teaching prose missing — review before use; invalid until filled",
                "error_codes": codes,
            }
        )
    return drafts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate error cases and emit error-learning records.")
    ap.add_argument("--cases", default=DEFAULT_CASES)
    ap.add_argument("--out-records", help="write error-learning training records here (JSONL)")
    ap.add_argument("--from-suite", help="build skeleton cases from a failed evaluation suite result")
    ap.add_argument("--skeletons-out", help="where to write skeleton cases")
    args = ap.parse_args(argv)

    if args.from_suite:
        drafts = cases_from_evaluation(args.from_suite)
        print(f"{len(drafts)} failed case(s) converted to skeleton error cases")
        if args.skeletons_out:
            p = resolve(args.skeletons_out)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in drafts) + ("\n" if drafts else ""), encoding="utf-8")
            print(f"written: {rel(p)}")
        return 0

    cases = load_cases(args.cases)
    errs = [e for case in cases for e in case.validate()]
    print(f"cases: {len(cases)} | problems: {len(errs)}")
    for e in errs[:10]:
        print(f"  - {e}")
    if args.out_records and cases and not errs:
        records = [r for case in cases for r in case.to_training_records()]
        print(f"records: {rel(write_records(resolve(args.out_records), records))}")
    return 0 if not errs else 1


if __name__ == "__main__":
    raise SystemExit(main())
