"""Multi-teacher consensus: never quietly trust one model.

Two teachers agreeing is weak evidence. Two teachers disagreeing is strong evidence that *something*
is wrong — and the correct response is to say so and try to resolve it deterministically, not to pick
whichever answer reads better.

The decision procedure, in order:

1. **No usable drafts** → `INSUFFICIENT_EVIDENCE`. Nothing is produced.
2. **One draft only** (or everyone from the same model id) → `INSUFFICIENT_EVIDENCE` unless
   `allow_single_teacher` is explicitly set, which the default config forbids.
3. **All drafts agree** → `AGREED`. The canonical answer is recorded with every teacher named.
4. **Disagreement** → each distinct answer is checked with a deterministic verifier
   (`teachers.verify`, or a caller-supplied callable):
   * exactly one answer verifies → `RESOLVED_BY_VERIFICATION`;
   * several verify, or none do → `DISAGREEMENT` (escalate to a human/verifier model; do not choose);
   * every answer fails verification → `REJECTED`, with the error codes attached.
5. **Mode is `manual`/`none`** → disagreement stays unresolved: `DISAGREEMENT`.

The outcome object always records how many drafts came from how many distinct models, the answer
groups, and the verification evidence, because "two of three teachers agreed" is the kind of claim
that must be re-checkable later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable

from teachers.providers import TeacherDraft
from teachers.verify import CheckResult, CheckStatus, VerificationMode, normalise, verify_answer


class ConsensusStatus(str, Enum):
    AGREED = "AGREED"
    RESOLVED_BY_VERIFICATION = "RESOLVED_BY_VERIFICATION"
    DISAGREEMENT = "DISAGREEMENT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    REJECTED = "REJECTED"


@dataclass
class ConsensusOutcome:
    request_id: str
    status: ConsensusStatus
    canonical_answer: str | None
    teachers: list[str] = field(default_factory=list)
    groups: dict[str, list[str]] = field(default_factory=dict)
    verification: dict[str, dict[str, Any]] = field(default_factory=dict)
    error_codes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.status in (ConsensusStatus.AGREED, ConsensusStatus.RESOLVED_BY_VERIFICATION)

    def verification_status_for_record(self) -> str:
        return {
            ConsensusStatus.AGREED: "multi_teacher_agreement",
            ConsensusStatus.RESOLVED_BY_VERIFICATION: "deterministically_verified",
            ConsensusStatus.DISAGREEMENT: "critic_reviewed",
            ConsensusStatus.INSUFFICIENT_EVIDENCE: "single_teacher",
            ConsensusStatus.REJECTED: "rejected",
        }[self.status]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "canonical_answer": self.canonical_answer,
            "teachers": list(self.teachers),
            "groups": {k: list(v) for k, v in self.groups.items()},
            "verification": self.verification,
            "error_codes": list(self.error_codes),
            "notes": list(self.notes),
        }


def _distinct_models(drafts: Iterable[TeacherDraft]) -> list[str]:
    return sorted({d.teacher.model_id for d in drafts if d.teacher.model_id})


def reach_consensus(
    drafts: Iterable[TeacherDraft],
    *,
    request_id: str = "",
    mode: str | VerificationMode = VerificationMode.EXACT,
    expected: str | None = None,
    min_teachers: int = 2,
    allow_single_teacher: bool = False,
    verifier: Callable[[str], CheckResult] | None = None,
) -> ConsensusOutcome:
    drafts = list(drafts)
    models = _distinct_models(drafts)
    outcome = ConsensusOutcome(request_id=request_id or (drafts[0].request_id if drafts else ""), status=ConsensusStatus.INSUFFICIENT_EVIDENCE, canonical_answer=None)

    if not drafts:
        outcome.notes.append("no drafts were produced; nothing is emitted (no fabrication fallback)")
        return outcome

    outcome.teachers = models
    outcome.groups = {}
    for d in drafts:
        key = normalise(d.answer)
        outcome.groups.setdefault(key, []).append(d.teacher.model_id or "(unnamed)")

    if len(models) < min_teachers and len(drafts) < min_teachers:
        if not allow_single_teacher:
            outcome.notes.append(
                f"{len(drafts)} draft(s) from {len(models)} model(s) < min_teachers={min_teachers}; "
                "single-teacher material is not treated as consensus evidence"
            )
            return outcome
        outcome.notes.append("single-teacher material accepted only because allow_single_teacher is set")

    if len(outcome.groups) == 1:
        answer = next(iter(outcome.groups))
        # canonical answer is the raw text of the first draft in the group, not the normalised key
        raw = next(d.answer for d in drafts if normalise(d.answer) == answer)
        outcome.status = ConsensusStatus.AGREED
        outcome.canonical_answer = raw
        return outcome

    # ---- disagreement: try deterministic resolution ---------------------------------
    mode_enum = VerificationMode(mode) if not isinstance(mode, VerificationMode) else mode
    deterministic = mode_enum not in (VerificationMode.MANUAL, VerificationMode.NONE) or verifier is not None
    if not deterministic:
        outcome.status = ConsensusStatus.DISAGREEMENT
        outcome.notes.append(
            f"teachers disagree across {len(outcome.groups)} answer group(s) and the declared mode "
            f"{mode_enum.value!r} cannot resolve them mechanically; flagged for review, not silently chosen"
        )
        return outcome

    passed: list[tuple[str, str]] = []  # (group_key, raw_answer)
    failed_codes: list[str] = []
    for group_key, teachers in outcome.groups.items():
        draft = next((d for d in drafts if normalise(d.answer) == group_key), drafts[0])
        raw = draft.answer
        # claim-checking modes must inspect the text that carries the claims, not the short answer,
        # otherwise "17% of 240 = 40.8" could never be verified
        if mode_enum is VerificationMode.ARITHMETIC_CLAIMS:
            text = str((draft.payload or {}).get("response") or raw)
        else:
            text = raw
        result = verifier(text) if verifier is not None else verify_answer(text, expected, mode_enum)
        outcome.verification[group_key] = {**result.to_dict(), "teachers": teachers}
        if result.status == CheckStatus.PASS.value:
            passed.append((group_key, raw))
        elif result.status == CheckStatus.FAIL.value:
            failed_codes.extend(result.error_codes)
            outcome.error_codes.extend(result.error_codes)
        else:
            outcome.error_codes.extend(result.error_codes)

    if len(passed) == 1:
        outcome.status = ConsensusStatus.RESOLVED_BY_VERIFICATION
        outcome.canonical_answer = passed[0][1]
        outcome.notes.append(
            f"{len(outcome.groups)} answers disagreed; deterministic verification selected exactly one "
            f"({passed[0][0][:60]!r}) — the losing answers are kept in the outcome for audit"
        )
        return outcome

    if not passed:
        outcome.status = ConsensusStatus.REJECTED
        outcome.notes.append(
            "no teacher answer passed verification; the item is rejected rather than voting on which "
            "wrong answer is least wrong"
        )
        outcome.error_codes = sorted(set(outcome.error_codes or failed_codes))
        return outcome

    outcome.status = ConsensusStatus.DISAGREEMENT
    outcome.notes.append(
        f"{len(passed)} answers passed verification — verification cannot choose between them; "
        "escalate to a human/verifier reviewer"
    )
    return outcome
