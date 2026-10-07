"""Critics: cheap, deterministic checks that run on every draft before consensus.

A critic is not a judge of quality. It is a filter for the failure modes that are mechanically
detectable, so that model-based review (when it happens) is spent on questions no script can answer:

* the draft is empty, a stub, or still contains generator placeholders;
* the answer is copy-pasted from the prompt (no work was done);
* the declared language does not match the script actually used (a "Hindi" lesson written in Latin
  script, or an "English" one written in Devanagari);
* the draft repeats itself (a classic degenerate generation);
* arithmetic claims inside the draft are *false* — checked by execution, not by opinion;
* required output fields are missing.

Critic verdicts are `ok` / `warn` / `reject`. `reject` removes a draft from consensus; `warn` keeps
it but records the finding, because a warning is evidence a reviewer should see rather than a reason
to silently discard material.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from teachers.providers import TeacherDraft
from teachers.verify import check_arithmetic, normalise

PLACEHOLDERS = ("todo", "tbd", "lorem ipsum", "{{", "}}", "<insert", "xxx", "placeholder")

# codes are strings; development/errors.py maps them onto the error taxonomy
EMPTY_DRAFT = "EMPTY_DRAFT"
MISSING_FIELD = "MISSING_FIELD"
PLACEHOLDER_TEXT = "PLACEHOLDER_TEXT"
ANSWER_ECHOES_PROMPT = "ANSWER_ECHOES_PROMPT"
LANGUAGE_SCRIPT_MISMATCH = "LANGUAGE_SCRIPT_MISMATCH"
DEGENERATE_REPETITION = "DEGENERATE_REPETITION"
CALCULATION_ERROR = "CALCULATION_ERROR"
TOO_SHORT = "TOO_SHORT"


@dataclass
class Critique:
    critic: str
    verdict: str  # ok | warn | reject
    detail: str
    codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"critic": self.critic, "verdict": self.verdict, "detail": self.detail, "codes": list(self.codes)}


def script_profile(text: str) -> dict[str, float]:
    """Fraction of letters in each script family. Cheap, dependency-free, good enough to catch swaps."""
    devanagari = len(re.findall(r"[\u0900-\u097F]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    total = devanagari + latin
    if total == 0:
        return {"devanagari": 0.0, "latin": 0.0}
    return {"devanagari": round(devanagari / total, 3), "latin": round(latin / total, 3)}


def _expected_script(language: str) -> str | None:
    if language in ("hi", "hinc-deva", "sa", "mr"):
        return "devanagari"
    if language in ("en", "hinc-latn"):
        return "latin"
    return None  # multi / regional languages: not mechanically checkable here


def critique_draft(
    draft: TeacherDraft,
    *,
    language: str,
    required_fields: Iterable[str] = ("task", "response", "answer"),
    min_chars: int = 12,
    prompt_text: str = "",
) -> list[Critique]:
    out: list[Critique] = []
    payload: dict[str, Any] = draft.payload or {}
    response = str(payload.get("response", ""))
    task = str(payload.get("task", ""))
    answer = str(payload.get("answer", ""))

    missing = [f for f in required_fields if not str(payload.get(f, "")).strip()]
    out.append(
        Critique(
            "field_completeness",
            "ok" if not missing else "reject",
            "all required fields present" if not missing else f"missing fields: {missing}",
            [] if not missing else [MISSING_FIELD],
        )
    )

    body = f"{task} {response} {answer}".strip()
    if len(body) < min_chars:
        out.append(Critique("substance", "reject", f"draft body is {len(body)} chars (<{min_chars})", [TOO_SHORT]))

    low = body.casefold()
    found = [p for p in PLACEHOLDERS if p in low]
    if found:
        out.append(Critique("placeholder_scan", "reject", f"placeholder text present: {found}", [PLACEHOLDER_TEXT]))

    if prompt_text and answer and normalise(answer) and normalise(answer) in normalise(prompt_text):
        out.append(
            Critique(
                "prompt_echo",
                "warn",
                "the draft's answer appears verbatim in the prompt: check that work was actually done",
                [ANSWER_ECHOES_PROMPT],
            )
        )

    expected = _expected_script(language)
    if expected and payload.get("template") is not True:
        profile = script_profile(response or task)
        if profile[expected] < 0.5:
            out.append(
                Critique(
                    "language_script",
                    "warn",
                    f"declared language {language!r} implies {expected} script but the draft is "
                    f"{profile} — a script swap or a mislabelled language",
                    [LANGUAGE_SCRIPT_MISMATCH],
                )
            )

    sentences = [s for s in re.split(r"[।.!?\n]+", response) if len(s.strip()) > 8]
    if sentences:
        counts = Counter(s.strip() for s in sentences)
        top, n = counts.most_common(1)[0]
        if n >= 3:
            out.append(
                Critique(
                    "degeneracy",
                    "warn",
                    f"the same sentence appears {n} times: likely degenerate generation",
                    [DEGENERATE_REPETITION],
                )
            )

    arith = check_arithmetic(response)
    if arith.status == "fail":
        out.append(Critique("arithmetic", "reject", arith.detail, [CALCULATION_ERROR]))
    elif arith.status == "pass":
        out.append(Critique("arithmetic", "ok", arith.detail, []))

    if not out:
        out.append(Critique("baseline", "ok", "no mechanical problem found", []))
    return out


def worst_verdict(critiques: Iterable[Critique]) -> str:
    order = {"ok": 0, "warn": 1, "reject": 2}
    worst = "ok"
    for c in critiques:
        if order[c.verdict] > order[worst]:
            worst = c.verdict
    return worst


def summarise(critiques: Iterable[Critique]) -> dict[str, Any]:
    critiques = list(critiques)
    by_verdict: dict[str, int] = {}
    codes: dict[str, int] = {}
    by_critic: dict[str, int] = {}
    for c in critiques:
        by_verdict[c.verdict] = by_verdict.get(c.verdict, 0) + 1
        by_critic[c.critic] = by_critic.get(c.critic, 0) + 1
        for code in c.codes:
            codes[code] = codes.get(code, 0) + 1
    return {
        "findings": len(critiques),
        "by_verdict": dict(sorted(by_verdict.items())),
        "by_critic": dict(sorted(by_critic.items())),
        "codes": dict(sorted(codes.items())),
    }
