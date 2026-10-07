"""Deterministic verification — the part of the teacher pipeline that does not need a model.

A language model asked "is 17% of 240 equal to 40.8?" can be confidently wrong. Python cannot. So
whenever a claim is checkable by execution, this module checks it by execution and the model's
opinion is not consulted.

What is implemented (all offline, all deterministic):

* `check_arithmetic` — extracts arithmetic claims of the form `a OP b = c`, `p% of n = m`, and
  fraction/percentage comparisons, then evaluates them with exact rational arithmetic (`Fraction`),
  so `1/3 + 1/6 = 1/2` is verified as exactly true rather than approximately true.
* `check_numeric` / `check_exact` / `check_set` — comparison modes for short answers.
* `verify_answer` — dispatches on a declared verification mode and returns explicit `PASS`,
  `FAIL` or `UNVERIFIABLE`.

The `UNVERIFIABLE` status matters as much as the other two: a checker that silently returns PASS
when it cannot parse the answer is how unverified content enters a dataset. Anything this module
cannot check is handed back with `REQUIRES_HUMAN_REVIEW`.

Deliberately absent: any attempt to judge prose quality, factual accuracy of free text, or reasoning
validity. Those need a reviewer (model or human) and are recorded as such, not faked here.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import Iterable

from training.data.records import DIFFICULTIES  # noqa: F401  (re-exported for pipeline config checks)

NUMBER = r"-?\d+(?:\.\d+)?"
FRACTION = rf"(?:{NUMBER}\s*/\s*{NUMBER}|{NUMBER})"
_OP_MAP = {"×": "*", "·": "*", "x": "*", "÷": "/", "−": "-", "^": "**"}

_CLAIM_RE = re.compile(
    rf"(?P<a>{FRACTION})\s*(?P<op>[+\-*/×÷^])\s*(?P<b>{FRACTION})\s*=\s*(?P<c>{FRACTION})"
)
_PERCENT_CLAIM_RE = re.compile(
    rf"(?P<p>{NUMBER})\s*%\s*of\s*(?P<n>{NUMBER})\s*(?:=|is)\s*(?P<m>{NUMBER})"
)


class VerificationMode(str, Enum):
    EXACT = "exact"
    NUMERIC = "numeric"
    SET = "set"
    ARITHMETIC_CLAIMS = "arithmetic_claims"
    INTEGRITY_ONLY = "integrity_only"
    MANUAL = "manual"
    NONE = "none"


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNVERIFIABLE = "unverifiable"


# Error codes are strings on purpose: `development/errors.py` owns the taxonomy mapping, and
# teachers/ must not depend on development/ (data generation must not need the promotion layer).
CALCULATION_ERROR = "CALCULATION_ERROR"
ANSWER_MISMATCH = "ANSWER_MISMATCH"
FORMAT_MISMATCH = "FORMAT_MISMATCH"
NO_VERIFIABLE_CLAIM = "NO_VERIFIABLE_CLAIM"
REQUIRES_HUMAN_REVIEW = "REQUIRES_HUMAN_REVIEW"
EMPTY_ANSWER = "EMPTY_ANSWER"


@dataclass
class CheckResult:
    mode: str
    status: str
    detail: str
    error_codes: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == CheckStatus.PASS.value

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "status": self.status,
            "detail": self.detail,
            "error_codes": list(self.error_codes),
            "evidence": self.evidence,
        }


# --------------------------------------------------------------------------------------
# exact rational arithmetic
# --------------------------------------------------------------------------------------
def _to_fraction(text: str) -> Fraction | None:
    text = text.strip().replace(" ", "")
    if "/" in text:
        num, _, den = text.partition("/")
        try:
            d = Fraction(den.replace(",", ""))
        except (ValueError, ZeroDivisionError):
            return None
        if d == 0:
            return None
        try:
            return Fraction(num.replace(",", "")) / d
        except ValueError:
            return None
    try:
        return Fraction(text.replace(",", ""))
    except ValueError:
        return None


def _eval_node(node: ast.AST) -> Fraction | None:
    """Evaluate an already-parsed arithmetic expression, allowing only + - * / ** and unary minus."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            return None
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _eval_node(node.operand)
        if inner is None:
            return None
        return -inner if isinstance(node.op, ast.USub) else inner
    if isinstance(node, ast.BinOp):
        left, right = _eval_node(node.left), _eval_node(node.right)
        if left is None or right is None:
            return None
        try:
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return None if right == 0 else left / right
            if isinstance(node.op, ast.Pow):
                if right.denominator != 1 or abs(int(right)) > 64:
                    return None
                return left ** int(right)
        except (ZeroDivisionError, OverflowError, ValueError):
            return None
    return None


def evaluate_expression(expr: str) -> Fraction | None:
    """Safely evaluate a small arithmetic expression exactly. Returns None when not evaluable."""
    cleaned = expr
    for src, dst in _OP_MAP.items():
        cleaned = cleaned.replace(src, dst)
    cleaned = cleaned.replace(",", "")
    if not re.fullmatch(r"[\d\s.+\-*/()]+", cleaned):
        return None
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError:
        return None
    return _eval_node(tree)


# --------------------------------------------------------------------------------------
# claim extraction and checking
# --------------------------------------------------------------------------------------
@dataclass
class ArithmeticClaim:
    text: str
    left: str
    op: str
    right: str
    claimed: str
    kind: str = "binary"  # binary | percent_of

    def is_true(self) -> bool | None:
        if self.kind == "percent_of":
            p = _to_fraction(self.left)
            n = _to_fraction(self.right)
            claimed = _to_fraction(self.claimed)
            if p is None or n is None or claimed is None:
                return None
            return (p / 100) * n == claimed
        a, b, c = _to_fraction(self.left), _to_fraction(self.right), _to_fraction(self.claimed)
        if a is None or b is None or c is None:
            return None
        got = evaluate_expression(f"({self.left}) {self.op} ({self.right})")
        if got is None:
            return None
        return got == c


def extract_arithmetic_claims(text: str) -> list[ArithmeticClaim]:
    claims: list[ArithmeticClaim] = []
    for m in _PERCENT_CLAIM_RE.finditer(text):
        claims.append(
            ArithmeticClaim(
                text=m.group(0),
                left=m.group("p"),
                op="%of",
                right=m.group("n"),
                claimed=m.group("m"),
                kind="percent_of",
            )
        )
    for m in _CLAIM_RE.finditer(text):
        claims.append(
            ArithmeticClaim(
                text=m.group(0),
                left=m.group("a"),
                op=m.group("op"),
                right=m.group("b"),
                claimed=m.group("c"),
            )
        )
    return claims


def check_arithmetic(text: str) -> CheckResult:
    claims = extract_arithmetic_claims(text)
    if not claims:
        return CheckResult(
            mode=VerificationMode.ARITHMETIC_CLAIMS.value,
            status=CheckStatus.UNVERIFIABLE.value,
            detail="no arithmetic claim of the form 'a OP b = c' or 'p% of n = m' found in the text",
            error_codes=[NO_VERIFIABLE_CLAIM],
        )
    bad: list[str] = []
    checked = 0
    for claim in claims:
        verdict = claim.is_true()
        if verdict is None:
            continue
        checked += 1
        if not verdict:
            bad.append(claim.text)
    if checked == 0:
        return CheckResult(
            mode=VerificationMode.ARITHMETIC_CLAIMS.value,
            status=CheckStatus.UNVERIFIABLE.value,
            detail=f"{len(claims)} claim(s) found but none evaluable (unparsable operand)",
            error_codes=[NO_VERIFIABLE_CLAIM],
            evidence={"claims": [c.text for c in claims]},
        )
    if bad:
        return CheckResult(
            mode=VerificationMode.ARITHMETIC_CLAIMS.value,
            status=CheckStatus.FAIL.value,
            detail=f"{len(bad)} of {checked} arithmetic claim(s) are false: {bad[:5]}",
            error_codes=[CALCULATION_ERROR],
            evidence={"false_claims": bad, "claims_checked": checked, "claims_found": len(claims)},
        )
    return CheckResult(
        mode=VerificationMode.ARITHMETIC_CLAIMS.value,
        status=CheckStatus.PASS.value,
        detail=f"all {checked} evaluable arithmetic claim(s) are exactly true",
        evidence={"claims_checked": checked, "claims_found": len(claims)},
    )


# --------------------------------------------------------------------------------------
# answer comparison modes
# --------------------------------------------------------------------------------------
def normalise(text: str) -> str:
    text = text.strip().casefold()
    text = re.sub(r"[\s\u200b]+", " ", text)
    text = re.sub(r"[।.,;:!?\"'`]+$", "", text)
    return text.strip()


def _first_number(text: str) -> Fraction | None:
    text = text.strip()
    percent = re.fullmatch(rf"({NUMBER})\s*%", text)
    if percent:
        value = _to_fraction(percent.group(1))
        return None if value is None else value / 100
    m = re.search(FRACTION, text)
    if not m:
        return None
    return _to_fraction(m.group(0))


def check_exact(answer: str, expected: str) -> CheckResult:
    if not answer.strip():
        return CheckResult(VerificationMode.EXACT.value, CheckStatus.FAIL.value, "empty answer", [EMPTY_ANSWER])
    ok = normalise(answer) == normalise(expected)
    return CheckResult(
        VerificationMode.EXACT.value,
        CheckStatus.PASS.value if ok else CheckStatus.FAIL.value,
        "exact match after normalisation" if ok else f"expected {expected!r}, got {answer!r}",
        [] if ok else [ANSWER_MISMATCH],
        {"expected": expected, "answer": answer},
    )


def check_numeric(answer: str, expected: str, tolerance: float = 0.0) -> CheckResult:
    got, want = _first_number(answer), _first_number(expected)
    if not answer.strip():
        return CheckResult(VerificationMode.NUMERIC.value, CheckStatus.FAIL.value, "empty answer", [EMPTY_ANSWER])
    if got is None or want is None:
        return CheckResult(
            VerificationMode.NUMERIC.value,
            CheckStatus.UNVERIFIABLE.value,
            "could not parse a number to compare (numeric mode expects a numeric answer)",
            [FORMAT_MISMATCH],
            {"expected": expected, "answer": answer},
        )
    diff = abs(got - want)
    allowed = Fraction(str(tolerance)) if tolerance else Fraction(0)
    ok = diff <= allowed
    return CheckResult(
        VerificationMode.NUMERIC.value,
        CheckStatus.PASS.value if ok else CheckStatus.FAIL.value,
        f"{float(got)} vs expected {float(want)} (tolerance {tolerance})",
        [] if ok else [ANSWER_MISMATCH],
        {"expected": float(want), "answer": float(got), "abs_error": float(diff)},
    )


def check_set(answer: str, expected: str) -> CheckResult:
    def split(text: str) -> set[str]:
        parts = re.split(r"[,;/\n]| और | and ", text)
        return {normalise(p) for p in parts if normalise(p)}

    if not answer.strip():
        return CheckResult(VerificationMode.SET.value, CheckStatus.FAIL.value, "empty answer", [EMPTY_ANSWER])
    got, want = split(answer), split(expected)
    ok = got == want
    return CheckResult(
        VerificationMode.SET.value,
        CheckStatus.PASS.value if ok else CheckStatus.FAIL.value,
        "sets match" if ok else f"missing={sorted(want - got)}, extra={sorted(got - want)}",
        [] if ok else [ANSWER_MISMATCH],
        {"expected": sorted(want), "answer": sorted(got)},
    )


def verify_answer(
    answer: str,
    expected: str | None = None,
    mode: str | VerificationMode = VerificationMode.EXACT,
    *,
    tolerance: float = 0.0,
) -> CheckResult:
    mode = VerificationMode(mode) if not isinstance(mode, VerificationMode) else mode
    if mode is VerificationMode.EXACT:
        if expected is None:
            return CheckResult(mode.value, CheckStatus.UNVERIFIABLE.value, "exact mode needs an expected answer", [REQUIRES_HUMAN_REVIEW])
        return check_exact(answer, expected)
    if mode is VerificationMode.NUMERIC:
        if expected is None:
            return CheckResult(mode.value, CheckStatus.UNVERIFIABLE.value, "numeric mode needs an expected answer", [REQUIRES_HUMAN_REVIEW])
        return check_numeric(answer, expected, tolerance)
    if mode is VerificationMode.SET:
        if expected is None:
            return CheckResult(mode.value, CheckStatus.UNVERIFIABLE.value, "set mode needs an expected answer", [REQUIRES_HUMAN_REVIEW])
        return check_set(answer, expected)
    if mode is VerificationMode.ARITHMETIC_CLAIMS:
        return check_arithmetic(answer)
    if mode is VerificationMode.INTEGRITY_ONLY:
        ok = bool(answer.strip())
        return CheckResult(
            mode.value,
            CheckStatus.PASS.value if ok else CheckStatus.FAIL.value,
            "non-empty answer only; content was NOT verified" if ok else "empty answer",
            [] if ok else [EMPTY_ANSWER],
            {"note": "integrity_only checks presence, not correctness"},
        )
    return CheckResult(
        mode.value,
        CheckStatus.UNVERIFIABLE.value,
        "mode requires a human or model reviewer; nothing was verified automatically",
        [REQUIRES_HUMAN_REVIEW],
    )


def summarise(results: Iterable[CheckResult]) -> dict:
    results = list(results)
    out = {s.value: 0 for s in CheckStatus}
    for r in results:
        out[r.status] = out.get(r.status, 0) + 1
    return {
        "checks": len(results),
        "pass": out.get(CheckStatus.PASS.value, 0),
        "fail": out.get(CheckStatus.FAIL.value, 0),
        "unverifiable": out.get(CheckStatus.UNVERIFIABLE.value, 0),
        "error_codes": sorted({c for r in results for c in r.error_codes}),
    }
