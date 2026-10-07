"""teacher / critic / verifier pipeline.

External models (GPT, Claude, Gemini, Grok, Qwen, …) are used here as teachers and evaluators,
never as SIR's runtime. This package contains no API client and no network call: requests are
deterministic functions of the curriculum graph, and teacher output arrives as recorded drafts
that carry their own provenance. See docs/curriculum-data-pipeline.md for the policy.

Exports are lazy (PEP 562) so that `python -m <package>.<module>` does not import every sibling module:
a CLI run must not pay for, or be affected by, unrelated imports.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY: dict[str, str] = {
    "CheckResult": "teachers.verify",
    "CheckStatus": "teachers.verify",
    "ConsensusOutcome": "teachers.consensus",
    "ConsensusStatus": "teachers.consensus",
    "Critique": "teachers.critics",
    "GenerationConfig": "teachers.pipeline",
    "ProviderRegistry": "teachers.providers",
    "RecordedProvider": "teachers.providers",
    "StaticProvider": "teachers.providers",
    "TeacherDraft": "teachers.providers",
    "TeacherPipeline": "teachers.pipeline",
    "TeacherSpec": "teachers.providers",
    "TemplateProvider": "teachers.providers",
    "VerificationMode": "teachers.verify",
    "build_request": "teachers.pipeline",
    "critique_draft": "teachers.critics",
    "reach_consensus": "teachers.consensus",
    "verify_answer": "teachers.verify",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        value = getattr(importlib.import_module(_LAZY[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(_LAZY)
