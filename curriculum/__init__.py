"""curriculum: graph, requirements, coverage, spaced review, sampling.

Import-light on purpose: these modules are used by data generation, evaluation and promotion,
and none of them may pull in torch or a model implementation.

Exports are lazy (PEP 562) so that `python -m <package>.<module>` does not import every sibling module:
a CLI run must not pay for, or be affected by, unrelated imports.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY: dict[str, str] = {
    "CoverageReport": "curriculum.coverage",
    "CoverageState": "curriculum.coverage",
    "CoverageThresholds": "curriculum.coverage",
    "CurriculumGraph": "curriculum.graph",
    "CurriculumNode": "curriculum.schema",
    "CurriculumSampler": "curriculum.sampler",
    "SamplingPolicy": "curriculum.sampler",
    "SpacedReviewScheduler": "curriculum.review",
    "build_report": "curriculum.coverage",
    "check_requirements": "curriculum.requirements",
    "load_default_graph": "curriculum.graph",
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
