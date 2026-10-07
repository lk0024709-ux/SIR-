"""developmental program: generations, promotion gates, regression, experiments, error learning.

Encodes *when SIR is allowed to claim progress*: generation contracts, promotion gates that fail
closed, per-axis regression protection, reproducibility records, the error taxonomy and the
brainstorming rubric. Deliberately independent of the model implementation.

Exports are lazy (PEP 562) so that `python -m <package>.<module>` does not import every sibling module:
a CLI run must not pay for, or be affected by, unrelated imports.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY: dict[str, str] = {
    "BrainstormAttempt": "development.brainstorming",
    "BrainstormScore": "development.brainstorming",
    "CATEGORY_BY_CODE": "development.errors",
    "ErrorCase": "development.errors",
    "ErrorCategory": "development.errors",
    "ExperimentRecord": "development.experiments",
    "GenerationContract": "development.generations",
    "PromotionDecision": "development.promotion",
    "RegressionReport": "development.regression",
    "categories_for_codes": "development.errors",
    "check_training_readiness": "development.readiness",
    "compare_runs": "development.regression",
    "evaluate_promotion": "development.promotion",
    "from_train_log": "development.experiments",
    "load_contracts": "development.generations",
    "load_records": "development.experiments",
    "score_attempt": "development.brainstorming",
    "validate_against_program": "development.generations",
    "write_record": "development.experiments",
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
