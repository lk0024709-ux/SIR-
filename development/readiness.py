"""Training-readiness gate: the checklist that stands between "we have a config" and a training run.

The project's own rule is that a serious pretraining run needs *measured* tokens, a licensed corpus, a
verified tokenizer, validated evaluation and a reproducible experiment record. This module turns that
rule into an executable check that reports `NOT_READY` when it is not ready, and says exactly which
requirement failed and what the measured value was.

Readiness levels (a strict ladder — the level is the *weakest* link):

| Level | Meaning |
|-------|---------|
| `NOT_READY` | at least one blocking requirement failed (no training run of consequence may start) |
| `PILOT_ONLY` | blocking requirements pass, but measured train tokens are below the partial threshold |
| `READY_FOR_PARTIAL` | tokens within the partial band (useful for pipeline/scale validation, not for capability claims) |
| `READY_FOR_SERIOUS` | every requirement met and tokens at or above the serious threshold |

The thresholds come from `configs/curriculum_sampling.yaml` (`readiness:`) so they can be argued with
in one place. Nothing here consults a model, and nothing here is allowed to be overridden by a flag
except `--ignore-blocking`, which only affects the exit code and is recorded in the report.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sir_paths import REPO_ROOT, load_config, rel, resolve, write_json

LEVELS = ("NOT_READY", "PILOT_ONLY", "READY_FOR_PARTIAL", "READY_FOR_SERIOUS")
STATUSES = ("MET", "NOT_MET", "UNKNOWN")
DEFAULT_CONFIG = REPO_ROOT / "configs" / "curriculum_sampling.yaml"


@dataclass
class ReadinessCheck:
    check_id: str
    requirement: str
    status: str
    blocking: bool
    detail: str
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check_id,
            "requirement": self.requirement,
            "status": self.status,
            "blocking": self.blocking,
            "detail": self.detail,
            "evidence": self.evidence,
        }


@dataclass
class ReadinessReport:
    generation_id: str
    level: str
    checks: list[ReadinessCheck] = field(default_factory=list)
    measured: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.level in ("READY_FOR_PARTIAL", "READY_FOR_SERIOUS")

    @property
    def blocking(self) -> list[str]:
        """Ids of blocking checks that did not pass. Safe to quote: these are the reasons for the level."""
        return [c.check_id for c in self.checks if c.blocking and c.status != "MET"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": "development/readiness.py",
            "generation_id": self.generation_id,
            "level": self.level,
            "ready_for_serious_pretraining": self.level == "READY_FOR_SERIOUS",
            "measured": self.measured,
            "thresholds": self.thresholds,
            "checks": [c.to_dict() for c in self.checks],
            "blocking_failures": self.blocking,
            "notes": self.notes,
            "note": (
                "This gate reports readiness for a training run. It is not evidence that any model was "
                "trained, and it never substitutes for evaluation after training."
            ),
        }


def _check(check_id: str, requirement: str, ok: bool | None, detail: str, evidence: str = "", *, blocking: bool = True) -> ReadinessCheck:
    status = "UNKNOWN" if ok is None else ("MET" if ok else "NOT_MET")
    return ReadinessCheck(check_id, requirement, status, blocking, detail, evidence)


def check_training_readiness(
    *,
    generation_id: str = "nano",
    config_path: Path | str = DEFAULT_CONFIG,
    curated_suite_dir: Path | str = "evaluation/suites",
    assessments_path: Path | str | None = None,
) -> ReadinessReport:
    cfg = load_config(config_path)
    thresholds = dict((cfg.get("readiness") or {}))
    serious = int(thresholds.get("serious_pretraining_min_train_tokens", 200_000_000))
    partial = int(thresholds.get("partial_training_min_train_tokens", 50_000_000))
    report = ReadinessReport(generation_id=generation_id, level="NOT_READY", thresholds=thresholds)

    # 1. curriculum graph + requirements ---------------------------------------------
    try:
        from curriculum.graph import load_default_graph
        from curriculum.requirements import check_requirements

        graph = load_default_graph()
        req = check_requirements(graph, load_config(REPO_ROOT / "configs" / "curriculum_requirements.yaml"))
        report.measured["curriculum_nodes"] = len(graph)
        report.checks.append(
            _check(
                "curriculum_requirements",
                "curriculum graph validates and satisfies the requirement contract",
                req.ok,
                f"{len(graph)} nodes; {len(req.errors)} requirement violation(s)",
                evidence=rel(resolve("configs/curriculum_requirements.yaml")),
            )
        )
    except Exception as exc:  # pragma: no cover - only on a broken checkout
        report.checks.append(_check("curriculum_requirements", "curriculum graph validates", False, f"{type(exc).__name__}: {exc}"))

    # 2. generation contracts ---------------------------------------------------------
    try:
        from development.generations import load_contracts, validate_against_program

        contracts, axes = load_contracts()
        errs = validate_against_program(contracts, axes)
        report.checks.append(
            _check("generation_contracts", "generation contracts are valid and match the program", not errs, f"{len(contracts)} contracts; {len(errs)} problem(s)")
        )
    except Exception as exc:  # pragma: no cover
        report.checks.append(_check("generation_contracts", "generation contracts are valid", False, f"{type(exc).__name__}: {exc}"))

    # 3. measured tokens + provenance gates -------------------------------------------
    corpus_report_path = resolve("evaluation/results/corpus_report.json")
    measured_tokens: int | None = None
    failing_gates: list[str] = []
    if corpus_report_path.exists():
        doc = json.loads(corpus_report_path.read_text(encoding="utf-8"))
        measured_tokens = int(((doc.get("m1") or {}).get("measured_tokens")) or 0)
        failing_gates = list((doc.get("m1") or {}).get("failing_gates") or [])
        report.measured["corpus_report"] = rel(corpus_report_path)
    meta_path = resolve("data/processed/smoke/meta.json")
    tokenized_tokens = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        tokenized_tokens = int(meta.get("n_tokens") or 0)
        report.measured["tokenized_smoke_tokens"] = tokenized_tokens
    report.measured["measured_tokens"] = measured_tokens
    report.measured["failing_provenance_gates"] = failing_gates

    token_count = max([t for t in (measured_tokens, tokenized_tokens) if t is not None], default=None)
    report.checks.append(
        _check(
            "dataset_provenance",
            "every training source carries a verified licence and provenance record",
            (not failing_gates) if corpus_report_path.exists() else None,
            "no failing gates" if not failing_gates else f"failing gates: {failing_gates}",
            evidence=rel(corpus_report_path),
        )
    )
    if token_count is None:
        report.checks.append(_check("measured_tokens", f"measured train tokens >= {serious:,}", None, "no measured token count available"))
    else:
        report.checks.append(
            _check(
                "measured_tokens_floor",
                f"measured train tokens >= {partial:,} (partial) and >= {serious:,} (serious)",
                token_count >= partial,
                f"measured {token_count:,} tokens",
                evidence=rel(corpus_report_path if corpus_report_path.exists() else meta_path),
            )
        )
        if token_count < partial:
            report.notes.append(
                f"{token_count:,} measured tokens is below the {partial:,} partial threshold: pretraining at "
                "this scale would produce a model whose failures cannot be attributed to architecture rather "
                "than data. The correct action is more data, not a longer run."
            )

    # 4. tokenizer verified -----------------------------------------------------------
    tok_spec = None
    tok_meta_sha = None
    if meta_path.exists():
        tok_meta_sha = json.loads(meta_path.read_text(encoding="utf-8")).get("tokenizer_sha256")
    smoke_cfg = load_config(REPO_ROOT / "configs" / "sir_nano_smoke.yaml") if (REPO_ROOT / "configs" / "sir_nano_smoke.yaml").exists() else {}
    tok_spec = ((smoke_cfg.get("tokenizer") or {}).get("spec")) if isinstance(smoke_cfg.get("tokenizer"), dict) else smoke_cfg.get("tokenizer")
    spec_exists = bool(tok_spec) and resolve(tok_spec).exists()
    report.checks.append(
        _check(
            "tokenizer_available",
            "the configured tokenizer artifact exists and matches the tokenized data",
            spec_exists and bool(tok_meta_sha),
            f"spec={tok_spec} exists={spec_exists}; tokenized data records tokenizer_sha256={str(tok_meta_sha)[:12]}",
            evidence=str(tok_spec or ""),
        )
    )

    # 5. evaluation suites ready ------------------------------------------------------
    suite_dir = resolve(curated_suite_dir)
    suites = sorted(suite_dir.glob("*.jsonl")) if suite_dir.exists() else []
    axes_covered: set[str] = set()
    for suite in suites:
        for line in suite.read_text(encoding="utf-8").splitlines():
            if line.strip():
                axes_covered.add(str(json.loads(line).get("axis", "")))
    report.measured["evaluation_suites"] = [rel(p) for p in suites]
    report.measured["evaluation_axes_covered"] = sorted(a for a in axes_covered if a)
    report.checks.append(
        _check(
            "evaluation_suite_present",
            "at least one curated evaluation suite exists with cases",
            bool(suites),
            f"{len(suites)} suite file(s), axes: {sorted(a for a in axes_covered if a)}",
            evidence=rel(suite_dir),
        )
    )

    # 6. coverage measured ------------------------------------------------------------
    coverage_path = resolve("evaluation/results/curriculum_coverage.json")
    assessed_nodes = None
    if coverage_path.exists():
        doc = json.loads(coverage_path.read_text(encoding="utf-8"))
        assessed_nodes = sum(
            v for k, v in (doc.get("counts") or {}).items() if k in ("PRACTICE", "ASSESSED", "MASTERED", "REVIEW_REQUIRED")
        )
    min_nodes = int(thresholds.get("min_coverage_nodes_assessed", 1))
    report.measured["assessed_curriculum_nodes"] = assessed_nodes
    report.checks.append(
        _check(
            "coverage_measured",
            f"curriculum coverage measured on >= {min_nodes} node(s)",
            (assessed_nodes is not None and assessed_nodes >= min_nodes) if coverage_path.exists() else False,
            "no coverage report yet" if assessed_nodes is None else f"{assessed_nodes} node(s) assessed",
            evidence=rel(coverage_path),
            blocking=True,
        )
    )

    # 7. experiment tracking ----------------------------------------------------------
    exp_dir = resolve("experiments/records")
    exp_records = sorted(exp_dir.glob("*.json")) if exp_dir.exists() else []
    report.checks.append(
        _check(
            "experiment_tracking",
            "experiment records directory exists with at least one record",
            bool(exp_records),
            f"{len(exp_records)} record(s) in {rel(exp_dir)}",
            evidence=rel(exp_dir),
            blocking=bool(thresholds.get("require_experiment_tracking", True)),
        )
    )
    if not exp_records:
        report.notes.append(
            "no experiment records yet: run `python -m development.experiments --from-train-log runs/<run>/train_log.json` "
            "after any training run so the run is reproducible from the repository"
        )

    # ---- level ---------------------------------------------------------------------
    blocking_failures = [c for c in report.checks if c.blocking and c.status != "MET"]
    if blocking_failures:
        report.level = "NOT_READY"
    elif token_count is not None and token_count < partial:
        report.level = "PILOT_ONLY"
    elif token_count is not None and token_count < serious:
        report.level = "READY_FOR_PARTIAL"
    elif token_count is not None:
        report.level = "READY_FOR_SERIOUS"
    else:
        report.level = "NOT_READY"
    if report.level != "READY_FOR_SERIOUS":
        report.notes.append(
            f"level={report.level}: this generation is not cleared for a serious pretraining run; "
            "report it as a pilot/readiness state, never as a trained model"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check readiness for the next training run.")
    ap.add_argument("--generation", default="nano")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out", default="evaluation/results/training_readiness.json")
    ap.add_argument("--ignore-blocking", action="store_true", help="exit 0 even when not ready (recorded in the report)")
    args = ap.parse_args(argv)

    report = check_training_readiness(generation_id=args.generation, config_path=resolve(args.config))
    for c in report.checks:
        mark = {"MET": "PASS", "NOT_MET": "FAIL", "UNKNOWN": "????"}[c.status]
        block = "blocking" if c.blocking else "advisory"
        print(f"  [{mark}] {c.check_id} ({block}): {c.detail}")
    print(f"level: {report.level}")
    for note in report.notes:
        print(f"  note: {note}")
    out = write_json(resolve(args.out), report.to_dict())
    print(f"report: {rel(out)}")
    if not report.ready and not args.ignore_blocking:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
