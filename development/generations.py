"""Generation contracts: the machine-readable definition of each developmental stage.

A generation (SIR-Nano … SIR-Expert) is a *stage of development with evidence requirements*, not a
parameter count. This module loads the contracts, checks them against the program config
(`configs/sir_human_development.yaml`, which owns the stable ids and order), and enforces the honesty
rules that stop a contract from describing an achievement that has not happened:

* `TRAINED_VERIFIED` requires `verification_status: VERIFIED` **and** evidence paths;
* any status that claims training requires evidence paths (a run log at minimum);
* `PLANNED` / `ARCHITECTURE_ONLY` must state `known_limitations`, so unreleased stages are explicit
  about what does not exist;
* ids, names and order must match the program config — a rename is a breaking change, not a detail.

The vocabulary is deliberately small and pessimistic: it is easy to move *up* only by adding evidence.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sir_paths import REPO_ROOT, load_config, rel, resolve, write_json

DEFAULT_CONTRACTS = REPO_ROOT / "configs" / "generation_contracts.yaml"
PROGRAM_CONFIG = REPO_ROOT / "configs" / "sir_human_development.yaml"

TRAINING_STATUSES = (
    "PLANNED",
    "ARCHITECTURE_ONLY",
    "UNTRAINED",
    "PARTIALLY_TRAINED",
    "TRAINED_UNVERIFIED",
    "TRAINED_VERIFIED",
    "DEPRECATED",
)
VERIFICATION_STATUSES = ("UNVERIFIED", "PARTIALLY_VERIFIED", "VERIFIED", "FAILED")
_TRAINED = ("PARTIALLY_TRAINED", "TRAINED_UNVERIFIED", "TRAINED_VERIFIED")
_UNBUILT = ("PLANNED", "ARCHITECTURE_ONLY")


@dataclass
class GenerationContract:
    generation_id: str
    name: str
    training_stage: str
    curriculum_scope: dict[str, Any]
    target_capabilities: list[str]
    required_evaluations: list[str]
    promotion_requirements: dict[str, Any]
    known_limitations: list[str]
    training_status: str
    verification_status: str
    model_config_ref: str | None = None
    target_params: int | None = None
    evidence: list[str] = field(default_factory=list)
    evidence_note: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GenerationContract":
        return cls(
            generation_id=str(d.get("generation_id", "")),
            name=str(d.get("name", "")),
            training_stage=str(d.get("training_stage", "")),
            curriculum_scope=dict(d.get("curriculum_scope") or {}),
            target_capabilities=[str(x) for x in (d.get("target_capabilities") or [])],
            required_evaluations=[str(x) for x in (d.get("required_evaluations") or [])],
            promotion_requirements=dict(d.get("promotion_requirements") or {}),
            known_limitations=[str(x) for x in (d.get("known_limitations") or [])],
            training_status=str(d.get("training_status", "")),
            verification_status=str(d.get("verification_status", "")),
            model_config_ref=d.get("model_config_ref"),
            target_params=d.get("target_params"),
            evidence=[str(x) for x in (d.get("evidence") or [])],
            evidence_note=str(d.get("evidence_note", "") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation_id": self.generation_id,
            "name": self.name,
            "training_stage": self.training_stage,
            "curriculum_scope": self.curriculum_scope,
            "target_capabilities": list(self.target_capabilities),
            "required_evaluations": list(self.required_evaluations),
            "promotion_requirements": self.promotion_requirements,
            "known_limitations": list(self.known_limitations),
            "training_status": self.training_status,
            "verification_status": self.verification_status,
            "model_config_ref": self.model_config_ref,
            "target_params": self.target_params,
            "evidence": list(self.evidence),
            "evidence_note": self.evidence_note,
        }

    # ---- honesty checks ---------------------------------------------------------------
    def validate(self, axes: dict[str, str]) -> list[str]:
        errs: list[str] = []
        if not self.generation_id or not self.name:
            errs.append("generation_id and name are required")
        if self.training_status not in TRAINING_STATUSES:
            errs.append(f"{self.generation_id}: training_status {self.training_status!r} not in {TRAINING_STATUSES}")
        if self.verification_status not in VERIFICATION_STATUSES:
            errs.append(
                f"{self.generation_id}: verification_status {self.verification_status!r} not in {VERIFICATION_STATUSES}"
            )
        if not self.target_capabilities:
            errs.append(f"{self.generation_id}: target_capabilities must be non-empty")
        unknown_axes = [a for a in self.required_evaluations if a not in axes]
        if unknown_axes:
            errs.append(f"{self.generation_id}: required_evaluations names unknown axes {unknown_axes}")
        if not self.required_evaluations:
            errs.append(f"{self.generation_id}: required_evaluations must be non-empty (promotion needs evidence axes)")
        scores = (self.promotion_requirements or {}).get("min_axis_scores") or {}
        missing_threshold = [a for a in self.required_evaluations if a not in scores]
        if missing_threshold:
            errs.append(
                f"{self.generation_id}: promotion_requirements.min_axis_scores has no threshold for {missing_threshold}"
            )
        for axis, value in scores.items():
            if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                errs.append(f"{self.generation_id}: min_axis_scores[{axis}]={value!r} must be within [0, 1]")
        if int((self.promotion_requirements or {}).get("min_cases_per_axis", 0)) < 1:
            errs.append(f"{self.generation_id}: promotion_requirements.min_cases_per_axis must be >= 1")

        if self.training_status in _TRAINED and not self.evidence:
            errs.append(
                f"{self.generation_id}: training_status {self.training_status} requires evidence paths — a "
                "status that claims training without a single artifact is a claim, not a record"
            )
        if self.training_status in _UNBUILT and not self.known_limitations:
            errs.append(f"{self.generation_id}: {self.training_status} generations must state known_limitations")
        if self.training_status == "TRAINED_VERIFIED" and self.verification_status != "VERIFIED":
            errs.append(
                f"{self.generation_id}: TRAINED_VERIFIED requires verification_status VERIFIED "
                f"(got {self.verification_status})"
            )
        if self.verification_status in ("VERIFIED", "PARTIALLY_VERIFIED") and not self.evidence:
            errs.append(f"{self.generation_id}: verification_status {self.verification_status} requires evidence paths")
        if self.verification_status == "FAILED" and self.training_status in _TRAINED:
            errs.append(
                f"{self.generation_id}: content says training happened but verification FAILED; report it as "
                "UNTRAINED/PLANNED with the failure recorded, or fix the evidence — do not relabel it"
            )
        return errs


def load_contracts(path: Path | str = DEFAULT_CONTRACTS) -> tuple[dict[str, GenerationContract], dict[str, str]]:
    cfg = load_config(path)
    axes = {str(k): str(v) for k, v in (cfg.get("evaluation_axes") or {}).items()}
    contracts = {c["generation_id"]: GenerationContract.from_dict(c) for c in cfg.get("generations") or []}
    return contracts, axes


def validate_against_program(
    contracts: dict[str, GenerationContract], axes: dict[str, str], program_path: Path | str = PROGRAM_CONFIG
) -> list[str]:
    """Contract ids/names/order must equal the program config, and every contract must be sound."""
    errs: list[str] = []
    program = load_config(program_path)
    declared = [(g["id"], g["name"]) for g in program.get("generations") or []]
    mine = [(c.generation_id, c.name) for c in contracts.values()]
    if [gid for gid, _ in declared] != [gid for gid, _ in mine]:
        errs.append(f"generation ids/order differ from program config: program={declared}, contracts={mine}")
    for (pid, pname), (cid, cname) in zip(declared, mine):
        if pid == cid and pname != cname:
            errs.append(f"{cid}: name {cname!r} does not match program label {pname!r} (labels are stable)")
    for contract in contracts.values():
        errs += contract.validate(axes)
    return errs


def generation_order(contracts: dict[str, GenerationContract]) -> list[str]:
    return [c.generation_id for c in contracts.values()]


def next_generation(contracts: dict[str, GenerationContract], generation_id: str) -> str | None:
    order = generation_order(contracts)
    if generation_id not in order:
        raise KeyError(f"unknown generation {generation_id!r}")
    i = order.index(generation_id)
    return order[i + 1] if i + 1 < len(order) else None


def status_summary(contracts: dict[str, GenerationContract]) -> dict[str, Any]:
    """What is actually true today, counted — safe to quote in a report."""
    by_training: dict[str, int] = {}
    by_verification: dict[str, int] = {}
    trained: list[str] = []
    for c in contracts.values():
        by_training[c.training_status] = by_training.get(c.training_status, 0) + 1
        by_verification[c.verification_status] = by_verification.get(c.verification_status, 0) + 1
        if c.training_status in _TRAINED:
            trained.append(c.generation_id)
    return {
        "generations": len(contracts),
        "by_training_status": dict(sorted(by_training.items())),
        "by_verification_status": dict(sorted(by_verification.items())),
        "generations_with_training_evidence": sorted(trained),
        "order": generation_order(contracts),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate SIR generation contracts.")
    ap.add_argument("--contracts", default=str(DEFAULT_CONTRACTS))
    ap.add_argument("--out", help="write the summary JSON here")
    args = ap.parse_args(argv)

    contracts, axes = load_contracts(resolve(args.contracts))
    errs = validate_against_program(contracts, axes)
    summary = status_summary(contracts)
    print(json.dumps(summary, indent=2))
    if errs:
        print(f"{len(errs)} contract problem(s):")
        for e in errs:
            print(f"  - {e}")
    else:
        print("contracts valid: ids, names and honesty rules all pass")
    if args.out:
        payload = {"tool": "development/generations.py", "summary": summary, "problems": errs}
        print(f"report: {rel(write_json(resolve(args.out), payload))}")
    return 0 if not errs else 1


if __name__ == "__main__":
    raise SystemExit(main())
