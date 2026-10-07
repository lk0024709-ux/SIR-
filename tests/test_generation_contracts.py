"""Generation contracts: the machine-readable promises each SIR generation makes.

The contract file is where "we have a 25M model" could most easily become a lie, so the tests are
written around the honesty rules: a trained status needs an artifact path, a planned status needs
stated limitations, and promotion thresholds must exist for every required axis.
"""

from __future__ import annotations

import copy

from development.generations import (
    GenerationContract,
    TRAINING_STATUSES,
    VERIFICATION_STATUSES,
    load_contracts,
    next_generation,
    status_summary,
    validate_against_program,
)
from sir_paths import REPO_ROOT, load_config

ORDER = ["nano", "lite", "flash_lite", "flash", "pro", "pro_plus", "pro_max", "ultra", "expert"]


def test_committed_contracts_load_validate_and_match_the_program():
    contracts, axes = load_contracts()
    assert list(contracts) == ORDER
    assert [c.name for c in contracts.values()] == [
        "SIR-Nano",
        "SIR-Lite",
        "SIR-Flash Lite",
        "SIR-Flash",
        "SIR-Pro",
        "SIR-Pro+",
        "SIR-Pro Max",
        "SIR-Ultra",
        "SIR-Expert",
    ]
    for contract in contracts.values():
        assert contract.validate(axes) == [], contract.generation_id
    assert validate_against_program(contracts, axes) == []


def test_every_generation_covers_all_eleven_axes_at_the_top():
    contracts, axes = load_contracts()
    assert len(axes) == 11
    assert set(contracts["expert"].required_evaluations) == set(axes)
    # ladders only grow: a later generation may not drop an axis the earlier one required
    previous: set[str] = set()
    for contract in contracts.values():
        current = set(contract.required_evaluations)
        assert previous <= current, f"{contract.generation_id} dropped required axes: {previous - current}"
        previous = current


def test_promotion_thresholds_never_decrease_along_the_ladder():
    contracts, _axes = load_contracts()
    previous: dict[str, float] = {}
    for contract in contracts.values():
        scores = contract.promotion_requirements["min_axis_scores"]
        for axis, value in previous.items():
            if axis in scores:
                assert scores[axis] >= value - 1e-9, f"{contract.generation_id} lowered {axis}"
        previous.update({k: float(v) for k, v in scores.items()})


def test_only_the_smoke_artifact_exists_and_no_generation_claims_training():
    contracts, _axes = load_contracts()
    summary = status_summary(contracts)
    assert summary["generations"] == 9
    assert summary["generations_with_training_evidence"] == []
    assert contracts["nano"].training_status == "UNTRAINED"
    assert contracts["nano"].verification_status == "UNVERIFIED"
    assert contracts["nano"].target_params == 25_000_000
    assert contracts["nano"].evidence, "the smoke-run evidence paths must stay attached to the contract"
    assert "NOT evidence" in contracts["nano"].evidence_note


def test_status_summary_counts_are_measured_from_the_file():
    contracts, _axes = load_contracts()
    summary = status_summary(contracts)
    assert sum(summary["by_training_status"].values()) == 9
    assert summary["by_verification_status"] == {"UNVERIFIED": 9}


def test_next_generation_walks_the_ladder_and_ends():
    contracts, _axes = load_contracts()
    assert next_generation(contracts, "nano") == "lite"
    assert next_generation(contracts, "ultra") == "expert"
    assert next_generation(contracts, "expert") is None
    import pytest

    with pytest.raises(KeyError):
        next_generation(contracts, "nonexistent")


def test_trained_without_evidence_is_rejected():
    contracts, axes = load_contracts()
    c = copy.deepcopy(contracts["nano"])
    c.training_status = "TRAINED_VERIFIED"
    c.verification_status = "VERIFIED"
    c.evidence = []
    errors = c.validate(axes)
    assert any("requires evidence paths" in e for e in errors)


def test_trained_verified_requires_verified_status():
    contracts, axes = load_contracts()
    c = copy.deepcopy(contracts["nano"])
    c.training_status = "TRAINED_VERIFIED"
    c.verification_status = "UNVERIFIED"
    c.evidence = ["evaluation/results/x.json"]
    assert any("requires verification_status VERIFIED" in e for e in c.validate(axes))


def test_planned_without_limitations_is_rejected():
    contracts, axes = load_contracts()
    c = copy.deepcopy(contracts["lite"])
    c.known_limitations = []
    assert any("known_limitations" in e for e in c.validate(axes))


def test_unknown_axis_or_missing_threshold_is_rejected():
    contracts, axes = load_contracts()
    c = copy.deepcopy(contracts["nano"])
    c.required_evaluations = list(c.required_evaluations) + ["vibes"]
    assert any("unknown axes" in e for e in c.validate(axes))

    c = copy.deepcopy(contracts["nano"])
    c.promotion_requirements = dict(c.promotion_requirements)
    c.promotion_requirements["min_axis_scores"] = {
        k: v for k, v in c.promotion_requirements["min_axis_scores"].items() if k != "language"
    }
    assert any("no threshold for" in e for e in c.validate(axes))

    c = copy.deepcopy(contracts["nano"])
    c.training_status = "IN_PROGRESS"
    assert any("training_status" in e for e in c.validate(axes))
    assert "UNTRAINED" in TRAINING_STATUSES and "VERIFIED" in VERIFICATION_STATUSES


def test_program_config_and_contracts_agree():
    contracts, axes = load_contracts()
    program = load_config(REPO_ROOT / "configs" / "sir_human_development.yaml")
    program_order = [g["id"] for g in program["generations"]]
    assert program_order == ORDER
    assert [g["name"] for g in program["generations"]] == [c.name for c in contracts.values()]
    # the program lists a broader set of developmental axes; every axis a contract may require must be
    # part of the program's developmental vocabulary
    assert {"knowledge", "reasoning", "application", "verification", "transfer", "brainstorming"} <= set(
        program["development_axes"]
    )


def test_contract_is_json_serialisable():
    contracts, _axes = load_contracts()
    import json

    blob = json.dumps({k: v.to_dict() for k, v in contracts.items()}, ensure_ascii=False)
    assert "SIR-Nano" in blob
    summary = status_summary(contracts)
    assert "TRAINED_VERIFIED" not in summary["by_training_status"]
