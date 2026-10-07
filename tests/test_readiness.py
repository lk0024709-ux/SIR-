"""Training readiness: the gate exists to stop a hopeful run, and to say why in print.

Two properties matter: an unknown check is never a pass, and the level is the weakest link — no
amount of met checks outvotes one blocking failure.
"""

from __future__ import annotations

from development.readiness import ReadinessCheck, ReadinessReport, _check, check_training_readiness


def test_unknown_is_not_a_pass():
    check = _check("tokenizer_available", "tokenizer exists", None, "no evidence")
    assert check.status == "UNKNOWN"
    assert check.to_dict()["status"] == "UNKNOWN"


def test_report_is_ready_only_for_the_positive_levels():
    report = ReadinessReport(generation_id="nano", level="NOT_READY")
    report.checks.append(_check("a", "r", True, "ok"))
    report.checks.append(_check("b", "r", False, "broken"))
    assert report.ready is False
    payload = report.to_dict()
    assert payload["ready_for_serious_pretraining"] is False
    assert payload["blocking_failures"] == ["b"]
    assert "not evidence that any model was trained" in payload["note"]

    report.level = "READY_FOR_SERIOUS"
    assert report.ready is True


def test_live_readiness_report_is_honest_about_the_current_repository():
    report = check_training_readiness(generation_id="nano")
    statuses = {c.check_id: c.status for c in report.checks}

    # these two are implemented and must pass against the committed repository
    assert statuses["curriculum_requirements"] == "MET", report.to_dict()
    assert statuses["generation_contracts"] == "MET", report.to_dict()

    partial = int(report.thresholds["partial_training_min_train_tokens"])
    tokenized = report.measured.get("tokenized_smoke_tokens")
    measured = report.measured.get("measured_tokens")
    best = max([t for t in (measured, tokenized) if t is not None], default=0)
    if best < partial:
        assert report.level == "NOT_READY"
    assert report.blocking, "with 0 measured tokens and no experiment records the gate cannot be ready"
    assert any("not cleared for a serious pretraining run" in n for n in report.notes)


def test_check_ids_are_unique_and_reported():
    report = check_training_readiness()
    ids = [c.check_id for c in report.checks]
    assert len(ids) == len(set(ids))
    payload = report.to_dict()
    assert len(payload["checks"]) == len(ids)
    assert payload["generation_id"] == "nano"


def test_missing_corpus_report_is_reported_as_not_measured(tmp_path):
    # point the suite directory at an empty dir: nothing should crash, the check just reports MISSING
    report = check_training_readiness(curated_suite_dir=tmp_path / "no_suites")
    statuses = {c.check_id: c.status for c in report.checks}
    assert statuses["evaluation_suite_present"] == "NOT_MET"
    assert report.level == "NOT_READY"
