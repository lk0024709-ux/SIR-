"""Regression protection: a candidate may not trade one axis for another.

The gate's whole value is that it refuses comparisons it cannot justify (suite mismatch) and that it
treats a missing axis as a regression rather than as "no data, no problem".
"""

from __future__ import annotations

import json

from development.regression import compare_runs, load_run

SUITE = "sha256:alpha"


def _run(axes: dict[str, float], suite: str = SUITE) -> dict:
    return {"axes": {k: {"score": v, "cases": 10} for k, v in axes.items()}, "suite_hash": suite}


def test_identical_runs_are_stable_and_pass():
    report = compare_runs(_run({"mathematics": 0.5}), _run({"mathematics": 0.5}))
    assert report.verdict == "PASS"
    assert report.axes[0].verdict == "stable"
    assert report.regressed_axes == []


def test_improvement_passes_and_is_reported():
    report = compare_runs(_run({"mathematics": 0.4}), _run({"mathematics": 0.6}))
    assert report.verdict == "PASS"
    assert report.improved_axes == ["mathematics"]


def test_regression_fails_even_when_another_axis_improves_sharply():
    report = compare_runs(_run({"mathematics": 0.6, "science": 0.4}), _run({"mathematics": 0.4, "science": 0.8}))
    assert report.verdict == "FAIL"
    assert report.regressed_axes == ["mathematics"]
    assert "does not compensate" in report.notes[-1]


def test_tolerance_absorbs_noise_and_can_be_overridden_per_axis():
    assert compare_runs(_run({"mathematics": 0.5}), _run({"mathematics": 0.49})).verdict == "PASS"
    strict = compare_runs(_run({"mathematics": 0.5}), _run({"mathematics": 0.49}), per_axis_tolerance={"mathematics": 0.001})
    assert strict.verdict == "FAIL"


def test_different_suite_hashes_invalidate_the_comparison():
    report = compare_runs(_run({"mathematics": 0.9}, "hash-a"), _run({"mathematics": 0.1}, "hash-b"))
    assert report.verdict == "INVALID"
    assert report.axes[0].verdict == "incomparable"
    assert "different test sets" in report.notes[0]


def test_missing_axis_in_the_candidate_is_a_regression():
    report = compare_runs(_run({"mathematics": 0.5, "science": 0.5}), _run({"mathematics": 0.5}))
    assert report.verdict == "FAIL"
    assert report.regressed_axes == ["science"]
    assert report.axes[1].verdict == "missing_in_candidate"


def test_load_run_reads_a_suite_result_file(tmp_path):
    path = tmp_path / "run.json"
    payload = {
        "suite_hash": "abc",
        "axes": {"mathematics": {"score": 0.75, "cases": 8}, "language": {"score": None, "cases": 0}},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    axes, suite = load_run(path)
    assert suite == "abc"
    assert axes["mathematics"] == 0.75
    # a None score means "no scored cases": it must not be invented as a number, so the axis is absent
    # and the comparison reports it as missing rather than as a healthy zero
    assert "language" not in axes


def test_flat_score_mapping_is_accepted():
    report = compare_runs({"scores": {"mathematics": 0.5}}, {"scores": {"mathematics": 0.5}})
    assert report.axes[0].axis == "mathematics"
