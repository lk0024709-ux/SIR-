"""The requirement contract must actually be enforced against the graph.

These tests exist because the interesting failure is not "the graph is broken" — it is "a topic the
project promised to cover quietly disappeared". The second test proves the check can fail.
"""

from __future__ import annotations

import copy

from curriculum.graph import load_default_graph
from curriculum.requirements import check_requirements
from sir_paths import REPO_ROOT, load_config

REQUIREMENTS = REPO_ROOT / "configs" / "curriculum_requirements.yaml"


def test_committed_requirements_are_satisfied():
    graph = load_default_graph()
    report = check_requirements(graph, load_config(REQUIREMENTS))
    assert report.ok, report.errors
    assert all(c["status"] == "pass" for c in report.checks)


def test_every_required_subject_is_present_and_tagged():
    graph = load_default_graph()
    req = load_config(REQUIREMENTS)
    for subject, topics in req["required_topics"].items():
        assert subject in graph.subjects, f"required subject {subject} missing from the graph"
        missing = set(topics) - graph.topics(subject)
        assert not missing, f"{subject}: no node covers {sorted(missing)}"


def test_missing_topic_is_reported_as_a_violation():
    graph = load_default_graph()
    req = copy.deepcopy(load_config(REQUIREMENTS))
    req["required_topics"]["mathematics"].append("quantum_arithmetic_nonsense")
    report = check_requirements(graph, req)
    assert not report.ok
    assert any("quantum_arithmetic_nonsense" in e for e in report.errors)


def test_grade_coverage_requirement_is_checked():
    graph = load_default_graph()
    req = copy.deepcopy(load_config(REQUIREMENTS))
    req["required_grade_coverage"]["mathematics"] = [1, 12]
    report = check_requirements(graph, req)
    assert not report.ok
    assert any("no foundation node for grade(s) [11, 12]" in e for e in report.errors)


def test_learning_loop_stages_must_be_record_kinds():
    graph = load_default_graph()
    req = copy.deepcopy(load_config(REQUIREMENTS))
    req["learning_loop"].append("telepathy")
    report = check_requirements(graph, req)
    assert not report.ok
    assert any("telepathy" in e for e in report.errors)


def test_generation_order_must_match_the_development_program():
    graph = load_default_graph()
    req = copy.deepcopy(load_config(REQUIREMENTS))
    req["generation_order"] = list(reversed(req["generation_order"]))
    report = check_requirements(graph, req)
    assert not report.ok
    assert any("generation order" in e for e in report.errors)


def test_required_sequence_must_be_real_dependencies():
    graph = load_default_graph()
    req = copy.deepcopy(load_config(REQUIREMENTS))
    req["required_sequences"]["scientific_thinking"] = ["st.g04", "st.g06", "st.g05"]
    report = check_requirements(graph, req)
    assert not report.ok
    assert any("not a real dependency chain" in e for e in report.errors)
