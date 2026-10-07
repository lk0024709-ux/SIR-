"""Curriculum graph: structure, ladder integrity, and the failure modes validation must catch."""

from __future__ import annotations

import copy

import pytest
import yaml

from curriculum.graph import DEFAULT_GRAPH_PATH, CurriculumGraph, load_default_graph
from curriculum.schema import CurriculumNode

MINI_DOC = {
    "schema_version": 1,
    "graph_id": "test_graph",
    "subjects": [{"id": "mathematics", "title": "Mathematics"}],
    "nodes": [
        {
            "id": "math.g01",
            "title": "Numbers",
            "subject": "mathematics",
            "track": "foundation",
            "band": "primary",
            "grade": 1,
            "depends_on": [],
            "topics": ["numbers"],
        },
        {
            "id": "math.g02",
            "title": "Addition",
            "subject": "mathematics",
            "track": "foundation",
            "band": "primary",
            "grade": 2,
            "depends_on": ["math.g01"],
            "topics": ["arithmetic"],
        },
        {
            "id": "adv.math.algebra",
            "title": "Advanced algebra",
            "subject": "mathematics",
            "track": "advanced",
            "band": "undergraduate",
            "grade": None,
            "depends_on": ["math.g02"],
            "topics": ["algebra"],
        },
    ],
}


def _graph(doc: dict) -> CurriculumGraph:
    return CurriculumGraph.from_dict(copy.deepcopy(doc))


# --------------------------------------------------------------------------------------
# the committed graph
# --------------------------------------------------------------------------------------
def test_committed_graph_loads_and_validates():
    graph = load_default_graph()
    assert len(graph) > 100, "the curriculum ladder should be broad, not a stub"
    assert graph.validate() == []
    assert DEFAULT_GRAPH_PATH.exists()


def test_committed_graph_covers_every_required_grade_ladder():
    graph = load_default_graph()
    for subject in ("mathematics", "science", "hindi", "english", "computer_science", "reasoning"):
        grades = sorted(
            n.grade for n in graph.nodes.values() if n.subject == subject and n.track == "foundation"
        )
        assert grades == list(range(1, 11)), f"{subject} ladder has gaps: {grades}"


def test_graph_order_is_topological_and_deterministic():
    graph = load_default_graph()
    order = graph.order()
    position = {nid: i for i, nid in enumerate(order)}
    for node in graph.nodes.values():
        for dep in node.depends_on:
            assert position[dep] < position[node.id], f"{node.id} came before its prerequisite {dep}"
    assert order == graph.order()


def test_prerequisite_queries_are_consistent():
    graph = load_default_graph()
    assert "math.g01" in graph.ancestors("math.g10")
    assert "math.g10" in graph.descendants("math.g01")
    assert graph.children("math.g01") == ("math.g02",)
    assert graph.prerequisites("math.g02") == ("math.g01",)


def test_cross_domain_dependencies_exist_where_they_are_required():
    graph = load_default_graph()
    # Hinglish cannot precede Hindi and English, and Indian knowledge must be reachable from school subjects
    assert set(graph.prerequisites("hinglish.g06")) >= {"hindi.g06", "english.g06"}
    assert "ss.g05" in graph.prerequisites("ik.g05")


def test_every_indian_knowledge_node_carries_an_epistemic_note():
    graph = load_default_graph()
    for node in graph.nodes.values():
        if node.subject == "indian_knowledge":
            assert len(node.epistemic_note) > 40, f"{node.id} lacks an epistemic note"


# --------------------------------------------------------------------------------------
# failure modes validation must catch
# --------------------------------------------------------------------------------------
def test_valid_mini_graph_passes():
    assert _graph(MINI_DOC).validate() == []


def test_grade_ladder_gap_is_an_error():
    doc = copy.deepcopy(MINI_DOC)
    doc["nodes"][1]["grade"] = 3  # Class 3 depends on Class 1: the Class 2 rung is missing
    errors = _graph(doc).validate()
    assert any("must depend on the Class 2 node" in e or "gaps" in e for e in errors), errors


def test_cycle_is_detected_not_silently_ordered():
    doc = copy.deepcopy(MINI_DOC)
    doc["nodes"][0]["depends_on"] = ["math.g02"]
    errors = _graph(doc).validate()
    assert any("cycle" in e for e in errors), errors
    with pytest.raises(ValueError):
        _graph(doc).order()


def test_missing_dependency_is_an_error():
    doc = copy.deepcopy(MINI_DOC)
    doc["nodes"][2]["depends_on"] = ["does.not.exist"]
    errors = _graph(doc).validate()
    assert any("unknown node" in e for e in errors), errors


def test_forward_dependency_within_foundation_is_an_error():
    doc = copy.deepcopy(MINI_DOC)
    doc["nodes"][0]["depends_on"] = ["math.g02"]  # Class 1 depending on Class 2
    errors = _graph(doc).validate()
    assert any("higher-grade" in e or "cycle" in e for e in errors), errors


def test_undeclared_subject_is_an_error():
    doc = copy.deepcopy(MINI_DOC)
    doc["nodes"][0]["subject"] = "astrology"
    errors = _graph(doc).validate()
    assert any("not declared" in e for e in errors), errors


def test_advanced_orphan_is_an_error():
    doc = copy.deepcopy(MINI_DOC)
    doc["nodes"][2]["depends_on"] = []
    errors = _graph(doc).validate()
    assert any("no prerequisites" in e for e in errors), errors


def test_non_self_authored_provenance_is_rejected_in_the_graph():
    node = CurriculumNode.from_dict(
        {
            "id": "math.g01",
            "title": "Numbers",
            "subject": "mathematics",
            "track": "foundation",
            "band": "primary",
            "grade": 1,
            "depends_on": [],
            "topics": ["numbers"],
            "provenance": "licensed",
        }
    )
    assert any("provenance" in e for e in node.validate(["mathematics"]))


def test_node_without_topics_is_rejected():
    node = CurriculumNode.from_dict(
        {
            "id": "math.g01",
            "title": "Numbers",
            "subject": "mathematics",
            "track": "foundation",
            "band": "primary",
            "grade": 1,
            "depends_on": [],
        }
    )
    assert any("topic tag" in e for e in node.validate(["mathematics"]))


def test_committed_graph_file_is_parseable_yaml_with_card():
    doc = yaml.safe_load(DEFAULT_GRAPH_PATH.read_text(encoding="utf-8"))
    assert doc["graph_id"] == "sir_curriculum_v0"
    assert doc["card"] == "data/curriculum/CARD.md"
    assert (DEFAULT_GRAPH_PATH.parent / "CARD.md").exists()
