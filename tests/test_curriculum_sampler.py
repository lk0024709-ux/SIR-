"""Curriculum sampling: deterministic plans, honest shortfalls, and bucket precedence.

The properties worth protecting are (a) a plan is reproducible from (records, policy, seed), and
(b) an empty bucket is *reported*, never silently padded with whatever is available.
"""

from __future__ import annotations

import pytest

from curriculum.coverage import CoverageReport, CoverageState, CoverageThresholds, NodeCoverage
from curriculum.graph import load_default_graph
from curriculum.sampler import BUCKETS, CurriculumSampler, SamplingPolicy
from training.data.records import TrainingRecord

TH = CoverageThresholds()


def _record(
    rid: str,
    node: str = "math.g05",
    *,
    kind: str = "practice",
    language: str = "en",
    difficulty: str = "intermediate",
    error_tags: list[str] | None = None,
) -> TrainingRecord:
    return TrainingRecord(
        record_id=rid,
        kind=kind,
        task="t",
        response="r",
        curriculum_node=node,
        domain="foundation",
        subject="mathematics",
        language=language,
        difficulty=difficulty,
        provenance="self_authored",
        verification_status="human_reviewed",
        license_status="cleared",
        source="test",
        error_tags=error_tags or [],
    )


def _coverage(states: dict[str, tuple[CoverageState, float | None]]) -> CoverageReport:
    nodes = {}
    for node, (state, acc) in states.items():
        cov = NodeCoverage(node_id=node, state=state)
        if acc is not None:
            cov.total = 10
            cov.correct = int(acc * 10)
        nodes[node] = cov
    return CoverageReport(graph_id="g", thresholds=TH, nodes=nodes)


def test_policy_from_config_is_validated(tmp_path):
    from sir_paths import REPO_ROOT, load_config

    cfg = load_config(REPO_ROOT / "configs" / "curriculum_sampling.yaml")
    policy = SamplingPolicy.from_config(cfg)
    assert abs(sum(policy.mixture.values()) - 1.0) < 1e-9, "config mixture must sum to exactly 1.0"
    with pytest.raises(ValueError):
        SamplingPolicy(mixture={"current": 0.5, "nonsense": 0.5}).check()
    with pytest.raises(ValueError):
        SamplingPolicy(mixture={"current": 0.5, "review": 0.1}).check()


def test_plan_is_deterministic_for_the_same_seed():
    graph = load_default_graph()
    records = [_record(f"r{i:03d}") for i in range(40)]
    policy = SamplingPolicy(seed=7, max_items=20)
    plan_a = CurriculumSampler(graph, policy).plan(records)
    plan_b = CurriculumSampler(graph, policy).plan(list(reversed(records)))
    assert [i.record_id for i in plan_a.items] == [i.record_id for i in plan_b.items]
    assert plan_a.size == 20


def test_bucket_precedence_error_correction_beats_transfer():
    graph = load_default_graph()
    records = [
        _record("r000", kind="transfer", error_tags=["CALCULATION_ERROR"]),
        _record("r001", kind="transfer"),
    ]
    policy = SamplingPolicy(seed=1, max_items=2, mixture={"current": 0.5, "transfer": 0.25, "error_correction": 0.25})
    plan = CurriculumSampler(graph, policy).plan(records)
    by_id = {i.record_id: i.bucket for i in plan.items}
    assert by_id["r000"] == "error_correction"
    assert by_id["r001"] == "transfer"


def test_review_bucket_follows_the_review_schedule():
    graph = load_default_graph()
    records = [_record("r000", node="math.g05"), _record("r001", node="math.g06")]
    plan = CurriculumSampler(graph, SamplingPolicy(seed=3, max_items=2, mixture={"current": 0.5, "review": 0.5}), review_due={"math.g05"}).plan(records)
    by_id = {i.record_id: i.bucket for i in plan.items}
    assert by_id["r000"] == "review"
    assert by_id["r001"] == "current"


def test_weak_area_bucket_uses_measured_accuracy():
    graph = load_default_graph()
    coverage = _coverage({"math.g05": (CoverageState.PRACTICE, 0.40), "math.g06": (CoverageState.MASTERED, 0.95)})
    records = [_record("r000", node="math.g05"), _record("r001", node="math.g06")]
    plan = CurriculumSampler(graph, SamplingPolicy(seed=3, max_items=2, mixture={"current": 0.5, "weak_areas": 0.5}), coverage=coverage).plan(records)
    by_id = {i.record_id: i.bucket for i in plan.items}
    assert by_id["r000"] == "weak_areas"
    assert by_id["r001"] == "current"


def test_empty_bucket_is_reported_and_shortfall_redistributed():
    graph = load_default_graph()
    records = [_record(f"r{i:03d}") for i in range(20)]  # no transfer, no error tags at all
    policy = SamplingPolicy(seed=5, max_items=10)
    plan = CurriculumSampler(graph, policy).plan(records)
    # nothing in this record set can fill review/weak/transfer/error buckets, and every shortfall
    # must be visible rather than quietly ignored
    assert sum(plan.shortfall.values()) >= 1
    assert plan.redistributed_to_current >= 1
    assert any("shortfall is redistributed" in n for n in plan.notes)
    assert plan.size == 10, "the plan must still reach max_items by redistributing to `current`"
    assert sum(plan.achieved.values()) == plan.size


def test_max_items_is_respected_and_reported():
    graph = load_default_graph()
    records = [_record(f"r{i:03d}") for i in range(6)]
    plan = CurriculumSampler(graph, SamplingPolicy(seed=1, max_items=3)).plan(records)
    assert plan.size == 3
    assert sum(plan.target.values()) == 3


def test_language_mix_is_applied_as_a_soft_target():
    graph = load_default_graph()
    records = [_record(f"hi{i:03d}", language="hi") for i in range(20)] + [
        _record(f"en{i:03d}", language="en") for i in range(20)
    ]
    policy = SamplingPolicy(seed=11, max_items=20, language_mix={"hi": 0.5, "en": 0.5})
    plan = CurriculumSampler(graph, policy).plan(records)
    counts = plan.language_achieved
    assert abs(counts.get("hi", 0) - counts.get("en", 0)) <= 4, counts
    assert sum(plan.language_target.values()) == plan.size


def test_buckets_vocabulary_is_complete():
    assert set(BUCKETS) == {"current", "review", "weak_areas", "transfer", "error_correction"}


def test_plan_serialises_bucket_counts():
    graph = load_default_graph()
    records = [_record("r000", kind="transfer")]
    payload = CurriculumSampler(graph, SamplingPolicy(seed=1, max_items=1, mixture={"transfer": 1.0})).plan(records).to_dict()
    assert payload["size"] == 1
    assert payload["achieved"]["transfer"] == 1
    assert payload["items"][0]["bucket"] == "transfer"
    assert payload["policy_fingerprint"]
