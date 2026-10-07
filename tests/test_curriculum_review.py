"""Spaced review: deterministic scheduling, no wall clock, and a monotone forgetting-risk signal."""

from __future__ import annotations

import pytest

from curriculum.review import ReviewItem, SpacedReviewScheduler


def test_first_schedule_uses_the_first_interval():
    s = SpacedReviewScheduler()
    item = s.start("math.g05", day=0, performance=0.7)
    assert item.stage == 0
    assert item.review_due_day == 1
    assert item.reviews == 1


def test_good_performance_stretches_the_interval():
    s = SpacedReviewScheduler()
    item = s.start("math.g05", day=0, performance=0.9)
    item = s.review(item, day=1, performance=0.95)
    assert item.stage >= 1
    assert item.review_due_day == 1 + s.interval_for(item.stage)


def test_poor_performance_shrinks_the_interval_but_never_below_zero():
    s = SpacedReviewScheduler()
    item = s.start("math.g05", day=0, performance=0.2)
    first_due = item.review_due_day
    item = s.review(item, day=first_due, performance=0.1)
    assert item.stage == 0
    assert item.review_due_day - item.last_seen_day <= max(s.intervals)


def test_moving_backwards_in_time_is_refused():
    s = SpacedReviewScheduler()
    item = s.start("math.g05", day=10, performance=0.9)
    with pytest.raises(ValueError):
        s.review(item, day=5, performance=0.9)


def test_due_and_latest_day():
    s = SpacedReviewScheduler()
    items = [s.start("math.g05", 0, 0.7), s.start("math.g06", 5, 0.7)]
    assert s.latest_day(items) == 5
    assert s.due_ids(items, day=1) == ["math.g05"]
    assert set(s.due_ids(items, day=6)) == {"math.g05", "math.g06"}


def test_forgetting_risk_is_monotone_in_time():
    s = SpacedReviewScheduler()
    item = s.start("math.g05", day=0, performance=0.9)
    risks = [s.forgetting_risk(item, day) for day in range(0, 40)]
    assert risks == sorted(risks)
    assert risks[0] == 0.0
    assert risks[-1] == 1.0


def test_summary_reports_high_risk_concepts():
    s = SpacedReviewScheduler()
    items = [s.start("math.g05", 0, 0.9), s.start("math.g06", 300, 0.9)]
    summary = s.summary(items, day=300)
    assert summary["tracked_concepts"] == 2
    assert any(node == "math.g05" for node, _risk in summary["high_risk_concepts"])
    assert "not a fitted probability model" in summary["method_note"]


def test_state_round_trips_through_jsonl(tmp_path):
    s = SpacedReviewScheduler()
    items = [s.start("math.g05", 0, 0.9), s.review(s.start("math.g06", 2, 0.7), 5, 0.85)]
    path = tmp_path / "review_state.jsonl"
    s.save(items, path)
    loaded = s.load(path)
    assert [i.node_id for i in loaded] == ["math.g05", "math.g06"]
    assert loaded[1].reviews == 2
    assert loaded[1].history == [2, 5]


def test_invalid_intervals_are_rejected():
    for kwargs in ({"intervals": ()}, {"intervals": (0, 1)}, {"promote_threshold": 0.5, "reset_threshold": 0.6}):
        with pytest.raises(ValueError):
            SpacedReviewScheduler(**kwargs)


def test_item_dict_is_stable_for_identical_state():
    item = ReviewItem(node_id="math.g05", stage=2, last_seen_day=4, performance=0.8, review_due_day=11)
    again = ReviewItem.from_dict(item.to_dict())
    assert again.to_dict() == item.to_dict()
