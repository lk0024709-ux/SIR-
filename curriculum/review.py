"""Spaced review: revisiting mastered material so it does not quietly decay.

A curriculum that only moves forward produces a model that "mastered" Class 3 fractions in March and
cannot do them in September. This module keeps the minimum bookkeeping needed to notice that:
per concept — last seen, performance, current interval stage, when it is next due, and a forgetting
risk derived from days-since-seen relative to the interval.

Deliberate simplifications, stated rather than hidden:

* **Days are integers supplied by the caller.** Nothing reads the wall clock, so schedules are
  reproducible and testable; a nightly job decides what "today" is.
* **The interval ladder is a fixed, transparent sequence** (1, 3, 7, 16, 35, 75, 160 days) with a
  performance-driven step up or down. It is not spaced-repetition research; it is a defensible
  first approximation that can be replaced later without changing the data format.
* **Forgetting risk is a heuristic ratio, not a measured probability.** It is reported as
  `days_since_seen / interval` because inventing a probability curve nobody validated would be
  worse than exposing the simple quantity.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from sir_paths import rel, resolve

DEFAULT_INTERVALS = (1, 3, 7, 16, 35, 75, 160)
DEFAULT_STATE_PATH = "data/curriculum/review_state.jsonl"


@dataclass
class ReviewItem:
    node_id: str
    stage: int = 0
    last_seen_day: int = 0
    performance: float = 0.0
    difficulty: str = "intermediate"
    review_due_day: int = 0
    reviews: int = 0
    history: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "ReviewItem":
        return cls(
            node_id=str(d.get("node_id", "")),
            stage=int(d.get("stage", 0)),
            last_seen_day=int(d.get("last_seen_day", 0)),
            performance=float(d.get("performance", 0.0)),
            difficulty=str(d.get("difficulty", "intermediate")),
            review_due_day=int(d.get("review_due_day", 0)),
            reviews=int(d.get("reviews", 0)),
            history=[int(x) for x in (d.get("history") or [])],
        )

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "stage": self.stage,
            "last_seen_day": self.last_seen_day,
            "performance": round(self.performance, 4),
            "difficulty": self.difficulty,
            "review_due_day": self.review_due_day,
            "reviews": self.reviews,
            "history": list(self.history),
        }


class SpacedReviewScheduler:
    def __init__(
        self,
        intervals: Iterable[int] = DEFAULT_INTERVALS,
        *,
        promote_threshold: float = 0.8,
        reset_threshold: float = 0.6,
    ):
        self.intervals = tuple(int(i) for i in intervals)
        if not self.intervals or any(i <= 0 for i in self.intervals):
            raise ValueError("review intervals must be positive day counts")
        if not 0.0 <= reset_threshold < promote_threshold <= 1.0:
            raise ValueError("require reset_threshold < promote_threshold within [0, 1]")
        self.promote_threshold = promote_threshold
        self.reset_threshold = reset_threshold

    # ---- scheduling ------------------------------------------------------------------
    def interval_for(self, stage: int) -> int:
        return self.intervals[max(0, min(int(stage), len(self.intervals) - 1))]

    def start(self, node_id: str, day: int, performance: float, difficulty: str = "intermediate") -> ReviewItem:
        """Register a first assessment of a concept (usually the point it is reached in the curriculum)."""
        stage = self._next_stage(0, performance)
        return ReviewItem(
            node_id=node_id,
            stage=stage,
            last_seen_day=int(day),
            performance=float(performance),
            difficulty=difficulty,
            review_due_day=int(day) + self.interval_for(stage),
            reviews=1,
            history=[int(day)],
        )

    def review(self, item: ReviewItem, day: int, performance: float, difficulty: str | None = None) -> ReviewItem:
        """Record a review outcome and reschedule. Day must not go backwards."""
        if int(day) < item.last_seen_day:
            raise ValueError(
                f"review day {day} is before last_seen_day {item.last_seen_day} for {item.node_id}: "
                "a schedule that moves backwards makes forgetting risk meaningless"
            )
        stage = self._next_stage(item.stage, performance)
        return ReviewItem(
            node_id=item.node_id,
            stage=stage,
            last_seen_day=int(day),
            performance=float(performance),
            difficulty=difficulty or item.difficulty,
            review_due_day=int(day) + self.interval_for(stage),
            reviews=item.reviews + 1,
            history=[*item.history, int(day)],
        )

    def _next_stage(self, stage: int, performance: float) -> int:
        if performance >= self.promote_threshold:
            return min(stage + 1, len(self.intervals) - 1)
        if performance < self.reset_threshold:
            return max(stage - 1, 0)
        return stage

    # ---- queries ---------------------------------------------------------------------
    def due(self, items: Iterable[ReviewItem], day: int) -> list[ReviewItem]:
        return sorted((i for i in items if i.review_due_day <= int(day)), key=lambda i: (i.review_due_day, i.node_id))

    def due_ids(self, items: Iterable[ReviewItem], day: int) -> list[str]:
        return [i.node_id for i in self.due(items, day)]

    def latest_day(self, items: Iterable[ReviewItem], default: int = 0) -> int:
        return max((i.last_seen_day for i in items), default=default)

    def forgetting_risk(self, item: ReviewItem, day: int) -> float:
        """Heuristic 0..1: how far past its interval this concept is. Not a measured probability."""
        interval = max(1, self.interval_for(item.stage))
        days_since = max(0, int(day) - item.last_seen_day)
        overdue = max(0.0, (days_since - interval) / interval)
        base = days_since / interval
        return round(min(1.0, 0.5 * base + 0.5 * overdue), 4)

    def summary(self, items: Iterable[ReviewItem], day: int) -> dict:
        items = list(items)
        risks = [self.forgetting_risk(i, day) for i in items]
        return {
            "tracked_concepts": len(items),
            "due": len(self.due(items, day)),
            "day": int(day),
            "mean_forgetting_risk": round(sum(risks) / len(risks), 4) if risks else None,
            "high_risk_concepts": sorted(
                [
                    (i.node_id, self.forgetting_risk(i, day))
                    for i in items
                    if self.forgetting_risk(i, day) >= 0.75
                ]
            ),
            "method_note": (
                "forgetting_risk is days_since_seen/interval blended with overdue time — a transparent "
                "heuristic, not a fitted probability model"
            ),
        }

    # ---- persistence -----------------------------------------------------------------
    def save(self, items: Iterable[ReviewItem], path: Path | str = DEFAULT_STATE_PATH) -> Path:
        p = resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            for item in sorted(items, key=lambda i: i.node_id):
                fh.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
        return p

    def load(self, path: Path | str = DEFAULT_STATE_PATH) -> list[ReviewItem]:
        p = resolve(path)
        if not p.exists():
            return []
        out: list[ReviewItem] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(ReviewItem.from_dict(json.loads(line)))
        return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Inspect or advance the spaced-review schedule.")
    ap.add_argument("--state", default=DEFAULT_STATE_PATH)
    ap.add_argument("--day", type=int, required=True, help="integer day index used as 'today'")
    ap.add_argument("--out", help="optional JSON summary path")
    args = ap.parse_args(argv)

    scheduler = SpacedReviewScheduler()
    items = scheduler.load(args.state)
    summary = scheduler.summary(items, args.day) if items else {"tracked_concepts": 0, "note": "no review state yet"}
    print(json.dumps(summary, indent=2))
    if args.out:
        from sir_paths import write_json

        print(f"written: {rel(write_json(resolve(args.out), summary))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
