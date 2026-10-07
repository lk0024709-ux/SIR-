"""Curriculum-aware sampling.

A trainer that samples uniformly from whatever records happen to exist will over-train the easy,
abundant material and under-train the parts of the curriculum that are actually missing. This module
implements the *policy* layer: given records, coverage and the review schedule, decide which
records a batch should draw from — and report honestly when a requested bucket is empty.

Bucket precedence (a record lands in exactly one bucket — the first that matches):

1. `error_correction` — records generated from diagnosed mistakes (`error_tags` or the correction kinds)
2. `review`            — nodes the spaced-review schedule marks due
3. `transfer`          — transfer tasks (the only evidence that generalisation happened)
4. `weak_areas`        — nodes whose measured accuracy is below `weak_accuracy`
5. `current`           — everything else at the active frontier

Everything is deterministic: the same records, policy and seed produce the same plan, in the same
order, on any machine. Language and difficulty mixes are applied as soft targets with largest-remainder
rounding, and the achieved distribution is reported next to the target so nobody has to guess whether
the mix was honoured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from curriculum.coverage import CoverageReport, CoverageState
from curriculum.graph import DEFAULT_GRAPH, CurriculumGraph
from sir_paths import REPO_ROOT, load_config, rel, resolve, write_json
from training.data.records import TrainingRecord, load_records

DEFAULT_CONFIG = REPO_ROOT / "configs" / "curriculum_sampling.yaml"
BUCKETS = ("current", "review", "weak_areas", "transfer", "error_correction")
ERROR_KINDS = ("error_diagnosis", "correction")


def _hash_key(seed: int, record_id: str) -> str:
    return hashlib.sha256(f"{seed}:{record_id}".encode("utf-8")).hexdigest()


def _largest_remainder(weights: dict[str, float], total: int) -> dict[str, int]:
    """Integer allocation closest to the weights, deterministic under ties (by key order)."""
    if total <= 0:
        return {k: 0 for k in weights}
    positive = {k: max(0.0, float(v)) for k, v in weights.items()}
    s = sum(positive.values())
    if s <= 0:
        return {k: 0 for k in weights}
    raw = {k: v / s * total for k, v in positive.items()}
    floor = {k: int(v) for k, v in raw.items()}
    remainder = total - sum(floor.values())
    order = sorted(raw, key=lambda k: (-(raw[k] - floor[k]), k))
    for k in order[:remainder]:
        floor[k] += 1
    return floor


@dataclass
class SamplingPolicy:
    mixture: dict[str, float] = field(
        default_factory=lambda: {"current": 0.65, "review": 0.15, "weak_areas": 0.10, "transfer": 0.05, "error_correction": 0.05}
    )
    difficulty_mix: dict[str, float] = field(default_factory=dict)
    language_mix: dict[str, float] = field(default_factory=dict)
    max_items: int = 2000
    seed: int = 0
    redistribute_shortfall: bool = True

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "SamplingPolicy":
        s = (cfg or {}).get("sampler") or {}
        policy = cls(
            mixture={str(k): float(v) for k, v in (s.get("mixture") or {}).items()} or cls().mixture,
            difficulty_mix={str(k): float(v) for k, v in (s.get("difficulty_mix") or {}).items()},
            language_mix={str(k): float(v) for k, v in (s.get("language_mix") or {}).items()},
            max_items=int(s.get("max_items", 2000)),
            seed=int(s.get("seed", 0)),
            redistribute_shortfall=bool(s.get("redistribute_shortfall", True)),
        )
        policy.check()
        return policy

    def check(self) -> None:
        unknown = set(self.mixture) - set(BUCKETS)
        if unknown:
            raise ValueError(f"unknown sampler mixture buckets: {sorted(unknown)} (known: {list(BUCKETS)})")
        total = sum(self.mixture.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"sampler.mixture must sum to 1.0 (±0.01); got {total:.4f}")
        if any(v < 0 for v in self.mixture.values()):
            raise ValueError("sampler.mixture entries must be non-negative")
        if self.max_items <= 0:
            raise ValueError("sampler.max_items must be > 0")

    def fingerprint(self) -> str:
        blob = json.dumps(
            {
                "mixture": self.mixture,
                "difficulty_mix": self.difficulty_mix,
                "language_mix": self.language_mix,
                "seed": self.seed,
            },
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class SampleItem:
    record_id: str
    bucket: str
    curriculum_node: str
    difficulty: str
    language: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "bucket": self.bucket,
            "curriculum_node": self.curriculum_node,
            "difficulty": self.difficulty,
            "language": self.language,
        }


@dataclass
class SamplePlan:
    items: list[SampleItem]
    target: dict[str, int]
    achieved: dict[str, int]
    shortfall: dict[str, int]
    redistributed_to_current: int
    seed: int
    policy_fingerprint: str
    language_target: dict[str, int] = field(default_factory=dict)
    language_achieved: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": "curriculum/sampler.py",
            "size": self.size,
            "seed": self.seed,
            "policy_fingerprint": self.policy_fingerprint,
            "target": self.target,
            "achieved": self.achieved,
            "shortfall": self.shortfall,
            "redistributed_to_current": self.redistributed_to_current,
            "language_target": self.language_target,
            "language_achieved": self.language_achieved,
            "notes": self.notes,
            "items": [i.to_dict() for i in self.items],
        }


class CurriculumSampler:
    def __init__(
        self,
        graph: CurriculumGraph,
        policy: SamplingPolicy,
        *,
        coverage: CoverageReport | None = None,
        review_due: Iterable[str] = (),
    ):
        self.graph = graph
        self.policy = policy
        self.coverage = coverage
        self.review_due = set(review_due)

    # ---- bucket assignment -----------------------------------------------------------
    def bucket_for(self, rec: TrainingRecord) -> str:
        if rec.error_tags or rec.kind in ERROR_KINDS:
            return "error_correction"
        if rec.curriculum_node in self.review_due:
            return "review"
        if rec.kind == "transfer":
            return "transfer"
        if self.coverage is not None:
            cov = self.coverage.nodes.get(rec.curriculum_node)
            weak_accuracy = self.coverage.thresholds.weak_accuracy
            if cov is not None and cov.accuracy is not None and cov.accuracy < weak_accuracy:
                return "weak_areas"
        return "current"

    def _frontier_nodes(self) -> set[str]:
        """Nodes that are legitimately in play: started, plus not-started nodes whose prerequisites are begun."""
        if self.coverage is None:
            return set()
        started = {
            nid
            for nid, cov in self.coverage.nodes.items()
            if cov.state not in (CoverageState.NOT_STARTED,)
        }
        frontier = set(started)
        for nid in self.coverage.ready_to_learn:
            frontier.add(nid)
        return frontier

    # ---- planning --------------------------------------------------------------------
    def plan(self, records: Sequence[TrainingRecord]) -> SamplePlan:
        records = list(records)
        frontier = self._frontier_nodes()
        by_bucket: dict[str, list[TrainingRecord]] = {b: [] for b in BUCKETS}
        for rec in records:
            bucket = self.bucket_for(rec)
            if bucket == "current" and frontier and rec.curriculum_node not in frontier:
                # material that is neither started nor unlocked: keep it out of "current"
                bucket = "review" if rec.curriculum_node in self.review_due else "current"
            by_bucket[bucket].append(rec)

        target_total = min(self.policy.max_items, len(records))
        target = _largest_remainder(self.policy.mixture, target_total)
        chosen: list[SampleItem] = []
        achieved: dict[str, int] = {}
        shortfall: dict[str, int] = {}
        notes: list[str] = []

        for bucket in BUCKETS:
            want = target.get(bucket, 0)
            picked = self._pick(by_bucket[bucket], want, bucket)
            achieved[bucket] = len(picked)
            shortfall[bucket] = max(0, want - len(picked))
            if shortfall[bucket]:
                notes.append(
                    f"bucket '{bucket}' supplied {len(picked)} of {want} requested records "
                    f"({shortfall[bucket]} short); shortfall is redistributed to 'current'"
                )
            chosen.extend(picked)

        redistributed = 0
        if self.policy.redistribute_shortfall:
            missing = sum(shortfall.values())
            if missing:
                used = {i.record_id for i in chosen}
                # preference order: the active frontier first, then the other buckets, each
                # deterministic — a record keeps the bucket it was classified into
                preference = ("current", "weak_areas", "review", "transfer", "error_correction")
                leftovers: list[TrainingRecord] = []
                for bucket in preference:
                    leftovers.extend(r for r in by_bucket[bucket] if r.record_id not in used)
                extra = self._pick(leftovers, missing, "redistributed")
                if extra:
                    for item in extra:
                        item.bucket = self.bucket_for(
                            next(r for r in leftovers if r.record_id == item.record_id)
                        )
                        achieved[item.bucket] = achieved.get(item.bucket, 0) + 1
                    redistributed = len(extra)
                    chosen.extend(extra)
                    supplied = sorted({i.bucket for i in extra})
                    notes.append(f"redistributed {len(extra)} record(s) from buckets: {supplied}")
                if len(extra) < missing:
                    notes.append(
                        f"only {len(extra)} of {missing} redistributed records were available; "
                        "the plan is smaller than max_items rather than padded"
                    )

        lang_target = _largest_remainder(self.policy.language_mix, len(chosen)) if self.policy.language_mix else {}
        lang_achieved: dict[str, int] = {}
        for item in chosen:
            lang_achieved[item.language] = lang_achieved.get(item.language, 0) + 1
        if lang_target:
            drift = {
                k: lang_achieved.get(k, 0) - v for k, v in sorted(lang_target.items()) if lang_achieved.get(k, 0) != v
            }
            if drift:
                notes.append(f"language mix drift (achieved - target): {drift}")

        return SamplePlan(
            items=chosen,
            target=target,
            achieved=achieved,
            shortfall=shortfall,
            redistributed_to_current=redistributed,
            seed=self.policy.seed,
            policy_fingerprint=self.policy.fingerprint(),
            language_target=lang_target,
            language_achieved=dict(sorted(lang_achieved.items())),
            notes=notes,
        )

    def _pick(self, candidates: Sequence[TrainingRecord], count: int, bucket: str) -> list[SampleItem]:
        if count <= 0 or not candidates:
            return []
        ordered = sorted(candidates, key=lambda r: (_hash_key(self.policy.seed, r.record_id), r.record_id))
        if self.policy.language_mix:
            per_lang = _largest_remainder(self.policy.language_mix, count)
            picked: list[TrainingRecord] = []
            used: set[str] = set()
            for lang, want in sorted(per_lang.items(), key=lambda kv: (-kv[1], kv[0])):
                if want <= 0:
                    continue
                for rec in ordered:
                    if len([p for p in picked if p.language == lang]) >= want:
                        break
                    if rec.record_id in used:
                        continue
                    if rec.language == lang or (lang == "other" and rec.language not in self.policy.language_mix):
                        picked.append(rec)
                        used.add(rec.record_id)
            for rec in ordered:
                if len(picked) >= count:
                    break
                if rec.record_id not in used:
                    picked.append(rec)
                    used.add(rec.record_id)
            ordered = sorted(picked[:count], key=lambda r: _hash_key(self.policy.seed, r.record_id))
        return [
            SampleItem(
                record_id=r.record_id,
                bucket=bucket,
                curriculum_node=r.curriculum_node,
                difficulty=r.difficulty,
                language=r.language,
            )
            for r in ordered[:count]
        ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Plan a curriculum-aware training batch from records.")
    ap.add_argument("--records", required=True, help="JSONL of training records")
    ap.add_argument("--graph", default=str(DEFAULT_GRAPH))
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--coverage", help="optional coverage report JSON (from curriculum.coverage)")
    ap.add_argument("--out", default="evaluation/results/curriculum_sample_plan.json")
    args = ap.parse_args(argv)

    graph = CurriculumGraph.load(resolve(args.graph))
    policy = SamplingPolicy.from_config(load_config(resolve(args.config)))
    records = load_records(args.records)
    coverage = None
    if args.coverage and resolve(args.coverage).exists():
        from curriculum.coverage import CoverageReport, CoverageThresholds, NodeCoverage

        raw = json.loads(resolve(args.coverage).read_text(encoding="utf-8"))
        coverage = CoverageReport(
            graph_id=raw.get("graph_id", ""),
            thresholds=CoverageThresholds(**raw.get("thresholds", {})),
            nodes={nid: NodeCoverage(node_id=nid, state=CoverageState(d["state"])) for nid, d in raw.get("nodes", {}).items()},
        )
    sampler = CurriculumSampler(graph, policy, coverage=coverage)
    plan = sampler.plan(records)
    out = write_json(resolve(args.out), plan.to_dict())
    print(f"plan: {plan.size} records | target={plan.target} achieved={plan.achieved}")
    if plan.notes:
        for n in plan.notes:
            print(f"  note: {n}")
    print(f"written: {rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
