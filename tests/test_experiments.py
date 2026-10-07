"""Experiment tracking: a run that cannot be re-identified must not be recorded as if it could.

The validation rules are the point of the module — an experiment record without a commit, a dataset
hash, a measured parameter count, or (for a completed run) an evaluation is not a record of an
experiment, it is a claim.
"""

from __future__ import annotations

import json

from development.experiments import (
    RECORD_STATUSES,
    ExperimentRecord,
    file_sha256,
    from_train_log,
    index,
    load_records,
    tree_fingerprint,
    write_record,
)


def _record(**overrides) -> ExperimentRecord:
    base = dict(
        experiment_id="unit-s7-c0a1b2c3",
        created_utc="2026-10-07T00:00:00Z",
        status="completed",
        git={"commit": "abc123", "branch": "arena/01a10c78-sir", "dirty": False},
        dataset={"present": True, "hash": {"tree_sha256": "d" * 64}},
        tokenizer={"spec": "data/tokenizers/sp_bpe_2048/spec.json", "sha256": "t" * 64},
        model={"parameter_count": 897_152, "config": {"vocab_size": 2048}},
        training={"steps": 40, "tokens_seen": 20_480, "seed": 7},
        evaluation={"trainer_val_loss": 5.1, "external_results": {}},
    )
    base.update(overrides)
    return ExperimentRecord(**base)


def test_a_complete_record_validates():
    assert _record().validate() == []


def test_status_vocabulary_is_closed():
    assert "completed" in RECORD_STATUSES and "aborted" in RECORD_STATUSES
    assert "finished" not in RECORD_STATUSES
    assert any("status" in e for e in _record(status="finished").validate())


def test_missing_identity_fields_are_refused():
    errors = _record(git={}, dataset={"hash": {}}, tokenizer={}, model={"parameter_count": 0}, training={}).validate()
    joined = "\n".join(errors)
    assert "git.commit" in joined
    assert "tree_sha256" in joined
    assert "tokenizer identity" in joined
    assert "parameter_count" in joined
    assert "training.steps" in joined


def test_completed_run_without_evaluation_is_refused():
    errors = _record(evaluation={}).validate()
    assert any("no evaluation results" in e for e in errors)


def test_failed_or_aborted_runs_must_state_what_went_wrong():
    assert any("known_failures" in e for e in _record(status="failed").validate())
    assert _record(status="failed", known_failures=["loss diverged at step 30"]).validate() == []


def test_write_refuses_an_invalid_record_and_round_trips_a_valid_one(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        write_record(_record(status="completed", evaluation={}), tmp_path)
    path = write_record(_record(), tmp_path)
    assert path.exists()
    loaded = load_records(tmp_path)
    assert len(loaded) == 1 and loaded[0].experiment_id == _record().experiment_id
    assert loaded[0].training["seed"] == 7


def test_index_is_a_measured_summary():
    rows = index([_record(), _record(experiment_id="other", status="planned")])
    assert [r["experiment_id"] for r in rows] == sorted(["unit-s7-c0a1b2c3", "other"])
    assert rows[0]["params"] == 897_152


def _fake_train_log(tmp_path) -> tuple:
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "tokens.bin").write_bytes(b"\x01\x02" * 50)
    (processed / "meta.json").write_text(
        json.dumps(
            {
                "file": "tokens.bin",
                "dtype": "uint16",
                "n_tokens": 100,
                "n_docs": 2,
                "seq_len": 16,
                "packed": True,
                "tokenizer_sha256": "t" * 64,
            }
        ),
        encoding="utf-8",
    )
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"model_type": "byte_bpe"}), encoding="utf-8")
    log = {
        "experiment": "unit_test",
        "seed": 7,
        "config_sha256": "deadbeefcafe",
        "config": "configs/sir_nano_smoke.yaml",
        "git": {"commit": "abc123", "branch": "test", "dirty": True},
        "data": {"train_tokens": 1000, "val_tokens": 100, "processed_dir": str(processed)},
        "counts": {"params": {"total": 897_152, "embedding": 45_056}},
        "tokens_seen": 1024,
        "batch": 4,
        "optimizer": "adamw",
        "log": [{"step": 1}, {"step": 40}],
        "final_eval": {"val_loss": 5.0, "val_perplexity": 148.4, "tokens_scored": 100},
        "status_note": "smoke run",
    }
    log_path = tmp_path / "train_log.json"
    log_path.write_text(json.dumps(log), encoding="utf-8")
    return log_path, processed, spec


def test_from_train_log_builds_a_complete_record(tmp_path):
    log_path, processed, spec = _fake_train_log(tmp_path)
    rec = from_train_log(
        log_path,
        processed_dir=processed,
        tokenizer_spec=spec,
        evaluation_results={"suite_v0": {"mathematics": 0.5}},
        created_utc="2026-10-07T12:00:00Z",
    )
    assert rec.validate() == []
    assert rec.experiment_id.startswith("unit_test-s7-c")
    assert rec.model["parameter_count"] == 897_152
    assert rec.training["steps"] == 40, "steps must come from the training log, not from the best checkpoint"
    assert rec.training["tokens_seen"] == 1024
    assert rec.dataset["hash"]["tree_sha256"]
    assert rec.tokenizer["sha256"] == file_sha256(spec)
    assert rec.evaluation["trainer_val_loss"] == 5.0
    assert rec.evaluation["external_results"]["suite_v0"]["mathematics"] == 0.5


def test_tree_fingerprint_changes_when_the_data_changes(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"one")
    first = tree_fingerprint([tmp_path])
    (tmp_path / "a.bin").write_bytes(b"two")
    second = tree_fingerprint([tmp_path])
    assert first["tree_sha256"] != second["tree_sha256"]
    assert first["files"] == 1 and first["bytes"] == 3
