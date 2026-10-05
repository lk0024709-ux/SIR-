"""Acquisition tests: the license gate, hashing, resume, bounded retries, blocked hosts, offline mode.

These tests never touch the network. Local HTTP servers, a local fixture tree and an injected socket
error exercise every path, because a test suite that needs GitHub is not a test suite.
"""

from __future__ import annotations

import hashlib
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _no_repo_writes(tmp_path, monkeypatch):
    """No test may write into data/manifests: logs and ledgers are redirected into tmp_path."""
    from training.data import acquire as acq

    monkeypatch.setattr(acq, "LOG_PATH", tmp_path / "manifests" / "acquisition_log.jsonl")
    monkeypatch.setattr(
        acq, "artifact_ledger_path", lambda sid: tmp_path / "manifests" / "artifacts" / f"{sid}.jsonl"
    )
    yield

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------------------
def _manifest(tmp_path: Path, sources: list[dict]) -> Path:
    p = tmp_path / "sources.yaml"
    p.write_text(
        yaml.safe_dump(
            {"schema_version": 2, "updated": "2026-10-05", "policy": {"require_training_permission": True}, "sources": sources},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return p


def _source(**over) -> dict:
    base = {
        "id": "unit_source",
        "name": "Unit source",
        "provider": "tests",
        "kind": "web-corpus",
        "url": "https://example.invalid/data",
        "license": "CC0-1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "license_verified": True,
        "license_evidence": "read the licence text on 2026-10-05: CC0 1.0",
        "languages": ["en"],
        "domain": "test",
        "natural": True,
        "synthetic": False,
        "acquisition": {"method": "github_tarball", "repo": "example/data", "ref": "main", "include": ["**/*.txt"]},
        "download_method": "GET",
        "redistribution_allowed": True,
        "commercial_use": "allowed",
        "derivatives": "allowed",
        "model_training_allowed": True,
        "attribution_required": False,
        "share_alike": False,
        "provenance": "unit test",
        "size_tokens": None,
        "status": "available",
        "notes": "unit test source",
    }
    base.update(over)
    return base


def _load(path: Path):
    from training.data.manifest import load_manifest

    return load_manifest(path)


@pytest.fixture()
def local_http_server():
    """Tiny HTTP server with Range support and an injectable failure counter."""
    payload = b"SIR acquisition test payload.\n" * 200
    state = {"requests": 0, "fail_next": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_GET(self):  # noqa: N802
            state["requests"] += 1
            if state["fail_next"] > 0:
                state["fail_next"] -= 1
                self.send_response(500)
                self.end_headers()
                return
            rng = self.headers.get("Range")
            body, status = payload, 200
            if rng and rng.startswith("bytes="):
                body, status = payload[int(rng.split("=")[1].split("-")[0]) :], 206
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(body)

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield {"url": f"http://127.0.0.1:{srv.server_address[1]}/data.txt", "payload": payload, "state": state}
    srv.shutdown()


# --------------------------------------------------------------------------------------
# the gate: what may be fetched at all
# --------------------------------------------------------------------------------------
def test_gate_refuses_needs_review_blocked_and_rejected_sources(tmp_path):
    from training.data.acquire import gate_reason

    for status, reason_key in (
        ("needs_review", "needs_review_reason"),
        ("blocked", "blocked_reason"),
        ("rejected", "rejected_reason"),
    ):
        p = _manifest(tmp_path, [_source(status=status, **{reason_key: "the licence is unread"})])
        assert gate_reason(_load(p), "unit_source"), f"{status} must never be fetchable"


def test_gate_refuses_available_without_training_permission(tmp_path):
    from training.data.acquire import gate_reason

    assert "training permission" in gate_reason(_load(_manifest(tmp_path, [_source(model_training_allowed=False)])), "unit_source")
    assert "license_verified" in gate_reason(_load(_manifest(tmp_path, [_source(license_verified=False)])), "unit_source")


def test_manifest_validation_enforces_the_permission_gate(tmp_path):
    from training.data.manifest import validate_manifest

    issues, _ = validate_manifest(_load(_manifest(tmp_path, [_source(license_verified=False)])), {"records": {}})
    assert any(i.code == "G1-license-not-verified" for i in issues if i.level == "error")


def test_undeclared_source_is_refused(tmp_path):
    from training.data.acquire import acquire_source

    with pytest.raises(SystemExit):
        acquire_source(_load(_manifest(tmp_path, [_source()])), "not_declared")


# --------------------------------------------------------------------------------------
# downloads: size, hash, resume, bounded retries
# --------------------------------------------------------------------------------------
def test_download_measures_bytes_and_sha256(local_http_server, tmp_path):
    from training.data.acquire import download

    dest = tmp_path / "out.bin"
    info = download(local_http_server["url"], dest)
    assert info["bytes"] == len(local_http_server["payload"])
    assert info["sha256"] == hashlib.sha256(local_http_server["payload"]).hexdigest()
    assert dest.read_bytes() == local_http_server["payload"]


def test_download_resumes_from_partial_file(local_http_server, tmp_path):
    from training.data.acquire import download

    dest = tmp_path / "resume.bin"
    part = dest.with_suffix(dest.suffix + ".part")
    part.write_bytes(local_http_server["payload"][:100])  # an interrupted download
    info = download(local_http_server["url"], dest)
    assert info["resumed_from"] == 100
    assert dest.read_bytes() == local_http_server["payload"]
    assert info["sha256"] == hashlib.sha256(local_http_server["payload"]).hexdigest()


def test_download_retries_then_succeeds(local_http_server, tmp_path):
    from training.data.acquire import download

    local_http_server["state"]["fail_next"] = 2
    info = download(local_http_server["url"], tmp_path / "retry.bin", retries=5)
    assert info["attempts"] == 3
    assert local_http_server["state"]["requests"] >= 3


def test_download_gives_up_after_bounded_retries(local_http_server, tmp_path):
    from training.data.acquire import FetchError, download

    local_http_server["state"]["fail_next"] = 10
    with pytest.raises(FetchError) as exc:
        download(local_http_server["url"], tmp_path / "never.bin", retries=3)
    assert exc.value.attempts == 3, "retries must be bounded; an infinite loop is not retrying"


def test_max_bytes_bound_is_enforced(local_http_server, tmp_path):
    from training.data.acquire import FetchError, download

    with pytest.raises(FetchError) as exc:
        download(local_http_server["url"], tmp_path / "big.bin", max_bytes=16, retries=1)
    assert exc.value.kind == "too_large"


# --------------------------------------------------------------------------------------
# blocked hosts, offline mode, fixtures
# --------------------------------------------------------------------------------------
def test_blocked_host_is_classified_and_logged(tmp_path, monkeypatch):
    from training.data import acquire as acq

    log_path = tmp_path / "acquire_log.jsonl"
    monkeypatch.setattr(acq, "LOG_PATH", log_path)

    def refused(*a, **k):
        raise ConnectionResetError("connection reset by peer (simulated egress proxy)")

    monkeypatch.setattr(acq, "open_stream", refused)
    m = _load(_manifest(tmp_path, [_source()]))
    outcome = acq.acquire_source(m, "unit_source", raw_dir=tmp_path / "raw", retries=2)

    assert outcome.status == "blocked"
    entry = json.loads(log_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    for key in ("hostname", "attempted_url", "error", "ts_utc", "retry_status", "blocked_reason_label"):
        assert key in entry, f"a blocked host must record {key}"
    assert entry["blocked_reason_label"].startswith("BLOCKED")
    assert entry["hostname"] == "github.com"


def test_offline_mode_never_touches_the_network(tmp_path, monkeypatch):
    from training.data import acquire as acq

    def explode(*a, **k):  # pragma: no cover
        raise AssertionError("offline mode attempted a network call")

    monkeypatch.setattr(acq, "open_stream", explode)
    outcome = acq.acquire_source(_load(_manifest(tmp_path, [_source()])), "unit_source", raw_dir=tmp_path / "raw", offline=True, retries=1)
    assert outcome.status == "blocked" and outcome.kind == "offline_mode"


def test_local_fixture_fetcher_copies_and_hashes(tmp_path, monkeypatch):
    from training.data import acquire as acq
    from training.data.acquire import acquire_source

    # a test must never write into the repository's manifests: redirect both outputs into tmp_path
    monkeypatch.setattr(acq, "artifact_ledger_path", lambda sid: tmp_path / "manifests" / "artifacts" / f"{sid}.jsonl")

    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "docs.json").write_text(json.dumps([{"id": "a", "text": "hello"}]), encoding="utf-8")
    p = _manifest(
        tmp_path,
        [_source(id="fixture_source", local_path=str(fixture), acquisition={"method": "local_fixture", "path": str(fixture)})],
    )
    outcome = acquire_source(
        _load(p), "fixture_source", raw_dir=tmp_path / "raw", acquisition_path=tmp_path / "manifests" / "acquisition.json", retries=1
    )
    assert outcome.ok and len(outcome.artifacts) == 1
    art = outcome.artifacts[0]
    assert art.bytes == (fixture / "docs.json").stat().st_size and len(art.sha256) == 64
    copied = json.loads((tmp_path / "raw" / "fixture_source" / "docs.json").read_text(encoding="utf-8"))
    assert copied[0]["text"] == "hello"
    assert (tmp_path / "manifests" / "artifacts" / "fixture_source.jsonl").exists()
    assert (tmp_path / "manifests" / "acquisition.json").exists()


# --------------------------------------------------------------------------------------
# the repository's own manifest
# --------------------------------------------------------------------------------------
def test_repo_manifest_declares_the_full_license_record():
    from training.data.manifest import load_manifest

    m = load_manifest(REPO_ROOT / "data" / "manifests" / "sources.yaml")
    required = (
        "provider", "license", "languages", "domain", "download_method", "redistribution_allowed",
        "commercial_use", "derivatives", "model_training_allowed", "attribution_required",
        "share_alike", "provenance", "status",
    )
    for sid, src in m.sources.items():
        for field in required:
            assert field in src.raw, f"{sid} is missing the license-record field {field!r}"


def test_every_available_source_has_verified_training_permission():
    from training.data.manifest import load_manifest

    m = load_manifest(REPO_ROOT / "data" / "manifests" / "sources.yaml")
    available = [s for s in m.sources.values() if s.raw.get("status") == "available"]
    assert available, "the repository must have at least one usable source"
    for src in available:
        assert src.raw["license_verified"] is True, src.id
        assert src.raw["model_training_allowed"] is True, src.id
        assert str(src.raw["license_evidence"]).strip(), f"{src.id}: quote the licence evidence"
        assert str(src.raw["provenance"]).strip(), f"{src.id}: say where the data comes from"


def test_non_available_sources_all_carry_a_reason():
    from training.data.manifest import load_manifest

    m = load_manifest(REPO_ROOT / "data" / "manifests" / "sources.yaml")
    for sid, src in m.sources.items():
        status = src.raw.get("status")
        if status == "blocked":
            assert str(src.raw.get("blocked_reason") or "").strip(), sid
        elif status == "rejected":
            assert str(src.raw.get("rejected_reason") or "").strip(), sid
        elif status == "needs_review":
            assert str(src.raw.get("needs_review_reason") or "").strip(), sid
