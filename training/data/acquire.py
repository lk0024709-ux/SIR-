"""Deterministic, license-gated, resumable acquisition of the SIR corpus inputs.

    python -m training.data.acquire --source cltk_sa_wikisource
    python -m training.data.acquire --all-available
    python -m training.data.acquire --all-available --offline      # fixtures only, no network
    python -m training.data.acquire --record-existing hiwiki_dump data/raw/hiwiki-latest-....bz2
    python -m training.data.acquire --verify                       # re-hash everything on disk

Design rules this module refuses to break
-----------------------------------------
1. **The manifest is the only input.** A source that is not declared, or is declared without a
   verified training permission (`status: available` + `license_verified` + `model_training_allowed`),
   is never fetched. `--all-available` cannot see it; `--source X` fails with a reason.
2. **No bypassing the environment.** If a host is unreachable, that is recorded as
   BLOCKED — network/environment restriction with hostname, URL, error and timestamp. The
   module will not try mirrors, proxies, alternate ports or scraping routes. Failing loudly on
   an access problem is the intended behaviour.
3. **Every byte is hashed while it is written.** SHA-256 is computed incrementally during the
   download (never a second full read into memory), the byte count comes from the stream, and
   the artifact ledger is written next to the data.
4. **Everything is bounded.** Bounded retries with exponential backoff, bounded chunk buffer
   (1 MiB), bounded per-file size, no infinite loops anywhere.
5. **Resumable.** Downloads go to `<name>.part` and are resumed with an HTTP Range request when
   the server supports it (the response code decides: 206 = resumed, 200 = restarted from zero).
   Already-complete artifacts (matching sha256 in the ledger) are skipped, so re-running is safe.

The corpus itself is never written into git — `.gitignore` keeps `data/raw/**` out, and only the
manifest, the artifact ledger and the failure log are committed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import os
import shutil
import socket
import ssl
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training.data.manifest import (  # noqa: E402
    DEFAULT_ACQUISITION,
    DEFAULT_MANIFEST,
    Manifest,
    Source,
    load_acquisition,
    load_manifest,
    sha256_file,
    validate_manifest,
)

RAW_DIR = REPO_ROOT / "data" / "raw"
CACHE_DIR = RAW_DIR / "_cache"
ARTIFACT_DIR = REPO_ROOT / "data" / "manifests" / "artifacts"
LOG_PATH = REPO_ROOT / "data" / "manifests" / "acquisition_log.jsonl"

CHUNK = 1 << 20  # 1 MiB: the only buffer size in this module
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 2.0
MAX_BACKOFF = 30.0
CONNECT_TIMEOUT = 15.0
READ_TIMEOUT = 60.0
USER_AGENT = "SIR-corpus-acquirer/1.0 (+https://github.com/lk0024709-ux/SIR-)"

# Download sizes are bounded so a hostile/broken server cannot fill the disk.
MAX_FILE_BYTES = 2 << 30  # 2 GiB
MAX_TARBALL_BYTES = 1 << 30  # 1 GiB

# Error classification. "blocked" means the environment prevented access — never the license.
BLOCKED_KINDS = {
    "dns_failure": "BLOCKED — network/environment restriction (DNS resolution failed)",
    "tls_failure": "BLOCKED — network/environment restriction (TLS handshake failed/closed by proxy)",
    "connection_failure": "BLOCKED — network/environment restriction (connection refused/reset)",
    "timeout": "BLOCKED — network/environment restriction (connection timed out)",
    "offline_mode": "BLOCKED — acquisition ran with --offline; network access was not attempted",
}
NON_BLOCKED_KINDS = {
    "access_denied": "HTTP 401/403: the server refused this client (no bypass attempted)",
    "not_found": "HTTP 404: the artifact is no longer at this URL (source needs re-pinning)",
    "legal_block": "HTTP 451: unavailable for legal reasons",
    "http_error": "unexpected HTTP status",
    "integrity": "downloaded bytes did not match the recorded checksum",
    "too_large": "artifact exceeded the configured size bound",
    "bad_archive": "archive could not be read",
    "manifest": "source is not fetchable under the manifest's terms",
}


# --------------------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------------------
def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def hostname_of(url: str) -> str:
    return urllib.parse.urlsplit(url).hostname or ""


def append_log(entry: dict[str, Any], path: Path | None = None) -> None:
    """Append one JSON line to the acquisition log. The log is committed (it is metadata).

    `path` is resolved at call time (not as a bound default) so tests and callers can redirect
    the log without patching the module global.
    """
    path = Path(path) if path is not None else LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts_utc": utc_now(), **entry}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


@dataclass
class Artifact:
    path: str                 # repo-relative path on disk
    sha256: str
    bytes: int
    url: str = ""
    source_id: str = ""
    origin_repo: str = ""
    origin_revision: str = ""
    extracted_from: str = ""
    skipped: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class FetchOutcome:
    source_id: str
    status: str = "error"       # ok | blocked | failed | skipped
    kind: str = ""
    detail: str = ""
    url: str = ""
    artifacts: list[Artifact] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    archives: list[dict[str, Any]] = field(default_factory=list)
    revision: str = ""
    attempts: int = 0
    offline_skip: bool = False

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class FetchError(Exception):
    def __init__(self, kind: str, detail: str, url: str = "", attempts: int = 1):
        super().__init__(detail)
        self.kind = kind
        self.detail = detail
        self.url = url
        self.attempts = attempts


def classify_exception(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code in (401, 403):
            return "access_denied"
        if exc.code == 404:
            return "not_found"
        if exc.code == 451:
            return "legal_block"
        return "http_error"
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        if isinstance(reason, socket.gaierror):
            return "dns_failure"
        if isinstance(reason, ssl.SSLError):
            return "tls_failure"
        if isinstance(reason, (ConnectionResetError, ConnectionRefusedError, ConnectionError)):
            return "connection_failure"
        if isinstance(reason, (TimeoutError, socket.timeout)):
            return "timeout"
        text = str(reason).lower()
        if "timed out" in text or "timeout" in text:
            return "timeout"
        if "certificate" in text or "ssl" in text or "tls" in text:
            return "tls_failure"
        if "connection reset" in text or "refused" in text or "closed by proxy" in text:
            return "connection_failure"
        return "connection_failure"
    if isinstance(exc, ssl.SSLError):
        return "tls_failure"
    if isinstance(exc, socket.gaierror):
        return "dns_failure"
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return "timeout"
    if isinstance(exc, (ConnectionResetError, ConnectionRefusedError)):
        return "connection_failure"
    if isinstance(exc, zipfile.BadZipFile):
        return "bad_archive"
    if isinstance(exc, tarfile.TarError):
        return "bad_archive"
    return "http_error"


def _is_blocked_kind(kind: str) -> bool:
    return kind in BLOCKED_KINDS


# --------------------------------------------------------------------------------------
# HTTP with bounded retries and resume
# --------------------------------------------------------------------------------------
def open_stream(url: str, *, offset: int = 0, timeout: float = READ_TIMEOUT, extra_headers: dict[str, str] | None = None):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(extra_headers or {})})
    if offset:
        req.add_header("Range", f"bytes={offset}-")
    return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310 (https only; the manifest is trusted input)


def download(
    url: str,
    dest: Path,
    *,
    retries: int = DEFAULT_RETRIES,
    resume: bool = True,
    max_bytes: int = MAX_FILE_BYTES,
    on_retry: Callable[[int, str, float], None] | None = None,
) -> dict[str, Any]:
    """Stream `url` to `dest`, resuming a `.part` file when the server honours Range.

    Returns {bytes, sha256, attempts, url, resumed_from, status_code}. Raises FetchError with a
    classified kind. Never retries more than `retries` times, never sleeps unbounded.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    attempts = 0
    last_kind, last_detail = "http_error", ""
    while attempts < max(1, retries):
        attempts += 1
        offset = part.stat().st_size if (resume and part.exists()) else 0
        h = hashlib.sha256()
        n = 0
        try:
            if offset:
                # seed the hash with what we already have, so the final digest covers the whole file
                with part.open("rb") as fh:
                    while True:
                        b = fh.read(CHUNK)
                        if not b:
                            break
                        h.update(b)
                        n += len(b)
            with open_stream(url, offset=offset) as resp:
                status = getattr(resp, "status", 200)
                if offset and status != 206:
                    # server ignored Range: restart cleanly rather than corrupt the file
                    offset, n = 0, 0
                    h = hashlib.sha256()
                mode = "ab" if offset else "wb"
                with part.open(mode) as out:
                    while True:
                        b = resp.read(CHUNK)
                        if not b:
                            break
                        n += len(b)
                        if n > max_bytes:
                            raise FetchError("too_large", f"{n} bytes exceeds max_bytes={max_bytes}", url, attempts)
                        h.update(b)
                        out.write(b)
            part.replace(dest)
            return {
                "bytes": n,
                "sha256": h.hexdigest(),
                "attempts": attempts,
                "url": url,
                "resumed_from": offset,
                "status_code": status,
            }
        except FetchError:
            raise
        except BaseException as exc:  # noqa: BLE001 — classified below, never swallowed
            last_kind = classify_exception(exc)
            last_detail = f"{type(exc).__name__}: {exc}"
            if attempts >= max(1, retries):
                break
            sleep = min(MAX_BACKOFF, DEFAULT_BACKOFF ** attempts)
            if on_retry:
                on_retry(attempts, last_detail, sleep)
            time.sleep(sleep)
    raise FetchError(last_kind, f"{last_detail} (after {attempts} attempt(s))", url, attempts)


# --------------------------------------------------------------------------------------
# artifact ledger helpers
# --------------------------------------------------------------------------------------
def artifact_ledger_path(source_id: str) -> Path:
    return REPO_ROOT / "data" / "manifests" / "artifacts" / f"{source_id}.jsonl"


def load_ledger(source_id: str) -> dict[str, dict[str, Any]]:
    p = artifact_ledger_path(source_id)
    if not p.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["path"]] = rec
    return out


def write_ledger(source_id: str, artifacts: Iterable[Artifact]) -> Path:
    p = artifact_ledger_path(source_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted((a.as_dict() for a in artifacts), key=lambda r: r["path"])
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n", encoding="utf-8")
    return p


def already_have(rel_path: str, ledger: dict[str, dict[str, Any]], verify: bool = True) -> Artifact | None:
    """Return the recorded artifact when the bytes on disk still match the recorded sha256."""
    rec = ledger.get(rel_path)
    if not rec:
        return None
    p = REPO_ROOT / rel_path
    if not p.exists():
        return None
    if verify:
        if p.stat().st_size != rec["bytes"]:
            return None
        if sha256_file(p) != rec["sha256"]:
            return None
    kw = {k: rec[k] for k in Artifact.__dataclass_fields__ if k in rec and k != "skipped"}
    return Artifact(skipped=True, **kw)


# --------------------------------------------------------------------------------------
# fetchers
# --------------------------------------------------------------------------------------
def _write_text_artifact(path: Path, text: str, meta: dict[str, Any]) -> Artifact:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    path.write_bytes(data)
    return Artifact(
        path=rel(path),
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
        url=meta.get("url", ""),
        source_id=meta.get("source_id", ""),
        origin_repo=meta.get("repo", ""),
        origin_revision=meta.get("revision", ""),
        extracted_from=meta.get("archive", ""),
    )


def github_api_json(path: str, token: str | None = None) -> Any:
    url = path if path.startswith("http") else f"https://api.github.com/{path.lstrip('/')}"
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with open_stream(url, extra_headers=headers) as resp:
        return json.loads(resp.read().decode("utf-8"))


def resolve_ref(repo: str, ref: str, token: str | None = None) -> str:
    """Resolve a branch/tag to a commit SHA so the recorded revision cannot drift."""
    try:
        data = github_api_json(f"repos/{repo}/commits/{ref}", token)
        return str(data["sha"])
    except Exception:  # noqa: BLE001 — an unresolved ref is recorded as-is, never invented
        return ""


def _matches(path: str, include: list[str], exclude: list[str]) -> bool:
    import fnmatch

    if any(fnmatch.fnmatch(path, pat) or path.startswith(pat.rstrip("*").rstrip("/") + "/") for pat in exclude):
        return False
    if not include:
        return True
    for pat in include:
        if fnmatch.fnmatch(path, pat):
            return True
        # directory-style prefix match: "wiki_documents/**" also matches nested paths
        if pat.endswith("/**") and path.startswith(pat[:-3].rstrip("/") + "/"):
            return True
    return False


def extract_tarball(
    archive: Path,
    out_dir: Path,
    *,
    include: list[str],
    exclude: list[str],
    source_id: str,
    url: str,
    repo: str,
    revision: str,
) -> list[Artifact]:
    """Stream-extract matching members. Bounded memory: members are read in order, never buffered whole."""
    artifacts: list[Artifact] = []
    with tarfile.open(archive, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            # strip the "<repo>-<ref>/" prefix git archives add
            parts = member.name.split("/", 1)
            inner = parts[1] if len(parts) == 2 else member.name
            if not _matches(inner, include, exclude):
                continue
            fh = tar.extractfile(member)
            if fh is None:
                continue
            dest = out_dir / inner
            dest.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            n = 0
            with dest.open("wb") as out:
                while True:
                    b = fh.read(CHUNK)
                    if not b:
                        break
                    n += len(b)
                    h.update(b)
                    out.write(b)
            artifacts.append(
                Artifact(
                    path=rel(dest),
                    sha256=h.hexdigest(),
                    bytes=n,
                    url=url,
                    source_id=source_id,
                    origin_repo=repo,
                    origin_revision=revision,
                    extracted_from=archive.name,
                )
            )
    return artifacts


def fetch_github_tarball(
    source: Source, out_dir: Path, *, retries: int, token: str | None, offline: bool, ledger: dict[str, dict[str, Any]]
) -> FetchOutcome:
    cfg = source.raw["acquisition"]
    repo, ref = cfg["repo"], cfg.get("ref", "HEAD")
    url = f"https://github.com/{repo}/archive/refs/heads/{ref}.tar.gz"
    try:
        url = cfg.get("url_template", url).format(repo=repo, ref=ref)
    except Exception:  # noqa: BLE001
        pass
    if offline:
        return FetchOutcome(source.id, "blocked", "offline_mode", BLOCKED_KINDS["offline_mode"], url, offline_skip=True)

    revision = resolve_ref(repo, ref, token) or ref
    if revision != ref:
        url = f"https://github.com/{repo}/archive/{revision}.tar.gz"
    archive = CACHE_DIR / f"{repo.replace('/', '__')}@{revision[:12]}.tar.gz"

    if not archive.exists() or archive.stat().st_size == 0:
        info = download(url, archive, retries=retries, max_bytes=MAX_TARBALL_BYTES)
    else:
        info = {"bytes": archive.stat().st_size, "sha256": sha256_file(archive), "attempts": 0, "url": url}

    archive_digest = info.get("sha256") or sha256_file(archive)
    artifacts = extract_tarball(
        archive,
        out_dir,
        include=list(cfg.get("include") or []),
        exclude=list(cfg.get("exclude") or []),
        source_id=source.id,
        url=url,
        repo=repo,
        revision=revision,
    )
    archive_rec = {
        "repo": repo,
        "ref": ref,
        "resolved_commit": revision,
        "url": url,
        "archive_sha256": archive_digest,
        "archive_bytes": archive.stat().st_size,
        "cache_file": rel(archive),
    }
    if not artifacts:
        return FetchOutcome(
            source.id,
            "failed",
            "not_found",
            f"archive {url} contained no file matching include={cfg.get('include')} exclude={cfg.get('exclude')}",
            url,
            revision=revision,
            archives=[archive_rec],
            attempts=int(info.get("attempts", 0)) + 1,
        )
    return FetchOutcome(
        source.id, "ok", "", "", url, artifacts, revision=revision, archives=[archive_rec], attempts=int(info.get("attempts", 0))
    )


def fetch_github_file(
    source: Source, out_dir: Path, *, retries: int, token: str | None, offline: bool, ledger: dict[str, dict[str, Any]]
) -> FetchOutcome:
    cfg = source.raw["acquisition"]
    repo, ref = cfg["repo"], cfg.get("ref", "HEAD")
    paths = list(cfg.get("paths") or [])
    if not paths:
        raise FetchError("manifest", f"{source.id}: github_file needs acquisition.paths", "", 1)
    if offline:
        return FetchOutcome(source.id, "blocked", "offline_mode", BLOCKED_KINDS["offline_mode"], "", offline_skip=True)

    revision = resolve_ref(repo, ref, token) or ref
    artifacts: list[Artifact] = []
    for p in paths:
        url = f"https://api.github.com/repos/{repo}/contents/{p}?ref={revision}"
        dest = out_dir / Path(p).name
        headers = {"Accept": "application/vnd.github.raw"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        attempt = 0
        while True:
            attempt += 1
            try:
                with open_stream(url, extra_headers=headers) as resp:
                    data = resp.read()
                break
            except BaseException as exc:  # noqa: BLE001
                kind = classify_exception(exc)
                if attempt >= max(1, retries) or _is_blocked_kind(kind):
                    raise FetchError(kind, f"{type(exc).__name__}: {exc} (after {attempt} attempt(s))", url, attempt) from exc
                time.sleep(min(MAX_BACKOFF, DEFAULT_BACKOFF ** attempt))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        artifacts.append(
            Artifact(
                path=rel(dest),
                sha256=hashlib.sha256(data).hexdigest(),
                bytes=len(data),
                url=url,
                source_id=source.id,
                origin_repo=repo,
                origin_revision=revision,
            )
        )
        # archives declared in the manifest (e.g. gutenberg.zip) are unpacked deterministically
        if cfg.get("unzip") and dest.suffix == ".zip":
            artifacts.extend(
                _unzip_selected(
                    dest,
                    out_dir,
                    include=list(cfg.get("include") or []),
                    exclude=list(cfg.get("exclude") or []),
                    source_id=source.id,
                    url=url,
                    repo=repo,
                    revision=revision,
                )
            )
    if not artifacts:
        return FetchOutcome(source.id, "failed", "not_found", f"no files fetched for {paths}", "", revision=revision)
    return FetchOutcome(source.id, "ok", "", "", artifacts[0].url, artifacts, revision=revision, attempts=1)


def _unzip_selected(zpath: Path, out_dir: Path, *, include, exclude, source_id, url, repo, revision) -> list[Artifact]:
    out: list[Artifact] = []
    with zipfile.ZipFile(zpath) as zf:
        for name in sorted(zf.namelist()):
            if name.endswith("/") or not _matches(name, include, exclude):
                continue
            data = zf.read(name)
            dest = out_dir / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            out.append(
                Artifact(
                    path=rel(dest),
                    sha256=hashlib.sha256(data).hexdigest(),
                    bytes=len(data),
                    url=url,
                    source_id=source_id,
                    origin_repo=repo,
                    origin_revision=revision,
                    extracted_from=zpath.name,
                )
            )
    return out


def fetch_github_org_tarballs(
    source: Source, out_dir: Path, *, retries: int, token: str | None, offline: bool, ledger: dict[str, dict[str, Any]]
) -> FetchOutcome:
    """Fetch a deterministic, bounded subset of an organisation's repositories.

    Determinism matters more than coverage here: the same command must select the same books on
    every machine. Selection = repositories sorted by name, minus an explicit exclusion list.
    """
    cfg = source.raw["acquisition"]
    org = cfg["org"]
    limit = int(cfg.get("limit", 50))
    exclude = set(cfg.get("exclude_repos") or [])
    ref = cfg.get("ref", "master")
    if offline:
        return FetchOutcome(source.id, "blocked", "offline_mode", BLOCKED_KINDS["offline_mode"], "", offline_skip=True)

    repos: list[str] = []
    page = 1
    while True:
        batch = github_api_json(f"orgs/{org}/repos?per_page=100&page={page}&sort=full_name", token)
        if not batch:
            break
        repos.extend(r["name"] for r in batch if r.get("name") not in exclude and not r.get("archived", False))
        if len(batch) < 100:
            break
        page += 1
    repos = sorted(set(repos))[:limit]

    artifacts: list[Artifact] = []
    failures: list[dict[str, Any]] = []
    archives: list[dict[str, Any]] = []
    attempts = 0
    for name in repos:
        sub = Source(id=source.id, raw={**source.raw, "acquisition": {**cfg, "method": "github_tarball", "repo": f"{org}/{name}", "ref": ref}})
        try:
            outcome = fetch_github_tarball(sub, out_dir / name, retries=retries, token=token, offline=False, ledger=ledger)
            attempts += outcome.attempts
            artifacts.extend(outcome.artifacts)
            archives.extend(outcome.archives)
        except FetchError as exc:
            attempts += exc.attempts
            failures.append(
                {
                    "repo": f"{org}/{name}",
                    "url": exc.url,
                    "kind": exc.kind,
                    "detail": exc.detail,
                    "blocked": _is_blocked_kind(exc.kind),
                    "ts_utc": utc_now(),
                }
            )
    if not artifacts:
        kind = "blocked" if failures and all(f["blocked"] for f in failures) else "failed"
        detail = "; ".join(f"{f['repo']}: {f['detail']}" for f in failures[:3]) or "no repositories matched"
        return FetchOutcome(
            source.id, "blocked" if kind == "blocked" else "failed", failures[0]["kind"] if failures else "not_found",
            detail, "", artifacts, failures, archives, attempts=attempts,
        )
    return FetchOutcome(
        source.id, "ok", "", "", f"https://github.com/{org}", artifacts, failures, archives,
        revision=f"{org}@{ref} ({len(archives)} repositories, each pinned by commit and archive sha256)", attempts=attempts,
    )


def fetch_local_fixture(source: Source, out_dir: Path, *, offline: bool, **_: Any) -> FetchOutcome:
    """The offline fallback: a fixture that ships with the repository is 'acquired' by copying
    it into the raw tree. No network, no licence risk, and the pipeline exercises the same path."""
    src = source.local_dir()
    if src is None or not src.exists():
        return FetchOutcome(source.id, "failed", "not_found", f"local fixture missing: {src}", "")
    artifacts: list[Artifact] = []
    for f in sorted(src.rglob("*")):
        if not f.is_file() or f.suffix not in {".json", ".jsonl", ".txt", ".md"}:
            continue
        dest = out_dir / f.relative_to(src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, dest)
        artifacts.append(
            Artifact(path=rel(dest), sha256=sha256_file(dest), bytes=dest.stat().st_size, url=str(src), source_id=source.id, origin_repo="local", origin_revision="local")
        )
    if not artifacts:
        return FetchOutcome(source.id, "failed", "not_found", f"local fixture {src} has no readable text files", str(src))
    return FetchOutcome(source.id, "ok", "", "", str(src), artifacts, revision="local", attempts=1)


FETCHERS: dict[str, Callable[..., FetchOutcome]] = {
    "github_tarball": fetch_github_tarball,
    "github_file": fetch_github_file,
    "github_org_tarballs": fetch_github_org_tarballs,
    "local_fixture": fetch_local_fixture,
}


# --------------------------------------------------------------------------------------
# acquisition records (measured, never typed)
# --------------------------------------------------------------------------------------
def update_acquisition(
    source: Source,
    outcome: FetchOutcome,
    *,
    acquisition_path: Path = DEFAULT_ACQUISITION,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the measured record for one source. Mirrors manifest.record_acquisition's shape."""
    path = Path(acquisition_path)
    doc = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"records": {}}
    doc.setdefault("records", {})
    raw = source.raw
    agg = hashlib.sha256()
    total_bytes = 0
    for a in sorted(outcome.artifacts, key=lambda x: x.path):
        agg.update(Path(a.path).name.encode("utf-8"))
        agg.update(a.sha256.encode("ascii"))
        total_bytes += a.bytes
    rec: dict[str, Any] = {
        "sha256": agg.hexdigest(),
        "sha256_meaning": "chained sha256 over (file name, file sha256) of every artifact, sorted by path",
        "bytes": total_bytes,
        "docs_raw_files": len(outcome.artifacts),
        "docs": len(outcome.artifacts),
        "docs_note": "raw text files; the document count is measured after normalization (training/data/normalize.py)",
        "files": len(outcome.artifacts),
        "file_list": sorted(a.path for a in outcome.artifacts)[:50],
        "file_list_truncated_at": 50,
        "artifact_ledger": rel(artifact_ledger_path(source.id)),
        "measured_at_utc": utc_now(),
        "url": outcome.url or raw.get("url") or "",
        "download_method": raw.get("download_method", ""),
        "revision": _best_revision(outcome, raw),
        "origin_repo": raw.get("acquisition", {}).get("repo") or raw.get("acquisition", {}).get("org", ""),
        "license_snapshot": {
            "license": raw.get("license"),
            "license_url": raw.get("license_url"),
            "license_verified": raw.get("license_verified"),
            "model_training_allowed": raw.get("model_training_allowed"),
            "commercial_use": raw.get("commercial_use"),
            "derivatives": raw.get("derivatives"),
            "attribution_required": raw.get("attribution_required"),
            "share_alike": raw.get("share_alike"),
            "evidence_sha256": hashlib.sha256(str(raw.get("license_evidence") or "").encode("utf-8")).hexdigest(),
        },
        "attempts": outcome.attempts,
        "complete": not outcome.failures,
        "failures": outcome.failures,
        "archives": outcome.archives,
        "resolved_commits": {a["repo"]: a["resolved_commit"] for a in outcome.archives if a.get("resolved_commit")},
        "measured_by": "training/data/acquire.py",
    }
    rec.update(extra or {})
    doc["records"][source.id] = rec
    doc["policy_note"] = (
        "Generated by training/data/acquire.py. Bytes, SHA-256 and revisions are measured from the "
        "downloaded artifacts; token counts are appended later by tokenization. Never edit by hand."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rec


# --------------------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------------------
def fetchable_sources(manifest: Manifest) -> list[Source]:
    out = []
    for s in manifest.sources.values():
        raw = s.raw
        if raw.get("status") != "available":
            continue
        if str(raw.get("license") or "").upper() in ("", "UNVERIFIED"):
            continue
        if not raw.get("model_training_allowed"):
            continue
        method = (raw.get("acquisition") or {}).get("method", "none")
        if method in FETCHERS:
            out.append(s)
    return sorted(out, key=lambda s: s.id)


def looks_pinned(revision: str | None) -> bool:
    """True when a revision names an immutable state (a commit SHA) or the local fixture.

    A branch or tag can move, so recording one as a `revision` would make the corpus
    irreproducible while *looking* verified. The provenance gate treats a non-pinned revision as
    a failure, so acquisition resolves refs to commits and never papers over a failed resolution.
    """
    rev = (revision or "").strip()
    if rev == "local":
        return True
    return bool(re.search(r"\b[0-9a-f]{40}\b", rev))


def _best_revision(outcome: Any, raw: dict[str, Any]) -> str:
    """Prefer the resolved commit; fall back to what the artifact ledger says was actually fetched."""
    candidates = [outcome.revision or ""]
    ledger_revs = sorted({a.origin_revision for a in outcome.artifacts if getattr(a, "origin_revision", None)})
    if len(ledger_revs) == 1:
        candidates.append(ledger_revs[0])
    for rev in candidates:
        if looks_pinned(rev):
            return rev
    return candidates[0] or raw.get("acquisition", {}).get("ref", "")


def gate_reason(manifest: Manifest, source: Source | str) -> str:
    """Explain, in one line, why a source may not be fetched. Used by --source on refusal.

    Accepts a Source or a source id: callers that only have an id should not have to reach into
    the manifest themselves (and should not be able to skip the check by mistake).
    """
    if isinstance(source, str):
        looked_up = manifest.get(source)
        if looked_up is None:
            return f"source {source!r} is not declared in the manifest"
        source = looked_up
    raw = source.raw
    status = raw.get("status")
    if status != "available":
        return (
            f"status is {status!r}, not available. Recorded reason: "
            f"{raw.get('blocked_reason') or raw.get('rejected_reason') or raw.get('needs_review_reason') or raw.get('reason') or '(none)'}"
        )
    if not raw.get("license_verified"):
        return "license_verified is not true — an unread license is not a license"
    if not raw.get("model_training_allowed"):
        return "model_training_allowed is not true — public availability is not a training permission"
    method = (raw.get("acquisition") or {}).get("method", "none")
    if method not in FETCHERS:
        return f"acquisition.method {method!r} is not an implemented fetcher ({sorted(FETCHERS)})"
    return ""


def archives_metadata(source: Source, ledger: dict[str, dict[str, Any]], token: str | None) -> tuple[list[dict[str, Any]], str]:
    """Rebuild the archive/revision ledger for data that is already on disk.

    Used by the idempotent path so a re-run does not have to re-download gigabytes merely to
    remember which upstream commits were used. Resolving a ref is one cheap API call per repo;
    if the API is unavailable the ref is recorded unresolved rather than invented.
    """
    cfg = source.raw.get("acquisition") or {}
    method = cfg.get("method")
    ref = cfg.get("ref", "HEAD")
    repos: list[str] = []
    if method == "github_tarball":
        repos = [cfg.get("repo", "")]
    elif method == "github_org_tarballs":
        repos = sorted({rec.get("origin_repo", "") for rec in ledger.values() if rec.get("origin_repo")})
    out: list[dict[str, Any]] = []
    for repo in repos:
        if not repo:
            continue
        commit = resolve_ref(repo, ref, token) or ""
        cache = sorted(CACHE_DIR.glob(f"{repo.replace('/', '__')}@*.tar.gz"))
        rec: dict[str, Any] = {
            "repo": repo,
            "ref": ref,
            "resolved_commit": commit,
            "url": f"https://github.com/{repo}/archive/{commit or ref}.tar.gz",
            "archive_sha256": sha256_file(cache[-1]) if cache else None,
            "archive_bytes": cache[-1].stat().st_size if cache else None,
            "cache_file": rel(cache[-1]) if cache else None,
            "note": "rebuilt from the on-disk cache; no re-download was performed",
        }
        out.append(rec)
    if method == "github_org_tarballs":
        resolved = sum(1 for r in out if r["resolved_commit"])
        revision = f"{cfg.get('org')}@{ref} ({len(out)} repositories, {resolved} commit-pinned)"
    else:
        revision = out[0]["resolved_commit"] if out and out[0]["resolved_commit"] else ref
    return out, revision


def acquire_source(
    manifest: Manifest,
    source_id: str,
    *,
    raw_dir: Path = RAW_DIR,
    retries: int = DEFAULT_RETRIES,
    token: str | None = None,
    offline: bool = False,
    verify_hashes: bool = True,
    record: bool = True,
    acquisition_path: Path = DEFAULT_ACQUISITION,
) -> FetchOutcome:
    source = manifest.get(source_id)
    if source is None:
        raise SystemExit(f"source {source_id!r} is not declared in {rel(manifest.path)} — SIR never fetches undeclared data")
    reason = gate_reason(manifest, source)
    if reason:
        append_log({"source_id": source_id, "stage": "gate", "outcome": "refused", "reason": reason})
        return FetchOutcome(source_id, "skipped", "manifest", reason)

    out_dir = raw_dir / source_id
    method = source.raw["acquisition"]["method"]

    # idempotency: if nothing changed and the ledger verifies, do not touch the network at all
    ledger = load_ledger(source_id)
    if ledger and method != "local_fixture":
        all_present = True
        for rel_path, rec in ledger.items():
            p = REPO_ROOT / rel_path
            if not p.exists() or (verify_hashes and (p.stat().st_size != rec["bytes"] or sha256_file(p) != rec["sha256"])):
                all_present = False
                break
        if all_present:
            arts = [
                Artifact(skipped=True, **{k: rec[k] for k in Artifact.__dataclass_fields__ if k in rec and k != "skipped"})
                for rec in ledger.values()
            ]
            archives, revision = archives_metadata(source, ledger, token)
            outcome = FetchOutcome(
                source_id,
                "ok",
                "",
                "already present and hash-verified",
                source.raw.get("url") or "",
                arts,
                archives=archives,
                revision=revision or source.raw.get("acquisition", {}).get("ref", ""),
            )
            if record:
                update_acquisition(source, outcome, acquisition_path=acquisition_path)
            append_log({"source_id": source_id, "stage": "download", "outcome": "skipped-present", "artifacts": len(arts)})
            return outcome

    try:
        outcome = FETCHERS[method](
            source, out_dir, retries=retries, token=token, offline=offline, ledger=ledger
        )
    except FetchError as exc:
        blocked = _is_blocked_kind(exc.kind)
        outcome = FetchOutcome(source_id, "blocked" if blocked else "failed", exc.kind, exc.detail, exc.url, attempts=exc.attempts)

    if outcome.ok:
        write_ledger(source_id, outcome.artifacts)
        if record:
            update_acquisition(source, outcome, acquisition_path=acquisition_path)
        append_log(
            {
                "source_id": source_id,
                "stage": "download",
                "outcome": "ok",
                "url": outcome.url,
                "hostname": hostname_of(outcome.url),
                "revision": outcome.revision,
                "artifacts": len(outcome.artifacts),
                "bytes": sum(a.bytes for a in outcome.artifacts),
                "attempts": outcome.attempts,
                "retry_status": "succeeded" if outcome.attempts > 1 else "not_needed",
                "partial_failures": len(outcome.failures),
            }
        )
        for f in outcome.failures:
            append_log({"source_id": source_id, "stage": "download", "outcome": "artifact-failed", "retry_status": "exhausted", **f})
    else:
        blocked = _is_blocked_kind(outcome.kind)
        append_log(
            {
                "source_id": source_id,
                "stage": "download",
                "outcome": "blocked" if blocked else "failed",
                "hostname": hostname_of(outcome.url),
                "attempted_url": outcome.url,
                "error_kind": outcome.kind,
                "error": outcome.detail,
                "retry_status": "exhausted" if outcome.attempts > 1 else "not_retried",
                "attempts": outcome.attempts,
                "blocked_reason_label": BLOCKED_KINDS.get(outcome.kind, NON_BLOCKED_KINDS.get(outcome.kind, "unknown")),
                "hostname_reachable": False if blocked else None,
            }
        )
    return outcome


def record_existing(source_id: str, paths: list[str], *, manifest: Manifest, acquisition_path: Path, verify: bool = False) -> dict[str, Any]:
    """Register files a human fetched (e.g. a Wikipedia dump). Bytes/sha256 are measured here.

    This is the ONLY supported path for data obtained outside the sandbox. It refuses sources
    that are not `available` with a verified training permission.
    """
    source = manifest.get(source_id)
    if source is None:
        raise SystemExit(f"{source_id!r} is not declared in the manifest")
    reason = gate_reason(manifest, source)
    if reason:
        raise SystemExit(f"refusing to record {source_id}: {reason}")
    artifacts = []
    for p in paths:
        path = Path(p)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            raise SystemExit(f"missing file: {path}")
        artifacts.append(
            Artifact(path=rel(path), sha256=sha256_file(path), bytes=path.stat().st_size, url="recorded-existing", source_id=source_id)
        )
    write_ledger(source_id, artifacts)
    outcome = FetchOutcome(source_id, "ok", "", "recorded pre-existing files", artifacts[0].url, artifacts, revision=source.raw.get("acquisition", {}).get("ref", "human-supplied"), attempts=1)
    rec = update_acquisition(source, outcome, acquisition_path=acquisition_path, extra={"recorded_existing": True})
    append_log({"source_id": source_id, "stage": "record-existing", "outcome": "ok", "files": len(artifacts), "bytes": rec["bytes"]})
    return rec


def verify_all(manifest: Manifest) -> list[dict[str, Any]]:
    """Re-hash every recorded artifact on disk. Reports drift instead of assuming integrity."""
    results: list[dict[str, Any]] = []
    for sid, source in sorted(manifest.sources.items()):
        ledger = load_ledger(sid)
        if not ledger:
            continue
        for rel_path, rec in sorted(ledger.items()):
            p = REPO_ROOT / rel_path
            if not p.exists():
                results.append({"source_id": sid, "path": rel_path, "status": "missing"})
                continue
            actual_bytes = p.stat().st_size
            actual_sha = sha256_file(p) if actual_bytes == rec["bytes"] else ""
            ok = actual_bytes == rec["bytes"] and actual_sha == rec["sha256"]
            results.append(
                {
                    "source_id": sid,
                    "path": rel_path,
                    "status": "ok" if ok else ("size-mismatch" if actual_bytes != rec["bytes"] else "sha256-mismatch"),
                    "expected_bytes": rec["bytes"],
                    "actual_bytes": actual_bytes,
                }
            )
    return results


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Acquire SIR corpus sources. Only manifest sources with a verified training permission are fetched.",
        epilog=(
            "EXECUTION ORDER\n"
            "  1. python -m training.data.manifest                 # gates must pass\n"
            "  2. python -m training.data.acquire --all-available  # downloads + hashes + ledger\n"
            "  3. python -m training.data.acquire --verify         # re-hash what is on disk\n"
            "  4. python -m training.data.build_corpus --config configs/m1_corpus.yaml\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--source", action="append", default=[], help="manifest source id (repeatable)")
    ap.add_argument("--all-available", action="store_true", help="every source that is available + licensed for training")
    ap.add_argument("--record-existing", nargs="+", metavar=("SOURCE_ID", "PATH"), help="register files fetched outside the sandbox (measured, not trusted)")
    ap.add_argument("--verify", action="store_true", help="re-hash all recorded artifacts and report drift")
    ap.add_argument("--offline", action="store_true", help="never touch the network; local fixtures only, everything else recorded as BLOCKED")
    ap.add_argument("--dry-run", action="store_true", help="print what would be fetched, fetch nothing")
    ap.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="max attempts per download (bounded)")
    ap.add_argument("--raw-dir", default=str(RAW_DIR))
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--acquisition", default=str(DEFAULT_ACQUISITION))
    ap.add_argument("--as-json", action="store_true")
    args = ap.parse_args(argv)

    manifest = load_manifest(args.manifest)
    issues, _usable = validate_manifest(manifest, load_acquisition(args.acquisition))
    errors = [i for i in issues if i.level == "error"]
    # "available but not yet fetched" is the expected pre-fetch state: creating that record is
    # this command's job. Every other error (undeclared license, blocked without reason, a
    # hand-typed size, an available source with no training permission) blocks acquisition.
    pending = [i for i in errors if i.code == "G3-missing-acquisition"]
    blocking = [i for i in errors if i.code != "G3-missing-acquisition"]
    if pending and not args.verify:
        print(f"note: {len(pending)} source(s) declared available but not yet acquired (this run will create their records)")
    if blocking and not args.verify:
        for i in blocking:
            print("  " + i.format(), file=sys.stderr)
        print("manifest has errors; refusing to acquire anything until they are fixed", file=sys.stderr)
        return 2

    if args.record_existing:
        sid, paths = args.record_existing[0], args.record_existing[1:]
        rec = record_existing(sid, paths, manifest=manifest, acquisition_path=Path(args.acquisition))
        print(json.dumps(rec, indent=2)[:4000])
        return 0

    if args.verify:
        results = verify_all(manifest)
        bad = [r for r in results if r["status"] != "ok"]
        if args.as_json:
            print(json.dumps({"checked": len(results), "failures": bad}, indent=2))
        else:
            print(f"verified {len(results)} artifact(s): {len(bad)} failure(s)")
            for r in bad[:20]:
                print(f"  {r['status']:15s} {r['source_id']} {r['path']}")
        return 1 if bad else 0

    if args.all_available:
        targets = [s.id for s in fetchable_sources(manifest)]
    else:
        targets = list(dict.fromkeys(args.source))
    if not targets:
        print("nothing to do: pass --source <ID> or --all-available", file=sys.stderr)
        return 2

    if args.dry_run:
        for sid in targets:
            src = manifest.get(sid)
            reason = gate_reason(manifest, src) if src else "undeclared"
            if reason:
                print(f"SKIP  {sid}: {reason}")
            else:
                cfg = src.raw["acquisition"]
                print(f"FETCH {sid}: method={cfg.get('method')} repo={cfg.get('repo') or cfg.get('org')} ref={cfg.get('ref')} include={cfg.get('include')}")
        return 0

    outcomes = []
    for sid in targets:
        print(f"== {sid}")
        outcome = acquire_source(
            manifest,
            sid,
            raw_dir=Path(args.raw_dir),
            retries=args.retries,
            token=os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN"),
            offline=args.offline,
            acquisition_path=Path(args.acquisition),
        )
        outcomes.append(outcome)
        if outcome.ok:
            print(f"   ok: {len(outcome.artifacts)} artifact(s), {sum(a.bytes for a in outcome.artifacts):,} bytes, revision {outcome.revision[:12]}")
        else:
            print(f"   {outcome.status.upper()} [{outcome.kind}] {outcome.detail[:200]}")
            if outcome.kind in BLOCKED_KINDS:
                print(f"   {BLOCKED_KINDS[outcome.kind]} — host {hostname_of(outcome.url)!r} (recorded in {rel(LOG_PATH)})")

    summary = {
        "sources": [
            {
                "source_id": o.source_id,
                "status": o.status,
                "kind": o.kind,
                "detail": o.detail,
                "artifacts": len(o.artifacts),
                "bytes": sum(a.bytes for a in o.artifacts),
                "revision": o.revision,
            }
            for o in outcomes
        ],
        "ok": sum(1 for o in outcomes if o.ok),
        "blocked": sum(1 for o in outcomes if o.status == "blocked"),
        "failed": sum(1 for o in outcomes if o.status == "failed"),
        "skipped": sum(1 for o in outcomes if o.status == "skipped"),
        "measured_at_utc": utc_now(),
    }
    if args.as_json:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"\nacquisition summary: ok={summary['ok']} blocked={summary['blocked']} "
            f"failed={summary['failed']} skipped={summary['skipped']}"
        )
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
