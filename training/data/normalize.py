"""Stage 0.5 — normalization: raw acquired artifacts into SIR's common document schema.

    python -m training.data.normalize --source cltk_sa_wikisource --out data/normalized

Why this stage exists separately from cleaning
----------------------------------------------
Acquisition gets bytes onto disk; cleaning makes text *good*; normalization makes it *uniform*
and keeps the paper trail. Every document that leaves this module carries:

    {doc_id, source_id, language, text, license, license_verified,
     synthetic, natural, provenance:{artifact, artifact_sha256, url, origin_repo,
     origin_revision, extractor, chunk_index, chunk_of}}

Provenance is never dropped: it is the only reason a later dedup decision ("source A won against
source B") can be explained, and the only way a licence obligation can be traced to the text it
covers.

Formats handled (deliberately few, deliberately explicit):
  * .txt            — blank-line separated blocks, packed into documents of at most
                      `--max-doc-chars` characters (deterministic: block order is file order)
  * .json           — CLTK-style nested dicts of strings, or a flat list of strings
  * .xhtml/.html/.xml — markup stripped with a stdlib HTMLParser (block tags become newlines)

Low-memory contract: files are processed one at a time, documents are written to disk as they are
produced, and an incremental SHA-256 is kept over the output file. Nothing accumulates in RAM.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import html
import io
import json
import sys
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training.data.acquire import load_ledger, rel  # noqa: E402
from training.data.manifest import DEFAULT_MANIFEST, load_manifest  # noqa: E402

NORMALIZED_DIR = REPO_ROOT / "data" / "normalized"
EXTRACTOR = "training/data/normalize.py"
DEFAULT_MAX_DOC_CHARS = 6_000
DEFAULT_MIN_DOC_CHARS = 200


# --------------------------------------------------------------------------------------
# markup
# --------------------------------------------------------------------------------------
BLOCK_TAGS = {
    "p", "div", "br", "li", "tr", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6",
    "section", "article", "blockquote", "pre", "figure", "figcaption", "table",
}
DROP_CONTENT_TAGS = {"script", "style", "head", "nav", "footer"}


class _TextExtractor(HTMLParser):
    """Minimal, predictable HTML/XHTML → text. No third-party dependency, no network."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._drop_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in DROP_CONTENT_TAGS:
            self._drop_depth += 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in DROP_CONTENT_TAGS and self._drop_depth:
            self._drop_depth -= 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._drop_depth:
            self.parts.append(data)

    def text(self) -> str:
        return "".join(self.parts)


def html_to_text(markup: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(markup)
        parser.close()
    except Exception:  # noqa: BLE001 — malformed markup must not kill a corpus build
        pass
    return html.unescape(parser.text())


# --------------------------------------------------------------------------------------
# readers — each yields raw text blocks in file order
# --------------------------------------------------------------------------------------
def _blocks_from_text(text: str) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if line.strip():
            buf.append(line.rstrip())
        elif buf:
            out.append("\n".join(buf))
            buf = []
    if buf:
        out.append("\n".join(buf))
    return [b for b in (s.strip() for s in out) if b]


def _flatten_json_strings(obj: Any) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for key in sorted(obj, key=lambda k: (len(str(k)), str(k))):  # stable, order-independent
            if isinstance(obj.get(key), str):
                yield obj[key]
            else:
                yield from _flatten_json_strings(obj[key])
    elif isinstance(obj, list):
        for item in obj:
            yield from _flatten_json_strings(item)


def read_blocks(path: Path) -> tuple[list[str], str]:
    """Return (blocks, format) for one artifact. Raises ValueError on unusable content."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ""}:
        return _blocks_from_text(path.read_text(encoding="utf-8", errors="replace")), "text"
    if suffix in {".xhtml", ".html", ".htm", ".xml"}:
        return _blocks_from_text(html_to_text(path.read_text(encoding="utf-8", errors="replace"))), "markup"
    if suffix == ".json":
        # a whole-document JSON read is bounded by the file size, which acquisition caps
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        text = "\n".join(_flatten_json_strings(data))
        return _blocks_from_text(text), "json"
    raise ValueError(f"unsupported artifact format: {path.suffix}")


def read_blocks_with_language(path: Path) -> tuple[list[tuple[str, str | None]], str]:
    """Like `read_blocks`, but keeps a per-record language when the artifact carries one.

    A JSON artifact that is a list of records with `text` and `language` (the SIR fixture, most
    extracted corpora) knows its own languages better than a filename ever will. Dropping that and
    labelling every chunk with the source's first language is how a corpus ends up with 57%
    "script mismatch" on clean data.
    """
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (ValueError, OSError) as exc:
            raise ValueError(f"{path.name}: unreadable JSON ({exc})") from exc
        if isinstance(data, list) and data and all(isinstance(x, dict) and "text" in x for x in data):
            out: list[tuple[str, str | None]] = []
            for rec in data:
                lang = rec.get("language")
                for piece in _blocks_from_text(str(rec.get("text") or "")):
                    out.append((piece, str(lang) if lang else None))
            return out, "json-records"
    blocks, fmt = read_blocks(path)
    return [(b, None) for b in blocks], fmt


def pack_blocks_with_language(
    pairs: Iterable[tuple[str, str | None]], *, max_doc_chars: int, min_doc_chars: int
) -> Iterator[tuple[str, str | None]]:
    """Pack blocks into documents, never mixing two declared languages in one document."""
    group: list[str] = []
    lang: str | None = None
    for text, block_lang in pairs:
        if lang is not None and block_lang != lang and group:
            for doc in pack_blocks(group, max_doc_chars=max_doc_chars, min_doc_chars=min_doc_chars):
                yield doc, lang
            group = []
        lang = block_lang
        group.append(text)
    if group:
        for doc in pack_blocks(group, max_doc_chars=max_doc_chars, min_doc_chars=min_doc_chars):
            yield doc, lang


def pack_blocks(blocks: Iterable[str], *, max_doc_chars: int, min_doc_chars: int) -> Iterator[str]:
    """Pack blocks into documents without exceeding max_doc_chars.

    Deterministic and streaming: same blocks in, same documents out. A block longer than
    max_doc_chars is hard-split at that length so no single document can blow up a later stage.
    """
    buf: list[str] = []
    size = 0
    for block in blocks:
        for piece in ([block] if len(block) <= max_doc_chars else [block[i : i + max_doc_chars] for i in range(0, len(block), max_doc_chars)]):
            if size and size + len(piece) + 2 > max_doc_chars:
                if size >= min_doc_chars:
                    yield "\n\n".join(buf)
                buf, size = [], 0
            buf.append(piece)
            size += len(piece) + 2
    if buf and size >= min_doc_chars:
        yield "\n\n".join(buf)


# --------------------------------------------------------------------------------------
# normalization
# --------------------------------------------------------------------------------------
def language_for_path(path: str, declared: list[str], language_by_path: dict[str, str] | None) -> str:
    """Assign a language label to a file. Patterns are matched in order; first match wins."""
    for pattern, lang in (language_by_path or {}).items():
        if fnmatch.fnmatch(path, pattern) or path.startswith(pattern.rstrip("*").rstrip("/") + "/"):
            return lang
    return declared[0] if declared else "unknown"


@dataclass
class NormalizeReport:
    source_id: str
    files_seen: int = 0
    files_unreadable: int = 0
    documents: int = 0
    chars: int = 0
    bytes_out: int = 0
    sha256: str = ""
    out_path: str = ""
    by_language: dict[str, int] = field(default_factory=dict)
    by_file: dict[str, int] = field(default_factory=dict)
    files_skipped_archive_container: int = 0
    files_excluded_from_corpus: int = 0
    excluded_from_corpus: list[str] = field(default_factory=list)
    policy: str = "normalize-v2"
    notes: list[str] = field(default_factory=list)
    failures: list[dict[str, str]] = field(default_factory=list)
    started_utc: str = ""
    finished_utc: str = ""

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def corpus_exclusions(source_id: str) -> list[str]:
    """Artifacts the ledger flags as excluded from the corpus (documentation, data cards)."""
    return sorted(p for p, rec in load_ledger(source_id).items() if rec.get("excluded_from_corpus"))


def normalize_source(
    source_id: str,
    *,
    out_dir: Path = NORMALIZED_DIR,
    manifest_path: Path | str = DEFAULT_MANIFEST,
    max_doc_chars: int = DEFAULT_MAX_DOC_CHARS,
    min_doc_chars: int = DEFAULT_MIN_DOC_CHARS,
    limit_files: int | None = None,
) -> NormalizeReport:
    """Normalize every artifact in the source's ledger into one JSONL file. Streams to disk."""
    manifest = load_manifest(manifest_path)
    source = manifest.get(source_id)
    if source is None:
        raise SystemExit(f"{source_id!r} is not declared in the manifest")
    ledger = load_ledger(source_id)
    if not ledger:
        raise SystemExit(
            f"{source_id}: no artifact ledger — run `python -m training.data.acquire --source {source_id}` first"
        )

    raw = source.raw
    declared = list(raw.get("languages") or [])
    lang_map = dict((raw.get("acquisition") or {}).get("language_by_path") or {})
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source_id}.jsonl"
    rep = NormalizeReport(source_id=source_id, out_path=rel(out_path), started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    h = hashlib.sha256()
    # An archive that was unpacked during acquisition is a container, not a document: its contents
    # appear in the ledger as their own artifacts. Reading the .zip itself is not an error, and
    # reporting it as one hides how many files were really processed.
    containers: set[str] = set()
    for rec in ledger.values():
        src = rec.get("extracted_from")
        if src:
            containers.add(str(src))
            containers.add(Path(str(src)).name)  # ledgers may record either the full path or the file name
    entries = sorted(ledger.items(), key=lambda kv: kv[0])
    if limit_files:
        entries = entries[:limit_files]
    with out_path.open("w", encoding="utf-8") as out:
        for artifact_path, rec in entries:
            rep.files_seen += 1
            path = REPO_ROOT / artifact_path
            if rec.get("excluded_from_corpus"):
                rep.files_excluded_from_corpus += 1
                rep.excluded_from_corpus.append(artifact_path)
                rep.notes.append(
                    f"{artifact_path}: excluded from the corpus (role={rec.get('role') or 'documentation'})"
                )
                continue
            if artifact_path in containers or Path(artifact_path).name in containers:
                rep.files_skipped_archive_container += 1
                rep.notes.append(f"{artifact_path}: archive container, contents unpacked into separate artifacts")
                continue
            if not path.exists():
                rep.files_unreadable += 1
                rep.failures.append({"artifact": artifact_path, "reason": "missing on disk"})
                continue
            try:
                pairs, fmt = read_blocks_with_language(path)
            except Exception as exc:  # noqa: BLE001 — logged, never silent
                rep.files_unreadable += 1
                rep.failures.append({"artifact": artifact_path, "reason": f"{type(exc).__name__}: {exc}"})
                continue
            manifest_lang = language_for_path(artifact_path, declared, lang_map)
            n_docs_here = 0
            for idx, (text, record_lang) in enumerate(
                pack_blocks_with_language(pairs, max_doc_chars=max_doc_chars, min_doc_chars=min_doc_chars)
            ):
                lang = record_lang or manifest_lang
                lang_source = "artifact-record" if record_lang else "manifest"
                doc_id = f"{source_id}:{Path(artifact_path).stem}:{idx:06d}"
                doc = {
                    "doc_id": doc_id,
                    "id": doc_id,  # clean_stream keys on 'id'; keep both so identity survives every stage
                    "source_id": source_id,
                    "language": lang,
                    "text": text,
                    "license": raw.get("license"),
                    "license_verified": bool(raw.get("license_verified")),
                    "license_url": raw.get("license_url"),
                    "synthetic": bool(raw.get("synthetic")),
                    "natural": bool(raw.get("natural")),
                    "share_alike": raw.get("share_alike"),
                    "attribution_required": bool(raw.get("attribution_required")),
                    "language_source": lang_source,
                    "provenance": {
                        "artifact": artifact_path,
                        "artifact_sha256": rec.get("sha256"),
                        "url": rec.get("url"),
                        "origin_repo": rec.get("origin_repo"),
                        "origin_revision": rec.get("origin_revision"),
                        "format": fmt,
                        "extractor": EXTRACTOR,
                        "chunk_index": idx,
                        "chunk_of": Path(artifact_path).name,
                    },
                }
                line = json.dumps(doc, ensure_ascii=False) + "\n"
                out.write(line)
                # hash exactly the bytes that were written (including the newline), so the
                # recorded sha256 is the file's sha256 and G7 can verify it later
                h.update(line.encode("utf-8"))
                rep.documents += 1
                n_docs_here += 1
                rep.chars += len(text)
                rep.bytes_out += len(line.encode("utf-8")) + 1
                rep.by_language[lang] = rep.by_language.get(lang, 0) + 1
            rep.by_file[artifact_path] = n_docs_here
    rep.sha256 = h.hexdigest()
    rep.finished_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out_dir / f"{source_id}.normalize_report.json").write_text(json.dumps(rep.as_dict(), indent=2) + "\n", encoding="utf-8")
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Normalize acquired raw artifacts into SIR JSONL documents.")
    ap.add_argument("--source", action="append", required=True, help="manifest source id (repeatable)")
    ap.add_argument("--out", default=str(NORMALIZED_DIR))
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--max-doc-chars", type=int, default=DEFAULT_MAX_DOC_CHARS)
    ap.add_argument("--min-doc-chars", type=int, default=DEFAULT_MIN_DOC_CHARS)
    ap.add_argument("--limit-files", type=int, default=None, help="debug: normalize only the first N artifacts")
    ap.add_argument("--as-json", action="store_true")
    args = ap.parse_args(argv)

    reports = []
    for sid in args.source:
        try:
            rep = normalize_source(
                sid,
                out_dir=Path(args.out),
                manifest_path=args.manifest,
                max_doc_chars=args.max_doc_chars,
                min_doc_chars=args.min_doc_chars,
                limit_files=args.limit_files,
            )
        except SystemExit as exc:
            print(f"{sid}: {exc}", file=sys.stderr)
            return 2
        reports.append(rep.as_dict())
        print(
            f"{sid}: {rep.documents:,} documents, {rep.chars:,} chars from {rep.files_seen} file(s)"
            + (f" [{rep.files_unreadable} unreadable]" if rep.files_unreadable else "")
        )
    if args.as_json:
        print(json.dumps({"sources": reports}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
