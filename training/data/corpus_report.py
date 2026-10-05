"""Machine-readable + human-readable corpus report, generated from a build's stats.

Nothing in this module computes a fact of its own: it rearranges `stats.json` into the tables the
sprint requires (language / documents / characters / tokens / share, source / licence / status,
split, quality, gates) and renders them as Markdown. If a number is missing it is printed as
"not measured" rather than as zero, because "0 tokens" and "not measured" mean different things.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _fmt(n: Any) -> str:
    if n is None:
        return "not measured"
    if isinstance(n, float):
        return f"{n:,.4f}" if n < 1 else f"{n:,.2f}"
    return f"{int(n):,}"


def _pct(n: Any, total: Any) -> str:
    try:
        if not total:
            return "—"
        return f"{100.0 * float(n) / float(total):.2f}%"
    except (TypeError, ValueError):
        return "—"


# --------------------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------------------
def _sources_by_status(manifest: Any) -> dict[str, Any]:
    """Every manifest entry, grouped by status — including the ones that contributed nothing."""
    counts: dict[str, int] = {}
    by_status: dict[str, list[str]] = {}
    raw_sources = getattr(manifest, "sources", {})
    entries = list(raw_sources.values()) if isinstance(raw_sources, dict) else list(raw_sources)
    for src in sorted(entries, key=lambda x: (str(x.raw.get("status")), str(x.raw.get("id")))):
        status = str(src.raw.get("status"))
        counts[status] = counts.get(status, 0) + 1
        by_status.setdefault(status, []).append(str(src.raw.get("id")))
    return {"counts": counts, "ids": by_status}


def build_report(stats: dict[str, Any], manifest: Any) -> dict[str, Any]:
    tokens = stats.get("tokens") or {}
    lang_stats = stats.get("per_language") or {}
    source_stats = stats.get("per_source") or {}
    split = stats.get("split") or {}
    m1 = stats.get("m1") or {}
    gates = stats.get("gates") or []

    token_total = int(tokens.get("total") or 0)
    docs = stats.get("documents") or {}
    chars = stats.get("characters") or {}
    synthetic_docs = int(docs.get("synthetic_documents") or 0)
    natural_docs = max(0, int(docs.get("after_dedup") or 0) - synthetic_docs)

    by_language = {
        lang: {
            "documents": int(info.get("documents") or 0),
            "characters": int(info.get("chars") or 0),
            "tokens": int((tokens.get("by_language") or {}).get(lang) or 0) if tokens.get("measured") else None,
            "share_of_documents": None,
            "share_of_tokens": None,
            "synthetic_documents": int(info.get("synthetic_documents") or 0),
            "train_documents": int(info.get("train_documents") or 0),
            "validation_documents": int(info.get("validation_documents") or 0),
            "measured_documents": int(info.get("measured_documents") or 0),
        }
        for lang, info in sorted(lang_stats.items())
    }
    total_docs = sum(v["documents"] for v in by_language.values()) or 1
    for lang, v in by_language.items():
        v["share_of_documents"] = round(v["documents"] / total_docs, 6)
        if token_total:
            v["share_of_tokens"] = round((v["tokens"] or 0) / token_total, 6)

    sources = {}
    for sid, info in sorted(source_stats.items()):
        src = manifest.get(sid)
        raw = src.raw if src else {}
        sources[sid] = {
            "documents": int(info.get("documents") or 0),
            "characters": int(info.get("chars") or 0),
            "tokens": int((tokens.get("by_source") or {}).get(sid) or 0) if tokens.get("measured") else None,
            "synthetic_documents": int(info.get("synthetic_documents") or 0),
            "train_documents": int(info.get("train_documents") or 0),
            "validation_documents": int(info.get("validation_documents") or 0),
            "license": raw.get("license"),
            "license_url": raw.get("license_url"),
            "license_verified": raw.get("license_verified"),
            "model_training_allowed": raw.get("model_training_allowed"),
            "attribution_required": raw.get("attribution_required"),
            "share_alike": raw.get("share_alike"),
            "commercial_use": raw.get("commercial_use"),
            "status": raw.get("status"),
            "provider": raw.get("provider"),
            "provenance": raw.get("provenance"),
            "url": raw.get("url"),
            "revision": (stats.get("acquisition") or {}).get(sid, {}).get("revision"),
            "synthetic": raw.get("synthetic"),
        }

    report = {
        "generated_utc": stats.get("generated_utc"),
        "experiment": (stats.get("config") or {}).get("path"),
        "corpus": {
            "documents": {
                "raw_files": docs.get("raw_files"),
                "normalized": docs.get("normalized"),
                "cleaned_kept": docs.get("cleaned_kept"),
                "after_dedup": docs.get("after_dedup"),
                "natural": natural_docs,
                "synthetic": synthetic_docs,
                "train": split.get("train_docs"),
                "validation": split.get("val_docs"),
            },
            "characters": chars,
            "tokens": {
                "measured": bool(tokens.get("measured")),
                "tool": tokens.get("tool"),
                "tokenizer_spec": tokens.get("tokenizer_spec"),
                "tokenizer_kind": tokens.get("tokenizer_kind"),
                "tokenizer_sha256": tokens.get("tokenizer_sha256"),
                "total": tokens.get("total"),
                "train": tokens.get("train"),
                "validation": tokens.get("validation"),
                "contexts_at_seq_len": tokens.get("contexts_at_seq_len"),
            },
            "per_language": by_language,
            "per_source": sources,
        },
        "split": {
            "method": split.get("method"),
            "seed": split.get("seed"),
            "val_frac_requested": split.get("val_frac_requested"),
            "val_frac_achieved": split.get("val_frac_achieved"),
            "components": split.get("components"),
            "train_docs": split.get("train_docs"),
            "val_docs": split.get("val_docs"),
            "train_chars": split.get("train_chars"),
            "val_chars": split.get("val_chars"),
            "per_language": split.get("per_language"),
            "notes": split.get("notes"),
            "leak_repair": stats.get("leak_repair"),
        },
        "quality": {
            "exact_duplicates_removed": (stats.get("dedup") or {}).get("exact", {}).get("removed"),
            "near_duplicates_removed": (stats.get("dedup") or {}).get("near", {}).get("removed"),
            "cross_source_near_duplicate_removals": (stats.get("dedup") or {}).get("near", {}).get("cross_source_removals"),
            "leakage": stats.get("validation"),
            "language_quality": stats.get("language_quality"),
            "rejects_by_reason": {
                sid: (info.get("reject_reasons") or {}) for sid, info in (stats.get("cleaning") or {}).items()
            },
            "cleaner_repairs": {
                sid: (info.get("repaired") or {}) for sid, info in (stats.get("cleaning") or {}).items()
            },
        },
        "gates": gates,
        "m1": m1,
        "reproducibility": stats.get("reproducibility"),
        "measurements": {
            "raw_documents": docs.get("normalized"),
            "clean_documents": docs.get("cleaned_kept"),
            "documents_after_dedup": docs.get("after_dedup"),
            "characters": {
                "raw": chars.get("normalized"),
                "cleaned": chars.get("cleaned"),
                "after_dedup": chars.get("after_dedup"),
            },
            "bytes_utf8_after_dedup": chars.get("bytes_utf8_after_dedup"),
            "tokens": token_total,
            "tokens_train": tokens.get("train"),
            "tokens_validation": tokens.get("validation"),
            "tokens_by_language": tokens.get("by_language"),
            "tokens_by_source": tokens.get("by_source"),
        },
        "sources_by_status": _sources_by_status(manifest),
        "sources_requested": stats.get("sources"),
        "sources_refused": stats.get("sources_refused"),
        "environment": stats.get("environment"),
    }
    return report


# --------------------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------------------
def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in r) + " |")
    return out


def render_markdown(report: dict[str, Any]) -> str:
    c = report["corpus"]
    t = c["tokens"]
    m1 = report["m1"] or {}
    docs = c["documents"]
    lines: list[str] = []
    add = lines.append

    add("# SIR — M1 corpus report")
    add("")
    add(f"*Generated {report.get('generated_utc') or 'unknown time'} by `training/data/build_corpus.py` from `stats.json`. "
        "Every number below was measured by the pipeline; nothing here is estimated or hand-entered.*")
    add("")

    add("## Headline")
    add("")
    add(f"- **M1 status: {m1.get('status', 'not evaluated')}** — {m1.get('reason', '')}")
    add(f"- **Training readiness:** {(m1.get('training_readiness') or {}).get('state', 'unknown')} — "
        f"{(m1.get('training_readiness') or {}).get('reason', '')}")
    add(f"- **Measured tokens:** {_fmt(t.get('total'))}"
        + (f" (train {_fmt(t.get('train'))} / validation {_fmt(t.get('validation'))})" if t.get("total") else ""))
    add(f"- **Documents:** {_fmt(docs.get('after_dedup'))} after dedup "
        f"({_fmt(docs.get('natural'))} natural, {_fmt(docs.get('synthetic'))} synthetic)")
    add(f"- **Characters:** {_fmt((c.get('characters') or {}).get('after_dedup'))} after dedup")
    add(f"- **Tokenizer:** {t.get('tokenizer_kind') or 'not selected'}"
        + (f" (`{t.get('tokenizer_spec')}`, sha256 {str(t.get('tokenizer_sha256'))[:12]}…)" if t.get("tokenizer_sha256") else ""))
    add("")

    add("## Languages")
    add("")
    rows = []
    for lang, v in c["per_language"].items():
        rows.append([
            lang, _fmt(v["documents"]), _fmt(v["characters"]), _fmt(v["tokens"]),
            _pct(v["documents"], docs.get("after_dedup")), v["share_of_tokens"] and f"{100 * v['share_of_tokens']:.2f}%" or "—",
            _fmt(v["synthetic_documents"]), _fmt(v["train_documents"]), _fmt(v["validation_documents"]),
        ])
    lines += _table(
        ["Language", "Documents", "Characters", "Tokens", "Share of docs", "Share of tokens", "Synthetic docs", "Train docs", "Val docs"],
        rows,
    )
    add("")

    add("## Sources")
    add("")
    rows = []
    for sid, v in c["per_source"].items():
        rows.append([
            sid, v["license"] or "—", _fmt(v["documents"]), _fmt(v["tokens"]),
            "yes" if v["license_verified"] else "**no**",
            "yes" if v["model_training_allowed"] else "**no**",
            ("yes" if v["attribution_required"] else "no"),
            ("yes" if v["share_alike"] else "no"),
            v["status"] or "—",
        ])
    lines += _table(["Source", "Licence", "Documents", "Tokens", "Licence verified", "Training allowed", "Attribution", "Share-alike", "Status"], rows)
    add("")

    add("## Split")
    add("")
    s = report["split"]
    add(f"- Method: `{s.get('method')}`, seed `{s.get('seed')}`")
    add(f"- Requested validation fraction: {_fmt(s.get('val_frac_requested'))}; achieved: {_fmt(s.get('val_frac_achieved'))}")
    add(f"- Components (shared-sentence groups that move together): {_fmt(s.get('components'))}")
    add(f"- Train: {_fmt(s.get('train_docs'))} documents / {_fmt(s.get('train_chars'))} characters; "
        f"validation: {_fmt(s.get('val_docs'))} documents / {_fmt(s.get('val_chars'))} characters")
    for note in (s.get("notes") or []):
        add(f"- note: {note}")
    repair = s.get("leak_repair") or {}
    if repair:
        add(f"- leakage repair: {repair.get('moved_documents_total', repair.get('moved_documents'))} document(s) moved from validation to train "
            f"over {len(repair.get('log') or [])} iteration(s); converged: {repair.get('converged')}")
    add("")

    add("## Quality")
    add("")
    q = report["quality"]
    add(f"- Exact duplicates removed: {_fmt(q.get('exact_duplicates_removed'))}")
    add(f"- Near duplicates removed: {_fmt(q.get('near_duplicates_removed'))} "
        f"(cross-source: {_fmt(q.get('cross_source_near_duplicate_removals'))})")
    leak = (q.get("leakage") or {}).get("leakage_8gram") or {}
    near = (q.get("leakage") or {}).get("cross_split_near_dups") or {}
    add(f"- Train/validation 8-gram leakage: {_fmt(leak.get('unique_overlapping_ngrams'))} unique "
        f"({_fmt(leak.get('overlapping_ngram_occurrences_train'))} occurrences in train), "
        f"cross-split near-duplicate pairs: {_fmt(near.get('pairs'))}")
    lq = q.get("language_quality") or {}
    add(f"- Language quality: {_fmt(lq.get('unknown'))} unknown-language and {_fmt(lq.get('script_mismatch'))} script-mismatch "
        f"of {_fmt(lq.get('documents_seen_cleaning') or lq.get('documents'))} documents the cleaner examined; "
        f"{_fmt(lq.get('label_disagreements'))} label disagreements with the source's own claim")
    if lq.get("hinc_latn_documents") or lq.get("hinc_deva_documents"):
        add(f"- Hinglish evidence: {_fmt(lq.get('hinc_latn_documents'))} romanized-Hindi (`hinc-latn`) and "
            f"{_fmt(lq.get('hinc_deva_documents'))} mixed-script (`hinc-deva`) documents; of the mixed-script ones, "
            f"{_fmt(lq.get('code_switch_documents'))} contain whole Latin words beside Devanagari words "
            "(the code-switching evidence the sprint requires). Romanized Hindi is *not* counted as code-switching.")
    det = lq.get("detector") or {}
    if det:
        add(f"- Independent detector cross-check: `{det.get('tool')}` agreed with {_fmt(det.get('agree'))} of "
            f"{_fmt(det.get('compared'))} labels ({100 * float(det.get('agreement_rate') or 0):.2f}%). "
            "The detector is a check, not the labeller (it has no Hinglish label).")
        for pair, n in (det.get("top_disagreements") or [])[:5]:
            add(f"  - disagreement: {pair} — {n} document(s)")
    add(f"- Corpus size after dedup: {_fmt((c.get('characters') or {}).get('bytes_utf8_after_dedup'))} UTF-8 bytes "
        f"({_fmt((c.get('characters') or {}).get('after_dedup'))} characters)")
    add("")

    add("## Gates")
    add("")
    rows = [[g["id"], "PASS" if g["passed"] else "**FAIL**", g["name"], g["detail"]] for g in (report.get("gates") or [])]
    lines += _table(["Gate", "Result", "Name", "Detail"], rows)
    add("")

    add("## What this corpus is not")
    add("")
    add("- It is not a claim about language quality: the language labels come from a **script + marker-lexicon heuristic** "
        "(`training/data/langid.py`). A statistical detector (py3langid, when installed) is reported as an independent "
        "cross-check in the Quality section; it never overrules the labeller.")
    add("- It is not 200M tokens unless the measured token count above says so; the M1 target is a floor, not a goal to be rounded to.")
    add("- It is not a distribution target: the language mix is whatever the legally usable sources contain.")
    add("- Synthetic documents (if any) are labelled `synthetic: true` and counted separately; they never stand in for natural text.")
    add("")

    add("## Environment and limits")
    add("")
    env = report.get("environment") or {}
    for key in ("cpu", "cpu_count", "memory_mb", "ram_mb", "python", "platform", "torch", "network", "notes"):
        if env.get(key) is not None:
            add(f"- {key}: {env.get(key)}")
    add("")
    add("## Sources considered (whole manifest, not only the corpus)")
    add("")
    sbs = report.get("sources_by_status") or {}
    for status, n in sorted((sbs.get("counts") or {}).items()):
        add(f"- **{status}**: {n} — {', '.join(sbs.get('ids', {}).get(status, [])[:12])}"
            + (" …" if len(sbs.get("ids", {}).get(status, [])) > 12 else ""))
    refused = report.get("sources_refused") or {}
    if refused:
        add("")
        add("Refused / not acquired in this build:")
        for sid, reason in sorted(refused.items()):
            add(f"- `{sid}`: {reason}")
    add("")
    add("## Reproducibility")
    add("")
    rep = report.get("reproducibility") or {}
    add(f"- Git commit: `{rep.get('git_commit')}`")
    add(f"- Config: `{(rep.get('config') or {}).get('path')}` sha256 `{str((rep.get('config') or {}).get('sha256'))[:16]}…`")
    add(f"- Seed: `{rep.get('seed')}`")
    add(f"- Source revisions pinned: {len(rep.get('source_revisions') or {})}; archive hashes recorded: "
        f"{sum(len(v) for v in (rep.get('source_archive_sha256') or {}).values())}")
    add(f"- Normalized files re-hashed on disk: {len(rep.get('normalized_files') or {})}")
    add("")
    return "\n".join(lines) + "\n"


def write_reports(report: dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
