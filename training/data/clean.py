"""Stage 1 — cleaning: raw text to documents that are safe to deduplicate and train on.

Everything here is *pure, deterministic and auditable*: no network, no randomness, and every
dropped document is logged with a reason so that "we filtered 40% of the data" can be checked
instead of asserted. Language handling is honest about its limits — Devanagari/Latin ratios are
a script heuristic, not a language model, and romanized-Hindi detection is a low-confidence
lexicon check. Nothing here claims to "detect 22 Indian languages".
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# scripts and ranges
# ---------------------------------------------------------------------------
DEVANAGARI = (0x0900, 0x097F)
BENGALI = (0x0980, 0x09FF)
GURMUKHI = (0x0A00, 0x0A7F)
GUJARATI = (0x0A80, 0x0AFF)
ODIA = (0x0B00, 0x0B7F)
TAMIL = (0x0B80, 0x0BFF)
TELUGU = (0x0C00, 0x0C7F)
KANNADA = (0x0C80, 0x0CFF)
MALAYALAM = (0x0D00, 0x0D7F)
INDIC_RANGES = {
    "devanagari": DEVANAGARI,
    "bengali": BENGALI,
    "gurmukhi": GURMUKHI,
    "gujarati": GUJARATI,
    "odia": ODIA,
    "tamil": TAMIL,
    "telugu": TELUGU,
    "kannada": KANNADA,
    "malayalam": MALAYALAM,
}
LATIN = (0x0041, 0x024F)

ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u200e\u200f\u2060\ufeff"))
CTRL = dict.fromkeys([c for c in range(0x00, 0x20) if c not in (0x09, 0x0A, 0x0D)] + [0x7F])

RE_TAG = re.compile(r"</?(?:ref|p|br|div|span|table|tr|td|th|ul|ol|li|h[1-6]|center|small|sup|sub|nowiki|source)\b[^>]*>", re.I)
RE_REF_BLOCK = re.compile(r"<ref\b[^>]*/>|<ref\b[^>]*>.*?</ref>", re.S | re.I)
RE_TEMPLATE = re.compile(r"\{\{[^{}]{0,400}?\}\}")
RE_WIKILINK = re.compile(r"\[\[((?:[^|\]]*\|)?)([^\]]*)\]\]")
RE_MD_LINK = re.compile(r"\[([^\]]{0,120})\]\((?:[^)]*)\)")
RE_CITE = re.compile(r"\[(?:citation needed|update|when|who|cn|dubious)\]", re.I)
RE_URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
RE_FILE = re.compile(r"\b(?:File|Image|Category|Wikidata):\s*\S+", re.I)
RE_HDR = re.compile(r"^\s*={2,}\s*[^=]*\s*={2,}\s*$", re.M)
RE_MULNL = re.compile(r"\n{3,}")
RE_MULSP = re.compile(r"[ \t\u00a0]{2,}")
RE_SENT_PUNCT = re.compile(r"[.!?।|:;,\"'()\u0964\u0965\-–—\s]")


@dataclass
class CleanConfig:
    min_chars: int = 120
    max_chars: int = 100_000
    min_sentences: int = 1
    max_symbol_ratio: float = 0.35
    max_url_ratio: float = 0.15
    max_repeat_frac: float = 0.55  # most frequent line's share of total lines
    min_alnum_ratio: float = 0.35
    drop_boilerplate: tuple[str, ...] = (
        "all rights reserved",
        "click here to subscribe",
        "enable javascript",
        "404 not found",
        "this page requires cookies",
    )
    recover_mojibake: bool = True


@dataclass
class CleanReport:
    read: int = 0
    kept: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    repaired: dict[str, int] = field(default_factory=dict)
    chars_in: int = 0
    chars_out: int = 0

    def note_drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1

    def note_fix(self, kind: str, n: int = 1) -> None:
        self.repaired[kind] = self.repaired.get(kind, 0) + n

    def as_dict(self) -> dict[str, Any]:
        return {
            "read": self.read,
            "kept": self.kept,
            "dropped": dict(sorted(self.dropped.items())),
            "repaired": dict(sorted(self.repaired.items())),
            "chars_in": self.chars_in,
            "chars_out": self.chars_out,
        }


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
def script_counts(text: str) -> dict[str, int]:
    counts = {k: 0 for k in INDIC_RANGES}
    counts["latin"] = 0
    counts["other"] = 0
    for ch in text:
        o = ord(ch)
        if LATIN[0] <= o <= LATIN[1]:
            counts["latin"] += 1
        else:
            for name, (lo, hi) in INDIC_RANGES.items():
                if lo <= o <= hi:
                    counts[name] += 1
                    break
            else:
                counts["other"] += 1
    return counts


def lang_guess(text: str) -> tuple[str, str, float]:
    """Return (guess, method, confidence). Deliberately a *script* guess.

    Honest limits: 'hi' here means 'mostly Devanagari'; it does not distinguish Hindi from
    Marathi/Bhojpuri/Nepali text written in the same script. 'hinc-latn' is a low-confidence
    lexicon hit for romanized Hindi and 'en' means 'mostly Latin script'.
    """
    c = script_counts(text)
    letters = sum(c.values()) or 1
    dev = c["devanagari"] / letters
    lat = c["latin"] / letters
    other = sum(v for k, v in c.items() if k not in ("latin", "devanagari")) / letters
    if dev >= 0.4 and lat >= 0.20:
        # Devanagari document with substantial Latin-script English: mixed-script code-switching,
        # not plain Hindi. Treating it as "hi" would hide exactly the tokenisation cost we measure.
        # Threshold 0.20 (not 0.25) captures cases like "मैं अभी report finalize..." where Latin is ~24%.
        return "hinc-deva", "script-heuristic-mixed", min(0.7, 0.35 + dev * lat * 2)
    if dev >= 0.5:
        return "hi", "script-heuristic", min(0.95, 0.55 + dev)
    if other >= 0.5:
        # other dominates: could be digits/symbols/punctuation-heavy text
        # only report an Indic script if at least one Indic character was actually seen
        indic = [(k, v) for k, v in c.items() if k not in ("latin", "devanagari", "other")]
        top_name, top_val = max(indic, key=lambda kv: kv[1])
        if top_val == 0:
            return "unknown", "script-heuristic", 0.0
        return f"script:{top_name}", "script-heuristic", 0.4
    if lat >= 0.5:
        roman = romanized_hindi_score(text)
        if roman >= 0.18:
            return "hinc-latn", "latin+romanized-hindi-lexicon", min(0.6, 0.25 + roman)
        return "en", "script-heuristic", min(0.9, 0.45 + lat - roman)
    return "unknown", "script-heuristic", 0.0


# A short, deliberately conservative lexicon. High precision, low recall — it exists so that
# romanized Hindi does not get silently counted as English during the bake-off.
# Note: common English words like "the" are intentionally excluded — they would cause
# pure English sentences to be misclassified as Hinglish.
ROMAN_HI = set(
    """hai hain tha thi aur ya yaar ka ki ke ko me mein se par tak bhi nahin nahi kya kyu kyun
    kaise kabhi abhikal kal aaj ja raha ja rahi gaya gayi diya diya de do de dena le lo aa raha
    aa rahi ho raha hota hoti hua hokar samajh padh likh bol suna dekh karo kar do kar na
    bhool jaldi tension paisa rupee school college office meeting class exam notes rate
    chhota bada zyada kam bohot bahut thoda thoda accha achha bura saras mujhe tumhe tumko
    hamara unka uska iskaiskaiskaiska""".split()
)


def romanized_hindi_score(text: str) -> float:
    toks = re.findall(r"[a-zA-Z']+", text.lower())
    if not toks:
        return 0.0
    return sum(1 for t in toks if t in ROMAN_HI) / len(toks)


def try_recover_mojibake(text: str) -> str | None:
    """Undo UTF-8-decoded-as-latin-1 damage (common in scraped Indic dumps)."""
    if not re.search(r"[ÂÃàáâã][\x80-\xbf]", text):
        return None
    try:
        fixed = text.encode("latin-1", errors="strict").decode("utf-8", errors="strict")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None
    before = script_counts(text)["devanagari"]
    after = script_counts(fixed)["devanagari"]
    return fixed if after > before else None


def normalize_text(text: str, cfg: CleanConfig, report: CleanReport) -> str:
    t = unicodedata.normalize("NFC", text)
    t = t.translate(ZERO_WIDTH)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = unicodedata.normalize("NFC", t)
    if cfg.recover_mojibake:
        fixed = try_recover_mojibake(t)
        if fixed is not None:
            report.note_fix("mojibake_recovered")
            t = fixed
    t = RE_REF_BLOCK.sub(" ", t)
    t = RE_TAG.sub(" ", t)
    t = RE_TEMPLATE.sub(" ", t)
    t = RE_WIKILINK.sub(lambda m: m.group(2) or " ", t)
    t = RE_MD_LINK.sub(lambda m: m.group(1), t)
    t = RE_CITE.sub(" ", t)
    n_url = len(RE_URL.findall(t))
    if n_url:
        report.note_fix(f"urls_removed")
        t = RE_URL.sub(" ", t)
    t = RE_FILE.sub(" ", t)
    t = RE_HDR.sub("\n", t)
    t = t.translate(CTRL)
    t = t.replace("\t", " ")
    t = RE_MULSP.sub(" ", t)
    t = RE_MULNL.sub("\n\n", t)
    t = "\n".join(line.strip() for line in t.split("\n")).strip()
    return t


def quality_reject_reasons(text: str, cfg: CleanConfig) -> list[str]:
    reasons: list[str] = []
    n = len(text)
    if n < cfg.min_chars:
        reasons.append("too_short")
    if n > cfg.max_chars:
        reasons.append("too_long")
    letters_digits = sum(ch.isalnum() for ch in text)
    if n and letters_digits / n < cfg.min_alnum_ratio:
        reasons.append("low_alnum_ratio")
    lines = [l for l in text.split("\n") if l.strip()]
    if lines:
        top = max(lines.count(l) for l in set(lines)) / len(lines)
        if top > cfg.max_repeat_frac and len(lines) > 2:
            reasons.append("line_repetition")
    urlish = len(re.findall(r"[^ ]*(?:http|www\.|\.com|\.org|\.gov)[^ ]*", text, re.I))
    words = max(1, len(text.split()))
    if urlish / words > cfg.max_url_ratio:
        reasons.append("link_farm")
    symbols = sum(unicodedata.category(ch).startswith("S") for ch in text)
    if n and symbols / n > cfg.max_symbol_ratio:
        reasons.append("symbol_soup")
    if not cfg.min_sentences or RE_SENT_PUNCT.search(text) is None:
        reasons.append("no_punctuation_structure")
    low = text.lower()
    if any(b in low for b in cfg.drop_boilerplate):
        reasons.append("boilerplate")
    return reasons


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_documents(path: Path) -> Iterable[dict[str, Any]]:
    """Accept .jsonl, .json (list), or .txt (blank-line separated) — corpora arrive as all three."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such input: {path}")
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"{path}:{i}: malformed jsonl line: {e}") from e
    elif path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "documents" in data:
            data = data["documents"]
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected a list of documents or {{'documents': [...]}}")
        yield from data
    else:
        blob = path.read_text(encoding="utf-8")
        for i, block in enumerate(re.split(r"\n\s*\n", blob)):
            if block.strip():
                yield {"id": f"{path.name}:{i}", "text": block}


def clean_stream(
    docs: Iterable[dict[str, Any]], cfg: CleanConfig | None = None, source_id: str = "unknown"
) -> tuple[list[dict[str, Any]], CleanReport, list[dict[str, Any]]]:
    cfg = cfg or CleanConfig()
    rep = CleanReport()
    kept: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    for doc in docs:
        rep.read += 1
        raw = doc.get("text")
        if not isinstance(raw, str):
            rep.note_drop("missing_text_field")
            continue
        rep.chars_in += len(raw)
        text = normalize_text(raw, cfg, rep)
        reasons = quality_reject_reasons(text, cfg)
        if reasons:
            rep.note_drop(reasons[0])
            rejects.append({"id": doc.get("id"), "source_id": source_id, "reasons": reasons, "chars": len(text)})
            continue
        guess, method, conf = lang_guess(text)
        declared = doc.get("language")
        # keep the source's own label when present, but surface disagreements rather than hiding them
        if declared and declared not in ("unknown", "multi") and declared != guess:
            rep.note_fix("language_label_disagreement")
        c = script_counts(text)
        # Carry the document's own metadata through cleaning. Provenance, licence and the
        # synthetic/natural flag are *not* cleaning decisions and must not be dropped here: a
        # cleaner that returns only {text, language} silently destroys the audit trail.
        out = dict(doc)
        out.update(
            {
                "id": doc.get("id") or f"{source_id}:{rep.kept}",
                "doc_id": doc.get("doc_id") or doc.get("id") or f"{source_id}:{rep.kept}",
                "source_id": source_id,
                "language": declared or guess,
                "language_guess": guess,
                "language_method": method,
                "language_confidence": round(conf, 3),
                "text": text,
                "chars": len(text),
                "scripts": c,
            }
        )
        kept.append(out)
        rep.kept += 1
        rep.chars_out += len(text)
    return kept, rep, rejects


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 1: clean raw text into JSONL documents.")
    ap.add_argument("--in", dest="inp", required=True, help="raw .json/.jsonl/.txt (or directory)")
    ap.add_argument("--out", required=True, help="cleaned .jsonl to write")
    ap.add_argument("--source-id", default="unknown")
    ap.add_argument("--reject-log", help="write dropped-doc log here")
    ap.add_argument("--min-chars", type=int, default=CleanConfig.min_chars)
    args = ap.parse_args(argv)

    src = Path(args.inp)
    files = sorted([src] if src.is_file() else src.glob("*"))
    files = [f for f in files if f.suffix in {".json", ".jsonl", ".txt"}]
    if not files:
        print(f"no .json/.jsonl/.txt inputs found under {src}", file=__import__("sys").stderr)
        return 2
    docs: list[dict[str, Any]] = []
    for f in files:
        docs.extend(load_documents(f))
    cfg = CleanConfig(min_chars=args.min_chars)
    kept, rep, rejects = clean_stream(docs, cfg, args.source_id)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for d in kept:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    if args.reject_log:
        Path(args.reject_log).write_text(json.dumps(rejects, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"input_files": len(files), **rep.as_dict(), "out": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
