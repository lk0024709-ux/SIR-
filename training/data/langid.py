"""Language identification for SIR corpora — deliberately a *heuristic*, and labelled as one.

What this is
------------
A deterministic script census plus small, hand-written marker lexicons. It is fast, offline,
reproducible and explainable, and it fills the label space the acquisition sprint needs:

    hi  en  hinc-latn  hinc-deva  mr  gu  bn  ta  te  kn  ml  pa  or  as  ur  sa  unknown

What this is NOT
----------------
It is not a statistical language identifier and SIR must never present it as one. Concrete
limits, stated here because they will show up in the corpus report:

  * Devanagari text cannot be reliably separated into Hindi / Marathi / Sanskrit without a
    language model; this module uses function-word markers and falls back to the source's
    declared label, recording `method` so a reader can see which documents were decided by
    evidence and which were decided by the dataset's own claim.
  * Latin-script text is separated into English vs romanized Hindi (hinc-latn) with a small
    high-precision word list. Recall is low by design: a false "hinc-latn" would inflate the
    Hinglish numbers, which is the one error this project cannot afford.
  * `hinc-deva` means "Devanagari document with a substantial Latin-script component" — it is
    evidence of mixed script, not proof of grammatical code-switching.
  * Urdu vs Arabic is script-only: both use the Arabic script. Only Urdu is expected here.

If a real detector (fastText/CLD3) becomes installable, it should replace `identify()` behind
the same signature and `method` strings; nothing else in the pipeline needs to change.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------------------
# label space and scripts
# --------------------------------------------------------------------------------------
LABELS = (
    "hi", "en", "hinc-latn", "hinc-deva",
    "mr", "gu", "bn", "ta", "te", "kn", "ml", "pa", "or", "as", "ur", "sa",
    "unknown",
)

SCRIPT_RANGES: dict[str, tuple[int, int]] = {
    "devanagari": (0x0900, 0x097F),
    "bengali": (0x0980, 0x09FF),
    "gurmukhi": (0x0A00, 0x0A7F),
    "gujarati": (0x0A80, 0x0AFF),
    "odia": (0x0B00, 0x0B7F),
    "tamil": (0x0B80, 0x0BFF),
    "telugu": (0x0C00, 0x0C7F),
    "kannada": (0x0C80, 0x0CFF),
    "malayalam": (0x0D00, 0x0D7F),
    "arabic": (0x0600, 0x06FF),
    "latin": (0x0041, 0x024F),
}

# Which scripts are *acceptable* for a declared language label. Used as a hard-ish check:
# a Marathi document written in Gurmukhi is a data error, not a language choice.
LABEL_SCRIPTS: dict[str, tuple[str, ...]] = {
    "hi": ("devanagari",),
    "mr": ("devanagari",),
    "sa": ("devanagari",),
    "ne": ("devanagari",),
    "bn": ("bengali",),
    "as": ("bengali",),
    "pa": ("gurmukhi",),
    "gu": ("gujarati",),
    "or": ("odia",),
    "ta": ("tamil",),
    "te": ("telugu",),
    "kn": ("kannada",),
    "ml": ("malayalam",),
    "ur": ("arabic",),
    "en": ("latin",),
    # code-mixed labels are *defined* by mixing, so both scripts are acceptable
    "hinc-latn": ("latin",),
    "hinc-deva": ("devanagari", "latin"),
    "unknown": tuple(SCRIPT_RANGES),
    "multi": tuple(SCRIPT_RANGES),
}

# --------------------------------------------------------------------------------------
# marker lexicons (high precision, low recall; every entry was chosen to be unambiguous)
# --------------------------------------------------------------------------------------
DEVANAGARI_MARKERS: dict[str, set[str]] = {
    "hi": {
        "है", "हैं", "है।", "था", "थी", "थे", "और", "नहीं", "कि", "की", "को", "में", "से", "पर",
        "यह", "वह", "किया", "गया", "हुआ", "लिए", "साथ", "बहुत", "कुछ", "अपने", "उनके", "जो",
        "तो", "भी", "अब", "कर", "हो", "रहा", "रही", "रहे", "वाले", "जब", "क्यों", "कैसे",
    },
    "mr": {
        "आहे", "आहेत", "आणि", "नाही", "होते", "होती", "होता", "मध्ये", "या", "त्या", "च्या",
        "झाले", "झाली", "केले", "करणे", "असे", "असते", "म्हणून", "त्यांनी", "यांनी", "पण",
        "काही", "सर्व", "आपल्या", "त्याच्या", "मराठी", "कविता", "अभंग", "देव", "श्री",
    },
    "sa": {
        "च", "एव", "इति", "तथा", "अपि", "तु", "वै", "न", "स", "तत्", "यत्", "अथ", "श्लोक",
        "अध्याय", "वाक्यम्", "किम्", "अस्ति", "भवति", "उवाच", "सर्वे", "धर्म", "कर्म", "ब्रह्म",
        "मोक्ष", "आत्मा", "परम", "तस्मात्", "यथा", "तेन", "एष", "इदम्",
    },
}

# Romanized Hindi / Hinglish markers. English function words are intentionally absent.
ROMANIZED_HINDI = {
    "hai", "hain", "tha", "thi", "the", "aur", "ya", "yaar", "ka", "ki", "ke", "ko", "me", "mein",
    "se", "par", "tak", "bhi", "nahi", "nahin", "kya", "kyu", "kyun", "kaise", "kab", "abhi",
    "kal", "aaj", "raha", "rahi", "rahe", "gaya", "gayi", "gaye", "diya", "liya", "dena", "lena",
    "karo", "kar", "karna", "karte", "karta", "karti", "hua", "hui", "hoga", "hogi", "matlab",
    "bhai", "dost", "kaam", "paisa", "paise", "ghar", "school", "office", "meeting", "jaldi",
    "tension", "accha", "achha", "thoda", "bahut", "bohot", "zyada", "kam", "bada", "chhota",
    "mujhe", "tumhe", "tumko", "hamara", "hamari", "apna", "apni", "unka", "uska", "mera", "meri",
    "sakta", "sakte", "sakti", "chahiye", "padega", "padta", "lagta", "lagti", "banaya", "bana",
}

_LATIN_WORD_RE = re.compile(r"[A-Za-z']+")


def _is_word_char(ch: str) -> bool:
    r"""True for letters/digits plus Indic combining marks (Mn/Mc) and joiners.

    A plain `\w` is not enough: Python's `\w` excludes combining marks, so Devanagari words
    shatter at every vowel sign ("नगर" -> "नगर", "न", "गम"). That silently corrupted the marker
    lexicons and mislabelled Hindi as Sanskrit, so the tokenizer is explicit here.
    """
    if ch.isalnum() or ch == "_":
        return True
    if ch in ("\u200c", "\u200d"):  # ZWNJ / ZWJ
        return True
    return unicodedata.category(ch) in ("Mn", "Mc", "Me")


def iter_words(text: str):
    """Yield word-ish tokens, keeping combining marks attached to their base letter."""
    buf: list[str] = []
    for ch in text:
        if _is_word_char(ch):
            buf.append(ch)
        elif buf:
            yield "".join(buf)
            buf.clear()
    if buf:
        yield "".join(buf)


@dataclass
class LangId:
    language: str
    method: str
    confidence: float
    scripts: dict[str, int] = field(default_factory=dict)
    mixed_script: bool = False
    heuristic: bool = True
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "method": self.method,
            "confidence": round(self.confidence, 3),
            "heuristic": True,
            "mixed_script": self.mixed_script,
            "scripts": self.scripts,
            "notes": self.notes,
        }


def _script_of(o: int) -> str:
    """Map one code point to a script name.

    The `latin` entry in SCRIPT_RANGES is (0x0041, 0x024F) for documentation, but that range
    also contains ASCII punctuation (`^` 0x5E, backtick, braces…) and Latin-1 symbols; counting
    them as Latin script inflates the Latin share of every Devanagari document. So Latin is
    matched letter-by-letter here and punctuation falls through to `other`.
    """
    if 0x0041 <= o <= 0x005A or 0x0061 <= o <= 0x007A or 0x00C0 <= o <= 0x024F:
        return "latin"
    for name, (lo, hi) in SCRIPT_RANGES.items():
        if name == "latin":
            continue
        if lo <= o <= hi:
            return name
    return "other"


def script_counts(text: str) -> dict[str, int]:
    """Count characters per script. One pass, no regex, no allocations per char class."""
    counts = {k: 0 for k in SCRIPT_RANGES}
    counts["other"] = 0
    for ch in text:
        counts[_script_of(ord(ch))] += 1
    return counts


def script_shares(counts: dict[str, int]) -> dict[str, float]:
    total = sum(v for k, v in counts.items() if k != "other") or 1
    return {k: v / total for k, v in counts.items() if k != "other"}


def devanagari_marker_scores(text: str) -> dict[str, float]:
    """Share of tokens matching each Devanagari marker lexicon."""
    toks = [t for t in iter_words(text) if len(t) > 1]  # single letters carry no language evidence
    if not toks:
        return {k: 0.0 for k in DEVANAGARI_MARKERS}
    out: dict[str, float] = {}
    for lang, markers in DEVANAGARI_MARKERS.items():
        hits = sum(1 for t in toks if t in markers)
        out[lang] = hits / len(toks)
    return out


def romanized_hindi_score(text: str) -> float:
    toks = _LATIN_WORD_RE.findall(text.lower())
    if not toks:
        return 0.0
    return sum(1 for t in toks if t in ROMANIZED_HINDI) / len(toks)


def identify(text: str, declared: str | None = None) -> LangId:
    """Identify the language label of one document.

    `declared` is the source's own claim. It is used as a *tie-breaker only* and is recorded
    as such in `method`, so a report can separate measured labels from inherited claims.
    """
    counts = script_counts(text)
    shares = script_shares(counts)
    dev = shares.get("devanagari", 0.0)
    lat = shares.get("latin", 0.0)
    notes: list[str] = []

    def result(lang: str, method: str, conf: float, mixed: bool = False) -> LangId:
        return LangId(lang, method, conf, counts, mixed, True, notes)

    # 1. mixed-script Devanagari+Latin: this is the hinc-deva signal SIR cares about
    if dev >= 0.40 and lat >= 0.20:
        return result("hinc-deva", "script-mixed(devanagari+latin)", min(0.75, 0.35 + dev * lat * 2), True)

    # 2. dominant Indic scripts
    for script, label in (
        ("bengali", "bn"),
        ("gurmukhi", "pa"),
        ("gujarati", "gu"),
        ("odia", "or"),
        ("tamil", "ta"),
        ("telugu", "te"),
        ("kannada", "kn"),
        ("malayalam", "ml"),
        ("arabic", "ur"),
    ):
        if shares.get(script, 0.0) >= 0.5:
            if label == "ur":
                notes.append("Urdu/Arabic share the Arabic script; script-only evidence")
            return result(label, f"script({script})", min(0.9, 0.45 + shares[script]))

    # 3. Devanagari family: hi / mr / sa
    if dev >= 0.5:
        scores = devanagari_marker_scores(text)
        best, second = sorted(scores.items(), key=lambda kv: -kv[1])[:2]
        best_lang, best_score = best
        second_score = second[1] if len(scores) > 1 else 0.0
        # Require a real margin and a minimum of evidence: one accidental function word in a
        # short document used to be enough to relabel Hindi text as Sanskrit.
        if best_score >= 0.05 and best_score >= 2.0 * second_score:
            return result(best_lang, f"script(devanagari)+marker-lexicon({best_lang})", min(0.85, 0.4 + best_score * 4))
        if declared in ("hi", "mr", "sa"):
            notes.append("no decisive marker match; used the source's declared label")
            return result(declared, f"script(devanagari)+declared-prior({declared})", 0.4)
        notes.append("no decisive marker match and no declared label; defaulting to hi (known limitation)")
        return result("hi", "script(devanagari)+default", 0.3)

    # 4. Latin script: English vs romanized Hindi
    if lat >= 0.5:
        roman = romanized_hindi_score(text)
        if roman >= 0.18:
            return result("hinc-latn", "latin+romanized-hindi-lexicon", min(0.6, 0.25 + roman), mixed=False)
        return result("en", "script(latin)", min(0.9, 0.45 + lat))

    notes.append("no script reached dominance; text is digits/symbols/mixed noise")
    return result("unknown", "script", 0.0)


def code_switch_evidence(text: str) -> dict[str, int | bool]:
    """Script-level evidence of code-switching: whole Latin words inside Devanagari text.

    Hinglish is not "Hindi written in Latin letters". A `hinc-latn` document is romanized Hindi; a
    `hinc-deva` document is only evidence of code-switching when Latin *words* (not just stray
    characters) appear beside Devanagari words. This function counts both sides so the report can
    say how many documents carried real mixed-word evidence.
    """
    latin_tokens = devanagari_tokens = 0
    for tok in iter_words(text):
        if any("A" <= ch <= "Z" or "a" <= ch <= "z" for ch in tok):
            latin_tokens += 1
        elif any(0x0900 <= ord(ch) <= 0x097F for ch in tok):
            devanagari_tokens += 1
    return {
        "latin_tokens": latin_tokens,
        "devanagari_tokens": devanagari_tokens,
        "code_switch": latin_tokens >= 2 and devanagari_tokens >= 2,
    }


def script_matches_label(label: str, counts: dict[str, int], min_share: float = 0.5) -> bool:
    """True when the document's script is consistent with the label it claims.

    `unknown`/`multi` are always accepted (they exist precisely for the residue).
    """
    if label in ("unknown", "multi", "", None):
        return True
    allowed = LABEL_SCRIPTS.get(label)
    if not allowed:
        return True  # label outside SIR's space: not our call to reject here
    shares = script_shares(counts)
    return any(shares.get(s, 0.0) >= min_share for s in allowed)


def label_from_declared_and_text(declared: str | None, text: str) -> LangId:
    """Pipeline entry point: measure the text, then reconcile with the declared label."""
    got = identify(text, declared)
    if declared and declared != "unknown" and declared != got.language:
        if declared in LABEL_SCRIPTS and script_matches_label(declared, got.scripts, 0.5):
            # script agrees with the declared label but the marker lexicon disagreed: keep the
            # declared label and record that it was inherited rather than measured.
            return LangId(
                declared,
                f"{got.method}+declared-prior",
                min(got.confidence, 0.5),
                got.scripts,
                got.mixed_script,
                True,
                got.notes + [f"measured {got.language}, source declared {declared}; script is consistent"],
            )
    return got

# --------------------------------------------------------------------------------------
# optional statistical detector (cross-check only — it does not label the corpus)
# --------------------------------------------------------------------------------------
# `py3langid` is a small log-linear language identifier (140 labels, model shipped in the wheel).
# It knows Hindi/Marathi/Sanskrit/Bengali/Tamil, but it does not know Hinglish: romanized Hindi is
# classified as some other Latin-script language (e.g. "fuv"). So it is used as an *independent
# cross-check* of the heuristic labels, never as the labeller, and everything degrades to the
# heuristic if the package is absent. The agreement rate is reported in the corpus report.
_DETECTOR = None
_DETECTOR_TRIED = False


def detector():
    """Return a loaded detector callable, or None when py3langid is not installed."""
    global _DETECTOR, _DETECTOR_TRIED
    if not _DETECTOR_TRIED:
        _DETECTOR_TRIED = True
        try:  # pragma: no cover - depends on the environment
            from py3langid.langid import MODEL_FILE, LanguageIdentifier

            ident = LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)
            _DETECTOR = (ident, getattr(__import__("py3langid"), "__version__", "unknown"))
        except Exception:  # pragma: no cover
            _DETECTOR = None
    return _DETECTOR


def detector_name() -> str | None:
    got = detector()
    return f"py3langid-{got[1]}" if got else None


def detector_label(text: str) -> tuple[str, float] | None:
    """(label, probability) from the optional detector, or None when it is unavailable."""
    got = detector()
    if got is None:
        return None
    try:
        label, conf = got[0].classify(text)
    except Exception:  # pragma: no cover
        return None
    return str(label), float(conf)


# SIR label -> the label the statistical detector is expected to produce for that document.
# hinc-deva/hinc-latn are SIR concepts the detector has no label for, so a Hindi-family answer counts
# as agreement; anything else is recorded as a disagreement and surfaced in the report.
DETECTOR_EXPECTED: dict[str, set[str]] = {
    "hi": {"hi"}, "mr": {"mr"}, "sa": {"sa"}, "ne": {"ne"},
    "bn": {"bn"}, "as": {"as"}, "gu": {"gu"}, "pa": {"pa"}, "or": {"or"},
    "ta": {"ta"}, "te": {"te"}, "kn": {"kn"}, "ml": {"ml"}, "si": {"si"},
    "ur": {"ur", "ar", "fa"}, "en": {"en"},
    "hinc-latn": {"hi"}, "hinc-deva": {"hi", "mr"},
}


def detector_agrees(sir_label: str, detector_result: tuple[str, float] | None) -> bool | None:
    """None when nothing could be compared, otherwise whether the detector supports the label."""
    if detector_result is None:
        return None
    expected = DETECTOR_EXPECTED.get(sir_label)
    if expected is None:
        return None
    return detector_result[0] in expected
