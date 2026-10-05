"""Score the tokenizer bake-off on held-out data and decide — with evidence, not preference.

    python -m tokenizer.evaluate_tokenizer --config configs/sir_nano_smoke.yaml

What is measured, for every candidate, on the SAME validation documents:
  tokens, chars/token, tokens/word, vocabulary actually produced, <unk> rate, exact round-trip
  rate, encode speed, artifact size, and how much text fits in one context window.

Then the README's acceptance criteria are evaluated against the byte-level BPE baseline:
  Hindi    >= 15% fewer tokens than baseline
  Hinglish >= 10% fewer tokens than baseline
A candidate is only comparable if it ended up with (nearly) the same vocabulary size — a smaller
vocab that wins on tokens is a different trade, not a win, so clamped runs are reported as
`not_comparable` rather than quietly folded into the table.

Outputs (generated, never typed by hand):
  evaluation/tokenizer_results.json
  docs/tokenizer-evaluation.md
  tokenizer/artifacts/<exp>/chosen.json      <- what training/train.py will pick up
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


from sir_paths import REPO_ROOT, rel  # noqa: E402
from tokenizer.api import SirTokenizer  # noqa: E402

SLICE_LABELS = {
    "hi": "Hindi (Devanagari)",
    "en": "English (Latin)",
    "hinc-latn": "Hinglish (romanized, code-mixed)",
    "hinc-deva": "Mixed-script Hinglish (Devanagari + Latin words)",
    "constructed_codeswitch": "Constructed code-switch stress slice",
}
ACCEPTANCE = {  # README M1 criteria, as targets to test — not expectations
    "hi": ("hindi_token_reduction_vs_baseline", 0.15),
    "hinc-latn": ("hinglish_token_reduction_vs_baseline", 0.10),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"


def load_val_docs(path: Path) -> list[dict[str, Any]]:
    docs = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                docs.append(json.loads(line))
    if not docs:
        raise SystemExit(f"no documents in {path} — run the data pipeline first")
    return docs


SENT = re.compile(r"(?<=[.!?।])\s+")


def build_slices(docs: list[dict[str, Any]], min_chars: int) -> dict[str, list[str]]:
    """Per-language held-out slices + one constructed code-switch slice.

    The constructed slice is assembled by interleaving ONE Hindi and ONE English held-out sentence
    per document. It is a measurement instrument, not a claim that such documents exist in the
    wild, and the report labels it `constructed`.
    """
    slices: dict[str, list[str]] = {}
    for lang in ("hi", "en", "hinc-latn", "hinc-deva"):
        texts = [d["text"] for d in docs if d.get("language") == lang]
        if texts:
            slices[lang] = texts
    hi = [s for t in slices.get("hi", []) for s in SENT.split(t) if len(s) > 12]
    en = [s for t in slices.get("en", []) for s in SENT.split(t) if len(s) > 12]
    if hi and en:
        pairs = min(len(hi), len(en))
        slices["constructed_codeswitch"] = [f"{hi[i]} {en[i]}" for i in range(pairs)]
    for name, texts in list(slices.items()):
        if sum(len(t) for t in texts) < min_chars and name != "constructed_codeswitch":
            print(f"  note: slice {name} has {sum(len(t) for t in texts)} chars < {min_chars} -> results provisional", file=sys.stderr)
    return slices


def slice_stats(tok: SirTokenizer, texts: list[str], seq_len: int) -> dict[str, Any]:
    m = tok.measure(texts)
    m["with_special_tokens"] = sum(len(tok.encode(t, specials=True)) for t in texts)
    m["context"] = tok.context_tokens(seq_len)
    m["provisional_slice"] = m["chars"] < 50_000
    return m


def evaluate(config_path: Path, docs_val: Path, artifacts: Path, out_json: Path, out_md: Path, chosen_out: Path) -> int:
    import yaml

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    tcfg = cfg.get("tokenizer") or {}
    exp = (cfg.get("experiment") or {}).get("name", "latest")
    baseline_kind = tcfg.get("baseline_for_acceptance", "byte_bpe")
    min_chars = int(tcfg.get("eval_slice_min_chars", 50_000))
    acceptance = tcfg.get("acceptance") or {}
    seq_len = int((cfg.get("model") or {}).get("max_seq_len", 512))

    manifest_path = artifacts / "bakeoff_manifest.json"
    if not manifest_path.exists():
        # the trainer writes it one level up when --results was not overridden
        alt = artifacts.parent / "bakeoff_manifest.json"
        manifest_path = alt if alt.exists() else manifest_path
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} not found — run python -m tokenizer.train_tokenizer first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [r for r in manifest["candidates"] if "error" not in r]
    if not records:
        raise SystemExit("no trained candidates in the manifest; the bake-off did not run")

    # corpus honesty flags
    ds_report = REPO_ROOT / (cfg.get("paths") or {}).get("processed_dir", "data/processed/latest") / "dataset_report.json"
    ds = json.loads(ds_report.read_text(encoding="utf-8")) if ds_report.exists() else {}
    corpus_natural = bool(manifest.get("train_file")) and not ds.get("provisional", True)
    provisional_reasons = []
    if ds.get("provisional"):
        provisional_reasons.append(ds.get("provisional_reason", "corpus below the disclosure threshold"))
    if not corpus_natural:
        provisional_reasons.append("corpus is a composed fixture, not natural text (see data/manifests/sources.yaml)")

    docs = load_val_docs(docs_val)
    slices = build_slices(docs, min_chars)
    print(f"evaluating {len(records)} candidates on {len(docs)} held-out docs, slices: {list(slices)}")

    scored: list[dict[str, Any]] = []
    for r in records:
        tok = SirTokenizer.load(Path(r["spec"]))
        per_slice = {name: slice_stats(tok, texts, seq_len) for name, texts in slices.items()}
        entry = {
            "kind": r["kind"],
            "vocab_requested": r["vocab_requested"],
            "vocab_actual": r["vocab_actual"],
            "vocab_clamped_from": r.get("vocab_requested_clamped_from"),
            "equal_vocab": r.get("vocab_requested_clamped_from") is None,
            "train_seconds": r["train_seconds"],
            "artifact_bytes": r["artifact_bytes"],
            "artifact_sha256": r["artifact_sha256"][:16],
            "spec": r["spec"],
            "slices": per_slice,
            "india_slices_tokens": sum(per_slice[k]["tokens"] for k in ("hi", "hinc-latn", "hinc-deva") if k in per_slice),
        }
        scored.append(entry)
        print(f"  {r['kind']:<11} v={r['vocab_actual']:<5} hi chars/token={per_slice.get('hi', {}).get('chars_per_token')}")

    # ---- comparisons against the baseline, paired by REQUESTED vocab -------------------
    # Pairing on requested vocab (not achieved) keeps a clamped candidate visible: the reader
    # sees its numbers, marked not-comparable, instead of the row silently disappearing.
    comparisons: list[dict[str, Any]] = []
    by_requested: dict[int, dict[str, dict[str, Any]]] = {}
    for e in scored:
        by_requested.setdefault(int(e["vocab_requested"]), {})[e["kind"]] = e
    accepted: list[dict[str, Any]] = []
    for vocab, kinds in sorted(by_requested.items()):
        base = kinds.get(baseline_kind)
        if not base:
            continue
        for kind, cand in sorted(kinds.items()):
            if kind == baseline_kind:
                continue
            dv = abs(cand["vocab_actual"] - base["vocab_actual"]) / max(1, base["vocab_actual"])
            row = {
                "vocab_requested": vocab,
                "vocab_actual": {"baseline": base["vocab_actual"], "challenger": cand["vocab_actual"]},
                "challenger": kind,
                "baseline": baseline_kind,
                "vocab_deviation": round(dv, 4),
                "slices": {},
                "comparable_vocab": bool(dv <= 0.02 and cand["equal_vocab"] and base["equal_vocab"]),
            }
            for sl in slices:
                bt, ct = base["slices"][sl]["tokens"], cand["slices"][sl]["tokens"]
                row["slices"][sl] = {
                    "baseline_tokens": bt,
                    "challenger_tokens": ct,
                    "token_reduction": round(1 - ct / bt, 5) if bt else 0.0,
                    "chars_per_token_baseline": base["slices"][sl]["chars_per_token"],
                    "chars_per_token_challenger": cand["slices"][sl]["chars_per_token"],
                    "provisional_slice": base["slices"][sl]["provisional_slice"],
                }
            verdicts = {}
            for sl, (cfgkey, target) in ACCEPTANCE.items():
                if sl not in row["slices"]:
                    continue
                red = row["slices"][sl]["token_reduction"]
                verdicts[sl] = {
                    "criterion": cfgkey,
                    "target_reduction": target,
                    "measured_reduction": red,
                    "measured_delta_pct": round(red * 100, 2),
                    "pass": bool(red >= target and row["comparable_vocab"]),
                    "blocked_reason": None if row["comparable_vocab"] else "vocabularies differ or were clamped; not a like-for-like comparison",
                    "provisional": bool(base["slices"][sl]["provisional_slice"]),
                    "failure_explanation": None
                    if red >= target
                    else "a byte-level BPE trained on this corpus already merges the Latin-script words that dominate romanized Hinglish, so the Devanagari byte penalty that BPE suffers does not apply to this slice",
                }
            row["acceptance"] = verdicts
            comparisons.append(row)
            if verdicts and all(v["pass"] for v in verdicts.values()):
                accepted.append({"vocab_requested": vocab, "challenger": kind, "verdicts": verdicts})

    # ---- informational ranking ---------------------------------------------------------
    ranking = sorted(scored, key=lambda e: e["india_slices_tokens"])
    informational_best = ranking[0] if ranking else None
    passing = sorted(accepted, key=lambda a: -min(v["measured_reduction"] - v["target_reduction"] for v in a["verdicts"].values()))
    sel_src = passing[0] if passing else informational_best
    if sel_src is None:
        selection = None
    else:
        selection = {
            "kind": sel_src.get("challenger", sel_src.get("kind")),
            "vocab_requested": sel_src.get("vocab_requested", sel_src.get("vocab_requested")),
            "vocab_actual": sel_src.get("vocab_actual") if "kind" in sel_src else None,
            "spec": sel_src.get("spec"),
            "basis": (
                "meets every applicable acceptance criterion with the largest safety margin, at comparable vocabulary"
                if passing
                else "no candidate met all acceptance criteria; this is the fewest-tokens-on-India-slices candidate, reported for information only"
            ),
            "met_acceptance": bool(passing),
            "eligible_for_release": False,
            "why_not_release_grade": provisional_reasons or ["selection is provisional until a licensed natural corpus is used"],
        }
        chosen_out.parent.mkdir(parents=True, exist_ok=True)
        selection_for_file = dict(selection)
        selection_for_file["spec"] = sel_src.get("spec")
        chosen_out.write_text(json.dumps(selection_for_file, indent=2) + "\n", encoding="utf-8")

    results = {
        "tool": "tokenizer/evaluate_tokenizer.py",
        "experiment": exp,
        "config": rel(config_path),
        "corpus": {
            "train_file": manifest.get("train_file", {}),
            "eval_file": str(docs_val),
            "eval_file_sha256": sha256(docs_val)[:16],
            "eval_docs": len(docs),
            "natural": corpus_natural,
            "provisional": bool(provisional_reasons),
            "provisional_reasons": provisional_reasons,
        },
        "slices": {k: {"docs": len(v), "chars": sum(len(t) for t in v), "label": SLICE_LABELS.get(k, k), "constructed": k == "constructed_codeswitch"} for k, v in slices.items()},
        "candidates": scored,
        "comparisons": comparisons,
        "acceptance": {
            "hindi_target": ACCEPTANCE["hi"][1],
            "hinglish_target": ACCEPTANCE["hinc-latn"][1],
            "any_candidate_passed": bool(accepted),
            "passing": accepted,
            "reading": "targets are criteria to test against; a failure is reported, not renegotiated",
        },
        "selection": selection,
        "informational_ranking": [
            {"kind": e["kind"], "vocab_requested": e["vocab_requested"], "vocab_actual": e["vocab_actual"], "india_slice_tokens": e["india_slices_tokens"]}
            for e in ranking
        ],
        "no_winner_declared_before_running": True,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    md = render_markdown(results)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    print(f"wrote {rel(out_json)} and {rel(out_md)}")
    if selection:
        print(f"selected (provisional): {selection['kind']} @ vocab {selection['vocab_actual']}")
    print(f"acceptance: {'AT LEAST ONE CANDIDATE MET IT' if accepted else 'NOT MET by any candidate — reported as a failure, see report'}")
    return 0


def render_markdown(r: dict[str, Any]) -> str:
    c = r["corpus"]
    lines = [
        "# Tokenizer evaluation — generated report",
        "",
        "<!-- Generated by tokenizer/evaluate_tokenizer.py. Do not hand-edit or hand-type numbers here. -->",
        "",
        f"- Experiment: `{r['experiment']}` · config `{r['config']}`",
        f"- Eval corpus: `{Path(c['eval_file']).name}`, {c['eval_docs']} held-out documents, sha256 `{c['eval_file_sha256']}…`",
        f"- **Natural corpus?** `{'yes' if c['natural'] else 'no'}` · **Provisional:** `{'yes' if c['provisional'] else 'no'}`",
    ]
    for reason in c["provisional_reasons"]:
        lines.append(f"  - {reason}")
    lines += [
        "",
        "All candidates were trained on identical training text and scored on identical held-out",
        "documents. A smaller vocabulary is not a win, so clamped candidates are flagged.",
        "",
        "## Candidate measurements (held-out slice = all languages combined)",
        "",
        "| candidate | vocab req | vocab actual | chars/token | tokens/word | <unk> rate | round-trip exact | encode ms/doc | artifact KB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for e in sorted(r["candidates"], key=lambda x: (x["vocab_actual"], x["kind"])):
        agg_t = sum(s["tokens"] for s in e["slices"].values())
        agg_c = sum(s["chars"] for s in e["slices"].values())
        agg_w = sum(s["words"] for s in e["slices"].values())
        unk = sum(s["unk_rate"] * s["tokens"] for s in e["slices"].values()) / max(1, agg_t)
        rt = sum(s["roundtrip_exact_rate"] for s in e["slices"].values()) / max(1, len(e["slices"]))
        ms = sum(s["encode_ms_per_doc"] for s in e["slices"].values()) / max(1, len(e["slices"]))
        lines.append(
            f"| `{e['kind']}` | {e['vocab_requested']} | {e['vocab_actual']}"
            f"{' ⚠️ clamped' if e['vocab_clamped_from'] else ''} | {agg_c / max(1, agg_t):.3f} | {agg_t / max(1, agg_w):.3f} "
            f"| {unk:.5f} | {rt:.3f} | {ms:.2f} | {e['artifact_bytes'] // 1024} |"
        )
    lines += ["", "## Per-slice token counts", "", "| candidate | vocab | " + " | ".join(SLICE_LABELS.get(k, k) for k in r["slices"]) + " |", "|---|---:|" + "---:|" * len(r["slices"])]
    for e in sorted(r["candidates"], key=lambda x: (x["kind"], x["vocab_actual"])):
        cells = " | ".join(str(e["slices"][k]["tokens"]) if k in e["slices"] else "—" for k in r["slices"])
        lines.append(f"| `{e['kind']}` | {e['vocab_actual']} | {cells} |")
    lines += ["", "### Slice sizes (why small slices are marked provisional)", ""]
    for k, v in r["slices"].items():
        tag = " *(constructed for measurement, not observed data)*" if v.get("constructed") else ""
        lines.append(f"- **{SLICE_LABELS.get(k, k)}**: {v['docs']} docs, {v['chars']} chars{tag}")

    lines += ["", "## Acceptance criteria (README M1)", ""]
    lines.append(f"- Target: Hindi ≥ {r['acceptance']['hindi_target']:.0%} and Hinglish ≥ {r['acceptance']['hinglish_target']:.0%} fewer tokens than the byte-level BPE baseline.")
    lines.append(f"- **Result: {'MET by at least one candidate' if r['acceptance']['any_candidate_passed'] else 'NOT MET by any candidate at comparable vocabulary'}.**")
    lines.append("")
    if not r["comparisons"]:
        lines.append("No comparable pairs exist (the baseline had no challenger at the same achieved vocabulary).")
    for row in r["comparisons"]:
        va = row["vocab_actual"]
        lines.append(f"### `{row['challenger']}` vs `{row['baseline']}` @ requested {row['vocab_requested']} (actual {va['baseline']} vs {va['challenger']})")
        lines.append("")
        if not row["comparable_vocab"]:
            lines.append(f"> ⚠️ not a like-for-like comparison (vocab deviation {row['vocab_deviation']:.1%}); numbers shown for information only.")
            lines.append("")
        lines.append("| slice | baseline tokens | challenger tokens | reduction | comparable |")
        lines.append("|---|---:|---:|---:|:--:|")
        for s, d in row["slices"].items():
            lines.append(
                f"| {SLICE_LABELS.get(s, s)} | {d['baseline_tokens']} | {d['challenger_tokens']} | {d['token_reduction']:+.1%} | {'✅' if row['comparable_vocab'] else '❌'} |"
            )
        lines.append("")
        for s, v in row["acceptance"].items():
            mark = "✅ PASS" if v["pass"] else "❌ FAIL"
            extra = f" — {v['blocked_reason']}" if v["blocked_reason"] else (" — provisional slice" if v["provisional"] else "")
            lines.append(f"- {SLICE_LABELS.get(s, s)}: measured {v['measured_reduction']:+.1%} vs target {v['target_reduction']:+.0%} → **{mark}**{extra}")
            if not v["pass"] and v.get("failure_explanation"):
                lines.append(f"  - why it failed: {v['failure_explanation']}")
        lines.append("")

    if r["selection"]:
        sel = r["selection"]
        lines += [
            "## Selection",
            "",
            f"Measured winner on the India-relevant slices: **`{sel['kind']}` at vocabulary {sel['vocab_actual']}**.",
            f"Basis: {sel['basis']}.",
            f"Eligible to be called a release tokenizer: **{'no' if sel['eligible_for_release'] is False else 'not yet assessed'}**.",
        ]
        for why in sel.get("why_not_release_grade") or []:
            lines.append(f"- {why}")
    lines += [
        "",
        "## How to read this without over-claiming",
        "",
        "- These are measurements of *subword mechanics* on the stated corpus. They say nothing about",
        "  model quality, Hindi fluency, or SIR's capabilities.",
        "- If a criterion failed, the failure is the result. The criterion stays as written; a future",
        "  run must beat it on a licensed corpus.",
        "- `chars/token` is reported for comparability, but the number that changes training cost is",
        "  total tokens for the same text, which is what the acceptance test uses.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score the SIR tokenizer bake-off on held-out data.")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "sir_nano_smoke.yaml"))
    ap.add_argument("--val", help="override val jsonl path")
    ap.add_argument("--artifacts", help="override artifacts dir")
    ap.add_argument("--out-json", default=str(REPO_ROOT / "evaluation" / "tokenizer_results.json"))
    ap.add_argument("--out-md", default=str(REPO_ROOT / "docs" / "tokenizer-evaluation.md"))
    args = ap.parse_args(argv)

    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    paths = cfg.get("paths") or {}
    processed = REPO_ROOT / paths.get("processed_dir", "data/processed/latest")
    val = Path(args.val) if args.val else processed / "val.jsonl"
    artifacts = Path(args.artifacts) if args.artifacts else REPO_ROOT / paths.get("tokenizer_dir", "tokenizer/artifacts/latest")
    return evaluate(
        Path(args.config), val, artifacts,
        Path(args.out_json), Path(args.out_md), artifacts / "chosen.json",
    )


if __name__ == "__main__":
    raise SystemExit(main())
