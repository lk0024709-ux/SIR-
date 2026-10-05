# Data card: `sir_fixture_v0`

| Field | Value |
|-------|-------|
| License | CC0-1.0 |
| Authorship | authored for the SIR repository; seed sentences are human-written, documents are composed |
| Natural corpus? | **No** — composed from 153 seed sentences by `scripts/build_fixture_corpus.py` (seed 20261005) |
| Documents | 551 |
| Characters | 188908 (measured) |
| Languages | {'hi': 207, 'en': 132, 'hinc-latn': 132, 'hinc-deva': 79, 'unknown': 1} |
| Dirty/markup samples | 8 |
| sha256(docs.json) | `9ba4e52a8117e2bb41b814d156046f780f6c6470da407a106de5f01766c2b6b8` |
| Built | 2026-10-05T11:54:48Z |

## What it is for
Exercising every stage that a real corpus must pass — cleaning, dedup, near-dup, splitting,
leakage measurement, tokenizer bake-off plumbing, one-step and short training runs, and CPU
inference — **without network access and without any licensing risk**.

## What it is not
It is not a Hindi, English, or Hinglish corpus for training a model worth releasing. Documents
repeat a small sentence pool, so:

- token-efficiency numbers are biased toward whatever merges the tiny training slice supports;
- perplexity is not comparable to any external number;
- generation quality is meaningless as a capability statement.

Scripts therefore attach `provisional: true` to results computed on this fixture, and SIR's
README must never quote a number that came only from here as a capability claim.

## Deliberate flaws included (so the pipeline cannot pass by ignoring dirt)
markup/`{{update}}` templates, `<ref>`/`<p>` tags, bare URLs, mojibake (`à¤®à¥‡à¤‚ …`),
zero-width spaces inside words, CRLF/trailing whitespace, exact + near + extended duplicates,
and one repeated-sentence document.

## Rebuilding
```bash
python -m scripts.build_fixture_corpus --seed 20261005   # byte-identical output
```
