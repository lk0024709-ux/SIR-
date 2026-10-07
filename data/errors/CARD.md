# Error-case card — `data/errors/`

## What this directory contains

`error_cases.jsonl` — self-authored **error-learning cases** for SIR's error-based learning loop. Each
line is one mistake, its classification, why it is wrong, the correct principle, the corrected answer,
a new similar problem, and a transfer problem. The twelve-category taxonomy lives in
`development/errors.py`.

## Provenance

- **Authorship:** written inside this repository by the SIR maintainer agent on 2026-10-07. No
  third-party text, no copied problem sets, no scraped content.
- **Licence:** self-authored original text; distribution follows this repository (there is no separate
  LICENSE file yet — see `README.md`).
- **Confidence:** each case is a **draft for review**. `verification_status` is `unverified`: the prose
  has not been independently checked by a teacher, critic or human reviewer. The arithmetic and code
  claims are machine-checkable and are meant to be checked by `teachers/verify.py` before use.
- **Geography and science facts** (e.g. the Tropic of Cancer claim) must be re-checked against a cited
  source before they are used as training material.

## Rules this directory follows

1. A case that is missing any stage of the learning chain fails `ErrorCase.validate()` and cannot be
   converted into training records. Skeletons produced from failed evaluation runs stay invalid until
   a reviewer fills the missing prose.
2. Every case names `classified_by` (which verifier/critic/reviewer assigned the category) and
   `source` (where the mistake came from). A category that nobody is willing to sign is not a category.
3. Turning cases into records never upgrades `verification_status`. Self-authored and unverified means
   exactly that.
