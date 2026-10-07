# Curriculum Graph Card — `sir_curriculum_v0`

## What this is

`curriculum_graph.yaml` is SIR's machine-readable curriculum graph: the prerequisite-aware
list of learning nodes that the developmental program (see `docs/human-development.md`) is
organised around. It is **content structure, not content**: a node says *what has to be
learned, what it depends on, and which topics it must cover*. It contains no textbook text,
no scraped syllabus, and no training examples — those live in datasets with their own cards.

## Authorship & provenance

- **Authored for this repository** (self-authored). No third-party text is reproduced.
- Topic sequencing is *inspired by* the general structure of Indian school and undergraduate
  education (Class 1–10 → Class 11–12 → undergraduate → advanced/research). It is **not** a
  transcription of any board's syllabus and carries no claim of alignment with NCERT, CBSE or
  any state board; no board document was copied, and none is quoted.
- No web page, textbook, or dataset was scraped to build this file.
- `provenance: self_authored` on every node records that claim explicitly, so a future node that
  comes from somewhere else cannot hide.

## License status

- The repository has **no `LICENSE` file yet** (tracked as a blocker in `README.md`). This graph
  is therefore offered under the same undecided status as the rest of the repository: no reuse
  rights are granted by this file, and no third-party licence is being asserted.
- Because it is self-authored and contains no third-party expression, it does not add licence
  risk to SIR's data position. It is *not* a training corpus.

## Honesty rules baked into the format

1. A node's existence means "this is planned curriculum", never "SIR knows this".
   Learning progress is measured separately (`curriculum/coverage.py`) from *assessment records*;
   the graph alone can never report a node as mastered.
2. Nodes for `indian_knowledge` carry an `epistemic_note` requiring textbook claims and
   traditional/mythological narrative to be separated. A test enforces that these notes exist.
3. Grade-ladder nodes are checked for continuity: a Class 7 mathematics node must depend on the
   Class 6 mathematics node. Gaps and cycles are validation errors, not warnings.

## Structure

| Field | Meaning |
|-------|---------|
| `id` | Stable dotted identifier (`math.g07`), never renamed once published |
| `title` | Human-readable node name |
| `subject` | Declared subject id (see `subjects:` block) |
| `track` | `foundation` · `bridge` · `advanced` · `meta` |
| `band` | `primary` … `research` (coarse stage used for reporting) |
| `grade` | School grade 1–10 for foundation nodes, `null` otherwise |
| `depends_on` | Prerequisite node ids (must exist; graph must be acyclic) |
| `topics` | Topic tags the node must cover; checked against `configs/curriculum_requirements.yaml` |
| `language` | `multi`, `hi`, `en`, `hinc-latn`, … (per-node default language of the material) |
| `provenance` | `self_authored` (only value currently permitted) |
| `epistemic_note` | Required for `indian_knowledge`; how to distinguish claim types |

## What this card does NOT claim

- It does not claim SIR has completed, or even started, any node.
- It does not claim the graph is a complete specification of any real curriculum.
- It does not claim pedagogical correctness; it is a dependency structure that evaluation can
  measure against, and it is expected to be revised by evidence.
