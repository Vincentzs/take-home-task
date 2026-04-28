# Computer Vision Take-Home: P&ID → Graph + SOP Cross-Reference

Converts a P&ID PDF into a NetworkX graph and cross-references it
against an SOP DOCX, emitting discrepancies, annotated page renderings,
and a structured log.

## Quick start

```bash
make install      # uv sync --extra dev (creates .venv, installs deps)
make run          # runs full pipeline on data/p&id/diagram.pdf + data/sop/sop.docx
```

The first run downloads EasyOCR weights (~100MB) to `~/.EasyOCR/`.

Outputs land in `output/`:

```
output/
├── graph.graphml          # open in Gephi/yEd or load via networkx.read_graphml
├── graph.json             # node-link JSON (cytoscape.js / d3 / vis.js / jq)
├── graph.dot              # Graphviz source; re-render with `dot -Kneato -Tsvg ...`
├── graph.svg              # rendered topology (skipped with warning if `dot` missing)
├── report.md              # human-readable discrepancy report
├── pipeline.log           # JSONL, one event per line
└── annotated/
    ├── page_0.png         # detected components + edges overlaid
    ├── page_1.png
    └── page_2.png
```

`graph.svg` is rendered by invoking the system `graphviz` binary
(`brew install graphviz` on macOS). If unavailable, the pipeline emits a
`dot_binary_missing` warning and skips the SVG — `graph.dot` and the
other outputs are unaffected.

## Approach

Classical CV + OCR-anchored tag detection. No ML model training, no
GPU required. ISA-5.1 tag conventions drive component classification
(see `src/pipeline/isa_lookup.py`). Pipeline is split into pure-function
stages composed by a thin runner (`src/pipeline/runner.py`); per-page
parallelism via `ProcessPoolExecutor`; errors flow as structured
warnings rather than exceptions.

Real-world P&IDs introduce OCR corruption — hyphens read as underscores,
digits as letters, multi-character prefixes split across two text boxes.
The tag parser (`src/pipeline/stages/tags.py`) recovers these via:
- Underscore-to-hyphen + uppercase normalisation.
- A loose digit-confusion regex (Z↔7, O↔0, I/L↔1, S↔5, B↔8) used as a
  fallback when the strict regex fails.
- Adjacency-based prefix merging for tags split across two OCR boxes
  (e.g. an isolated "F" + "715A" → "F-715A").
- Sibling-prefix inference for orphan number tokens whose prefix was
  entirely lost by OCR, anchored to a strict-match sibling tag with the
  same number on the same page.

Architecture, data shapes, error model, and testing strategy are
documented in `docs/superpowers/specs/2026-04-27-pid-sop-design.md`.

## Development

```bash
make test         # unit tests only (fast)
make test-all     # unit + integration (slow; loads OCR models)
make lint
make typecheck
make ci           # lint + typecheck + test-all
```

Stack: Python 3.12+, `uv` (deps), `ruff` (lint+format), `mypy --strict`,
`pytest`, `opencv-python`, `easyocr`, `pypdfium2`, `python-docx`,
`networkx`.

## Configuration

Defaults live in `src/pipeline/config.py`. Override via:

- `config.toml` at the project root, e.g.
  ```toml
  dpi = 400
  ocr_confidence_threshold = 0.5
  max_workers = 4
  ```
- Environment variables: `PIPELINE_DPI=400 PIPELINE_MAX_WORKERS=4 make run`.

## Assumptions

- The SOP follows the structure in `data/sop/sop.docx`: an Operating
  Limits / Design Limits table with one tag-bearing row per equipment.
- ISA-5.1 tag conventions (e.g. `V-745`, `E-742`, `AC-746`).
- Solid lines on the drawing represent process piping; dashed/dotted
  lines are signal lines and are filtered out of the graph.

## Submission

The implementation plan that produced this code is at
`docs/superpowers/plans/2026-04-27-pid-sop-pipeline.md`. The design
spec is at `docs/superpowers/specs/2026-04-27-pid-sop-design.md`.
