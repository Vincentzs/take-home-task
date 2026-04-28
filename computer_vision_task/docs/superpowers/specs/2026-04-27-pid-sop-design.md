# P&ID → Graph + SOP Cross-Reference: Design

**Status:** Approved
**Date:** 2026-04-27
**Owner:** Vincent Zhao

---

## 1. Context

A take-home assignment to convert a Piping & Instrumentation Diagram (P&ID) into a structured graph and cross-reference that graph against a Standard Operating Procedure (SOP), flagging any inconsistencies.

The pipeline ingests a 3-page P&ID PDF (`data/p&id/diagram.pdf`) and an SOP DOCX (`data/sop/sop.docx`) and produces a NetworkX graph, a Markdown discrepancy report, annotated page images, and a structured log. It runs end-to-end with a single command, requires no GPU, ships no model weights, and is deterministic for a given input + configuration.

The chosen approach is **classical CV + OCR-anchored tag detection**: lines and shapes come from OpenCV, tag text comes from an OCR engine, component types are inferred from ISA-5.1 tag prefixes. No detector training. Cross-reference compares tag presence, inferred component type, and design pressure/temperature when annotated on the drawing.

**References**
- ISA-5.1-2009 instrument symbology, summarised in the [Kimray oil & gas P&ID reference guide](https://kimray.com/sites/default/files/uploads/training-demos/Kimray%20How%20to%20Read%20an%20Oil%20%26%20Gas%20P%26ID%20Reference%20Guide.pdf). Drives tag conventions and instrument-letter semantics.
- Provided inputs: `data/p&id/diagram.pdf`, `data/sop/sop.docx`.

---

## 2. Goals and non-goals

### Goals
- End-to-end pipeline from PDF + DOCX to graph + report with one command.
- ISA-5.1-aware tag extraction for both equipment tags (e.g. `F-715A`) and instrument balloons (e.g. `PIC`/`02`).
- Five concrete discrepancy kinds: missing in diagram, extra in diagram, type mismatch, pressure mismatch, temperature mismatch.
- Per-page parallelism and stage-level caching for fast iteration.
- `mypy --strict` clean, `ruff` clean, ≥90% unit coverage on rule-encoding modules.
- Reviewer can clone, run `make install && make run`, and see the full output.

### Non-goals
- Training new symbol detectors. (Discussed and rejected — vendor weights of unknown provenance, no labelled data.)
- Web UI, REST API, or database. (Bonus, deferred.)
- Generalising to non-ISA tag schemes.
- Recovering process flow direction. (Reliable arrow detection is out of scope.)
- Multi-sheet cross-page connections via off-page connectors.

---

## 3. Inputs and outputs

### Inputs
- `data/p&id/diagram.pdf` — 3 pages, ~7.6 MB, 11"×17" landscape.
- `data/sop/sop.docx` — Operating Limits table covering `F-715 A/B`, `V-745`, `E-742` (shell + tube), `AC-746`.
- `config.toml` (optional) — overrides defaults defined in `pipeline/config.py`.

### Outputs
```
output/
├── graph.graphml                  # NetworkX export, opens in Gephi/yEd
├── report.md                      # human-readable discrepancy report
├── pipeline.log                   # JSONL, one event per line
├── annotated/
│   ├── page_0.png
│   ├── page_1.png
│   └── page_2.png
└── intermediate/<stage>/<sha256>.json   # cache, gitignored
```

Exit code is `0` even when discrepancies > 0. Discrepancies are the product, not failures.

---

## 4. Architecture

Three architectural ideas drive the rest of the design: **pure stages**, **single side-effect boundary**, and **late join**.

```
                    ┌─────────────── Settings (typed, immutable) ──────────────┐
                    │       loaded once from config.toml + env at startup       │
                    └─────────┬───────────────────────────────────────┬─────────┘
                              │                                       │
                              ▼                                       ▼
            ┌──────────────────────────────┐         ┌──────────────────────────────┐
            │      P&ID branch (per-page)   │         │        SOP branch            │
            │                                │        │                              │
            │   pdf_loader → preprocess →    │        │  sop_parser                  │
            │   ocr → tags → components →    │        │  (docx → expected list)      │
            │   attributes → connections →   │        │                              │
            │   page_subgraph                │        │                              │
            │                                │        │                              │
            │   pure functions, run in       │        │  pure function,              │
            │   parallel across pages        │        │  single-shot                 │
            └────────────┬─────────────────┘          └──────────────┬───────────────┘
                          │ list[PageSubgraph]                       │ list[ExpectedComponent]
                          ▼                                          │
                ┌────────────────────────┐                            │
                │   graph_builder         │                           │
                │   (merge subgraphs into │                           │
                │   one NetworkX graph)   │                           │
                └────────────┬───────────┘                            │
                              │ Graph                                 │
                              ▼                                       ▼
                          ┌──────────────────────────────────────────────────┐
                          │              cross_reference (late join)         │
                          │   compares Graph against ExpectedComponent list │
                          └────────────────────────┬─────────────────────────┘
                                                    │ Report + Warnings
            ┌──────────────────────────────┐        │
            │         visualizer            │ ◀──── used by all branches for annotation
            │  (annotated PNG per page)     │
            └──────────────────────────────┘

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Side effects live ONLY in `runner.py`:
     • disk reads (PDF, DOCX, cache)
     • disk writes (intermediates, graph.graphml, report.md, annotated PNGs)
     • parallel dispatch (ProcessPoolExecutor over pages)
     • content-hash caching (skip stages whose inputs haven't changed)
     • structured logging (stdlib logging → JSON to pipeline.log)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Pure stages, side effects at the edge.** Every stage in `pipeline/stages/` is a pure function: data in, data out. The runner is the only module that touches the filesystem, spawns processes, or logs. This is what makes the rest of the codebase readable (each stage in isolation) and scalable (fan out, cache, retry, or replace any stage without ripple).

**Late join.** P&ID extraction and SOP parsing are independent until cross-reference. They share only `Settings` and the type system. Either branch can be run, debugged, and tested alone.

**Per-page fan-out.** The P&ID branch is a sequence applied to one page; pages are independent; the runner uses `ProcessPoolExecutor` to run them concurrently. 3 pages today, 30 pages tomorrow — same code, near-linear scaling.

**Properties that fall out of this shape:**

| Property | Mechanism |
|---|---|
| Readable | Each stage file is ~50–150 lines of pure logic with type hints. No I/O mixed in. |
| Scalable (more pages) | Per-page fan-out + caching. |
| Scalable (richer detection) | Stages depend on `typing.Protocol`s; OCR/detector/line-extractor implementations are swappable without touching downstream code. |

---

## 5. Module layout

```
src/pipeline/
├── __init__.py
├── cli.py                  # CLI subcommands
├── runner.py               # orchestrator: side effects, parallelism, caching
├── protocols.py            # OCREngine, LineDetector, SymbolClassifier, ...
├── types.py                # Component, TextBox, Discrepancy, BBox, ...
├── config.py               # Settings (typed) loaded from config.toml + env
├── isa_lookup.py           # ISA-5.1 letter table + abbreviation dict (frozen)
├── logging_config.py
└── stages/
    ├── __init__.py
    ├── pdf_loader.py
    ├── preprocess.py
    ├── ocr.py              # EasyOCREngine (impl of OCREngine)
    ├── tags.py
    ├── components.py
    ├── attributes.py
    ├── connections.py      # HoughLineDetector + dash filter
    ├── graph_builder.py
    ├── sop_parser.py
    ├── cross_reference.py
    └── visualizer.py
```

Stage contracts live in `protocols.py` so the orchestrator depends on interfaces, not implementations:

```python
class OCREngine(Protocol):
    def read(self, image: np.ndarray) -> list[TextBox]: ...

class LineDetector(Protocol):
    def detect(self, image: np.ndarray) -> list[LineSegment]: ...

class SymbolClassifier(Protocol):
    def classify(self, tag: str) -> ComponentType | None: ...
```

`isa_lookup.py` ships the ISA-5.1 first-letter table and the standard abbreviation dictionary (~150 entries: `PCV`, `FIC`, `LSL`, `PSV`, `ESD`, …) as frozen mappings. One source of truth, easy to extend.

---

## 6. Domain types

Frozen dataclasses with `slots=True` in `pipeline/types.py`. Importable without OpenCV/NetworkX/EasyOCR.

```python
BBox = tuple[int, int, int, int]   # (x0, y0, x1, y1) in page-pixel space

@dataclass(frozen=True, slots=True)
class PageImage:
    page_idx: int
    image: np.ndarray                # H×W or H×W×3, uint8
    dpi: int

@dataclass(frozen=True, slots=True)
class TextBox:
    text: str
    bbox: BBox
    confidence: float                # 0.0–1.0

class TagKind(StrEnum):
    EQUIPMENT = "equipment"
    INSTRUMENT = "instrument"

@dataclass(frozen=True, slots=True)
class EquipmentTag:                  # F-715A, V-745, AC-746
    raw: str
    prefix: str                      # "F", "V", "AC"
    number: int                      # 715
    suffix: str | None               # "A" or None
    bbox: BBox
    page: int

@dataclass(frozen=True, slots=True)
class InstrumentTag:                 # PIC/02 inside a balloon
    letters: str                     # "PIC"
    number: str                      # "02"
    variable: str                    # "Pressure"
    functions: tuple[str, ...]       # ("Indicator", "Controller")
    description: str                 # "Pressure Indicator Controller"
    bbox: BBox
    page: int

Tag = EquipmentTag | InstrumentTag

class ComponentType(StrEnum):
    VESSEL = "vessel"
    HEAT_EXCHANGER = "heat_exchanger"
    AIR_COOLED_EXCHANGER = "air_cooled_exchanger"
    FILTER = "filter"
    PUMP = "pump"
    TANK = "tank"
    INSTRUMENT = "instrument"
    UNKNOWN = "unknown"

@dataclass(frozen=True, slots=True)
class Component:
    tag: str                         # canonical, e.g. "F-715A"
    type: ComponentType
    page: int
    bbox: BBox                       # symbol bbox, NOT label bbox
    attrs: Mapping[str, str | float] = field(default_factory=dict)
    # attrs may include: "design_pressure_psig", "design_temp_f", "description"

@dataclass(frozen=True, slots=True)
class Edge:
    src_tag: str
    dst_tag: str
    page: int
    evidence: BBox                   # bbox of the line segment chain

@dataclass(frozen=True, slots=True)
class ExpectedComponent:
    tag: str                         # "E-742"
    type: ComponentType
    side: str | None                 # "shell" | "tube" | None
    design_pressure_psig: float | None
    design_temp_f_min: float | None
    design_temp_f_max: float | None  # SOP can carry a range, e.g. AC-746

class DiscrepancyKind(StrEnum):
    MISSING_IN_DIAGRAM = "missing_in_diagram"
    EXTRA_IN_DIAGRAM = "extra_in_diagram"
    TYPE_MISMATCH = "type_mismatch"
    PRESSURE_MISMATCH = "pressure_mismatch"
    TEMPERATURE_MISMATCH = "temperature_mismatch"

@dataclass(frozen=True, slots=True)
class Discrepancy:
    kind: DiscrepancyKind
    tag: str
    side: str | None
    expected: str | float | None
    actual: str | float | None
    page: int | None
    message: str

@dataclass(frozen=True, slots=True)
class PipelineWarning:               # non-fatal issue from any stage
    stage: str
    page: int | None
    code: str
    message: str

@dataclass(frozen=True, slots=True)
class Report:
    discrepancies: tuple[Discrepancy, ...]
    warnings: tuple[PipelineWarning, ...]
    summary: Mapping[str, int]       # counts by kind
```

---

## 7. Data flow

Each stage boundary is a typed value with a JSON-roundtrippable representation. The same shapes are used in tests, in cache files, and in intermediates.

```
─── ENTRY ───────────────────────────────────────────────────────────────────────
Settings  ← config.toml + env
inputs    ← (Path("data/p&id/diagram.pdf"), Path("data/sop/sop.docx"))

─── P&ID BRANCH (mapped over pages, parallel) ───────────────────────────────────
pdf_loader.load(pdf_path, dpi)            → list[PageImage]
                                           [cache key: sha256(pdf_bytes + dpi)]
per page:
preprocess.clean(PageImage)               → PageImage          # binarised + deskewed
ocr.read(PageImage)                       → list[TextBox]
tags.parse(list[TextBox])                 → list[Tag]
components.locate(list[Tag], PageImage)   → list[Component]
attributes.enrich(
    list[Component], list[TextBox])       → list[Component]    # design_p/t if found
connections.detect(
    PageImage, list[Component])           → list[Edge]
                       │
                       ▼
PageSubgraph = (list[Component], list[Edge], list[PipelineWarning])

─── FAN-IN ──────────────────────────────────────────────────────────────────────
graph_builder.assemble(list[PageSubgraph]) → networkx.MultiDiGraph

─── SOP BRANCH (single shot) ────────────────────────────────────────────────────
sop_parser.parse(docx_path)                → list[ExpectedComponent]
                                           [cache key: sha256(docx_bytes)]

─── LATE JOIN ───────────────────────────────────────────────────────────────────
cross_reference.compare(
    graph, list[ExpectedComponent])        → Report

─── PRESENTATION ────────────────────────────────────────────────────────────────
visualizer.render(
    PageImage, PageSubgraph, Report)       → bytes (PNG)        # per page
runner.persist(graph, report, pngs, log)   → output/
```

### Tag parsing rules

Two regex families, since equipment tags and instrument balloons use different shapes:

```
EQUIPMENT_RE        = ^[A-Z]{1,3}-\d{2,4}(\s?[A-Z])?$
INSTRUMENT_LETTERS  = ^[A-Z]{2,4}$        # paired with adjacent number box
INSTRUMENT_NUMBER   = ^\d{2,3}$
```

Equipment tag prefixes map to `ComponentType` via a small dict in `isa_lookup.py`:
`V → VESSEL`, `E → HEAT_EXCHANGER`, `AC → AIR_COOLED_EXCHANGER`, `F → FILTER`, `P → PUMP`, `T → TANK`. Unknown prefixes yield `ComponentType.UNKNOWN` with a warning.

Instrument letters decompose per ISA-5.1: first letter = measured variable, subsequent letters = function modifiers. `tags.py` produces a human-readable `description` (e.g. `"Pressure Indicator Controller"`) by consulting the ISA tables.

### Connection extraction

Skeletonise the binarised page, run probabilistic Hough line detection, then **filter dashed/dotted segments** before accepting any line as a process edge. Dash detection samples pixels along each segment and rejects lines whose black-pixel run length falls below a threshold — these are signal lines (electrical/pneumatic/hydraulic), not pipes. Per the ISA-5.1 conventions captured in the Kimray guide, only solid lines represent process piping.

### `E-742` aggregation

The SOP carries `E-742` twice — once for shell side, once for tube side, with different design limits. The drawing has one `E-742` symbol. `cross_reference` aggregates the two SOP rows into one expected node with two design-limit sub-records, and reports mismatches as e.g. `E-742 (tube): TEMPERATURE_MISMATCH …`.

### Cache key invariant

A stage's cache key is `sha256(stage_name + serialized_inputs + settings_subset_used_by_stage)`. Inputs serialise via the dataclass schemas; numpy arrays hash by their bytes. A change to the OCR confidence threshold invalidates only OCR (and downstream); a change to the dash-line threshold invalidates only `connections` (and downstream).

### Concurrency

```python
with ProcessPoolExecutor() as pool:
    page_subgraphs = list(pool.map(process_page, page_images))
```

`process_page` is a top-level pure function that runs the full per-page pipeline. The SOP branch executes on the main process while pages process. `graph_builder` waits for both.

---

## 8. Error handling

Three tiers, distinguished by who decides what to do.

### Tier 1 — Fatal (raise, log, exit non-zero)

Conditions: missing input file, malformed PDF, invalid TOML config, output dir not writable, zero-page PDF, missing required dependency, Python version mismatch.

Caught in `cli.py`, logged as a single structured `ERROR` event, single-line human-readable message printed to stderr. Full traceback goes to `pipeline.log`. `--debug` echoes the traceback to stderr.

Exit codes:
```
0   success (possibly with discrepancies)
1   unspecified fatal error
2   invalid arguments / config
3   missing or unreadable input
4   output path not writable
5   unsupported input format
```

### Tier 2 — Per-stage degraded (record `PipelineWarning`, return partial result)

Stage signature: `def stage(inputs, settings) -> tuple[Output, list[PipelineWarning]]`.

| Stage | Failure | Handling |
|---|---|---|
| `ocr` | image too dark / blank | returns `[]`, emits `PipelineWarning(code="no_text_found")` |
| `ocr` | textbox confidence < threshold | dropped silently if not a tag candidate; if it would match a tag regex, emit `PipelineWarning(code="low_confidence_tag")` |
| `tags` | regex match but ISA prefix unknown | `ComponentType.UNKNOWN`, `PipelineWarning(code="unknown_prefix")` |
| `components` | tag has no nearby symbol contour | bbox falls back to label bbox, `PipelineWarning(code="symbol_not_localized")` |
| `attributes` | numeric token nearby but no unit | not a warning; value simply unattached |
| `connections` | Hough returns 0 lines on a non-empty page | `PipelineWarning(code="no_lines_detected")`, edges empty |
| `connections` | edge between two tags but the line is dashed | filtered, no warning (correct behaviour) |
| `sop_parser` | design-limits table header not recognised | returns `[]`, `PipelineWarning(code="sop_table_not_found")` — surfaces prominently |
| `sop_parser` | numeric cell unparseable | row skipped, `PipelineWarning(code="sop_value_unparseable")` |
| per-page worker | unhandled exception in pool | runner catches, page contributes empty subgraph + `PipelineWarning(code="page_failed")` |
| per-page worker | wall-clock exceeds `Settings.page_timeout_s` (default 300) | terminated, `PipelineWarning(code="page_timeout")` |

Warnings flow `PageSubgraph.warnings` → `runner` → `Report.warnings` → "Warnings" section of `report.md`, grouped by stage with counts in the summary table.

### Tier 3 — Discrepancies (the product, not failures)

Mismatches between the graph and the SOP populate `Report.discrepancies`.

Cross-reference rules that prevent false positives:
- `PRESSURE_MISMATCH` and `TEMPERATURE_MISMATCH` fire **only when both sides have a value**. SOP value present + drawing value absent → silent.
- `EXTRA_IN_DIAGRAM` is only flagged for *equipment* tags, never instrument balloons.
- `E-742` shell vs tube aggregates to one node with two design-limit sub-records.

### Validation

`Settings` validates on load — out-of-range values raise at startup (Tier 1). Public stage functions do not re-validate well-typed inputs. Output schemas are validated once, in `runner.persist`, before writing.

### Logging

- JSONL to `output/pipeline.log`, one event per line.
- Required keys: `ts, level, stage, page, code, message, duration_ms`.
- Levels: `DEBUG` (off by default), `INFO` (stage start/end, counts), `WARNING` (Tier 2), `ERROR` (Tier 1).
- Console output is human-readable, level-coloured. JSONL for grep/jq on disk.
- `--quiet` suppresses console INFO; `--debug` enables DEBUG and tracebacks.

---

## 9. Testing

Three layers, ordered by what they protect.

### Layer 1 — Unit tests on pure-function modules

Fast (<1s total), no I/O, no model loads. Coverage target: ≥90% on each.

| Module | What's tested |
|---|---|
| `tags.py` | Equipment regex, instrument letter decomposition, ISA-5.1 first-letter parsing |
| `isa_lookup.py` | Letter table & abbreviation dict well-formed and correct against the Kimray guide |
| `sop_parser.py` | Recognises design-limits table; emits expected rows; aggregates `E-742` shell+tube; warns on unparseable cells |
| `cross_reference.py` | Each discrepancy kind fires only on the right condition; pressure/temp mismatches require both values present; instrument tags don't trigger `EXTRA_IN_DIAGRAM`; `E-742` aggregation works |
| `connections.py::is_dashed` | Dash-pattern filter on synthetic pixel rows |

Tests use hand-crafted dataclass instances and small synthetic numpy arrays — never real images.

### Layer 2 — One integration test

`tests/test_pipeline_smoke.py` runs `pipeline run` against the provided files into a `tmp_path` and asserts:
1. Artifacts exist and are non-empty: `graph.graphml`, `report.md`, `pipeline.log`, three annotated PNGs.
2. Graph contains all five SOP tags as nodes: `F-715A`, `F-715B`, `V-745`, `E-742`, `AC-746`.
3. None of those five has `ComponentType.UNKNOWN`.
4. At least one edge exists.
5. `report.md` contains sections `## Summary`, `## Discrepancies`, `## Warnings`.
6. Exit code 0 even when discrepancies > 0.

Marked `@pytest.mark.slow`. Excluded from the dev's tight loop (`pytest -m 'not slow'`); included in CI.

### Layer 3 — Static checks

- `mypy --strict` on the whole `src/pipeline/` package — catches contract drift between stages and types. Strict mode is non-negotiable; the Protocol-based architecture only pays off if types are checked.
- `ruff check` and `ruff format --check`.

### What we explicitly do NOT test

- OCR output snapshots — too brittle to model version and platform.
- Pixel-perfect annotated PNGs — assert non-empty and roughly the right size.
- Exact bbox coordinates — they shift with DPI and binarisation parameters.
- Hough line counts — implementation-detail flux.

### Fixtures

```
tests/
├── conftest.py
├── fixtures/
│   ├── sample_textboxes.json
│   ├── tiny_sop.docx
│   └── synthetic_lines.npy
├── test_tags.py
├── test_isa_lookup.py
├── test_sop_parser.py
├── test_cross_reference.py
├── test_connections_dash_filter.py
└── test_pipeline_smoke.py     # @pytest.mark.slow
```

Fixture factories (`make_component(tag="V-745", **overrides)`) keep tests readable.

---

## 10. Tooling

| Tool | Purpose |
|---|---|
| `uv` | dependency + venv management; lockfile in repo |
| `ruff` | lint + format (replaces black, isort, flake8) |
| `mypy --strict` | static typing, enforced on commit |
| `pytest` + `pytest-cov` | unit + integration testing, coverage reports |
| `pre-commit` | runs the four above before each commit |
| `Makefile` | reviewer-facing entry points |

`Makefile` targets:
```
make install     # uv sync
make lint        # ruff check + ruff format --check
make typecheck   # mypy --strict
make test        # pytest -m 'not slow'
make test-all    # pytest (includes integration)
make run         # python -m pipeline.cli run
make ci          # lint + typecheck + test-all
```

---

## 11. Risks

- **OCR quality on engineering text.** EasyOCR/PaddleOCR perform well on horizontal text but struggle with rotated tags. Mitigation: rotate-and-retry on low-confidence regions; explicit confidence threshold in `Settings`.
- **Symbol localisation accuracy.** Without trained detectors, component bbox is inferred from tag proximity to unlabelled contour regions. Mitigation: bbox falls back to the tag's bbox with a warning; cross-reference does not depend on bbox correctness.
- **Dash-pattern false positives.** Solid lines that are partially eroded by binarisation may look dashed. Mitigation: tunable threshold in `Settings`.
- **SOP format brittleness.** The current SOP uses a header-row-based design-limits table. A stricter parser would handle multiple table styles. Mitigation: explicit `PipelineWarning(code="sop_table_not_found")` makes failure visible in the report.

---

## 12. Out of scope / future work

- Web UI to browse the graph and discrepancies (bonus).
- Database for graph + report storage (bonus).
- YOLO-based symbol detection — revisit if labelled data appears.
- Connection direction inference (would need flow-arrow detection).
- Multi-sheet cross-page connections via off-page connectors.

---

## 13. Implementation sequencing

The implementation plan is the next document (writing-plans skill). Suggested ordering — rule-encoding modules first, OpenCV last:

1. Tooling: `uv`, `ruff`, `mypy`, `pytest`, `pre-commit`, `Makefile`.
2. `pipeline/types.py` + `pipeline/config.py` + `pipeline/protocols.py` — the contracts.
3. `pipeline/isa_lookup.py` + `pipeline/stages/tags.py` (with unit tests) — pure rules.
4. `pipeline/stages/sop_parser.py` (with unit tests) — independent of P&ID branch.
5. `pipeline/stages/cross_reference.py` (with unit tests) — final shape of the report.
6. `pipeline/stages/pdf_loader.py` + `preprocess.py` + `ocr.py` — image pipeline.
7. `pipeline/stages/components.py` + `attributes.py` + `connections.py` — symbol localisation and lines.
8. `pipeline/stages/graph_builder.py` + `visualizer.py`.
9. `pipeline/runner.py` + `pipeline/cli.py` — orchestrator.
10. Integration test + reviewer-facing README update.
