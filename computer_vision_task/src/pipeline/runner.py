"""Pipeline orchestrator. The only module with side effects.

Owns:
- disk reads (PDF, DOCX) and writes (graph.graphml, report.md, log, PNGs)
- per-page parallelism via ProcessPoolExecutor
- stage-level caching keyed by content hash (TODO in a future task; not
  required for the smoke test)
- structured logging
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from pipeline.config import Settings
from pipeline.logging_config import configure as configure_logging
from pipeline.logging_config import event
from pipeline.stages import (
    components as components_stage,
)
from pipeline.stages import (
    cross_reference,
    graph_builder,
    pdf_loader,
    preprocess,
    sop_parser,
    visualizer,
)
from pipeline.stages import (
    tags as tags_stage,
)
from pipeline.stages.attributes import enrich
from pipeline.stages.connections import detect as detect_connections
from pipeline.stages.ocr import EasyOCREngine, run_ocr
from pipeline.types import (
    Component,
    Edge,
    PageImage,
    PipelineWarning,
    Report,
)


@dataclass(frozen=True, slots=True)
class PageSubgraph:
    page: int
    components: list[Component]
    edges: list[Edge]
    warnings: list[PipelineWarning]


def _process_page(
    page_image: PageImage,
    settings: Settings,
) -> PageSubgraph:
    """Pure per-page pipeline. Suitable for ProcessPoolExecutor."""
    rgb_page, binary = preprocess.clean(page_image)
    engine = EasyOCREngine()
    text_boxes, ocr_warnings = run_ocr(rgb_page.image, engine, page_image.page_idx)
    parsed_tags, tag_warnings = tags_stage.parse_tags(
        text_boxes,
        page=page_image.page_idx,
        min_confidence=settings.ocr_confidence_threshold,
    )
    components, comp_warnings = components_stage.locate(
        parsed_tags,
        binary,
        page=page_image.page_idx,
    )
    enriched = enrich(components, text_boxes, settings.attribute_search_radius_px)
    edges, edge_warnings = detect_connections(
        binary,
        enriched,
        page=page_image.page_idx,
        dash_solid_run_min_px=settings.dash_solid_run_min_px,
    )
    return PageSubgraph(
        page=page_image.page_idx,
        components=enriched,
        edges=edges,
        warnings=ocr_warnings + tag_warnings + comp_warnings + edge_warnings,
    )


def run_pipeline(
    pdf_path: Path,
    sop_path: Path,
    settings: Settings,
) -> Report:
    """End-to-end run. Returns the final Report; persistence handled here too."""
    output_dir = settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir / "pipeline.log")

    started = time.perf_counter()
    event("runner", "pipeline starting", code="start")

    pages = pdf_loader.load(pdf_path, dpi=settings.dpi)
    event("pdf_loader", f"loaded {len(pages)} pages", code="pages_loaded")

    if settings.max_workers > 1:
        with ProcessPoolExecutor(max_workers=settings.max_workers) as pool:
            subgraphs = list(pool.map(_process_page, pages, [settings] * len(pages)))
    else:
        subgraphs = [_process_page(p, settings) for p in pages]
    for sg in subgraphs:
        event(
            "runner",
            f"page {sg.page}: {len(sg.components)} components, {len(sg.edges)} edges, "
            f"{len(sg.warnings)} warnings",
            page=sg.page,
            code="page_done",
        )

    expected, sop_warnings = sop_parser.parse_sop(sop_path)
    event("sop_parser", f"{len(expected)} expected components", code="sop_parsed")

    per_page_graphs = [(sg.components, sg.edges) for sg in subgraphs]
    graph = graph_builder.assemble(per_page_graphs)
    graph_builder.write_graphml(graph, output_dir / "graph.graphml")
    graph_builder.write_node_link_json(graph, output_dir / "graph.json")
    dot_path = output_dir / "graph.dot"
    graph_builder.write_dot(graph, dot_path)
    graph_warnings: list[PipelineWarning] = []
    svg_warning = graph_builder.render_svg(dot_path, output_dir / "graph.svg")
    if svg_warning is not None:
        graph_warnings.append(svg_warning)
    event(
        "graph_builder",
        f"graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges",
        code="graph_built",
    )

    all_components = [c for sg in subgraphs for c in sg.components]
    report = cross_reference.compare(all_components, expected)

    extra_warnings = sop_warnings + [w for sg in subgraphs for w in sg.warnings] + graph_warnings
    md = cross_reference.render_report(report, extra_warnings=extra_warnings)
    (output_dir / "report.md").write_text(md, encoding="utf-8")
    event(
        "cross_reference",
        f"{len(report.discrepancies)} discrepancies, {len(extra_warnings)} warnings",
        code="report_written",
    )

    annotated_dir = output_dir / "annotated"
    for page_image, sg in zip(pages, subgraphs, strict=True):
        out = annotated_dir / f"page_{page_image.page_idx}.png"
        visualizer.render(page_image, sg.components, sg.edges, out)
    event("visualizer", "annotated pages written", code="annotated_written")

    duration_ms = (time.perf_counter() - started) * 1000.0
    event(
        "runner",
        f"pipeline complete in {duration_ms:.0f} ms",
        code="done",
        duration_ms=duration_ms,
    )
    logging.shutdown()
    return Report(
        discrepancies=report.discrepancies,
        warnings=tuple(extra_warnings),
        summary=report.summary,
    )
