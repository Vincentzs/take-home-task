"""Assemble per-page (components, edges) tuples into a single NetworkX graph
and serialise to GraphML / JSON / DOT, with optional SVG rendering via Graphviz.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import networkx as nx
from networkx.drawing import nx_pydot

from pipeline.types import Component, Edge, PipelineWarning


def assemble(per_page: list[tuple[list[Component], list[Edge]]]) -> nx.MultiDiGraph:
    graph: nx.MultiDiGraph = nx.MultiDiGraph()
    for components, _edges in per_page:
        for c in components:
            attrs: dict[str, str | float] = {
                "type": c.type.value,
                "page": c.page,
                "bbox": ",".join(str(v) for v in c.bbox),
            }
            for k, v in c.attrs.items():
                attrs[k] = v
            graph.add_node(c.tag, **attrs)
    for components, edges in per_page:
        present = {c.tag for c in components}
        for edge in edges:
            if edge.src_tag in present and edge.dst_tag in present:
                graph.add_edge(
                    edge.src_tag,
                    edge.dst_tag,
                    page=edge.page,
                    evidence=",".join(str(v) for v in edge.evidence),
                )
    return graph


def write_graphml(graph: nx.MultiDiGraph, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, output_path)


def write_node_link_json(graph: nx.MultiDiGraph, output_path: Path) -> None:
    """Diff-friendly node-link JSON; consumable by cytoscape.js / d3 / vis.js."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = nx.node_link_data(graph)
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_dot(graph: nx.MultiDiGraph, output_path: Path) -> None:
    """Graphviz DOT for direct rendering via `dot -Tsvg`."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx_pydot.write_dot(graph, str(output_path))


def render_svg(dot_path: Path, svg_path: Path) -> PipelineWarning | None:
    """Invoke graphviz `dot` to render DOT to SVG.

    Returns a warning (rather than raising) if the system `dot` binary is
    unavailable or fails — keeps the pipeline's "errors as warnings" contract.
    """
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            ["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)],
            capture_output=True,
            check=False,
            timeout=30,
        )
    except FileNotFoundError:
        return PipelineWarning(
            stage="graph_builder",
            page=None,
            code="dot_binary_missing",
            message="graphviz `dot` not found on PATH; skipping graph.svg",
        )
    if result.returncode != 0:
        return PipelineWarning(
            stage="graph_builder",
            page=None,
            code="dot_render_failed",
            message=result.stderr.decode("utf-8", errors="replace").strip()[:300],
        )
    return None
