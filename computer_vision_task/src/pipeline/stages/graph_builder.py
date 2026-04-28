"""Assemble per-page (components, edges) tuples into a single NetworkX graph
and serialise to GraphML.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from pipeline.types import Component, Edge


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
