from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from pipeline.stages.graph_builder import (
    assemble,
    render_svg,
    write_dot,
    write_node_link_json,
)
from pipeline.types import Component, ComponentType, Edge


def _component(tag: str, type_: ComponentType = ComponentType.VESSEL) -> Component:
    return Component(tag=tag, type=type_, page=0, bbox=(0, 0, 10, 10))


def _edge(src: str, dst: str) -> Edge:
    return Edge(src_tag=src, dst_tag=dst, page=0, evidence=(0, 0, 10, 10))


def test_write_node_link_json_round_trips(tmp_path: Path) -> None:
    graph = assemble([([_component("V-745"), _component("P-1")], [_edge("V-745", "P-1")])])
    out = tmp_path / "graph.json"
    write_node_link_json(graph, out)

    data = json.loads(out.read_text(encoding="utf-8"))
    node_ids = {n["id"] for n in data["nodes"]}
    assert node_ids == {"V-745", "P-1"}
    assert data["directed"] is True
    assert data["multigraph"] is True

    edges_key = "links" if "links" in data else "edges"
    edge_pairs = {(e["source"], e["target"]) for e in data[edges_key]}
    assert edge_pairs == {("V-745", "P-1")}


def test_write_dot_produces_nonempty_file(tmp_path: Path) -> None:
    graph = assemble([([_component("V-745"), _component("P-1")], [_edge("V-745", "P-1")])])
    out = tmp_path / "graph.dot"
    write_dot(graph, out)
    assert out.stat().st_size > 0
    text = out.read_text(encoding="utf-8")
    assert '"V-745"' in text
    assert '"P-1"' in text


def test_render_svg_warns_when_dot_binary_missing(tmp_path: Path) -> None:
    dot = tmp_path / "graph.dot"
    dot.write_text("digraph G {}\n", encoding="utf-8")
    svg = tmp_path / "graph.svg"

    with patch("pipeline.stages.graph_builder.subprocess.run", side_effect=FileNotFoundError):
        warning = render_svg(dot, svg)

    assert warning is not None
    assert warning.code == "dot_binary_missing"
    assert warning.stage == "graph_builder"
    assert not svg.exists()
