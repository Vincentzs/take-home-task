"""End-to-end smoke test. Loads OCR models; marked slow."""

from pathlib import Path

import networkx as nx
import pytest

from pipeline.config import Settings
from pipeline.runner import run_pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
PDF = REPO_ROOT / "data" / "p&id" / "diagram.pdf"
SOP = REPO_ROOT / "data" / "sop" / "sop.docx"
SOP_TAGS = {"F-715A", "F-715B", "V-745", "E-742", "AC-746"}


@pytest.mark.slow
def test_pipeline_runs_end_to_end_on_sample(tmp_path: Path) -> None:
    if not PDF.exists() or not SOP.exists():
        pytest.skip("sample data not present")

    settings = Settings(dpi=200, max_workers=1, output_dir=tmp_path)
    report = run_pipeline(PDF, SOP, settings)

    assert (tmp_path / "graph.graphml").stat().st_size > 0
    assert (tmp_path / "report.md").stat().st_size > 0
    assert (tmp_path / "pipeline.log").stat().st_size > 0
    for i in range(3):
        png = tmp_path / "annotated" / f"page_{i}.png"
        assert png.exists() and png.stat().st_size > 0

    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "## Summary" in md
    assert "## Discrepancies" in md
    assert "## Warnings" in md

    graph = nx.read_graphml(tmp_path / "graph.graphml")
    nodes = set(graph.nodes())
    missing = SOP_TAGS - nodes
    assert not missing, f"SOP-referenced tags missing from graph: {missing}"

    for tag in SOP_TAGS:
        node = graph.nodes[tag]
        assert node["type"] != "unknown", f"{tag} classified as UNKNOWN"

    assert graph.number_of_edges() >= 1, "expected at least one edge"
    # Discrepancies are allowed; the test passes regardless of count.
    assert isinstance(report.summary, dict)
