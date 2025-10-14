from __future__ import annotations

from pathlib import Path

from oscillink.adapters.recall import recall_and_settle_jsonl
from oscillink.ingest import (
    build_jsonl_index,
    chunk_paragraphs,
    extract_text,
    load_embedding_model,
)


def test_docx_heading_metadata(tmp_path: Path) -> None:
    # Simulate DOCX-extracted text with markdown headings
    p = tmp_path / "doc.md"
    p.write_text("# Title\n\nIntro para.\n\n## Section A\n\nPara A1.\n\nPara A2.", encoding="utf-8")
    ex = extract_text([p])
    pages = [pg for r in ex for pg in r.pages]
    ch = chunk_paragraphs(pages)
    # First chunk under Title
    assert any(c.meta and c.meta.get("section_title") == "Title" for c in ch.chunks)
    # Later chunks under Section A
    assert any(c.meta and c.meta.get("section_title") == "Section A" for c in ch.chunks)


def test_csv_row_metadata_and_ingest_receipt(tmp_path: Path) -> None:
    # Create simple CSV with headers and two rows
    src = tmp_path / "rows.csv"
    src.write_text("id,name\n1,Alice\n2,Bob\n", encoding="utf-8")
    ex = extract_text([src], parser="auto")
    pages = [pg for r in ex for pg in r.pages]
    # Ensure row-index metadata attached
    assert len(pages) == 2
    assert all(pg.meta and "row_index" in pg.meta for pg in pages)
    # Chunk -> embed -> index
    ch = chunk_paragraphs(pages)
    model = load_embedding_model("bge-small-en-v1.5")
    vecs = model.embed([c.text for c in ch.chunks])
    built = build_jsonl_index(ch.chunks, vecs, out_path=str(tmp_path / "idx.jsonl"))
    # Ingest sidecar is created by CLI; here we only check query-time settle attaches parent_ingest_sig
    # Emulate settle to validate parent linkage presence in receipt meta
    bundle, settle = recall_and_settle_jsonl(built.index_path, model.embed(["q"])[0])
    assert isinstance(bundle, list)
    assert isinstance(settle, dict)
    # parent_ingest_sig in settle.meta is optional here since CLI persists it; ensure presence of candidate_set_hash
    assert settle.get("meta", {}).get("candidate_set_hash")

