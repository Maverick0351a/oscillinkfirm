from __future__ import annotations

from pathlib import Path

from oscillink.ingest import build_jsonl_index, chunk_paragraphs, extract_text, load_embedding_model


def test_jsonl_index_determinism(tmp_path: Path) -> None:
    # Prepare a tiny deterministic input
    doc = tmp_path / "doc.txt"
    doc.write_text("A\n\nB\n\nC", encoding="utf-8")

    # Run extract -> chunk -> embed -> index twice and compare digests
    ex1 = extract_text([doc])
    pages1 = [p for r in ex1 for p in r.pages]
    ch1 = chunk_paragraphs(pages1)

    ex2 = extract_text([doc])
    pages2 = [p for r in ex2 for p in r.pages]
    ch2 = chunk_paragraphs(pages2)

    assert [ (c.start, c.end) for c in ch1.chunks ] == [ (c.start, c.end) for c in ch2.chunks ]

    model = load_embedding_model("bge-small-en-v1.5")
    vecs1 = model.embed([c.text for c in ch1.chunks])
    vecs2 = model.embed([c.text for c in ch2.chunks])
    assert vecs1 == vecs2

    idx1 = build_jsonl_index(ch1.chunks, vecs1, out_path=str(tmp_path / "idx.jsonl"))
    idx2 = build_jsonl_index(ch2.chunks, vecs2, out_path=str(tmp_path / "idx.jsonl"))

    assert idx1.index_sha256 == idx2.index_sha256
