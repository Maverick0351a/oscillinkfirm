from __future__ import annotations

from pathlib import Path

from oscillink.ingest import (
    RecallParams,
    build_jsonl_index,
    chunk_paragraphs,
    extract_text,
    load_embedding_model,
    recall_and_settle_jsonl,
)


def test_recall_settle_jsonl_determinism(tmp_path: Path) -> None:
    # Create tiny corpus
    doc = tmp_path / "doc.txt"
    doc.write_text("Alpha\n\nBeta\n\nGamma", encoding="utf-8")

    # Build index deterministically
    ex = extract_text([doc])
    pages = [p for r in ex for p in r.pages]
    ch = chunk_paragraphs(pages)
    model = load_embedding_model("bge-small-en-v1.5")
    vecs = model.embed([c.text for c in ch.chunks])
    idx = build_jsonl_index(ch.chunks, vecs, out_path=str(tmp_path / "idx.jsonl"))

    # Query twice and compare bundles + key receipt meta
    qvec = model.embed(["Beta"])[0]
    params = RecallParams(kneighbors=3, bundle_k=3, tol=1e-3)
    b1, r1 = recall_and_settle_jsonl(idx.index_path, qvec, params=params)
    b2, r2 = recall_and_settle_jsonl(idx.index_path, qvec, params=params)

    assert [x["id"] for x in b1] == [x["id"] for x in b2]
    assert r1.get("meta", {}).get("candidate_set_hash") == r2.get("meta", {}).get("candidate_set_hash")
