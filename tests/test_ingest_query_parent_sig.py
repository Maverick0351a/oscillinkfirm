from __future__ import annotations

from pathlib import Path

from oscillink.adapters.recall import RecallParams, recall_and_settle_jsonl
from oscillink.ingest import (
    build_jsonl_index,
    chunk_paragraphs,
    extract_text,
    load_embedding_model,
)


def test_parent_ingest_sig_roundtrip(tmp_path: Path) -> None:
    # Minimal synthetic input content
    src = tmp_path / "doc.txt"
    src.write_text("Hello world.\nThis is Oscillink.\n", encoding="utf-8")

    # Deterministic ingest (extract -> chunk -> embed -> index)
    ex = extract_text([str(src)])
    pages = [p for r in ex for p in r.pages]
    ch = chunk_paragraphs(pages)
    model = load_embedding_model("bge-small-en-v1.5")
    vecs = model.embed([c.text for c in ch.chunks])
    idx = build_jsonl_index(ch.chunks, vecs, out_path=str(tmp_path / "index.jsonl"))

    # Query end-to-end and verify parent_ingest_sig appears in receipt
    qvec = model.embed(["hello oscillink"])[0]
    params = RecallParams(kneighbors=3, bundle_k=3)
    _bundle, receipt = recall_and_settle_jsonl(idx.index_path, qvec, params=params)
    assert isinstance(receipt, dict)
    meta = receipt.get("meta", {}) if isinstance(receipt.get("meta"), dict) else {}
    assert meta.get("parent_ingest_sig") == idx.index_sha256
