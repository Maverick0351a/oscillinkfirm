from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from oscillink.ingest.chunk import Chunk
from oscillink.ingest.embed import load_embedding_model
from oscillink.ingest.index_simple import build_jsonl_index
from oscillink.ingest.models_registry import ModelSpec
from oscillink.ingest.query_service import query_index


def _stable_chunk(path: Path, text: str, page: int, start: int, end: int, meta: Dict[str, Any]) -> Chunk:
    return Chunk(
        source_path=str(path),
        page_number=page,
        start=start,
        end=end,
        text=text,
        meta=meta,
    )


def test_candidate_set_hash_changes_but_receipt_shape_constant(tmp_path: Path, monkeypatch):
    # Ensure embedding model stub has matching dim with our index and no hash enforcement
    def _stub(name: str) -> ModelSpec:
        return ModelSpec(name=name, revision=None, sha256_weights=None, sha256_tokenizer=None, license=None, dim=384, path=None)

    monkeypatch.setattr("oscillink.ingest.embed.get_model_spec", _stub)
    model = load_embedding_model("bge-small-en-v1.5")

    # Create a tiny index with two chunks from the same file and distinct metadata
    doc_path = tmp_path / "demo.txt"
    doc_path.write_text("alpha one\nbeta two\n", encoding="utf-8")
    ch0 = _stable_chunk(doc_path, "alpha one", 1, 0, 5, {"dept": "A", "client_id": "X"})
    ch1 = _stable_chunk(doc_path, "beta two", 1, 6, 11, {"dept": "B", "client_id": "X"})
    vecs = model.embed([ch0.text, ch1.text])
    idx = build_jsonl_index([ch0, ch1], vecs, out_path=str(tmp_path / "idx.jsonl"))

    # Run e2e without filters (R0) and with filters that keep one candidate (R1)
    R0 = query_index(index_path=idx.index_path, backend="jsonl", q="alpha", e2e=True, epsilon=1e-3, tau=0.3)
    R1 = query_index(index_path=idx.index_path, backend="jsonl", q="alpha", e2e=True, epsilon=1e-3, tau=0.3, filters={"dept": "A"})

    # Both should be non-abstain and include settle receipts
    assert not R0.get("abstain") and not R1.get("abstain")
    s0 = R0["settle_receipt"]
    s1 = R1["settle_receipt"]
    m0 = s0.get("meta", {})
    m1 = s1.get("meta", {})

    # Candidate set hash must differ since the candidate universe changed
    assert m0.get("candidate_set_hash") and m1.get("candidate_set_hash")
    assert m0["candidate_set_hash"] != m1["candidate_set_hash"]

    # Query receipt invariants: filters change what we consider, not how we compute
    qr0 = R0["receipt"]
    qr1 = R1["receipt"]
    for key in ("dim", "epsilon", "tau", "query_model_sha256", "index_model_sha256"):
        assert qr0.get(key) == qr1.get(key)

    # Schema/shape invariants on settle receipts: same key set and meta key set
    assert set(s0.keys()) == set(s1.keys())
    assert set(m0.keys()) == set(m1.keys())

    # State signature may differ if top-K differs; assert presence and type only
    assert isinstance(m0.get("state_sig"), str) and isinstance(m1.get("state_sig"), str)
