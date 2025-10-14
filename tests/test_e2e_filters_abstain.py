from __future__ import annotations

from pathlib import Path

from oscillink.ingest.models_registry import ModelSpec
from oscillink.ingest.query_service import query_index


def test_e2e_filters_no_candidates(tmp_path: Path, monkeypatch):
    # Use the bundled small demo index
    repo_root = Path.cwd()
    index_path = str(repo_root / "data" / "demo_index.jsonl")
    # Make embedding model loader return a 384-dim stub without hash enforcement
    def _stub(name: str) -> ModelSpec:
        return ModelSpec(name=name, revision=None, sha256_weights=None, sha256_tokenizer=None, license=None, dim=384, path=None)
    monkeypatch.setattr("oscillink.ingest.embed.get_model_spec", _stub)
    # Apply a filter that cannot possibly match
    out = query_index(index_path=index_path, backend="jsonl", q="test", e2e=True, epsilon=1e-3, tau=0.9, filters={"no_such_key": "nope"})
    assert out.get("abstain") is True
    assert out.get("reason") == "no candidates after filter"
    # Receipt should be present with dims and model hashes (hashes may be None)
    rec = out.get("receipt")
    assert isinstance(rec, dict) and "dim" in rec and "epsilon" in rec and "tau" in rec
    # Ingest receipt passes through when available
    if out.get("ingest_receipt"):
        assert isinstance(out["ingest_receipt"], dict)
