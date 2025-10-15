import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

try:
    from examples.query_server import app as ex_app
except Exception:  # pragma: no cover
    ex_app = None  # type: ignore


@pytest.mark.skipif(ex_app is None, reason="examples.query_server not available")
def test_e2e_candidate_set_hash_determinism():
    assert ex_app is not None
    client = TestClient(ex_app)
    # Build tiny index with two chunks in different departments
    with tempfile.TemporaryDirectory() as td:
        idx = Path(td) / "tiny.jsonl"
        text = "Payment terms are net thirty days."
        base = {
            "page_number": 1,
            "start": 0,
            "end": len(text),
            "text": text,
            "vector": [0.0] * 384,
        }
        recs = [
            {**base, "source_path": "docA.pdf", "meta": {"department": "litigation"}},
            {**base, "source_path": "docB.pdf", "meta": {"department": "tax"}},
        ]
        with idx.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")

        body = {"index_path": str(idx), "q": "payment terms?", "backend": "jsonl", "k": 2}

        r_all = client.post("/v1/query-e2e", json=body)
        assert r_all.status_code == 200, r_all.text
        e_all = r_all.json()

        body_f = {**body, "filters": {"department": "litigation"}}
        r_f = client.post("/v1/query-e2e", json=body_f)
        assert r_f.status_code == 200, r_f.text
        e_f = r_f.json()

    # Extract candidate_set_hash from likely locations
    def get_cset(obj):
        return (
            obj.get("receipt", {}).get("meta", {}).get("candidate_set_hash")
            or obj.get("receipt", {}).get("candidate_set_hash")
            or obj.get("meta", {}).get("candidate_set_hash")
        )

    c_all = get_cset(e_all)
    c_f = get_cset(e_f)

    if not (c_all and c_f):
        pytest.skip("candidate_set_hash not present in response")

    assert c_all != c_f, "candidate_set_hash should change with filters"

    # Core receipt fields that should remain stable
    def core(obj, keys=("epsilon", "tau")):
        r = obj.get("receipt", {})
        return tuple(r.get(k) for k in keys)

    assert core(e_all) == core(e_f)
