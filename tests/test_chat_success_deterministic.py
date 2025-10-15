import json

import pytest
from fastapi.testclient import TestClient

from opra.api.app import app


@pytest.mark.parametrize("k", [5, 10, 20])
def test_chat_success_deterministic(k):
    payload = {
        "index_path": "_tmp_demo/demo_index.jsonl",
        "q": "Summarize clause 9",
        "backend": "jsonl",
        "k": k,
        "synth_mode": "extractive",
    }
    client = TestClient(app)
    r1 = client.post("/v1/chat", json=payload)
    assert r1.status_code == 200
    r2 = client.post("/v1/chat", json=payload)
    assert r2.status_code == 200

    j1, j2 = r1.json(), r2.json()
    assert j1.get("abstain") is not True
    assert isinstance(j1.get("receipt"), dict)

    def top_id(j):
        cits = j.get("citations") or []
        if not cits:
            return None
        c0 = cits[0]
        # Normalize field names; our ChatResponse uses page_number not page
        return (c0.get("source_path"), c0.get("page_number"))

    assert top_id(j1) == top_id(j2), "Top citation changed between identical runs"
    # Receipts stable across identical runs
    assert json.dumps(j1.get("receipt"), sort_keys=True) == json.dumps(j2.get("receipt"), sort_keys=True)
