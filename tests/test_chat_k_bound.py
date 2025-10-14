from __future__ import annotations

from fastapi.testclient import TestClient

from opra.api.app import app


def test_chat_k_bound():
    client = TestClient(app)
    payload = {
        "index_path": "_tmp_demo/demo_index.jsonl",
        "q": "What is Oscillink?",
        "backend": "jsonl",
        "k": 60,
        "embed_model": "bge-small-en-v1.5",
        "synth_mode": "extractive",
    }
    r = client.post("/v1/chat", json=payload)
    assert r.status_code == 422
    data = r.json()
    assert "error" in data
    assert "k>20" in data["error"]
