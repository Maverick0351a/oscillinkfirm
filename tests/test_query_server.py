from __future__ import annotations

from fastapi.testclient import TestClient

from examples.query_server import app


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_query_jsonl_demo():
    client = TestClient(app)
    payload = {"index_path": "_tmp_demo/demo_index.jsonl", "q": "What is Oscillink?", "backend": "jsonl", "k": 3}
    r = client.post("/v1/query", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "results" in data and isinstance(data["results"], list)
    assert len(data["results"]) >= 1
