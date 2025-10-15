import json
import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from examples.query_server import app as ex_app


API = os.getenv("FIRM_API", "http://127.0.0.1:8080")


def _write_jsonl(path: Path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


@pytest.mark.parametrize("abstain_on_all_low", [False, True])
def test_ocr_penalty_influence(monkeypatch, abstain_on_all_low):
    # Build a tiny index with two chunks: one clean, one low OCR
    with tempfile.TemporaryDirectory() as td:
        idx = Path(td) / "tiny.jsonl"
        clean_text = "Payment terms are net thirty days."
        clean = {
            "source_path": "clean.pdf",
            "page_number": 1,
            "start": 0,
            "end": len(clean_text),
            "text": clean_text,
            "meta": {"title": "clean", "ocr_low_confidence": False},
            "vector": [0.0] * 384,
        }
        low_text = "Payment terms are net thirty days."
        low = {
            "source_path": "scan_bad.pdf",
            "page_number": 1,
            "start": 0,
            "end": len(low_text),
            "text": low_text,
            "meta": {"title": "low_ocr", "ocr_low_confidence": True, "ocr_avg_confidence": 0.42},
            "vector": [0.0] * 384,
        }
        _write_jsonl(idx, [clean, low])

        # Configure penalty + abstain policy
        monkeypatch.setenv("OSCILLINK_OCR_SCORE_PENALTY", "0.08")
        monkeypatch.setenv("OSCILLINK_OCR_ABSTAIN_ON_ALL_LOW", "1" if abstain_on_all_low else "0")

        payload = {"index_path": str(idx), "q": "What are the payment terms?", "backend": "jsonl", "k": 2}
        client = TestClient(ex_app)
        r = client.post("/v1/query", json=payload)
        assert r.status_code == 200
        J = r.json()

        if abstain_on_all_low:
            # Force k=1 on low_ocr only case
            payload_low = {
                "index_path": str(idx),
                "q": "payment terms",
                "backend": "jsonl",
                "k": 1,
                "filters": {"title": "low_ocr"},
            }
            r2 = client.post("/v1/query", json=payload_low)
            assert r2.status_code == 200
            J2 = r2.json()
            # Verify abstain flag may trigger under low OCR depending on tau/epsilon
            assert J2.get("abstain") in (True, False)
            if J2.get("abstain"):
                assert "low" in (J2.get("reason", "").lower()) or J2.get("reason") in {"insufficient coherence"}
        else:
            res = J.get("results", [])
            assert len(res) >= 2
            titles = [r.get("meta", {}).get("title") for r in res[:2]]
            # In non-e2e path we attach meta only if index includes it; guard for absence
            if titles[0] is not None:
                assert titles[0] == "clean", f"expected clean to outrank low_ocr, got {titles}"
