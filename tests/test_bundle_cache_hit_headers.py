import os

import httpx
import pytest

# Use examples query server which exposes cache at /v1/query; adjust via FIRM_API
API = os.getenv("FIRM_API", "http://127.0.0.1:8080")


@pytest.mark.skipif(os.getenv("FIRM_HAS_BUNDLE") != "1", reason="Set FIRM_HAS_BUNDLE=1 if cache headers are exposed on /v1/bundle or equivalent")
def test_bundle_cache_hit_headers():
    payload = {
        "index_path": "_tmp_demo/demo_index.jsonl",
        "q": "Payment terms summary",
        "backend": "jsonl",
        "k": 12,
    }
    # Try /v1/bundle first, fallback to /v1/query
    with httpx.Client(timeout=15.0) as c:
        r1 = c.post(f"{API}/v1/bundle", json=payload)
        if r1.status_code == 404:
            r1 = c.post(f"{API}/v1/query", json=payload)
        r1.raise_for_status()
        miss = r1.headers.get("X-Cache")

        r2 = c.post(f"{API}/v1/bundle", json=payload)
        if r2.status_code == 404:
            r2 = c.post(f"{API}/v1/query", json=payload)
        r2.raise_for_status()
        hit = r2.headers.get("X-Cache")
        hits = r2.headers.get("X-Cache-Hits")

    assert miss in ("MISS", None)
    assert hit == "HIT"
    assert hits in ("1", "2")
