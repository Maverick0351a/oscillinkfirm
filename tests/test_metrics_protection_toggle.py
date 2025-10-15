from fastapi.testclient import TestClient

from examples.query_server import app as ex_app


def test_metrics_protection_toggle(monkeypatch):
    client = TestClient(ex_app)
    # Protected, no secret → non-200
    monkeypatch.setenv("OSCILLINK_METRICS_PROTECTED", "1")
    r_forbidden = client.get("/metrics")
    assert r_forbidden.status_code in (401, 403, 503)

    # Provide secret → 200
    monkeypatch.setenv("OSCILLINK_ADMIN_SECRET", "abc123")
    r_ok = client.get("/metrics", headers={"X-Admin-Secret": "abc123"})
    assert r_ok.status_code == 200
    text = r_ok.text
    assert ("process_cpu_seconds_total" in text) or ("osc_http_requests_total" in text) or ("osc_query_abstain_total" in text)
