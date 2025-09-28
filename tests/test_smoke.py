# =============================
# ===== tests/test_smoke.py ===
# =============================
"""
Minimal smoke tests (pytest) for OEE API.
Run: pytest -q
"""
from fastapi.testclient import TestClient
from oee_api.api import app

def test_health():
    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_oee_line_defaults():
    c = TestClient(app)
    r = c.get("/oee/line/0")
    # We cannot guarantee DB data here; just assert payload shape
    assert r.status_code == 200
    j = r.json()
    assert "meta" in j
    assert "gauges" in j or "linechart" in j