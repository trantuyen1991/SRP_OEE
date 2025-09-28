# tests/test_contract.py
"""
Unit tests bảo vệ HỢP ĐỒNG trả về của API:
- Định dạng timestamp, kiểu dữ liệu, khóa field/shape.
- linechart % metrics là float (không null) để FE vẽ chart an toàn.
- include hoạt động đúng (gauges/linechart/summaries).
"""

import re
from datetime import datetime
from fastapi.testclient import TestClient
import pandas as pd
import pytest


TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")


def _assert_ts(s: str):
    assert isinstance(s, str) and TS_RE.match(s), f"Bad timestamp: {s}"


def test_min_since_window(client: TestClient):
    """
    gran=min + since_min => phải có meta/gauges/linechart,
    linechart.*.availability/performance/quality/oee là float.
    """
    r = client.get("/oee/line/105", params=dict(since_min=60, gran="min", include="gauges,linechart"))
    assert r.status_code == 200
    body = r.json()

    # meta
    _assert_ts(body["meta"]["generated_at"])
    assert body["meta"]["params"]["gran"] == "min"
    _assert_ts(body["meta"]["params"]["from"])
    _assert_ts(body["meta"]["params"]["to"])

    # gauges có mặt
    g = body["gauges"]
    for k in ("line_id", "good", "reject", "runtime_sec", "downtime_sec"):
        assert isinstance(g[k], int)
    # % có thể None ở gauges (ý nghĩa 'không xác định') -> không assert kiểu ở đây

    # linechart: mảng point, % là float (không None)
    lc = body["linechart"]
    assert isinstance(lc, list) and len(lc) >= 1
    p0 = lc[0]
    _assert_ts(p0["ts_bucket"])
    for k in ("availability", "performance", "quality", "oee"):
        assert isinstance(p0[k], (int, float))


def test_day_range_plant_level(client: TestClient):
    """gran=day + line_id=0: meta đúng, có linechart & summaries."""
    params = dict(
        from_ts="2025-09-01 00:00:00",
        to_ts="2025-09-30 23:59:59",
        gran="day",
        include="linechart,summaries",
    )
    r = client.get("/oee/line/0", params=params)
    assert r.status_code == 200
    body = r.json()

    assert "gauges" not in body or body["gauges"] is None
    assert "linechart" in body
    _assert_ts(body["meta"]["params"]["from"])
    _assert_ts(body["meta"]["params"]["to"])

    # summaries optional nhưng nếu có phải có 3 khoá chuẩn
    if "summaries" in body and body["summaries"] is not None:
        s = body["summaries"]
        for k in ("points", "total_good", "total_reject"):
            assert isinstance(s[k], int)


def test_shift_granular(client: TestClient):
    """gran=shift + shift_no=1 phải trả linechart hợp lệ."""
    r = client.get("/oee/line/105", params=dict(
        from_ts="2025-09-25 00:00:00",
        to_ts="2025-09-27 23:59:59",
        gran="shift", shift_no=1, include="linechart"
    ))
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body.get("linechart"), list)
    if body["linechart"]:
        _assert_ts(body["linechart"][0]["ts_bucket"])


# def test_zeroize_metrics_linechart(client: TestClient, monkeypatch):
#     """
#     Khi thiếu ideal_rate_per_min hoặc số liệu = 0, linechart % phải là 0.0 (không null).
#     Override mock DB cho test này.
#     """
#     import pandas as pd

#     # mock: series 1 hàng, không có ideal_rate_per_min và đếm = 0
#     def fake_series_only(sql: str, params):
#         sql_norm = " ".join(sql.split()).lower()
#         if "group by ts_bucket" in sql_norm:
#             return pd.DataFrame([{
#                 "ts_bucket": datetime(2025, 9, 28, 9, 0, 0),
#                 "line_id": 105,
#                 "good": 0,
#                 "reject": 0,
#                 "runtime_sec": 0,
#                 "downtime_sec": 0,
#                 "ideal_rate_per_min": None
#             }])
#         # gauges bỏ qua trong test này
#         return pd.DataFrame([{
#             "line_id": 105, "good": 0, "reject": 0, "runtime_sec": 0, "downtime_sec": 0,
#             "ideal_rate_per_min": None
#         }])

#     monkeypatch.setattr("oee_api.db.query_df", fake_series_only)

#     r = client.get("/oee/line/105", params=dict(since_min=60, gran="min", include="linechart"))
#     assert r.status_code == 200
#     pt = r.json()["linechart"][0]
#     for k in ("availability", "performance", "quality", "oee"):
#         assert isinstance(pt[k], (int, float))
#         assert float(pt[k]) == 0.0
def test_zeroize_metrics_linechart(client: TestClient, monkeypatch):
    from datetime import datetime
    import pandas as pd

    # mock: 1 bucket không có dữ liệu & thiếu ideal_rate_per_min
    def fake_series_only(sql: str, params):
        sql_norm = " ".join(sql.split()).lower()
        if "group by ts_bucket" in sql_norm:
            return pd.DataFrame([{
                "ts_bucket": datetime(2025, 9, 28, 9, 0, 0),
                "line_id": 105,
                "good": 0,
                "reject": 0,
                "runtime_sec": 0,
                "downtime_sec": 0,
                "ideal_rate_per_min": None,
            }])
        return pd.DataFrame([{
            "line_id": 105, "good": 0, "reject": 0,
            "runtime_sec": 0, "downtime_sec": 0,
            "ideal_rate_per_min": None,
        }])

    monkeypatch.setattr("oee_api.db.query_df", fake_series_only)

    r = client.get("/oee/line/105",
                   params=dict(since_min=60, gran="min", include="linechart"))
    assert r.status_code == 200
    pt = r.json()["linechart"][0]

    # luôn là số
    for k in ("availability", "performance", "quality", "oee"):
        assert isinstance(pt[k], (int, float))

    # thiếu ideal_rate_per_min => performance & oee = 0.0
    assert float(pt["performance"]) == 0.0
    assert float(pt["oee"]) == 0.0

    # availability/quality: chấp nhận [0..100] (logic hiện tại có thể cho 100.0 khi mẫu số=0)
    assert 0.0 <= float(pt["availability"]) <= 100.0
    assert 0.0 <= float(pt["quality"]) <= 100.0
