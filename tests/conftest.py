# tests/conftest.py
"""
Shared fixtures cho toàn bộ test:
- client: FastAPI TestClient
- patch_db_default: monkeypatch oee_api.db.query_df với data mẫu "đẹp"
"""

from datetime import datetime, timedelta
from typing import Any, Sequence
import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def client() -> TestClient:
    from oee_api.api import app  # import app thật
    return TestClient(app)


@pytest.fixture(autouse=True)
def patch_db_default(monkeypatch):
    """
    Mock mặc định cho mọi test: trả về
      - series (khi SQL có 'GROUP BY ts_bucket')
      - gauges (ngược lại)
    Bạn có thể override ở từng test bằng cách set monkeypatch.setattr(...) lại.
    """
    base = datetime(2025, 9, 28, 9, 0, 0)
    series_rows = [
        # 3 bucket liên tiếp (min/hour/day... đều OK vì API chỉ cần ts_bucket + sums)
        dict(ts_bucket=base + timedelta(minutes=i * 5),
             line_id=105, good=10 + i, reject=1, runtime_sec=300, downtime_sec=0,
             ideal_rate_per_min=600.0)
        for i in range(3)
    ]
    gauges_row = dict(line_id=105, good=123, reject=4, runtime_sec=1800,
                      downtime_sec=60, ideal_rate_per_min=600.0)

    def fake_query_df(sql: str, params: Sequence[Any]):
        sql_norm = " ".join(sql.split()).lower()
        if "group by ts_bucket" in sql_norm:
            return pd.DataFrame(series_rows)
        return pd.DataFrame([gauges_row])

    monkeypatch.setattr("oee_api.db.query_df", fake_query_df)
