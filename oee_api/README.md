# =============================
# ========= README.md =========
# =============================
# OEE API V1 (Refactor)

### 1) Configure environment
```
set ODBC_CONN_STR=Driver={MySQL ODBC 9.3 Unicode Driver};Server=192.168.58.6;Port=3306;Database=bms_db;Uid=thingsboard;Pwd=@thingsboard;
# optional
set API_KEY=
set CORS_ALLOW_ALL=1
```

### 2) Install & run
```
python -m venv .venv && . .venv/Scripts/activate  # Windows
pip install -r requirements.txt
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### 3) Sample calls
- `GET /health`
- `GET /oee/line/105?since_min=240&gran=min&include=gauges,linechart`
- `GET /oee/line/0?from_ts=2025-09-01T00:00:00+07:00&to_ts=2025-09-26T23:59:59+07:00&gran=day&include=linechart,summaries`
- http://127.0.0.1:8080/oee/line/105?since_min=15&gran=min&include=gauges,linechart
- http://127.0.0.1:8080/oee/line/0?from_ts=2025-09-01T00:00:00&to_ts=2025-09-27T23:59:59&gran=day&include=linechart,summaries
- http://127.0.0.1:8080/oee/line/105?from_ts=2025-09-25T00:00:00&to_ts=2025-09-27T23:59:59&gran=shift&shift_no=1&include=linechart
- http://127.0.0.1:8080/oee/line/105?since_min=360&gran=min&limit=120&include=linechart


### 4) Notes
- Align table/column names in `sql_texts.py` with your database.
- If you already have battle-tested SQL, paste them into `sql_texts.py` keeping the same parameters order.
- All timestamps returned in ISO 8601 with `+07:00`. Default lookback `since_min=240`.
- `line_id=0` aggregates across all lines.
- `limit` controls number of buckets returned; internal `MAX_ROWS_FUSE` protects the DB.
- Shift filter uses `dim_shift_calendar` (`shift_no`, `start_time`, `end_time`, `active`). Overnight handled.

### 4) Dạng module (mình khuyên dùng):
    python -m oee_api.api
Hoặc uvicorn:
    uvicorn oee_api.api:app --host 127.0.0.1 --port 8080
    
Thấy log: Uvicorn running on http://127.0.0.1:8080 là OK.