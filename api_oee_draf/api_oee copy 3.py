"""
api_oee.py
==========

FastAPI mini-service cung cấp JSON cho dashboard OEE:
- /oee/line/{line_id}: hỗ trợ range mode (from_ts/to_ts) và gran mode (min).
- /oee/dashboard/line/{line_id}: gom snapshot gồm gauges + linechart + summaries (shift,day,week,month,quarter,year).
- Có cache TTL để nhiều client cùng gọi thì dùng chung kết quả.
"""

import os, time, math, logging
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta, date, time as dtime
from decimal import Decimal

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# Config & logging
# -----------------------------------------------------------------------------

DB_URL = os.getenv("DB_URL", "mysql+mysqlconnector://root:root@127.0.0.1:3306/mpy_oee")
DEFAULT_TTL_SEC = int(os.getenv("CACHE_TTL_SEC", "30"))
TARGET_POINTS = int(os.getenv("TARGET_POINTS", "2000"))
MIN_BUCKET_SEC = 60
MAX_ROWS = int(os.getenv("MAX_ROWS", "5000"))

def _setup_logging() -> None:
    """Setup logging format & handler"""
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_fmt = "%(asctime)s [api_oee] %(levelname)s: %(message)s"
    logging.basicConfig(level=getattr(logging, level, logging.INFO),
                        format=log_fmt, handlers=[logging.StreamHandler()])

_setup_logging()

# -----------------------------------------------------------------------------
# Engine & cache TTL
# -----------------------------------------------------------------------------

def get_engine() -> Engine:
    """Create SQLAlchemy engine for MySQL"""
    logging.info("Creating engine to %s", DB_URL)
    return create_engine(DB_URL, pool_recycle=3600, pool_pre_ping=True)

ENGINE: Engine = get_engine()

class TTLCache:
    """Simple TTL cache for storing API responses"""
    def __init__(self, ttl_sec: int):
        self.ttl = ttl_sec
        self.store: Dict[Tuple, Tuple[float, Dict[str, Any]]] = {}
    def get(self, key: Tuple) -> Optional[Dict[str, Any]]:
        rec = self.store.get(key)
        if not rec: return None
        exp, val = rec
        if time.time() > exp:
            self.store.pop(key, None)
            return None
        return val
    def set(self, key: Tuple, val: Dict[str, Any]) -> None:
        self.store[key] = (time.time() + self.ttl, val)

CACHE = TTLCache(DEFAULT_TTL_SEC)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    """Parse ISO datetime string safely"""
    if not s: return None
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"):
        try: return datetime.strptime(s, fmt)
        except ValueError: continue
    raise HTTPException(status_code=400, detail=f"Invalid datetime format: {s}")

def _choose_bucket_sec(dt_from: datetime, dt_to: datetime, target: int = TARGET_POINTS) -> int:
    """Auto choose bucket seconds for line chart"""
    total = (dt_to - dt_from).total_seconds()
    if total <= 0: return MIN_BUCKET_SEC
    raw = max(MIN_BUCKET_SEC, math.ceil(total / target))
    return int(math.ceil(raw / 60.0) * 60)

def _df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert DataFrame to JSON-safe list of dicts"""
    if df.empty:
        return []
    out = df.copy()

    # Convert all NaN/inf thành None
    out = out.replace([np.inf, -np.inf], np.nan).where(pd.notnull(out), None)

    # Format datetime
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d %H:%M:%S")

    def _json_safe(x):
        if x is None: return None
        if isinstance(x, datetime): return x.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(x, date): return x.strftime("%Y-%m-%d")
        if isinstance(x, dtime): return x.strftime("%H:%M:%S")
        if isinstance(x, Decimal): return float(x)
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        return x

    for c in out.columns:
        out[c] = out[c].map(_json_safe)

    return out.to_dict(orient="records")

def _build_oee_expr() -> str:
    """Return OEE formula string for SQL"""
    return """
    ROUND(
      (SUM(f.runtime_sec)/NULLIF(SUM(f.planned_sec),0)) *
      (SUM(f.produced)/NULLIF(SUM((f.runtime_sec/60.0)*COALESCE(dp.ideal_rate_per_min,0)),0)) *
      (SUM(f.good)/NULLIF(SUM(f.produced),0)) * 100, 2
    ) AS oee_pct
    """
# -----------------------------------------------------------------------------
# SQL builder cho range (from_ts/to_ts) + auto bucket
# -----------------------------------------------------------------------------
def _build_sql_range(
    line_id: int,
    process_order: Optional[str],
    packaging_id: Optional[int],
    dt_from: datetime,
    dt_to: datetime,
    limit: int,
    bucket_sec: Optional[int] = None,
) -> Tuple[str, Dict[str, Any], str, Dict[str, Any], int]:
    """
    Tạo 2 câu SQL:
      - sql_series: dữ liệu line chart (đã group theo bucket thời gian)
      - sql_gauges: tổng hợp cho 4 đồng hồ (A/P/Q/OEE) trong range
    Trả về:
      (sql_series, params_series, sql_gauges, params_gauges, bucket_used)
    """
    # Params chung
    p_base: Dict[str, Any] = {
        "line_id": line_id,
        "from_ts": dt_from.strftime("%Y-%m-%d %H:%M:%S"),
        "to_ts": dt_to.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # WHERE động theo filter
    wheres = ["f.line_id = :line_id", "f.ts_min BETWEEN :from_ts AND :to_ts"]
    if process_order:
        wheres.append("f.process_order = :po")
        p_base["po"] = process_order
    if packaging_id is not None:
        wheres.append("f.packaging_id = :pkg")
        p_base["pkg"] = packaging_id
    where_sql = " WHERE " + " AND ".join(wheres)

    # Chọn bucket
    if not bucket_sec or bucket_sec <= 0:
        bucket_sec = _choose_bucket_sec(dt_from, dt_to, TARGET_POINTS)

    if bucket_sec <= 60:
        # từng phút (giữ nguyên mốc ts_min)
        select_time = "f.ts_min AS bucket"
        group_clause = "GROUP BY f.ts_min"
        order_expr = "bucket"
    else:
        # gộp theo giây bucket_sec (bội số của 60)
        grp = f"FLOOR(UNIX_TIMESTAMP(f.ts_min)/{bucket_sec})"
        select_time = f"FROM_UNIXTIME({grp}*{bucket_sec}) AS bucket"
        group_clause = f"GROUP BY {grp}"
        order_expr = "bucket"

    # --- SQL cho line chart (series) ---
    sql_series = f"""
    SELECT
      {select_time},
      ROUND(100.0 * SUM(f.runtime_sec) / NULLIF(SUM(f.planned_sec), 0), 2) AS availability_pct,
      ROUND(100.0 * SUM(f.produced) /
            NULLIF(SUM((f.runtime_sec/60.0) * COALESCE(dp.ideal_rate_per_min, 0)), 0), 2) AS performance_pct,
      ROUND(100.0 * SUM(f.good) / NULLIF(SUM(f.produced), 0), 2) AS quality_pct,
      ROUND(
        (SUM(f.runtime_sec) / NULLIF(SUM(f.planned_sec), 0)) *
        (SUM(f.produced) / NULLIF(SUM((f.runtime_sec/60.0) * COALESCE(dp.ideal_rate_per_min, 0)), 0)) *
        (SUM(f.good)      / NULLIF(SUM(f.produced), 0)) * 100
      , 2) AS oee_pct
    FROM fact_production_min f
    LEFT JOIN dim_packaging dp ON dp.packaging_id = f.packaging_id
    {where_sql}
    {group_clause}
    ORDER BY {order_expr}
    LIMIT :limit
    """

    # params cho series (có :limit)
    p_series = dict(p_base)
    p_series["limit"] = min(limit, MAX_ROWS)

    # --- SQL cho gauges (tổng hợp) ---
    sql_gauges = f"""
    SELECT
      ROUND(100.0 * SUM(f.runtime_sec) / NULLIF(SUM(f.planned_sec), 0), 2) AS availability_pct,
      ROUND(100.0 * SUM(f.produced) /
            NULLIF(SUM((f.runtime_sec/60.0) * COALESCE(dp.ideal_rate_per_min, 0)), 0), 2) AS performance_pct,
      ROUND(100.0 * SUM(f.good) / NULLIF(SUM(f.produced), 0), 2) AS quality_pct,
      ROUND(
        (SUM(f.runtime_sec) / NULLIF(SUM(f.planned_sec), 0)) *
        (SUM(f.produced) / NULLIF(SUM((f.runtime_sec/60.0) * COALESCE(dp.ideal_rate_per_min, 0)), 0)) *
        (SUM(f.good)      / NULLIF(SUM(f.produced), 0)) * 100
      , 2) AS oee_pct
    FROM fact_production_min f
    LEFT JOIN dim_packaging dp ON dp.packaging_id = f.packaging_id
    {where_sql}
    """

    # params cho gauges (không có :limit)
    p_gauges = dict(p_base)

    return sql_series, p_series, sql_gauges, p_gauges, bucket_sec

# -----------------------------------------------------------------------------
# Param normalization
# -----------------------------------------------------------------------------

class FilterParams(BaseModel):
    gran: Optional[str]=None
    process_order: Optional[str]=None
    packaging_id: Optional[int]=None
    from_ts: Optional[str]=None
    to_ts: Optional[str]=None
    limit: Optional[int]=Field(default=None, ge=1, le=MAX_ROWS)
    bucket_sec: Optional[int]=Field(default=None, ge=60)
    since_min: Optional[int]=Field(default=None, ge=1)

def _normalize_params(req:Request)->FilterParams:
    """Normalize query params from request"""
    qp=dict(req.query_params)
    alias_map={"po":"process_order","pkg":"packaging_id","from":"from_ts","to":"to_ts","max_points":"limit"}
    for k_alias,k_std in alias_map.items():
        if k_alias in qp and k_std not in qp: qp[k_std]=qp[k_alias]
    try: return FilterParams(**qp)
    except Exception as e:
        logging.error("Param normalize error: %s | raw=%s", e, qp)
        raise HTTPException(status_code=400, detail=f"Bad query params: {e}")
# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------

app = FastAPI(title="OEE API", version="0.4.0")

# Cho phép gọi từ browser / widget
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    """Health check endpoint"""
    try:
        with ENGINE.connect() as conn:
            row = conn.execute(
                text("SELECT NOW() AS now_ts, DATABASE() AS db")
            ).mappings().first()
        return {"ok": True, "db_time": str(row["now_ts"]), "db": row["db"]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# -----------------------------------------------------------------------------
# Endpoint /oee/line/{line_id}
# -----------------------------------------------------------------------------

@app.get("/oee/line/{line_id}")
def get_oee(line_id: int, q: FilterParams = Depends(_normalize_params)) -> JSONResponse:
    """
    API trả về dữ liệu OEE cho 1 line.
    - Nếu có from_ts/to_ts: dùng range mode (linechart auto bucket + gauges).
    - Nếu không: dùng gran mode (min), query từ fact_production_min với since_min.
    """
    gran = q.gran or "min"
    limit = q.limit or TARGET_POINTS
    bucket_sec = q.bucket_sec

    # ------------------------------
    # Range mode: có from/to
    # ------------------------------
    if q.from_ts and q.to_ts:
        dt_from = _parse_iso(q.from_ts)
        dt_to = _parse_iso(q.to_ts)
        if dt_from >= dt_to:
            raise HTTPException(status_code=400, detail="from_ts must < to_ts")

        key = ("range", line_id, q.process_order, q.packaging_id, q.from_ts, q.to_ts, limit, bucket_sec)
        cached = CACHE.get(key)
        if cached:
            return JSONResponse(cached)

        sql_s, p_s, sql_g, p_g, bucket_used = _build_sql_range(
            line_id, q.process_order, q.packaging_id, dt_from, dt_to, limit, bucket_sec
        )

        with ENGINE.connect() as conn:
            df_series = pd.read_sql(text(sql_s), conn, params=p_s)
            df_gauges = pd.read_sql(text(sql_g), conn, params=p_g)

        data = _df_to_records(df_series)
        gauges = _df_to_records(df_gauges)
        gauges = gauges[0] if gauges else {}

        payload = {
            "meta": {
                "mode": "range",
                "line_id": line_id,
                "from_ts": q.from_ts,
                "to_ts": q.to_ts,
                "process_order": q.process_order,
                "packaging_id": q.packaging_id,
                "row_count": len(data),
                "limit": limit,
                "bucket_sec": bucket_used,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "gauges": gauges,
            "data": data,
        }
        CACHE.set(key, payload)
        return JSONResponse(payload)

    # ------------------------------
    # Gran mode: không có from/to
    # ------------------------------
    dt_to = datetime.now()
    since = q.since_min or 240
    dt_from = dt_to - timedelta(minutes=since)

    key = ("gran", line_id, gran, q.process_order, q.packaging_id, since, limit, bucket_sec)
    cached = CACHE.get(key)
    if cached:
        return JSONResponse(cached)

    sql_s, p_s, sql_g, p_g, bucket_used = _build_sql_range(
        line_id, q.process_order, q.packaging_id, dt_from, dt_to, limit, bucket_sec
    )

    with ENGINE.connect() as conn:
        df_series = pd.read_sql(text(sql_s), conn, params=p_s)
        df_gauges = pd.read_sql(text(sql_g), conn, params=p_g)

    data = _df_to_records(df_series)
    gauges = _df_to_records(df_gauges)
    gauges = gauges[0] if gauges else {}

    payload = {
        "meta": {
            "mode": "gran",
            "gran": gran,
            "line_id": line_id,
            "since_min": since,
            "row_count": len(data),
            "limit": limit,
            "bucket_sec": bucket_used,
            "process_order": q.process_order,
            "packaging_id": q.packaging_id,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "gauges": gauges,
        "data": data,
    }
    CACHE.set(key, payload)
    return JSONResponse(payload)
# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------

app = FastAPI(title="OEE API", version="0.4.0")

# Cho phép gọi từ browser / widget
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    """Health check endpoint"""
    try:
        with ENGINE.connect() as conn:
            row = conn.execute(
                text("SELECT NOW() AS now_ts, DATABASE() AS db")
            ).mappings().first()
        return {"ok": True, "db_time": str(row["now_ts"]), "db": row["db"]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# -----------------------------------------------------------------------------
# Endpoint /oee/line/{line_id}
# -----------------------------------------------------------------------------

@app.get("/oee/line/{line_id}")
def get_oee(line_id: int, q: FilterParams = Depends(_normalize_params)) -> JSONResponse:
    """
    API trả về dữ liệu OEE cho 1 line.
    - Nếu có from_ts/to_ts: dùng range mode (linechart auto bucket + gauges).
    - Nếu không: dùng gran mode (min), query từ fact_production_min với since_min.
    """
    gran = q.gran or "min"
    limit = q.limit or TARGET_POINTS
    bucket_sec = q.bucket_sec

    # ------------------------------
    # Range mode: có from/to
    # ------------------------------
    if q.from_ts and q.to_ts:
        dt_from = _parse_iso(q.from_ts)
        dt_to = _parse_iso(q.to_ts)
        if dt_from >= dt_to:
            raise HTTPException(status_code=400, detail="from_ts must < to_ts")

        key = ("range", line_id, q.process_order, q.packaging_id, q.from_ts, q.to_ts, limit, bucket_sec)
        cached = CACHE.get(key)
        if cached:
            return JSONResponse(cached)

        sql_s, p_s, sql_g, p_g, bucket_used = _build_sql_range(
            line_id, q.process_order, q.packaging_id, dt_from, dt_to, limit, bucket_sec
        )

        with ENGINE.connect() as conn:
            df_series = pd.read_sql(text(sql_s), conn, params=p_s)
            df_gauges = pd.read_sql(text(sql_g), conn, params=p_g)

        data = _df_to_records(df_series)
        gauges = _df_to_records(df_gauges)
        gauges = gauges[0] if gauges else {}

        payload = {
            "meta": {
                "mode": "range",
                "line_id": line_id,
                "from_ts": q.from_ts,
                "to_ts": q.to_ts,
                "process_order": q.process_order,
                "packaging_id": q.packaging_id,
                "row_count": len(data),
                "limit": limit,
                "bucket_sec": bucket_used,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "gauges": gauges,
            "data": data,
        }
        CACHE.set(key, payload)
        return JSONResponse(payload)

    # ------------------------------
    # Gran mode: không có from/to
    # ------------------------------
    dt_to = datetime.now()
    since = q.since_min or 240
    dt_from = dt_to - timedelta(minutes=since)

    key = ("gran", line_id, gran, q.process_order, q.packaging_id, since, limit, bucket_sec)
    cached = CACHE.get(key)
    if cached:
        return JSONResponse(cached)

    sql_s, p_s, sql_g, p_g, bucket_used = _build_sql_range(
        line_id, q.process_order, q.packaging_id, dt_from, dt_to, limit, bucket_sec
    )

    with ENGINE.connect() as conn:
        df_series = pd.read_sql(text(sql_s), conn, params=p_s)
        df_gauges = pd.read_sql(text(sql_g), conn, params=p_g)

    data = _df_to_records(df_series)
    gauges = _df_to_records(df_gauges)
    gauges = gauges[0] if gauges else {}

    payload = {
        "meta": {
            "mode": "gran",
            "gran": gran,
            "line_id": line_id,
            "since_min": since,
            "row_count": len(data),
            "limit": limit,
            "bucket_sec": bucket_used,
            "process_order": q.process_order,
            "packaging_id": q.packaging_id,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "gauges": gauges,
        "data": data,
    }
    CACHE.set(key, payload)
    return JSONResponse(payload)

# -----------------------------------------------------------------------------
# Endpoint /oee/dashboard/line/{line_id}
# -----------------------------------------------------------------------------

@app.get("/oee/dashboard/line/{line_id}")
def get_dashboard(line_id: int, q: FilterParams = Depends(_normalize_params)) -> JSONResponse:
    """
    Return snapshot for OEE dashboard:
    - Gauges (A/P/Q/OEE)
    - Linechart (auto bucket)
    - Summaries: shift, day, week, month, quarter, year
    """
    limit = q.limit or TARGET_POINTS
    bucket_sec = q.bucket_sec
    now = datetime.now()

    # Determine range
    if q.from_ts and q.to_ts:
        dt_from = _parse_iso(q.from_ts)
        dt_to = _parse_iso(q.to_ts)
    else:
        since = q.since_min or 240
        dt_to = now
        dt_from = now - timedelta(minutes=since)

    if dt_from >= dt_to:
        raise HTTPException(status_code=400, detail="from_ts must < to_ts")

    # Cache key
    key = ("dashboard", line_id, q.process_order, q.packaging_id, dt_from, dt_to, limit, bucket_sec)
    cached = CACHE.get(key)
    if cached:
        return JSONResponse(cached)

    # Build SQL series + gauges
    sql_s, p_s, sql_g, p_g, bucket_used = _build_sql_range(
        line_id, q.process_order, q.packaging_id, dt_from, dt_to, limit, bucket_sec
    )

    p_common = {
        "line_id": line_id,
        "from_ts": dt_from.strftime("%Y-%m-%d %H:%M:%S"),
        "to_ts": dt_to.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # --- SQL summaries ---
    sql_shift = """
    SELECT s.shift_date, s.shift_no,
      SUM(f.produced) AS produced, SUM(f.good) AS good, SUM(f.ng) AS ng,
      ROUND(100.0*SUM(f.runtime_sec)/NULLIF(SUM(f.planned_sec),0),2) AS availability_pct,
      ROUND(100.0*SUM(f.produced)/NULLIF(SUM((f.runtime_sec)/60.0)*COALESCE(dp.ideal_rate_per_min,0)),2) AS performance_pct,
      ROUND(100.0*SUM(f.good)/NULLIF(SUM(f.produced),0),2) AS quality_pct,
      ROUND(
        (SUM(f.runtime_sec)/NULLIF(SUM(f.planned_sec),0)) *
        (SUM(f.produced)/NULLIF(SUM((f.runtime_sec)/60.0)*COALESCE(dp.ideal_rate_per_min,0)),0)) *
        (SUM(f.good)/NULLIF(SUM(f.produced),0)) * 100, 2
      ) AS oee_pct
    FROM dim_shift_calendar s
    JOIN fact_production_min f
      ON f.ts_min >= CONCAT(s.shift_date,' ',s.start_time)
     AND f.ts_min < DATE_ADD(CONCAT(s.shift_date,' ',s.end_time),
                      INTERVAL (s.end_time < s.start_time) DAY)
    LEFT JOIN dim_packaging dp ON dp.packaging_id=f.packaging_id
    WHERE f.line_id=:line_id AND f.ts_min BETWEEN :from_ts AND :to_ts
    GROUP BY s.shift_date, s.shift_no
    ORDER BY s.shift_date, s.shift_no
    """

    sql_day = """
    SELECT DATE(f.ts_min) AS bucket,
      SUM(f.produced) AS produced, SUM(f.good) AS good, SUM(f.ng) AS ng,
      ROUND(100.0*SUM(f.runtime_sec)/NULLIF(SUM(f.planned_sec),0),2) AS availability_pct,
      ROUND(100.0*SUM(f.produced)/NULLIF(SUM((f.runtime_sec)/60.0)*COALESCE(dp.ideal_rate_per_min,0)),2) AS performance_pct,
      ROUND(100.0*SUM(f.good)/NULLIF(SUM(f.produced),0),2) AS quality_pct,
      ROUND(
        (SUM(f.runtime_sec)/NULLIF(SUM(f.planned_sec),0)) *
        (SUM(f.produced)/NULLIF(SUM((f.runtime_sec)/60.0)*COALESCE(dp.ideal_rate_per_min,0)),0)) *
        (SUM(f.good)/NULLIF(SUM(f.produced),0)) * 100, 2
      ) AS oee_pct
    FROM fact_production_min f
    LEFT JOIN dim_packaging dp ON dp.packaging_id=f.packaging_id
    WHERE f.line_id=:line_id AND f.ts_min BETWEEN :from_ts AND :to_ts
    GROUP BY DATE(f.ts_min)
    ORDER BY DATE(f.ts_min)
    """

    sql_week = """
    SELECT YEARWEEK(f.ts_min,3) AS bucket,
      SUM(f.produced) AS produced, SUM(f.good) AS good, SUM(f.ng) AS ng,
      ROUND(100.0*SUM(f.runtime_sec)/NULLIF(SUM(f.planned_sec),0),2) AS availability_pct,
      ROUND(100.0*SUM(f.produced)/NULLIF(SUM((f.runtime_sec)/60.0)*COALESCE(dp.ideal_rate_per_min,0)),2) AS performance_pct,
      ROUND(100.0*SUM(f.good)/NULLIF(SUM(f.produced),0),2) AS quality_pct,
      ROUND(
        (SUM(f.runtime_sec)/NULLIF(SUM(f.planned_sec),0)) *
        (SUM(f.produced)/NULLIF(SUM((f.runtime_sec)/60.0)*COALESCE(dp.ideal_rate_per_min,0)),0)) *
        (SUM(f.good)/NULLIF(SUM(f.produced),0)) * 100, 2
      ) AS oee_pct
    FROM fact_production_min f
    LEFT JOIN dim_packaging dp ON dp.packaging_id=f.packaging_id
    WHERE f.line_id=:line_id AND f.ts_min BETWEEN :from_ts AND :to_ts
    GROUP BY YEARWEEK(f.ts_min,3)
    ORDER BY YEARWEEK(f.ts_min,3)
    """

    sql_month = """
    SELECT DATE_FORMAT(f.ts_min,'%Y-%m') AS bucket,
      SUM(f.produced) AS produced, SUM(f.good) AS good, SUM(f.ng) AS ng,
      ROUND(100.0*SUM(f.runtime_sec)/NULLIF(SUM(f.planned_sec),0),2) AS availability_pct,
      ROUND(100.0*SUM(f.produced)/NULLIF(SUM((f.runtime_sec)/60.0)*COALESCE(dp.ideal_rate_per_min,0)),2) AS performance_pct,
      ROUND(100.0*SUM(f.good)/NULLIF(SUM(f.produced),0),2) AS quality_pct,
      ROUND(
        (SUM(f.runtime_sec)/NULLIF(SUM(f.planned_sec),0)) *
        (SUM(f.produced)/NULLIF(SUM((f.runtime_sec)/60.0)*COALESCE(dp.ideal_rate_per_min,0)),0)) *
        (SUM(f.good)/NULLIF(SUM(f.produced),0)) * 100, 2
      ) AS oee_pct
    FROM fact_production_min f
    LEFT JOIN dim_packaging dp ON dp.packaging_id=f.packaging_id
    WHERE f.line_id=:line_id AND f.ts_min BETWEEN :from_ts AND :to_ts
    GROUP BY DATE_FORMAT(f.ts_min,'%Y-%m')
    ORDER BY bucket
    """

    sql_quarter = """
    SELECT CONCAT(YEAR(f.ts_min),'-Q',QUARTER(f.ts_min)) AS bucket,
      SUM(f.produced) AS produced, SUM(f.good) AS good, SUM(f.ng) AS ng,
      ROUND(100.0*SUM(f.runtime_sec)/NULLIF(SUM(f.planned_sec),0),2) AS availability_pct,
      ROUND(100.0*SUM(f.produced)/NULLIF(SUM((f.runtime_sec)/60.0)*COALESCE(dp.ideal_rate_per_min,0)),2) AS performance_pct,
      ROUND(100.0*SUM(f.good)/NULLIF(SUM(f.produced),0),2) AS quality_pct,
      ROUND(
        (SUM(f.runtime_sec)/NULLIF(SUM(f.planned_sec),0)) *
        (SUM(f.produced)/NULLIF(SUM((f.runtime_sec)/60.0)*COALESCE(dp.ideal_rate_per_min,0)),0)) *
        (SUM(f.good)/NULLIF(SUM(f.produced),0)) * 100, 2
      ) AS oee_pct
    FROM fact_production_min f
    LEFT JOIN dim_packaging dp ON dp.packaging_id=f.packaging_id
    WHERE f.line_id=:line_id AND f.ts_min BETWEEN :from_ts AND :to_ts
    GROUP BY YEAR(f.ts_min), QUARTER(f.ts_min)
    ORDER BY YEAR(f.ts_min), QUARTER(f.ts_min)
    """

    sql_year = """
    SELECT YEAR(f.ts_min) AS bucket,
      SUM(f.produced) AS produced, SUM(f.good) AS good, SUM(f.ng) AS ng,
      ROUND(100.0*SUM(f.runtime_sec)/NULLIF(SUM(f.planned_sec),0),2) AS availability_pct,
      ROUND(100.0*SUM(f.produced)/NULLIF(SUM((f.runtime_sec)/60.0)*COALESCE(dp.ideal_rate_per_min,0)),2) AS performance_pct,
      ROUND(100.0*SUM(f.good)/NULLIF(SUM(f.produced),0),2) AS quality_pct,
      ROUND(
        (SUM(f.runtime_sec)/NULLIF(SUM(f.planned_sec),0)) *
        (SUM(f.produced)/NULLIF(SUM((f.runtime_sec)/60.0)*COALESCE(dp.ideal_rate_per_min,0)),0)) *
        (SUM(f.good)/NULLIF(SUM(f.produced),0)) * 100, 2
      ) AS oee_pct
    FROM fact_production_min f
    LEFT JOIN dim_packaging dp ON dp.packaging_id=f.packaging_id
    WHERE f.line_id=:line_id AND f.ts_min BETWEEN :from_ts AND :to_ts
    GROUP BY YEAR(f.ts_min)
    ORDER BY YEAR(f.ts_min)
    """

    # Run queries
    with ENGINE.connect() as conn:
        df_series = pd.read_sql(text(sql_s), conn, params=p_s)
        df_gauges = pd.read_sql(text(sql_g), conn, params=p_g)
        df_shift  = pd.read_sql(text(sql_shift), conn, params=p_common)
        df_day    = pd.read_sql(text(sql_day), conn, params=p_common)
        df_week   = pd.read_sql(text(sql_week), conn, params=p_common)
        df_month  = pd.read_sql(text(sql_month), conn, params=p_common)
        df_quarter= pd.read_sql(text(sql_quarter), conn, params=p_common)
        df_year   = pd.read_sql(text(sql_year), conn, params=p_common)

    data   = _df_to_records(df_series)
    gauges = _df_to_records(df_gauges)
    gauges = gauges[0] if gauges else {}

    payload = {
        "meta": {
            "mode":"dashboard",
            "line_id":line_id,
            "from_ts":dt_from.strftime("%Y-%m-%d %H:%M:%S"),
            "to_ts":dt_to.strftime("%Y-%m-%d %H:%M:%S"),
            "process_order":q.process_order,
            "packaging_id":q.packaging_id,
            "row_count":len(data),
            "limit":limit,
            "bucket_sec":bucket_used,
            "generated_at":now.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "gauges":gauges,
        "linechart":data,
        "shift_summary":_df_to_records(df_shift),
        "day_summary":_df_to_records(df_day),
        "week_summary":_df_to_records(df_week),
        "month_summary":_df_to_records(df_month),
        "quarter_summary":_df_to_records(df_quarter),
        "year_summary":_df_to_records(df_year)
    }

    CACHE.set(key,payload)
    return JSONResponse(payload)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_oee:app", host="0.0.0.0", port=8088, reload=False, workers=1)
