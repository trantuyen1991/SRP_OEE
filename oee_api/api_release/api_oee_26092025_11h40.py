"""
api_oee.py
==========

FastAPI mini-service cung cấp JSON cho dashboard AVEVA:
- Đọc dữ liệu OEE từ MySQL (ưu tiên VIEW theo granularity).
- Hỗ trợ filter: line_id, process_order, packaging_id.
- Hỗ trợ granularity: min | shift | day | week | month | year.
- Có cache in-memory TTL để giảm tải DB khi nhiều màn hình cùng truy cập.

Chạy:   uvicorn api_oee:app --host 0.0.0.0 --port 8088 --reload
Yêu cầu: fastapi, uvicorn, sqlalchemy, mysql-connector-python, pandas

Tác giả: <Bạn>
"""

import os
import time
import logging
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta, date, time as dtime

import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from decimal import Decimal
import numpy as np
# -----------------------------------------------------------------------------
# Cấu hình & logging
# -----------------------------------------------------------------------------

DB_URL = os.getenv("DB_URL", "mysql+mysqlconnector://root:root@127.0.0.1:3306/mpy_oee")
DEFAULT_TTL_SEC = int(os.getenv("CACHE_TTL_SEC", "30"))     # TTL cache
DEFAULT_SINCE_MIN = int(os.getenv("SINCE_MIN", "240"))      # min: mặc định 4 giờ gần nhất
MAX_ROWS = int(os.getenv("MAX_ROWS", "5000"))               # an toàn cho trả JSON

def _setup_logging() -> None:
    """Cấu hình logging chuẩn, có thể override LOG_LEVEL/LOG_FILE qua env."""
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_fmt = "%(asctime)s [api_oee] %(levelname)s: %(message)s"
    handlers = []
    log_file = os.getenv("LOG_FILE")
    if log_file:
        from logging.handlers import RotatingFileHandler
        handlers.append(RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3))
    else:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=getattr(logging, level, logging.INFO),
                        format=log_fmt, handlers=handlers)

_setup_logging()

# -----------------------------------------------------------------------------
# Kết nối DB & cache TTL đơn giản
# -----------------------------------------------------------------------------

def get_engine() -> Engine:
    """
    Tạo engine SQLAlchemy. Dùng pool_pre_ping để tự kiểm tra kết nối trước khi dùng.
    """
    logging.info("Creating engine to %s", DB_URL)
    return create_engine(DB_URL, pool_recycle=3600, pool_pre_ping=True)

ENGINE: Engine = get_engine()

class TTLCache:
    """
    Cache in-memory đơn giản với TTL (giây).
    Key là tuple tham số; value là (expire_epoch, data_dict).
    """
    def __init__(self, ttl_sec: int):
        self.ttl = ttl_sec
        self.store: Dict[Tuple, Tuple[float, Dict[str, Any]]] = {}

    def get(self, key: Tuple) -> Optional[Dict[str, Any]]:
        now = time.time()
        rec = self.store.get(key)
        if not rec:
            return None
        exp, val = rec
        if now > exp:
            # hết hạn
            self.store.pop(key, None)
            return None
        return val

    def set(self, key: Tuple, val: Dict[str, Any]) -> None:
        exp = time.time() + self.ttl
        self.store[key] = (exp, val)

CACHE = TTLCache(DEFAULT_TTL_SEC)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _view_name_for(line_id: int, gran: str) -> Optional[str]:
    """
    Trả về tên VIEW cho line và granularity; None nếu chưa có view tương ứng.
    Bạn đã có: v_oee_min_line105, v_oee_shift_line105, v_oee_day_line105, ...
    Khi mở rộng các line khác, chỉ cần tạo view cùng quy ước tên.
    """
    allowed = {"min", "shift", "day", "week", "month", "year"}
    if gran not in allowed:
        return None
    if line_id == 105:
        return f"v_oee_{gran}_line105"
    # TODO: thêm các line khác khi có view
    return None

def _build_sql(line_id: int,
               gran: str,
               process_order: Optional[str],
               packaging_id: Optional[int],
               since_min: Optional[int],
               limit: int) -> Tuple[str, Dict[str, Any]]:
    """
    Tạo SQL lấy dữ liệu cho API.
    - gran='min': nếu có view -> đọc view; thêm since_min để giới hạn thời gian.
    - gran∈{shift,day,week,month,year}:
        * Nếu KHÔNG có filter (po/pkg) và có view -> dùng view gộp (nhanh).
        * Nếu CÓ filter -> tổng hợp trực tiếp từ fact_production_min (áp filter trước khi gộp).
    """
    params: Dict[str, Any] = {}
    # -------------------- MINUTE --------------------
    if gran == "min":
        view = _view_name_for(line_id, "min")
        if view:
            where = ["1=1"]
            if process_order:
                where.append("process_order = :po"); params["po"] = process_order
            if packaging_id is not None:
                where.append("packaging_id = :pkg"); params["pkg"] = packaging_id
            if since_min and since_min > 0:
                where.append("ts_min >= :since_ts")
                params["since_ts"] = (datetime.now() - timedelta(minutes=since_min)).strftime("%Y-%m-%d %H:%M:00")
            sql = f"SELECT * FROM {view} WHERE {' AND '.join(where)} ORDER BY ts_min LIMIT :limit"
            params["limit"] = min(limit, MAX_ROWS)
            return sql, params

        # Fallback min (không view)
        where = ["line_id = :line_id"]; params["line_id"] = line_id
        if process_order:
            where.append("process_order = :po"); params["po"] = process_order
        if packaging_id is not None:
            where.append("packaging_id = :pkg"); params["pkg"] = packaging_id
        if since_min and since_min > 0:
            where.append("ts_min >= :since_ts")
            params["since_ts"] = (datetime.now() - timedelta(minutes=since_min)).strftime("%Y-%m-%d %H:%M:00")
        where_sql = " WHERE " + " AND ".join(where)
        sql = f"""
        SELECT
          f.ts_min, f.line_id, f.process_order, f.packaging_id,
          f.produced, f.good, f.ng, f.runtime_sec, f.planned_sec,
          ROUND(100.0 * (CASE WHEN f.planned_sec>0 THEN f.runtime_sec/f.planned_sec ELSE 0 END),2) AS availability_pct,
          ROUND(100.0 * (CASE WHEN f.runtime_sec>0 AND COALESCE(dp.ideal_rate_per_min,0)>0
                              THEN f.produced/((f.runtime_sec/60.0)*dp.ideal_rate_per_min) ELSE 0 END),2) AS performance_pct,
          ROUND(100.0 * (CASE WHEN f.produced>0 THEN f.good/f.produced ELSE 0 END),2) AS quality_pct,
          ROUND(
            (CASE WHEN f.planned_sec>0 THEN f.runtime_sec/f.planned_sec ELSE 0 END) *
            (CASE WHEN f.runtime_sec>0 AND COALESCE(dp.ideal_rate_per_min,0)>0
                  THEN f.produced/((f.runtime_sec/60.0)*dp.ideal_rate_per_min) ELSE 0 END) *
            (CASE WHEN f.produced>0 THEN f.good/f.produced ELSE 0 END) * 100
          ,2) AS oee_pct
        FROM fact_production_min f
        LEFT JOIN dim_packaging dp ON dp.packaging_id = f.packaging_id
        {where_sql}
        ORDER BY f.ts_min
        LIMIT :limit
        """
        params["limit"] = min(limit, MAX_ROWS)
        return sql, params

    # -------------------- AGGREGATES --------------------
    # Nếu có view và KHÔNG có filter → dùng view
    view = _view_name_for(line_id, gran)
    if view and not process_order and packaging_id is None:
        order_sql = " ORDER BY 1"
        sql = f"SELECT * FROM {view}{order_sql} LIMIT :limit"
        return sql, {"limit": min(limit, MAX_ROWS)}

    # Có filter → tổng hợp trực tiếp từ fact_production_min
    base_where = ["f.line_id = :line_id"]; params["line_id"] = line_id
    if process_order:
        base_where.append("f.process_order = :po"); params["po"] = process_order
    if packaging_id is not None:
        base_where.append("f.packaging_id = :pkg"); params["pkg"] = packaging_id
    where_sql = " WHERE " + " AND ".join(base_where)

    # Chọn group theo gran
    if gran == "shift":
        # cần join dim_shift_calendar
        sql = f"""
        SELECT
          s.shift_date, s.shift_no,
          {line_id} AS line_id,
          MIN(f.ts_min) AS from_ts, MAX(f.ts_min) AS to_ts,
          SUM(f.produced) AS produced,
          SUM(f.good) AS good,
          SUM(f.ng) AS ng,
          SUM(f.runtime_sec) AS runtime_sec,
          SUM(f.planned_sec) AS planned_sec,
          ROUND(100.0 * SUM(f.runtime_sec)/NULLIF(SUM(f.planned_sec),0),2) AS availability_pct,
          ROUND(100.0 * SUM(f.produced)
                / NULLIF(SUM((f.runtime_sec/60.0) * COALESCE(dp.ideal_rate_per_min,0)),0),2) AS performance_pct,
          ROUND(100.0 * SUM(f.good)/NULLIF(SUM(f.produced),0),2) AS quality_pct,
          ROUND(
            (SUM(f.runtime_sec)/NULLIF(SUM(f.planned_sec),0)) *
            (SUM(f.produced)/NULLIF(SUM((f.runtime_sec/60.0)*COALESCE(dp.ideal_rate_per_min,0)),0)) *
            (SUM(f.good)/NULLIF(SUM(f.produced),0)) * 100
          ,2) AS oee_pct
        FROM dim_shift_calendar s
        JOIN fact_production_min f
          ON f.ts_min >= CONCAT(s.shift_date,' ',s.start_time)
         AND f.ts_min <  DATE_ADD(CONCAT(s.shift_date,' ',s.end_time),
                      INTERVAL (s.end_time < s.start_time) DAY)
        LEFT JOIN dim_packaging dp ON dp.packaging_id = f.packaging_id
        {where_sql}
        GROUP BY s.shift_date, s.shift_no
        ORDER BY s.shift_date, s.shift_no
        LIMIT :limit
        """
        params["limit"] = min(limit, MAX_ROWS)
        return sql, params

    if gran == "day":
        group_expr = "DATE(f.ts_min)"
        order_expr = "DATE(f.ts_min)"
    elif gran == "week":
        group_expr = "YEARWEEK(f.ts_min,3)"
        order_expr = "YEARWEEK(f.ts_min,3)"
    elif gran == "month":
        group_expr = "DATE_FORMAT(f.ts_min,'%Y-%m')"
        order_expr = "DATE_FORMAT(f.ts_min,'%Y-%m')"
    elif gran == "year":
        group_expr = "YEAR(f.ts_min)"
        order_expr = "YEAR(f.ts_min)"
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported gran='{gran}'")

    sql = f"""
    SELECT
      {group_expr} AS bucket,
      {line_id} AS line_id,
      SUM(f.produced) AS produced,
      SUM(f.good) AS good,
      SUM(f.ng) AS ng,
      SUM(f.runtime_sec) AS runtime_sec,
      SUM(f.planned_sec) AS planned_sec,
      ROUND(100.0 * SUM(f.runtime_sec)/NULLIF(SUM(f.planned_sec),0),2) AS availability_pct,
      ROUND(100.0 * SUM(f.produced)
            / NULLIF(SUM((f.runtime_sec/60.0) * COALESCE(dp.ideal_rate_per_min,0)),0),2) AS performance_pct,
      ROUND(100.0 * SUM(f.good)/NULLIF(SUM(f.produced),0),2) AS quality_pct,
      ROUND(
        (SUM(f.runtime_sec)/NULLIF(SUM(f.planned_sec),0)) *
        (SUM(f.produced)/NULLIF(SUM((f.runtime_sec/60.0)*COALESCE(dp.ideal_rate_per_min,0)),0)) *
        (SUM(f.good)/NULLIF(SUM(f.produced),0)) * 100
      ,2) AS oee_pct
    FROM fact_production_min f
    LEFT JOIN dim_packaging dp ON dp.packaging_id = f.packaging_id
    {where_sql}
    GROUP BY {group_expr}
    ORDER BY {order_expr}
    LIMIT :limit
    """
    params["limit"] = min(limit, MAX_ROWS)
    return sql, params

def _df_to_records(df: pd.DataFrame) -> list[dict]:
    """
    Chuyển DataFrame -> list[dict] an toàn cho JSON.
    - Chuẩn hóa: datetime -> "YYYY-mm-dd HH:MM:SS"
                  date     -> "YYYY-mm-dd"
                  time     -> "HH:MM:SS"
                  Decimal/numpy types -> float/int/str phù hợp.
    """
    if df.empty:
        return []

    out = df.copy()

    # 1) Datetime64 columns -> string
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d %H:%M:%S")

    # 2) Object columns: map từng ô để convert date/time/Decimal/numpy types
    def _json_safe(x):
        if isinstance(x, datetime):
            return x.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(x, date):
            return x.strftime("%Y-%m-%d")
        if isinstance(x, dtime):
            return x.strftime("%H:%M:%S")
        if isinstance(x, Decimal):
            # đổi sang float; nếu cần giữ chính xác tuyệt đối, dùng str(x)
            return float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        return x

    obj_cols = [c for c in out.columns if out[c].dtype == object]
    for c in obj_cols:
        out[c] = out[c].map(_json_safe)

    return out.to_dict(orient="records")

# -----------------------------------------------------------------------------
# FastAPI app & endpoints
# -----------------------------------------------------------------------------

# app = FastAPI(title="OEE Mini API", version="0.1.0",
#               description="API JSON (cache) phục vụ AVEVA Edge & client khác")

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="OEE Mini API", version="0.1.0",
              description="API JSON (cache) phục vụ AVEVA Edge & client khác")

# CORS — cho phép gọi từ WebView/localhost (đơn giản: *). 
ALLOWED_ORIGINS = [
    "http://localhost",
    "http://localhost:15509",     # nếu AVEVA/Live Server dùng port này
    "http://127.0.0.1",
    "http://127.0.0.1:5500",      # <-- Live Server của bạn
    "http://192.168.58.6",        # máy AVEVA/browser trong LAN
    "http://192.168.58.6:8088",   # (không bắt buộc, nhưng thêm cũng ok)
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/health")
def health() -> Dict[str, Any]:
    """Kiểm tra sức khỏe dịch vụ & DB."""
    try:
        with ENGINE.connect() as conn:
            row = conn.execute(text("SELECT NOW() AS now_ts, DATABASE() AS db")).mappings().first()
        return {"ok": True, "db_time": str(row["now_ts"]), "db": row["db"]}
    except Exception as e:
        logging.error("Health DB error: %s", e, exc_info=True)
        return {"ok": False, "error": str(e)}
@app.get("/oee/line/{line_id}")
def get_oee(
    line_id: int,
    gran: str = Query("min", pattern="^(min|shift|day|week|month|year)$",
                      description="Granularity: min|shift|day|week|month|year"),
    process_order: Optional[str] = Query(None, description="Filter theo process order"),
    packaging_id: Optional[int] = Query(None, description="Filter theo packaging_id"),
    since_min: Optional[int] = Query(DEFAULT_SINCE_MIN, ge=1, description="Chỉ áp dụng cho gran=min"),
    limit: int = Query(2000, ge=1, le=MAX_ROWS, description="Giới hạn số dòng trả về")
) -> JSONResponse:
    """
    Trả về dữ liệu OEE theo tham số lọc.
    - Ưu tiên đọc từ VIEW nếu tồn tại (ví dụ v_oee_min_line105).
    - Có cache TTL để giảm tải DB cho cùng một tham số trong khoảng thời gian ngắn.
    """
    key = (line_id, gran, process_order, packaging_id, since_min if gran == "min" else None, limit)
    cached = CACHE.get(key)
    if cached:
        return JSONResponse(cached)

    try:
        sql, params = _build_sql(line_id, gran, process_order, packaging_id, since_min, limit)
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error("Build SQL error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Build SQL error")

    try:
        with ENGINE.connect() as conn:
            df = pd.read_sql(text(sql), conn, params=params)
    except Exception as e:
        logging.error("DB query error: %s | sql=%s | params=%s", e, sql, params, exc_info=True)
        raise HTTPException(status_code=500, detail="DB query error")

    data = _df_to_records(df)
    payload = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "line_id": line_id,
            "gran": gran,
            "process_order": process_order,
            "packaging_id": packaging_id,
            "since_min": since_min if gran == "min" else None,
            "row_count": len(data),
            "limit": limit,
            "from_view": _view_name_for(line_id, gran) is not None,
            # ép kiểu JSON-safe
            "sql": str(sql),                 # ép TextClause -> string
            "params": {k: str(v) for k, v in params.items()}  # ép mọi value -> string
        },
        "data": data
    }


    CACHE.set(key, payload)
    return JSONResponse(payload)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_oee:app", host="0.0.0.0", port=8088, reload=False, workers=1)
