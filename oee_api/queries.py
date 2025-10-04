# =============================
# ======== queries.py =========
# =============================
"""
Query orchestration:
- Chooses correct SQL by granularity
- Applies filters (line, po, pkg, shift_no)
- Executes via db.query_df
- Computes OEE (A/P/Q/OEE) for gauges and per-bucket rows
- Shapes payload according to `include`
"""
from typing import Optional, List, Tuple
from datetime import datetime, timedelta
from fastapi import HTTPException
import logging
import pandas as pd

from oee_api.config import SETTINGS
from oee_api.db import query_df
from oee_api.utils import (
    df_to_records,
    sanitize_json_deep,
    choose_bucket_seconds,
    now_local,
    to_iso8601,
    to_local_str,
    compute_oee,
    choose_bucket_seconds_auto,
    parse_csv_int,
    TTLCache,
    previous_window,
    CompareMode,
)
from oee_api import sql_texts as SQL
import math
import os
import numpy as np

CACHE_TTL_SEC = int(os.getenv("CACHE_TTL_SEC", "30"))   # TTL mặc định 30s
CACHE_MAX_KEYS = int(os.getenv("CACHE_MAX_KEYS", "1024"))
CACHE_WINDOW_SEC = int(os.getenv("CACHE_WINDOW_SEC", "10"))   # 10 giây là hợp lý cho dashboard

CACHE_HIT   = 0
CACHE_MISS  = 0

CACHE = TTLCache(ttl_sec=CACHE_TTL_SEC, maxsize=CACHE_MAX_KEYS)

logger = logging.getLogger("oee.queries")

def _int_or_none(x):
    try:
        if x is None:
            return None
        # pandas/float NaN
        if isinstance(x, float) and math.isnan(x):
            return None
        if 'pandas' in str(type(x)) and pd.isna(x):
            return None
        return int(x)
    except Exception:
        return None

def _is_nan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)

def _nz_int(x) -> int:
    if x is None or _is_nan(x):
        return 0
    try:
        return int(x)
    except Exception:
        return 0

def _nz_float(x) -> float:
    if x is None or _is_nan(x):
        return 0.0
    try:
        return float(x)
    except Exception:
        return 0.0

def _ideal_or_none(x):
    """Trả float nếu hợp lệ, còn lại None (kể cả NaN, 0 hoặc âm)."""
    if x is None or _is_nan(x):
        return None
    try:
        v = float(x)
    except Exception:
        return None
    return v if v > 0 else None

def _assemble_where(
    dt_from: datetime,
    dt_to: datetime,
    line_id: int,
    po: Optional[str],
    pkg: Optional[int],
    shift_no: Optional[int],
    shift_join: bool,
) -> Tuple[str, list, str]:
    """Create WHERE + params list respecting plant-level and optional filters."""
    parts = ["WHERE f.ts_min BETWEEN ? AND ?"] 
    params: list = [dt_from.strftime("%Y-%m-%d %H:%M:%S"), dt_to.strftime("%Y-%m-%d %H:%M:%S")]  # inclusive to seconds

    if line_id > 0:
        parts.append(SQL.LINE_FILTER)
        params.append(line_id)

    if po:
        parts.append(SQL.PO_FILTER)
        params.append(po)
    if pkg:
        parts.append(SQL.PKG_FILTER)
        params.append(pkg)

    if shift_no is not None:
        # shift filter requires CROSS JOIN dim_shift_calendar
        parts.append(SQL.SHIFT_TOD_WHERE)
        params.append(shift_no)

    where = "\n".join(parts)
    sj = SQL.SHIFT_CROSS_JOIN if shift_join or (shift_no is not None) else ""
    return where, params, sj

def _floor_dt(dt, sec: int):
    if sec <= 0:
        return dt.replace(microsecond=0)
    ts = int(dt.timestamp())
    floored = ts - (ts % sec)
    return dt.__class__.fromtimestamp(floored, tz=dt.tzinfo)

def _merge_segments(segs: list[dict]) -> list[dict]:
    """Merge adjacent segments with same state_id/reason_id."""
    if not segs:
        return []
    merged = []
    prev = segs[0].copy()
    for cur in segs[1:]:
        if (cur["line_id"] == prev["line_id"]
            and cur["state_id"] == prev["state_id"]
            and (cur.get("reason_id") or 0) == (prev.get("reason_id") or 0)
            and cur["start_ts"] == prev["end_ts"]):
            # nối tiếp -> merge
            prev["end_ts"] = cur["end_ts"]
            prev["duration_sec"] = (
                (prev["end_ts"] - prev["start_ts"]).total_seconds()
            )
        else:
            merged.append(prev)
            prev = cur.copy()
    merged.append(prev)
    return merged

def _build_gantt_summary(df_seg):
    # Query toàn bộ state 1 lần (cache lại để dùng nhiều lần)
    df_states = query_df("SELECT state_id, state_code FROM dim_state ORDER BY state_id")
    ALL_STATES = df_states.to_dict(orient="records")

    summary = {}
    if not df_seg.empty:
        # Group by state_id: tính tổng duration và số lần xuất hiện
        gp = df_seg.groupby("state_id").agg(
            total_duration=("duration_sec", "sum"),
            count=("state_id", "count")
        ).to_dict(orient="index")

        # Điền vào đủ tất cả state
        for s in ALL_STATES:
            sid = s["state_id"]
            scode = s["state_code"]
            if sid in gp:
                summary[sid] = {
                    "state_code": scode,
                    "duration": float(gp[sid]["total_duration"]),
                    "count": int(gp[sid]["count"]),
                }
            else:
                summary[sid] = {
                    "state_code": scode,
                    "duration": 0.0,
                    "count": 0,
                }
    else:
        # Nếu không có segment nào -> fill tất cả = 0
        for s in ALL_STATES:
            summary[s["state_id"]] = {
                "state_code": s["state_code"],
                "duration": 0.0,
                "count": 0,
            }

    return summary

async def build_payload(
    line_id: int,
    dt_from: datetime,
    dt_to: datetime,
    gran: str,
    limit: int,
    bucket_sec: Optional[int],
    po: Optional[str],
    pkg: Optional[int],
    shift_no: Optional[int],
    include: List[str],
) -> dict:
    """Build the JSON payload for the endpoint.
    Returns dict with keys: meta, gauges?, linechart?, summaries?
    """
    span_seconds = int((dt_to - dt_from).total_seconds())

    # --- pick bucket for linechart (auto density) ---
    auto_bucket_sec = choose_bucket_seconds_auto(span_seconds, limit)
    logger.info("series:auto gran, window=%ss, limit=%s -> bucket_sec=%s", span_seconds, limit, auto_bucket_sec)

    # --- Compose cache key -------------------------------------------------------
    # include có thể là "gauges,linechart" -> chuẩn hóa thành tuple có thứ tự ổn định
    _inc = tuple(sorted(p.strip() for p in include if p.strip()))

    # --- floor thời điểm kết thúc để tăng tỷ lệ HIT ---
    dt_to_key = _floor_dt(dt_to, CACHE_WINDOW_SEC)
    # (tuỳ chọn) nếu muốn chắc ăn hơn, có thể floor cả dt_from khi scope=minute/between
    dt_from_key = _floor_dt(dt_from, CACHE_WINDOW_SEC)  
    # dt_from_key = dt_from
    cache_key = (
        "oee.v2",                # thay đổi khi bạn đổi format payload -> tự vô hiệu cache cũ
        int(line_id or 0),
        gran or "",
        dt_from_key.isoformat(),
        dt_to_key.isoformat(),
        int(limit or 0),
        po or "", int(pkg or 0), int(shift_no or 0),
        _inc,
    )
    cached = CACHE.get(cache_key)
    # logger.info("cached=%s  cache_key=%s", cached, cache_key)
    if cached is not None:
        global CACHE_HIT
        CACHE_HIT += 1
        # logger.info("CACHE HIT=%s  key=%s from=%s to=%s include=%s",
                    # CACHE_HIT, cache_key[:3], to_local_str(dt_from), to_local_str(dt_to), _inc)
        return cached                      

    else:
        global CACHE_MISS
        CACHE_MISS += 1
        # logger.info("CACHE MISS=%s  key=%s", CACHE_MISS, cache_key[:3])

    meta = {
        "generated_at": to_local_str(now_local()),
        "params": {
            "line_id": line_id,
            "from": to_local_str(dt_from),
            "to": to_local_str(dt_to),
            "gran": gran,
            "limit": limit,
            "bucket_sec": bucket_sec,
            "po": po,
            "pkg": pkg,
            "shift_no": shift_no,
        }
    }

    out = {"meta": meta}

    # --------- decide SQL by gran ---------
    if gran in ("min", "hour"):
        bsec = choose_bucket_seconds(span_seconds, limit, gran, bucket_sec)
        where, params, shift_join = _assemble_where(dt_from, dt_to, line_id, po, pkg, shift_no, shift_join=False)
        line_expr = "?" if line_id == 0 else "f.line_id"
        series_sql = SQL.SERIES_MIN_HOUR.format(
            line_id_expr=line_expr,
            shift_join=shift_join,
            where=where,
        )
        # params: bucket_sec, bucket_sec, [from,to,(line),(po),(pkg),(shift)], limit
        sparams = [bsec, bsec] + params
        if line_id == 0:
            sparams.append(0)  # placeholder to project line_id=0
        sparams += [limit]

    elif gran in ("day", "week", "month", "quarter", "year"):
        where, params, shift_join = _assemble_where(dt_from, dt_to, line_id, po, pkg, shift_no, shift_join=False)
        line_expr = "?" if line_id == 0 else "f.line_id"
        series_sql = SQL.SERIES_BY_CAL[gran].format(
            line_id_expr=line_expr,
            shift_join=shift_join,
            where=where,
        )
        sparams = params.copy()
        if line_id == 0:
            sparams.append(0)
        sparams += [limit]
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported gran: {gran}")

    # --------- execute series SQL ---------
    logger.info("series SQL gran=%s", gran)
    series_df = query_df(series_sql, sparams)
    series_df = compute_oee(series_df)
    # --------- Gauges (aggregate over the window) ---------
    if "gauges" in include:
        # --- Gauges (aggregate over the window) ---
        logger.debug("gauges window: %s .. %s", dt_from, dt_to)
        gw, gparams, gsj = _assemble_where(dt_from, dt_to, line_id, po, pkg, shift_no, shift_join=False)

        lid = int(line_id or 0)
        line_expr = "0" if lid == 0 else "f.line_id"

        sql = SQL.GAUGES.format(line_id_expr=line_expr, shift_join=gsj, where=gw)
        params = gparams.copy()   # KHÔNG append 0 nữa

        # logger.info("SQL:\n%s", sql)
        # logger.info("PARAMS: %s", params)

        gauges_df = query_df(sql, params)

        if "reject" not in gauges_df.columns and "ng" in gauges_df.columns:
            gauges_df = gauges_df.rename(columns={"ng": "reject"})

        # helpers an toàn hơn (xem mục 2)
        def _sum_int(df, col):
            if col not in df.columns:
                return 0
            s = pd.to_numeric(df[col], errors="coerce")
            return int(s.fillna(0).astype("int64").sum())

        def _sum_float(df, col):
            if col not in df.columns:
                return 0.0
            s = pd.to_numeric(df[col], errors="coerce")
            return float(s.fillna(0).astype("float64").sum())

        if gauges_df.empty:
            totals = {
                "good": 0,
                "reject": 0,
                "runtime_sec": 0,
                "downtime_sec": 0,
                "planned_sec": 0,
                "ideal_capacity_cnt": 0.0,
            }
        else:
            good         = _sum_int(gauges_df, "good")
            reject       = _sum_int(gauges_df, "reject") if "reject" in gauges_df.columns else _sum_int(gauges_df, "ng")
            runtime_sec  = _sum_int(gauges_df, "runtime_sec")
            downtime_sec = _sum_int(gauges_df, "downtime_sec")
            planned_sec  = _sum_int(gauges_df, "planned_sec")
            ideal_cap    = _sum_float(gauges_df, "ideal_capacity_cnt")

            totals = {
                "good": good,
                "reject": reject,
                "runtime_sec": runtime_sec,
                "downtime_sec": downtime_sec,
                "planned_sec": planned_sec,
                "ideal_capacity_cnt": ideal_cap,
            }

        one = pd.DataFrame([totals])        # 1 hàng tổng để compute_oee()
        one_oee = compute_oee(one).iloc[0].to_dict()
        gauges = {
            "line_id": int(line_id or 0),
            **totals,
            "availability": float(one_oee.get("availability", 0.0)),
            "performance":  float(one_oee.get("performance", 0.0)),
            "quality":      float(one_oee.get("quality", 0.0)),
            "oee":          float(one_oee.get("oee", 0.0)),
        }
        out["gauges"] = gauges

    # ---- LINECHART (auto-bucket by time window) ----
    if "linechart" in include:
        # compose WHERE and params như bạn đang làm (dựa vào dt_from/dt_to/line_id/po/pkg/shift_no)
        where_sql, where_params, shift_join = _assemble_where(
            dt_from=dt_from, dt_to=dt_to,
            line_id=line_id, po=po, pkg=pkg, shift_no=shift_no, shift_join=False
        )

        # chọn biểu thức line_id cho plant-level vs line-level
        line_id_expr = "0" if line_id == 0 else str(int(line_id))

        # lắp template
        sql = SQL.SERIES_AUTO.format(
            line_id_expr=line_id_expr,
            shift_join=shift_join,
            where=where_sql,
        )

        params = [auto_bucket_sec, auto_bucket_sec] + where_params + [int(limit)]
        logger.debug("run SERIES_AUTO: bucket_sec=%s, limit=%s, params=%s", auto_bucket_sec, limit, where_params)

        linechart_df = query_df(sql, params)
        logger.info("linechart points=%s (before normalize)", len(linechart_df))

        # tính lại các chỉ số OEE cho từng bucket
        linechart_df = compute_oee(linechart_df)
        
        out["linechart"]= df_to_records(linechart_df)

    # -------- Race (downtime top reasons) --------
    if "race" in include:
        params = {
            "line_id": line_id,          # 0 = plant, >0 = 1 line
            "from_ts": dt_from,          # datetime (UTC/local tuỳ chuẩn)
            "to_ts": dt_to,
            "limit": limit or 10,
            "offset": 0,
        }

        race_df = query_df(SQL.RACE, params)   # top 10
        race = []
        total_downtime = race_df["total_downtime_sec"].sum() if not race_df.empty else 0
        for _, r in race_df.iterrows():
            race.append({
                "reason_id": _int_or_none(r.get("reason_id")),
                "reason_name": r["reason_name"] or "Unknown",
                "reason_group": r["reason_group"] or "Unknown",
                "total_downtime_sec": int(r["total_downtime_sec"]),
                "percent": (r["total_downtime_sec"] / total_downtime * 100.0) if total_downtime > 0 else 0
            })
        out["race"] = race

    # -------- Gantt (timeline of states) --------
    if "gantt" in include:
        params = {
            "line_id": line_id,          # 0 = plant, >0 = 1 line
            "from_ts": dt_from,          # datetime (UTC/local tuỳ chuẩn)
            "to_ts": dt_to,
            "limit": limit or 50,
            "offset": 0,
        }
        gantt_df = query_df(SQL.GANTT, params)
        gantt = []
        for _, r in gantt_df.iterrows():
            gantt.append({
                "event_id": _int_or_none(r["event_id"]),
                "line_id": _int_or_none(r["line_id"]),
                "state_id": _int_or_none(r["state_id"]),
                "reason_id": _int_or_none(r["reason_id"]) if r["reason_id"] else None,
                "start_ts": str(r["start_ts"]),
                "end_ts": str(r["end_ts"]),
                "duration_sec": _int_or_none(r["duration_sec"]),
                "note": r["note"] or ""
            })
        out["gantt"] = gantt

    # --------- summaries (basic counts by gran for convenience) ---------
    if "summaries" in include:
        # simple derived summaries from the already computed series
        if series_df.empty:
            out["summaries"] = {"points": 0, "total_good": 0, "total_reject": 0}
        else:
            out["summaries"] = {
                "points": int(series_df.shape[0]),
                "total_good": int(series_df["good"].fillna(0).sum()),
                "total_reject": int(series_df["reject"].fillna(0).sum()),
            }
    # --- Set cache ---------------------------------------------------------------
    CACHE.set(cache_key, out)
    logger.info("CACHE SET key=%s size=%s", cache_key[:3], len(out.get("linechart", [])) if isinstance(out.get("linechart"), list) else "-")
    return out

def _gauges_query(line_id, dt_from, dt_to, po, pkg, shift_no, shift_join=False):
    gw, gparams, gsj = _assemble_where(dt_from, dt_to, line_id, po, pkg, shift_no, shift_join)

    lid = int(line_id or 0)
    line_expr = "0" if lid == 0 else "f.line_id"

    sql = SQL.GAUGES.format(line_id_expr=line_expr, shift_join=gsj, where=gw)
    params = gparams.copy()   # KHÔNG append 0 nữa

    gauges_df = query_df(sql, params)

    if "reject" not in gauges_df.columns and "ng" in gauges_df.columns:
        gauges_df = gauges_df.rename(columns={"ng": "reject"})

    # helpers an toàn hơn (xem mục 2)
    def _sum_int(df, col):
        if col not in df.columns:
            return 0
        s = pd.to_numeric(df[col], errors="coerce")
        return int(s.fillna(0).astype("int64").sum())

    def _sum_float(df, col):
        if col not in df.columns:
            return 0.0
        s = pd.to_numeric(df[col], errors="coerce")
        return float(s.fillna(0).astype("float64").sum())

    if gauges_df.empty:
        totals = {
            "good": 0,
            "reject": 0,
            "runtime_sec": 0,
            "downtime_sec": 0,
            "planned_sec": 0,
            "ideal_capacity_cnt": 0.0,
        }
    else:
        good         = _sum_int(gauges_df, "good")
        reject       = _sum_int(gauges_df, "reject") if "reject" in gauges_df.columns else _sum_int(gauges_df, "ng")
        runtime_sec  = _sum_int(gauges_df, "runtime_sec")
        downtime_sec = _sum_int(gauges_df, "downtime_sec")
        planned_sec  = _sum_int(gauges_df, "planned_sec")
        ideal_cap    = _sum_float(gauges_df, "ideal_capacity_cnt")

        totals = {
            "good": good,
            "reject": reject,
            "runtime_sec": runtime_sec,
            "downtime_sec": downtime_sec,
            "planned_sec": planned_sec,
            "ideal_capacity_cnt": ideal_cap,
        }

    one = pd.DataFrame([totals])        # 1 hàng tổng để compute_oee()
    one_oee = compute_oee(one).iloc[0].to_dict()
    gauges = {
        "line_id": int(line_id or 0),
        **totals,
        "availability": float(one_oee.get("availability", 0.0)),
        "performance":  float(one_oee.get("performance", 0.0)),
        "quality":      float(one_oee.get("quality", 0.0)),
        "oee":          float(one_oee.get("oee", 0.0)),
    }

    return gauges

def _safe_div(n, d):
    try:
        n = float(n)
        d = float(d)
        if d == 0:
            return 0.0
        return n / d
    except Exception:
        return 0.0

def _calc_apq_from_row(r: dict) -> tuple[float, float, float, float]:
    """
    Lấy A/P/Q/OEE từ 1 bucket. Ưu tiên dùng các cột đã có trong series.
    Nếu không có thì tính lại từ các cột thô.
    Trả về: (A, P, Q, OEE) tính theo %
    """
    # 1) Ưu tiên dùng column đã có
    if all(k in r for k in ("availability", "performance", "quality", "oee")):
        A = float(r.get("availability") or 0.0)
        P = float(r.get("performance") or 0.0)
        Q = float(r.get("quality") or 0.0)
        O = float(r.get("oee") or (A * P * Q / 10000.0))
        return A, P, Q, O

    # 2) Tính từ cột thô
    runtime_sec       = float(r.get("runtime_sec") or 0.0)
    planned_sec       = float(r.get("planned_sec") or 0.0)
    downtime_sec      = float(r.get("downtime_sec") or 0.0)
    good              = float(r.get("good") or 0.0)
    reject            = float(r.get("reject") or 0.0)
    ideal_capacity    = float(r.get("ideal_capacity_cnt") or 0.0)

    # Availability: thời gian chạy / (thời gian kế hoạch)
    # Chọn mẫu an toàn: planned = max(planned_sec, runtime_sec + downtime_sec)
    planned = max(planned_sec, runtime_sec + downtime_sec, 1.0)
    A = 100.0 * _safe_div(runtime_sec, planned)

    # Performance: output thực tế / output lý tưởng
    # Nếu bạn đang dùng công thức khác ở gauges, có thể thay block này để khớp 100%
    actual_output = good + reject
    P = 100.0 * _safe_div(actual_output, ideal_capacity) if ideal_capacity > 0 else 0.0

    # Quality: good / (good + reject)
    Q = 100.0 * _safe_div(good, actual_output) if actual_output > 0 else 0.0

    # OEE = A * P * Q / 10000
    O = (A * P * Q) / 10000.0
    return A, P, Q, O

def _clean_values(values: list[float]) -> list[float]:
    out = []
    for v in values:
        if v is None:
            continue
        try:
            f = float(v)
            if math.isfinite(f):
                out.append(f)
        except Exception:
            pass
    return out

def _series_request(line_id, gran, dt_from, dt_to, po, pkg, shift_no, limit=2000, shift_join=False):
    span_seconds = int((dt_to - dt_from).total_seconds())
        # --------- decide SQL by gran ---------
    if gran in ("min", "hour"):
        bsec = choose_bucket_seconds(span_seconds, limit, gran)
        where, params, shift_join = _assemble_where(dt_from, dt_to, line_id, po, pkg, shift_no, shift_join)
        line_expr = "?" if line_id == 0 else "f.line_id"
        series_sql = SQL.SERIES_MIN_HOUR.format(
            line_id_expr=line_expr,
            shift_join=shift_join,
            where=where,
        )
        # params: bucket_sec, bucket_sec, [from,to,(line),(po),(pkg),(shift)], limit
        sparams = [bsec, bsec] + params
        if line_id == 0:
            sparams.append(0)  # placeholder to project line_id=0
        sparams += [limit]

    elif gran in ("day", "week", "month", "quarter", "year"):
        where, params, shift_join = _assemble_where(dt_from, dt_to, line_id, po, pkg, shift_no, shift_join=False)
        line_expr = "?" if line_id == 0 else "f.line_id"
        series_sql = SQL.SERIES_BY_CAL[gran].format(
            line_id_expr=line_expr,
            shift_join=shift_join,
            where=where,
        )
        sparams = params.copy()
        if line_id == 0:
            sparams.append(0)
        sparams += [limit]
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported gran: {gran}")

    # --------- execute series SQL ---------
    logger.info("series SQL gran=%s", gran)
    series_df = query_df(series_sql, sparams)
    series_df = compute_oee(series_df)

    return series_df

def _linechart_query(line_id, dt_from, dt_to, po, pkg, shift_no, limit, bucket_mode = True):
    span_seconds = int((dt_to - dt_from).total_seconds())
    # --- pick bucket for linechart (auto density) ---
    if bucket_mode:
        auto_bucket_sec = choose_bucket_seconds_auto(span_seconds, limit)
    else:
        auto_bucket_sec = 60
    logger.info("series:auto gran, window=%ss, limit=%s -> bucket_sec=%s", span_seconds, limit, auto_bucket_sec)
        # compose WHERE and params như bạn đang làm (dựa vào dt_from/dt_to/line_id/po/pkg/shift_no)
    where_sql, where_params, shift_join = _assemble_where(
        dt_from=dt_from, dt_to=dt_to,
        line_id=line_id, po=po, pkg=pkg, shift_no=shift_no, shift_join=False
    )

    # chọn biểu thức line_id cho plant-level vs line-level
    line_id_expr = "0" if line_id == 0 else str(int(line_id))

    # lắp template
    sql = SQL.SERIES_AUTO.format(
        line_id_expr=line_id_expr,
        shift_join=shift_join,
        where=where_sql,
    )

    params = [auto_bucket_sec, auto_bucket_sec] + where_params + [int(limit)]
    logger.debug("run SERIES_AUTO: bucket_sec=%s, limit=%s, params=%s", auto_bucket_sec, limit, where_params)

    linechart_df = query_df(sql, params)
    # tính lại các chỉ số OEE cho từng bucket
    linechart_df = compute_oee(linechart_df)
    return linechart_df, auto_bucket_sec

def build_gauges_stats(line_id: int, gran: str, dt_from, dt_to, po, pkg, shift_no, limit=20000) -> dict:
    """
    Tính min/max/avg của A/P/Q/OEE theo từng bucket trong khoảng thời gian.
    Dùng series SQL (giống linechart) đang có sẵn.

    Trả về:
    {
      "oee": {"min": ..., "max": ..., "avg": ...},
      "availability": {...},
      "performance": {...},
      "quality": {...},
      "runtime_sec_per_bucket": {"avg": ..., "max": ...},
      "downtime_sec_per_bucket": {"avg": ..., "max": ...}
    }
    """
    # 1) Lấy series (giống logic linechart)
    df, bucket_sec = _linechart_query(line_id, dt_from, dt_to, po, pkg, shift_no, limit, bucket_mode = False)

    if df.empty:
        # thống nhất schema rỗng
        return {
            "oee": {"min": 0.0, "max": 0.0, "avg": 0.0},
            "availability": {"min": 0.0, "max": 0.0, "avg": 0.0},
            "performance": {"min": 0.0, "max": 0.0, "avg": 0.0},
            "quality": {"min": 0.0, "max": 0.0, "avg": 0.0},
            "runtime_sec_per_bucket": {"avg": 0.0, "max": 0.0},
            "downtime_sec_per_bucket": {"avg": 0.0, "max": 0.0},
        }

    # 2) Chuẩn hoá records
    rows = df.to_dict("records")

    list_A, list_P, list_Q, list_O = [], [], [], []
    list_runtime = []
    list_downtime = []

    for r in rows:
        A, P, Q, O = _calc_apq_from_row(r)
        list_A.append(A)
        list_P.append(P)
        list_Q.append(Q)
        list_O.append(O)
        list_runtime.append(float(r.get("runtime_sec") or 0.0))
        list_downtime.append(float(r.get("downtime_sec") or 0.0))

    list_A = _clean_values(list_A)
    list_P = _clean_values(list_P)
    list_Q = _clean_values(list_Q)
    list_O = _clean_values(list_O)

    list_runtime   = _clean_values(list_runtime)
    list_downtime  = _clean_values(list_downtime)

    def _stats(xs: list[float]) -> dict:
        if not xs:
            return {"min": 0.0, "max": 0.0, "avg": 0.0}
        return {
            "min": float(np.min(xs)),
            "max": float(np.max(xs)),
            "avg": float(np.mean(xs)),
        }

    out = {
        "points": int(df.shape[0]),
        "bucket_sec": int(bucket_sec),
        "oee":          _stats(list_O),
        "availability": _stats(list_A),
        "performance":  _stats(list_P),
        "quality":      _stats(list_Q),
        "runtime_sec_per_bucket": {
            "avg": float(np.mean(list_runtime)) if list_runtime else 0.0,
            "max": float(np.max(list_runtime))  if list_runtime else 0.0,
        },
        "downtime_sec_per_bucket": {
            "avg": float(np.mean(list_downtime)) if list_downtime else 0.0,
            "max": float(np.max(list_downtime))  if list_downtime else 0.0,
        },
    }
    return out

async def build_gauges_payload(
    line_id: int,
    dt_from: datetime,
    dt_to: datetime,
    gran: str,
    po: Optional[str],
    pkg: Optional[int],
    shift_no: Optional[int],
    compare: str,
    stats: str, 
) -> dict:
    """Build the JSON payload for the endpoint.
    Returns dict with keys: meta, gauges?, linechart?, summaries?
    """
    span_seconds = int((dt_to - dt_from).total_seconds())
    logger.info("series:auto gran, window=%ss", span_seconds)

    # --- Compose cache key -------------------------------------------------------
    # --- floor thời điểm kết thúc để tăng tỷ lệ HIT ---
    dt_to_key = _floor_dt(dt_to, CACHE_WINDOW_SEC)
    # (tuỳ chọn) nếu muốn chắc ăn hơn, có thể floor cả dt_from khi scope=minute/between
    dt_from_key = _floor_dt(dt_from, CACHE_WINDOW_SEC)  
    # dt_from_key = dt_from
    cache_key = (
        "oee.v2",                # thay đổi khi bạn đổi format payload -> tự vô hiệu cache cũ
        int(line_id or 0),
        gran or "",
        dt_from_key.isoformat(),
        dt_to_key.isoformat(),
        po or "", int(pkg or 0), int(shift_no or 0),
        compare or "",
        stats or "",
    )
    cached = CACHE.get(cache_key)
    # logger.info("cached=%s  cache_key=%s", cached, cache_key)
    if cached is not None:
        global CACHE_HIT
        CACHE_HIT += 1
        # logger.info("CACHE HIT=%s  key=%s from=%s to=%s include=%s",
                    # CACHE_HIT, cache_key[:3], to_local_str(dt_from), to_local_str(dt_to), _inc)
        return cached                      

    else:
        global CACHE_MISS
        CACHE_MISS += 1
        # logger.info("CACHE MISS=%s  key=%s", CACHE_MISS, cache_key[:3])

    meta = {
        "generated_at": to_local_str(now_local()),
        "params": {
            "line_id": line_id,
            "from": to_local_str(dt_from),
            "to": to_local_str(dt_to),
            "gran": gran,
            "po": po,
            "pkg": pkg,
            "shift_no": shift_no,
        }
    }

    out = {"meta": meta}

    # --------- decide SQL by gran ---------
    if gran in ("min", "hour"):
        pass
    elif gran in ("day", "week", "month", "quarter", "year"):
        pass
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported gran: {gran}")
    # --------- Gauges (aggregate over the window) ---------
    logger.debug("gauges window: %s .. %s", dt_from, dt_to)
    out["gauges"] = _gauges_query(line_id, dt_from, dt_to, po, pkg, shift_no, shift_join=False)

    # --- 4) compare=prev
    if compare == CompareMode.prev:
        p_from, p_to = previous_window(gran, dt_from, dt_to)

        gauge_prev = _gauges_query(line_id, p_from, p_to, po, pkg, shift_no, shift_join=False)

        # delta = cur - prev (số % cũng trừ thẳng)
        delta = {}
        for k in out["gauges"].keys():
            cv = out["gauges"].get(k, 0)
            pv = gauge_prev.get(k, 0)
            try:
                delta[k] = float(cv) - float(pv)
            except Exception:
                delta[k] = 0.0

        out["gauges_prev"]  = gauge_prev
        out["gauges_delta"] = delta
    
    # >>> NEW: thống kê min/max/avg theo bucket
    if stats == "basic":
        out["gauges_stats"] = build_gauges_stats(
            line_id=line_id,
            gran=gran,
            dt_from=dt_from,
            dt_to=dt_to,
            po = po, pkg = pkg, shift_no = shift_no
        )
    # --- Set cache ---------------------------------------------------------------
    CACHE.set(cache_key, out)
    logger.info("CACHE SET key=%s size=%s", cache_key[:3], len(out.get("linechart", [])) if isinstance(out.get("linechart"), list) else "-")
    return out

async def build_line_payload(
    line_id: int,
    dt_from: datetime,
    dt_to: datetime,
    gran: str,
    limit: int,
    bucket_sec: Optional[int],
    po: Optional[str],
    pkg: Optional[int],
    shift_no: Optional[int],
    include: List[str],
) -> dict:
    """Build the JSON payload for the endpoint.
    Returns dict with keys: meta, gauges?, linechart?, summaries?
    """
    # --- Compose cache key -------------------------------------------------------
    # include có thể là "gauges,linechart" -> chuẩn hóa thành tuple có thứ tự ổn định
    _inc = tuple(sorted(p.strip() for p in include if p.strip()))

    # --- floor thời điểm kết thúc để tăng tỷ lệ HIT ---
    dt_to_key = _floor_dt(dt_to, CACHE_WINDOW_SEC)
    # (tuỳ chọn) nếu muốn chắc ăn hơn, có thể floor cả dt_from khi scope=minute/between
    dt_from_key = _floor_dt(dt_from, CACHE_WINDOW_SEC)  
    # dt_from_key = dt_from
    cache_key = (
        "oee.v2",                # thay đổi khi bạn đổi format payload -> tự vô hiệu cache cũ
        int(line_id or 0),
        gran or "",
        dt_from_key.isoformat(),
        dt_to_key.isoformat(),
        int(limit or 0),
        po or "", int(pkg or 0), int(shift_no or 0),
        _inc,
    )
    cached = CACHE.get(cache_key)
    # logger.info("cached=%s  cache_key=%s", cached, cache_key)
    if cached is not None:
        global CACHE_HIT
        CACHE_HIT += 1
        # logger.info("CACHE HIT=%s  key=%s from=%s to=%s include=%s",
                    # CACHE_HIT, cache_key[:3], to_local_str(dt_from), to_local_str(dt_to), _inc)
        return cached                      

    else:
        global CACHE_MISS
        CACHE_MISS += 1
        # logger.info("CACHE MISS=%s  key=%s", CACHE_MISS, cache_key[:3])

    meta = {
        "generated_at": to_local_str(now_local()),
        "params": {
            "line_id": line_id,
            "from": to_local_str(dt_from),
            "to": to_local_str(dt_to),
            "gran": gran,
            "limit": limit,
            "bucket_sec": bucket_sec,
            "po": po,
            "pkg": pkg,
            "shift_no": shift_no,
        }
    }

    out = {"meta": meta}

    # --------- execute series SQL ---------
    # series_df = _series_request(line_id, gran, dt_from, dt_to, po, pkg, shift_no, limit, shift_join=False)

    # ---- LINECHART (auto-bucket by time window) ----
    linechart_df, bucket_sec = _linechart_query(line_id, dt_from, dt_to, po, pkg, shift_no, limit, bucket_mode=True)
    logger.info("linechart points=%s (before normalize)", len(linechart_df))
 
    out["linechart"]= df_to_records(linechart_df)
    # --------- summaries (basic counts by gran for convenience) ---------
    if "summaries" in include:
        # simple derived summaries from the already computed series
        if linechart_df.empty:
            out["summaries"] = {"points": 0, "total_good": 0, "total_reject": 0}
        else:
            out["summaries"] = {
                "bucket_sec": int(bucket_sec),
                "points": int(linechart_df.shape[0]),
                "total_good": int(linechart_df["good"].fillna(0).sum()),
                "total_reject": int(linechart_df["reject"].fillna(0).sum()),
            }
    # --- Set cache ---------------------------------------------------------------
    CACHE.set(cache_key, out)
    logger.info("CACHE SET key=%s size=%s", cache_key[:3], len(out.get("linechart", [])) if isinstance(out.get("linechart"), list) else "-")
    return out

async def build_race_payload(
    line_id: int,
    dt_from: datetime,
    dt_to: datetime,
    gran: str,
    limit: int,
    bucket_sec: Optional[int],
    po: Optional[str],
    pkg: Optional[int],
    shift_no: Optional[int],
    include: List[str],
) -> dict:
    """Build the JSON payload for the endpoint.
    Returns dict with keys: meta, race, summaries?
    """
    span_seconds = int((dt_to - dt_from).total_seconds())

    # --- pick bucket for linechart (auto density) ---
    auto_bucket_sec = choose_bucket_seconds_auto(span_seconds, limit)
    logger.info("series:auto gran, window=%ss, limit=%s -> bucket_sec=%s", span_seconds, limit, auto_bucket_sec)

    # --- Compose cache key -------------------------------------------------------
    # include có thể là "gauges,linechart" -> chuẩn hóa thành tuple có thứ tự ổn định
    _inc = tuple(sorted(p.strip() for p in include if p.strip()))

    # --- floor thời điểm kết thúc để tăng tỷ lệ HIT ---
    dt_to_key = _floor_dt(dt_to, CACHE_WINDOW_SEC)
    # (tuỳ chọn) nếu muốn chắc ăn hơn, có thể floor cả dt_from khi scope=minute/between
    dt_from_key = _floor_dt(dt_from, CACHE_WINDOW_SEC)  
    # dt_from_key = dt_from
    cache_key = (
        "oee.v2",                # thay đổi khi bạn đổi format payload -> tự vô hiệu cache cũ
        int(line_id or 0),
        gran or "",
        dt_from_key.isoformat(),
        dt_to_key.isoformat(),
        int(limit or 0),
        po or "", int(pkg or 0), int(shift_no or 0),
        _inc,
    )
    cached = CACHE.get(cache_key)
    # logger.info("cached=%s  cache_key=%s", cached, cache_key)
    if cached is not None:
        global CACHE_HIT
        CACHE_HIT += 1
        # logger.info("CACHE HIT=%s  key=%s from=%s to=%s include=%s",
                    # CACHE_HIT, cache_key[:3], to_local_str(dt_from), to_local_str(dt_to), _inc)
        return cached                      

    else:
        global CACHE_MISS
        CACHE_MISS += 1
        # logger.info("CACHE MISS=%s  key=%s", CACHE_MISS, cache_key[:3])

    meta = {
        "generated_at": to_local_str(now_local()),
        "params": {
            "line_id": line_id,
            "from": to_local_str(dt_from),
            "to": to_local_str(dt_to),
            "gran": gran,
            "limit": limit,
            "bucket_sec": bucket_sec,
            "po": po,
            "pkg": pkg,
            "shift_no": shift_no,
        }
    }

    out = {"meta": meta}

    # --------- decide SQL by gran ---------
    if gran in ("min", "hour"):
        bsec = choose_bucket_seconds(span_seconds, limit, gran, bucket_sec)
        where, params, shift_join = _assemble_where(dt_from, dt_to, line_id, po, pkg, shift_no, shift_join=False)
        line_expr = "?" if line_id == 0 else "f.line_id"
        series_sql = SQL.SERIES_MIN_HOUR.format(
            line_id_expr=line_expr,
            shift_join=shift_join,
            where=where,
        )
        # params: bucket_sec, bucket_sec, [from,to,(line),(po),(pkg),(shift)], limit
        sparams = [bsec, bsec] + params
        if line_id == 0:
            sparams.append(0)  # placeholder to project line_id=0
        sparams += [limit]

    elif gran in ("day", "week", "month", "quarter", "year"):
        where, params, shift_join = _assemble_where(dt_from, dt_to, line_id, po, pkg, shift_no, shift_join=False)
        line_expr = "?" if line_id == 0 else "f.line_id"
        series_sql = SQL.SERIES_BY_CAL[gran].format(
            line_id_expr=line_expr,
            shift_join=shift_join,
            where=where,
        )
        sparams = params.copy()
        if line_id == 0:
            sparams.append(0)
        sparams += [limit]
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported gran: {gran}")

    # --------- execute series SQL ---------
    logger.info("series SQL gran=%s", gran)
    series_df = query_df(series_sql, sparams)
    series_df = compute_oee(series_df)
    # -------- Race (downtime top reasons) --------
    params = {
        "line_id": line_id,          # 0 = plant, >0 = 1 line
        "from_ts": dt_from,          # datetime (UTC/local tuỳ chuẩn)
        "to_ts": dt_to,
        "limit": limit or 10,
        "offset": 0,
    }

    race_df = query_df(SQL.RACE, params)   # top 10
    race = []
    total_downtime = race_df["total_downtime_sec"].sum() if not race_df.empty else 0
    for _, r in race_df.iterrows():
        race.append({
            "reason_id": _int_or_none(r.get("reason_id")),
            "reason_name": r["reason_name"] or "Unknown",
            "reason_group": r["reason_group"] or "Unknown",
            "total_downtime_sec": int(r["total_downtime_sec"]),
            "percent": (r["total_downtime_sec"] / total_downtime * 100.0) if total_downtime > 0 else 0
        })
    out["race"] = race
    # --------- summaries (basic counts by gran for convenience) ---------
    if "summaries" in include:
        # simple derived summaries from the already computed series
        if series_df.empty:
            out["summaries"] = {"points": 0, "total_good": 0, "total_reject": 0}
        else:
            out["summaries"] = {
                "points": int(series_df.shape[0]),
                "total_good": int(series_df["good"].fillna(0).sum()),
                "total_reject": int(series_df["reject"].fillna(0).sum()),
            }
    # --- Set cache ---------------------------------------------------------------
    CACHE.set(cache_key, out)
    logger.info("CACHE SET key=%s size=%s", cache_key[:3], len(out.get("linechart", [])) if isinstance(out.get("linechart"), list) else "-")
    return out

async def build_gantt_payload(
    line_id: int,
    dt_from: datetime,
    dt_to: datetime,
    gran: str,
    limit: int,
    bucket_sec: Optional[int],
    po: Optional[str],
    pkg: Optional[int],
    shift_no: Optional[int],
    include: List[str],
    lines=None, lines_mode="top_downtime",
    lines_limit=10, lines_offset=0, **kwargs
) -> dict:
    """Build the JSON payload for the endpoint.
    Returns dict with keys: meta, gauges?, linechart?, summaries?
    """
    span_seconds = int((dt_to - dt_from).total_seconds())

    # --- pick bucket for linechart (auto density) ---
    auto_bucket_sec = choose_bucket_seconds_auto(span_seconds, limit)
    logger.info("series:auto gran, window=%ss, limit=%s -> bucket_sec=%s", span_seconds, limit, auto_bucket_sec)

    # --- Compose cache key -------------------------------------------------------
    # include có thể là "gauges,linechart" -> chuẩn hóa thành tuple có thứ tự ổn định
    _inc = tuple(sorted(p.strip() for p in include if p.strip()))

    # --- floor thời điểm kết thúc để tăng tỷ lệ HIT ---
    dt_to_key = _floor_dt(dt_to, CACHE_WINDOW_SEC)
    # (tuỳ chọn) nếu muốn chắc ăn hơn, có thể floor cả dt_from khi scope=minute/between
    dt_from_key = _floor_dt(dt_from, CACHE_WINDOW_SEC)  
    # dt_from_key = dt_from
    cache_key = (
        "oee.v2",                # thay đổi khi bạn đổi format payload -> tự vô hiệu cache cũ
        int(line_id or 0),
        gran or "",
        dt_from_key.isoformat(),
        dt_to_key.isoformat(),
        int(limit or 0),
        po or "", int(pkg or 0), int(shift_no or 0),
        _inc,
    )
    cached = CACHE.get(cache_key)
    # logger.info("cached=%s  cache_key=%s", cached, cache_key)
    if cached is not None:
        global CACHE_HIT
        CACHE_HIT += 1
        # logger.info("CACHE HIT=%s  key=%s from=%s to=%s include=%s",
                    # CACHE_HIT, cache_key[:3], to_local_str(dt_from), to_local_str(dt_to), _inc)
        return cached                      

    else:
        global CACHE_MISS
        CACHE_MISS += 1
        # logger.info("CACHE MISS=%s  key=%s", CACHE_MISS, cache_key[:3])

    meta = {
        "generated_at": to_local_str(now_local()),
        "params": {
            "line_id": line_id,
            "from": to_local_str(dt_from),
            "to": to_local_str(dt_to),
            "gran": gran,
            "limit": limit,
            "bucket_sec": bucket_sec,
            "po": po,
            "pkg": pkg,
            "shift_no": shift_no,
        }
    }

    out = {"meta": meta}

    # --------- decide SQL by gran ---------
    if gran in ("min", "hour"):
        bsec = choose_bucket_seconds(span_seconds, limit, gran, bucket_sec)
        where, params, shift_join = _assemble_where(dt_from, dt_to, line_id, po, pkg, shift_no, shift_join=False)
        line_expr = "?" if line_id == 0 else "f.line_id"
        series_sql = SQL.SERIES_MIN_HOUR.format(
            line_id_expr=line_expr,
            shift_join=shift_join,
            where=where,
        )
        # params: bucket_sec, bucket_sec, [from,to,(line),(po),(pkg),(shift)], limit
        sparams = [bsec, bsec] + params
        if line_id == 0:
            sparams.append(0)  # placeholder to project line_id=0
        sparams += [limit]

    elif gran in ("day", "week", "month", "quarter", "year"):
        where, params, shift_join = _assemble_where(dt_from, dt_to, line_id, po, pkg, shift_no, shift_join=False)
        line_expr = "?" if line_id == 0 else "f.line_id"
        series_sql = SQL.SERIES_BY_CAL[gran].format(
            line_id_expr=line_expr,
            shift_join=shift_join,
            where=where,
        )
        sparams = params.copy()
        if line_id == 0:
            sparams.append(0)
        sparams += [limit]
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported gran: {gran}")

    # --------- execute series SQL ---------
    logger.info("series SQL gran=%s", gran)
    series_df = query_df(series_sql, sparams)
    series_df = compute_oee(series_df)
    # -------- Gantt (timeline of states) --------
    if "gantt" in include:
        if line_id > 0:
            params = {
                "line_id": line_id,          # 0 = plant, >0 = 1 line
                "from_ts": dt_from,          # datetime (UTC/local tuỳ chuẩn)
                "to_ts": dt_to,
                "limit": limit or 50,
                "offset": 0,
            }
            gantt_df = query_df(SQL.GANTT, params)
            gantt = []
            for _, r in gantt_df.iterrows():
                gantt.append({
                    "event_id": _int_or_none(r["event_id"]), # Số thứ tự của event trong bảng fact_state_event
                    "line_id": _int_or_none(r["line_id"]),
                    "state_id": _int_or_none(r["state_id"]),# Trạng thái hoạt động "1=RUN/2=STOP/5=IDLE..."
                    "reason_id": _int_or_none(r["reason_id"]) if r["reason_id"] else None, # Mã lỗi "9999/4100..."
                    "start_ts": str(r["start_ts"]),
                    "end_ts": str(r["end_ts"]),
                    "duration_sec": _int_or_none(r["duration_sec"]),# Tổng thời gian xuất hiện của event
                    "note": r["note"] or "" # trạng thái tương ứng với state_id "RUN/STOP/IDLE..."
                })
            out["gantt"] = gantt
            pass
        else:
            # PLANT -> multi-gantt per line
            # 1) xác định danh sách line
            selected_lines: list[int]
            if lines:
                selected_lines = parse_csv_int(lines)
            else:
                pick_sql = {
                    "all": SQL.LINES_ALL_IN_WINDOW,
                    "top_run": SQL.LINES_TOP_RUN,
                    "top_downtime": SQL.LINES_TOP_DOWNTIME,
                }.get(lines_mode, SQL.LINES_TOP_DOWNTIME)

                df_pick = query_df(pick_sql, {
                    "from_ts": dt_from, "to_ts": dt_to,
                    "lines_limit": int(lines_limit),
                    "lines_offset": int(lines_offset),
                })
                selected_lines = sorted({int(x) for x in df_pick["line_id"].tolist()}) if not df_pick.empty else []

            # 2) lấy segments cho từng line (dùng SQL đơn giản)
            gantt_lines = []
            for lid in selected_lines:
                df_seg = query_df(SQL.GANTT_SEGMENTS_SIMPLE, {
                    "from_ts": dt_from, "to_ts": dt_to,
                    "line_id": int(lid),
                    "limit": int(limit),
                    "offset": 0,  # có thể mở param riêng nếu muốn phân trang theo line
                })
                segs = [] if df_seg.empty else df_seg.to_dict("records")
                # segs = _merge_segments(segs)
                # (tuỳ chọn) summary by state cho từng line
                summary = _build_gantt_summary(df_seg)
                gantt_lines.append({
                    "line_id": int(lid),
                    "segments": segs,
                    "summary_by_state": summary,  # cho panel nhỏ bên phải mỗi row (nếu UI cần)
                })

            out["gantt"] = {
                "lines": gantt_lines,
                "has_more_lines": len(selected_lines) == int(lines_limit)  # gần đúng
            }

    # --------- summaries (basic counts by gran for convenience) ---------
    if "summaries" in include:
        # simple derived summaries from the already computed series
        if series_df.empty:
            out["summaries"] = {"points": 0, "total_good": 0, "total_reject": 0}
        else:
            out["summaries"] = {
                "points": int(series_df.shape[0]),
                "total_good": int(series_df["good"].fillna(0).sum()),
                "total_reject": int(series_df["reject"].fillna(0).sum()),
            }
    # --- Set cache ---------------------------------------------------------------
    CACHE.set(cache_key, out)
    logger.info("CACHE SET key=%s size=%s", cache_key[:3], len(out.get("linechart", [])) if isinstance(out.get("linechart"), list) else "-")
    return out