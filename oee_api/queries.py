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
    OeeInputs,
    compute_oee,
    choose_bucket_seconds_auto,
    TTLCache,
)
from oee_api import sql_texts as SQL
import math
import os

CACHE_TTL_SEC = int(os.getenv("CACHE_TTL_SEC", "30"))   # TTL mặc định 30s
CACHE_MAX_KEYS = int(os.getenv("CACHE_MAX_KEYS", "1024"))
CACHE_WINDOW_SEC = int(os.getenv("CACHE_WINDOW_SEC", "10"))   # 10 giây là hợp lý cho dashboard

CACHE_HIT   = 0
CACHE_MISS  = 0

CACHE = TTLCache(ttl_sec=CACHE_TTL_SEC, maxsize=CACHE_MAX_KEYS)

logger = logging.getLogger("oee.queries")

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
) -> Tuple[str, list]:
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

def _postprocess_series(lineID,df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-row OEE metrics từ raw sums.
    Trả về DataFrame đã 'zeroize' các % metrics để vẽ chart không lỗi.
    """
    rows = []
    for _, r in df.iterrows():
        good = _nz_int(r.get("good"))
        reject = _nz_int(r.get("reject"))
        runtime_sec = _nz_int(r.get("runtime_sec"))
        downtime_sec = _nz_int(r.get("downtime_sec"))
        ideal = _ideal_or_none(r.get("ideal_rate_per_min"))
        ideal_cnt = _ideal_or_none(r.get("ideal_capacity_cnt"))

        metrics = compute_oee(OeeInputs(
            line_id=lineID,
            good=good, reject=reject,
            runtime_sec=runtime_sec, downtime_sec=downtime_sec,
            ideal_rate_per_min=ideal,ideal_capacity_cnt=ideal_cnt,
        ))

        row = dict(r)
        # ép số đếm về int để JSON gọn và ổn định
        row["good"] = good
        row["reject"] = reject
        row["runtime_sec"] = runtime_sec
        row["downtime_sec"] = downtime_sec
        row.update(metrics)
        rows.append(row)

    return pd.DataFrame(rows)

def _floor_dt(dt, sec: int):
    if sec <= 0:
        return dt.replace(microsecond=0)
    ts = int(dt.timestamp())
    floored = ts - (ts % sec)
    return dt.__class__.fromtimestamp(floored, tz=dt.tzinfo)

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

    elif gran == "shift":
        # uses CROSS JOIN dim_shift_calendar internally, no extra shift_join needed here
        where, params, _ = _assemble_where(dt_from, dt_to, line_id, po, pkg, None, shift_join=True)
        line_expr = "?" if line_id == 0 else "f.line_id"
        series_sql = SQL.SERIES_SHIFT.format(
            line_id_expr=line_expr,
            shift_no_filter=("AND s.shift_no = ?" if shift_no is not None else ""),
            where=where,
        )
        sparams = params.copy()
        if line_id == 0:
            sparams.append(0)
        if shift_no is not None:
            sparams.append(shift_no)
        sparams += [limit]

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported gran: {gran}")

    # --------- execute series SQL ---------
    logger.info("series SQL gran=%s", gran)
    series_df = query_df(series_sql, sparams)
    series_df = _postprocess_series(lineID=line_id, df= series_df)
    # ---- Belt & suspenders: enforce hợp đồng ở cấp DataFrame ----
    if not series_df.empty:
        # 1) Chuẩn hóa kiểu & fillna cho 4 % metrics
        for c in ("availability", "performance", "quality", "oee"):
            series_df[c] = pd.to_numeric(series_df.get(c), errors="coerce").fillna(0.0)

        # 2) Chuẩn hóa bộ đếm để tạo mask
        good      = pd.to_numeric(series_df.get("good"), errors="coerce").fillna(0).astype(int)
        reject    = pd.to_numeric(series_df.get("reject"), errors="coerce").fillna(0).astype(int)
        runtime   = pd.to_numeric(series_df.get("runtime_sec"), errors="coerce").fillna(0).astype(int)
        downtime  = pd.to_numeric(series_df.get("downtime_sec"), errors="coerce").fillna(0).astype(int)

        # 3) Thiếu ideal -> performance = 0.0
        ideal = pd.to_numeric(series_df.get("ideal_rate_per_min"), errors="coerce")
        mask_ideal   = ideal.isna() | (ideal <= 0)

        # 4) Bucket không có dữ liệu -> 4% = 0.0
        mask_nodata  = (good + reject == 0) & (runtime + downtime == 0)

        series_df.loc[mask_ideal, "performance"] = 0.0
        series_df.loc[mask_nodata, ["availability", "performance", "quality", "oee"]] = 0.0

        # 5) Tính lại OEE cho nhất quán
        series_df["oee"] = (
            series_df["availability"] * series_df["performance"] * series_df["quality"] / 10000.0
        )

    # --------- Gauges (aggregate over the window) ---------
    if "gauges" in include:
        logger.debug("gauges window: %s .. %s", dt_from, dt_to)
        gw, gparams, gsj = _assemble_where(dt_from, dt_to, line_id, po, pkg, shift_no, shift_join=False)
        line_expr = "?" if line_id == 0 else "f.line_id"
        gauges_sql = SQL.GAUGES.format(line_id_expr=line_expr, shift_join=gsj, where=gw)
        gparams2 = gparams.copy()

        if line_id == 0:
            gparams2.append(0)

        gauges_df = query_df(gauges_sql, gparams2)
        gauges_row = (gauges_df.iloc[0].to_dict() if not gauges_df.empty else
                    {"line_id": line_id, "good": 0, "reject": 0, "runtime_sec": 0, "downtime_sec": 0, "ideal_rate_per_min": None})

        good = _nz_int(gauges_row.get("good"))
        reject = _nz_int(gauges_row.get("reject"))
        runtime_sec = _nz_int(gauges_row.get("runtime_sec"))
        downtime_sec = _nz_int(gauges_row.get("downtime_sec"))
        ideal = _ideal_or_none(gauges_row.get("ideal_rate_per_min"))
        ideal_cnt = _ideal_or_none(gauges_row.get("ideal_capacity_cnt"))

        gauges = compute_oee(OeeInputs(
            line_id=line_id,
            good=good, reject=reject,
            runtime_sec=runtime_sec, downtime_sec=downtime_sec,
            ideal_rate_per_min=ideal,ideal_capacity_cnt=ideal_cnt,
        ))
        
        gauges.update({
            "line_id": line_id,
            "good": good,
            "reject": reject,
            "runtime_sec": runtime_sec,
            "downtime_sec": downtime_sec,
        })
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
        linechart_df = _postprocess_series(lineID=line_id, df=linechart_df)
        
        out["linechart"]= df_to_records(linechart_df)

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