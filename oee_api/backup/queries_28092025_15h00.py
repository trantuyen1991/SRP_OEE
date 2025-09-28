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
from datetime import datetime
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
)
from oee_api import sql_texts as SQL
import math
logger = logging.getLogger("oee.queries")

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

def _postprocess_series(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-row OEE metrics từ raw sums.
    Trả về DataFrame đã 'zeroize' các % metrics để vẽ chart không lỗi.
    """
    rows = []
    for _, r in df.iterrows():
        metrics = compute_oee(OeeInputs(
            good=r.get("good", 0) or 0,
            reject=r.get("reject", 0) or 0,
            runtime_sec=r.get("runtime_sec", 0) or 0,
            downtime_sec=r.get("downtime_sec", 0) or 0,
            ideal_rate_per_min=r.get("ideal_rate_per_min"),
        ))
        no_data = ((r.get("good", 0) or 0) + (r.get("reject", 0) or 0) == 0) and \
                ((r.get("runtime_sec", 0) or 0) + (r.get("downtime_sec", 0) or 0) == 0)

        for k in ("availability", "performance", "quality", "oee"):
            v = metrics.get(k)
            if v is None or no_data:
                metrics[k] = 0.0

        # 👉 ép None -> 0.0 cho các % metrics (linechart friendly)
        for k in ("availability", "performance", "quality", "oee"):
            if metrics.get(k) is None:
                metrics[k] = 0.0
       
        row = dict(r)
        # (tuỳ chọn) ép các số đếm về int để JSON gọn
        for k in ("good", "reject", "runtime_sec", "downtime_sec"):
            row[k] = int(row.get(k, 0) or 0)

        row.update(metrics)
        rows.append(row)

    return pd.DataFrame(rows)


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
    series_df = _postprocess_series(series_df)

    # --------- Gauges (aggregate over the window) ---------
    if "gauges" in include:
        gw, gparams, gsj = _assemble_where(dt_from, dt_to, line_id, po, pkg, shift_no, shift_join=False)
        line_expr = "?" if line_id == 0 else "f.line_id"
        gauges_sql = SQL.GAUGES.format(line_id_expr=line_expr, shift_join=gsj, where=gw)
        gparams2 = gparams.copy()
        if line_id == 0:
            gparams2.append(0)
        gauges_df = query_df(gauges_sql, gparams2)
        gauges_row = gauges_df.iloc[0].to_dict() if not gauges_df.empty else {
            "line_id": line_id,
            "good": 0, "reject": 0, "runtime_sec": 0, "downtime_sec": 0, "ideal_rate_per_min": None,
        }
        gauges = compute_oee(OeeInputs(
            good=gauges_row.get("good", 0) or 0,
            reject=gauges_row.get("reject", 0) or 0,
            runtime_sec=gauges_row.get("runtime_sec", 0) or 0,
            downtime_sec=gauges_row.get("downtime_sec", 0) or 0,
            ideal_rate_per_min=gauges_row.get("ideal_rate_per_min"),
        ))
        gauges.update({
            "line_id": line_id,
            "good": int(gauges_row.get("good", 0) or 0),
            "reject": int(gauges_row.get("reject", 0) or 0),
            "runtime_sec": int(gauges_row.get("runtime_sec", 0) or 0),
            "downtime_sec": int(gauges_row.get("downtime_sec", 0) or 0),
        })
        out["gauges"] = gauges

    # --------- linechart ---------
    if "linechart" in include:
        out["linechart"] = df_to_records(series_df)

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

    return out