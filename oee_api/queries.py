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

def _postprocess_series(df: pd.DataFrame) -> pd.DataFrame:
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

        metrics = compute_oee(OeeInputs(
            good=good, reject=reject,
            runtime_sec=runtime_sec, downtime_sec=downtime_sec,
            ideal_rate_per_min=ideal,
        ))

        # === Cưỡng chế theo hợp đồng ===
        no_data = (good + reject == 0) and (runtime_sec + downtime_sec == 0)

        for k in ("availability", "performance", "quality", "oee"):
            v = metrics.get(k)
            metrics[k] = 0.0 if (v is None or _is_nan(v)) else float(v)

        if ideal is None:
            metrics["performance"] = 0.0

        if no_data:
            for k in ("availability", "performance", "quality", "oee"):
                metrics[k] = 0.0

        # tính lại OEE cho nhất quán
        metrics["oee"] = (
            metrics["availability"] * metrics["performance"] * metrics["quality"] / 10000.0
        )

        row = dict(r)
        # ép số đếm về int để JSON gọn và ổn định
        row["good"] = good
        row["reject"] = reject
        row["runtime_sec"] = runtime_sec
        row["downtime_sec"] = downtime_sec
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

        gauges = compute_oee(OeeInputs(
            good=good, reject=reject,
            runtime_sec=runtime_sec, downtime_sec=downtime_sec,
            ideal_rate_per_min=ideal,
        ))

        no_data = (good + reject == 0) and (runtime_sec + downtime_sec == 0)

        for k in ("availability", "performance", "quality", "oee"):
            v = gauges.get(k)
            gauges[k] = 0.0 if (v is None or _is_nan(v)) else float(v)

        if ideal is None:
            gauges["performance"] = 0.0

        if no_data:
            for k in ("availability", "performance", "quality", "oee"):
                gauges[k] = 0.0

        gauges["oee"] = (
            gauges["availability"] * gauges["performance"] * gauges["quality"] / 10000.0
        )

        gauges.update({
            "line_id": line_id,
            "good": good,
            "reject": reject,
            "runtime_sec": runtime_sec,
            "downtime_sec": downtime_sec,
        })
        out["gauges"] = gauges

    # --------- linechart ---------
    if "linechart" in include:
        recs = df_to_records(series_df)

        # ---- Last guard: đảm bảo hợp đồng ở cấp JSON records ----
        for pt in recs:
            # chuẩn hoá số đếm
            g  = _nz_int(pt.get("good"))
            rj = _nz_int(pt.get("reject"))
            rt = _nz_int(pt.get("runtime_sec"))
            dt = _nz_int(pt.get("downtime_sec"))
            ideal = _ideal_or_none(pt.get("ideal_rate_per_min"))

            # fill None/NaN -> 0.0 cho 4 chỉ số %
            for k in ("availability", "performance", "quality", "oee"):
                v = pt.get(k)
                if v is None or _is_nan(v):
                    pt[k] = 0.0
                else:
                    pt[k] = float(v)

            # thiếu ideal -> performance = 0.0
            if ideal is None:
                pt["performance"] = 0.0

            # bucket không có dữ liệu -> 4% = 0.0
            if (g + rj == 0) and (rt + dt == 0):
                for k in ("availability", "performance", "quality", "oee"):
                    pt[k] = 0.0

            # tính lại OEE cho nhất quán
            pt["oee"] = pt["availability"] * pt["performance"] * pt["quality"] / 10000.0

        out["linechart"] = recs

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