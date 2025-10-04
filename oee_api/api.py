# =============================
# ========== api.py ==========
# =============================
"""
FastAPI application for OEE API (Refactor V1)
- One main endpoint: /oee/line/{line_id}
- Supports both range mode (from_ts/to_ts) and gran mode (since_min or gran=day/week/month/quarter/year/shift)
- Returns a flexible payload composed by `include` query: gauges, linechart, summaries
- Timezone: Asia/Ho_Chi_Minh, all timestamps in ISO 8601 with offset
- Plant-level: line_id = 0 (aggregate all lines)
- Shift filter: optional shift_no, applied by time-of-day window (handles overnight shifts)
- Strong logging at each step, docstrings & inline comments

NOTE:
• SQLs in `sql_texts.py` are templates and may require alignment with your actual schema.
• If your existing SQLs already work, paste them into sql_texts.py 
  and keep the function signatures in queries.py unchanged.
"""
from fastapi import FastAPI, APIRouter, Query, Path, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List
import logging

from oee_api.config import SETTINGS
from oee_api.utils import (
    now_local,
    to_iso8601,
    to_local_str,  
    sanitize_json_deep,
    parse_from_to,
    normalize_oee_payload,
    caculate_dt_from_to,
)
from oee_api.utils import Scope,CompareMode
from oee_api.queries import (
    build_payload,
    build_gauges_payload,
    build_line_payload,
    build_race_payload,
    build_gantt_payload,
)
from oee_api.schemas import Payload, Granularity
from enum import Enum
from datetime import timedelta
import time
from typing import Dict, Any, Optional, Tuple, List

# --------------------------------------------------------------
# Logging
# --------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("oee.api")

app = FastAPI(
    title="OEE Service",
    version="2.0",
    openapi_tags = [
        {
            "name": "Gauges",
            "description": "Calculate the overall OEE indicators (Availability / Performance / Quality / OEE) for a specific line or the entire plant within a **time window**. "
                        "Supports *previous period* comparison via the `compare=prev` query parameter.",
        },
        {
            "name": "Linechart",
            "description": "Time-series OEE data by user-defined or auto-selected time buckets.",
        },
        {
            "name": "Race",
            "description": "Top downtime reasons with their respective proportions.",
        },
        {
            "name": "Gantt",
            "description": "Machine state timeline; at plant level, returns multiple Gantt charts for each line.",
        },
    ],
)

# --------------------------------------------------------------
# CORS (open for now; adjust via env later if needed)
# --------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    # allow_origins=SETTINGS.CORS_ALLOW_ORIGINS,
    allow_origins=["*"],
    # allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------
# Optional API key (disabled if not set)
# --------------------------------------------------------------
async def _check_api_key(x_api_key: Optional[str]):
    if not SETTINGS.API_KEY:
        return  # feature disabled
    if x_api_key != SETTINGS.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
def health():
    """Simple health check."""
    return {"status": "ok", "time": to_local_str(now_local())}

@app.get(
    "/oee/line/{line_id}",
    response_model=Payload,
    response_model_exclude_none=True,
    summary="Get All data OEE timeseries/gauges for a line or plant (line_id=0).",
    tags=["oee"],
)

async def get_oee_all(
    line_id: int = Path(..., ge=0, description="Line ID. Use 0 for Plant-level aggregation."),
    # --- scope-first parameters ---
    scope: Scope = Query(Scope.day, description="Time scope for view: minute/day/week/month/quarter/year/between."),
    since_min: Optional[int] = Query(
        SETTINGS.DEFAULT_SINCE_MIN, 
        ge=1, 
        description="Only when scope=minute. Lookback minutes.",
        examples={
            "last15":  {"summary": "Last 15 minutes", "value": 15},
            "last240": {"summary": "Last 4 hours",    "value": 240},
            "last720": {"summary": "Last 12 hours",   "value": 720},
        },
    ),
    from_ts: Optional[str] = Query(
        None, 
        description="Only when scope=between. 'YYYY-MM-DD HH:MM:SS' (local)",
        examples={
            "startOfWindow": {"summary": "Start of range", "value": "2025-09-27 07:00:00"},
            "monthStart":    {"summary": "Month start",   "value": "2025-09-01 00:00:00"},
        },
    ),
    to_ts: Optional[str] = Query(
        None, 
        description="Only when scope=between. 'YYYY-MM-DD HH:MM:SS' (local)",
        examples={
            "endOfWindow": {"summary": "End of range", "value": "2025-09-27 11:00:00"},
            "monthEnd":    {"summary": "Month end",    "value": "2025-09-30 23:59:59"},
        },
    ),
    # filters
    po: Optional[str] = Query(
        None, alias="process_order",
        description="Filter by process order (PO).",
        examples={"samplePO": {"summary": "Example PO", "value": "PO123"}},
    ),
    pkg: Optional[int] = Query(
        None, alias="packaging_id",
        description="Filter by packaging ID.",
        examples={"samplePkg": {"summary": "Example packaging", "value": 7}},
    ),
    shift_no: Optional[int] = Query(
        None, description="Shift number filter (1,2,3…). Applies by time-of-day window.",
        examples={"shift1": {"summary": "Shift #1", "value": 1}},
    ),
    # shaping
    limit: int = Query(
        SETTINGS.DEFAULT_LIMIT, ge=10, le=SETTINGS.HARD_MAX_LIMIT,
        description="Max points AFTER auto-bucketing.",
        examples={"tight": {"summary": "Cap at 120 points", "value": 120}},
    ),
    include: str = Query(
        "gauges,linechart",
        description="Comma list: gauges,linechart,summaries",
        examples={
            "realtime": {"summary": "Realtime board", "value": "gauges,linechart"},
            "report":   {"summary": "Report view",    "value": "linechart,summaries"},
            "all":      {"summary": "All parts",      "value": "gauges,linechart,summaries"},
        },
    ),
    # optional auth header
    x_api_key: Optional[str] = Header(None, convert_underscores=False),
):
    """
    Main OEE endpoint. Examples:
    - /oee/line/105?scope=minute&since_min=240&include=gauges,linechart
    - /oee/line/0?gran=between&from_ts=2025-09-20T00:00:00+07:00&to_ts=2025-09-26T23:59:59+07:00&include=linechart,summaries
    """
    await _check_api_key(x_api_key)

    # --- resolve time window from scope ---
    now = now_local()
    if scope == Scope.minute:
        m = int(since_min or SETTINGS.DEFAULT_SINCE_MIN)  # ví dụ mặc định 240 nếu bạn đang dùng
        dt_from, dt_to = now - timedelta(minutes=m), now
    elif scope == Scope.between:
        if not from_ts or not to_ts:
            raise HTTPException(status_code=400, detail="from_ts & to_ts are required when scope=between")
        dt_from, dt_to = parse_from_to(from_ts, to_ts, since_min=240)  # bạn đã có helper này
    elif scope == Scope.day:
        dt_from = now.replace(hour=0, minute=0, second=0, microsecond=0)
        dt_to = now
    elif scope == Scope.week:
        # tuần bắt đầu Thứ 2 00:00
        weekday = (now.weekday() + 7) % 7  # 0=Mon
        monday = (now - timedelta(days=weekday)).replace(hour=0, minute=0, second=0, microsecond=0)
        dt_from, dt_to = monday, now
    elif scope == Scope.month:
        first = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        dt_from, dt_to = first, now
    elif scope == Scope.quarter:
        q = (now.month - 1) // 3  # 0..3
        first_month = 1 + q*3
        first = now.replace(month=first_month, day=1, hour=0, minute=0, second=0, microsecond=0)
        dt_from, dt_to = first, now
    elif scope == Scope.year:
        first = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        dt_from, dt_to = first, now
    else:
        # fallback an toàn
        m = int(since_min or SETTINGS.DEFAULT_SINCE_MIN)
        dt_from, dt_to = now - timedelta(minutes=m), now

    logger.info("/oee/line start | line_id=%s scope=%s -> window %s .. %s include=%s", line_id, scope, to_local_str(dt_from), to_local_str(dt_to), include)

    # map scope -> gran 
    if scope == Scope.minute:
        gran_for_queries = "min"
    elif scope == Scope.between:
        # linechart dùng auto-bucket; gran chỉ để log/summary => chọn "min" làm giá trị an toàn
        gran_for_queries = "min"
    else:
        gran_for_queries = scope.value  # day/week/month/quarter/year


    # 2) Decide which parts to include
    include_parts: List[str] = [p.strip() for p in include.split(",") if p.strip()]

    # 3) Build the payload (queries + post-processing are in queries.py)
    payload = await build_payload(
        line_id=line_id,
        dt_from=dt_from,
        dt_to=dt_to,
        gran=gran_for_queries,
        limit=limit,
        bucket_sec=None,
        po=po,
        pkg=pkg,
        shift_no=shift_no,
        include=include_parts,
    )

    # 4) chốt hạ output theo hợp đồng
    # payload = normalize_oee_payload(payload)
    # 5) Final sanitation + response
    logger.info("linechart payload=%s", payload)
    payload = sanitize_json_deep(payload)
    return JSONResponse(payload)


@app.get(
    "/oee/gauges/{line_id}",
    tags=["oee"],
    response_model=Payload,
    response_model_exclude_none=True,
    summary = "OEE gauges for a specific line or the entire plant",
    description = (
        "Return gauges for current window. \n"
        "Optional `compare=prev` to include previous window and delta. \n"
        "Optional `stats=basic` to add min/max/avg of A/P/Q/OEE by time bucket.\n"
    ),
)
async def get_gauges(
    line_id: int = Path(..., ge=0, description="plant = 0 / line = 103 / 104 / 105..."),
    scope: Scope = Query(..., description="minute / day / week / month / quarter / year / between"),
    since_min: int | None = Query(None, description="Used only when scope=minute"),
    from_ts: str | None = Query(None, description="YYYY-MM-DD HH:MM:SS (local) or ISO format, used when scope=between"),
    to_ts: str | None = Query(None, description="YYYY-MM-DD HH:MM:SS (local) or ISO format, used when scope=between"),
    po: str | None = Query(None, description="Filter by PO. Example: ACB123"),
    pkg: int | None = Query(None, description="Filter by packaging_id"),
    shift_no: int | None = Query(None, description="Filter by work shift number"),
    compare: CompareMode = Query(CompareMode.none, description="`none` or `prev` to compare with the previous period"),
    stats: str | None = Query(None, description="Set 'basic' to include gauges_stats (min/max/avg by bucket)"),
    # optional auth header
    x_api_key: Optional[str] = Header(None, convert_underscores=False),
):
    """
    Endpoint. Examples:
    - GET /oee/gauges/0?scope=minute&since_min=240&compare=prev
    - GET /oee/gauges/105?scope=between&from_ts=2025-10-03 00:00:00&to_ts=2025-10-03 12:00:00
    - GET /oee/gauges/0?scope=day&po=AB123&shift_no=1
    Payload:
    {
      "meta": {...},
      "gauges": {...},
      "gauges_prev": {...},     # nếu compare=prev
      "gauges_delta": {...}     # nếu compare=prev
    }
    """
    await _check_api_key(x_api_key)
    # 1) resolve time window from scope ---
    dt_from, dt_to, gran_for_queries = caculate_dt_from_to(scope,since_min,from_ts,to_ts,HTTPException)
    logger.info("/oee/line start | line_id=%s scope=%s -> window %s .. %s", line_id, scope, to_local_str(dt_from), to_local_str(dt_to))
    # 2) Build the payload (queries + post-processing are in queries.py)
    payload = await build_gauges_payload(
        line_id=line_id,
        dt_from=dt_from,
        dt_to=dt_to,
        gran=gran_for_queries,
        po=po,
        pkg=pkg,
        shift_no=shift_no,
        compare=compare,
        stats=stats,
    )
    # 3) Final sanitation + response
    # Chuẩn hoá % và counter an toàn
    payload = normalize_oee_payload(payload)
    payload = sanitize_json_deep(payload)
    return JSONResponse(payload)

@app.get(
    "/oee/linechart/{line_id}",
    response_model=Payload,
    response_model_exclude_none=True,
    summary="Get data for line chart.",
    tags=["oee"],
)

async def get_linechart(
    line_id: int = Path(..., ge=0, description="Line ID. Use 0 for Plant-level aggregation."),
    # --- scope-first parameters ---
    scope: Scope = Query(Scope.day, description="Time scope for view: minute/day/week/month/quarter/year/between."),
    since_min: Optional[int] = Query(
        SETTINGS.DEFAULT_SINCE_MIN, 
        ge=1, 
        description="Only when scope=minute. Lookback minutes.",
        examples={
            "last15":  {"summary": "Last 15 minutes", "value": 15},
            "last240": {"summary": "Last 4 hours",    "value": 240},
            "last720": {"summary": "Last 12 hours",   "value": 720},
        },
    ),
    from_ts: Optional[str] = Query(
        None, 
        description="Only when scope=between. 'YYYY-MM-DD HH:MM:SS' (local)",
        examples={
            "startOfWindow": {"summary": "Start of range", "value": "2025-09-27 07:00:00"},
            "monthStart":    {"summary": "Month start",   "value": "2025-09-01 00:00:00"},
        },
    ),
    to_ts: Optional[str] = Query(
        None, 
        description="Only when scope=between. 'YYYY-MM-DD HH:MM:SS' (local)",
        examples={
            "endOfWindow": {"summary": "End of range", "value": "2025-09-27 11:00:00"},
            "monthEnd":    {"summary": "Month end",    "value": "2025-09-30 23:59:59"},
        },
    ),
    # filters
    po: Optional[str] = Query(
        None, alias="process_order",
        description="Filter by process order (PO).",
        examples={"samplePO": {"summary": "Example PO", "value": "PO123"}},
    ),
    pkg: Optional[int] = Query(
        None, alias="packaging_id",
        description="Filter by packaging ID.",
        examples={"samplePkg": {"summary": "Example packaging", "value": 7}},
    ),
    shift_no: Optional[int] = Query(
        None, description="Shift number filter (1,2,3…). Applies by time-of-day window.",
        examples={"shift1": {"summary": "Shift #1", "value": 1}},
    ),
    # shaping
    limit: int = Query(
        SETTINGS.DEFAULT_LIMIT, ge=10, le=SETTINGS.HARD_MAX_LIMIT,
        description="Max points AFTER auto-bucketing.",
        examples={"tight": {"summary": "Cap at 120 points", "value": 120}},
    ),
    include: str = Query(
        "gauges,linechart",
        description="Comma list: gauges,linechart,summaries",
        examples={
            "realtime": {"summary": "Realtime board", "value": "gauges,linechart"},
            "report":   {"summary": "Report view",    "value": "linechart,summaries"},
            "all":      {"summary": "All parts",      "value": "gauges,linechart,summaries"},
        },
    ),
    # optional auth header
    x_api_key: Optional[str] = Header(None, convert_underscores=False),
):
    """
    Main OEE endpoint. Examples:
    - /oee/linechart/105?scope=minute&since_min=240&include=gauges,linechart
    - /oee/linechart/0?gran=between&from_ts=2025-09-20T00:00:00+07:00&to_ts=2025-09-26T23:59:59+07:00&include=linechart,summaries
    """
    await _check_api_key(x_api_key)
    # 1) resolve time window from scope ---
    dt_from, dt_to, gran_for_queries = caculate_dt_from_to(scope,since_min,from_ts,to_ts,HTTPException)
    logger.info("/oee/line start | line_id=%s scope=%s -> window %s .. %s", line_id, scope, to_local_str(dt_from), to_local_str(dt_to))

    # 3) Decide which parts to include
    include_parts: List[str] = [p.strip() for p in include.split(",") if p.strip()]

    # 4) Build the payload (queries + post-processing are in queries.py)
    payload = await build_line_payload(
        line_id=line_id,
        dt_from=dt_from,
        dt_to=dt_to,
        gran=gran_for_queries,
        limit=limit,
        bucket_sec=None,
        po=po,
        pkg=pkg,
        shift_no=shift_no,
        include=include_parts,
    )
    # 5) Final sanitation + response
    # payload = normalize_oee_payload(payload)
    payload = sanitize_json_deep(payload)
    return JSONResponse(payload)

@app.get(
    "/oee/race/{line_id}",
    response_model=Payload,
    response_model_exclude_none=True,
    summary="Get data for Race chart.",
    tags=["oee"],
)

async def get_race(
    line_id: int = Path(..., ge=0, description="Line ID. Use 0 for Plant-level aggregation."),
    # --- scope-first parameters ---
    scope: Scope = Query(Scope.day, description="Time scope for view: minute/day/week/month/quarter/year/between."),
    since_min: Optional[int] = Query(
        SETTINGS.DEFAULT_SINCE_MIN, 
        ge=1, 
        description="Only when scope=minute. Lookback minutes.",
        examples={
            "last15":  {"summary": "Last 15 minutes", "value": 15},
            "last240": {"summary": "Last 4 hours",    "value": 240},
            "last720": {"summary": "Last 12 hours",   "value": 720},
        },
    ),
    from_ts: Optional[str] = Query(
        None, 
        description="Only when scope=between. 'YYYY-MM-DD HH:MM:SS' (local)",
        examples={
            "startOfWindow": {"summary": "Start of range", "value": "2025-09-27 07:00:00"},
            "monthStart":    {"summary": "Month start",   "value": "2025-09-01 00:00:00"},
        },
    ),
    to_ts: Optional[str] = Query(
        None, 
        description="Only when scope=between. 'YYYY-MM-DD HH:MM:SS' (local)",
        examples={
            "endOfWindow": {"summary": "End of range", "value": "2025-09-27 11:00:00"},
            "monthEnd":    {"summary": "Month end",    "value": "2025-09-30 23:59:59"},
        },
    ),
    # filters
    po: Optional[str] = Query(
        None, alias="process_order",
        description="Filter by process order (PO).",
        examples={"samplePO": {"summary": "Example PO", "value": "PO123"}},
    ),
    pkg: Optional[int] = Query(
        None, alias="packaging_id",
        description="Filter by packaging ID.",
        examples={"samplePkg": {"summary": "Example packaging", "value": 7}},
    ),
    shift_no: Optional[int] = Query(
        None, description="Shift number filter (1,2,3…). Applies by time-of-day window.",
        examples={"shift1": {"summary": "Shift #1", "value": 1}},
    ),
    # shaping
    limit: int = Query(
        SETTINGS.DEFAULT_LIMIT, ge=10, le=SETTINGS.HARD_MAX_LIMIT,
        description="Max points AFTER auto-bucketing.",
        examples={"tight": {"summary": "Cap at 120 points", "value": 120}},
    ),
    include: str = Query(
        "gauges,linechart",
        description="Comma list: gauges,linechart,summaries",
        examples={
            "realtime": {"summary": "Realtime board", "value": "gauges,linechart"},
            "report":   {"summary": "Report view",    "value": "linechart,summaries"},
            "all":      {"summary": "All parts",      "value": "gauges,linechart,summaries"},
        },
    ),
    # optional auth header
    x_api_key: Optional[str] = Header(None, convert_underscores=False),
):
    """
    Race chart endpoint. Examples:
    - /oee/race/105?scope=minute&since_min=240&include=summaries
    - /oee/race/0?gran=between&from_ts=2025-09-20T00:00:00+07:00&to_ts=2025-09-26T23:59:59+07:00&include=summaries
    """
    await _check_api_key(x_api_key)
    # 1) resolve time window from scope ---
    dt_from, dt_to, gran_for_queries = caculate_dt_from_to(scope,since_min,from_ts,to_ts,HTTPException)
    logger.info("/oee/line start | line_id=%s scope=%s -> window %s .. %s", line_id, scope, to_local_str(dt_from), to_local_str(dt_to))

    # 3) Decide which parts to include
    include_parts: List[str] = [p.strip() for p in include.split(",") if p.strip()]

    # 3) Build the payload (queries + post-processing are in queries.py)
    payload = await build_race_payload(
        line_id=line_id,
        dt_from=dt_from,
        dt_to=dt_to,
        gran=gran_for_queries,
        limit=limit,
        bucket_sec=None,
        po=po,
        pkg=pkg,
        shift_no=shift_no,
        include=include_parts,
    )

    # 4) chốt hạ output theo hợp đồng
    # payload = normalize_oee_payload(payload)
    # 5) Final sanitation + response
    # logger.info("linechart payload=%s", payload)
    payload = sanitize_json_deep(payload)
    return JSONResponse(payload)

@app.get(
    "/oee/gantt/{line_id}",
    response_model=Payload,
    response_model_exclude_none=True,
    summary="Get data for gantt chart.",
    tags=["oee"],
)

async def get_gantt(
    line_id: int = Path(..., ge=0, description="Line ID. Use 0 for Plant-level aggregation."),
    # --- scope-first parameters ---
    scope: Scope = Query(Scope.day, description="Time scope for view: minute/day/week/month/quarter/year/between."),
    since_min: Optional[int] = Query(
        SETTINGS.DEFAULT_SINCE_MIN, 
        ge=1, 
        description="Only when scope=minute. Lookback minutes.",
        examples={
            "last15":  {"summary": "Last 15 minutes", "value": 15},
            "last240": {"summary": "Last 4 hours",    "value": 240},
            "last720": {"summary": "Last 12 hours",   "value": 720},
        },
    ),
    from_ts: Optional[str] = Query(
        None, 
        description="Only when scope=between. 'YYYY-MM-DD HH:MM:SS' (local)",
        examples={
            "startOfWindow": {"summary": "Start of range", "value": "2025-09-27 07:00:00"},
            "monthStart":    {"summary": "Month start",   "value": "2025-09-01 00:00:00"},
        },
    ),
    to_ts: Optional[str] = Query(
        None, 
        description="Only when scope=between. 'YYYY-MM-DD HH:MM:SS' (local)",
        examples={
            "endOfWindow": {"summary": "End of range", "value": "2025-09-27 11:00:00"},
            "monthEnd":    {"summary": "Month end",    "value": "2025-09-30 23:59:59"},
        },
    ),
    # filters
    po: Optional[str] = Query(
        None, alias="process_order",
        description="Filter by process order (PO).",
        examples={"samplePO": {"summary": "Example PO", "value": "PO123"}},
    ),
    pkg: Optional[int] = Query(
        None, alias="packaging_id",
        description="Filter by packaging ID.",
        examples={"samplePkg": {"summary": "Example packaging", "value": 7}},
    ),
    shift_no: Optional[int] = Query(
        None, description="Shift number filter (1,2,3…). Applies by time-of-day window.",
        examples={"shift1": {"summary": "Shift #1", "value": 1}},
    ),
    # shaping
    limit: int = Query(
        SETTINGS.DEFAULT_LIMIT, ge=10, le=SETTINGS.HARD_MAX_LIMIT,
        description="Max points AFTER auto-bucketing.",
        examples={"tight": {"summary": "Cap at 120 points", "value": 120}},
    ),
    include: str = Query(
        "gauges,linechart",
        description="Comma list: gauges,linechart,summaries",
        examples={
            "realtime": {"summary": "Realtime board", "value": "gauges,linechart"},
            "report":   {"summary": "Report view",    "value": "linechart,summaries"},
            "all":      {"summary": "All parts",      "value": "gauges,linechart,summaries"},
        },
    ),
    # optional auth header
    x_api_key: Optional[str] = Header(None, convert_underscores=False),
    # --- NEW: multi-gantt khi line_id = 0 (plant) ---
    lines: Optional[str] = Query(
        None, description="CSV line IDs (khi line_id=0). Ví dụ: 103,104,105"),
    lines_mode: str = Query(
        "top_downtime", pattern="^(all|top_downtime|top_run)$",
        description="Cách chọn line tự động khi không truyền 'lines'"),
    lines_limit: int = Query(8, ge=1, le=50),
    lines_offset: int = Query(0, ge=0),
):
    """
    Main OEE endpoint. Examples:
    - /oee/line/105?scope=minute&since_min=240&include=gauges,linechart
    - /oee/line/0?gran=between&from_ts=2025-09-20T00:00:00+07:00&to_ts=2025-09-26T23:59:59+07:00&include=linechart,summaries
    """
    await _check_api_key(x_api_key)
    # 1) resolve time window from scope ---
    dt_from, dt_to, gran_for_queries = caculate_dt_from_to(scope,since_min,from_ts,to_ts,HTTPException)
    logger.info("/oee/line start | line_id=%s scope=%s -> window %s .. %s", line_id, scope, to_local_str(dt_from), to_local_str(dt_to))

    # 3) Decide which parts to include
    include_parts: List[str] = [p.strip() for p in include.split(",") if p.strip()]
    payload = await build_gantt_payload(
        line_id=line_id,# scope=scope, since_min=since_min,
        dt_from=dt_from, dt_to=dt_to,gran=gran_for_queries, 
        limit=limit, bucket_sec=None,
        include=include_parts, po=po, pkg=pkg, shift_no=shift_no,
        # NEW
        lines=lines, lines_mode=lines_mode,
        lines_limit=lines_limit, lines_offset=lines_offset,
    )

    # 4) chốt hạ output theo hợp đồng
    # payload = normalize_oee_payload(payload)
    # 5) Final sanitation + response
    logger.info("linechart payload=%s", payload)
    payload = sanitize_json_deep(payload)
    return JSONResponse(payload)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("oee_api.api:app", host="0.0.0.0", port=8088, reload=False, workers=1)