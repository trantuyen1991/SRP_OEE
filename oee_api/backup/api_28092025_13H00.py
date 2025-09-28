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
from fastapi import FastAPI, Query, Path, Header, HTTPException
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
)
from oee_api.queries import (
    build_payload,
)
from oee_api.schemas import Payload, Granularity

# --------------------------------------------------------------
# Logging
# --------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("oee.api")

app = FastAPI(title="OEE API V1", version="1.0.0")

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


# @app.get("/oee/line/{line_id}")
@app.get(
    "/oee/line/{line_id}",
    response_model=Payload,
    response_model_exclude_none=True,
    summary="Get OEE timeseries/gauges for a line or plant (line_id=0).",
    tags=["oee"],
)
# async def get_oee_line(
#     line_id: int = Path(..., ge=0, description="Line ID. Use 0 for Plant-level aggregation."),
#     # time selectors
#     from_ts: Optional[str] = Query(None, description="Start time in ISO 8601 or 'YYYY-MM-DD HH:MM:SS' local."),
#     to_ts: Optional[str] = Query(None, description="End time in ISO 8601 or 'YYYY-MM-DD HH:MM:SS' local."),
#     since_min: Optional[int] = Query(SETTINGS.DEFAULT_SINCE_MIN, ge=1, description="If from/to omitted: lookback minutes."),
#     # granularity
#     gran: str = Query("min", pattern="^(min|hour|shift|day|week|month|quarter|year)$",description="Series granularity"),
#     # filters
#     po: Optional[str] = Query(None, alias="process_order"),
#     pkg: Optional[int] = Query(None, alias="packaging_id"),
#     shift_no: Optional[int] = Query(None, description="Filter by shift number, applies by time-of-day window."),
#     # shaping
#     limit: int = Query(SETTINGS.DEFAULT_LIMIT, ge=10, le=SETTINGS.HARD_MAX_LIMIT, description="Target points after bucketing."),
#     bucket_sec: Optional[int] = Query(None, ge=60, description="Override bucket seconds for 'min'/'hour' gran."),
#     include: str = Query("gauges,linechart", description="Comma list: gauges,linechart,summaries"),
#     # optional auth header
#     x_api_key: Optional[str] = Header(None, convert_underscores=False),
# ):
async def get_oee_line(
    line_id: int = Path(..., ge=0, description="Line ID. Use 0 for Plant-level aggregation."),
    # time selectors
    from_ts: Optional[str] = Query(None, description="Local time 'YYYY-MM-DD HH:MM:SS'."),
    to_ts: Optional[str] = Query(None, description="Local time 'YYYY-MM-DD HH:MM:SS'."),
    since_min: Optional[int] = Query(SETTINGS.DEFAULT_SINCE_MIN, ge=1, description="If from/to omitted: lookback minutes."),
    # granularity (enum thay vì pattern)
    gran: Granularity = Query("min", description="Series granularity."),
    # filters
    po: Optional[str] = Query(None, alias="process_order"),
    pkg: Optional[int] = Query(None, alias="packaging_id"),
    shift_no: Optional[int] = Query(None, description="Filter by shift number, applies by time-of-day window."),
    # shaping
    limit: int = Query(SETTINGS.DEFAULT_LIMIT, ge=10, le=SETTINGS.HARD_MAX_LIMIT, description="Target points after bucketing."),
    bucket_sec: Optional[int] = Query(None, ge=60, description="Override bucket seconds for 'min'/'hour' gran."),
    include: str = Query("gauges,linechart", description="Comma list: gauges,linechart,summaries"),
    # optional auth header
    x_api_key: Optional[str] = Header(None, convert_underscores=False),
):
    """
    Main OEE endpoint. Examples:
    - /oee/line/105?since_min=240&gran=min&include=gauges,linechart
    - /oee/line/0?from_ts=2025-09-20T00:00:00+07:00&to_ts=2025-09-26T23:59:59+07:00&gran=day&include=linechart,summaries
    """
    await _check_api_key(x_api_key)

    logger.info("/oee/line start | line_id=%s gran=%s include=%s", line_id, gran, include)

    # 1) Decide time window (from_ts/to_ts)
    dt_from, dt_to = parse_from_to(from_ts, to_ts, since_min)
    logger.info("window: %s → %s", to_iso8601(dt_from), to_iso8601(dt_to))

    # 2) Decide which parts to include
    include_parts: List[str] = [p.strip() for p in include.split(",") if p.strip()]

    # 3) Build the payload (queries + post-processing are in queries.py)
    payload = await build_payload(
        line_id=line_id,
        dt_from=dt_from,
        dt_to=dt_to,
        gran=gran,
        limit=limit,
        bucket_sec=bucket_sec,
        po=po,
        pkg=pkg,
        shift_no=shift_no,
        include=include_parts,
    )

    # 4) Final sanitation + response
    payload = sanitize_json_deep(payload)
    return JSONResponse(payload)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("oee_api.api:app", host="0.0.0.0", port=8080, reload=False, workers=1)