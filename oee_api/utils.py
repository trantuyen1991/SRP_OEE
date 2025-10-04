# =============================
# ========= utils.py ==========
# =============================
"""Utilities: time helpers, JSON sanitation, DF conversions, bucketing.
"""
from __future__ import annotations
from typing import Any, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from decimal import Decimal
import math
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo
from oee_api.config import SETTINGS
import logging
import os, time, threading
from enum import Enum

# ---------------- Time helpers ----------------

def tz() -> ZoneInfo:
    return ZoneInfo(SETTINGS.TIMEZONE)

def now_local() -> datetime:
    return datetime.now(tz())

def to_iso8601(dt: datetime) -> str:
    return dt.astimezone(tz()).isoformat(timespec="seconds")

def to_local_str(dt: datetime) -> str:
    """
    Format 'YYYY-MM-DD HH:MM:SS' theo múi giờ local, KHÔNG kèm offset.
    - pandas.Timestamp -> datetime
    - naive -> coi là local
    - aware -> convert về local
    """
    if dt is None:
        return None
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    if dt.tzinfo is None:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return dt.astimezone(tz()).strftime("%Y-%m-%d %H:%M:%S")


def parse_from_to(from_ts: Optional[str], to_ts: Optional[str], since_min: Optional[int]) -> Tuple[datetime, datetime]:
    """Decide a time window from inputs. Accepts ISO 8601 or 'YYYY-MM-DD HH:MM:SS'.
    If from/to omitted: fallback to lookback minutes (default 240).
    """
    def _parse(s: str) -> datetime:
        if "T" in s:
            return datetime.fromisoformat(s)
        # assume local datetime string
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz())

    if from_ts and to_ts:
        start = _parse(from_ts)
        end = _parse(to_ts)
    else:
        minutes = since_min if since_min else SETTINGS.DEFAULT_SINCE_MIN
        end = now_local()
        start = end - timedelta(minutes=minutes)

    # normalize to seconds resolution
    return start.replace(microsecond=0), end.replace(microsecond=0)


# ---------------- DataFrame helpers ----------------

def df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to list[dict] and coerce NaN/Inf to None."""
    if df is None or df.empty:
        return []
    # Ensure plain Python types (avoid numpy types leaking to JSON)
    df = df.replace({np.nan: None, np.inf: None, -np.inf: None})
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")
    records = df.to_dict(orient="records")
    return records


# ---------------- JSON sanitizer ----------------

def sanitize_json_deep(obj: Any) -> Any:
    """Recursively sanitize an object for JSON encoding.
    - Convert NaN/Inf to None
    - Convert numpy scalars to Python scalars
    - Convert Decimal to float
    - Format datetimes to ISO 8601 with offset
    """
    if obj is None:
        return None
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (datetime, pd.Timestamp)):
        return to_local_str(obj)
    if isinstance(obj, date):
        # chuẩn hóa về 'YYYY-MM-DD 00:00:00' cho đồng nhất với datetime
        return f"{obj.strftime('%Y-%m-%d')} 00:00:00"
    if isinstance(obj, list):
        return [sanitize_json_deep(x) for x in obj]
    if isinstance(obj, dict):
        return {k: sanitize_json_deep(v) for k, v in obj.items()}
    return obj


# ---------------- Bucketing ----------------

def choose_bucket_seconds(span_seconds: int, limit: int, gran: str, override: Optional[int] = None) -> int:
    """Pick a bucket size (seconds) so that number of points <= limit.
    Respect explicit override if provided.
    """
    if override and override > 0:
        return override

    if gran == "min":
        # base 60s upwards
        candidates = [60, 120, 300, 600, 900, 1800, 3600]
    elif gran == "hour":
        candidates = [3600, 7200, 14400]
    else:
        # for day/week/month we don't apply second-based bucket
        return 60

    for b in candidates:
        if span_seconds / b <= max(10, min(limit, 20000)):
            return b
    return candidates[-1]


# ---------------- OEE calculations ----------------
@dataclass
class OeeInputs:
    good: float
    reject: float
    runtime_sec: float
    downtime_sec: float
    ideal_rate_per_min: Optional[float]  # may be None
    # NEW: tổng công suất lý thuyết (đã nhân theo runtime từng line)
    ideal_capacity_cnt: Optional[float] = None

class Scope(str, Enum):
    minute = "minute"
    day = "day"
    week = "week"
    month = "month"
    quarter = "quarter"
    year = "year"
    between = "between"

class CompareMode(str, Enum):
    none = "none"
    prev = "prev"

def compute_oee(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized OEE metrics for any scope (plant/line).
    Requires columns: good, ng, runtime_sec, planned_sec, ideal_capacity_cnt.
    Returns df with columns: availability, performance, quality, oee.
    NaN/None -> 0.0
    """
    s = df.copy()

    # Chuẩn hóa cột
    for c in ("good","runtime_sec","planned_sec","ideal_capacity_cnt"):
        if c not in s.columns: s[c] = 0
        s[c] = pd.to_numeric(s[c], errors="coerce").fillna(0)

    # 'reject' có thể tên 'reject' hoặc 'ng' tùy truy vấn
    if "reject" in s.columns:
        rej = pd.to_numeric(s["reject"], errors="coerce").fillna(0)
    else:
        rej = pd.to_numeric(s.get("ng", 0), errors="coerce").fillna(0)

    good = pd.to_numeric(s["good"], errors="coerce").fillna(0)
    produced = good + rej

    runtime_min = s["runtime_sec"] / 60.0
    planned_min = s["planned_sec"] / 60.0

    # Availability
    avail = np.where(planned_min > 0, runtime_min / planned_min, 0.0)
    # Performance (CHUẨN): dùng tổng sản lượng
    perf  = np.where(s["ideal_capacity_cnt"] > 0, produced / s["ideal_capacity_cnt"], 0.0)
    # Quality (CHUẨN): good / (good + reject)
    qual  = np.where(produced > 0, good / produced, 0.0)

    oee = avail * perf * qual

    s["availability"] = np.clip(np.nan_to_num(100.0 * avail), 0.0, None)
    s["performance"]  = np.clip(np.nan_to_num(100.0 * perf),  0.0, None)
    s["quality"]      = np.clip(np.nan_to_num(100.0 * qual),  0.0, None)
    s["oee"]          = np.clip(np.nan_to_num(100.0 * oee),   0.0, None)
    return s

# ---- OEE payload normalizer -------------------------------------------------
def normalize_oee_payload(payload: dict) -> dict:
    """Enforce OEE contract on output payload (gauges + linechart).
        Rules:
          - Any % None/NaN -> 0.0
          - Missing ideal_rate_per_min (None/NaN/<=0) -> performance = 0.0
          - No-data bucket (good+reject==0 and runtime+downtime==0) -> all % = 0.0
          - Recompute OEE = A*P*Q/10000
          - Cast counters to int
    """
    def _is_nan(x) -> bool:
        return isinstance(x, float) and math.isnan(x)

    def _nz_int(x) -> int:
        if x is None or _is_nan(x):
            return 0
        try:
            return int(float(x))
        except Exception:
            return 0

    def _ideal_ok(x) -> bool:
        """True nếu ideal_rate_per_min là số > 0, bất kể input là str/obj."""
        if x is None:
            return False
        try:
            v = float(x)
        except Exception:
            return False
        if math.isnan(v) or v <= 0:
            return False
        return True

    def _fix_point(pt: dict):
        g  = _nz_int(pt.get("good"))
        rj = _nz_int(pt.get("reject"))
        rt = _nz_int(pt.get("runtime_sec"))
        dt = _nz_int(pt.get("downtime_sec"))

        pt["good"] = g; pt["reject"] = rj
        pt["runtime_sec"] = rt; pt["downtime_sec"] = dt

        # fill None/NaN -> 0.0 cho 4 %
        for k in ("availability", "performance", "quality", "oee"):
            v = pt.get(k)
            try:
                v = float(v)
            except Exception:
                v = 0.0
            if _is_nan(v):
                v = 0.0
            pt[k] = v

        # Bucket không dữ liệu -> 4% = 0.0
        if (g + rj == 0) and (rt + dt == 0):
            for k in ("availability", "performance", "quality", "oee"):
                pt[k] = 0.0

        # Recompute OEE
        pt["oee"] = pt["availability"] * pt["performance"] * pt["quality"] / 10000.0

    # gauges
    if isinstance(payload.get("gauges"), dict):
        _fix_point(payload["gauges"])

    # linechart
    if isinstance(payload.get("linechart"), list):
        for pt in payload["linechart"]:
            if isinstance(pt, dict):
                _fix_point(pt)

    return payload
# ==== Auto bucketing helper ==================================================
logger = logging.getLogger("oee.utils")

def choose_bucket_seconds_auto(range_seconds: int, limit: int) -> int:
    """
    Chọn bucket_sec tự động cho linechart dựa trên độ dài khoảng thời gian và limit.
    - Ít nhất 60 giây (1 phút).
    - Làm tròn lên bội số của 60 cho đẹp (5 phút, 10 phút, ...).
    """
    try:
        limit = int(limit or 2000)
        range_seconds = int(max(0, range_seconds))
    except Exception:
        limit = 2000

    if limit <= 0 or range_seconds <= 0:
        logger.debug("auto-bucket: trivial, range=%s, limit=%s -> 60", range_seconds, limit)
        return 60

    raw = math.ceil(range_seconds / limit)  # số giây/bucket thô
    # làm tròn lên bội số 60
    bucket_sec = max(60, int(math.ceil(raw / 60.0) * 60))
    logger.info("auto-bucket: range_sec=%s, limit=%s -> bucket_sec=%s", range_seconds, limit, bucket_sec)
    return bucket_sec

# ==== Simple in-process TTL Cache ============================================
class TTLCache:
    """Rất gọn: lưu (expire_ts, value) theo key (hashable). Thread-safe."""
    def __init__(self, ttl_sec: int = 30, maxsize: int = 512):
        self.ttl_sec = int(ttl_sec)
        self.maxsize = int(maxsize)
        self._store: Dict[Any, Tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: Any) -> Optional[Any]:
        now = time.time()
        with self._lock:
            item = self._store.get(key)
            if not item:
                return None
            exp, val = item
            if exp < now:
                # hết hạn -> xóa
                self._store.pop(key, None)
                return None
            return val

    def set(self, key: Any, value: Any) -> None:
        exp = time.time() + self.ttl_sec
        with self._lock:
            if len(self._store) >= self.maxsize:
                # dọn thô: xóa các mục đã hết hạn; nếu vẫn đầy -> pop bất kỳ
                now = time.time()
                for k, (e, _) in list(self._store.items()):
                    if e < now:
                        self._store.pop(k, None)
                if len(self._store) >= self.maxsize:
                    self._store.pop(next(iter(self._store)), None)
            self._store[key] = (exp, value)
# --- NEW: parse CSV -> list[int] ---
def parse_csv_int(s: str | None) -> list[int]:
    if not s:
        return []
    out = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except Exception:
            continue
    # bỏ trùng và giữ thứ tự
    seen, uniq = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq

def caculate_dt_from_to(scope,since_min,from_ts,to_ts,HTTPException):
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

    # map scope -> gran 
    if scope == Scope.minute:
        gran = "min"
    elif scope == Scope.between:
        gran = "min"
    else:
        gran = scope.value  # day/week/month/quarter/year

    return dt_from, dt_to, gran

# === Gran & previous-window helpers =========================================
def infer_gran_from_scope(scope: str) -> str:
    """
    scope -> gran:
      - 'minute' và 'between'  => gran = 'min'
      - còn lại: gran = scope (day/week/month/quarter/year)
    """
    s = (scope or "").lower()
    if s in ("minute", "between"):
        return "min"
    return s

def previous_window(gran: str, start: datetime, end: datetime) -> tuple[datetime, datetime]:
    """
    Trả về cửa sổ liền kề trước [start, end) theo scope.
    minute/between: lùi đúng một khoảng thời gian (delta = end-start).
    day/week/month/quarter/year: lùi 1 đơn vị lịch tương ứng.
    """
    s = (gran or "").lower()
    if s == "min":
        delta = end - start
        return (start - delta, end - delta)

    # đơn vị lịch
    if s == "day":
        delta = timedelta(days=1)
        return (start - delta, end - delta)
    if s == "week":
        delta = timedelta(weeks=1)
        return (start - delta, end - delta)
    if s == "month":
        # lùi 1 tháng đơn giản (an toàn: rơi về ngày 1)
        a = start.replace(day=1)
        b = end.replace(day=1)
        prev_a_month = (a.month - 2) % 12 + 1
        prev_b_month = (b.month - 2) % 12 + 1
        prev_a_year  = a.year - (1 if a.month == 1 else 0)
        prev_b_year  = b.year - (1 if b.month == 1 else 0)
        prev_start = a.replace(year=prev_a_year, month=prev_a_month)
        prev_end   = b.replace(year=prev_b_year, month=prev_b_month)
        return (prev_start, prev_end)
    if s == "quarter":
        # quy ước q1 bắt đầu 01-01; lùi 1 quý
        def minus_quarter(d: datetime) -> datetime:
            q = ( (d.month-1)//3 )  # 0..3
            y = d.year
            q -= 1
            if q < 0:
                q += 4; y -= 1
            m = 1 + 3*q
            return d.replace(year=y, month=m, day=1, hour=0, minute=0, second=0, microsecond=0)
        return (minus_quarter(start), minus_quarter(end))
    if s == "year":
        prev_start = start.replace(year=start.year-1)
        prev_end   = end.replace(year=end.year-1)
        return (prev_start, prev_end)

    # fallback
    delta = end - start
    return (start - delta, end - delta)
