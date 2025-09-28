# oee_api/schemas.py
from __future__ import annotations
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, ConfigDict

# Granularity “khóa hợp đồng” với FE (enum)
Granularity = Literal["min", "hour", "day", "week", "month", "quarter", "year", "shift"]

class MetaParams(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    line_id: int
    # JSON key vẫn là "from"/"to" (alias), nhưng tên field hợp lệ trong Python
    from_ts: str = Field(
        serialization_alias="from",
        validation_alias="from",
        description="Local time 'YYYY-MM-DD HH:MM:SS' (no offset).",
    )
    to_ts: str = Field(
        serialization_alias="to",
        validation_alias="to",
        description="Local time 'YYYY-MM-DD HH:MM:SS' (no offset).",
    )
    gran: Granularity
    limit: int
    bucket_sec: Optional[int] = Field(default=None, description="Bucket seconds for min/hour gran.")
    po: Optional[str] = Field(default=None, description="Process order filter.")
    pkg: Optional[int] = Field(default=None, description="Packaging ID filter.")
    shift_no: Optional[int] = Field(default=None, description="Shift number filter.")

class Meta(BaseModel):
    generated_at: str = Field(description="Local time 'YYYY-MM-DD HH:MM:SS' (no offset).")
    params: MetaParams

class Gauge(BaseModel):
    line_id: int
    good: int
    reject: int
    runtime_sec: int
    downtime_sec: int
    # Gauges có thể 'không xác định' => cho phép null
    availability: Optional[float] = None
    performance: Optional[float] = None
    quality: Optional[float] = None
    oee: Optional[float] = None

class LinePoint(BaseModel):
    # linechart luôn là số (đã zeroize 0.0 nếu không tính được)
    ts_bucket: str = Field(description="Time bucket as 'YYYY-MM-DD HH:MM:SS' local.")
    line_id: int
    good: int
    reject: int
    runtime_sec: int
    downtime_sec: int
    ideal_rate_per_min: Optional[float] = None
    availability: float
    performance: float
    quality: float
    oee: float

class Summaries(BaseModel):
    points: int
    total_good: int
    total_reject: int

class Payload(BaseModel):
    """
    Hợp đồng trả về: ít nhất có 'meta'. Tùy 'include' sẽ có thêm 'gauges', 'linechart', 'summaries'.
    Tất cả timestamps trả về ở định dạng local 'YYYY-MM-DD HH:MM:SS' (không offset).
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "meta": {
                    "generated_at": "2025-09-28 09:12:00",
                    "params": {
                        "line_id": 105,
                        "from": "2025-09-28 05:12:00",
                        "to": "2025-09-28 09:12:00",
                        "gran": "min",
                        "limit": 2000,
                        "bucket_sec": 300,
                        "po": None,
                        "pkg": None,
                        "shift_no": None
                    }
                },
                "gauges": {
                    "line_id": 105,
                    "good": 15420,
                    "reject": 120,
                    "runtime_sec": 13200,
                    "downtime_sec": 600,
                    "availability": 95.45,
                    "performance": 88.1,
                    "quality": 99.23,
                    "oee": 83.52
                },
                "linechart": [
                    {
                        "ts_bucket": "2025-09-28 06:00:00",
                        "line_id": 105,
                        "good": 2600,
                        "reject": 20,
                        "runtime_sec": 3300,
                        "downtime_sec": 0,
                        "ideal_rate_per_min": 800.0,
                        "availability": 100.0,
                        "performance": 78.8,
                        "quality": 99.23,
                        "oee": 78.11
                    }
                ],
                "summaries": {
                    "points": 49,
                    "total_good": 15420,
                    "total_reject": 120
                }
            }
        }
    )

    meta: Meta
    gauges: Optional[Gauge] = None
    linechart: Optional[List[LinePoint]] = None
    summaries: Optional[Summaries] = None
