# =============================
# ======== sql_texts.py =======
# =============================
"""
Centralized SQL templates. Replace contents with your proven SQL if you already have them.
Conventions:
- Use positional placeholders '?' for pyodbc.
- All time filtering use BETWEEN ? AND ? where the end is inclusive to ':59'.
- For Plant-level (line_id=0), omit the line filter.
- Shift filter handled by time-of-day windows from dim_shift_calendar (overnight supported).

Tables expected (adapt names to your schema):
- fact_production_min f(line_id, time_stamp, good_count, reject_count, runtime_sec, downtime_sec, packaging_id, process_order)
- dim_packaging p(packaging_id, ideal_rate_per_min)
- dim_shift_calendar s(shift_no, start_time, end_time, active)
"""

# ---------- WHERE fragments ----------
WHERE_BASE = """
WHERE f.ts_min BETWEEN ? AND ?
{line_filter}
{po_filter}
{pkg_filter}
{shift_filter}
""".strip()

LINE_FILTER = "AND f.line_id = ?"
PO_FILTER   = "AND f.process_order = ?"
PKG_FILTER  = "AND f.packaging_id = ?"

# Shift: join theo ngày, lọc theo time-of-day; KHÔNG dùng 'active'
SHIFT_CROSS_JOIN = "JOIN dim_shift_calendar s ON DATE(f.ts_min) = s.shift_date"
SHIFT_TOD_WHERE = """
AND (
  (s.end_time >= s.start_time AND TIME(f.ts_min) BETWEEN s.start_time AND s.end_time)
  OR
  (s.end_time < s.start_time  AND (TIME(f.ts_min) >= s.start_time OR TIME(f.ts_min) < s.end_time))
)
AND s.shift_no = ?
""".strip()


# ---------- Series (min/hour) ----------
SERIES_MIN_HOUR = """
SELECT
  FROM_UNIXTIME(FLOOR(UNIX_TIMESTAMP(f.ts_min)/?)*?) AS ts_bucket,
  {line_id_expr} AS line_id,
  SUM(f.good)                                     AS good,
  SUM(f.ng)                                       AS reject,
  SUM(f.runtime_sec)                              AS runtime_sec,
  SUM(GREATEST(f.planned_sec - f.runtime_sec,0))  AS downtime_sec,
  MAX(p.ideal_rate_per_min)                       AS ideal_rate_per_min,
  SUM(p.ideal_rate_per_min * f.runtime_sec/60.0)  AS ideal_capacity_cnt
FROM fact_production_min f
LEFT JOIN dim_packaging p ON p.packaging_id = f.packaging_id
{shift_join}
{where}
GROUP BY ts_bucket
ORDER BY ts_bucket
LIMIT ?
""".strip()


# ---------- Series (day/week/month/quarter/year) ----------
# We compute canonical bucket timestamps using MySQL date functions
SERIES_BY_CAL = {
    "day": """
        SELECT
          DATE(f.ts_min) AS ts_bucket,
          {line_id_expr} AS line_id,
          SUM(f.good)                                     AS good,
          SUM(f.ng)                                       AS reject,
          SUM(f.runtime_sec)                              AS runtime_sec,
          SUM(GREATEST(f.planned_sec - f.runtime_sec,0))  AS downtime_sec,
          MAX(p.ideal_rate_per_min)                       AS ideal_rate_per_min,
          SUM(p.ideal_rate_per_min * f.runtime_sec/60.0)  AS ideal_capacity_cnt
        FROM fact_production_min f
        LEFT JOIN dim_packaging p ON p.packaging_id = f.packaging_id
        {shift_join}
        {where}
        GROUP BY DATE(f.ts_min)
        ORDER BY ts_bucket
        LIMIT ?
    """.strip(),
    "week": """
        SELECT
          STR_TO_DATE(CONCAT(YEARWEEK(f.ts_min, 3), ' Monday'), '%X%V %W') AS ts_bucket,
          {line_id_expr} AS line_id,
          SUM(f.good)                                  AS good,
          SUM(f.ng)                                     AS reject,
          SUM(f.runtime_sec)                            AS runtime_sec,
          SUM(GREATEST(f.planned_sec - f.runtime_sec,0)) AS downtime_sec,
          MAX(p.ideal_rate_per_min)                     AS ideal_rate_per_min
        FROM fact_production_min f
        LEFT JOIN dim_packaging p ON p.packaging_id = f.packaging_id
        {shift_join}
        {where}
        GROUP BY ts_bucket 
        ORDER BY ts_bucket
        LIMIT ?
    """.strip(),
    "month": """
        SELECT
          DATE_FORMAT(f.ts_min, '%Y-%m-01') AS ts_bucket,
          {line_id_expr} AS line_id,
          SUM(f.good)                                  AS good,
          SUM(f.ng)                                     AS reject,
          SUM(f.runtime_sec)                            AS runtime_sec,
          SUM(GREATEST(f.planned_sec - f.runtime_sec,0)) AS downtime_sec,
          MAX(p.ideal_rate_per_min)                     AS ideal_rate_per_min
        FROM fact_production_min f
        LEFT JOIN dim_packaging p ON p.packaging_id = f.packaging_id
        {shift_join}
        {where}
        GROUP BY DATE_FORMAT(f.ts_min, '%Y-%m-01')
        ORDER BY ts_bucket
        LIMIT ?
    """.strip(),
    "quarter": """
        SELECT
          STR_TO_DATE(CONCAT(YEAR(f.ts_min),'-',LPAD(1+3*(QUARTER(f.ts_min)-1),2,'0'),'-01'), '%Y-%m-%d') AS ts_bucket,
          {line_id_expr} AS line_id,
          SUM(f.good)                                  AS good,
          SUM(f.ng)                                     AS reject,
          SUM(f.runtime_sec)                            AS runtime_sec,
          SUM(GREATEST(f.planned_sec - f.runtime_sec,0)) AS downtime_sec,
          MAX(p.ideal_rate_per_min)                     AS ideal_rate_per_min
        FROM fact_production_min f
        LEFT JOIN dim_packaging p ON p.packaging_id = f.packaging_id
        {shift_join}
        {where}
        GROUP BY ts_bucket 
        ORDER BY ts_bucket
        LIMIT ?
    """.strip(),
    "year": """
        SELECT
          STR_TO_DATE(CONCAT(YEAR(f.ts_min),'-01-01'), '%Y-%m-%d') AS ts_bucket,
          {line_id_expr} AS line_id,
          SUM(f.good)                                  AS good,
          SUM(f.ng)                                     AS reject,
          SUM(f.runtime_sec)                            AS runtime_sec,
          SUM(GREATEST(f.planned_sec - f.runtime_sec,0)) AS downtime_sec,
          MAX(p.ideal_rate_per_min)                     AS ideal_rate_per_min
        FROM fact_production_min f
        LEFT JOIN dim_packaging p ON p.packaging_id = f.packaging_id
        {shift_join}
        {where}
        GROUP BY ts_bucket 
        ORDER BY ts_bucket
        LIMIT ?
    """.strip(),
}

# ---------- Series (shift) ----------
# Build canonical shift buckets by crossing each event with active shift windows.
SERIES_SHIFT = """
SELECT
  -- mốc bắt đầu ca theo ngày trong dim_shift_calendar
  STR_TO_DATE(CONCAT(s.shift_date, ' ', TIME_FORMAT(s.start_time, '%H:%i:%s')), '%Y-%m-%d %H:%i:%s') AS ts_bucket,
  {line_id_expr} AS line_id,
  s.shift_no,
  SUM(f.good)                                  AS good,
  SUM(f.ng)                                     AS reject,
  SUM(f.runtime_sec)                            AS runtime_sec,
  SUM(GREATEST(f.planned_sec - f.runtime_sec,0)) AS downtime_sec,
  MAX(p.ideal_rate_per_min)                     AS ideal_rate_per_min
FROM fact_production_min f
LEFT JOIN dim_packaging p ON p.packaging_id = f.packaging_id
JOIN dim_shift_calendar s ON DATE(f.ts_min) = s.shift_date
{where}
AND (
  (s.end_time >= s.start_time AND TIME(f.ts_min) BETWEEN s.start_time AND s.end_time)
  OR (s.end_time <  s.start_time AND (TIME(f.ts_min) >= s.start_time OR TIME(f.ts_min) < s.end_time))
)
{shift_no_filter}
GROUP BY ts_bucket, s.shift_no
ORDER BY ts_bucket
LIMIT ?
""".strip()

# ---------- Gauges (aggregate over window) ----------
GAUGES = """
SELECT
  {line_id_expr} AS line_id,
  SUM(f.good)                                     AS good,
  SUM(f.ng)                                       AS reject,
  SUM(f.planned_sec)                              AS planned_sec,
  SUM(f.runtime_sec)                              AS runtime_sec,
  SUM(GREATEST(f.planned_sec - f.runtime_sec,0))  AS downtime_sec,
  MAX(p.ideal_rate_per_min)                       AS ideal_rate_per_min,
  SUM(p.ideal_rate_per_min * f.runtime_sec/60.0)  AS ideal_capacity_cnt
FROM fact_production_min f
LEFT JOIN dim_packaging p ON p.packaging_id = f.packaging_id
{shift_join}
{where}
""".strip()

# ==== Auto-bucket series (dùng cho mọi scope from_ts/to_ts) ==================
# Thứ tự placeholder (?):
#   1: bucket_sec
#   2: bucket_sec (lặp lại trong FROM_UNIXTIME)
#   3..N: where params (tùy filter)
#   Cuối: LIMIT
SERIES_AUTO = r"""
SELECT
  FROM_UNIXTIME(FLOOR(UNIX_TIMESTAMP(f.ts_min)/?)*?)      AS ts_bucket,
  {line_id_expr}                                          AS line_id,
  SUM(f.good)                                             AS good,
  SUM(f.ng)                                               AS reject,
  SUM(f.planned_sec)                                      AS planned_sec,
  SUM(f.runtime_sec)                                      AS runtime_sec,
  SUM(GREATEST(f.planned_sec - f.runtime_sec, 0))         AS downtime_sec,
  MAX(p.ideal_rate_per_min)                               AS ideal_rate_per_min,
  SUM(p.ideal_rate_per_min * f.runtime_sec/60.0)          AS ideal_capacity_cnt
FROM fact_production_min f
LEFT JOIN dim_packaging p ON p.packaging_id = f.packaging_id
{shift_join}
{where}
GROUP BY ts_bucket
ORDER BY ts_bucket
LIMIT ?
""".strip()

RACE = """
SELECT 
    e.reason_id,
    COALESCE(r.reason_code, '0')        AS reason_code,
    COALESCE(r.reason_name, 'Unknown')  AS reason_name,
    COALESCE(r.reason_group, 'Unknown') AS reason_group,
    SUM(e.duration_sec)                 AS total_downtime_sec
FROM fact_state_event e
LEFT JOIN dim_reason r ON r.reason_id = e.reason_id
WHERE
    -- overlap window (đừng dùng start>=from AND end<=to)
    e.start_ts < :to_ts
    AND e.end_ts  > :from_ts
    AND (:line_id = 0 OR e.line_id = :line_id)   -- 0 = toàn plant
    AND e.state_id NOT IN (1, 5)            -- loại RUN (1) và IDLE (5)
GROUP BY
    e.reason_id, r.reason_code, r.reason_name, r.reason_group
ORDER BY total_downtime_sec DESC
LIMIT :limit;
""".strip()

GANTT = """
SELECT
    e.event_id,
    e.line_id,
    e.state_id,
    s.state_code,
    e.reason_id,
    COALESCE(r.reason_code, '0')        AS reason_code,
    COALESCE(r.reason_name, 'Unknown')  AS reason_name,
    COALESCE(r.reason_group, 'Unknown') AS reason_group,
    e.start_ts,
    e.end_ts,
    e.duration_sec,
    e.note
FROM fact_state_event e
JOIN dim_state  s ON s.state_id = e.state_id
LEFT JOIN dim_reason r ON r.reason_id = e.reason_id
WHERE
    e.start_ts < :to_ts
    AND e.end_ts  > :from_ts
    AND (:line_id = 0 OR e.line_id = :line_id)
ORDER BY e.start_ts
LIMIT :limit OFFSET :offset;
""".strip()
# --- NEW: chọn line theo tổng downtime / run trong cửa sổ ---
LINES_TOP_DOWNTIME = """
SELECT
  e.line_id,
  SUM(CASE WHEN e.state_id NOT IN (1,5) THEN e.duration_sec ELSE 0 END) AS downtime_sec
FROM fact_state_event e
WHERE e.start_ts < :to_ts AND e.end_ts > :from_ts
GROUP BY e.line_id
ORDER BY downtime_sec DESC
LIMIT :lines_limit OFFSET :lines_offset
""".strip()

LINES_TOP_RUN = """
SELECT
  e.line_id,
  SUM(CASE WHEN e.state_id = 1 THEN e.duration_sec ELSE 0 END) AS run_sec
FROM fact_state_event e
WHERE e.start_ts < :to_ts AND e.end_ts > :from_ts
GROUP BY e.line_id
ORDER BY run_sec DESC
LIMIT :lines_limit OFFSET :lines_offset
""".strip()

LINES_ALL_IN_WINDOW = """
SELECT DISTINCT e.line_id
FROM fact_state_event e
WHERE e.start_ts < :to_ts AND e.end_ts > :from_ts
ORDER BY e.line_id
LIMIT :lines_limit OFFSET :lines_offset
""".strip()

# --- NEW: Gantt cho 1 line (độc lập template cũ) ---
GANTT_SEGMENTS_SIMPLE = """
SELECT
  e.event_id,
  e.line_id,
  e.state_id,
  s.state_code,
  e.reason_id,
  COALESCE(r.reason_code, '0')         AS reason_code,
  COALESCE(r.reason_name, 'Unknown')   AS reason_name,
  GREATEST(e.start_ts, :from_ts)       AS start_ts,
  LEAST(e.end_ts,   :to_ts)            AS end_ts,
  TIMESTAMPDIFF(SECOND,
    GREATEST(e.start_ts, :from_ts),
    LEAST(e.end_ts,   :to_ts))         AS duration_sec,
  e.note
FROM fact_state_event e
JOIN dim_state  s ON s.state_id  = e.state_id
LEFT JOIN dim_reason r ON r.reason_id = e.reason_id
WHERE
  e.start_ts < :to_ts AND e.end_ts > :from_ts
  AND e.line_id = :line_id
ORDER BY e.start_ts
LIMIT :limit OFFSET :offset
""".strip()
