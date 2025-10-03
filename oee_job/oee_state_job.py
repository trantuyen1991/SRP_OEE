"""
oee_state_job.py

Job nén trạng thái máy:
- Đọc từ fact_state_raw (ts, line_id, state_code, note)
- Gom theo line, sắp thời gian, nén các block liên tiếp có cùng state_code
- Tạo event [start_ts, end_ts) và upsert vào fact_state_event
- Cắt "đuôi mở" của quá khứ đến thời điểm now (nếu bản ghi cuối chưa có next)

Chạy bằng Windows Task Scheduler (Python 64-bit) — độc lập AVEVA.

Author: <Bạn>
"""

import os
import logging
from datetime import datetime, timedelta, timezone
import pandas as pd
from sqlalchemy import create_engine, text

# ============================================================
# CẤU HÌNH
# ============================================================
DB_URL = os.getenv(
    "DB_URL",
    "mysql+mysqlconnector://root:root@127.0.0.1:3306/mpy_oee"
)

INCREMENTAL    = int(os.getenv("STATE_INCR", "1"))          # 1=incr (default), 0=full
OVERLAP_SEC    = int(os.getenv("STATE_OVERLAP_SEC", "300"))  # overlap để an toàn (giây)
LOOKBACK_HOURS = int(os.getenv("STATE_LOOKBACK_HOURS", "8")) # dùng cho full
LINE_IDS_ENV   = os.getenv("STATE_LINE_IDS", "")             # "103,104" hoặc rỗng
# Nếu muốn đóng event cuối “mở” (chưa có end) ở quá khứ, đặt True
CLOSE_LAST_TO_NOW = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [oee_state_job] %(levelname)s: %(message)s",
)

ENGINE = create_engine(DB_URL, pool_recycle=3600, pool_pre_ping=True)

# ============================================================
# TIỆN ÍCH
# ============================================================
def _parse_line_ids() -> list[int]:
    if not LINE_IDS_ENV.strip():
        # để trống => lấy tất cả line có raw trong khoảng đọc
        return []
    return [int(x) for x in LINE_IDS_ENV.split(",") if x.strip().isdigit()]

def _now_local_naive() -> datetime:
    return datetime.now(timezone.utc).astimezone().replace(tzinfo=None)

def get_last_starts_per_line(line_ids: list[int] | None = None) -> dict[int, datetime]:
    """
    Lấy watermark theo line: MAX(start_ts) từ fact_state_event.
    Nếu line_ids rỗng => lấy tất cả line đã có event.
    """
    cond = ""
    if line_ids:
        in_list = ",".join(str(i) for i in line_ids)
        cond = f"WHERE line_id IN ({in_list})"
    sql = text(f"""
      SELECT line_id, MAX(start_ts) AS last_start
      FROM fact_state_event
      {cond}
      GROUP BY line_id
    """)
    df = pd.read_sql(sql, ENGINE)
    out = {}
    for _, r in df.iterrows():
        out[int(r["line_id"])] = pd.to_datetime(r["last_start"]).to_pydatetime()
    return out

def load_raw_since(min_since: datetime, line_ids: list[int] | None = None) -> pd.DataFrame:
    base_sql = """
        SELECT ts, line_id, state_code, reason_code, note
        FROM fact_state_raw
        WHERE ts >= :since
    """
    params = {"since": min_since}
    if line_ids:
        in_list = ",".join(str(i) for i in line_ids)
        sql = text(base_sql + f" AND line_id IN ({in_list}) ORDER BY line_id, ts")
    else:
        sql = text(base_sql + " ORDER BY line_id, ts")
    df = pd.read_sql(sql, ENGINE, params=params)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"])
    logging.info("Loaded raw: %d rows (since=%s, lines=%s)", len(df), min_since, line_ids or "ALL")
    return df

def compress_events_with_cut(df: pd.DataFrame, end_clip: datetime,
                             cut_from_per_line: dict[int, datetime] | None = None) -> pd.DataFrame:
    """
    Như compress_events nhưng chỉ emit event có end_ts >= cut_from_per_line[line]
    để tránh tạo lại toàn bộ quá khứ trong incremental.
    """
    if df.empty:
        return pd.DataFrame(columns=["line_id","state_id","start_ts","end_ts","reason_id","note"])

    df = df.sort_values(["line_id","ts"]).copy()
    df["state_id"] = df["state_code"].astype(int)
    out_rows = []

    for lid, g in df.groupby("line_id", sort=False):
        g = g.sort_values("ts").reset_index(drop=True)
        changed = (g["state_id"] != g["state_id"].shift(1)).fillna(True)
        starts = g[changed].copy()
        starts["start_ts"] = starts["ts"]
        starts["end_ts"]   = starts["start_ts"].shift(-1)
        starts["end_ts"]   = starts["end_ts"].fillna(end_clip)

        cut_from = None
        if cut_from_per_line and lid in cut_from_per_line:
            cut_from = cut_from_per_line[lid]

        for _, r in starts.iterrows():
            s_ts = pd.to_datetime(r["start_ts"]).to_pydatetime()
            e_ts = pd.to_datetime(r["end_ts"]).to_pydatetime()
            # nếu có ngưỡng cut_from: chỉ lấy những event chạm vùng cần cập nhật
            if cut_from is not None and e_ts < cut_from:
                continue
            out_rows.append({
                "line_id": int(lid),
                "state_id": int(r["state_id"]),
                "start_ts": s_ts,
                "end_ts":   e_ts,
                "reason_id": None,
                "note": None if pd.isna(r.get("note")) else str(r.get("note")),
            })

    ev = pd.DataFrame(out_rows)
    logging.info("Compressed (filtered) to %d events", len(ev))
    return ev
# ============================================================
# I/O
# ============================================================
def load_raw(since_dt: datetime, line_ids: list[int]) -> pd.DataFrame:
    """
    Đọc raw trong cửa sổ từ since_dt → now.
    Nếu line_ids rỗng => lấy tất cả line.
    """
    base_sql = """
        SELECT ts, line_id, state_code, reason_code, note
        FROM fact_state_raw
        WHERE ts >= :since
    """
    params = {"since": since_dt}
    if line_ids:
        # lọc theo danh sách line
        in_list = ",".join([str(i) for i in line_ids])
        sql = text(base_sql + f" AND line_id IN ({in_list}) ORDER BY line_id, ts")
    else:
        sql = text(base_sql + " ORDER BY line_id, ts")
    df = pd.read_sql(sql, ENGINE, params=params)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"])
    logging.info("Loaded raw: %d rows (since=%s, lines=%s)", len(df), since_dt, line_ids or "ALL")
    return df

# ============================================================
# NÉN RAW → EVENT
# ============================================================
def compress_events(df: pd.DataFrame, end_clip: datetime) -> pd.DataFrame:
    """
    Nén theo từng line:
    - Với mỗi line: sort theo ts, tính 'changed' khi state_code != state_code.shift(1)
    - start = ts tại các điểm changed==True
    - end   = start của record kế tiếp (cùng line); nếu None -> end_clip
    Trả DataFrame: line_id, state_id, start_ts, end_ts, reason_id(NULL), note(last)
    """
    if df.empty:
        return pd.DataFrame(columns=["line_id","state_id","start_ts","end_ts","reason_id","note"])

    df = df.sort_values(["line_id","ts"]).copy()
    # state_code trong raw đã là INT — map sang state_id giữ nguyên
    df["state_id"] = df["state_code"].astype(int)

    out_rows = []
    for lid, g in df.groupby("line_id", sort=False):
        g = g.sort_values("ts").reset_index(drop=True)
        # vị trí bắt đầu block
        changed = (g["state_id"] != g["state_id"].shift(1)).fillna(True)
        starts = g[changed].copy()
        starts["start_ts"] = starts["ts"]

        # end = start kế tiếp (trong line)
        starts["end_ts"] = starts["start_ts"].shift(-1)
        # điền end None bằng end_clip (now) nếu muốn đóng “mở”
        if CLOSE_LAST_TO_NOW:
            starts["end_ts"] = starts["end_ts"].fillna(end_clip)
        else:
            # vẫn phải bỏ các block chưa có end (để lại cho lần sau)
            starts = starts[~starts["end_ts"].isna()]

        # note: lấy note tại dòng start (hoặc None)
        for _, r in starts.iterrows():
            out_rows.append({
                "line_id": int(lid),
                "state_id": int(r["state_id"]),
                "start_ts": pd.to_datetime(r["start_ts"]).to_pydatetime(),
                "end_ts":   pd.to_datetime(r["end_ts"]).to_pydatetime(),
                "reason_id": None,                       # chưa gán lý do ở mức raw
                "note": None if pd.isna(r.get("note")) else str(r.get("note")),
            })
    ev = pd.DataFrame(out_rows)
    logging.info("Compressed to %d events", len(ev))
    return ev

# ============================================================
# UPSERT EVENT
# ============================================================
UPSERT_SQL = text("""
INSERT INTO fact_state_event
  (line_id, machine_id, state_id, reason_id, start_ts, end_ts, shift_id, po, packaging_id, note)
VALUES
  (:line_id, NULL, :state_id, :reason_id, :start_ts, :end_ts, NULL, NULL, NULL, :note)
ON DUPLICATE KEY UPDATE
  state_id=VALUES(state_id),
  reason_id=VALUES(reason_id),
  end_ts=VALUES(end_ts),
  note=VALUES(note)
""")
# ============================================================
# LOOKUP reason_id cho một event theo khoảng thời gian
# - Chỉ lấy các mã trong dim_reason (tránh nhầm RUN/IDLE…)
# - Ưu tiên bản ghi gần end_ts nhất trong khoảng [start_ts, end_ts)
# ============================================================
REASON_LOOKUP_SQL = text("""
    SELECT r.reason_id
    FROM fact_state_raw f
    JOIN dim_reason r ON r.reason_code = f.state_code
    WHERE f.line_id = :line_id
      AND f.ts >= :start_ts
      AND f.ts <  :end_ts
    ORDER BY f.ts DESC
    LIMIT 1
""")
# ƯU TIÊN: tra theo reason_code (máy gửi), bỏ qua reason_code = 0
REASON_BY_CODE_SQL = text("""
    SELECT r.reason_id
    FROM fact_state_raw f
    JOIN dim_reason r ON r.reason_code = f.reason_code
    WHERE f.line_id = :line_id
      AND f.reason_code <> 0
      AND f.ts >= :start_ts
      AND f.ts <  :end_ts
    ORDER BY f.ts DESC            -- lấy mã gần end_ts nhất
    LIMIT 1
""")

# PHỤ: nếu chưa có reason_code, có thể tra theo state_id + nhóm thời gian (thường sẽ không ra)
# giữ để fallback về sau (tuỳ site có nhu cầu hay không)
REASON_FALLBACK_SQL = text("""
    SELECT r.reason_id
    FROM fact_state_raw f
    JOIN dim_reason r ON r.state_id = :state_id
    WHERE f.line_id = :line_id
      AND f.ts >= :start_ts
      AND f.ts <  :end_ts
    ORDER BY f.ts DESC
    LIMIT 1
""")

STATE_MAP = {
    "RUN":   1,
    "STOP":  2,
    "ALARM": 3,
    "SETUP": 4,
    "IDLE":  5,
    "BREAK": 6,
    "PM":    7,
}
def upsert_events(ev: pd.DataFrame) -> int:
    """
    Upsert từng event vào fact_state_event.
    Yêu cầu có UNIQUE KEY (line_id, start_ts) để tránh trùng.
    """
    if ev.empty:
        logging.info("No events to upsert.")
        return 0

    rows = 0
    with ENGINE.begin() as conn:
        for _, r in ev.iterrows():
            line_id  = int(r["line_id"])
            state_id = int(r["state_id"])
            start_ts = r["start_ts"]
            end_ts   = r["end_ts"]
            note     = None if pd.isna(r.get("note")) else str(r.get("note"))

            # --- Lookup reason_id ---
            reason_id = None
            try:
                # 1) Ưu tiên tìm theo reason_code (máy gửi) trong khoảng event
                q = conn.execute(
                    REASON_BY_CODE_SQL,
                    {"line_id": line_id, "start_ts": start_ts, "end_ts": end_ts}
                ).fetchone()
                if q and q[0] is not None:
                    reason_id = int(q[0])
                else:
                    # 2) (Tuỳ chọn) fallback theo state_id
                    #    thường chỉ dùng khi site không có reason_code
                    if state_id not in (STATE_MAP["RUN"], STATE_MAP["IDLE"]):
                        q2 = conn.execute(
                            REASON_FALLBACK_SQL,
                            {
                                "line_id": line_id,
                                "state_id": state_id,
                                "start_ts": start_ts,
                                "end_ts": end_ts,
                            }
                        ).fetchone()
                        if q2 and q2[0] is not None:
                            reason_id = int(q2[0])
            except Exception as ex:
                logging.warning("reason lookup failed | line=%s %s..%s | %s",
                                line_id, start_ts, end_ts, ex)

            # 3) Nếu DataFrame đã có cột reason_id (do bạn tính trước), ưu tiên giá trị đó
            if "reason_id" in r and pd.notna(r["reason_id"]):
                reason_id = int(r["reason_id"])

            # --- UPSERT ---
            conn.execute(UPSERT_SQL, {
                "line_id":  line_id,
                "state_id": state_id,
                "reason_id": reason_id,  # có thể None nếu RUN/IDLE hoặc chưa xác định
                "start_ts": start_ts,
                "end_ts":   end_ts,
                "note":     note,
            })
            rows += 1

    logging.info("Upserted %d events into fact_state_event.", rows)
    return rows

# ============================================================
# MAIN
# ============================================================
def main() -> None:
    try:
        now = _now_local_naive()
        line_ids = _parse_line_ids()

        if INCREMENTAL:
            # 1) Lấy watermark per line
            last_map = get_last_starts_per_line(line_ids if line_ids else None)
            # 2) Với line chưa có event → cho since mặc định (now - LOOKBACK_HOURS)
            default_since = now - timedelta(hours=max(LOOKBACK_HOURS, 1))
            since_per_line: dict[int, datetime] = {}
            # nếu người dùng cung cấp cụ thể line_ids, dùng đúng list đó; nếu không, lấy toàn bộ line xuất hiện trong raw gần đây
            if line_ids:
                candidate_lines = line_ids
            else:
                # lấy danh sách line từ fact_state_raw trong khoảng default_since để tránh quét cả DB
                tmp = pd.read_sql(
                    text("SELECT DISTINCT line_id FROM fact_state_raw WHERE ts >= :since"),
                    ENGINE,
                    params={"since": default_since}
                )
                candidate_lines = [int(x) for x in tmp["line_id"].tolist()]

            for lid in candidate_lines:
                last_start = last_map.get(lid, default_since)
                since_per_line[lid] = last_start - timedelta(seconds=OVERLAP_SEC)

            # 3) Load raw kể từ min(since_per_line) cho tất cả line
            if since_per_line:
                min_since = min(since_per_line.values())
            else:
                min_since = default_since
            raw = load_raw_since(min_since=min_since, line_ids=candidate_lines if candidate_lines else None)

            # 4) Nén và cắt theo từng line
            ev = compress_events_with_cut(raw, end_clip=now, cut_from_per_line=since_per_line)
        else:
            # FULL REBUILD theo LOOKBACK_HOURS
            since = now - timedelta(hours=LOOKBACK_HOURS)
            raw = load_raw_since(min_since=since, line_ids=line_ids if line_ids else None)
            ev  = compress_events_with_cut(raw, end_clip=now, cut_from_per_line=None)

        upsert_events(ev)
        logging.info("oee_state_job finished (%s).", "INCREMENTAL" if INCREMENTAL else "FULL")

    except Exception as e:
        logging.exception("Unhandled exception in oee_state_job: %s", e)

if __name__ == "__main__":
    main()
