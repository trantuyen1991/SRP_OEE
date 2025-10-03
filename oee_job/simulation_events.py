
# import pyodbc
# import random
# import datetime as dt
# from typing import List, Dict
# import math
# # ----------------------------------------------------------
# # Config (điều chỉnh theo site)
# # ----------------------------------------------------------
# DSN  = "MySQL_OEE"      # Tên DSN bạn đã tạo
# USER = "root"
# PWD  = "root"
# DB   = "mpy_oee"

# # ===== Batch config (dùng khi run_batch) =====
# BATCH_ENABLE   = False      # True => chạy batch; False => chạy live như cũ
# BATCH_MINUTES  = 60         # bơm 60 phút gần nhất
# BATCH_STEP_SEC = 30         # mỗi 30 giây 1 bước

# # Các line muốn giả lập
# LINE_IDS: List[int] = [103, 104, 105, 106, 107]

# # Trọng số giả lập trạng thái (tổng ~ 1.0)
# STATE_WEIGHTS: Dict[str, float] = {
#     "RUN":   0.78,
#     "STOP":  0.08,
#     "ALARM": 0.04,
#     "SETUP": 0.035,
#     "BREAK": 0.025,
#     "IDLE":  0.02,
#     "PM":  0.01,
# }

# # Map state string -> state_id INT (khớp dim_state)
# STATE_MAP: Dict[str, int] = {
#     "RUN":   1,
#     "STOP":  2,
#     "ALARM": 3,
#     "SETUP": 4,
#     "BREAK": 5,
#     "PM":    6,
#     "IDLE":  7,
# }

# # Bật nếu muốn seed nhanh dim_state (1..7) trước khi chạy
# SEED_DIM_STATE_IF_MISSING = False

# # Dùng local time hay UTC để ghi ts
# USE_LOCAL_TIME = True


# # ----------------------------------------------------------
# # Helpers nội bộ
# # ----------------------------------------------------------
# # Cấu hình dwell trung bình (giây) cho từng state
# STATE_MEAN_SEC = {
#     1: 8*60,   # RUN  ~ 8 phút
#     2: 2*60,   # STOP ~ 2 phút
#     3: 1*60,   # ALARM ~ 1 phút
#     4: 5*60,   # SETUP ~ 5 phút
#     5: 15*60,  # BREAK ~ 15 phút
#     6: 30*60,  # PM ~ 30 phút
#     7: 10*60,  # IDLE ~ 10 phút
# }

# def _get_last_state(conn, line_id: int):
#     """Trả (state_id, ts) gần nhất của line; None nếu chưa có."""
#     try:
#         with conn.cursor() as cur:
#             cur.execute(
#                 "SELECT state_code, ts FROM fact_state_raw WHERE line_id=? ORDER BY ts DESC LIMIT 1",
#                 (int(line_id),)
#             )
#             row = cur.fetchone()
#             if row:
#                 return int(row[0]), row[1]  # (state_id, ts)
#     except Exception as ex:
#         try:
#             log_warn(f"_get_last_state error (line={line_id}): {ex}")
#         except Exception:
#             pass
#     return None, None

# def _choose_next_state(conn, line_id: int, now_dt) -> (int, str):
#     """
#     Chọn state mới: giữ nguyên theo p_stay hoặc chuyển theo trọng số.
#     Trả về (state_id, state_name_note).
#     """
#     # Lấy state cũ
#     prev_id, prev_ts = _get_last_state(conn, line_id)

#     # Nếu chưa có state trước -> bốc theo weights ban đầu
#     if not prev_id:
#         s_name = _choose_state()               # "RUN"/"STOP"/...
#         return STATE_MAP.get(s_name, 1), s_name

#     # Tính p_stay theo mean dwell
#     try:
#         mean_sec = max(STATE_MEAN_SEC.get(prev_id, 60), 1)
#         # delta = thời gian từ lần ghi gần nhất; nếu AVEVA chạy mỗi 1-10s thì delta đó đủ dùng
#         if isinstance(prev_ts, str):
#             # một số ODBC trả str
#             prev_dt = dt.datetime.fromisoformat(prev_ts.split('.')[0])
#         else:
#             prev_dt = prev_ts  # pyodbc trả datetime
#         delta_sec = max((now_dt - prev_dt).total_seconds(), 0.0)
#         p_stay = math.exp(-delta_sec / float(mean_sec))
#     except Exception as ex:
#         try:
#             log_warn(f"p_stay calc error (line={line_id}): {ex}, fallback p_stay=0.8")
#         except Exception:
#             pass
#         p_stay = 0.8

#     try:
#         r = random.random()
#         if r < p_stay:
#             # Giữ nguyên
#             # Tìm tên state cho note
#             inv_map = {v: k for k, v in STATE_MAP.items()}
#             return prev_id, inv_map.get(prev_id, "RUN")
#     except Exception:
#         # nếu có lỗi vẫn tiếp tục chọn mới
#         pass

#     # Chuyển sang state khác theo trọng số, tránh quay về prev_id ngay
#     try:
#         keys = list(STATE_WEIGHTS.keys())
#         wts  = list(STATE_WEIGHTS.values())
#         # loại trừ state cũ
#         inv_map = {v: k for k, v in STATE_MAP.items()}
#         prev_name = inv_map.get(prev_id, None)
#         if prev_name in keys:
#             i = keys.index(prev_name)
#             keys.pop(i); wts.pop(i)
#         s_name = random.choices(keys, weights=wts, k=1)[0]
#         return STATE_MAP.get(s_name, 1), s_name
#     except Exception as ex:
#         try:
#             log_warn(f"choose transition failed: {ex}, fallback RUN")
#         except Exception:
#             pass
#         return 1, "RUN"

# def _now_str() -> str:
#     """MySQL DATETIME(6) string (microseconds để hạn chế đụng PK)."""
#     t = dt.datetime.now() if USE_LOCAL_TIME else dt.datetime.utcnow()
#     return t.strftime("%Y-%m-%d %H:%M:%S.%f")


# def _choose_state() -> str:
#     """Chọn 1 state theo trọng số."""
#     try:
#         keys = list(STATE_WEIGHTS.keys())
#         wts  = list(STATE_WEIGHTS.values())
#         return random.choices(keys, weights=wts, k=1)[0]
#     except Exception as ex:
#         # fallback an toàn
#         try:
#             log_warn(f"random.choices failed: {ex}, fallback RUN")
#         except Exception:
#             pass
#         return "RUN"


# def _connect() -> pyodbc.Connection:
#     """Kết nối ODBC DSN/USER/PWD/DB (autocommit)."""
#     conn = None
#     try:
#         conn = pyodbc.connect(
#             f"DSN={DSN};UID={USER};PWD={PWD};DATABASE={DB};",
#             autocommit=True,
#             timeout=5,
#         )
#         # Thiết lập encoding (nếu driver hỗ trợ)
#         try:
#             conn.setdecoding(pyodbc.SQL_CHAR, encoding="utf-8")
#             conn.setdecoding(pyodbc.SQL_WCHAR, encoding="utf-8")
#             conn.setencoding(encoding="utf-8")
#         except Exception:
#             pass
#         try:
#             log_info(f"Connected to DSN={DSN}, DB={DB}")
#         except Exception:
#             pass
#         return conn
#     except Exception as ex:
#         try:
#             log_error(f"DB connect failed: {ex}")
#         except Exception:
#             print(f"[ERROR] DB connect failed: {ex}")
#         # Trả None để caller xử lý
#         return None


# def _ensure_dim_state(conn: pyodbc.Connection) -> None:
#     """Tuỳ chọn: seed nhanh dim_state nếu thiếu các state id 1..7."""
#     if not SEED_DIM_STATE_IF_MISSING:
#         return
#     try:
#         with conn.cursor() as cur:
#             cur.execute("SELECT state_id, state_code FROM dim_state")
#             existing = {int(r[0]): str(r[1]) for r in cur.fetchall()}
#             need = []
#             required = [
#                 (1, 'RUN',   'Running',             'RUNNING',      0, 0, '#2ecc71', 10),
#                 (2, 'STOP',  'Unplanned Stop',      'STOPPED',      0, 1, '#e74c3c', 20),
#                 (3, 'ALARM', 'Alarm / Fault',       'STOPPED',      0, 1, '#c0392b', 25),
#                 (4, 'SETUP', 'Setup / Changeover',  'STOPPED',      1, 1, '#f39c12', 30),
#                 (5, 'BREAK', 'Break / Lunch',       'PLANNED_STOP', 1, 0, '#3498db', 50),
#                 (6, 'PM',    'Planned Maintenance', 'PLANNED_STOP', 1, 1, '#9b59b6', 60),
#                 (7, 'IDLE',  'Idle / No Order',     'STOPPED',      1, 0, '#95a5a6', 40),
#             ]
#             for row in required:
#                 sid, scode = row[0], row[1]
#                 if sid not in existing:
#                     need.append(row)
#             if need:
#                 sql = (
#                     "INSERT INTO dim_state "
#                     "(state_id, state_code, state_name, category, is_planned, is_downtime, color_hex, sort_order) "
#                     "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
#                 )
#                 for row in need:
#                     try:
#                         cur.execute(sql, row)
#                     except Exception as exx:
#                         try:
#                             log_warn(f"Seed dim_state failed for {row}: {exx}")
#                         except Exception:
#                             pass
#                 try:
#                     log_info(f"Seeded dim_state rows: {len(need)}")
#                 except Exception:
#                     pass
#             else:
#                 try:
#                     log_info("dim_state already has required 1..7")
#                 except Exception:
#                     pass
#     except Exception as ex:
#         try:
#             log_warn(f"ensure dim_state error: {ex}")
#         except Exception:
#             pass


# def _upsert_fact_state_raw(conn: pyodbc.Connection, ts_str: str, line_id: int, state_id: int, note: str) -> None:
#     """Ghi 1 dòng vào fact_state_raw (PK: ts + line_id)."""
#     try:
#         sql = (
#             "INSERT INTO fact_state_raw (ts, line_id, state_code, note) "
#             "VALUES (?, ?, ?, ?) "
#             "ON DUPLICATE KEY UPDATE state_code=VALUES(state_code), note=VALUES(note)"
#         )
#         with conn.cursor() as cur:
#             cur.execute(sql, (ts_str, int(line_id), int(state_id), note))
#     except Exception as ex:
#         # Để caller đếm số lỗi
#         raise ex

# def run_batch() -> int:
#     """
#     Backfill dữ liệu quá khứ cho từng line:
#       - Từ now - BATCH_MINUTES -> now, bước BATCH_STEP_SEC
#       - Giữ trạng thái theo p_stay
#       - CHỈ ghi khi state đổi (giống thực tế)
#     """
#     conn = _connect()
#     if conn is None:
#         return 2

#     try:
#         _ensure_dim_state(conn)
#     except Exception:
#         pass

#     now = dt.datetime.now() if USE_LOCAL_TIME else dt.datetime.utcnow()
#     start = now - dt.timedelta(minutes=BATCH_MINUTES)

#     try:
#         log_info(f"[BATCH] start={start} end={now} step={BATCH_STEP_SEC}s lines={LINE_IDS}")
#     except Exception:
#         pass

#     # trạng thái hiện tại theo từng line (để chỉ ghi khi đổi)
#     last_state: dict[int, int] = {}
#     last_ts:    dict[int, dt.datetime] = {}

#     # khởi tạo từ DB (nếu có)
#     for lid in LINE_IDS:
#         sid, ts_prev = _get_last_state(conn, lid)
#         if sid:
#             last_state[lid] = sid
#             last_ts[lid] = ts_prev if isinstance(ts_prev, dt.datetime) else start

#     inserted = 0
#     ts = start
#     while ts <= now:
#         ts_str = ts.strftime("%Y-%m-%d %H:%M:%S.%f")
#         for lid in LINE_IDS:
#             try:
#                 s_id, s_name = _choose_next_state(conn, lid, ts)
#                 # chỉ ghi khi có thay đổi hoặc line chưa có gì
#                 if last_state.get(lid) != s_id:
#                     _upsert_fact_state_raw(conn, ts_str, lid, s_id, s_name)
#                     last_state[lid] = s_id
#                     last_ts[lid] = ts
#                     inserted += 1
#             except Exception as ex:
#                 try:
#                     log_error(f"[BATCH] insert failed line={lid} ts={ts_str}: {ex}")
#                 except Exception:
#                     pass
#         ts += dt.timedelta(seconds=BATCH_STEP_SEC)

#     try:
#         log_info(f"[BATCH] done. inserted_rows={inserted}")
#     except Exception:
#         pass

#     try:
#         conn.close()
#     except Exception:
#         pass

#     return 0

# # ----------------------------------------------------------
# # Entry point (gọi từ Scheduler)
# # ----------------------------------------------------------
# def main() -> int:
#     if BATCH_ENABLE:
#         return run_batch()

#     # ---- live (1 tick) như bạn đang dùng ----
#     ts_str = _now_str()
#     try:
#         log_info(f"Inject at ts={ts_str} for lines={LINE_IDS}")
#     except Exception:
#         pass

#     conn = _connect()
#     if conn is None:
#         return 2

#     try:
#         _ensure_dim_state(conn)
#     except Exception:
#         pass

#     success = fail = 0
#     now_dt = dt.datetime.now() if USE_LOCAL_TIME else dt.datetime.utcnow()
#     for lid in LINE_IDS:
#         try:
#             s_id, s_name = _choose_next_state(conn, lid, now_dt)
#             _upsert_fact_state_raw(conn, ts_str, lid, s_id, s_name)
#             success += 1
#         except Exception as ex:
#             fail += 1
#             try:
#                 log_error(f"Insert failed (line={lid}): {ex}")
#             except Exception:
#                 pass

#     try:
#         conn.close()
#     except Exception:
#         pass

#     try:
#         log_info(f"Done: success={success} fail={fail}")
#     except Exception:
#         pass

#     return 0 if fail == 0 else 2


# if __name__ == "__main__":
#     rc = 0
#     try:
#         rc = main()
#     except Exception as ex:
#         try:
#             log_error(f"Unhandled error: {ex}")
#         except Exception:
#             print(f"[ERROR] Unhandled error: {ex}")
#         rc = 2

#     # reset trigger nếu bạn dùng tag kích
#     try:
#         HMI.SetTagValue("$MySQL_Upsert_state_TRG", 0)
#     except Exception:
#         pass

#     # chỉ exit khi chạy ngoài AVEVA
#     if "HMI" not in globals():
#         try:
#             import sys
#             sys.exit(rc)
#         except Exception:
#             pass
