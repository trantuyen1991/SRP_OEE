"""
oee_job.py

Job tính toán và upsert dữ liệu OEE theo phút cho Line-105.
Chạy bằng Windows Task Scheduler (Python 64-bit) bên ngoài AVEVA Edge.

Luồng xử lý:
1. Load dữ liệu raw từ fact_counter_raw trong X giờ gần nhất.
2. Tính delta sản lượng, good, ng, runtime_sec, planned_sec cho từng phút.
3. Upsert kết quả vào fact_production_min.
4. Ghi log chi tiết từng bước để tiện theo dõi/debug.

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
# Chuỗi kết nối SQLAlchemy
LINE_ID = int(os.getenv("LINE_ID", "105"))  # Line cần xử lý
LOOKBACK_HOURS = int(os.getenv("LOOKBACK_HOURS", "4"))  # Số giờ đọc ngược lại
RUNTIME_MODE = os.getenv("RUNTIME_MODE", "any_change_60s")
# "any_change_60s": nếu có delta trong phút thì runtime=60s
# "coarse": nếu produced>0 thì runtime=60s

logging.basicConfig(
    filename="oee_job.log",
    level=logging.INFO,
    format="%(asctime)s [oee_job] %(levelname)s: %(message)s",
)

ENGINE = create_engine(DB_URL, pool_recycle=3600, pool_pre_ping=True)

# ============================================================
# HÀM TIỆN ÍCH
# ============================================================

def floor_minute(ts: pd.Timestamp) -> pd.Timestamp:
    """
    Làm tròn xuống phút gần nhất cho timestamp.
    """
    return ts.replace(second=0, microsecond=0)

# ============================================================
# HÀM XỬ LÝ DỮ LIỆU
# ============================================================

def load_raw_since(since_dt: datetime) -> pd.DataFrame:
    """
    Load dữ liệu raw từ bảng fact_counter_raw kể từ thời điểm since_dt.

    Args:
        since_dt (datetime): thời điểm bắt đầu đọc dữ liệu.

    Returns:
        DataFrame chứa các cột ts, line_id, process_order, packaging_id,
        counter_total, good_qty, ng_qty.
    """
    sql = text("""
        SELECT ts, line_id, process_order, packaging_id, counter_total, good_qty, ng_qty
        FROM fact_counter_raw
        WHERE line_id=:line_id AND ts>=:since
        ORDER BY ts
    """)
    df = pd.read_sql(sql, ENGINE, params={"line_id": LINE_ID, "since": since_dt})
    if not df.empty:
        df['ts'] = pd.to_datetime(df['ts'])
    logging.info("Loaded %d rows raw data since %s", len(df), since_dt)
    return df


def compute_minute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính dữ liệu theo phút từ dữ liệu raw.

    Logic:
    - Delta sản lượng = counter_total.diff()
    - Nếu có delta > 0 trong phút => runtime_sec=60 (mode any_change_60s).
    - Nếu produced>0 => runtime_sec=60 (mode coarse).
    - planned_sec = 60 mặc định.

    Args:
        df (DataFrame): dữ liệu raw.

    Returns:
        DataFrame: dữ liệu tổng hợp theo phút.
    """
    if df.empty:
        logging.warning("No raw data to compute.")
        return pd.DataFrame(columns=[
            "ts_min","line_id","process_order","packaging_id",
            "produced","good","ng","runtime_sec","planned_sec"
        ])

    df = df.sort_values("ts").copy()
    # Điền forward PO/packaging nếu thiếu
    df['process_order'] = df['process_order'].ffill()
    df['packaging_id']  = df['packaging_id'].ffill()

    # Delta sản lượng (chặn reset lùi)
    df['delta_total'] = df['counter_total'].diff().clip(lower=0).fillna(0).astype(int)
    df['delta_good'] = df['good_qty'].diff().clip(lower=0).fillna(0).astype(int)
    df['delta_ng']   = df['ng_qty'].diff().clip(lower=0).fillna(0).astype(int)

    # Gán phút
    df['ts_min'] = df['ts'].dt.floor('T')

    # Gom theo phút
    g = df.groupby('ts_min', as_index=False).agg({
        'process_order': 'last',
        'packaging_id' : 'last',
        'delta_total'  : 'sum',
        'delta_good'     : 'sum',
        'delta_ng'       : 'sum',
    })
    g.rename(columns={
        'delta_total': 'produced',
        'delta_good' : 'good',
        'delta_ng'   : 'ng'
    }, inplace=True)

    # Runtime logic
    if RUNTIME_MODE == 'any_change_60s':
        has_change = df.groupby('ts_min')['delta_total'].apply(lambda s: int((s > 0).any()))
        g = g.merge(has_change.rename('is_run'), on='ts_min', how='left')
        g['runtime_sec'] = (g['is_run'] * 60).astype(int)
        g.drop(columns=['is_run'], inplace=True)
    else:
        g['runtime_sec'] = (g['produced'] > 0).astype(int) * 60

    g['planned_sec'] = 60
    g['line_id'] = LINE_ID

    logging.info("Computed %d minute rows.", len(g))
    return g[['ts_min','line_id','process_order','packaging_id',
              'produced','good','ng','runtime_sec','planned_sec']]


def upsert_minute(dfm: pd.DataFrame) -> None:
    """
    Upsert dữ liệu vào bảng fact_production_min.

    Args:
        dfm (DataFrame): dữ liệu đã tổng hợp theo phút.
    """
    if dfm.empty:
        logging.warning("No minute data to upsert.")
        return

    rows = 0
    with ENGINE.begin() as conn:
        for _, r in dfm.iterrows():
            conn.execute(text("""
                INSERT INTO fact_production_min
                (ts_min,line_id,process_order,packaging_id,produced,good,ng,runtime_sec,planned_sec)
                VALUES (:ts_min,:line_id,:process_order,:packaging_id,:produced,:good,:ng,:runtime_sec,:planned_sec)
                ON DUPLICATE KEY UPDATE
                  process_order=VALUES(process_order),
                  packaging_id=VALUES(packaging_id),
                  produced=VALUES(produced),
                  good=VALUES(good),
                  ng=VALUES(ng),
                  runtime_sec=VALUES(runtime_sec),
                  planned_sec=VALUES(planned_sec)
            """), {
                "ts_min": r['ts_min'].to_pydatetime() if hasattr(r['ts_min'], 'to_pydatetime') else r['ts_min'],
                "line_id": int(r['line_id']),
                "process_order": None if pd.isna(r['process_order']) else str(r['process_order']),
                "packaging_id": None if pd.isna(r['packaging_id']) else int(r['packaging_id']),
                "produced": int(r['produced']),
                "good": int(r['good']),
                "ng": int(r['ng']),
                "runtime_sec": int(r['runtime_sec']),
                "planned_sec": int(r['planned_sec']),
            })
            rows += 1
    logging.info("Upserted %d rows into fact_production_min.", rows)

# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Chạy toàn bộ job:
    1. Load raw data.
    2. Compute minute aggregates.
    3. Upsert vào fact_production_min.
    """
    try:
        now = datetime.now(timezone.utc).astimezone().replace(tzinfo=None)
        since = now - timedelta(hours=LOOKBACK_HOURS)

        raw = load_raw_since(since)
        dfm = compute_minute(raw)
        upsert_minute(dfm)

        logging.info("Job finished successfully.")
    except Exception as e:
        logging.exception("Unhandled exception in oee_job: %s", e)

if __name__ == "__main__":
    main()
