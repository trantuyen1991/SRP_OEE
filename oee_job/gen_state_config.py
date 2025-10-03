# gen_state_config.py
# Tạo STATE_MAP và STATE_WEIGHTS từ dim_reason.json

import json
from collections import Counter, defaultdict
from pathlib import Path

# === 1) Khai báo state_id đang dùng trong dim_state (CHỈNH NẾU KHÁC) ===
STATE_ID_DEFAULTS = {
    "RUN": 1,
    "STOP": 2,
    "ALARM": 3,
    "SETUP": 4,
    "BREAK": 5,
    "PM": 6,
    "IDLE": 7,
}

# === 2) Đường dẫn file JSON danh mục reason ===
JSON_PATH = Path(r"C:\Users\trant\Downloads\dim_reason.json")  # đổi nếu file nằm nơi khác

# === 3) Quy tắc map nhóm lỗi -> STATE (dùng chữ có dấu/không dấu đều ổn) ===
# Ưu tiên theo thứ tự: nếu khớp KEY nào trước thì dùng state đó
GROUP_KEYWORDS_TO_STATE = [
    # (tu_khoa, state)
    ("dừng có kế hoạch", "PM"),
    ("chuyển đổi mẹ", "SETUP"),
    ("cài đặt máy", "SETUP"),
    ("gián đoạn sản xuất", "BREAK"),   # họp/5S/nghỉ giải lao/ăn cơm
    ("chất lượng", "STOP"),            # lỗi chất lượng -> xem như STOP
    ("vận chuyển", "STOP"),            # thành phẩm đầy, vệ sinh đầu fill...
    ("dừng thiết bị", "STOP"),         # thiết bị (bơm/gantry/khác)
    ("dừng sản xuất", "STOP"),
    ("lỗi máy dán tem", "STOP"),
]

# Nếu muốn tách ALARM riêng: dùng severity >= ngưỡng
ALARM_SEVERITY_THRESHOLD = 3  # severity >= 3 -> ALARM, ngược lại theo rule ở trên

# === 4) Trọng số default (nếu không tính theo tần suất) ===
DEFAULT_WEIGHTS = {
    "RUN": 0.70,
    "STOP": 0.10,
    "SETUP": 0.05,
    "PM": 0.02,
    "BREAK": 0.04,
    "ALARM": 0.04,
    "IDLE": 0.05,   # nếu bạn không dùng IDLE có thể đặt 0.0
}

# === 5) Helper: bỏ dấu/chuẩn hóa tiếng Việt rất đơn giản cho so khớp từ khóa ===
import unicodedata
def normalize(s: str) -> str:
    s = (s or "").lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s

def map_reason_to_state(reason_group: str, severity: int | None) -> str:
    rg = normalize(reason_group)
    if severity is not None and severity >= ALARM_SEVERITY_THRESHOLD:
        return "ALARM"
    for key, state in GROUP_KEYWORDS_TO_STATE:
        if normalize(key) in rg:
            return state
    # không khớp thì coi như STOP (downtime)
    return "STOP"

def main():
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))

    # Hầu hết file dim_reason.json dạng list of objects
    # { reason_id, reason_code, reason_name, reason_group, is_planned, severity, ... }
    # Gom tần suất theo state
    state_counter = Counter()
    by_state_samples = defaultdict(list)

    for r in data:
        rg = r.get("reason_group") or ""
        sev = r.get("severity")
        try:
            sev = int(sev) if sev is not None else None
        except Exception:
            sev = None
        state = map_reason_to_state(rg, sev)
        state_counter[state] += 1
        by_state_samples[state].append((r.get("reason_id"), r.get("reason_name")))

    # Xây STATE_MAP (theo ID defaults ở trên)
    STATE_MAP = {k: int(v) for k, v in STATE_ID_DEFAULTS.items()}

    # Tính trọng số từ tần suất trong file (chuẩn hóa về tổng 1.0)
    total = sum(state_counter.values()) or 1
    weights_from_data = {k: state_counter[k] / total for k in STATE_MAP.keys()}

    # Nếu muốn dùng DEFAULT_WEIGHTS thay vì theo dữ liệu, đổi flag này
    USE_DEFAULT = False

    if USE_DEFAULT:
        weights_raw = {k: float(DEFAULT_WEIGHTS.get(k, 0.0)) for k in STATE_MAP.keys()}
    else:
        weights_raw = weights_from_data

    # Chuẩn hóa tổng = 1.0 (phòng sai số)
    s = sum(weights_raw.values()) or 1.0
    STATE_WEIGHTS = {k: round(v / s, 6) for k, v in weights_raw.items()}

    # In ra màn hình dưới dạng Python literal để copy vào AVEVA script
    print("# --- STATE_MAP (copy vào AVEVA) ---")
    print("STATE_MAP = {")
    for k in STATE_MAP:
        print(f"    '{k}': {STATE_MAP[k]},")
    print("}\n")

    print("# --- STATE_WEIGHTS (copy vào AVEVA) ---")
    print("STATE_WEIGHTS = {")
    for k in STATE_MAP:
        print(f"    '{k}': {STATE_WEIGHTS.get(k, 0.0)},")
    print("}\n")

    # Thông tin tham khảo: mỗi state có những reason nào (mẫu)
    print("# Tham khảo: nhóm -> ví dụ reason")
    for st, items in by_state_samples.items():
        sample = ", ".join(str(x[0]) for x in items[:5])
        print(f"#  {st}: {state_counter[st]} mã (ví dụ reason_id: {sample})")

if __name__ == "__main__":
    main()
