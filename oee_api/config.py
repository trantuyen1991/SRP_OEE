# config.py
import os
from dataclasses import dataclass

@dataclass
class _Settings:
    CORS_ALLOW_ORIGINS: list = None
    API_KEY: str = ""
    # --- NEW: SQLAlchemy DSN (for API server) ---
    DATABASE_URL: str = ""
    # (Giữ lại ODBC_CONN_STR cho AVEVA Edge nếu bạn cần ở nơi khác)
    ODBC_CONN_STR: str = ""

    DEFAULT_SINCE_MIN: int = 240
    DEFAULT_LIMIT: int = 2000
    HARD_MAX_LIMIT: int = 10000
    MAX_ROWS_FUSE: int = 500000
    TIMEZONE: str = "Asia/Ho_Chi_Minh"

    

    def __post_init__(self):
        if self.CORS_ALLOW_ORIGINS is None:
            allow_all = os.getenv("CORS_ALLOW_ALL", "1") == "1"
            if allow_all:
                self.CORS_ALLOW_ORIGINS = ["*"]
            else:
                origins = os.getenv("CORS_ALLOW_ORIGINS", "*")
                self.CORS_ALLOW_ORIGINS = [o.strip() for o in origins.split(",") if o.strip()]

        self.API_KEY = os.getenv("API_KEY", "")

        # --- NEW: prefer SQLAlchemy DSN for API server ---
        self.DATABASE_URL = os.getenv(
            "DATABASE_URL",
            "mysql+pymysql://root:root@127.0.0.1:3306/mpy_oee?charset=utf8mb4"
        )

        # Keep for AVEVA Edge or tools that still expect ODBC
        self.ODBC_CONN_STR = os.getenv(
            "ODBC_CONN_STR",
            os.getenv("MYSQL_ODBC_CONN_STR", "")
        )

SETTINGS = _Settings()
