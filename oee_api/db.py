# db.py
"""
DB access via SQLAlchemy Engine (MySQL + PyMySQL).
- Preserve old signature: query_df(sql: str, params: Optional[Sequence|dict])
- Accepts positional "?" placeholders and converts them to named binds (:p0, :p1, ...)
- Returns pandas.DataFrame
"""
from __future__ import annotations
from typing import Sequence, Optional, Tuple
import re
import logging
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from oee_api.config import SETTINGS

logger = logging.getLogger("oee.db")

# ---------- Engine ----------
_engine: Engine = create_engine(
    SETTINGS.DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=1800,   # recycle stale conns
    pool_size=10,
    max_overflow=20,
    isolation_level="AUTOCOMMIT",
    future=True,         # SQLAlchemy 2.0 style
)

def engine() -> Engine:
    return _engine

# ---------- Helpers ----------
_QUESTION_MARK = re.compile(r"\?")

def _to_named_binds(sql: str, params: Sequence) -> Tuple[str, dict]:
    """
    Convert positional '?' placeholders to named binds :p0, :p1, ...
    Only used when 'params' is a sequence (list/tuple).
    """
    bind_names = []
    def repl(_):
        idx = len(bind_names)
        name = f"p{idx}"
        bind_names.append(name)
        return f":{name}"

    sql_named = _QUESTION_MARK.sub(repl, sql)
    bind_dict = {name: params[i] for i, name in enumerate(bind_names)}
    return sql_named, bind_dict

# ---------- Public API ----------
def query_df(sql: str, params: Optional[Sequence|dict] = None) -> pd.DataFrame:
    """
    Execute a SELECT and return DataFrame.
    - If params is list/tuple → treat SQL as using '?' and convert to named.
    - If params is dict → use as-is (SQL should contain :named binds).
    """
    with engine().connect() as conn:
        if params is None:
            logger.debug("SQL(no params): %s", sql)
            df = pd.read_sql_query(text(sql), conn)
        elif isinstance(params, (list, tuple)):
            sql_named, bind_dict = _to_named_binds(sql, params)
            logger.debug("SQL(positional): %s | binds=%s", sql_named, bind_dict)
            df = pd.read_sql_query(text(sql_named), conn, params=bind_dict)
        else:
            # dict
            logger.debug("SQL(named): %s | binds=%s", sql, params)
            df = pd.read_sql_query(text(sql), conn, params=params)
    return df
