"""Data I/O helpers for SQLite loading and basic cleaning. (SQLite 读取与基础清洗工具)"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable
import sqlite3
import pandas as pd
import numpy as np

CORE_TABLES: Iterable[str] = (
    "bookings", "tickets", "ticket_flights", "flights",
    "boarding_passes", "seats", "aircrafts_data", "airports_data",
)

def _connect_readonly(db_path: Path | str) -> sqlite3.Connection:
    """Open SQLite DB in read-only mode to avoid accidental creation."""
    if isinstance(db_path, str) and db_path.strip() == ":memory:":
        return sqlite3.connect(":memory:")
    path = Path(db_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"SQLite DB not found: {path}")
    uri = path.resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def find_sqlite(db_dir: Path | str) -> Path:
    """Return the first *.db under db_dir. Raise if none is found. (返回目录下第一个 *.db/*.sqlite 文件；若不存在则抛错)"""
    db_dir = Path(db_dir)
    cand = sorted(list(db_dir.glob("*.db")) + list(db_dir.glob("*.sqlite")))
    if not cand:
        raise FileNotFoundError(
            f"No SQLite DB under {db_dir}. Run `python scripts/fetch_data.py` "
            "or place your database into data/raw/."
        )
    return cand[0]


def read_table(
    con: sqlite3.Connection,
    table: str,
    columns: Iterable[str] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Read a table with optional column selection and row limit. (读取表，可选列与行数上限)"""
    cols = "*"
    if columns:
        cols = ", ".join(columns)
    sql = f"SELECT {cols} FROM {table}"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return pd.read_sql(sql, con)


def load_core_tables(
    db_path: Path | str,
    table_names: Iterable[str] | None = None,
    columns: Dict[str, Iterable[str]] | None = None,
    row_limit: Dict[str, int] | None = None,
) -> Dict[str, pd.DataFrame]:
    """Load selected tables from the SQLite database. (从 SQLite 加载指定表，可按列/行数限制)"""
    con = _connect_readonly(db_path)
    out: Dict[str, pd.DataFrame] = {}
    if table_names is None:
        table_names = CORE_TABLES
    if columns is None:
        columns = {}
    if row_limit is None:
        row_limit = {}
    try:
        for t in table_names:
            try:
                out[t] = read_table(con, t, columns=columns.get(t), limit=row_limit.get(t))
            except sqlite3.OperationalError as e:
                # Skip missing tables to support dataset variants / 兼容数据变体中的缺表情况
                if "no such table" in str(e).lower():
                    continue
                raise
    finally:
        con.close()
    return out


def replace_literal_N(df: pd.DataFrame) -> pd.DataFrame:
    """Replace textual '\\N' sentinels with NaN for object/string columns. (将对象列/字符串列中的文本 '\\N' 替换为 NaN)"""
    d = df.copy()
    for c in d.select_dtypes(include=["object", "string"]).columns:
        d[c] = d[c].replace({"\\N": np.nan})
    return d
