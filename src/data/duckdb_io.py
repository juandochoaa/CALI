from __future__ import annotations

from pathlib import Path

import duckdb


def connect(db_path: str | Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db_path))


def load_csv(conn: duckdb.DuckDBPyConnection, path: str | Path, table: str) -> None:
    conn.execute(
        'CREATE OR REPLACE TABLE ' + table + ' AS SELECT * FROM read_csv_auto(?)',
        [str(path)],
    )
