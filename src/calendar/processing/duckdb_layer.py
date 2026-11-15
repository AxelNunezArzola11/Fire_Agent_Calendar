"""Minimal local DuckDB layer over normalized Parquet (views, no copies).

- Database file (default ``calendar.duckdb`` at the repo root): holds
  VIEWs over the hive-partitioned Parquet store plus nothing else.
- Recovery: delete the file and call :func:`init_db` again — views rebuild
  from the Parquet store in seconds. The Parquet files are the system of
  record; the DuckDB file is disposable cache.
- All query helpers open read-only connections with bound parameters.
"""

from __future__ import annotations

from pathlib import Path


def init_db(db_path: str, parquet_root: str) -> str:
    """Create (or recreate) views over the Parquet store. Idempotent."""
    import duckdb

    root = Path(parquet_root).resolve()
    # NOTE: DDL cannot take bound parameters in DuckDB; the glob is inlined.
    # It derives from our own settings (not user input) and is quote-escaped.
    glob = str(root / "*" / "product=*" / "sensor=*" / "year=*" / "month=*" / "*.parquet")
    glob = glob.replace("'", "''")
    con = duckdb.connect(db_path)
    try:
        # union_by_name: per-product src_* columns differ (e.g. MODIS has
        # src_brightness, VIIRS does not); missing columns read as NULL.
        # Found by Phase 2B integration (SELECT * over both products).
        con.execute(
            "CREATE OR REPLACE VIEW detections AS "
            f"SELECT * FROM read_parquet('{glob}', hive_partitioning = true, "
            "union_by_name = true)")
    finally:
        con.close()
    return db_path


def _connect(db_path: str):
    import duckdb

    return duckdb.connect(db_path, read_only=True)


def datasets(db_path: str) -> list[dict]:
    """Distinct (product, sensor) pairs present in the store."""
    con = _connect(db_path)
    try:
        rows = con.execute(
            "SELECT product, sensor, COUNT(*) n FROM detections "
            "GROUP BY product, sensor ORDER BY product, sensor").fetchall()
    finally:
        con.close()
    return [{"product": p, "sensor": s, "rows": n} for p, s, n in rows]


def count(db_path: str, *, product: str | None = None, sensor: str | None = None,
          start: str | None = None, end: str | None = None,
          bbox: tuple[float, float, float, float] | None = None) -> int:
    """Filtered record count (all filters optional, parameterized)."""
    clauses: list[str] = []
    params: list = []
    if product:
        clauses.append("product = ?")
        params.append(product)
    if sensor:
        clauses.append("sensor = ?")
        params.append(sensor)
    if start:
        clauses.append("acquisition_timestamp_utc >= ?")
        params.append(start)
    if end:
        clauses.append("acquisition_timestamp_utc < ?")
        params.append(end)
    if bbox:
        w, s, e, n = bbox
        clauses.append("longitude BETWEEN ? AND ? AND latitude BETWEEN ? AND ?")
        params.extend([w, e, s, n])
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    con = _connect(db_path)
    try:
        return con.execute(f"SELECT COUNT(*) FROM detections {where}", params).fetchone()[0]
    finally:
        con.close()


def fetch(db_path: str, *, product: str | None = None, sensor: str | None = None,
          start: str | None = None, end: str | None = None,
          bbox: tuple[float, float, float, float] | None = None,
          limit: int = 1000):
    """Filtered rows as a DataFrame (bounded by *limit*; same filters as count)."""
    clauses: list[str] = []
    params: list = []
    if product:
        clauses.append("product = ?")
        params.append(product)
    if sensor:
        clauses.append("sensor = ?")
        params.append(sensor)
    if start:
        clauses.append("acquisition_timestamp_utc >= ?")
        params.append(start)
    if end:
        clauses.append("acquisition_timestamp_utc < ?")
        params.append(end)
    if bbox:
        w, s, e, n = bbox
        clauses.append("longitude BETWEEN ? AND ? AND latitude BETWEEN ? AND ?")
        params.extend([w, e, s, n])
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    con = _connect(db_path)
    try:
        return con.execute(
            f"SELECT * FROM detections {where} ORDER BY acquisition_timestamp_utc "
            f"LIMIT {int(limit)}", params).df()
    finally:
        con.close()
