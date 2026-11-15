"""Idempotent Parquet writer (hive partitions, deterministic bytes).

Layout: ``<root>/<processing_version>/product=<p>/sensor=<s>/year=<y>/month=<m>/
part-<content_sha8>.parquet`` — one file per month-shard at pilot scale
(no tiny-file sprawl: 12 files/year/dataset). Re-running identical input
reproduces identical bytes and skips the write (sha match).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

COMPRESSION = "snappy"  # fast + splittable; zstd optional later


def _content_sha(df: pd.DataFrame) -> str:
    canonical = df.sort_values("detection_id").to_csv(index=False).encode()
    return hashlib.sha256(canonical).hexdigest()[:8]


def _month_shards(df: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    stamps = pd.to_datetime(df["acquisition_timestamp_utc"], utc=True)
    shards: dict[tuple[str, str], pd.DataFrame] = {}
    for (year, month), group in df.groupby([stamps.dt.strftime("%Y"), stamps.dt.strftime("%m")]):
        shards[(year, month)] = group.sort_values("detection_id").reset_index(drop=True)
    return shards


def write_dataset(df: pd.DataFrame, *, root: str, product: str, sensor: str,
                  processing_version: str) -> dict:
    """Write month-sharded Parquet idempotently. Returns {written, skipped, files}."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    report: dict = {"written": [], "skipped": [], "files": []}
    if df.empty:
        logger.warning("write_dataset: empty frame for %s/%s — nothing written", product, sensor)
        return report
    # Partition keys live in the directory path only; storing them again as
    # data columns breaks hive-partitioned reads (Arrow type-merge conflict).
    # Readers (pyarrow hive / DuckDB hive_partitioning) restore them from paths.
    store = df.drop(columns=["product", "sensor"], errors="ignore")
    for (year, month), shard in _month_shards(store).items():
        tag = _content_sha(shard)
        out_dir = (Path(root) / processing_version / f"product={product}"
                   / f"sensor={sensor}" / f"year={year}" / f"month={month}")
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(out_dir.glob("part-*.parquet"))
        if len(existing) == 1 and existing[0].stem == f"part-{tag}":
            report["skipped"].append(str(existing[0]))
            report["files"].append(str(existing[0]))
            continue
        for stale in existing:  # same shard, different content -> superseded version
            stale.unlink()
        out = out_dir / f"part-{tag}.parquet"
        table = pa.Table.from_pandas(shard, preserve_index=False)
        pq.write_table(table, out, compression=COMPRESSION)
        report["written"].append(str(out))
        report["files"].append(str(out))
        logger.info("wrote %s (%d rows)", out, len(shard))
    return report


def read_dataset(root: str, *, product: str | None = None, sensor: str | None = None,
                 processing_version: str | None = None) -> pd.DataFrame:
    """Read (a slice of) the normalized store with pandas/pyarrow."""
    import pyarrow.parquet as pq

    base = Path(root)
    if processing_version:
        base = base / processing_version
    filters = []
    if product:
        filters.append(("product", "=", product))
    if sensor:
        filters.append(("sensor", "=", sensor))
    dataset = pq.ParquetDataset(base, filters=filters or None)
    return dataset.read().to_pandas()
