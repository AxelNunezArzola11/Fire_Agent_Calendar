"""FIRMS CSV ingestion + canonical normalization (v1).

Reads NRT (14-col) and Standard-archive (15-col, extra ``type``) CSVs.
Required fields: latitude, longitude, acq_date, acq_time, satellite,
instrument. Everything else is preserved when present, never invented.

``detection_id`` = sha256 hex (32 chars) of
``product|satellite|instrument|acq_date|acq_time|latitude|longitude|source_sha256``
using the source's exact strings (no rounding). Limitation: truly identical
source rows (same second, same coords, same satellite, same file) share an
ID — flagged as duplicates in the quality report rather than silently kept.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED = ("latitude", "longitude", "acq_date", "acq_time", "satellite", "instrument")

CANONICAL = (
    "detection_id", "sensor", "satellite", "instrument", "product",
    "product_version", "acquisition_timestamp_utc", "original_acquisition_timestamp",
    "latitude", "longitude", "confidence", "frp", "daynight", "scan", "track",
    "source_file", "source_sha256", "ingestion_timestamp_utc", "processing_version",
)


@dataclass
class NormalizedBatch:
    accepted: pd.DataFrame  # canonical + preserved source columns
    rejected: list[dict]  # {row_index, reason}
    source_rows: int


def _parse_utc(acq_date: str, acq_time: str) -> str:
    """'2023-01-05' + '0130' -> '2023-01-05T01:30:00+00:00'. Raises ValueError."""
    stamp = f"{str(acq_date).strip()}{str(acq_time).strip().zfill(4)}"
    dt = datetime.strptime(stamp, "%Y-%m-%d%H%M").replace(tzinfo=timezone.utc)
    return dt.isoformat()


def detection_id(product: str, row: pd.Series, source_sha256: str) -> str:
    canonical = "|".join([
        product, str(row["satellite"]), str(row["instrument"]),
        str(row["acq_date"]), str(row["acq_time"]),
        str(row["latitude"]), str(row["longitude"]), source_sha256])
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


def normalize_frame(raw: pd.DataFrame, *, product: str, sensor: str,
                    source_file: str, source_sha256: str,
                    processing_version: str) -> NormalizedBatch:
    """Normalize a raw FIRMS frame. Pure function (no I/O)."""
    missing_cols = [c for c in REQUIRED if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"missing required columns: {missing_cols}")

    ingested_at = datetime.now(timezone.utc).isoformat()
    accepted_rows: list[dict] = []
    rejected: list[dict] = []

    for idx, row in raw.iterrows():
        try:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError(f"coords out of range ({lat}, {lon})")
            if pd.isna(row["satellite"]) or pd.isna(row["instrument"]):
                raise ValueError("missing satellite/instrument")
            utc = _parse_utc(row["acq_date"], row["acq_time"])
        except (ValueError, TypeError) as exc:
            rejected.append({"row_index": int(idx), "reason": str(exc)[:200]})
            continue

        def _num(value) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return float("nan")  # preserved as null; counted in quality

        record = {
            "detection_id": detection_id(product, row, source_sha256),
            "sensor": sensor,
            "satellite": str(row["satellite"]),
            "instrument": str(row["instrument"]),
            "product": product,
            "product_version": ("" if pd.isna(row.get("version")) else str(row.get("version"))),
            "acquisition_timestamp_utc": utc,
            "original_acquisition_timestamp": f"{row['acq_date']} {row['acq_time']}",
            "latitude": lat,
            "longitude": lon,
            "confidence": ("" if pd.isna(row.get("confidence")) else str(row.get("confidence"))),
            "frp": _num(row.get("frp")),
            "daynight": ("" if pd.isna(row.get("daynight")) else str(row.get("daynight"))),
            "scan": _num(row.get("scan")),
            "track": _num(row.get("track")),
            "source_file": source_file,
            "source_sha256": source_sha256,
            "ingestion_timestamp_utc": ingested_at,
            "processing_version": processing_version,
        }
        # Preserve every extra source column verbatim (brightness variants,
        # bright_t31/ti4/ti5, type, ...). Never overwrites canonical fields.
        for col in raw.columns:
            if col not in record and col not in (
                    "acq_date", "acq_time", "version", "latitude", "longitude",
                    "confidence", "frp", "daynight", "scan", "track",
                    "satellite", "instrument"):
                record[f"src_{col}"] = row[col]
        accepted_rows.append(record)

    accepted = pd.DataFrame(accepted_rows)
    logger.info("normalize %s: %d accepted, %d rejected of %d",
                source_file, len(accepted), len(rejected), len(raw))
    return NormalizedBatch(accepted, rejected, len(raw))


def normalize_csv(path: str, **kwargs) -> NormalizedBatch:
    """Read a FIRMS CSV from disk and normalize it (read-only on the file)."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=True)
    return normalize_frame(raw, source_file=path, **kwargs)
