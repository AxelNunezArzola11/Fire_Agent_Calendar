"""Minimal JSON catalog: raw -> normalized lineage with version history.

``catalog.json`` holds ``{"datasets": [...]}``; each dataset keeps a
``versions`` list (append-only — reprocessing adds an entry, never rewrites
history). Small enough for whole-file read/write with an atomic replace.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _blank() -> dict:
    return {"catalog_version": 1, "datasets": []}


def load_catalog(path: str) -> dict:
    catalog = Path(path)
    if not catalog.is_file():
        return _blank()
    return json.loads(catalog.read_text(encoding="utf-8"))


def _save(catalog_path: Path, data: dict) -> None:
    tmp = catalog_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(catalog_path)  # atomic on POSIX


def record_dataset(catalog_path: str, *, dataset_id: str, raw_files: list[dict],
                   processing_version: str, parquet_files: list[str],
                   validation_status: str, record_counts: dict,
                   coverage: dict, quality: dict) -> dict:
    """Append a processing version for *dataset_id* (history preserved)."""
    catalog_file = Path(catalog_path)
    catalog_file.parent.mkdir(parents=True, exist_ok=True)
    data = load_catalog(str(catalog_file))
    entry = next((d for d in data["datasets"] if d["dataset_id"] == dataset_id), None)
    if entry is None:
        entry = {"dataset_id": dataset_id, "versions": []}
        data["datasets"].append(entry)
    entry["versions"].append({
        "processing_version": processing_version,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "raw_files": raw_files,  # [{path, sha256, size_bytes}]
        "parquet_files": sorted(parquet_files),
        "validation_status": validation_status,
        "record_counts": record_counts,
        "coverage": coverage,
        "quality": quality,
    })
    _save(catalog_file, data)
    logger.info("catalog: %s now has %d version(s)", dataset_id, len(entry["versions"]))
    return entry


def latest_version(catalog_path: str, dataset_id: str) -> dict | None:
    """Newest processing version for a dataset, or None if unknown."""
    data = load_catalog(catalog_path)
    entry = next((d for d in data["datasets"] if d["dataset_id"] == dataset_id), None)
    if not entry or not entry["versions"]:
        return None
    return entry["versions"][-1]
