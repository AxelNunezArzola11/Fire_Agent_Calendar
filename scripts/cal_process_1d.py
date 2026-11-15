#!/usr/bin/env python3
"""Phase 1D processing runner (reproducible, one-off).

Reads the Phase 1C raw archive (read-only), normalizes both pilot files,
writes hive-partitioned Parquet, rebuilds the DuckDB views, cross-checks
California filtering via two independent paths (pandas regions filter vs
DuckDB bbox query), and records lineage + quality in catalog.json.

Raw files are never modified. Re-runs are idempotent (same bytes skipped).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.calendar.processing import (  # noqa: E402
    catalog, duckdb_layer, normalize, parquet, quality, regions)
from src.calendar.storage import integrity, manifest  # noqa: E402
from src.calendar import settings  # noqa: E402

RAW_ROOT = str(REPO / settings.RAW_ROOT)
PARQUET_ROOT = str(REPO / settings.PARQUET_ROOT)
CATALOG = str(REPO / settings.CATALOG_PATH)
DB_PATH = str(REPO / "calendar.duckdb")
PRODUCT_SENSOR = {"modis": "terra_aqua", "viirs-snpp": "snpp"}


def main() -> dict:
    t_start = time.monotonic()
    report: dict = {"datasets": [], "elapsed_s": 0.0}
    registered = manifest.registered_files(RAW_ROOT)
    if not registered:
        raise SystemExit("BLOCKED: raw archive is empty — run the 1C pilot first")

    for raw_path in sorted(registered):
        t0 = time.monotonic()
        event = registered[raw_path]
        product = event["product"]
        sensor = PRODUCT_SENSOR[product]
        batch = normalize.normalize_csv(
            raw_path, product=product, sensor=sensor,
            source_sha256=event["sha256"],
            processing_version=settings.PROCESSING_VERSION)
        # Duplicate protection: identical detection_ids collapse with a count.
        dupes = int(batch.accepted["detection_id"].duplicated().sum())
        deduped = batch.accepted.drop_duplicates(subset=["detection_id"]).reset_index(drop=True)
        written = parquet.write_dataset(
            deduped, root=PARQUET_ROOT, product=product, sensor=sensor,
            processing_version=settings.PROCESSING_VERSION)
        qual = quality.quality_report(source_rows=batch.source_rows,
                                      accepted=deduped,
                                      rejected=batch.rejected)
        qual["duplicate_detection_ids"] = dupes
        entry = catalog.record_dataset(
            CATALOG, dataset_id=f"{product}-2023-USA",
            raw_files=[{"path": raw_path, "sha256": event["sha256"],
                        "size_bytes": event["size_bytes"]}],
            processing_version=settings.PROCESSING_VERSION,
            parquet_files=written["files"], validation_status="valid",
            record_counts={"source": batch.source_rows, "accepted": len(deduped),
                           "rejected": len(batch.rejected)},
            coverage={"temporal": qual["temporal_coverage"],
                      "geographic": qual["geographic_coverage"]},
            quality=qual)
        report["datasets"].append({
            "dataset_id": f"{product}-2023-USA", "source_rows": batch.source_rows,
            "accepted": len(deduped), "rejected": len(batch.rejected),
            "parquet_files": len(written["files"]),
            "catalog_versions": len(entry["versions"]),
            "seconds": round(time.monotonic() - t0, 1)})

    duckdb_layer.init_db(DB_PATH, PARQUET_ROOT)
    # California cross-check: pandas region filter vs DuckDB bbox query.
    ca = regions.REGIONS["california"]
    counts = {}
    for product, sensor in PRODUCT_SENSOR.items():
        full = parquet.read_dataset(PARQUET_ROOT, product=product,
                                    processing_version=settings.PROCESSING_VERSION)
        _kept, stats = regions.filter_region(full, "california")
        via_duck = duckdb_layer.count(DB_PATH, product=product, bbox=ca.bbox)
        assert stats["kept_rows"] == via_duck, (product, stats, via_duck)
        counts[product] = {"ca_rows": via_duck, "total_rows": len(full)}
    report["california_crosscheck"] = counts
    report["duckdb_datasets"] = duckdb_layer.datasets(DB_PATH)
    report["elapsed_s"] = round(time.monotonic() - t_start, 1)
    out = REPO / "PHASE_1D_RUN.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)[:3000])
    print(f"\nwrote {out}")
    # Prove raw immutability after the whole run.
    print("raw verify_tree:", {k: len(v) for k, v in manifest.verify_tree(RAW_ROOT).items()
                               if k != "total"})
    print("raw sha sample:",
          integrity.sha256_file(sorted(registered)[0]) == registered[sorted(registered)[0]]["sha256"])
    return report


if __name__ == "__main__":
    main()
