#!/usr/bin/env python3
"""Phase 1C historical pilot runner (reproducible, one-off).

YEAR SELECTION (documented before acquisition, per brief §5.1):
  Year 2023 — latest year with both open archives verified published AND
  stable (VIIRS file last-modified Jun 2025; the 2025 MODIS file was still
  mutating in Oct 2026). Full-year Standard coverage for MODIS + S-NPP.
  Expected size ~40 MB total: safe on the measured 35 GB free.

Scope: USA per-country Standard CSVs (complete original files; California
filtering happens in the analytical layer, Phase 1D — never on raw files).
  - modis / 2023 / United_States  (~7.6 MB)
  - viirs-snpp / 2023 / United_States  (~31 MB)

Flow per file: capacity guard (HEAD size) -> staged download (byte cap) ->
register into data_raw_firms/ -> backup -> validation report (JSON Lines).

Recorded unavailable via this path: NOAA-20 / NOAA-21 per-country archives
(not published), Terra-only / Aqua-only splits (shared MODIS file).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.calendar.acquisition import archive  # noqa: E402
from src.calendar.storage import manager, manifest  # noqa: E402

YEAR = 2023
COUNTRY = "United_States"
FILES = [("modis", "terra_aqua"), ("viirs-snpp", "snpp")]
STAGING = "/tmp/cal_1c_staging"
RAW_ROOT = str(REPO / "data_raw_firms")
BACKUP_ROOT = str(REPO / "backup_mirror")
RESERVE_BYTES = 2 * 1024**3  # 2 GB floor: pilot must leave headroom


def main() -> dict:
    t_start = time.monotonic()
    report: dict = {"year": YEAR, "country": COUNTRY, "files": [], "capacity": {}}
    total_expected = 0
    probes = {}
    for slug, _sensor in FILES:
        probe = archive.probe_open_archive(slug, YEAR, COUNTRY)
        probes[slug] = {"published": probe.published, "size": probe.size_bytes,
                        "mtime": probe.last_modified, "etag": probe.etag}
        if not probe.published or probe.size_bytes is None:
            raise SystemExit(f"BLOCKED: {slug}/{YEAR} unpublished or unsized: {probe}")
        total_expected += probe.size_bytes
    report["probes"] = probes
    # Capacity guard BEFORE any download (temp ~2x: staged + raw copy).
    usage = manager.ensure_capacity(str(REPO), total_expected * 2,
                                    reserve_bytes=RESERVE_BYTES,
                                    what=f"1C pilot ({total_expected} bytes)")
    report["capacity"] = {"free_bytes": usage["free_bytes"], "band": usage["band"],
                          "needed_bytes": total_expected * 2, "reserve_bytes": RESERVE_BYTES}

    for slug, sensor in FILES:
        t0 = time.monotonic()
        staged = archive.fetch_open_archive(
            slug, YEAR, COUNTRY, STAGING, max_bytes=probes[slug]["size"] * 2)
        reg = manifest.register_staged(
            staged.path, raw_root=RAW_ROOT, product=slug, sensor=sensor,
            batch_id=f"open-archive-{YEAR}-{COUNTRY}",
            provenance={**staged.provenance, "stage": "1C",
                        "scope_note": "USA per-country file; CA filtering is analytical (1D)"},
            expected_sha256=staged.sha256)
        report["files"].append({
            "filename": staged.filename, "product": slug, "sensor": sensor,
            "size_bytes": staged.size_bytes, "sha256": staged.sha256,
            "raw_path": reg.path, "status": reg.status,
            "seconds": round(time.monotonic() - t0, 1)})

    backup = manager.backup_new_files(RAW_ROOT, BACKUP_ROOT)
    report["backup"] = {k: (len(v) if isinstance(v, list) else v) for k, v in backup.items()}
    report["elapsed_s"] = round(time.monotonic() - t_start, 1)
    report["free_bytes_after"] = manager.disk_usage(str(REPO))["free_bytes"]
    out = REPO / "PHASE_1C_PILOT_RUN.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return report


if __name__ == "__main__":
    main()
