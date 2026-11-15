"""Storage Manager (MVP): disk bands, capacity guard, backup, reporting.

Bands: Normal <70%, Warning 70–84%, High 85–94%, Critical ≥95%.
Never deletes anything. Backup = verified copy to a separate directory on
the same server: protects against accidental overwrite, NOT disk failure.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from . import integrity, manifest
from .. import settings

logger = logging.getLogger(__name__)

BAND_NORMAL, BAND_WARNING, BAND_HIGH, BAND_CRITICAL = (
    "Normal", "Warning", "High", "Critical")


class CapacityBlocked(RuntimeError):
    """Raised when a download/operation cannot proceed safely."""


def disk_usage(path: str) -> dict:
    """Current usage for the filesystem holding *path* (bytes + fraction).

    Fraction follows the ``df`` convention (1 - free/total), which counts
    root-reserved blocks as unavailable — the conservative choice for a
    capacity safeguard.
    """
    usage = shutil.disk_usage(path)
    frac = 1.0 - (usage.free / usage.total) if usage.total else 1.0
    return {"path": str(path), "total_bytes": usage.total,
            "used_bytes": usage.used, "free_bytes": usage.free,
            "used_frac": frac, "band": band(frac)}


def band(used_frac: float) -> str:
    """Map a used-fraction to its status band (thresholds from settings)."""
    if used_frac >= settings.STORAGE_CRITICAL:
        return BAND_CRITICAL
    if used_frac >= settings.STORAGE_HIGH:
        return BAND_HIGH
    if used_frac >= settings.STORAGE_WARN:
        return BAND_WARNING
    return BAND_NORMAL


def ensure_capacity(path: str, bytes_needed: int, *, reserve_bytes: int,
                    what: str = "operation") -> dict:
    """Block unless ``free - needed >= reserve``. Returns the usage dict.

    Raises CapacityBlocked with an explicit reason otherwise. Call BEFORE
    every historical acquisition (temp space ≈ 2× file size: staged + raw).
    """
    usage = disk_usage(path)
    free_after = usage["free_bytes"] - bytes_needed
    if free_after < reserve_bytes:
        raise CapacityBlocked(
            f"blocked {what}: need {bytes_needed} bytes + "
            f"{reserve_bytes}-byte reserve, only {usage['free_bytes']} free "
            f"on {usage['path']} (band {usage['band']})")
    logger.info("capacity ok for %s: free=%d need=%d reserve=%d",
                what, usage["free_bytes"], bytes_needed, reserve_bytes)
    return usage


def backup_new_files(raw_root: str, backup_root: str) -> dict:
    """Copy registered files missing (or differing) in backup; re-verify SHAs.

    Idempotent and pull-based. Returns {copied, skipped, failed}.
    """
    report: dict = {"copied": [], "skipped": [], "failed": []}
    for rel_path, event in manifest.registered_files(raw_root).items():
        src = Path(rel_path)
        try:
            target = Path(backup_root) / src.relative_to(raw_root)
        except ValueError:
            target = Path(backup_root) / src.name
        try:
            if target.is_file() and integrity.sha256_file(target) == event.get("sha256"):
                report["skipped"].append(str(target))
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            if integrity.sha256_file(target) != event.get("sha256"):
                target.unlink(missing_ok=True)
                raise IOError("post-copy hash mismatch")
            report["copied"].append(str(target))
        except Exception as exc:  # noqa: BLE001 - report, don't crash the batch
            logger.error("backup failed for %s: %s", src, exc)
            report["failed"].append({"path": str(src), "error": str(exc)[:200]})
    return report


def storage_report(raw_root: str, backup_root: str) -> dict:
    """Bytes per dataset, manifest counts, quarantine, disk band, backup lag."""
    events = manifest.load_events(raw_root)
    counts: dict[str, int] = {}
    dataset_bytes: dict[str, int] = {}
    quarantined: list[str] = []
    for event in events:
        kind = event.get("event", "?")
        counts[kind] = counts.get(kind, 0) + 1
        if kind == "registered":
            key = f"{event.get('product')}/{event.get('sensor')}"
            dataset_bytes[key] = dataset_bytes.get(key, 0) + int(event.get("size_bytes", 0))
        if kind == "quarantined":
            quarantined.append(event.get("path", "?"))
    registered = manifest.registered_files(raw_root)
    missing_backup = 0
    for rel_path in registered:
        try:
            target = Path(backup_root) / Path(rel_path).relative_to(raw_root)
        except ValueError:
            target = Path(backup_root) / Path(rel_path).name
        if not target.is_file():
            missing_backup += 1
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "disk": disk_usage(raw_root),
        "manifest_events": counts,
        "registered_files": len(registered),
        "dataset_bytes": dataset_bytes,
        "quarantined": quarantined,
        "backup_root": str(backup_root),
        "backup_missing_files": missing_backup,
    }
