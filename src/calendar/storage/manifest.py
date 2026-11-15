"""Append-only acquisition manifest + raw-archive registration.

Every stored raw file has exactly one ``registered`` event carrying its
SHA-256, size, product, sensor, batch, provenance, and timestamps.
Re-registration of identical bytes is idempotent (returns ``duplicate_of``);
different bytes under an occupied path are refused, never overwritten.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import integrity, layout

logger = logging.getLogger(__name__)


class RegistrationError(RuntimeError):
    """Refused registration (occupied path, hash mismatch, bad input)."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Registration:
    path: str  # repo-relative raw path
    sha256: str
    size_bytes: int
    product: str
    sensor: str
    batch_id: str
    filename: str
    status: str  # 'registered' | 'duplicate_of'
    duplicate_of: str | None = None
    provenance: dict = field(default_factory=dict)
    registered_at: str = ""


def _append_event(raw_root: str, event: dict) -> None:
    manifest = layout.manifest_path(raw_root)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    event = {"ts": _utcnow(), **event}
    with open(manifest, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def load_events(raw_root: str) -> list[dict]:
    """Read all manifest events (empty list if no manifest yet)."""
    manifest = layout.manifest_path(raw_root)
    if not manifest.is_file():
        return []
    events = []
    with open(manifest, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def registered_files(raw_root: str) -> dict[str, dict]:
    """path -> latest registration record (replays the append-only log)."""
    files: dict[str, dict] = {}
    for event in load_events(raw_root):
        if event.get("event") in ("registered", "duplicate_of"):
            files[event["path"]] = event
    return files


def find_by_sha(raw_root: str, sha256: str) -> list[str]:
    """All registered paths holding these exact bytes (duplicate detection)."""
    return [p for p, e in registered_files(raw_root).items()
            if e.get("sha256") == sha256]


def register_staged(staged_path: str, *, raw_root: str, product: str, sensor: str,
                    batch_id: str, provenance: dict | None = None,
                    expected_sha256: str | None = None) -> Registration:
    """Commit a staged file into the immutable raw archive.

    Steps: hash staged bytes -> refuse occupied-different -> copy ->
    re-hash copy (must match) -> sidecar -> manifest event.
    Idempotent: identical bytes already registered -> ``duplicate_of``.
    """
    src = Path(staged_path)
    if not src.is_file():
        raise RegistrationError(f"nothing to register: {staged_path}")
    digest = integrity.sha256_file(src)
    if expected_sha256 and digest != expected_sha256:
        raise RegistrationError(
            f"staged file hash {digest[:16]}… != expected {expected_sha256[:16]}…")

    dest = layout.raw_file_path(raw_root, product, sensor, batch_id, src.name)
    if dest.exists():
        existing = integrity.sha256_file(dest)
        if existing == digest:
            rel = str(dest)
            logger.info("duplicate registration (idempotent): %s", rel)
            _append_event(raw_root, {"event": "duplicate_of", "path": rel,
                                     "sha256": digest, "size_bytes": dest.stat().st_size,
                                     "product": product, "sensor": sensor,
                                     "batch_id": batch_id, "filename": src.name})
            return Registration(rel, digest, dest.stat().st_size, product, sensor,
                                batch_id, src.name, "duplicate_of", duplicate_of=rel,
                                provenance=provenance or {}, registered_at=_utcnow())
        raise RegistrationError(
            f"refusing to overwrite {dest} (stored sha {existing[:16]}… != {digest[:16]}…)")

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    if integrity.sha256_file(dest) != digest:  # copy verification
        dest.unlink(missing_ok=True)
        raise RegistrationError(f"copy verification failed for {dest}")
    integrity.write_sidecar(dest)
    rel = str(dest)
    size = dest.stat().st_size
    _append_event(raw_root, {"event": "registered", "path": rel, "sha256": digest,
                             "size_bytes": size, "product": product, "sensor": sensor,
                             "batch_id": batch_id, "filename": src.name,
                             "provenance": provenance or {}})
    logger.info("registered %s (%d bytes, sha256=%s…)", rel, size, digest[:16])
    return Registration(rel, digest, size, product, sensor, batch_id, src.name,
                        "registered", provenance=provenance or {}, registered_at=_utcnow())


def quarantine_file(staged_path: str, *, raw_root: str, quarantine_dir: str,
                    reasons: list[str]) -> str:
    """Preserve a failed file under quarantine/ (never registered as trusted)."""
    src = Path(staged_path)
    if not src.is_file():
        raise RegistrationError(f"nothing to quarantine: {staged_path}")
    dest = layout.quarantine_path(raw_root, quarantine_dir, src.name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise RegistrationError(f"quarantine slot occupied: {dest}")
    shutil.copy2(src, dest)
    digest = integrity.sha256_file(dest)
    _append_event(raw_root, {"event": "quarantined", "path": str(dest),
                             "sha256": digest, "size_bytes": dest.stat().st_size,
                             "filename": src.name, "reasons": reasons})
    logger.warning("quarantined %s: %s", dest, "; ".join(reasons))
    return str(dest)


def verify_tree(raw_root: str) -> dict:
    """Re-hash every registered file. Returns {ok, mismatch, missing, total}."""
    files = registered_files(raw_root)
    report: dict = {"ok": [], "mismatch": [], "missing": [], "total": len(files)}
    for path, event in files.items():
        target = Path(path)
        if not target.is_file():
            report["missing"].append(path)
        elif integrity.sha256_file(target) != event.get("sha256"):
            report["mismatch"].append(path)
        else:
            report["ok"].append(path)
    return report
