"""Raw-archive layout builders.

``<raw_root>/<product>/<sensor>/<batch_id>/<original_filename>``
plus ``<raw_root>/manifest.jsonl`` and ``<raw_root>/quarantine/``.

``batch_id`` is the archive request id, NRT poll id, or manual ``spec_id`` —
always traceable, never silently reused for different bytes (see manifest).
"""

from __future__ import annotations

from pathlib import Path

MANIFEST_NAME = "manifest.jsonl"


def raw_file_path(raw_root: str, product: str, sensor: str, batch_id: str,
                  filename: str) -> Path:
    """Destination path for a raw file. Rejects path-traversal segments."""
    for segment in (product, sensor, batch_id, filename):
        if not segment or segment in (".", "..") or "/" in segment or "\\" in segment:
            raise ValueError(f"invalid path segment: {segment!r}")
    return Path(raw_root) / product / sensor / batch_id / filename


def manifest_path(raw_root: str) -> Path:
    return Path(raw_root) / MANIFEST_NAME


def quarantine_path(raw_root: str, quarantine_dir: str, filename: str) -> Path:
    if not filename or filename in (".", "..") or "/" in filename or "\\" in filename:
        raise ValueError(f"invalid filename: {filename!r}")
    return Path(raw_root) / quarantine_dir / filename


def sha_sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")
