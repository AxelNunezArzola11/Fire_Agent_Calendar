"""Acquisition manifest: append-only JSONL + resume state.

Each record tracks one granule from inventory through verified
download. Re-runs skip completed entries and resume partial files
via HTTP Range; nothing is re-downloaded silently.
"""

from __future__ import annotations

import json
from pathlib import Path

STATUSES = ("inventoried", "downloading", "complete", "failed",
            "skipped-budget")


def manifest_path(root: str | Path) -> Path:
    return Path(root) / "l2_manifest.jsonl"


def append(root: str | Path, record: dict) -> None:
    """Append one record (creates parent dirs; never rewrites).

    Guards resume state: a terminal 'complete' record is never
    regressed to 'inventoried'/'downloading' by a fresh inventory
    pass (re-downloads record 'failed' or a new 'complete').
    """
    key = record.get("granule_ur") or ""
    if key and record.get("status") in ("inventoried", "downloading"):
        latest = latest_by_ur(root).get(key) or {}
        if latest.get("status") == "complete":
            return
    path = manifest_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str)
                     + "\n")


def load(root: str | Path) -> list[dict]:
    """All records in order (empty list when no manifest exists)."""
    path = manifest_path(root)
    if not path.is_file():
        return []
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def latest_by_ur(root: str | Path) -> dict:
    """Newest record per granule UR (resume view)."""
    view: dict[str, dict] = {}
    for record in load(root):
        key = record.get("granule_ur") or ""
        if key:
            view[key] = record
    return view


def pending_urs(records: list[dict]) -> list[str]:
    """URs still needing download (not complete, not budget-skipped)."""
    view: dict[str, dict] = {}
    for record in records:
        key = record.get("granule_ur") or ""
        if key:
            view[key] = record
    return sorted(ur for ur, record in view.items()
                  if record.get("status") not in ("complete",
                                                  "skipped-budget"))
