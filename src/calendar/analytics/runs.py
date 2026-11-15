"""Analytical execution registry (append-only JSONL).

Every analytical execution — matching, clustering, metrics, validation —
is recorded with its inputs, params, region, date range, and outputs, so
any result can be traced and reproduced. History is never rewritten.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_RUNS_PATH = "analytics_runs.jsonl"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_execution(runs_path: str, *, stage: str, inputs: dict,
                  params: dict | None = None, region: str | None = None,
                  date_range: tuple[str, str] | None = None) -> str:
    """Open an execution record. Returns the execution id (uuid4 hex)."""
    exec_id = uuid.uuid4().hex
    record = {
        "exec_id": exec_id,
        "stage": stage,
        "started_at": _utcnow(),
        "status": "running",
        "inputs": inputs,
        "params": params or {},
        "region": region,
        "date_range": list(date_range) if date_range else None,
        "outputs": {},
    }
    target = Path(runs_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    logger.info("execution %s opened (%s)", exec_id[:8], stage)
    return exec_id


def complete_execution(runs_path: str, exec_id: str, outputs: dict,
                       *, status: str = "complete") -> None:
    """Append the completion record (original entry is never modified)."""
    record = {"exec_id": exec_id, "status": status,
              "completed_at": _utcnow(), "outputs": outputs}
    with open(runs_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    logger.info("execution %s %s", exec_id[:8], status)


def load_runs(runs_path: str) -> list[dict]:
    """All execution records in append order (empty list if none)."""
    target = Path(runs_path)
    if not target.is_file():
        return []
    records = []
    with open(target, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_run(runs_path: str, exec_id: str) -> list[dict]:
    """All records (open + completion) for one execution id."""
    return [r for r in load_runs(runs_path) if r.get("exec_id") == exec_id]
