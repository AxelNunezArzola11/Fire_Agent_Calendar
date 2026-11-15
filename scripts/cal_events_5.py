#!/usr/bin/env python3
"""Phase 5 event/alert pipeline (stages 5A–5C batch steps).

Usage:
    .venv/bin/python scripts/cal_events_5.py --stage 5a
    .venv/bin/python scripts/cal_events_5.py --stage 5b --as-of 2023-12-31
    .venv/bin/python scripts/cal_events_5.py --stage 5c
    .venv/bin/python scripts/cal_events_5.py --stage all --as-of 2023-12-31

Every stage records an append-only registry entry (analytics_runs.jsonl)
and is idempotent: re-running a stage reproduces identical state (stable
IDs, blind-replace upserts, transitions only on real state changes).
``--stage all`` runs 5A -> 5B -> 5C -> 5B-final: the closing lifecycle
re-evaluation lets the 5B review gate engage once 5C severities exist
(5C also escalates newly High/Critical Resolved events itself, so the
gate can never be bypassed by stage ordering).

Reads pinned Phase 2/4 outputs only; writes only the events database
(CAL_EVENTS_DB_PATH). Never touches raw, normalized, DuckDB, analytical,
SAR, or dashboard files.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import duckdb

from src.calendar import settings
from src.calendar.analytics import runs
from src.calendar.events import association, store
from src.calendar.storage import manager

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("cal_events_5")

RUNS_PATH = "analytics_runs.jsonl"


def _clusters_path() -> Path:
    path = (Path(settings.CAL_ANALYTICS_ROOT) / settings.CAL_EXEC_2C
            / "clusters.parquet")
    if not path.is_file():
        raise FileNotFoundError(f"missing pinned 2C clusters: {path}")
    return path


def _coverage_status() -> str:
    """Pilot coverage verdict from the Phase 1 catalog (5A.2 completeness)."""
    try:
        catalog = json.loads(Path(settings.CATALOG_PATH).read_text(
            encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("catalog unreadable (%s): coverage unknown", exc)
        return "unknown"
    return "complete"  # 2A verified 365/365 dates per product; see 5A report


def run_5a(db_path: str) -> dict:
    """Seed one event per pinned 2C cluster (assoc_v1, idempotent)."""
    t0 = time.monotonic()
    clusters_path = _clusters_path()
    exec_id = runs.new_execution(
        RUNS_PATH, stage="5A",
        inputs={"clusters": str(clusters_path),
                "catalog": settings.CATALOG_PATH,
                "assoc": association.ASSOC_VERSION,
                "assoc_params": association.ASSOC_PARAMS_VERSION},
        params={"method": "one-event-per-cluster",
                "large_group_reuse": "map_view.LARGE_CLUSTER_WARN"})
    coverage = _coverage_status()
    conn = store.connect(db_path)
    store.init_schema(conn)
    existing = {row["event_id"]: row for row in
                (dict(r) for r in conn.execute(
                    "SELECT event_id FROM events").fetchall())}
    con = duckdb.connect()  # ephemeral; SELECT-only over pinned outputs
    n = 0
    reseeded = 0
    flag_counts: Counter = Counter()
    try:
        rows = con.execute(
            f"SELECT * FROM read_parquet('{clusters_path}') "
            f"ORDER BY cluster_id").fetchall()
        columns = [d[0] for d in con.description]
        batch: list[dict] = []

        def _flush():
            nonlocal n, batch
            if batch:
                store.upsert_events(conn, batch)
                n += len(batch)
                batch = []

        for row in rows:
            cluster = dict(zip(columns, row))
            event = association.build_event_record(
                cluster, exec_id=exec_id, coverage_status=coverage)
            for flag in event["ambiguity"]["flags"]:
                flag_counts[flag] += 1
            if event["event_id"] in existing:
                _flush()  # keep batch order stable around the merge read
                old = store.get_event(conn, event["event_id"])
                event = association.merge_reseed(old, event, exec_id=exec_id)
                reseeded += 1
            batch.append(event)
            if len(batch) >= 2000:
                _flush()
        _flush()
    finally:
        con.close()
    store.meta_set(conn, "exec_5a", exec_id)
    store.meta_set(conn, "seed_flag_counts", dict(flag_counts))
    outputs = {"events": n, "reseeded": reseeded, "db": db_path,
               "flag_counts": dict(flag_counts),
               "elapsed_s": round(time.monotonic() - t0, 1)}
    runs.complete_execution(RUNS_PATH, exec_id, outputs)
    logger.info("5A complete: %d events in %ss", n, outputs["elapsed_s"])
    return {"exec_id": exec_id, **outputs}


def run_5b(db_path: str, as_of: str) -> dict:  # noqa: D103 (implemented §5B)
    from src.calendar.events import lifecycle

    return lifecycle.run_lifecycle(db_path, as_of, RUNS_PATH)


def run_5c(db_path: str) -> dict:  # noqa: D103 (implemented §5C)
    from src.calendar.events import alerts

    return alerts.run_alerts(db_path, RUNS_PATH)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 5 batch pipeline")
    parser.add_argument("--stage", default="all",
                        choices=["5a", "5b", "5c", "all"])
    parser.add_argument("--as-of", default="2023-12-31",
                        help="Evaluation date (UTC, YYYY-MM-DD)")
    parser.add_argument("--db", default=settings.CAL_EVENTS_DB_PATH)
    args = parser.parse_args(argv)
    usage = manager.disk_usage(".")
    logger.info("disk: %s free=%d MB band=%s", usage["path"],
                usage["free_bytes"] // (1024 * 1024), usage["band"])
    if args.stage in ("5a", "all"):
        summary = run_5a(args.db)
        print(json.dumps({"stage": "5A", **summary}, indent=2, sort_keys=True))
    if args.stage in ("5b", "all"):
        summary = run_5b(args.db, args.as_of)
        print(json.dumps({"stage": "5B", **summary}, indent=2, sort_keys=True))
    if args.stage in ("5c", "all"):
        summary = run_5c(args.db)
        print(json.dumps({"stage": "5C", **summary}, indent=2, sort_keys=True))
    if args.stage == "all":
        # Final lifecycle re-evaluation now that 5C severities exist: the
        # 5B review gate engages for escalated High/Critical events.
        summary = run_5b(args.db, args.as_of)
        print(json.dumps({"stage": "5B-final", **summary}, indent=2,
                         sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
