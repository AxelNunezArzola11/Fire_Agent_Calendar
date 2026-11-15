"""Stage 5A: event/alert/notification persistence (event_store_v1).

One SQLite database (stdlib sqlite3; default ``calendar_events.sqlite``).
No PostgreSQL, PostGIS, broker, or distributed stack — the MVP stays
local-first like the rest of FireAgent Calendar.

Tables: events, transitions (audit), reviews, alerts, alert_history,
subscriptions, notifications, qwen_log, meta. All writes are transactional
(``with conn:`` blocks); event/alert upserts are idempotent; the
transitions/alert_history tables are append-only audit logs. Raw NASA
FIRMS data, normalized Parquet, DuckDB, and Phase 2/4 outputs are never
written here — this store only references them.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .model import (STORE_VERSION, check_alert_type, check_severity,
                    check_state)

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    event_id            TEXT PRIMARY KEY,
    model_version       TEXT NOT NULL,
    assoc_version       TEXT NOT NULL,
    assoc_params_version TEXT NOT NULL,
    first_observed_utc  TEXT NOT NULL,
    last_observed_utc   TEXT NOT NULL,
    last_evaluated_utc  TEXT,
    lifecycle_state     TEXT NOT NULL,
    severity_name       TEXT NOT NULL,
    severity_level      INTEGER NOT NULL,
    severity_provenance TEXT NOT NULL,
    sensor_modis_n      INTEGER NOT NULL,
    sensor_viirs_n      INTEGER NOT NULL,
    satellites          TEXT NOT NULL,
    cluster_ids         TEXT NOT NULL,
    anomaly_refs        TEXT NOT NULL,
    detection_count     INTEGER NOT NULL,
    detection_refs      TEXT,
    detection_resolution TEXT NOT NULL,
    activity            TEXT NOT NULL,
    frp                 TEXT NOT NULL,
    bbox                TEXT NOT NULL,
    centroid            TEXT NOT NULL,
    completeness        TEXT NOT NULL,
    ambiguity           TEXT NOT NULL,
    review_required     INTEGER NOT NULL DEFAULT 0,
    review_status       TEXT NOT NULL DEFAULT 'none',
    provenance          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_state ON events (lifecycle_state);
CREATE INDEX IF NOT EXISTS idx_events_severity ON events (severity_level);
CREATE INDEX IF NOT EXISTS idx_events_last_obs ON events (last_observed_utc);
CREATE TABLE IF NOT EXISTS transitions (
    trans_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id        TEXT NOT NULL,
    prev_state      TEXT NOT NULL,
    new_state       TEXT NOT NULL,
    at_utc          TEXT NOT NULL,
    reason          TEXT NOT NULL,
    evidence        TEXT NOT NULL,
    completeness    TEXT NOT NULL,
    algo_version    TEXT NOT NULL,
    config_version  TEXT NOT NULL,
    review_required INTEGER NOT NULL DEFAULT 0,
    review_performed INTEGER NOT NULL DEFAULT 0,
    exec_id         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trans_event ON transitions (event_id);
CREATE TABLE IF NOT EXISTS reviews (
    review_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id       TEXT NOT NULL,
    reviewer_label TEXT NOT NULL,
    decision       TEXT NOT NULL,
    note           TEXT NOT NULL DEFAULT '',
    at_utc         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reviews_event ON reviews (event_id);
CREATE TABLE IF NOT EXISTS alerts (
    alert_id       TEXT PRIMARY KEY,
    alert_type     TEXT NOT NULL,
    severity_name  TEXT NOT NULL,
    severity_level INTEGER NOT NULL,
    event_id       TEXT,
    scope          TEXT NOT NULL,
    metric         TEXT NOT NULL,
    grain          TEXT NOT NULL DEFAULT '',
    period         TEXT NOT NULL DEFAULT '',
    trigger        TEXT NOT NULL,
    baseline_ref   TEXT NOT NULL,
    data_quality   TEXT NOT NULL,
    explanation    TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    rule_version   TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'open',
    ack_at         TEXT,
    ack_note       TEXT NOT NULL DEFAULT '',
    resolved_at    TEXT,
    resolve_note   TEXT NOT NULL DEFAULT '',
    fingerprint    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alerts_event ON alerts (event_id);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts (status);
CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts (alert_type);
CREATE TABLE IF NOT EXISTS alert_history (
    hist_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id TEXT NOT NULL,
    at_utc   TEXT NOT NULL,
    action   TEXT NOT NULL,
    detail   TEXT NOT NULL,
    exec_id  TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_ahist_alert ON alert_history (alert_id);
CREATE TABLE IF NOT EXISTS subscriptions (
    sub_id              TEXT PRIMARY KEY,
    email               TEXT NOT NULL,
    status              TEXT NOT NULL,
    confirm_token_hash  TEXT,
    confirm_expires_utc TEXT,
    manage_token_hash   TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    confirmed_at        TEXT,
    unsubscribed_at     TEXT,
    prefs               TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS notifications (
    notif_id    TEXT PRIMARY KEY,
    alert_id    TEXT NOT NULL,
    sub_id      TEXT NOT NULL,
    kind        TEXT NOT NULL,
    status      TEXT NOT NULL,
    attempts    INTEGER NOT NULL DEFAULT 0,
    last_error  TEXT,
    created_at  TEXT NOT NULL,
    sent_at     TEXT,
    fingerprint TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_notif_alert_sub ON notifications (alert_id, sub_id);
CREATE TABLE IF NOT EXISTS qwen_log (
    log_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    at_utc      TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    status      TEXT NOT NULL,
    latency_ms  REAL,
    model       TEXT NOT NULL DEFAULT '',
    error       TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_qwen_target ON qwen_log (target_type, target_id);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS custom_rules (
    rule_id         TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    metric          TEXT NOT NULL,
    scope           TEXT NOT NULL,
    grain           TEXT NOT NULL,
    period          TEXT NOT NULL,
    geo_scope       TEXT NOT NULL,
    operator        TEXT,
    threshold       REAL,
    anomaly_class   TEXT,
    severity_name   TEXT NOT NULL,
    severity_level  INTEGER NOT NULL,
    notify          INTEGER NOT NULL DEFAULT 0,
    enabled         INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    version         TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS custom_rule_evals (
    eval_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_id   TEXT NOT NULL,
    at_utc    TEXT NOT NULL,
    triggered INTEGER NOT NULL,
    reason    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_creval_rule ON custom_rule_evals (rule_id);
CREATE TABLE IF NOT EXISTS custom_hits (
    hit_id        TEXT PRIMARY KEY,
    rule_id       TEXT NOT NULL,
    rule_name     TEXT NOT NULL,
    severity_name TEXT NOT NULL,
    period        TEXT NOT NULL,
    metric        TEXT NOT NULL,
    scope         TEXT NOT NULL,
    summary       TEXT NOT NULL,
    trigger_refs  TEXT NOT NULL,
    triggered_at  TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'open',
    version       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chit_rule ON custom_hits (rule_id);
CREATE TABLE IF NOT EXISTS custom_notifications (
    notif_id    TEXT PRIMARY KEY,
    hit_id      TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'queued',
    attempts    INTEGER NOT NULL DEFAULT 0,
    last_error  TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    sent_at     TEXT
);
"""

_JSON_COLUMNS = {
    "events": ("satellites", "cluster_ids", "anomaly_refs", "detection_refs",
               "detection_resolution", "activity", "frp", "bbox", "centroid",
               "completeness", "ambiguity", "provenance"),
    "transitions": ("evidence", "completeness"),
    "alerts": ("trigger", "baseline_ref", "data_quality"),
    "alert_history": ("detail",),
    "subscriptions": ("prefs",),
    "custom_hits": ("trigger_refs",),
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextlib.contextmanager
def transaction(conn: sqlite3.Connection):
    """Join the ambient transaction, or open one when there is none.

    Batch pipelines wrap a whole page in ``batch(conn)`` so thousands of
    single-row helpers commit once; standalone calls behave exactly as
    before (one commit per call). Plain ``with conn:`` does NOT group:
    ``with`` only commits on exit while BEGIN is lazy, so use batch().
    """
    if conn.in_transaction:
        yield
    else:
        with conn:
            yield


@contextlib.contextmanager
def batch(conn: sqlite3.Connection):
    """Explicit batch transaction (one commit for the whole block).

    Unlike ``with conn:``, this BEGINs immediately so nested
    ``transaction()`` helpers genuinely join instead of committing one
    by one. A crash or exception rolls the block back; batch callers are
    idempotent, so a re-run resumes cleanly.
    """
    if conn.in_transaction:
        yield
    else:
        conn.execute("BEGIN")
        try:
            yield
        except BaseException:
            conn.rollback()
            raise
        else:
            conn.commit()


def _dump(value) -> str | None:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, default=str)


def _loads(value):
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    return json.loads(value)


def connect(db_path: str, *, timeout: float = 60.0) -> sqlite3.Connection:
    """Open the events database (creates parent dirs; Row factory).

    WAL mode + 60 s busy timeout: UI readers never block batch writers
    (a blocked writer crashed the first 5B run), and concurrent batch
    stages wait instead of failing. In-memory databases keep the default
    journal (WAL is a no-op there).
    """
    target = Path(db_path)
    if str(target.parent) not in ("", "."):
        target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target), check_same_thread=False,
                           timeout=timeout)
    conn.row_factory = sqlite3.Row
    if str(target) != ":memory:":
        conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={int(timeout * 1000)}")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create all Phase 5 tables and indexes (idempotent)."""
    with transaction(conn):
        conn.executescript(SCHEMA_SQL)
        conn.execute(
            "INSERT INTO meta (key, value) VALUES ('store_version', ?) "
            "ON CONFLICT(key) DO NOTHING", (STORE_VERSION,))
    logger.debug("event store schema initialised (%s).", STORE_VERSION)


def _row_to_dict(row: sqlite3.Row, table: str) -> dict:
    record = dict(row)
    for column in _JSON_COLUMNS.get(table, ()):
        if column in record:
            record[column] = _loads(record[column])
    return record


# --- meta ---

def meta_set(conn: sqlite3.Connection, key: str, value) -> None:
    with transaction(conn):
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, _dump(value)))


def meta_get(conn: sqlite3.Connection, key: str):
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return _loads(row["value"]) if row else None


# --- events ---

EVENT_COLUMNS = (
    "event_id", "model_version", "assoc_version", "assoc_params_version",
    "first_observed_utc", "last_observed_utc", "last_evaluated_utc",
    "lifecycle_state", "severity_name", "severity_level",
    "severity_provenance", "sensor_modis_n", "sensor_viirs_n", "satellites",
    "cluster_ids", "anomaly_refs", "detection_count", "detection_refs",
    "detection_resolution", "activity", "frp", "bbox", "centroid",
    "completeness", "ambiguity", "review_required", "review_status",
    "provenance",
)


def _event_params(record: dict) -> dict:
    check_state(record["lifecycle_state"])
    check_severity(record["severity_name"])
    params = {}
    for column in EVENT_COLUMNS:
        value = record.get(column)
        if column in _JSON_COLUMNS["events"]:
            value = _dump(value)
        elif column == "review_required":
            value = int(bool(value))
        params[column] = value
    if not params["event_id"]:
        raise ValueError("event record lacks event_id")
    return params


def upsert_event(conn: sqlite3.Connection, record: dict) -> None:
    """Insert or replace one event (idempotent; full-row replace).

    Transitions are the audit trail for state changes — upsert itself is
    a blind replace used only by the 5A seeder (which never changes the
    state of an existing event) and by later stages through their own
    audited helpers.
    """
    params = _event_params(record)
    columns = ", ".join(EVENT_COLUMNS)
    placeholders = ", ".join(f":{c}" for c in EVENT_COLUMNS)
    with transaction(conn):
        conn.execute(f"INSERT INTO events ({columns}) VALUES ({placeholders}) "
                     f"ON CONFLICT(event_id) DO UPDATE SET "
                     + ", ".join(f"{c} = excluded.{c}" for c in EVENT_COLUMNS
                                 if c != "event_id"),
                     params)


def upsert_events(conn: sqlite3.Connection, records: list[dict]) -> int:
    """Batch idempotent upsert (one transaction; returns row count)."""
    if not records:
        return 0
    rows = [_event_params(record) for record in records]
    columns = ", ".join(EVENT_COLUMNS)
    placeholders = ", ".join(f":{c}" for c in EVENT_COLUMNS)
    with transaction(conn):
        conn.executemany(
            f"INSERT INTO events ({columns}) VALUES ({placeholders}) "
            f"ON CONFLICT(event_id) DO UPDATE SET "
            + ", ".join(f"{c} = excluded.{c}" for c in EVENT_COLUMNS
                        if c != "event_id"),
            rows)
    return len(rows)


def get_event(conn: sqlite3.Connection, event_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM events WHERE event_id = ?",
                       (event_id,)).fetchone()
    return _row_to_dict(row, "events") if row else None


def count_events(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()
    return int(row["n"])


def list_events(conn: sqlite3.Connection, *, state: str | None = None,
                severity: str | None = None, limit: int = 100,
                offset: int = 0, order: str = "rank") -> list[dict]:
    """Bounded event listing.

    order="rank" (UI default): highest severity, then most recent, then
    id. order="id": stable event_id order with no sort, for full-table
    batch scans (immune to mid-scan state/severity changes).
    """
    if limit < 1 or limit > 5000:
        raise ValueError("limit must be in 1..5000")
    if order not in ("rank", "id"):
        raise ValueError(f"unknown order {order!r}")
    clauses: list = []
    params: list = []
    if state is not None:
        check_state(state)
        clauses.append("lifecycle_state = ?")
        params.append(state)
    if severity is not None:
        check_severity(severity)
        clauses.append("severity_name = ?")
        params.append(severity)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    ordering = ("severity_level DESC, last_observed_utc DESC, event_id"
                if order == "rank" else "event_id")
    rows = conn.execute(
        f"SELECT * FROM events {where} ORDER BY {ordering} "
        f"LIMIT ? OFFSET ?",
        (*params, int(limit), int(offset))).fetchall()
    return [_row_to_dict(row, "events") for row in rows]


def update_event_fields(conn: sqlite3.Connection, event_id: str,
                        fields: dict) -> None:
    """Update scalar/JSON columns of one event (validated, transactional)."""
    if not fields:
        return
    if "lifecycle_state" in fields:
        check_state(fields["lifecycle_state"])
    if "severity_name" in fields:
        level = check_severity(fields["severity_name"])
        fields = {**fields, "severity_level": level}
    if "event_id" in fields:
        raise ValueError("event_id is immutable")
    unknown = [k for k in fields if k not in EVENT_COLUMNS]
    if unknown:
        raise ValueError(f"unknown event columns: {unknown}")
    serialised = {k: (_dump(v) if k in _JSON_COLUMNS["events"]
                      or (k == "review_required") else v)
                  for k, v in fields.items()}
    if "review_required" in serialised:
        serialised["review_required"] = int(bool(fields["review_required"]))
    with transaction(conn):
        conn.execute(
            f"UPDATE events SET {', '.join(f'{k} = ?' for k in serialised)} "
            f"WHERE event_id = ?",
            (*serialised.values(), event_id))


# --- transitions (append-only audit) ---

def record_transition(conn: sqlite3.Connection, *, event_id: str,
                      prev_state: str, new_state: str, reason: str,
                      evidence: dict, completeness: dict, algo_version: str,
                      config_version: str, review_required: bool = False,
                      review_performed: bool = False, exec_id: str,
                      at_utc: str | None = None) -> int:
    """Append one lifecycle transition (prev must match current state).

    The event row update and the audit insert happen in one transaction,
    so an interrupted run can never leave a state change unrecorded.
    Returns the transition id.
    """
    check_state(prev_state)
    check_state(new_state)
    if not reason:
        raise ValueError("transition reason is required")
    stamp = at_utc or _utcnow()
    with transaction(conn):
        current = conn.execute(
            "SELECT lifecycle_state FROM events WHERE event_id = ?",
            (event_id,)).fetchone()
        if current is None:
            raise ValueError(f"unknown event {event_id!r}")
        if current["lifecycle_state"] != prev_state:
            raise ValueError(
                f"stale transition for {event_id}: event is "
                f"{current['lifecycle_state']!r}, prev_state is {prev_state!r}")
        cursor = conn.execute(
            "INSERT INTO transitions (event_id, prev_state, new_state, at_utc,"
            " reason, evidence, completeness, algo_version, config_version,"
            " review_required, review_performed, exec_id)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (event_id, prev_state, new_state, stamp, reason,
             _dump(evidence), _dump(completeness), algo_version,
             config_version, int(review_required), int(review_performed),
             exec_id))
        conn.execute(
            "UPDATE events SET lifecycle_state = ?, last_evaluated_utc = ?, "
            "review_required = ? WHERE event_id = ?",
            (new_state, stamp, int(review_required), event_id))
    return int(cursor.lastrowid)


def list_transitions(conn: sqlite3.Connection, event_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM transitions WHERE event_id = ? ORDER BY trans_id",
        (event_id,)).fetchall()
    return [_row_to_dict(row, "transitions") for row in rows]


# --- reviews ---

def record_review(conn: sqlite3.Connection, *, event_id: str,
                  reviewer_label: str, decision: str, note: str = "",
                  at_utc: str | None = None) -> int:
    """Append one auditable human review (decision: approved|rejected)."""
    if decision not in ("approved", "rejected"):
        raise ValueError(f"unknown review decision {decision!r}")
    label = (reviewer_label or "").strip() or "anonymous-reviewer"
    stamp = at_utc or _utcnow()
    with transaction(conn):
        if conn.execute("SELECT 1 FROM events WHERE event_id = ?",
                        (event_id,)).fetchone() is None:
            raise ValueError(f"unknown event {event_id!r}")
        cursor = conn.execute(
            "INSERT INTO reviews (event_id, reviewer_label, decision, note,"
            " at_utc) VALUES (?, ?, ?, ?, ?)",
            (event_id, label, decision, note, stamp))
        conn.execute(
            "UPDATE events SET review_status = ? WHERE event_id = ?",
            (decision, event_id))
    return int(cursor.lastrowid)


def list_reviews(conn: sqlite3.Connection, event_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM reviews WHERE event_id = ? ORDER BY review_id",
        (event_id,)).fetchall()
    return [dict(row) for row in rows]


# --- alerts ---

ALERT_COLUMNS = (
    "alert_id", "alert_type", "severity_name", "severity_level", "event_id",
    "scope", "metric", "grain", "period", "trigger", "baseline_ref",
    "data_quality", "explanation", "created_at", "updated_at",
    "rule_version", "status", "ack_at", "ack_note", "resolved_at",
    "resolve_note", "fingerprint",
)


def _alert_params(record: dict) -> dict:
    check_alert_type(record["alert_type"])
    check_severity(record["severity_name"])
    if record.get("status", "open") not in ("open", "acknowledged", "resolved"):
        raise ValueError(f"unknown alert status {record.get('status')!r}")
    params = {}
    for column in ALERT_COLUMNS:
        value = record.get(column, "" if column not in
                           ("event_id", "ack_at", "resolved_at") else None)
        if column in _JSON_COLUMNS["alerts"]:
            value = _dump(value if value not in (None, "") else {})
        if column == "severity_level":
            value = record.get("severity_level")
            if value is None:
                value = check_severity(record["severity_name"])
        params[column] = value
    if not params["alert_id"]:
        raise ValueError("alert record lacks alert_id")
    return params


def upsert_alert(conn: sqlite3.Connection, record: dict) -> None:
    """Idempotent alert upsert (blind replace; history via alert_history)."""
    params = _alert_params(record)
    columns = ", ".join(ALERT_COLUMNS)
    placeholders = ", ".join(f":{c}" for c in ALERT_COLUMNS)
    with transaction(conn):
        conn.execute(f"INSERT INTO alerts ({columns}) VALUES ({placeholders}) "
                     f"ON CONFLICT(alert_id) DO UPDATE SET "
                     + ", ".join(f"{c} = excluded.{c}" for c in ALERT_COLUMNS
                                 if c not in ("alert_id", "created_at")),
                     params)


def get_alert(conn: sqlite3.Connection, alert_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM alerts WHERE alert_id = ?",
                       (alert_id,)).fetchone()
    return _row_to_dict(row, "alerts") if row else None


def count_alerts(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS n FROM alerts").fetchone()
    return int(row["n"])


def list_alerts(conn: sqlite3.Connection, *, status: str | None = None,
                alert_type: str | None = None, event_id: str | None = None,
                limit: int = 100, offset: int = 0) -> list[dict]:
    """Bounded alert listing (highest severity, then most recent, then id)."""
    if limit < 1 or limit > 5000:
        raise ValueError("limit must be in 1..5000")
    clauses, params = [], []
    if status is not None:
        if status not in ("open", "acknowledged", "resolved"):
            raise ValueError(f"unknown alert status {status!r}")
        clauses.append("status = ?")
        params.append(status)
    if alert_type is not None:
        check_alert_type(alert_type)
        clauses.append("alert_type = ?")
        params.append(alert_type)
    if event_id is not None:
        clauses.append("event_id = ?")
        params.append(event_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"SELECT * FROM alerts {where} ORDER BY severity_level DESC, "
        f"updated_at DESC, alert_id LIMIT ? OFFSET ?",
        (*params, int(limit), int(offset))).fetchall()
    return [_row_to_dict(row, "alerts") for row in rows]


def set_alert_status(conn: sqlite3.Connection, alert_id: str, status: str, *,
                     note: str = "", exec_id: str = "",
                     at_utc: str | None = None) -> None:
    """Acknowledge or resolve an alert (transactional + history entry)."""
    if status not in ("acknowledged", "resolved", "open"):
        raise ValueError(f"unknown alert status {status!r}")
    stamp = at_utc or _utcnow()
    with transaction(conn):
        current = conn.execute("SELECT status FROM alerts WHERE alert_id = ?",
                               (alert_id,)).fetchone()
        if current is None:
            raise ValueError(f"unknown alert {alert_id!r}")
        if status == "acknowledged":
            conn.execute("UPDATE alerts SET status = 'acknowledged', "
                         "ack_at = ?, ack_note = ?, updated_at = ? "
                         "WHERE alert_id = ?", (stamp, note, stamp, alert_id))
        elif status == "resolved":
            conn.execute("UPDATE alerts SET status = 'resolved', "
                         "resolved_at = ?, resolve_note = ?, updated_at = ? "
                         "WHERE alert_id = ?", (stamp, note, stamp, alert_id))
        else:
            conn.execute("UPDATE alerts SET status = 'open', updated_at = ? "
                         "WHERE alert_id = ?", (stamp, alert_id))
        conn.execute("INSERT INTO alert_history (alert_id, at_utc, action,"
                     " detail, exec_id) VALUES (?, ?, ?, ?, ?)",
                     (alert_id, stamp, status,
                      _dump({"note": note,
                             "previous": current["status"]}), exec_id))


def append_alert_history(conn: sqlite3.Connection, alert_id: str, action: str,
                         detail: dict | None = None, exec_id: str = "",
                         at_utc: str | None = None) -> int:
    with transaction(conn):
        cursor = conn.execute(
            "INSERT INTO alert_history (alert_id, at_utc, action, detail,"
            " exec_id) VALUES (?, ?, ?, ?, ?)",
            (alert_id, at_utc or _utcnow(), action, _dump(detail or {}),
             exec_id))
    return int(cursor.lastrowid)


def list_alert_history(conn: sqlite3.Connection, alert_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM alert_history WHERE alert_id = ? ORDER BY hist_id",
        (alert_id,)).fetchall()
    return [_row_to_dict(row, "alert_history") for row in rows]


# --- subscriptions ---

SUB_COLUMNS = (
    "sub_id", "email", "status", "confirm_token_hash", "confirm_expires_utc",
    "manage_token_hash", "created_at", "confirmed_at", "unsubscribed_at",
    "prefs",
)


def upsert_subscription(conn: sqlite3.Connection, record: dict) -> None:
    if record.get("status") not in ("pending", "confirmed", "unsubscribed"):
        raise ValueError(f"unknown subscription status {record.get('status')!r}")
    params = {c: (_dump(record.get(c)) if c in _JSON_COLUMNS["subscriptions"]
                  else record.get(c)) for c in SUB_COLUMNS}
    if not params["sub_id"]:
        raise ValueError("subscription record lacks sub_id")
    columns = ", ".join(SUB_COLUMNS)
    placeholders = ", ".join(f":{c}" for c in SUB_COLUMNS)
    with transaction(conn):
        conn.execute(f"INSERT INTO subscriptions ({columns}) "
                     f"VALUES ({placeholders}) "
                     f"ON CONFLICT(sub_id) DO UPDATE SET "
                     + ", ".join(f"{c} = excluded.{c}" for c in SUB_COLUMNS
                                 if c != "sub_id"),
                     params)


def get_subscription(conn: sqlite3.Connection, sub_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM subscriptions WHERE sub_id = ?",
                       (sub_id,)).fetchone()
    return _row_to_dict(row, "subscriptions") if row else None


def find_subscription_by_email(conn: sqlite3.Connection,
                               email: str) -> dict | None:
    row = conn.execute("SELECT * FROM subscriptions WHERE email = ? "
                       "ORDER BY created_at DESC LIMIT 1", (email,)).fetchone()
    return _row_to_dict(row, "subscriptions") if row else None


def find_subscription_by_token_hash(conn: sqlite3.Connection,
                                    token_hash: str,
                                    kind: str) -> dict | None:
    """Look up a subscription by a token hash (kind: confirm|manage)."""
    column = {"confirm": "confirm_token_hash",
              "manage": "manage_token_hash"}[kind]
    row = conn.execute(f"SELECT * FROM subscriptions WHERE {column} = ?",
                       (token_hash,)).fetchone()
    return _row_to_dict(row, "subscriptions") if row else None


def list_subscriptions(conn: sqlite3.Connection, *,
                       status: str | None = None) -> list[dict]:
    if status is not None and status not in ("pending", "confirmed",
                                             "unsubscribed"):
        raise ValueError(f"unknown subscription status {status!r}")
    if status is None:
        rows = conn.execute(
            "SELECT * FROM subscriptions ORDER BY created_at").fetchall()
    else:
        rows = conn.execute("SELECT * FROM subscriptions WHERE status = ? "
                            "ORDER BY created_at", (status,)).fetchall()
    return [_row_to_dict(row, "subscriptions") for row in rows]


# --- notifications ---

NOTIF_COLUMNS = (
    "notif_id", "alert_id", "sub_id", "kind", "status", "attempts",
    "last_error", "created_at", "sent_at", "fingerprint",
)


def upsert_notification(conn: sqlite3.Connection, record: dict) -> None:
    params = {c: record.get(c) for c in NOTIF_COLUMNS}
    if not params["notif_id"]:
        raise ValueError("notification record lacks notif_id")
    columns = ", ".join(NOTIF_COLUMNS)
    placeholders = ", ".join(f":{c}" for c in NOTIF_COLUMNS)
    with transaction(conn):
        conn.execute(f"INSERT INTO notifications ({columns}) "
                     f"VALUES ({placeholders}) "
                     f"ON CONFLICT(notif_id) DO UPDATE SET "
                     + ", ".join(f"{c} = excluded.{c}" for c in NOTIF_COLUMNS
                                 if c != "notif_id"),
                     params)


def get_notification(conn: sqlite3.Connection, notif_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM notifications WHERE notif_id = ?",
                       (notif_id,)).fetchone()
    return dict(row) if row else None


def list_notifications(conn: sqlite3.Connection, *,
                       status: str | None = None,
                       limit: int = 200) -> list[dict]:
    if limit < 1 or limit > 5000:
        raise ValueError("limit must be in 1..5000")
    if status is None:
        rows = conn.execute("SELECT * FROM notifications ORDER BY created_at"
                            " DESC LIMIT ?", (limit,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM notifications WHERE status = ? "
                            "ORDER BY created_at DESC LIMIT ?",
                            (status, limit,)).fetchall()
    return [dict(row) for row in rows]


# --- qwen_log ---

def append_qwen_log(conn: sqlite3.Connection, *, target_type: str,
                    target_id: str, status: str,
                    latency_ms: float | None = None, model: str = "",
                    error: str = "", at_utc: str | None = None) -> int:
    with transaction(conn):
        cursor = conn.execute(
            "INSERT INTO qwen_log (at_utc, target_type, target_id, status,"
            " latency_ms, model, error) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (at_utc or _utcnow(), target_type, target_id, status,
             latency_ms, model, error[:500]))
    return int(cursor.lastrowid)


def list_qwen_log(conn: sqlite3.Connection, *, target_type: str | None = None,
                  target_id: str | None = None,
                  limit: int = 200) -> list[dict]:
    clauses: list = []
    params: list = []
    if target_type is not None:
        clauses.append("target_type = ?")
        params.append(target_type)
    if target_id is not None:
        clauses.append("target_id = ?")
        params.append(target_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(f"SELECT * FROM qwen_log {where} ORDER BY log_id DESC"
                        f" LIMIT ?", (*params, int(limit))).fetchall()
    return [dict(row) for row in rows]


# --- custom rules (user-defined; separate from system alerts) ---

CUSTOM_RULE_COLUMNS = (
    "rule_id", "name", "description", "metric", "scope", "grain", "period",
    "geo_scope", "operator", "threshold", "anomaly_class", "severity_name",
    "severity_level", "notify", "enabled", "created_at", "updated_at",
    "version",
)


def upsert_custom_rule(conn: sqlite3.Connection, record: dict) -> None:
    """Create or replace one custom rule (validated, idempotent)."""
    check_severity(record.get("severity_name", "Moderate"))
    if not record.get("rule_id"):
        raise ValueError("custom rule lacks rule_id")
    params = {c: record.get(c) for c in CUSTOM_RULE_COLUMNS}
    params["severity_level"] = check_severity(record["severity_name"])
    params["notify"] = int(bool(record.get("notify")))
    params["enabled"] = int(bool(record.get("enabled", True)))
    columns = ", ".join(CUSTOM_RULE_COLUMNS)
    placeholders = ", ".join(f":{c}" for c in CUSTOM_RULE_COLUMNS)
    with transaction(conn):
        conn.execute(f"INSERT INTO custom_rules ({columns}) "
                     f"VALUES ({placeholders}) "
                     f"ON CONFLICT(rule_id) DO UPDATE SET "
                     + ", ".join(f"{c} = excluded.{c}" for c in CUSTOM_RULE_COLUMNS
                                 if c not in ("rule_id", "created_at")),
                     params)


def get_custom_rule(conn: sqlite3.Connection, rule_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM custom_rules WHERE rule_id = ?",
                       (rule_id,)).fetchone()
    return dict(row) if row else None


def list_custom_rules(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM custom_rules ORDER BY created_at").fetchall()
    return [dict(row) for row in rows]


def delete_custom_rule(conn: sqlite3.Connection, rule_id: str) -> bool:
    """Delete a rule (hits and evals are retained as audit history)."""
    with transaction(conn):
        cursor = conn.execute("DELETE FROM custom_rules WHERE rule_id = ?",
                              (rule_id,))
    return cursor.rowcount > 0


def set_custom_rule_enabled(conn: sqlite3.Connection, rule_id: str,
                            enabled: bool, at_utc: str | None = None) -> None:
    with transaction(conn):
        cursor = conn.execute("UPDATE custom_rules SET enabled = ?, "
                              "updated_at = ? WHERE rule_id = ?",
                              (int(enabled), at_utc or _utcnow(), rule_id))
        if cursor.rowcount == 0:
            raise ValueError(f"unknown custom rule {rule_id!r}")


def append_custom_eval(conn: sqlite3.Connection, *, rule_id: str,
                       at_utc: str, triggered: bool, reason: str) -> int:
    with transaction(conn):
        cursor = conn.execute(
            "INSERT INTO custom_rule_evals (rule_id, at_utc, triggered,"
            " reason) VALUES (?, ?, ?, ?)",
            (rule_id, at_utc, int(triggered), reason))
    return int(cursor.lastrowid)


def list_custom_evals(conn: sqlite3.Connection, rule_id: str,
                      limit: int = 100) -> list[dict]:
    rows = conn.execute("SELECT * FROM custom_rule_evals WHERE rule_id = ? "
                        "ORDER BY eval_id DESC LIMIT ?",
                        (rule_id, int(limit))).fetchall()
    return [dict(row) for row in rows]


CUSTOM_HIT_COLUMNS = (
    "hit_id", "rule_id", "rule_name", "severity_name", "period", "metric",
    "scope", "summary", "trigger_refs", "triggered_at", "status", "version",
)


def upsert_custom_hit(conn: sqlite3.Connection, record: dict) -> None:
    """Idempotent hit upsert (stable hit_id ⇒ re-eval never duplicates)."""
    if not record.get("hit_id"):
        raise ValueError("custom hit lacks hit_id")
    if record.get("status", "open") not in ("open", "acknowledged"):
        raise ValueError(f"unknown hit status {record.get('status')!r}")
    params = {c: (_dump(record.get(c)) if c in _JSON_COLUMNS["custom_hits"]
                  else record.get(c)) for c in CUSTOM_HIT_COLUMNS}
    columns = ", ".join(CUSTOM_HIT_COLUMNS)
    placeholders = ", ".join(f":{c}" for c in CUSTOM_HIT_COLUMNS)
    with transaction(conn):
        conn.execute(f"INSERT INTO custom_hits ({columns}) "
                     f"VALUES ({placeholders}) "
                     f"ON CONFLICT(hit_id) DO UPDATE SET "
                     "summary = excluded.summary, "
                     "trigger_refs = excluded.trigger_refs, "
                     "triggered_at = excluded.triggered_at",
                     params)


def get_custom_hit(conn: sqlite3.Connection, hit_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM custom_hits WHERE hit_id = ?",
                       (hit_id,)).fetchone()
    return _row_to_dict(row, "custom_hits") if row else None


def list_custom_hits(conn: sqlite3.Connection, *,
                     rule_id: str | None = None,
                     limit: int = 200) -> list[dict]:
    if rule_id is None:
        rows = conn.execute("SELECT * FROM custom_hits ORDER BY triggered_at"
                            " DESC LIMIT ?", (int(limit),)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM custom_hits WHERE rule_id = ? "
                            "ORDER BY triggered_at DESC LIMIT ?",
                            (rule_id, int(limit))).fetchall()
    return [_row_to_dict(row, "custom_hits") for row in rows]


def set_custom_hit_status(conn: sqlite3.Connection, hit_id: str,
                          status: str) -> None:
    if status not in ("open", "acknowledged"):
        raise ValueError(f"unknown hit status {status!r}")
    with transaction(conn):
        cursor = conn.execute("UPDATE custom_hits SET status = ? "
                              "WHERE hit_id = ?", (status, hit_id))
        if cursor.rowcount == 0:
            raise ValueError(f"unknown custom hit {hit_id!r}")


def queue_custom_notification(conn: sqlite3.Connection, *, hit_id: str,
                              fingerprint: str, created_at: str) -> str:
    """Queue one notification intent per hit (idempotent, never requeues)."""
    notif_id = f"cn-{hit_id}"
    with transaction(conn):
        current = conn.execute("SELECT status FROM custom_notifications "
                               "WHERE notif_id = ?", (notif_id,)).fetchone()
        if current is not None and current["status"] == "sent":
            return notif_id
        conn.execute(
            "INSERT INTO custom_notifications (notif_id, hit_id, fingerprint,"
            " status, attempts, last_error, created_at, sent_at)"
            " VALUES (?, ?, ?, 'queued', 0, '', ?, NULL)"
            " ON CONFLICT(notif_id) DO UPDATE SET fingerprint = excluded."
            "fingerprint", (notif_id, hit_id, fingerprint, created_at))
    return notif_id


def list_custom_notifications(conn: sqlite3.Connection, *,
                              status: str | None = None,
                              limit: int = 200) -> list[dict]:
    if status is None:
        rows = conn.execute("SELECT * FROM custom_notifications ORDER BY "
                            "created_at DESC LIMIT ?", (int(limit),)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM custom_notifications WHERE "
                            "status = ? ORDER BY created_at DESC LIMIT ?",
                            (status, int(limit))).fetchall()
    return [dict(row) for row in rows]


def set_custom_notification(conn: sqlite3.Connection, notif_id: str,
                            status: str, error: str = "",
                            at_utc: str | None = None) -> None:
    if status not in ("queued", "sent", "failed"):
        raise ValueError(f"unknown notification status {status!r}")
    stamp = at_utc or _utcnow()
    with transaction(conn):
        conn.execute("UPDATE custom_notifications SET status = ?, "
                     "last_error = ?, sent_at = CASE WHEN ? = 'sent' THEN ? "
                     "ELSE sent_at END WHERE notif_id = ?",
                     (status, error[:500], status, stamp, notif_id))


def bump_custom_notification(conn: sqlite3.Connection, notif_id: str,
                             at_utc: str | None = None) -> None:
    _ = at_utc
    with transaction(conn):
        conn.execute("UPDATE custom_notifications SET attempts = attempts + 1,"
                     " last_error = 'sender reported failure' "
                     "WHERE notif_id = ?", (notif_id,))
