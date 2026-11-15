"""Stage 5B: conservative, auditable event lifecycle (lifecycle_v1).

States: Active -> Monitoring -> Pending Closure -> Resolved, with return
to Active/Monitoring on new qualifying activity (including reopening a
Resolved event). Closure-side moves advance one step per evaluation so
the audit trail shows the full ladder; activity-side moves apply
immediately (the more alert state always wins).

Safety core (non-negotiable):
- Resolved never means contained or extinguished (see RESOLVED_DISCLAIMER).
- Missing/stale data can never advance an event toward closure: at most
  an Active event steps to Monitoring as an explicit data-hold, and every
  other state is preserved with an insufficient-evidence reason.
- High/Critical events require an approved human review before Resolved.
- Every transition records prev/new state, timestamp, reason, evidence,
  completeness, versions, and review flags (store.transitions).

Evaluation is data-relative: the caller passes an explicit UTC as-of
date. For the historical pilot the canonical as-of is the last verified
coverage date (2023-12-31); an NRT deployment passes the current date.
Day granularity matches the FIRMS daily acquisition cadence.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .. import settings
from ..analytics import runs
from ..ui import query
from .model import (LIFECYCLE_CFG_VERSION, LIFECYCLE_STATES, LIFECYCLE_VERSION,
                    STATE_ACTIVE, STATE_MONITORING, STATE_PENDING,
                    STATE_RESOLVED, check_state)

from . import store as store_mod

logger = logging.getLogger(__name__)

STATE_INDEX = {name: idx for idx, name in enumerate(LIFECYCLE_STATES)}

# Severities that require an approved human review before final resolution.
REVIEW_SEVERITIES = ("High", "Critical")


@dataclass(frozen=True)
class LifecycleConfig:
    """Versioned lifecycle policy (lifecycle_cfg_v1).

    Quiet periods are conservative MVP policy, not scientific claims: a
    full week of verified-coverage silence before leaving Active, three
    weeks before Pending Closure, 45 days before Resolved. All values are
    explicit, configurable, and recorded on every transition.
    """

    quiet_to_monitoring_d: int = 7
    quiet_to_pending_d: int = 21
    quiet_to_resolved_d: int = 45
    version: str = LIFECYCLE_CFG_VERSION

    def target_state(self, quiet_days: int) -> str:
        if quiet_days >= self.quiet_to_resolved_d:
            return STATE_RESOLVED
        if quiet_days >= self.quiet_to_pending_d:
            return STATE_PENDING
        if quiet_days >= self.quiet_to_monitoring_d:
            return STATE_MONITORING
        return STATE_ACTIVE


def _parse_day(value: str, *, what: str) -> date:
    try:
        return date.fromisoformat(value[:10])
    except (ValueError, TypeError, AttributeError):
        raise ValueError(f"unparseable {what}: {value!r}")


def coverage_context(as_of: str) -> dict:
    """Verified-coverage context for an as-of date (catalog-derived).

    Returns {as_of, coverage_max, coverage_complete, spans}. Outside every
    verified span the evaluation is stale and the lifecycle holds.
    """
    as_of_day = _parse_day(as_of, what="as_of").isoformat()
    catalog = json.loads(Path(settings.CATALOG_PATH).read_text(
        encoding="utf-8"))
    spans = query.catalog_spans(catalog)
    coverage_max = max((info["max"] for info in spans.values()),
                       default="") if spans else ""
    complete = bool(spans) and all(
        info["min"] <= as_of_day <= info["max"] for info in spans.values())
    return {"as_of": as_of_day, "coverage_max": coverage_max,
            "coverage_complete": complete, "spans": spans}


def evaluate(event: dict, config: LifecycleConfig | None = None,
             ctx: dict | None = None) -> dict:
    """Decide the next lifecycle step for one event (pure, deterministic)."""
    config = config or LifecycleConfig()
    ctx = ctx or {}
    as_of = ctx.get("as_of", "")
    as_of_day = _parse_day(as_of, what="as_of")
    last_day = _parse_day(event["last_observed_utc"], what="last_observed_utc")
    quiet_days = (as_of_day - last_day).days
    current = check_state(event["lifecycle_state"])
    completeness = {
        "coverage_status": ("complete" if ctx.get("coverage_complete")
                            else "incomplete-or-stale"),
        "coverage_max": ctx.get("coverage_max", ""),
        "as_of": as_of_day.isoformat(),
        "event_coverage": event.get("completeness", {}).get(
            "coverage_status", "unknown"),
    }
    base_evidence = {
        "quiet_days": quiet_days,
        "last_observed_utc": event["last_observed_utc"],
        "as_of": as_of_day.isoformat(),
        "coverage_max": ctx.get("coverage_max", ""),
        "severity": event.get("severity_name"),
    }

    def decision(new_state: str, reason: str, *,
                 review_required: bool = False,
                 review_performed: bool = False,
                 extra: dict | None = None) -> dict:
        return {
            "event_id": event["event_id"], "prev_state": current,
            "new_state": new_state, "changed": new_state != current,
            "reason": reason,
            "evidence": {**base_evidence, **(extra or {})},
            "completeness": completeness,
            "review_required": review_required,
            "review_performed": review_performed,
            "algo_version": LIFECYCLE_VERSION,
            "config_version": config.version,
        }

    # New qualifying activity (last observation advanced since the last
    # evaluation) reactivates from ANY state, including Resolved.
    last_eval = event.get("last_evaluated_utc")
    if last_eval is not None and event["last_observed_utc"] > last_eval:
        reopened = current == STATE_RESOLVED
        return decision(STATE_ACTIVE, "new-qualifying-activity",
                        extra={"reopened": reopened,
                               "previous_evaluation": last_eval})

    # As-of preceding the observation is incoherent input: hold Active.
    if quiet_days < 0:
        return decision(STATE_ACTIVE, "as-of-precedes-observation",
                        extra={"held_state": current})

    # Data-sufficiency gate: stale or incomplete coverage can never move
    # an event toward closure.
    if not ctx.get("coverage_complete"):
        if current == STATE_ACTIVE:
            return decision(STATE_MONITORING, "data-hold-stale-coverage",
                            extra={"held_closure": True,
                                   "note": "Active stepped to Monitoring as "
                                   "an explicit data hold, not a closure"})
        return decision(current, "insufficient-evidence-hold",
                        extra={"held_closure": True})

    target = config.target_state(quiet_days)
    target_idx, current_idx = STATE_INDEX[target], STATE_INDEX[current]
    if target_idx <= current_idx:
        # Activity-side: the evidence supports a more alert (or equal)
        # state — apply immediately (a quiet revision, e.g. backfill).
        if target == current:
            return decision(current, "state-sustained",
                            extra={"target_state": target})
        return decision(target, "evidence-supports-alert-state",
                        extra={"target_state": target})
    # Closure-side: advance exactly one ladder step per evaluation.
    nxt = LIFECYCLE_STATES[current_idx + 1]
    if nxt == STATE_RESOLVED and event.get("severity_name") in REVIEW_SEVERITIES:
        approved = event.get("review_status") == "approved"
        if not approved:
            held = STATE_PENDING if current_idx + 1 >= STATE_INDEX[STATE_PENDING] \
                else nxt
            return decision(held, "review-required-hold",
                            review_required=True,
                            extra={"target_state": target,
                                   "required": "approved human review for "
                                   f"{event.get('severity_name')} events"})
        return decision(nxt, "quiet-period-elapsed",
                        review_performed=True,
                        extra={"target_state": target})
    reason = {STATE_MONITORING: "quiet-period-elapsed",
              STATE_PENDING: "quiet-period-elapsed",
              STATE_RESOLVED: "quiet-period-elapsed"}[nxt]
    return decision(nxt, reason, extra={"target_state": target})


def apply_decision(conn, decision: dict, exec_id: str,
                   at_utc: str | None = None) -> bool:
    """Apply one decision (transition iff the state changes; else stamp).

    Always stamps last_evaluated_utc so a later observation advance is
    detectable as new activity. Returns True when a transition occurred.
    """
    event_id = decision["event_id"]
    if decision["changed"]:
        store_mod.record_transition(
            conn, event_id=event_id, prev_state=decision["prev_state"],
            new_state=decision["new_state"], reason=decision["reason"],
            evidence=decision["evidence"],
            completeness=decision["completeness"],
            algo_version=decision["algo_version"],
            config_version=decision["config_version"],
            review_required=decision["review_required"],
            review_performed=decision["review_performed"],
            exec_id=exec_id, at_utc=at_utc)
        return True
    store_mod.update_event_fields(
        conn, event_id,
        {"last_evaluated_utc": at_utc or decision["evidence"]["as_of"]
         + "T23:59:59+00:00",
         "review_required": decision["review_required"]})
    return False


def _evaluation_stamp(as_of_day: str) -> str:
    return f"{as_of_day}T23:59:59+00:00"


def run_lifecycle(db_path: str, as_of: str, runs_path: str) -> dict:
    """Evaluate every event to quiescence (bounded ladder passes)."""
    from datetime import datetime, timezone  # local: keeps import light

    t0 = time.monotonic()
    config = LifecycleConfig()
    ctx = coverage_context(as_of)
    exec_id = runs.new_execution(
        runs_path, stage="5B",
        inputs={"db": db_path, "catalog": settings.CATALOG_PATH},
        params={"as_of": ctx["as_of"], "coverage_max": ctx["coverage_max"],
                "coverage_complete": ctx["coverage_complete"],
                "quiet_d": [config.quiet_to_monitoring_d,
                            config.quiet_to_pending_d,
                            config.quiet_to_resolved_d],
                "algo": LIFECYCLE_VERSION,
                "config": config.version})
    conn = store_mod.connect(db_path)
    store_mod.init_schema(conn)
    stamp = _evaluation_stamp(ctx["as_of"])
    started = datetime.now(timezone.utc).isoformat()
    transitions = 0
    moves: dict[str, int] = {}
    # At most 3 ladder steps exist (Active->…->Resolved); 4 passes bound
    # the loop even under pathological input.
    for _ in range(4):
        moved_this_pass = 0
        offset = 0
        while True:
            batch = store_mod.list_events(conn, limit=5000, offset=offset,
                                          order="id")
            if not batch:
                break
            # One commit per page (was: one per event — 100x slower and
            # vulnerable to lock timeouts under concurrent readers).
            with store_mod.batch(conn):
                for event in batch:
                    decision = evaluate(event, config, ctx)
                    if apply_decision(conn, decision, exec_id, at_utc=stamp):
                        transitions += 1
                        moved_this_pass += 1
                        key = (f"{decision['prev_state']}"
                               f"->{decision['new_state']}")
                        moves[key] = moves.get(key, 0) + 1
            offset += len(batch)
            if len(batch) < 5000:
                break
        if moved_this_pass == 0:
            break
    rows = conn.execute("SELECT lifecycle_state, COUNT(*) AS n FROM events "
                        "GROUP BY lifecycle_state").fetchall()
    states = {row["lifecycle_state"]: row["n"] for row in rows}
    store_mod.meta_set(conn, "exec_5b", exec_id)
    outputs = {"as_of": ctx["as_of"], "coverage_complete":
               ctx["coverage_complete"], "transitions": transitions,
               "moves": moves, "states": states, "started_at": started,
               "elapsed_s": round(time.monotonic() - t0, 1)}
    runs.complete_execution(runs_path, exec_id, outputs)
    logger.info("5B complete: %d transitions, states=%s in %ss",
                transitions, states, outputs["elapsed_s"])
    return {"exec_id": exec_id, **outputs}
