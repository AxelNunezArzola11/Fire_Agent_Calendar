"""User-created custom alerts (custom_rules_v1) — manual evaluation only.

Separate from the immutable Phase 5 predefined rules (`alerts.py`,
`alert_rules_v1`): custom rules live in their own tables, use stable
`crt-`/`crh-` ids, and are evaluated ONLY when the user explicitly runs
an evaluation. There is no background scheduler, so nothing here
implies continuous monitoring.

Scientific safeguards (enforced, not just documented):
- Conditions read stored values only (2D metrics, 4C anomaly records):
  a threshold crossing is never proof of a wildfire.
- (metric, scope) must be an existing supported series (no pooling,
  no invented combinations, no composite scores).
- FRP missing (None) never triggers (never treated as zero).
- Baseline-relative conditions reuse 4C classifications verbatim and
  refuse evaluation unless reference_end strictly precedes the
  evaluated period (leakage-free by construction).
- Custom hits are labeled user-defined everywhere and never enter
  the system `alerts` table.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone

from ..analytics import baselines
from .model import check_severity

from . import store as store_mod

logger = logging.getLogger(__name__)

CUSTOM_VERSION = "custom_rules_v1"

OPERATORS = ("gt", "gte", "lt", "lte")
OPERATOR_LABELS = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}
GRAINS = ("daily", "weekly", "monthly")
ANOMALY_CLASSES = ("notable-high", "notable-low", "provisional-notable-high",
                   "provisional-notable-low", "typical", "provisional-typical")
GEO_SCOPE = "USA-pilot"  # the only honest scope: metrics are pilot-wide


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_rule_id() -> str:
    return f"crt-{uuid.uuid4().hex[:12]}-r1"


def validate_rule(rule: dict) -> dict:
    """Validate + normalize a custom rule (raises ValueError).

    Returns the normalized rule dict (does not persist).
    """
    name = (rule.get("name") or "").strip()
    if not name or len(name) > 120:
        raise ValueError("rule name is required (1-120 chars)")
    metric = rule.get("metric")
    scope = rule.get("scope")
    if (metric, scope) not in baselines.SERIES_SPECS:
        raise ValueError(f"unsupported series {(metric, scope)}; no "
                         f"sensor pooling or invented combinations allowed")
    grain = rule.get("grain")
    if grain not in GRAINS:
        raise ValueError(f"unknown grain {grain!r}; want {list(GRAINS)}")
    period = (rule.get("period") or "").strip()
    if not period:
        raise ValueError("evaluation period is required")
    try:
        baselines.slot_index(grain, period)
    except ValueError:
        raise ValueError(f"period {period!r} is not a valid {grain} period")
    operator = rule.get("operator")
    threshold = rule.get("threshold")
    if (operator is None) != (threshold is None):
        raise ValueError("operator and threshold must be set together")
    if operator is not None and operator not in OPERATORS:
        raise ValueError(f"unknown operator {operator!r}")
    if threshold is not None:
        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            raise ValueError(f"threshold {threshold!r} is not a number")
    anomaly_class = rule.get("anomaly_class")
    if anomaly_class is not None and anomaly_class not in ANOMALY_CLASSES:
        raise ValueError(f"unknown anomaly class {anomaly_class!r}")
    if operator is None and anomaly_class is None:
        raise ValueError("rule needs a threshold, an anomaly class, or both")
    check_severity(rule.get("severity_name", "Moderate"))
    return {
        "rule_id": rule.get("rule_id") or new_rule_id(),
        "name": name,
        "description": (rule.get("description") or "").strip()[:500],
        "metric": metric, "scope": scope, "grain": grain,
        "period": period, "geo_scope": GEO_SCOPE,
        "operator": operator, "threshold": threshold,
        "anomaly_class": anomaly_class,
        "severity_name": rule.get("severity_name", "Moderate"),
        "notify": bool(rule.get("notify", False)),
        "enabled": bool(rule.get("enabled", True)),
        "version": CUSTOM_VERSION,
    }


def _compare(operator: str, value: float, threshold: float) -> bool:
    return {"gt": value > threshold, "gte": value >= threshold,
            "lt": value < threshold, "lte": value <= threshold}[operator]


def evaluate_rule(rule: dict, *, metrics: dict,
                  anomalies: list[dict]) -> dict:
    """Evaluate one rule against stored outputs (pure, deterministic).

    Returns {triggered, reason, value, anomaly_id, classification}.
    Never raises on missing data: absence yields triggered=False with
    an explicit reason.
    """
    metric, scope, grain, period = (rule["metric"], rule["scope"],
                                    rule["grain"], rule["period"])
    try:
        entries = baselines.extract_series(metrics, grain, metric, scope)
    except ValueError as exc:
        return {"triggered": False, "reason": f"unsupported series: {exc}",
                "value": None, "anomaly_id": None, "classification": None}
    observed = next((e["value"] for e in entries if e["period"] == period),
                    None)
    if rule.get("operator") is not None:
        if observed is None:
            return {"triggered": False,
                    "reason": f"no verified {metric}/{scope} value for "
                    f"{period} (missing, not zero)",
                    "value": None, "anomaly_id": None, "classification": None}
        if not _compare(rule["operator"], float(observed),
                        float(rule["threshold"])):
            return {"triggered": False,
                    "reason": f"observed {observed} does not satisfy "
                    f"{OPERATOR_LABELS[rule['operator']]} {rule['threshold']}",
                    "value": observed, "anomaly_id": None,
                    "classification": None}
    classification = None
    anomaly_id = None
    if rule.get("anomaly_class") is not None:
        match = next((a for a in anomalies
                      if a.get("metric") == metric and a.get("scope") == scope
                      and a.get("baseline_method") == "sensor_aware"
                      and a.get("grain") == grain
                      and a.get("eval_period") == period), None)
        if match is None:
            return {"triggered": False,
                    "reason": f"no eligible sensor_aware anomaly record for "
                    f"{period} (missing evidence, not a negative)",
                    "value": observed, "anomaly_id": None,
                    "classification": None}
        ref_end = match.get("reference_end") or ""
        if not ref_end or not ref_end < period:
            return {"triggered": False,
                    "reason": "refused: reference period does not strictly "
                    "precede the evaluation period (leakage guard)",
                    "value": observed, "anomaly_id": match.get("anomaly_id"),
                    "classification": match.get("classification")}
        classification = match.get("classification")
        anomaly_id = match.get("anomaly_id")
        if classification != rule["anomaly_class"]:
            return {"triggered": False,
                    "reason": f"anomaly class is {classification}, rule "
                    f"requires {rule['anomaly_class']}",
                    "value": observed, "anomaly_id": anomaly_id,
                    "classification": classification}
    parts = []
    if rule.get("operator") is not None:
        parts.append(f"{metric}/{scope} {observed} "
                     f"{OPERATOR_LABELS[rule['operator']]} {rule['threshold']}")
    if rule.get("anomaly_class") is not None:
        parts.append(f"anomaly {anomaly_id} is {classification}")
    return {"triggered": True, "reason": " AND ".join(parts),
            "value": observed, "anomaly_id": anomaly_id,
            "classification": classification}


def hit_id_for(rule_id: str, period: str, condition: dict) -> str:
    """Stable hit id: same rule + period + condition never duplicates."""
    digest = hashlib.sha256(json.dumps(condition, sort_keys=True,
                                       default=str).encode()).hexdigest()[:8]
    return f"crh-{rule_id}-{period}-{digest}"


def run_evaluation(conn, rule_id: str, *, metrics: dict,
                   anomalies: list[dict], at_utc: str | None = None) -> dict:
    """Evaluate one enabled rule now (manual/on-demand only).

    Records eval history; upserts the hit on trigger (idempotent);
    queues a notification intent when rule.notify is set. Returns
    {rule_id, triggered, reason, hit_id}.
    """
    rule = store_mod.get_custom_rule(conn, rule_id)
    if rule is None:
        raise ValueError(f"unknown custom rule {rule_id!r}")
    if not rule["enabled"]:
        raise ValueError(f"custom rule {rule_id} is disabled")
    stamp = at_utc or _utcnow()
    outcome = evaluate_rule(rule, metrics=metrics, anomalies=anomalies)
    store_mod.append_custom_eval(
        conn, rule_id=rule_id, at_utc=stamp,
        triggered=bool(outcome["triggered"]), reason=outcome["reason"])
    hit_id = None
    if outcome["triggered"]:
        condition = {"operator": rule["operator"],
                     "threshold": rule["threshold"],
                     "anomaly_class": rule["anomaly_class"]}
        hit_id = hit_id_for(rule_id, rule["period"], condition)
        store_mod.upsert_custom_hit(conn, {
            "hit_id": hit_id, "rule_id": rule_id,
            "rule_name": rule["name"], "severity_name": rule["severity_name"],
            "period": rule["period"], "metric": rule["metric"],
            "scope": rule["scope"], "summary": outcome["reason"],
            "trigger_refs": {"value": outcome["value"],
                             "anomaly_id": outcome["anomaly_id"],
                             "classification": outcome["classification"]},
            "triggered_at": stamp, "status": "open",
            "version": CUSTOM_VERSION})
        if rule["notify"]:
            store_mod.queue_custom_notification(
                conn, hit_id=hit_id, fingerprint=hit_id,
                created_at=stamp)
    logger.info("custom rule %s evaluated: triggered=%s", rule_id,
                outcome["triggered"])
    return {"rule_id": rule_id, "triggered": bool(outcome["triggered"]),
            "reason": outcome["reason"], "hit_id": hit_id}


def render_custom_notification(hit: dict, rule_name: str) -> tuple[str, str]:
    """Subject + body for a user-defined alert hit (English)."""
    subject = (f"[FireAgent Calendar] Custom alert ({hit['severity_name']}): "
               f"{rule_name} ({hit['hit_id']})")
    body = "\n".join([
        "FireAgent Calendar — user-defined custom alert",
        "=" * 50,
        "",
        f"Rule        : {rule_name} ({hit['rule_id']})",
        f"Severity    : {hit['severity_name']} (operator-chosen label, "
        "not a system severity)",
        f"Period      : {hit['period']} ({hit['metric']}/{hit['scope']})",
        f"Triggered   : {hit['triggered_at']}",
        "",
        "Trigger:",
        hit.get("summary", "unavailable"),
        "",
        "This is a USER-DEFINED threshold record, not a system-generated "
        "scientific alert. A threshold crossing is not proof of a wildfire.",
        "Caveat: satellite detections are not confirmed ground truth.",
    ])
    return subject, body


def send_pending_custom(conn, *, sender, at_utc: str | None = None,
                        max_attempts: int = 3) -> dict:
    """Deliver queued custom notifications (explicit operator action).

    Each queued row fans out to confirmed subscribers; a row is marked
    sent only when the sender reports success (idempotent per hit).
    Never sends when SMTP is unavailable.
    """
    from . import notify as notify_mod

    stamp = at_utc or _utcnow()
    stats = {"sent": 0, "failed": 0, "skipped": 0}
    if not notify_mod.smtp_status()["available"]:
        return {"sent": 0, "failed": 0, "skipped": 0,
                "reason": "SMTP unavailable: queue retained, nothing sent"}
    subs = [s for s in store_mod.list_subscriptions(conn, status="confirmed")]
    if not subs:
        return {"sent": 0, "failed": 0, "skipped": 0,
                "reason": "no confirmed subscribers: queue retained"}
    for row in store_mod.list_custom_notifications(conn, status="queued"):
        if row["attempts"] >= max_attempts:
            store_mod.set_custom_notification(
                conn, row["notif_id"], "failed",
                "retry budget exhausted", at_utc=stamp)
            stats["failed"] += 1
            continue
        hit = store_mod.get_custom_hit(conn, row["hit_id"])
        if hit is None:
            store_mod.set_custom_notification(
                conn, row["notif_id"], "failed", "hit no longer exists",
                at_utc=stamp)
            stats["failed"] += 1
            continue
        subject, body = render_custom_notification(hit, hit["rule_name"])
        ok_all = True
        for sub in subs:
            try:
                ok = bool(sender(sub["email"], subject, body))
            except Exception as exc:  # noqa: BLE001 - sanitized below
                logger.error("custom notify sender failure: %s",
                             notify_mod.redact(str(exc))[:200])
                ok = False
            ok_all = ok_all and ok
        if ok_all:
            store_mod.set_custom_notification(
                conn, row["notif_id"], "sent", "", at_utc=stamp)
            stats["sent"] += 1
        else:
            store_mod.bump_custom_notification(
                conn, row["notif_id"], at_utc=stamp)
            stats["failed"] += 1
    return stats
