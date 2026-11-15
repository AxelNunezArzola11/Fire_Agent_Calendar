"""Stage 5D: email subscriptions and notifications (notify_v1).

Public subscription workflow with NO accounts, login, passwords, roles,
or permissions:

- Subscribe with an email address -> pending + confirmation token.
- Confirm via the unguessable one-time token (48 h expiry) -> confirmed.
- Manage/unsubscribe via a separate unguessable token link.
- Only the minimum is stored (address, status, token hashes, prefs,
  timestamps). Token hashes only — tokens themselves never persist.
  Addresses, tokens, and secrets never appear in logs or errors.

Delivery reuses the proven email_alerts.py pattern (stdlib smtplib,
STARTTLS, fail-open when unconfigured) against CAL_SMTP_* settings, with
a per-subscription outbox: idempotent jobs, delivery history, bounded
retries (3 attempts), deduplication (alert fingerprints), cooldowns (no
repeat mail for unchanged alerts), unsubscribe enforcement, and honest
"skipped-not-configured" behavior when SMTP is absent. A notification is
marked delivered only when the transport reports success.

Policy (initial, configurable per subscription):
- Critical/High -> immediate email.
- Moderate/Low -> digest (default daily).
- Informational -> digest only when the subscription opts in.
"""

from __future__ import annotations

import hashlib
import logging
import re
import secrets
import smtplib
import time
import uuid
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from .. import settings
from .model import NOTIFY_VERSION, SEVERITY_LEVELS

from . import store as store_mod

logger = logging.getLogger(__name__)

TOKEN_BYTES = 32
CONFIRM_EXPIRY_H = 48
MAX_ATTEMPTS = 3
DIGEST_CHOICES = ("daily", "weekly")

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9.-]{1,253}\.[A-Za-z]{2,}$")

# Matches email-like substrings anywhere in free text (for redaction).
_EMAIL_FINDER = re.compile(r"[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9.-]{1,253}\.[A-Za-z]{2,}")

IMMEDIATE_SEVERITIES = ("Critical", "High")


def redact(text: str) -> str:
    """Redact email-like substrings (log hygiene for SMTP exceptions).

    `smtplib` errors can embed recipient addresses (e.g. recipient
    refused: user@example.com); truncation alone does not remove them.
    """
    return _EMAIL_FINDER.sub("[redacted-address]", str(text))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(moment: datetime | None = None) -> str:
    return (moment or _utcnow()).isoformat()


def new_token() -> str:
    """Unguessable URL-safe token (never logged, never stored raw)."""
    return secrets.token_urlsafe(TOKEN_BYTES)


def hash_token(token: str) -> str:
    """SHA-256 hex of a token (the only persisted form)."""
    return hashlib.sha256(token.encode()).hexdigest()


def normalize_email(email: str) -> str:
    """Validate and normalize a subscriber address (ValueError if bad)."""
    candidate = (email or "").strip().lower()
    if len(candidate) > 320 or not _EMAIL_RE.match(candidate):
        raise ValueError("invalid email address")
    return candidate


def default_prefs() -> dict:
    digest = settings.CAL_DIGEST_DEFAULT if settings.CAL_DIGEST_DEFAULT in \
        DIGEST_CHOICES else "daily"
    return {"digest": digest, "include_informational": False,
            "version": NOTIFY_VERSION}


# --- subscription workflow ---

def subscribe(conn, email: str) -> dict:
    """Create (or refresh) a pending subscription.

    Returns {sub_id, status, confirm_token, manage_token}. Tokens are
    returned ONLY here (to be emailed); afterwards only hashes exist.
    Re-subscribing an address issues fresh tokens and re-pends it unless
    already confirmed (confirmed stays confirmed; unsubscribed re-pends).
    """
    address = normalize_email(email)
    now = _stamp()
    current = store_mod.find_subscription_by_email(conn, address)
    if current is not None and current["status"] == "confirmed":
        return {"sub_id": current["sub_id"], "status": "confirmed",
                "confirm_token": None, "manage_token": None,
                "note": "already confirmed"}
    confirm_token = new_token()
    manage_token = new_token()
    record = {
        "sub_id": current["sub_id"] if current else uuid.uuid4().hex,
        "email": address,
        "status": "pending",
        "confirm_token_hash": hash_token(confirm_token),
        "confirm_expires_utc": _stamp(_utcnow() + timedelta(
            hours=CONFIRM_EXPIRY_H)),
        "manage_token_hash": hash_token(manage_token),
        "created_at": current["created_at"] if current else now,
        "confirmed_at": None,
        "unsubscribed_at": None,
        "prefs": current["prefs"] if current else default_prefs(),
    }
    store_mod.upsert_subscription(conn, record)
    logger.info("subscription %s pending confirmation", record["sub_id"][:8])
    return {"sub_id": record["sub_id"], "status": "pending",
            "confirm_token": confirm_token, "manage_token": manage_token}


def confirm(conn, token: str) -> dict:
    """Confirm a pending subscription (one-time token, expiry enforced)."""
    if not token:
        return {"status": "invalid", "reason": "empty token"}
    sub = store_mod.find_subscription_by_token_hash(
        conn, hash_token(token), "confirm")
    if sub is None:
        return {"status": "invalid", "reason": "unknown or consumed token"}
    if sub["status"] == "confirmed":
        return {"status": "confirmed", "sub_id": sub["sub_id"],
                "note": "already confirmed"}
    if sub["status"] == "unsubscribed":
        return {"status": "invalid",
                "reason": "subscription was cancelled; re-subscribe"}
    expires = sub.get("confirm_expires_utc") or ""
    if expires < _stamp():
        return {"status": "expired", "sub_id": sub["sub_id"],
                "reason": "confirmation token expired; re-subscribe"}
    updated = dict(sub)
    updated["status"] = "confirmed"
    updated["confirm_token_hash"] = None  # one-time: consumed
    updated["confirm_expires_utc"] = None
    updated["confirmed_at"] = _stamp()
    store_mod.upsert_subscription(conn, updated)
    logger.info("subscription %s confirmed", sub["sub_id"][:8])
    return {"status": "confirmed", "sub_id": sub["sub_id"]}


def unsubscribe(conn, manage_token: str) -> dict:
    """Cancel a subscription via its management token (idempotent)."""
    if not manage_token:
        return {"status": "invalid", "reason": "empty token"}
    sub = store_mod.find_subscription_by_token_hash(
        conn, hash_token(manage_token), "manage")
    if sub is None:
        return {"status": "invalid", "reason": "unknown management token"}
    if sub["status"] == "unsubscribed":
        return {"status": "unsubscribed", "sub_id": sub["sub_id"],
                "note": "already unsubscribed"}
    updated = dict(sub)
    updated["status"] = "unsubscribed"
    updated["unsubscribed_at"] = _stamp()
    store_mod.upsert_subscription(conn, updated)
    logger.info("subscription %s unsubscribed", sub["sub_id"][:8])
    return {"status": "unsubscribed", "sub_id": sub["sub_id"]}


def set_prefs(conn, manage_token: str, prefs: dict) -> dict:
    """Update digest preferences via the management token."""
    if not manage_token:
        return {"status": "invalid", "reason": "empty token"}
    sub = store_mod.find_subscription_by_token_hash(
        conn, hash_token(manage_token), "manage")
    if sub is None:
        return {"status": "invalid", "reason": "unknown management token"}
    merged = dict(sub["prefs"] or {})
    if "digest" in prefs:
        if prefs["digest"] not in DIGEST_CHOICES:
            return {"status": "invalid",
                    "reason": f"digest must be one of {DIGEST_CHOICES}"}
        merged["digest"] = prefs["digest"]
    if "include_informational" in prefs:
        merged["include_informational"] = bool(
            prefs["include_informational"])
    merged["version"] = NOTIFY_VERSION
    updated = dict(sub)
    updated["prefs"] = merged
    store_mod.upsert_subscription(conn, updated)
    return {"status": "ok", "sub_id": sub["sub_id"], "prefs": merged}


# --- SMTP transport (fail-open) ---

def smtp_status() -> dict:
    """Availability verdict (never exposes secrets or addresses)."""
    if settings.CAL_SMTP_ENABLED.lower() != "true":
        return {"available": False, "reason": "CAL_SMTP_ENABLED is not true"}
    if not (settings.CAL_SMTP_USER and settings.CAL_SMTP_PASSWORD):
        return {"available": False,
                "reason": "SMTP credentials are not configured"}
    return {"available": True,
            "reason": f"SMTP {settings.CAL_SMTP_HOST}:{settings.CAL_SMTP_PORT} "
            f"configured"}


def send_via_smtp(to_address: str, subject: str, body: str) -> bool:
    """Send one message; True only when the transport reports success."""
    status = smtp_status()
    if not status["available"]:
        logger.info("notify skipped-not-configured (%s)", status["reason"])
        return False
    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = settings.CAL_SMTP_FROM or settings.CAL_SMTP_USER
    message["To"] = to_address
    message.attach(MIMEText(body, "plain"))
    try:
        with smtplib.SMTP(settings.CAL_SMTP_HOST,
                          int(settings.CAL_SMTP_PORT)) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(settings.CAL_SMTP_USER, settings.CAL_SMTP_PASSWORD)
            smtp.sendmail(message["From"], [to_address],
                          message.as_string())
        return True
    except Exception as exc:  # noqa: BLE001 - transport failure, sanitized
        logger.error("notify smtp failure: %s", redact(str(exc))[:200])
        return False


# --- content (English, satellite-honest) ---

def manage_instructions(manage_token: str | None = None) -> str:
    lines = ["Manage this subscription: "]
    if settings.CAL_APP_URL:
        lines.append(f"open {settings.CAL_APP_URL} -> Events & Alerts -> "
                     f"Notifications and paste your management token.")
    else:
        lines.append("open the FireAgent Calendar app -> Events & Alerts -> "
                     "Notifications and paste your management token.")
    if manage_token:
        lines.append(f"Management token: {manage_token}")
    lines.append("Tokens are secret: do not share them.")
    return "\n".join(lines)


def confirmation_email(sub_id: str, confirm_token: str) -> tuple[str, str]:
    subject = "Confirm your FireAgent Calendar alert subscription"
    body = "\n".join([
        "FireAgent Calendar — email alert subscription",
        "=" * 50,
        "",
        "Confirm this subscription by opening the app (Events & Alerts ->",
        "Notifications) and pasting this confirmation token:",
        "",
        f"Confirmation token: {confirm_token}",
        "",
        f"This token expires in {CONFIRM_EXPIRY_H} hours and can be used "
        "once. If you did not request this subscription, ignore this email.",
        "",
        "Satellite fire observations are not confirmed wildfires; alerts "
        "describe observation activity, not ground truth.",
    ])
    return subject, body


def render_immediate(alert: dict, event: dict | None,
                     manage_token: str | None = None) -> tuple[str, str]:
    """Subject + body for one immediate (High/Critical) alert."""
    severity = alert["severity_name"]
    subject = (f"[FireAgent Calendar] {severity}: {alert['alert_type']} "
               f"({alert['alert_id']})")
    lines = [
        f"FireAgent Calendar alert — {alert['alert_type']}",
        "=" * 50,
        "",
        f"Severity    : {severity} (activity signal, not a danger rating)",
        f"Alert       : {alert['alert_id']}",
        f"Metric      : {alert['metric']} / scope {alert['scope']}",
        f"Period      : {alert['period']}",
    ]
    if event is not None:
        centroid = event.get("centroid") or {}
        lines += [
            f"Event       : {event['event_id']} ({event['lifecycle_state']})",
            f"Observed    : {event['first_observed_utc'][:10]} to "
            f"{event['last_observed_utc'][:10]}",
            f"Approx. loc.: lon {centroid.get('lon')}, "
            f"lat {centroid.get('lat')} (cluster centroid, not an ignition)",
            f"Sensors     : {event['sensor_modis_n']} MODIS, "
            f"{event['sensor_viirs_n']} VIIRS observations",
        ]
    lines += ["", "Evidence:", alert.get("explanation", "unavailable"), ""]
    if settings.CAL_APP_URL:
        lines += [f"View in the app: {settings.CAL_APP_URL}", ""]
    lines += [manage_instructions(manage_token),
              "",
              "Caveat: satellite detections are not confirmed ground truth."]
    return subject, "\n".join(lines)


def render_digest(alerts_list: list[dict], period_label: str,
                  manage_token: str | None = None) -> tuple[str, str]:
    """Subject + body for one digest batch (Moderate/Low/Info)."""
    subject = (f"[FireAgent Calendar] Activity digest ({period_label}): "
               f"{len(alerts_list)} alert(s)")
    lines = ["FireAgent Calendar activity digest", "=" * 50, "",
             f"Period: {period_label}",
             f"Alerts: {len(alerts_list)} (Moderate/Low severity "
             "activity signals)", ""]
    for alert in alerts_list:
        lines += [f"- [{alert['severity_name']}] {alert['alert_type']}: "
                  f"{alert['alert_id']} (period {alert['period']})"
                  + (f" event {alert['event_id']}"
                     if alert.get("event_id") else " (pilot-wide)")]
    lines += ["",
              "Caveat: satellite detections are not confirmed ground truth.",
              "", manage_instructions(manage_token)]
    if settings.CAL_APP_URL:
        lines.append(f"View in the app: {settings.CAL_APP_URL}")
    return subject, "\n".join(lines)


# --- notification policy + outbox processing ---

def wants_alert(sub: dict, alert: dict) -> bool:
    """Subscription-level policy gate (unsubscribed/pending get nothing)."""
    if sub["status"] != "confirmed":
        return False
    severity = alert["severity_name"]
    if severity not in SEVERITY_LEVELS:
        return False
    if severity in IMMEDIATE_SEVERITIES:
        return True
    if severity == "Informational":
        return bool((sub["prefs"] or {}).get("include_informational"))
    return True  # Moderate/Low via digest


def _notif_id(alert_id: str, sub_id: str, kind: str) -> str:
    return hashlib.sha256(
        f"{alert_id}|{sub_id}|{kind}".encode()).hexdigest()[:24]


def process_outbox(conn, *, kind: str = "immediate",
                   sender=send_via_smtp,
                   manage_tokens: dict[str, str] | None = None,
                   at_utc: str | None = None) -> dict:
    """Deliver due notifications of one kind (idempotent, bounded retry).

    - Immediate: every open Critical/High alert per confirmed subscriber.
    - Digest: every open Moderate/Low (+opt-in Informational) alert per
      confirmed subscriber, one email per subscriber.
    Re-runs never duplicate: a sent fingerprint is final; failures retry
    up to MAX_ATTEMPTS; unchanged alerts never re-mail.
    """
    if kind not in ("immediate", "digest"):
        raise ValueError(f"unknown outbox kind {kind!r}")
    stamp = at_utc or _stamp()
    manage_tokens = manage_tokens or {}
    stats = {"sent": 0, "failed": 0, "skipped": 0}
    subs = [s for s in store_mod.list_subscriptions(conn, status="confirmed")]
    for sub in subs:
        if kind == "immediate":
            due = [a for a in store_mod.list_alerts(conn, status="open",
                                                    limit=5000)
                   if a["severity_name"] in IMMEDIATE_SEVERITIES
                   and wants_alert(sub, a)]
            for alert in due:
                _deliver_one(conn, alert, sub, kind, sender,
                             manage_tokens.get(sub["sub_id"]), stamp, stats)
        else:
            due = [a for a in store_mod.list_alerts(conn, status="open",
                                                    limit=5000)
                   if a["severity_name"] not in IMMEDIATE_SEVERITIES
                   and wants_alert(sub, a)]
            if due:
                _deliver_digest(conn, due, sub, sender,
                                manage_tokens.get(sub["sub_id"]), stamp,
                                stats)
    return stats


def _prior(conn, notif_id: str):
    return store_mod.get_notification(conn, notif_id)


def _deliver_one(conn, alert, sub, kind, sender, manage_token, stamp,
                 stats) -> None:
    notif_id = _notif_id(alert["alert_id"], sub["sub_id"], kind)
    prior = _prior(conn, notif_id)
    if prior is not None and prior["status"] == "sent" and \
            prior["fingerprint"] == alert["fingerprint"]:
        stats["skipped"] += 1  # unchanged alert: never re-mail
        return
    if prior is not None and prior["status"] == "sent":
        # Materially changed evidence: new notification row.
        notif_id = f"{notif_id}-{alert['fingerprint'][:8]}"
        prior = None
    attempts = (prior["attempts"] if prior else 0) + 1
    if attempts > MAX_ATTEMPTS:
        store_mod.upsert_notification(conn, {
            "notif_id": notif_id, "alert_id": alert["alert_id"],
            "sub_id": sub["sub_id"], "kind": kind, "status": "failed",
            "attempts": attempts - 1,
            "last_error": "retry budget exhausted",
            "created_at": prior["created_at"] if prior else stamp,
            "sent_at": None, "fingerprint": alert["fingerprint"]})
        stats["failed"] += 1
        return
    event = None
    if alert.get("event_id"):
        event = store_mod.get_event(conn, alert["event_id"])
    subject, body = render_immediate(alert, event, manage_token)
    ok = False
    try:
        ok = bool(sender(sub["email"], subject, body))
    except Exception as exc:  # noqa: BLE001 - sender failure, sanitized
        logger.error("notify sender failure: %s", redact(str(exc))[:200])
    if ok:
        store_mod.upsert_notification(conn, {
            "notif_id": notif_id, "alert_id": alert["alert_id"],
            "sub_id": sub["sub_id"], "kind": kind, "status": "sent",
            "attempts": attempts, "last_error": None,
            "created_at": prior["created_at"] if prior else stamp,
            "sent_at": stamp, "fingerprint": alert["fingerprint"]})
        stats["sent"] += 1
    else:
        reason = ("skipped-not-configured"
                  if not smtp_status()["available"] and sender is send_via_smtp
                  else "failed")
        store_mod.upsert_notification(conn, {
            "notif_id": notif_id, "alert_id": alert["alert_id"],
            "sub_id": sub["sub_id"], "kind": kind,
            "status": reason if attempts >= MAX_ATTEMPTS else "queued",
            "attempts": attempts,
            "last_error": reason if sender is send_via_smtp
            else "sender reported failure",
            "created_at": prior["created_at"] if prior else stamp,
            "sent_at": None, "fingerprint": alert["fingerprint"]})
        if reason == "skipped-not-configured":
            stats["skipped"] += 1
        else:
            stats["failed"] += 1


def _deliver_digest(conn, due, sub, sender, manage_token, stamp, stats
                    ) -> None:
    digest_fp = hashlib.sha256("|".join(sorted(
        f"{a['alert_id']}:{a['fingerprint']}" for a in due)).encode()
    ).hexdigest()[:16]
    notif_id = _notif_id(f"digest-{digest_fp}", sub["sub_id"], "digest")
    if _prior(conn, notif_id) is not None:
        stats["skipped"] += 1  # identical digest already handled
        return
    period = time.strftime("%Y-%m-%d", time.gmtime())
    subject, body = render_digest(due, period, manage_token)
    ok = False
    try:
        ok = bool(sender(sub["email"], subject, body))
    except Exception as exc:  # noqa: BLE001 - sender failure, sanitized
        logger.error("notify sender failure: %s", redact(str(exc))[:200])
    store_mod.upsert_notification(conn, {
        "notif_id": notif_id,
        "alert_id": f"digest:{len(due)}",
        "sub_id": sub["sub_id"], "kind": "digest",
        "status": "sent" if ok else "queued",
        "attempts": 1, "last_error": None if ok else "pending",
        "created_at": stamp, "sent_at": stamp if ok else None,
        "fingerprint": digest_fp})
    stats["sent" if ok else "failed"] += 1
