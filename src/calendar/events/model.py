"""Phase 5 shared model: versions, severity scale, lifecycle states.

Single source of truth for the constants every Phase 5 stage shares, so
the event record, lifecycle engine, alert engine, notifications, and Qwen
layer can never disagree on vocabulary. Pure data + validation; no I/O.
"""

from __future__ import annotations

# --- Version pins (recorded on every event/alert/transition record) ---
MODEL_VERSION = "event_model_v1"
ASSOC_VERSION = "assoc_v1"
ASSOC_PARAMS_VERSION = "assoc_params_v1"
LIFECYCLE_VERSION = "lifecycle_v1"
LIFECYCLE_CFG_VERSION = "lifecycle_cfg_v1"
ALERT_RULES_VERSION = "alert_rules_v1"
STORE_VERSION = "event_store_v1"
NOTIFY_VERSION = "notify_v1"
QWEN_VERSION = "qwen_v1"

# --- Severity taxonomy (activity SIGNAL severity, never a danger rating) ---
# Exact scale required by the Phase 5 specification. This classifies the
# strength of the satellite-observation activity signal only: it is not a
# fire danger rating, emergency classification, evacuation risk, or any
# prediction of fire behavior or consequences.
SEVERITY_LEVELS: dict[str, int] = {
    "Informational": 0,
    "Low": 1,
    "Moderate": 2,
    "High": 3,
    "Critical": 4,
}
SEVERITY_NAMES: dict[int, str] = {v: k for k, v in SEVERITY_LEVELS.items()}
SEVERITY_SCALE_NOTE = (
    "Activity signal severity: strength of the satellite-observation "
    "activity signal. Not a fire danger rating, emergency classification, "
    "evacuation risk, or prediction of fire behavior or consequences."
)

# --- Lifecycle states (conservative; Resolved != extinguished) ---
STATE_ACTIVE = "Active"
STATE_MONITORING = "Monitoring"
STATE_PENDING = "Pending Closure"
STATE_RESOLVED = "Resolved"
LIFECYCLE_STATES: tuple[str, ...] = (
    STATE_ACTIVE, STATE_MONITORING, STATE_PENDING, STATE_RESOLVED,
)
RESOLVED_DISCLAIMER = (
    "Resolved means no qualifying satellite activity was observed for the "
    "configured quiet period with sufficient data. It does not mean the fire "
    "is contained or extinguished; containment/extinction require an "
    "authoritative external source."
)

# --- Alert types ---
ALERT_TYPES: tuple[str, ...] = (
    "Activity Spike",
    "Cluster Anomaly",
    "High FRP",
    "Historical Anomaly",
    "Data Quality",
)
ALERT_STATUS = ("open", "acknowledged", "resolved")

# --- Subscription states ---
SUB_STATUS = ("pending", "confirmed", "unsubscribed")


def check_severity(name: str) -> int:
    """Numeric level for a severity name (ValueError on unknown names)."""
    if name not in SEVERITY_LEVELS:
        raise ValueError(f"unknown severity {name!r}; want {sorted(SEVERITY_LEVELS)}")
    return SEVERITY_LEVELS[name]


def check_state(state: str) -> str:
    """Validate a lifecycle state (ValueError on unknown states)."""
    if state not in LIFECYCLE_STATES:
        raise ValueError(f"unknown lifecycle state {state!r}; want {list(LIFECYCLE_STATES)}")
    return state


def check_alert_type(alert_type: str) -> str:
    """Validate an alert type (ValueError on unknown types)."""
    if alert_type not in ALERT_TYPES:
        raise ValueError(f"unknown alert type {alert_type!r}; want {list(ALERT_TYPES)}")
    return alert_type
