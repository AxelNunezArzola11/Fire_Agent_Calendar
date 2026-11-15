"""Event report builder: one-click markdown evidence package.

Pure string assembly (unit-tested; no Streamlit, no store access).
Callers supply stored records; this module computes no analytics.
Output is byte-identical for identical inputs + generated_at.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..events.model import RESOLVED_DISCLAIMER
from .calendar_view import CITATIONS


def _fmt(value) -> str:
    """Missing-tolerant scalar (None -> 'unavailable', never blank)."""
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


def _join(items) -> str:
    if not items:
        return "unavailable"
    return ", ".join(str(v) for v in items)


def event_report_markdown(event: dict, *, transitions=(), alerts=(),
                          exec_chain=None, generated_at=None) -> str:
    """Full markdown report for one activity event (deterministic)."""
    stamp = generated_at or datetime.now(timezone.utc).isoformat()
    chain = exec_chain or {}
    frp = event.get("frp") or {}
    lines = [
        f"# FireAgent Calendar event report: {event.get('event_id', 'unavailable')}",
        f"Generated (UTC): {stamp}",
        "",
        "## Observation summary",
        f"- Lifecycle state: {_fmt(event.get('lifecycle_state'))}",
        f"- Severity: {_fmt(event.get('severity_name'))}",
        f"- Observations: {_fmt(event.get('detection_count'))} "
        f"(MODIS {_fmt(event.get('sensor_modis_n'))} / VIIRS "
        f"{_fmt(event.get('sensor_viirs_n'))})",
        f"- Satellites: {_join(event.get('satellites'))}",
        f"- Observed: {_fmt(event.get('first_observed_utc'))} to "
        f"{_fmt(event.get('last_observed_utc'))}",
        f"- Source clusters: {_join(event.get('cluster_ids'))}",
        f"- Severity basis: {_fmt(event.get('severity_provenance'))}",
        "",
        "## Fire Radiative Power (reported)",
        f"- FRP sum (MW): {_fmt(frp.get('frp_sum'))} | mean: "
        f"{_fmt(frp.get('frp_mean'))} | max: {_fmt(frp.get('frp_max'))}",
        f"- Contributing: {_fmt(frp.get('frp_n'))} | missing: "
        f"{_fmt(frp.get('frp_null'))} (excluded, never zero-filled)",
        "- Sums of reported per-observation FRP; repeated observations "
        "included, never total emitted energy. MODIS and VIIRS FRP are "
        "not directly comparable.",
        "",
        "## Lifecycle history",
    ]
    if transitions:
        for tr in transitions:
            lines.append(
                f"- {str(tr.get('at_utc', ''))[:10]}: "
                f"{tr.get('prev_state')} → {tr.get('new_state')} "
                f"({tr.get('reason')})")
    else:
        lines.append("- No transitions yet (seed state).")
    lines += ["", "## Linked alerts"]
    if alerts:
        for alert in alerts:
            lines.append(
                f"- {alert.get('alert_id')} [{alert.get('alert_type')} / "
                f"{alert.get('severity_name')} / {alert.get('status')}] "
                f"period {alert.get('period')}: "
                f"{alert.get('explanation')}")
    else:
        lines.append("- No linked alerts for this event.")
    lines += ["", "## Methods and provenance"]
    for stage in sorted(chain):
        stamp8 = str(chain[stage])[:8] if chain[stage] else "not built"
        lines.append(f"- Phase {stage}: exec `{stamp8}`")
    lines += [
        f"- Association record: {_fmt(event.get('provenance'))}",
        f"- Detection resolution: {_fmt(event.get('detection_resolution'))}",
        "",
        "## Limitations",
        "- Satellite observations are not confirmed wildfires; one fire "
        "can be observed many times.",
        "- Events group observations over time; they are not confirmed "
        "wildfires, not perimeters, and not incidents.",
        f"- {RESOLVED_DISCLAIMER}",
        "- Alert severity bands are retrospective percentiles over the "
        "available population, not validated real-time signals "
        "(Spike/High-FRP alerts).",
        "- Multi-year & multi-sensor support (local 2023 USA archive + live NASA FIRMS "
        "MODIS/VIIRS S-NPP/NOAA-20/NOAA-21); methods are deterministic and versioned, "
        "not autonomous agent judgments.",
        "",
        "## How to cite",
        *(f"- {citation}" for citation in CITATIONS),
    ]
    return "\n".join(lines) + "\n"
