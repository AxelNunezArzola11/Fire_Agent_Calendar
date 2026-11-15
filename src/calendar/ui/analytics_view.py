"""Historical Analytics view (Phase 4E): trends, baselines, anomalies.

Pure series helpers (unit-tested) plus a thin Streamlit render wrapper.
All figures come from the pinned 4C anomaly results; the UI computes no
statistics. Unavailable values render as "unavailable", never zero.
"""

from __future__ import annotations

from ..analytics import baselines, confidence
from . import filters, query

METHODOLOGY_NOTE = ("Statistical anomalies indicate unusual satellite "
                    "observation activity relative to an eligible historical "
                    "reference. They do not establish wildfire occurrence, "
                    "fire severity, or future fire danger.")
FRP_SCOPE_NOTE = ("FRP scopes are per-sensor sums of reported values; MODIS "
                  "and VIIRS FRP are not directly comparable.")
TRAILING_REFERENCE_NOTE = ("Trailing reference: strictly precedes the "
                           "evaluation period (no lookahead).")

SCOPES_BY_METRIC: dict[str, list[str]] = {}
for _metric, _scope in baselines.SERIES_SPECS:
    SCOPES_BY_METRIC.setdefault(_metric, []).append(_scope)

SCOPE_LABELS = {"raw": "Raw (all sensors)", "modis": "MODIS",
                "viirs-snpp": "VIIRS S-NPP", "Terra": "MODIS Terra",
                "Aqua": "MODIS Aqua", "N": "VIIRS S-NPP",
                "harmonized": "Harmonized", "count": "Cluster count",
                "viirs": "VIIRS detections in clusters"}

METHOD_LABELS = {"monthly_median": "Monthly Median (multi-year)",
                 "seasonal": "Seasonal, week-of-year (multi-year)",
                 "sensor_aware": "Sensor-Aware trailing"}

KEY_AN_METRIC = "cal_an_metric"
KEY_AN_SCOPE = "cal_an_scope"
KEY_AN_METHOD = "cal_an_method"
KEY_AN_GRAIN = "cal_an_grain"
KEY_AN_SELECTED = "cal_an_selected"

NOTABLE_CLASSES = ("notable-high", "notable-low", "provisional-notable-high",
                   "provisional-notable-low")


def series_for(records: list[dict], metric: str, scope: str,
               method: str, grain: str) -> list[dict]:
    """Ordered anomaly rows for one selection (period-sorted)."""
    rows = [r for r in records
            if r["metric"] == metric and r["scope"] == scope
            and r["baseline_method"] == method and r.get("grain") == grain]
    rows.sort(key=lambda r: r["eval_period"])
    return rows


def in_range(rows: list[dict], grain: str, start: str, end: str) -> list[dict]:
    """Rows whose period overlaps [start, end] (inclusive, UTC dates)."""
    kept = []
    for row in rows:
        try:
            lo, hi = filters.period_to_range(grain, row["eval_period"])
        except ValueError:
            continue
        if lo <= end and hi >= start:
            kept.append(row)
    return kept


def eligibility_counts(rows: list[dict]) -> dict[str, int]:
    """Classification histogram for one selection."""
    counts: dict[str, int] = {}
    for row in rows:
        key = row["classification"]
        counts[key] = counts.get(key, 0) + 1
    return counts


def notables(rows: list[dict]) -> list[dict]:
    """Notable rows (full + provisional evidence), period-ordered."""
    return sorted((r for r in rows if r["classification"] in NOTABLE_CLASSES),
                  key=lambda r: r["eval_period"])


def status_text(eligibility: str) -> str:
    """Human-readable eligibility (neutral language)."""
    return {"eligible": "Valid baseline",
            "provisional": "Provisional baseline (median only)",
            "unstable": "Unstable variability (z unavailable)",
            "insufficient-history": "Insufficient history"}.get(
                eligibility, f"Unknown ({eligibility})")


def _init_state() -> None:
    import streamlit as st

    st.session_state.setdefault(KEY_AN_METRIC, "detection")
    st.session_state.setdefault(KEY_AN_SCOPE, "raw")
    st.session_state.setdefault(KEY_AN_METHOD, "sensor_aware")
    st.session_state.setdefault(KEY_AN_GRAIN, "daily")
    st.session_state.setdefault(KEY_AN_SELECTED, "")


def _fmt_stat(value) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


def render_analytics_tab(records: list[dict]) -> None:
    """Full Historical Analytics tab (selectors + trend + notables + detail)."""
    import pandas as pd
    import streamlit as st

    _init_state()
    if st.session_state.get(filters.KEY_AOI_MODE, "full") != "full":
        st.info("Area of interest active (map only): historical analytics "
                "are pilot-wide (USA). Regional breakdowns are not part of "
                "the Phase 2 outputs.")
    metric = st.selectbox("Metric", options=sorted(SCOPES_BY_METRIC),
                          key=KEY_AN_METRIC,
                          format_func=lambda m: {"detection": "Detection Activity",
                                                 "cluster": "Hotspot Cluster Activity",
                                                 "frp": "Fire Radiative Power"}[m])
    scopes = SCOPES_BY_METRIC[metric]
    if st.session_state.get(KEY_AN_SCOPE) not in scopes:
        st.session_state[KEY_AN_SCOPE] = scopes[0]
    scope = st.selectbox("Scope", options=scopes, key=KEY_AN_SCOPE,
                         format_func=lambda s: SCOPE_LABELS.get(s, s))
    method = st.selectbox("Baseline method", options=list(baselines.METHODS),
                          key=KEY_AN_METHOD,
                          format_func=lambda m: METHOD_LABELS[m])
    grains = list(baselines.METHOD_GRAINS[method])
    if st.session_state.get(KEY_AN_GRAIN) not in grains:
        st.session_state[KEY_AN_GRAIN] = grains[0]
    grain = st.selectbox("Grain", options=grains, key=KEY_AN_GRAIN)
    st.caption("Sensor selection is intrinsic to scope here; sidebar sensor "
               "filters apply to the Map. The shared date range applies.")
    if metric == "frp":
        st.caption(FRP_SCOPE_NOTE)

    date_start = filters.coerce_day(st.session_state.get(filters.KEY_RANGE_START, "2023-01-01"))
    date_end = filters.coerce_day(st.session_state.get(filters.KEY_RANGE_END, "2023-12-31"))
    rows = in_range(series_for(records, metric, scope, method, grain),
                    grain, date_start, date_end)
    counts = eligibility_counts(rows)
    n_notable = sum(counts.get(c, 0) for c in NOTABLE_CLASSES)
    st.caption(f"{len(rows)} evaluated periods in range · {n_notable} notable · "
               f"{counts.get('typical', 0) + counts.get('provisional-typical', 0)} typical · "
               f"{counts.get('insufficient-evidence', 0)} insufficient evidence · "
               f"temporal basis: acquisition date (UTC)")
    if not rows or all(r["classification"] == "insufficient-evidence" for r in rows):
        st.warning("No eligible baselines for this selection on the 2023 "
                   "pilot: multi-year methods need prior cycles that do not "
                   "exist yet. This is insufficient history, not a software "
                   "failure. Try the Sensor-Aware trailing method.")
        st.info(METHODOLOGY_NOTE)
        return

    chart = pd.DataFrame({"period": [r["eval_period"] for r in rows],
                          "observed": [r["observed_value"] for r in rows],
                          "baseline": [r["baseline_value"] for r in rows]})
    st.line_chart(chart, x="period", y=["observed", "baseline"])
    st.caption("Baseline gaps = periods without an eligible baseline (never "
               "drawn artificially).")

    noted = notables(rows)
    if noted:
        from . import frames as frames_mod

        st.write("### Notable periods")
        st.dataframe(frames_mod.notables_frame(noted, _fmt_stat),
                     width="stretch")
        options = [r["eval_period"] for r in noted]
        if st.session_state.get(KEY_AN_SELECTED) not in options:
            st.session_state[KEY_AN_SELECTED] = options[0]
        picked = st.selectbox("Inspect period", options=options,
                              key=KEY_AN_SELECTED)
        detail = next(r for r in noted if r["eval_period"] == picked)
        _render_detail(detail)
    else:
        st.info("No notable anomalies in this selection and range "
                "(all evaluated periods are typical or provisional-typical).")
    st.info(METHODOLOGY_NOTE)


def _render_detail(record: dict) -> None:
    import streamlit as st

    st.write(f"### Anomaly {record['anomaly_id']}")
    cols = st.columns(4)
    cols[0].metric("Observed", _fmt_stat(record.get("observed_value")))
    cols[1].metric("Baseline", _fmt_stat(record.get("baseline_value")))
    cols[2].metric("Pct deviation", _fmt_stat(record.get("pct_deviation")) + "%"
                   if record.get("pct_deviation") is not None else "unavailable")
    cols[3].metric("Z-score", _fmt_stat(record.get("z_score")))
    st.write("Classification:", record["classification"],
             "| Quality:", status_text(record.get("eligibility", "")))
    st.write("Reference:", f"{record.get('reference_start')} to "
             f"{record.get('reference_end')} (n={record.get('reference_n')})",
             "| Method:", f"{record.get('baseline_method')} "
             f"({record.get('reference_strategy')})")
    if record.get("baseline_method") == "sensor_aware":
        st.caption(TRAILING_REFERENCE_NOTE)
    with st.expander("Confidence components"):
        comp = confidence.components(record)
        st.write({k: v for k, v in comp.items() if k != "disclaimer"})
        st.caption(comp["disclaimer"])
    with st.expander("Explanation", expanded=True):
        st.text(confidence.explain(record))
    st.caption(f"Provenance: anomaly exec `{record.get('exec_id', '')[:8]}` · "
               f"{record.get('algo_version')} / {record.get('classify_version')} / "
               f"{record.get('config_version')}.")
