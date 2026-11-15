"""Historical Analytics view (Phase 4E): trends, baselines, anomalies.

Pure series helpers (unit-tested) plus a thin Streamlit render wrapper.
Pinned Phase 4C results are read from verified Stage 4C execution outputs;
when a non-full geographic region or newly retrieved FIRMS dataset is active,
regional trailing series are computed deterministically over the observations
in that bounding box only (never mixing USA/California rows into other
regions). Unavailable values render as "unavailable", never zero.
"""

from __future__ import annotations

import pandas as pd

from ..analytics import baselines, confidence
from ..processing import regions
from . import aoi as aoi_mod
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
KEY_AN_RANGE_MODE = "cal_an_range_mode"
KEY_AN_START = "cal_an_start"
KEY_AN_END = "cal_an_end"
KEY_AN_SOURCE = "cal_an_source"

NOTABLE_CLASSES = ("notable-high", "notable-low", "provisional-notable-high",
                   "provisional-notable-low")

DEFAULT_HISTORY_START = "2023-01-01"
DEFAULT_HISTORY_END = "2023-12-31"


def series_for(records: list[dict], metric: str, scope: str,
               method: str, grain: str) -> list[dict]:
    """Ordered anomaly rows for one selection (period-sorted)."""
    rows = [r for r in records
            if r.get("metric") == metric and r.get("scope") == scope
            and r.get("baseline_method") == method and r.get("grain") == grain]
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


def prepare_chart_frame(rows: list[dict]) -> pd.DataFrame:
    """Build a strictly typed float64 DataFrame for Streamlit trend charts.

    Fixes Pandas 3.x object-dtype failures when baseline_value contains None
    and ensures chronological ordering by eval_period.
    """
    if not rows:
        return pd.DataFrame(
            columns=["period", "observed", "baseline", "z_score", "pct_deviation"]
        )
    ordered = sorted(rows, key=lambda r: str(r.get("eval_period", "")))
    frame = pd.DataFrame(
        {
            "period": [str(r.get("eval_period", "")) for r in ordered],
            "observed": pd.to_numeric(
                [r.get("observed_value") for r in ordered], errors="coerce"
            ).astype("float64"),
            "baseline": pd.to_numeric(
                [r.get("baseline_value") for r in ordered], errors="coerce"
            ).astype("float64"),
            "z_score": pd.to_numeric(
                [r.get("z_score") for r in ordered], errors="coerce"
            ).astype("float64"),
            "pct_deviation": pd.to_numeric(
                [r.get("pct_deviation") for r in ordered], errors="coerce"
            ).astype("float64"),
        }
    )
    return frame


def resolve_analytics_window(
    shared_start: str,
    shared_end: str,
    *,
    mode: str = "auto",
    custom_start: str | None = None,
    custom_end: str | None = None,
    available_start: str = DEFAULT_HISTORY_START,
    available_end: str = DEFAULT_HISTORY_END,
) -> tuple[str, str, str]:
    """Resolve the effective Historical Analytics date window.

    When the shared sidebar filter is still at a single daily snapshot (e.g.,
    ``INITIAL_DAY == '2023-08-01'``), ``auto`` mode automatically expands to
    the full historical dataset span so line charts render multi-period
    trends instead of a blank 1-point series.
    """
    if mode == "custom" and custom_start and custom_end:
        s = filters.coerce_day(custom_start)
        e = filters.coerce_day(custom_end)
        return (s, e, "custom") if s <= e else (e, s, "custom")
    if mode == "full":
        return (available_start, available_end, "full")
    s_shared = filters.coerce_day(shared_start)
    e_shared = filters.coerce_day(shared_end)
    if mode == "sidebar":
        return (s_shared, e_shared, "sidebar") if s_shared <= e_shared else (e_shared, s_shared, "sidebar")
    # auto: follow shared range when user picked a multi-day window; expand single-day map snapshots
    if s_shared < e_shared:
        return s_shared, e_shared, "sidebar"
    return available_start, available_end, "auto-expanded"


def _init_state() -> None:
    import streamlit as st

    st.session_state.setdefault(KEY_AN_METRIC, "detection")
    st.session_state.setdefault(KEY_AN_SCOPE, "raw")
    st.session_state.setdefault(KEY_AN_METHOD, "sensor_aware")
    st.session_state.setdefault(KEY_AN_GRAIN, "daily")
    st.session_state.setdefault(KEY_AN_SELECTED, "")
    st.session_state.setdefault(KEY_AN_RANGE_MODE, "auto")
    st.session_state.setdefault(KEY_AN_START, DEFAULT_HISTORY_START)
    st.session_state.setdefault(KEY_AN_END, DEFAULT_HISTORY_END)


def _fmt_stat(value) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


def _active_region_context() -> dict:
    import streamlit as st

    from .. import settings as _settings

    aoi_mode = st.session_state.get(filters.KEY_AOI_MODE, "full")
    reg_id = st.session_state.get(filters.KEY_AOI_REGION, "california")
    label = "Full 2023 Pilot Archive (United States)"
    bbox = None
    try:
        bbox = aoi_mod.resolve_bbox(
            mode=aoi_mode,
            region_id=reg_id,
            custom=(
                st.session_state.get(filters.KEY_AOI_W),
                st.session_state.get(filters.KEY_AOI_S),
                st.session_state.get(filters.KEY_AOI_E),
                st.session_state.get(filters.KEY_AOI_N),
            ),
            viewport=st.session_state.get("cal_map_bbox"),
            saved=(
                aoi_mod.load_aois(_settings.CAL_AOI_PATH).get(
                    st.session_state.get(filters.KEY_AOI_SAVED, "")
                )
                or None
            ),
        )
    except aoi_mod.AoIError:
        bbox = None
    if aoi_mode == "region":
        label = regions.region_label(reg_id)
    elif aoi_mode == "custom" and bbox:
        label = f"Custom bbox ({bbox[0]:.2f}..{bbox[2]:.2f}, {bbox[1]:.2f}..{bbox[3]:.2f})"
    elif aoi_mode == "viewport" and bbox:
        label = f"Map viewport ({bbox[0]:.2f}..{bbox[2]:.2f}, {bbox[1]:.2f}..{bbox[3]:.2f})"
    elif aoi_mode == "saved" and bbox:
        label = f"Saved AOI: {st.session_state.get(filters.KEY_AOI_SAVED, 'custom')}"
    return {"mode": aoi_mode, "region_id": reg_id, "label": label, "bbox": bbox}


def _render_regional_analytics(region_ctx: dict) -> None:
    """Render regional time-series analytics from local DuckDB or retrieved live FIRMS data."""
    import streamlit as st

    from ..pipeline import regional

    live_ds = st.session_state.get("cal_live_regional")
    bbox = region_ctx["bbox"]
    live_matches = (
        live_ds is not None
        and live_ds.bbox == bbox
    )

    metric_opts = ("detection", "frp")
    metric = st.selectbox(
        "Metric (regional series)",
        options=list(metric_opts),
        key="cal_an_reg_metric",
        format_func=lambda m: {
            "detection": "Detection Activity",
            "frp": "Fire Radiative Power (MW)",
        }[m],
    )
    if live_matches:
        avail_prods = sorted({str(p) for p in live_ds.df["product"].dropna().unique()})
        scope_opts = (
            ["raw", *avail_prods]
            if metric == "detection"
            else (avail_prods or ["modis", "viirs-snpp"])
        )
    else:
        scope_opts = (
            ["raw", "modis", "viirs-snpp", "Terra", "Aqua", "N"]
            if metric == "detection"
            else ["modis", "viirs-snpp"]
        )
    scope = st.selectbox(
        "Scope (regional series)",
        options=scope_opts,
        key="cal_an_reg_scope",
        format_func=lambda s: SCOPE_LABELS.get(s, s),
    )
    grain = st.selectbox(
        "Grain (regional series)",
        options=["daily", "weekly", "monthly"],
        key="cal_an_reg_grain",
    )

    if live_matches:
        d_start, d_end = live_ds.date_start, live_ds.date_end
        st.success(
            f"Using newly retrieved NASA FIRMS observations for **{live_ds.region_label}** "
            f"(`{d_start}` to `{d_end}`, exec `{live_ds.exec_id[:8]}`, "
            f"{live_ds.n_detections:,} normalized detections across {', '.join(live_ds.sources)})."
        )
        rows = regional.build_regional_timeseries(
            live_ds.df,
            metric=metric,
            scope=scope,
            grain=grain,
            date_start=d_start,
            date_end=d_end,
            region_label=live_ds.region_label,
            exec_id=live_ds.exec_id,
        )
    else:
        shared_s = filters.coerce_day(
            st.session_state.get(filters.KEY_RANGE_START, DEFAULT_HISTORY_START)
        )
        shared_e = filters.coerce_day(
            st.session_state.get(filters.KEY_RANGE_END, DEFAULT_HISTORY_END)
        )
        d_start, d_end, _ = resolve_analytics_window(
            shared_s,
            shared_e,
            mode=st.session_state.get(KEY_AN_RANGE_MODE, "auto"),
            custom_start=st.session_state.get(KEY_AN_START),
            custom_end=st.session_state.get(KEY_AN_END),
        )
        db_st = query.duckdb_status()
        if not db_st["available"]:
            st.warning(f"Local DuckDB geospatial archive unavailable ({db_st['status']}): {db_st['reason']}")
            return
        local_n = query.count_detections(date_start=d_start, date_end=d_end, bbox=bbox)
        if local_n == 0:
            st.info(
                f"**{region_ctx['label']}** has **0 recorded observations** in the local 2023 "
                f"USA pilot archive for `{d_start}` to `{d_end}` (lon `{bbox[0]:.2f}..{bbox[2]:.2f}`, "
                f"lat `{bbox[1]:.2f}..{bbox[3]:.2f}`). USA observations are never reused for other regions."
            )
            st.caption(
                "To analyze detections outside the 2023 USA pilot archive, use the "
                "Live NASA FIRMS retrieval control in the Activity Map or Pipeline section."
            )
            st.info(METHODOLOGY_NOTE)
            return
        st.caption(
            f"Computing regional trailing series for **{region_ctx['label']}** over {local_n:,} local "
            f"DuckDB detections (`{d_start}` to `{d_end}`, `{region_ctx['bbox'][0]:.2f}..{region_ctx['bbox'][2]:.2f}`, "
            f"`{region_ctx['bbox'][1]:.2f}..{region_ctx['bbox'][3]:.2f}`)."
        )
        rows = query.query_regional_timeseries(
            date_start=d_start,
            date_end=d_end,
            bbox=bbox,
            metric=metric,
            scope=scope,
            grain=grain,
            products=tuple(st.session_state.get(filters.KEY_PRODUCTS, list(query.PRODUCTS))),
            satellites=tuple(st.session_state.get(filters.KEY_SATELLITES, list(query.SATELLITES))),
            region_label=region_ctx["label"],
        )

    _render_charts_and_notables(rows, show_baseline_warning=False)


def _render_charts_and_notables(
    rows: list[dict],
    *,
    show_baseline_warning: bool = True,
) -> None:
    import streamlit as st

    counts = eligibility_counts(rows)
    n_notable = sum(counts.get(c, 0) for c in NOTABLE_CLASSES)
    n_typical = counts.get("typical", 0) + counts.get("provisional-typical", 0)
    n_insuff = counts.get("insufficient-evidence", 0)
    st.caption(
        f"{len(rows)} evaluated periods in range · {n_notable} notable · "
        f"{n_typical} typical · {n_insuff} insufficient evidence · "
        "temporal basis: acquisition date (UTC)"
    )

    if not rows:
        st.info(
            "Zero evaluated periods exist in this date range and selection. "
            "Try expanding the Historical Analytics date range."
        )
        st.info(METHODOLOGY_NOTE)
        return

    chart = prepare_chart_frame(rows)
    obs_valid = int(chart["observed"].notna().sum())
    base_valid = int(chart["baseline"].notna().sum())
    z_valid = int(chart["z_score"].notna().sum())

    if obs_valid == 0:
        st.info(
            "No numeric observations are available to plot in this selection "
            "(empty observation state, never zero-filled)."
        )
        st.info(METHODOLOGY_NOTE)
        return

    if show_baseline_warning and base_valid == 0:
        st.warning(
            "No eligible historical baselines for this method/selection on single-year data "
            "(multi-year methods need prior cycles; insufficient history, not a software failure). "
            "Observed series is still plotted below; switch to **Sensor-Aware trailing** for baselines."
        )

    st.write("#### Observed Activity & Historical Baseline")
    if len(chart) == 1:
        # Single-period selection: render bar metrics so one point is never an invisible line
        single_cols = ["observed"] + (["baseline"] if base_valid > 0 else [])
        st.bar_chart(chart, x="period", y=single_cols)
        st.caption(
            "Single-period window shown as comparison bars. Widen the Historical Analytics "
            "date range above to display a multi-period time-series trend."
        )
    else:
        y_cols = ["observed", "baseline"] if base_valid > 0 else ["observed"]
        st.line_chart(chart, x="period", y=y_cols)
        st.caption(
            "Baseline gaps = periods without an eligible baseline (n < 3 prior slots; "
            "never drawn artificially)."
        )

    if z_valid > 0:
        st.write("#### Statistical Deviation (Z-Score)")
        if len(chart) == 1:
            st.bar_chart(chart, x="period", y=["z_score"])
        else:
            st.bar_chart(chart, x="period", y=["z_score"])
        st.caption(
            "Z-score shown only when n >= 7 prior periods with non-zero variance "
            "(classify_v1 notable gate: |z| >= 2.0 and |pct| >= 25%)."
        )

    noted = notables(rows)
    if noted:
        from . import frames as frames_mod

        st.write("### Notable periods")
        st.dataframe(frames_mod.notables_frame(noted, _fmt_stat), width="stretch")
        options = [r["eval_period"] for r in noted]
        if st.session_state.get(KEY_AN_SELECTED) not in options:
            st.session_state[KEY_AN_SELECTED] = options[0]
        picked = st.selectbox("Inspect period", options=options, key=KEY_AN_SELECTED)
        detail = next(r for r in noted if r["eval_period"] == picked)
        _render_detail(detail)
    else:
        st.info(
            "No notable anomalies in this selection and range "
            "(all evaluated periods with eligible baselines are typical or provisional-typical)."
        )
    st.info(METHODOLOGY_NOTE)


def render_analytics_tab(records: list[dict] | None = None) -> None:
    """Full Historical Analytics tab (selectors + trend + notables + detail)."""
    import streamlit as st

    _init_state()
    anom_st = query.artifact_status("4C", "anomalies_4c.json") if records is None else None
    region_ctx = _active_region_context()
    live_ds = st.session_state.get("cal_live_regional")
    has_regional = (region_ctx["bbox"] is not None) or (live_ds is not None)

    if records is None and anom_st is not None:
        run_meta = anom_st.get("run_meta") or {}
        if anom_st["available"]:
            completed_txt = run_meta.get("completed_at", "")
            st.caption(
                f"Pinned Phase 4C execution `{anom_st['exec_id'][:8]}` · "
                f"registry status: **{anom_st.get('run_status', 'verified')}**"
                + (f" ({completed_txt[:19]} UTC)" if completed_txt else "")
            )
        else:
            st.warning(f"Historical anomaly output unavailable ({anom_st['status']}).")
            st.info(
                f"Required Phase 4C analytical artifact is missing or incomplete in this deployment.\n\n"
                f"- Artifact path: `{anom_st['path'] or 'anomalies_4c.json'}`\n"
                f"- Execution ID (`{anom_st['env_var']}`): "
                f"`{anom_st['exec_id'] or '(unconfigured)'}` "
                f"(run status: `{anom_st.get('run_status', 'unknown')}`)\n"
                f"- Detail: {anom_st['reason']}\n"
                f"- How to configure: {anom_st['remediation']}\n\n"
                "No anomaly scores or baselines are fabricated when Stage 4C outputs are absent or incomplete."
            )
            if has_regional:
                st.info("Regional analytics mode remains available below for the selected region.")
            else:
                st.info(METHODOLOGY_NOTE)
                return

    if records is None and anom_st is not None and anom_st["available"]:
        try:
            records = query.load_anomalies()
        except FileNotFoundError as exc:
            st.warning("Historical anomaly output unavailable.")
            st.info(f"{exc}. No anomalies are fabricated when Stage 4C outputs are absent.")
            if not has_regional:
                st.info(METHODOLOGY_NOTE)
                return
            records = []

    if has_regional:
        source_choices = ["regional", "pilot"] if records else ["regional"]
        source_labels = {
            "regional": f"Selected region ({region_ctx['label']})",
            "pilot": "Pinned 2023 USA pilot archive (Stage 4C)",
        }
        chosen_source = st.radio(
            "Analytics scope",
            options=source_choices,
            key=KEY_AN_SOURCE,
            horizontal=True,
            format_func=lambda s: source_labels[s],
        )
        if chosen_source == "regional":
            _render_regional_analytics(region_ctx)
            return

    if not records:
        st.info("No Phase 4C historical anomaly records are loaded.")
        st.info(METHODOLOGY_NOTE)
        return

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
               "filters apply to the Map.")
    if metric == "frp":
        st.caption(FRP_SCOPE_NOTE)

    all_series_rows = series_for(records, metric, scope, method, grain)
    min_avail = DEFAULT_HISTORY_START
    max_avail = DEFAULT_HISTORY_END
    if all_series_rows:
        try:
            min_avail = filters.period_to_range(grain, all_series_rows[0]["eval_period"])[0]
            max_avail = filters.period_to_range(grain, all_series_rows[-1]["eval_period"])[1]
        except ValueError:
            pass

    range_mode = st.selectbox(
        "Historical date window",
        options=["auto", "full", "sidebar", "custom"],
        key=KEY_AN_RANGE_MODE,
        format_func=lambda m: {
            "auto": f"Auto (full span {min_avail}..{max_avail} when sidebar is a single day)",
            "full": f"Full dataset span ({min_avail} to {max_avail})",
            "sidebar": "Follow sidebar From/To dates",
            "custom": "Custom Historical Analytics dates",
        }[m],
    )
    if range_mode == "custom":
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("Analytics Start (UTC)", key=KEY_AN_START, placeholder="YYYY-MM-DD")
        with c2:
            st.text_input("Analytics End (UTC)", key=KEY_AN_END, placeholder="YYYY-MM-DD")

    try:
        shared_s = filters.coerce_day(st.session_state.get(filters.KEY_RANGE_START, DEFAULT_HISTORY_START))
        shared_e = filters.coerce_day(st.session_state.get(filters.KEY_RANGE_END, DEFAULT_HISTORY_END))
        date_start, date_end, resolved_reason = resolve_analytics_window(
            shared_s,
            shared_e,
            mode=range_mode,
            custom_start=st.session_state.get(KEY_AN_START),
            custom_end=st.session_state.get(KEY_AN_END),
            available_start=min_avail,
            available_end=max_avail,
        )
    except ValueError as exc:
        st.error(f"Invalid analytics date range: {exc}")
        return

    if resolved_reason == "auto-expanded":
        st.caption(
            f"Showing full historical span `{date_start}` to `{date_end}` "
            f"(sidebar is set to single daily snapshot `{shared_s}`)."
        )
    else:
        st.caption(f"Analytics window: `{date_start}` to `{date_end}` ({resolved_reason}).")

    rows = in_range(all_series_rows, grain, date_start, date_end)
    _render_charts_and_notables(rows, show_baseline_warning=True)


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
