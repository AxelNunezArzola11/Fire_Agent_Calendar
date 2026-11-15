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

from .. import settings
from ..acquisition import sensors
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

DYNAMIC_SCOPES_BY_METRIC: dict[str, list[str]] = {
    "detection": [
        "raw", "modis", "viirs-snpp", "viirs-noaa20", "viirs-noaa21",
        "Terra", "Aqua", "N", "NOAA-20", "NOAA-21",
    ],
    "frp": ["modis", "viirs-snpp", "viirs-noaa20", "viirs-noaa21"],
}

SCOPE_LABELS = {"raw": "Raw (all sensors)", "modis": "MODIS",
                "viirs-snpp": "VIIRS S-NPP",
                "viirs-noaa20": "VIIRS NOAA-20",
                "viirs-noaa21": "VIIRS NOAA-21",
                "Terra": "MODIS Terra",
                "Aqua": "MODIS Aqua", "N": "VIIRS S-NPP",
                "NOAA-20": "VIIRS NOAA-20",
                "NOAA-21": "VIIRS NOAA-21",
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
    # auto: preserve dates outside local archive span; follow multi-day shared range;
    # otherwise expand single-day pilot snapshots to the available pilot span.
    if s_shared < available_start or e_shared > available_end:
        return (s_shared, e_shared, "sidebar") if s_shared <= e_shared else (e_shared, s_shared, "sidebar")
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


def is_date_range_outside_local_pilot(
    date_start: str,
    date_end: str,
    min_avail: str = DEFAULT_HISTORY_START,
    max_avail: str = DEFAULT_HISTORY_END,
) -> bool:
    """True when either bound falls outside the precomputed local archive span."""
    return date_start < min_avail or date_end > max_avail


def _active_region_context() -> dict:
    import streamlit as st

    aoi_mode = st.session_state.get(filters.KEY_AOI_MODE, "full")
    resolved = None
    aoi_error = ""
    try:
        resolved = aoi_mod.resolve_session_aoi(
            st.session_state,
            aoi_path=settings.CAL_AOI_PATH,
            include_session_fallback=True,
        )
    except aoi_mod.AoIError as exc:
        aoi_error = str(exc)
    if resolved is not None:
        return {
            "mode": aoi_mode,
            "region_id": resolved.region_id,
            "label": resolved.label,
            "bbox": resolved.bbox,
            "source": resolved.source,
            "point_radius": resolved.point_radius,
            "aoi_error": "",
        }
    return {
        "mode": aoi_mode,
        "region_id": None,
        "label": "CONUS / USA Archive Extent",
        "bbox": None,
        "source": "none",
        "point_radius": None,
        "aoi_error": aoi_error,
    }


def _safe_line_chart(st_mod, data, **kwargs) -> None:
    fn = getattr(st_mod, "line_chart", None)
    if callable(fn):
        fn(data, **kwargs)


def _safe_bar_chart(st_mod, data, **kwargs) -> None:
    fn = getattr(st_mod, "bar_chart", None)
    if callable(fn):
        fn(data, **kwargs)


def _render_distribution_charts(df: pd.DataFrame, region_label: str) -> None:
    import streamlit as st

    from ..pipeline import regional

    if df is None or df.empty:
        return
    dist = regional.summarize_regional_distributions(df)
    by_sat = dist["by_satellite"]
    by_prod = dist["by_product"]
    by_dn = dist["by_daynight"]
    frp_bins = dist["frp_bins"]

    st.write(f"#### Satellite, Sensor & FRP Distributions ({region_label})")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Detections by Satellite & Sensor")
        if not by_sat.empty:
            _safe_bar_chart(st, by_sat, x="satellite", y=["detections"])
            st.dataframe(by_sat, width="stretch")
    with c2:
        st.caption("FRP Intensity Distribution (MW · MODIS & VIIRS separated)")
        if not frp_bins.empty:
            _safe_bar_chart(st, frp_bins, x="frp_bin", y=["modis", "viirs"])
        if not by_prod.empty:
            st.dataframe(by_prod, width="stretch")

    if not by_dn.empty:
        parts = [
            f"`{r['daynight']}`: {int(r['detections']):,} detections ({r['frp_sum']:,.1f} MW)"
            for _, r in by_dn.iterrows()
        ]
        st.caption("Day/Night split (satellite overpass flag): " + " · ".join(parts))


def _render_regional_archives_catalog() -> None:
    import streamlit as st

    from ..pipeline import regional

    archives = regional.list_regional_archives()
    if not archives:
        return
    with st.expander(f"Independent Regional DuckDB Archives ({len(archives)} stored)"):
        rows = [
            {
                "archive_id": a["archive_id"],
                "region": a["region_label"],
                "detections": a["n_detections"],
                "min_date": a["min_date"] or "—",
                "max_date": a["max_date"] or "—",
                "bbox": f"{a['bbox'][0]:.2f},{a['bbox'][1]:.2f},{a['bbox'][2]:.2f},{a['bbox'][3]:.2f}",
                "updated_utc": (a["updated_utc"] or "")[:19],
            }
            for a in archives
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch")


def _render_regional_analytics(
    region_ctx: dict,
    *,
    forced_window: tuple[str, str] | None = None,
) -> None:
    """Render dynamic multi-year/regional time-series analytics from DuckDB or live FIRMS."""
    import streamlit as st

    from ..pipeline import regional

    if region_ctx.get("aoi_error"):
        st.error(f"Area of interest error: {region_ctx['aoi_error']}")
        return

    live_ds = st.session_state.get("cal_live_regional")
    bbox = region_ctx["bbox"]
    point_radius = region_ctx.get("point_radius")
    c_lat, c_lon, r_km = (
        point_radius
        if point_radius is not None and len(point_radius) == 3
        else (None, None, None)
    )
    products = tuple(st.session_state.get(filters.KEY_PRODUCTS, list(query.PRODUCTS)))
    satellites = tuple(st.session_state.get(filters.KEY_SATELLITES, list(query.SATELLITES)))
    modis_conf = st.session_state.get(filters.KEY_MODIS_CONF, 0) or None
    viirs_conf = tuple(st.session_state.get(filters.KEY_VIIRS_CONF, list(query.VIIRS_CONFIDENCE)))
    firms_ready = bool(settings.get_setting("FIRMS_MAP_KEY", ""))

    # Keep dedicated dynamic start/end inputs synced when sidebar dates or forced_window change
    if forced_window:
        st.session_state["cal_an_reg_start"], st.session_state["cal_an_reg_end"] = forced_window
    else:
        st.session_state.setdefault("cal_an_reg_start", DEFAULT_HISTORY_START)
        st.session_state.setdefault("cal_an_reg_end", DEFAULT_HISTORY_END)
        filters.sync_inputs_from_range(
            st.session_state,
            start_key="cal_an_reg_start",
            end_key="cal_an_reg_end",
            sentinel_key="cal_an_reg_synced",
        )

    c_d1, c_d2 = st.columns(2)
    with c_d1:
        raw_start = st.text_input("Analytics Start (UTC)", key="cal_an_reg_start", placeholder="YYYY-MM-DD")
    with c_d2:
        raw_end = st.text_input("Analytics End (UTC)", key="cal_an_reg_end", placeholder="YYYY-MM-DD")
    try:
        d_start = filters.coerce_day(raw_start)
        d_end = filters.coerce_day(raw_end)
        if d_start > d_end:
            d_start, d_end = d_end, d_start
    except ValueError as exc:
        st.error(f"Invalid analytics date range: {exc}")
        return

    external_aoi = aoi_mod.is_external_aoi(
        bbox=bbox,
        region_id=region_ctx.get("region_id"),
        point_radius=point_radius,
    )
    archive_id = (
        regional.compute_archive_id(
            bbox,
            region_id=region_ctx.get("region_id"),
            point_radius=point_radius,
        )
        if bbox is not None
        else ""
    )
    outside_dates = is_date_range_outside_local_pilot(d_start, d_end)
    only_noaa_selected = bool(satellites) and not any(s in ("Terra", "Aqua", "N") for s in satellites)
    db_st = query.duckdb_status()

    live_matches = (
        live_ds is not None
        and (bbox is None or live_ds.bbox == bbox)
        and (not archive_id or not getattr(live_ds, "archive_id", "") or live_ds.archive_id == archive_id)
        and live_ds.date_start == d_start
        and live_ds.date_end == d_end
    )

    metric_opts = ("detection", "frp")
    metric = st.selectbox(
        "Metric (multi-year / regional)",
        options=list(metric_opts),
        key="cal_an_reg_metric",
        format_func=lambda m: {
            "detection": "Detection Activity",
            "frp": "Fire Radiative Power (MW)",
        }[m],
    )
    if live_matches and not live_ds.df.empty:
        avail_prods = sorted({str(p) for p in live_ds.df["product"].dropna().unique()})
        avail_sats = sorted({
            sensors.canonical_satellite(str(s), product=str(p))
            for s, p in zip(live_ds.df["satellite"], live_ds.df["product"])
            if pd.notna(s)
        })
        if metric == "detection":
            scope_opts = list(dict.fromkeys(["raw", *avail_prods, *avail_sats]))
        else:
            scope_opts = avail_prods or [p for p in products if p in DYNAMIC_SCOPES_BY_METRIC["frp"]] or list(DYNAMIC_SCOPES_BY_METRIC["frp"])
    else:
        scope_opts = list(DYNAMIC_SCOPES_BY_METRIC[metric])
    if st.session_state.get("cal_an_reg_scope") not in scope_opts:
        st.session_state["cal_an_reg_scope"] = scope_opts[0]
    scope = st.selectbox(
        "Scope (multi-year / regional)",
        options=scope_opts,
        key="cal_an_reg_scope",
        format_func=lambda s: SCOPE_LABELS.get(s, s),
    )
    grain = st.selectbox(
        "Grain (multi-year / regional)",
        options=["daily", "weekly", "monthly"],
        key="cal_an_reg_grain",
    )

    # Decide between pinned 2023 USA Stage 4C local archive vs independent regional DuckDB + NASA FIRMS
    use_regional_archive = (
        bbox is not None
        and (external_aoi or outside_dates or only_noaa_selected or not db_st["available"])
    )

    ds: regional.RegionalDataset | None = None
    refresh_requested = False
    if bbox is not None and firms_ready:
        refresh_requested = bool(
            st.button(
                f"Refresh Regional Archive from NASA FIRMS ({d_start} to {d_end})",
                key="cal_an_fetch_firms",
            )
        )

    if use_regional_archive and bbox is not None:
        if live_matches and not refresh_requested:
            ds = live_ds
        else:
            cached_ds = None
            if not refresh_requested:
                cached_ds, _cov = regional.load_regional_dataset(
                    archive_id,
                    bbox=bbox,
                    date_start=d_start,
                    date_end=d_end,
                    products=products,
                    satellites=satellites,
                    region_id=region_ctx.get("region_id") or region_ctx["mode"],
                    region_label=region_ctx["label"],
                    center_lat=c_lat,
                    center_lon=c_lon,
                    radius_km=r_km,
                    aoi_source=region_ctx.get("source") or region_ctx["mode"],
                )
            if cached_ds is not None and not refresh_requested:
                ds = cached_ds
            else:
                with st.spinner(
                    f"Populating independent regional DuckDB archive `{archive_id}` "
                    f"from NASA FIRMS ({d_start}..{d_end})…"
                ):
                    ds = regional.ensure_regional_archive_data(
                        bbox=bbox,
                        date_start=d_start,
                        date_end=d_end,
                        region_id=region_ctx.get("region_id") or region_ctx["mode"],
                        region_label=region_ctx["label"],
                        products=products,
                        satellites=satellites,
                        center_lat=c_lat,
                        center_lon=c_lon,
                        radius_km=r_km,
                        aoi_source=region_ctx.get("source") or region_ctx["mode"],
                        force_refresh=refresh_requested,
                    )
        if ds is not None and ds.data_origin != "unavailable":
            st.session_state["cal_live_regional"] = ds

        avail = sensors.evaluate_availability(
            d_start,
            d_end,
            products,
            satellites,
            firms_key_configured=firms_ready,
            region_label=region_ctx["label"],
            live_dataset=ds,
        )
        reg_db_path = regional.resolve_regional_db_path()
        if ds.data_origin == "firms_live_api":
            st.success(
                f"Data origin: **Newly retrieved NASA FIRMS API -> Independent Regional DuckDB Archive** · "
                f"Archive: `{ds.archive_id}` · Exec: `{ds.exec_id[:8]}` · Region: **{ds.region_label}** · "
                f"Window: `{d_start}` to `{d_end}` · Sources: `{', '.join(ds.sources) or 'none'}` · "
                f"{ds.n_detections:,} normalized detections stored in `{reg_db_path.name}`."
            )
        elif ds.data_origin == "regional_duckdb_archive":
            st.success(
                f"Data origin: **Independent Regional DuckDB Archive (Previously Archived)** · "
                f"Archive: `{ds.archive_id}` · Exec: `{ds.exec_id[:8]}` · Region: **{ds.region_label}** · "
                f"Window: `{d_start}` to `{d_end}` · {ds.n_detections:,} detections."
            )
        else:
            st.warning(
                f"Data origin: **Unavailable for `{ds.archive_id}`** — "
                + ("; ".join(ds.errors) if ds.errors else "No regional archive or live FIRMS data available.")
            )
            for note in avail["sensor_notes"]:
                st.info(note)
            _render_regional_archives_catalog()
            st.info(METHODOLOGY_NOTE)
            return

        if ds.errors:
            st.warning("; ".join(ds.errors))
        for note in avail["sensor_notes"]:
            st.info(note)

        filtered_df = query._filter_regional_df(
            ds.df,
            bbox=ds.bbox,
            products=products,
            satellites=satellites,
            modis_conf_min=modis_conf,
            viirs_conf=viirs_conf,
        )
        if filtered_df.empty:
            st.info(
                f"Regional archive `{ds.archive_id}` (**{ds.region_label}**) contains **0 matching detections** "
                f"across `{d_start}` to `{d_end}` (sources checked: `{', '.join(ds.sources) or 'none'}`). "
                "Zero observations are recorded honestly without backfilling from the Stage 4C USA baseline."
            )
            _render_regional_archives_catalog()
            st.info(METHODOLOGY_NOTE)
            return

        m_cols = st.columns(4)
        modis_part = filtered_df[filtered_df["product"].astype(str) == "modis"]
        viirs_part = filtered_df[filtered_df["product"].astype(str).str.startswith("viirs")]
        m_cols[0].metric("Filtered Observations", f"{len(filtered_df):,}")
        m_cols[1].metric("Active Days", f"{filtered_df['acquisition_timestamp_utc'].astype(str).str.slice(0, 10).nunique()}")
        m_cols[2].metric(
            "MODIS (Terra+Aqua)",
            f"{len(modis_part):,} ({pd.to_numeric(modis_part['frp'], errors='coerce').fillna(0).sum():,.1f} MW)",
        )
        m_cols[3].metric(
            "VIIRS (S-NPP/NOAA)",
            f"{len(viirs_part):,} ({pd.to_numeric(viirs_part['frp'], errors='coerce').fillna(0).sum():,.1f} MW)",
        )

        rows = query.query_regional_timeseries(
            date_start=d_start,
            date_end=d_end,
            bbox=ds.bbox,
            metric=metric,
            scope=scope,
            grain=grain,
            products=products,
            satellites=satellites,
            modis_conf_min=modis_conf,
            viirs_conf=viirs_conf,
            region_label=ds.region_label,
            extra_df=ds.df,
            archive_id=ds.archive_id,
            region_id=ds.region_id,
            allow_usa_baseline=False,
        )
        _render_charts_and_notables(rows, show_baseline_warning=False)
        _render_distribution_charts(filtered_df, ds.region_label)
        _render_regional_archives_catalog()
        return

    # USA 2023 baseline local DuckDB path
    avail = sensors.evaluate_availability(
        d_start,
        d_end,
        products,
        satellites,
        firms_key_configured=firms_ready,
        region_label=region_ctx["label"],
        live_dataset=live_ds if live_matches else None,
    )
    if refresh_requested and live_ds is not None and live_matches:
        st.info("Live NASA FIRMS refresh requested for USA AOI.")
    local_n = 0
    if db_st["available"] and not outside_dates and not only_noaa_selected and not external_aoi:
        local_n = query.count_detections(
            date_start=d_start,
            date_end=d_end,
            products=products,
            satellites=satellites,
            bbox=bbox,
            modis_conf_min=modis_conf,
            viirs_conf=viirs_conf,
        )

    if local_n == 0:
        if bbox is None and outside_dates:
            st.info(
                f"Window `{d_start}` to `{d_end}` is outside the pinned 2023 USA archive. "
                "Select a Geographic area of interest in the sidebar to retrieve global/multi-year NASA FIRMS data."
            )
        elif not db_st["available"]:
            st.warning(
                f"Local Stage 4C USA DuckDB archive unavailable ({db_st['status']}: {db_st['reason']}). "
                "Select a Geographic area of interest in the sidebar to query independent regional DuckDB + NASA FIRMS."
            )
        else:
            st.info(
                f"**{region_ctx['label']}** has **0 matching observations** in the pinned 2023 USA archive "
                f"for `{d_start}` to `{d_end}`."
            )
        _render_regional_archives_catalog()
        st.info(METHODOLOGY_NOTE)
        return

    st.caption(
        f"Data origin: **Pinned Stage 4C / Phase 1D USA DuckDB Archive** · **{region_ctx['label']}** · "
        f"{local_n:,} detections (`{d_start}` to `{d_end}`)."
    )
    rows = query.query_regional_timeseries(
        date_start=d_start,
        date_end=d_end,
        bbox=bbox,
        metric=metric,
        scope=scope,
        grain=grain,
        products=products,
        satellites=satellites,
        modis_conf_min=modis_conf,
        viirs_conf=viirs_conf,
        region_label=region_ctx["label"],
    )
    _render_charts_and_notables(rows, show_baseline_warning=False)
    _render_regional_archives_catalog()


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
        _safe_bar_chart(st, chart, x="period", y=single_cols)
        st.caption(
            "Single-period window shown as comparison bars. Widen the Historical Analytics "
            "date range above to display a multi-period time-series trend."
        )
    else:
        y_cols = ["observed", "baseline"] if base_valid > 0 else ["observed"]
        _safe_line_chart(st, chart, x="period", y=y_cols)
        st.caption(
            "Baseline gaps = periods without an eligible baseline (n < 3 prior slots; "
            "never drawn artificially)."
        )

    if z_valid > 0:
        st.write("#### Statistical Deviation (Z-Score)")
        _safe_bar_chart(st, chart, x="period", y=["z_score"])
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
    products = tuple(st.session_state.get(filters.KEY_PRODUCTS, list(query.PRODUCTS)))
    satellites = tuple(st.session_state.get(filters.KEY_SATELLITES, list(query.SATELLITES)))
    only_noaa_selected = bool(satellites) and not any(s in ("Terra", "Aqua", "N") for s in satellites)
    try:
        shared_s = filters.coerce_day(
            st.session_state.get(filters.KEY_RANGE_START, DEFAULT_HISTORY_START)
        )
        shared_e = filters.coerce_day(
            st.session_state.get(filters.KEY_RANGE_END, DEFAULT_HISTORY_END)
        )
    except ValueError:
        shared_s, shared_e = DEFAULT_HISTORY_START, DEFAULT_HISTORY_END
    sidebar_outside = is_date_range_outside_local_pilot(shared_s, shared_e)
    reg_s_ss = st.session_state.get("cal_an_reg_start")
    reg_e_ss = st.session_state.get("cal_an_reg_end")
    reg_outside = bool(
        reg_s_ss
        and reg_e_ss
        and is_date_range_outside_local_pilot(str(reg_s_ss), str(reg_e_ss))
    )
    needs_dynamic = has_regional or sidebar_outside or only_noaa_selected or reg_outside

    db_st = query.duckdb_status()
    firms_ready = bool(settings.get_setting("FIRMS_MAP_KEY", ""))
    dynamic_available = bool(db_st["available"] or firms_ready or live_ds is not None or has_regional)

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
            if dynamic_available and needs_dynamic:
                st.info("Multi-year / regional DuckDB & NASA FIRMS analytics remain available below.")
            else:
                st.info(METHODOLOGY_NOTE)
                return

    if records is None and anom_st is not None and anom_st["available"]:
        try:
            records = query.load_anomalies()
        except FileNotFoundError as exc:
            st.warning("Historical anomaly output unavailable.")
            st.info(f"{exc}. No anomalies are fabricated when Stage 4C outputs are absent.")
            if not (dynamic_available and needs_dynamic):
                st.info(METHODOLOGY_NOTE)
                return
            records = []

    if records or dynamic_available:
        source_choices = ["pilot", "regional"] if (records and dynamic_available) else (
            ["regional"] if dynamic_available else ["pilot"]
        )
        default_source = "regional" if (needs_dynamic or not records) and "regional" in source_choices else source_choices[0]
        prev_dynamic = st.session_state.get("cal_an_prev_dynamic")
        if KEY_AN_SOURCE not in st.session_state or st.session_state[KEY_AN_SOURCE] not in source_choices:
            st.session_state[KEY_AN_SOURCE] = default_source
        elif needs_dynamic and prev_dynamic is False and "regional" in source_choices:
            st.session_state[KEY_AN_SOURCE] = "regional"
        st.session_state["cal_an_prev_dynamic"] = needs_dynamic

        if len(source_choices) > 1:
            source_labels = {
                "pilot": "Pinned Stage 4C Baseline Archive (2023 USA · MODIS + VIIRS S-NPP)",
                "regional": f"Dynamic Multi-Year / Regional ({region_ctx['label']} · DuckDB + NASA FIRMS)",
            }
            chosen_source = st.radio(
                "Analytics scope",
                options=source_choices,
                key=KEY_AN_SOURCE,
                horizontal=True,
                format_func=lambda s: source_labels[s],
            )
        else:
            chosen_source = source_choices[0]

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
    st.caption("Sensor selection is intrinsic to scope in Stage 4C; switch to "
               "Dynamic Multi-Year / Regional above for NOAA-20/21, custom years, or AOI bounds.")
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
            "auto": f"Auto (Stage 4C baseline {min_avail}..{max_avail}, or sidebar dates when outside)",
            "full": f"Stage 4C baseline span ({min_avail} to {max_avail})",
            "sidebar": "Follow sidebar From/To dates",
            "custom": "Custom Historical Analytics dates (any year via FIRMS)",
        }[m],
    )
    if range_mode == "custom":
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("Analytics Start (UTC)", key=KEY_AN_START, placeholder="YYYY-MM-DD")
        with c2:
            st.text_input("Analytics End (UTC)", key=KEY_AN_END, placeholder="YYYY-MM-DD")

    try:
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

    if is_date_range_outside_local_pilot(date_start, date_end, min_avail, max_avail):
        st.info(
            f"Selected window (`{date_start}` to `{date_end}`) extends beyond the pinned "
            f"Stage 4C archive ({min_avail} to {max_avail}). Displaying multi-year "
            "DuckDB / NASA FIRMS analysis below."
        )
        _render_regional_analytics(region_ctx, forced_window=(date_start, date_end))
        return

    if resolved_reason == "auto-expanded":
        st.caption(
            f"Showing full Stage 4C baseline span `{date_start}` to `{date_end}` "
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
