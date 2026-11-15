"""Fire Calendar rendering (Phase 3B).

Pure HTML builders (unit-tested) plus thin Streamlit `render_*` wrappers.
All figures come from the pinned Phase 2D metrics document — this module
computes no analytics. Color shows observation density only, never danger.
"""

from __future__ import annotations

from datetime import date

from . import filters, query

METHODOLOGY_NOTE = ("Satellite observations and analytical hotspot clusters "
                    "are not equivalent to confirmed individual wildfires.")
INTENSITY_CAPTION = ("Cell color shows the observation count for that day; "
                     "it does not indicate fire danger or severity.")
TEMPORAL_BASIS_LABEL = "Temporal basis: satellite acquisition date (UTC)."
ZERO_NA_LEGEND = ("0 means zero recorded observations inside verified "
                  "coverage; n/a means outside verified pilot coverage "
                  "(no data, not zero fire activity).")

# Single source of truth for citation copy (About tab + export sidecar).
CITATIONS = (
    "NASA FIRMS (Fire Information for Resource Management System), "
    "MODIS Collection 6.1 and VIIRS 375 m V2 active fire products via "
    "NASA LANCE (https://firms.modaps.eosdis.nasa.gov).",
    "FireAgent Calendar 2023 USA pilot: 482,672 accepted Standard "
    "detections; methods metrics_v1, matcher_v1, cluster_v1, "
    "dual_index_v1, baselines_v1, anomalies_v1 (pinned exec ids in "
    "About / export sidecar).",
    "Sofiev, M. and Kouznetsov, R. (2026). Observation operator and "
    "detection limits for MODIS and VIIRS Fire Radiative Power "
    "products. Geoscientific Model Development, 19, 10123-10142. "
    "https://doi.org/10.5194/gmd-19-10123-2026 (FRP observation "
    "science reference; research preview only, not used in "
    "production analytics).",
)

# Single-hue sequential scale (light -> dark amber). Steps, not a gradient,
# so adjacent levels stay distinguishable without implying severity bands.
INTENSITY_STEPS = ("#fff7e6", "#ffe8bf", "#ffd27a", "#ffab2e", "#e07b00")
UNAVAILABLE_CELL = "#f0f0f0"


def intensity_color(value: float, vmax: float) -> str:
    """Map a non-negative value onto the 5-step intensity scale."""
    if vmax <= 0 or value <= 0:
        return INTENSITY_STEPS[0]
    ratio = min(1.0, value / vmax)
    return INTENSITY_STEPS[min(4, int(ratio * 5))]


def _cell(day_num: int, text: str, bg: str, selected: bool = False,
          title: str = "") -> str:
    border = "3px solid #1a1a1a" if selected else "1px solid #d9d9d9"
    return (f'<td title="{title}" style="background:{bg};border:{border};'
            f'border-radius:6px;padding:6px 4px;text-align:center;min-width:64px">'
            f'<div style="font-weight:600;font-size:14px">{day_num}</div>'
            f'<div style="font-size:12px;color:#333">{text}</div></td>')


def _grid_table(headers: list[str], rows: list[str], caption: str = "") -> str:
    head = "".join(f"<th style='padding:4px;color:#555;font-weight:600'>"
                   f"{h}</th>" for h in headers)
    body = "".join(f"<tr>{r}</tr>" for r in rows)
    cap = f"<caption style='caption-side:bottom;padding:6px;color:#666'>{caption}</caption>" if caption else ""
    return (f"<table style='border-collapse:separate;border-spacing:4px'>{cap}"
            f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")


def month_grid_html(year: int, month: int, values: dict[str, int | None],
                    selected: str | None = None) -> str:
    """Month grid; None value = unavailable (outside verified coverage)."""
    present = [v for v in values.values() if v is not None]
    vmax = max(present) if present else 0
    rows = []
    for week in filters.month_weeks(year, month):
        cells = []
        for day in week:
            if day is None:
                cells.append("<td></td>")
                continue
            key = day.isoformat()
            val = values.get(key)
            if val is None:
                cells.append(_cell(day.day, "n/a", UNAVAILABLE_CELL,
                                    title="Outside verified pilot coverage"))
            else:
                cells.append(_cell(
                    day.day, f"{val:,}",
                    intensity_color(val, vmax), selected=(key == selected),
                    title=f"{key}: {val:,}"))
        rows.append("".join(cells))
    return _grid_table(list(filters.WEEKDAY_HEADERS), rows, INTENSITY_CAPTION)


def week_strip_html(monday: date, values: dict[str, int | None],
                    selected: str | None = None) -> str:
    """7-day strip for the ISO week starting Monday."""
    days = [monday.fromordinal(monday.toordinal() + i) for i in range(7)]
    present = [values.get(d.isoformat()) for d in days]
    vmax = max((v for v in present if v is not None), default=0)
    cells = []
    for day in days:
        val = values.get(day.isoformat())
        if val is None:
            cells.append(_cell(day.day, "n/a", UNAVAILABLE_CELL,
                                title="Outside verified pilot coverage"))
        else:
            cells.append(_cell(day.day, f"{val:,}",
                                intensity_color(val, vmax),
                                selected=(day.isoformat() == selected),
                                title=f"{day.isoformat()}: {val:,}"))
    headers = [f"{filters.WEEKDAY_HEADERS[i]}<br>{d.isoformat()[5:]}"
               for i, d in enumerate(days)]
    return _grid_table(headers, ["".join(cells)], INTENSITY_CAPTION)


def year_grid_html(year: int, monthly: dict[str, int | None],
                   selected: str | None = None) -> str:
    """12-month strip (YYYY-MM labels; None = unavailable)."""
    import calendar as _cal

    present = [v for v in monthly.values() if v is not None]
    vmax = max(present) if present else 0
    cells, headers = [], []
    for month in range(1, 13):
        key = f"{year:04d}-{month:02d}"
        val = monthly.get(key)
        headers.append(_cal.month_abbr[month])
        if val is None:
            cells.append(_cell(month, "n/a", UNAVAILABLE_CELL,
                                title="Outside verified pilot coverage"))
        else:
            cells.append(_cell(month, f"{val:,}", intensity_color(val, vmax),
                                selected=(key == selected), title=f"{key}: {val:,}"))
    return _grid_table(headers, ["".join(cells)], INTENSITY_CAPTION)


def metric_choices() -> list[str]:
    """Selectable calendar metrics (labels double as chart series names)."""
    return ["Detections (raw)", "Detections MODIS", "Detections VIIRS",
            "Harmonized observations", "Clusters",
            "FRP MODIS (MW)", "FRP VIIRS (MW)"]


def series_value(metrics: dict, grain: str, period: str, choice: str) -> int | float | None:
    """One chart/grid value from pinned 2D rows (None when no record)."""
    if choice == "Detections (raw)":
        row = query.detection_row(metrics, grain, period)
        return row["raw_observations"] if row else None
    if choice in ("Detections MODIS", "Detections VIIRS"):
        row = query.detection_row(metrics, grain, period)
        if row is None:
            return None
        product = "modis" if choice == "Detections MODIS" else "viirs-snpp"
        return row["by_product"].get(product, 0)
    if choice == "Harmonized observations":
        row = query.harmonized_row(metrics, grain, period)
        return row["harmonized_observations"] if row else None
    if choice == "Clusters":
        row = query.cluster_row(metrics, grain, period)
        return row["clusters"] if row else None
    if choice.startswith("FRP "):
        product = "modis" if "MODIS" in choice else "viirs-snpp"
        for row in query.frp_rows(metrics, grain, period):
            if row["product"] == product:
                return row["frp_sum"]
        return None
    raise ValueError(f"unknown metric choice {choice!r}")


def selection_summary(det_row: dict, products: tuple[str, ...],
                      satellites: tuple[str, ...]) -> dict:
    """Sensor-selection figures from one pinned detection row (additive only).

    Full verified numbers are always shown; this adds the in-selection
    subset so sensor filters visibly update the calendar without hiding or
    recomputing anything.
    """
    by_product = det_row.get("by_product", {}) or {}
    by_sat = det_row.get("by_satellite", {}) or {}
    return {"raw_in_selection": sum(by_product.get(p, 0) for p in products),
            "by_product": {p: by_product.get(p, 0) for p in products},
            "by_satellite": {s: by_sat.get(s, 0) for s in satellites}}


# --- Streamlit rendering (thin wrappers; logic stays in pure helpers) ---

KEY_VIEW_YEAR = "cal_view_year"
KEY_VIEW_MONTH = "cal_view_month"
KEY_METRIC = "cal_metric"
KEY_DAY_PICKER = "cal_day_picker"
KEY_MONTH_PICKER = "cal_month_picker"
KEY_JUMP_MONTH = "cal_jump_month"


def _shift_day(delta: int) -> None:
    """on_click: step the selected day (callback runs before the script)."""
    import datetime

    import streamlit as st

    from datetime import date as _date

    cur = _date(st.session_state[KEY_VIEW_YEAR],
                st.session_state[KEY_VIEW_MONTH],
                int(st.session_state.get(KEY_DAY_PICKER, 1)))
    new = cur + datetime.timedelta(days=delta)
    st.session_state[filters.KEY_PERIOD] = new.isoformat()
    st.session_state[KEY_VIEW_YEAR] = new.year
    st.session_state[KEY_VIEW_MONTH] = new.month
    st.session_state[KEY_DAY_PICKER] = new.day
    st.session_state[KEY_JUMP_MONTH] = _date(new.year, new.month, 1)


def _shift_month(delta: int) -> None:
    """on_click: step the viewed month, keeping the picked day if possible."""
    import calendar as _cal

    import streamlit as st

    from datetime import date as _date

    total = (st.session_state[KEY_VIEW_YEAR] * 12
             + (st.session_state[KEY_VIEW_MONTH] - 1)) + delta
    year, month = total // 12, total % 12 + 1
    day = min(int(st.session_state.get(KEY_DAY_PICKER, 1)),
              _cal.monthrange(year, month)[1])
    st.session_state[KEY_VIEW_YEAR] = year
    st.session_state[KEY_VIEW_MONTH] = month
    st.session_state[KEY_DAY_PICKER] = day
    st.session_state[KEY_JUMP_MONTH] = _date(year, month, 1)
    st.session_state[filters.KEY_PERIOD] = f"{year:04d}-{month:02d}-{day:02d}"


def _shift_week(delta: int) -> None:
    """on_click: step the selected ISO week."""
    import streamlit as st

    st.session_state[filters.KEY_PERIOD] = filters.shift_period(
        "weekly", st.session_state[filters.KEY_PERIOD], delta)


def _shift_year(delta: int) -> None:
    """on_click: step the viewed year (monthly view)."""
    import streamlit as st

    year = st.session_state[KEY_VIEW_YEAR] + delta
    st.session_state[KEY_VIEW_YEAR] = year
    month = int(st.session_state.get(KEY_MONTH_PICKER, 8))
    st.session_state[filters.KEY_PERIOD] = f"{year:04d}-{month:02d}"


KEY_TO_MAP_FLAG = "cal_to_map_flag"


def _calendar_to_map() -> None:
    """on_click: point the shared range at the calendar period (pre-script)."""
    import streamlit as st

    try:
        start, end = filters.period_to_range(
            st.session_state.get(filters.KEY_GRAIN, "daily"),
            st.session_state.get(filters.KEY_PERIOD, filters.INITIAL_DAY))
    except ValueError:
        return
    st.session_state[filters.KEY_RANGE_START] = start
    st.session_state[filters.KEY_RANGE_END] = end
    st.session_state[filters.KEY_MAP_TAB_HINT] = (start, end)
    st.session_state[KEY_TO_MAP_FLAG] = True


def _init_state() -> None:
    import streamlit as st

    st.session_state.setdefault(filters.KEY_GRAIN, "daily")
    st.session_state.setdefault(filters.KEY_PERIOD, filters.INITIAL_DAY)
    st.session_state.setdefault(KEY_VIEW_YEAR, 2023)
    st.session_state.setdefault(KEY_VIEW_MONTH, 8)
    st.session_state.setdefault(KEY_METRIC, metric_choices()[0])
    st.session_state.setdefault(filters.KEY_RANGE_START, "2023-01-01")
    st.session_state.setdefault(filters.KEY_RANGE_END, "2023-12-31")


def _day_value(metrics: dict, spans: dict, day: str, choice: str) -> int | float | None:
    """Grid value for one date: number, 0, or None when unavailable."""
    status = query.coverage_for(day, spans)["status"]
    if status == "unavailable":
        return None
    value = series_value(metrics, "daily", day, choice)
    if value is not None:
        return value
    return 0 if status == "complete" else None


def _range_sync_caption() -> None:
    """Visible note when the viewed period leaves the shared range."""
    import streamlit as st

    grain = st.session_state.get(filters.KEY_GRAIN, "daily")
    period = st.session_state.get(filters.KEY_PERIOD, "")
    start = st.session_state.get(filters.KEY_RANGE_START, "")
    end = st.session_state.get(filters.KEY_RANGE_END, "")
    if not filters.period_overlaps_range(grain, period, start, end):
        st.caption(f"Viewing {period}: outside the shared sidebar range "
                   f"({start} to {end}). Map, analytics, and exports "
                   f"follow the sidebar range.")


def _coverage_badge(status: str) -> None:
    import streamlit as st

    if status == "complete":
        st.success("Coverage: complete (verified pilot data).", icon="✅")
    elif status == "partial":
        st.warning("Coverage: partial — only some products available.", icon="⚠️")
    else:
        st.error("Outside verified pilot coverage: no data available.", icon="⛔")


def _render_detail(metrics: dict, spans: dict, grain: str, period: str) -> None:
    import streamlit as st

    try:
        start, end = filters.period_to_range(grain, period)
    except ValueError as exc:
        st.error(f"Invalid period: {exc}")
        return
    days = [start] if start == end else [start, end]
    statuses = {query.coverage_for(d, spans)["status"] for d in days}
    status = "complete" if statuses == {"complete"} else (
        "unavailable" if statuses == {"unavailable"} else "partial")
    _coverage_badge(status)
    if status == "unavailable":
        st.info("No observations, clusters, or FRP records exist in the precomputed "
                "calendar tables for this period. This is missing local archive data, "
                "not an observation of zero fire activity.")
        st.caption(
            "Precomputed Fire Calendar cells cover the local 2023 USA archive (2023-01-01 to 2023-12-31). "
            "For other years (2000–present) or VIIRS NOAA-20 / NOAA-21, use **Activity Map**, "
            "**Historical Analytics**, or **Events & Alerts → Pipeline** to retrieve and analyze NASA FIRMS data."
        )
        return

    det = query.detection_row(metrics, grain, period) or {}
    harm = query.harmonized_row(metrics, grain, period) or {}
    clu = query.cluster_row(metrics, grain, period) or {}
    frp = query.frp_rows(metrics, grain, period)
    products = tuple(st.session_state.get(filters.KEY_PRODUCTS, list(query.PRODUCTS)))
    satellites = tuple(st.session_state.get(filters.KEY_SATELLITES, list(query.SATELLITES)))
    sel = selection_summary(det, products, satellites)

    st.subheader("Detection Activity (satellite observations)")
    st.caption("Counts of satellite observations. Observations are not unique fires: "
               "the same fire can be observed many times.")
    cols = st.columns(4)
    cols[0].metric("Raw observations", f"{det.get('raw_observations', 0):,}")
    cols[1].metric("Harmonized", f"{harm.get('harmonized_observations', 0):,}",
                   help="Matched groups counted once each; ambiguous members "
                        "individually; singles once.")
    by_product = det.get("by_product", {})
    cols[2].metric("MODIS", f"{by_product.get('modis', 0):,}")
    cols[3].metric("VIIRS", f"{by_product.get('viirs-snpp', 0):,}")
    if set(products) != set(query.PRODUCTS) or set(satellites) != set(query.SATELLITES):
        st.caption(f"In current sensor selection "
                     f"({', '.join(products) or 'none'} / {', '.join(satellites) or 'none'}): "
                     f"{sel['raw_in_selection']:,} observations "
                     f"{sel['by_product']}. Harmonized and match-class figures "
                     f"below are cross-sensor (both sensors, for context).")
    with st.expander("Breakdowns (satellite, match class, harmonized parts)"):
        left, right = st.columns(2)
        with left:
            st.write("By satellite:", det.get("by_satellite", {}))
            st.write("By match class:", det.get("by_match_class", {}))
        with right:
            st.write("Matched groups attributed:",
                     harm.get("matched_groups_attributed", 0))
            st.write("Ambiguous members:", harm.get("ambiguous_members", 0))
            st.write("Singletons:", harm.get("singletons", 0))
            st.write("Composition:", harm.get("composition", {}))

    st.subheader("Hotspot Cluster Activity (observation groups)")
    st.caption("Analytical spatial-temporal groups of observations. Clusters are "
               "not confirmed individual wildfires. Cluster counts are "
               "cross-sensor objects and are shown unfiltered.")
    if clu:
        cols = st.columns(4)
        cols[0].metric("Clusters", f"{clu.get('clusters', 0):,}")
        cols[1].metric("Detections in clusters",
                       f"{clu.get('detections_in_clusters', 0):,}")
        cols[2].metric("Singletons", f"{clu.get('singletons', 0):,}")
        cols[3].metric("Multi-detection", f"{clu.get('multi_detection', 0):,}")
        with st.expander("Composition"):
            st.write("Size min/mean/max:", clu.get("size_min"),
                     clu.get("size_mean"), clu.get("size_max"))
            st.write("MODIS detections:", clu.get("modis_detections"),
                     "| VIIRS detections:", clu.get("viirs_detections"))
            st.write("Max observation span (h):", clu.get("span_h_max"))
    else:
        st.info("No clusters start in this period (complete coverage).")

    st.subheader("Fire Radiative Power (reported MW)")
    st.caption("Sums of reported per-observation FRP. Repeated observations are "
               "included; a sum is not total emitted fire energy. Cross-sensor "
               "FRP values are not directly comparable.")
    shown_frp = [r for r in frp if r["product"] in products]
    if len(shown_frp) != len(frp):
        hidden = sorted({r["product"] for r in frp} - set(products))
        st.caption(f"FRP rows filtered to the sensor selection "
                   f"(hidden: {', '.join(hidden)}).")
    frp = shown_frp
    if frp:
        cols = st.columns(len(frp))
        for col, row in zip(cols, sorted(frp, key=lambda r: r["product"])):
            with col:
                st.metric(f"{row['product']} sum (MW)", f"{row['frp_sum']:,.1f}")
                st.write("mean:", row["frp_mean"], "| max:", row["frp_max"])
                st.write("contributing:", row["n_contributing"],
                         "| missing:", row["n_missing_frp"])
    else:
        st.info("No FRP records for this period (complete coverage).")

    start_end = f"{start}" if start == end else f"{start} to {end}"
    st.button(f"Show {start_end} on the map", key="cal_to_map",
              on_click=_calendar_to_map)
    if st.session_state.get(KEY_TO_MAP_FLAG):
        st.info("Shared date range updated — open the Activity Map tab to see it.")
        st.session_state[KEY_TO_MAP_FLAG] = False


def _render_daily(metrics: dict, spans: dict, choice: str) -> None:
    import calendar as _cal

    import pandas as pd
    import streamlit as st

    nav1, nav2, nav3 = st.columns([1, 2, 1])
    nav1.button("Previous month", key="cal_prev_month",
                on_click=_shift_month, args=(-1,))
    nav3.button("Next month", key="cal_next_month",
                on_click=_shift_month, args=(1,))
    year = st.session_state[KEY_VIEW_YEAR]
    month = st.session_state[KEY_VIEW_MONTH]
    max_picker_date = date(date.today().year + 1, 12, 31)
    with nav2:
        picked = st.date_input(
            "Jump to month",
            value=min(max(date(year, month, 1), filters.MIN_ARCHIVE_DATE), max_picker_date),
            min_value=filters.MIN_ARCHIVE_DATE,
            max_value=max_picker_date,
            key=KEY_JUMP_MONTH,
        )
    if isinstance(picked, date) and (picked.year, picked.month) != (year, month):
        year, month = picked.year, picked.month
        st.session_state[KEY_VIEW_YEAR] = year
        st.session_state[KEY_VIEW_MONTH] = month
        day = min(int(st.session_state.get(KEY_DAY_PICKER, 1)),
                  _cal.monthrange(year, month)[1])
        st.session_state[KEY_DAY_PICKER] = day
        st.session_state[filters.KEY_PERIOD] = f"{year:04d}-{month:02d}-{day:02d}"

    st.write(f"### {_cal.month_name[month]} {year}")
    last_day = _cal.monthrange(year, month)[1]
    values = {}
    for day in range(1, last_day + 1):
        key = f"{year:04d}-{month:02d}-{day:02d}"
        values[key] = _day_value(metrics, spans, key, choice)
    period = st.session_state[filters.KEY_PERIOD]
    selected = period if period[:7] == f"{year:04d}-{month:02d}" else None
    st.markdown(month_grid_html(year, month, values, selected=selected),
                unsafe_allow_html=True)
    chart_days = [f"{year:04d}-{month:02d}-{d:02d}" for d in range(1, last_day + 1)]
    chart = pd.DataFrame({"day": [d[8:] for d in chart_days],
                          choice: [values[d] if values[d] is not None else 0
                                   for d in chart_days]})
    st.bar_chart(chart, x="day", y=choice)

    sel_day = st.number_input("Selected day", min_value=1, max_value=last_day,
                              value=int(st.session_state.get(KEY_DAY_PICKER, 1)),
                              key=KEY_DAY_PICKER)
    prev_d, next_d = st.columns(2)
    prev_d.button("Previous day", key="cal_prev_day",
                  on_click=_shift_day, args=(-1,))
    next_d.button("Next day", key="cal_next_day",
                  on_click=_shift_day, args=(1,))
    st.session_state[filters.KEY_PERIOD] = f"{year:04d}-{month:02d}-{int(sel_day):02d}"


def _render_weekly(metrics: dict, spans: dict, choice: str) -> None:
    from datetime import date as _date

    import pandas as pd
    import streamlit as st

    period = st.session_state[filters.KEY_PERIOD]
    try:
        filters.period_to_range("weekly", period)
    except ValueError:
        st.session_state[filters.KEY_PERIOD] = "2023-W31"
    prev_w, lab, next_w = st.columns([1, 2, 1])
    prev_w.button("Previous week", key="cal_prev_week",
                  on_click=_shift_week, args=(-1,))
    next_w.button("Next week", key="cal_next_week",
                  on_click=_shift_week, args=(1,))
    start, end = filters.period_to_range("weekly", st.session_state[filters.KEY_PERIOD])
    lab.write(f"### Week {st.session_state[filters.KEY_PERIOD]} ({start} to {end})")
    monday = _date.fromisoformat(start)
    values = {}
    for i in range(7):
        key = monday.fromordinal(monday.toordinal() + i).isoformat()
        values[key] = _day_value(metrics, spans, key, choice)
    st.markdown(week_strip_html(monday, values), unsafe_allow_html=True)
    days = sorted(values)
    chart = pd.DataFrame({"day": days, choice: [values[d] if values[d] is not None else 0
                                                for d in days]})
    st.bar_chart(chart, x="day", y=choice)


def _render_monthly(metrics: dict, spans: dict, choice: str) -> None:
    import pandas as pd
    import streamlit as st

    prev_y, lab, next_y = st.columns([1, 2, 1])
    prev_y.button("Previous year", key="cal_prev_year",
                  on_click=_shift_year, args=(-1,))
    next_y.button("Next year", key="cal_next_year",
                  on_click=_shift_year, args=(1,))
    year = st.session_state[KEY_VIEW_YEAR]
    lab.write(f"### {year}")
    monthly = {}
    for month in range(1, 13):
        key = f"{year:04d}-{month:02d}"
        mid = f"{key}-15"
        if query.coverage_for(mid, spans)["status"] == "unavailable":
            monthly[key] = None
        else:
            value = series_value(metrics, "monthly", key, choice)
            monthly[key] = value if value is not None else 0
    st.markdown(year_grid_html(year, monthly,
                               selected=st.session_state[filters.KEY_PERIOD]),
                unsafe_allow_html=True)
    labels = [f"{year:04d}-{m:02d}" for m in range(1, 13)]
    chart = pd.DataFrame({"month": [label[5:] for label in labels],
                          choice: [monthly[label] if monthly[label] is not None else 0
                                   for label in labels]})
    st.bar_chart(chart, x="month", y=choice)
    month = st.selectbox("Selected month", list(range(1, 13)),
                         index=min(max(int(st.session_state.get(KEY_MONTH_PICKER, 8)) - 1, 0), 11),
                         key=KEY_MONTH_PICKER)
    st.session_state[filters.KEY_PERIOD] = f"{year:04d}-{int(month):02d}"


def render_calendar_tab(metrics: dict, spans: dict) -> None:
    """Full calendar tab (grain navigation + grid + chart + detail)."""
    import streamlit as st

    _init_state()
    if st.session_state.get(filters.KEY_AOI_MODE, "full") != "full":
        st.info("Area of interest active (map only): calendar metrics are "
                "pilot-wide (USA). Regional metric breakdowns are not part of "
                "the Phase 2 outputs.")
    grain = st.radio("View", options=list(filters.GRAINS),
                     key=filters.KEY_GRAIN, horizontal=True,
                     format_func=lambda g: {"daily": "Daily", "weekly": "Weekly",
                                            "monthly": "Monthly"}[g])
    choice = st.selectbox("Metric", options=metric_choices(), key=KEY_METRIC)
    st.caption(TEMPORAL_BASIS_LABEL)
    st.caption(ZERO_NA_LEGEND)
    _range_sync_caption()
    current = st.session_state[filters.KEY_PERIOD]
    if grain == "daily" and len(current) != 10:
        st.session_state[filters.KEY_PERIOD] = filters.INITIAL_DAY
        st.session_state[KEY_VIEW_YEAR], st.session_state[KEY_VIEW_MONTH] = 2023, 8
        st.session_state[KEY_DAY_PICKER] = 1
    elif grain == "weekly" and "-W" not in current:
        st.session_state[filters.KEY_PERIOD] = "2023-W31"
    elif grain == "monthly" and (len(current) != 7 or "-W" in current):
        st.session_state[filters.KEY_PERIOD] = "2023-08"
        st.session_state[KEY_VIEW_YEAR] = 2023
        st.session_state[KEY_MONTH_PICKER] = 8
    if grain == "daily":
        _render_daily(metrics, spans, choice)
    elif grain == "weekly":
        _render_weekly(metrics, spans, choice)
    else:
        _render_monthly(metrics, spans, choice)
    st.divider()
    st.write(f"### Selected period: {st.session_state[filters.KEY_PERIOD]}")
    _render_detail(metrics, spans, grain, st.session_state[filters.KEY_PERIOD])
    st.info(METHODOLOGY_NOTE)
