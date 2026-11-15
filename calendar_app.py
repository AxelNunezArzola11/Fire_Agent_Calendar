#!/usr/bin/env python3
"""FireAgent Calendar — historical fire-observation explorer (Phase 3).

Run: .venv/bin/streamlit run calendar_app.py
Separate entry point: the original dashboard (app.py) is untouched.

All figures come from pinned Phase 2 analytical outputs (see
src/calendar/settings.py CAL_EXEC_*). This app computes no analytics.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(REPO))

import streamlit as st

from src.calendar import settings
from src.calendar.processing import regions
from src.calendar.ui import (analytics_view, aoi, calendar_view, events_view,
                             export_view, filters, frames, map_view, query)

st.set_page_config(page_title="FireAgent Calendar", layout="wide")


@st.cache_data(show_spinner=False)
def _metrics(fingerprint: tuple) -> dict:
    """Pinned Phase 2D metrics (fingerprint key: never stale)."""
    _ = fingerprint
    return query.load_metrics()


@st.cache_data(show_spinner="Loading anomalies…")
def _anomalies(fingerprint: tuple) -> list:
    """Pinned Phase 4C anomaly results (fingerprint key: never stale)."""
    _ = fingerprint
    return query.load_anomalies()


def _metrics_if_available(fingerprint: tuple) -> dict | None:
    """Load Phase 2D metrics when available, or return None without crashing."""
    if not query.artifact_status("2D", query.METRICS_FILENAME)["available"]:
        return None
    try:
        return _metrics(fingerprint)
    except FileNotFoundError:
        return None


def _anomalies_if_available(fingerprint: tuple) -> list | None:
    """Load Phase 4C anomalies when available, or return None without crashing."""
    if not query.artifact_status("4C", "anomalies_4c.json")["available"]:
        return None
    try:
        return _anomalies(fingerprint)
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner="Loading detections…")
def _detections(fingerprint: tuple, date_start: str, date_end: str,
                products: tuple, satellites: tuple, bbox, modis_conf_min,
                viirs_conf, limit: int):
    """Bounded detection query (fingerprint + all params are cache keys)."""
    _ = fingerprint
    return query.query_detections(date_start=date_start, date_end=date_end,
                                  products=tuple(products),
                                  satellites=tuple(satellites), bbox=bbox,
                                  modis_conf_min=modis_conf_min,
                                  viirs_conf=tuple(viirs_conf) if viirs_conf else viirs_conf,
                                  limit=limit)


@st.cache_data(show_spinner=False)
def _det_count(fingerprint: tuple, date_start: str, date_end: str,
               products: tuple, satellites: tuple, bbox, modis_conf_min,
               viirs_conf) -> int:
    _ = fingerprint
    return query.count_detections(date_start=date_start, date_end=date_end,
                                  products=tuple(products),
                                  satellites=tuple(satellites), bbox=bbox,
                                  modis_conf_min=modis_conf_min,
                                  viirs_conf=tuple(viirs_conf) if viirs_conf else viirs_conf)


@st.cache_data(show_spinner="Loading clusters…")
def _clusters(fingerprint: tuple, date_start: str, date_end: str, bbox,
              limit: int, columns=None):
    """Bounded cluster query (fingerprint + all params are cache keys)."""
    _ = fingerprint
    return query.query_clusters(date_start=date_start, date_end=date_end,
                                bbox=bbox, limit=limit,
                                columns=tuple(columns) if columns else None)


@st.cache_data(show_spinner=False)
def _clu_count(fingerprint: tuple, date_start: str, date_end: str, bbox) -> int:
    _ = fingerprint
    return query.count_clusters(date_start=date_start, date_end=date_end,
                                bbox=bbox)


@st.cache_data(show_spinner=False)
def _det_by_id(fingerprint: tuple, detection_id: str):
    _ = fingerprint
    return query.detection_by_id(detection_id)


@st.cache_data(show_spinner=False)
def _member(fingerprint: tuple, detection_id: str):
    _ = fingerprint
    return query.member_by_id(detection_id)


@st.cache_data(show_spinner=False)
def _group(fingerprint: tuple, match_id: str):
    _ = fingerprint
    return query.match_group(match_id)


@st.cache_data(show_spinner=False)
def _clu_by_id(fingerprint: tuple, cluster_id: str):
    _ = fingerprint
    return query.cluster_by_id(cluster_id)


@st.cache_data(show_spinner="Loading cluster members…")
def _cluster_dets(fingerprint: tuple, cluster_id: str):
    """Member detections of one cluster (fingerprint + id are cache keys)."""
    _ = fingerprint
    return query.detections_of_cluster(cluster_id)


@st.cache_resource(show_spinner="Building map…", max_entries=4)
def _det_map_cached(fingerprint: tuple, df, color_mode: str,
                    show_density: bool, grid_deg: float,
                    show_boundary: bool, fit):
    """Built detection map, cached (never handed out directly; see below)."""
    import folium as _folium

    _ = fingerprint
    param = map_view.build_detection_map(df, color_mode=color_mode,
                                         show_density=show_density,
                                         grid_deg=grid_deg,
                                         show_boundary=show_boundary, fit=fit)
    param.get_root().html.add_child(_folium.Element(
        map_view.legend_html("detections", color_mode,
                             grid_deg if show_density else None)))
    return param


def _det_map(fingerprint: tuple, df, color_mode: str, show_density: bool,
             grid_deg: float, show_boundary: bool, fit):
    """Private per-render copy of the cached map (concurrency safety).

    streamlit-folium mutates the map object while serializing it, so
    handing the shared cached object to overlapping renders raises
    "OrderedDict mutated during iteration" and the map is replaced by
    an error panel. Each render gets its own deep copy; the cached
    pristine original is never rendered directly.
    """
    return map_view.copy_map(_det_map_cached(
        fingerprint, df, color_mode, show_density, grid_deg,
        show_boundary, fit))


@st.cache_resource(show_spinner="Building map…", max_entries=4)
def _clu_map_cached(fingerprint: tuple, df, show_extent: bool,
                    show_boundary: bool, fit):
    """Built cluster map, cached (never handed out directly; see below)."""
    import folium as _folium

    _ = fingerprint
    param = map_view.build_cluster_map(df, show_extent=show_extent,
                                       show_boundary=show_boundary, fit=fit)
    param.get_root().html.add_child(
        _folium.Element(map_view.legend_html("clusters", "")))
    return param


def _clu_map(fingerprint: tuple, df, show_extent: bool, show_boundary: bool,
             fit):
    """Private per-render copy of the cached map (concurrency safety)."""
    return map_view.copy_map(_clu_map_cached(fingerprint, df, show_extent,
                                             show_boundary, fit))


def _map_callbacks(fingerprint: tuple) -> map_view.MapCallbacks:
    """Bind the fingerprint into cached query callables for the map tab."""
    return map_view.MapCallbacks(
        fetch_det=lambda **kw: _detections(fingerprint, **kw),
        count_det=lambda **kw: _det_count(fingerprint, **kw),
        fetch_clu=lambda **kw: _clusters(fingerprint, **kw),
        count_clu=lambda **kw: _clu_count(fingerprint, **kw),
        by_det=lambda det_id: _det_by_id(fingerprint, det_id),
        member=lambda det_id: _member(fingerprint, det_id),
        group=lambda match_id: _group(fingerprint, match_id),
        by_clu=lambda clu_id: _clu_by_id(fingerprint, clu_id),
        build_det_map=lambda *a, **kw: _det_map(fingerprint, *a, **kw),
        build_clu_map=lambda *a, **kw: _clu_map(fingerprint, *a, **kw),
        fetch_cluster_dets=lambda cluster_id: _cluster_dets(
            fingerprint, cluster_id))


@st.cache_data(show_spinner=False)
def _spans() -> dict:
    """Per-dataset verified coverage spans from the Phase 1 catalog."""
    cat_st = query.catalog_status()
    if not cat_st["available"]:
        return {}
    try:
        catalog = json.loads(Path(cat_st["resolved_path"]).read_text(encoding="utf-8"))
        return query.catalog_spans(catalog)
    except (OSError, ValueError):
        return {}


def _aoi_controls() -> None:
    """Controls for the active AOI mode (inside the sidebar expander)."""
    mode = st.session_state.get(filters.KEY_AOI_MODE, "full")
    if mode == "region":
        st.selectbox("Region", options=sorted(regions.REGIONS),
                     key=filters.KEY_AOI_REGION)
        region = regions.REGIONS.get(st.session_state.get(filters.KEY_AOI_REGION, ""))
        if region is not None:
            st.caption(f"{region.version} ({region.crs}): {region.source}")
    elif mode == "custom":
        with st.expander("Search by place or coordinates"):
            place = st.text_input("Place name", key="cal_geo_place",
                                  placeholder="e.g. Redding, California")
            col_lat, col_lon = st.columns(2)
            with col_lat:
                plat = st.number_input("Latitude", value=40.59,
                                       key="cal_geo_lat", format="%.4f")
            with col_lon:
                plon = st.number_input("Longitude", value=-122.39,
                                       key="cal_geo_lon", format="%.4f")
            prad = st.number_input("Radius (km)", value=50.0,
                                   key="cal_geo_radius")
            if st.button("Find place", key="cal_geo_find") and place:
                try:
                    found = aoi.geocode_nominatim(place.strip())
                except aoi.AoIError as exc:
                    st.error(str(exc))
                else:
                    st.session_state["cal_geo_lat"] = found["latitude"]
                    st.session_state["cal_geo_lon"] = found["longitude"]
                    st.caption(f"Found: {found['label']}")
            if st.button("Apply point + radius", key="cal_geo_apply"):
                try:
                    box = aoi.radius_bbox(
                        st.session_state.get("cal_geo_lat"),
                        st.session_state.get("cal_geo_lon"),
                        st.session_state.get("cal_geo_radius"))
                except aoi.AoIError as exc:
                    st.error(str(exc))
                else:
                    (st.session_state[filters.KEY_AOI_W],
                     st.session_state[filters.KEY_AOI_S],
                     st.session_state[filters.KEY_AOI_E],
                     st.session_state[filters.KEY_AOI_N]) = box
                    st.success("Custom area set from search.")
        col_w, col_e = st.columns(2)
        with col_w:
            st.number_input("West", value=-124.48, key=filters.KEY_AOI_W,
                            format="%.2f")
            st.number_input("South", value=32.53, key=filters.KEY_AOI_S,
                            format="%.2f")
        with col_e:
            st.number_input("East", value=-114.13, key=filters.KEY_AOI_E,
                            format="%.2f")
            st.number_input("North", value=42.01, key=filters.KEY_AOI_N,
                            format="%.2f")
        try:
            aoi.validate_bbox((st.session_state[filters.KEY_AOI_W],
                               st.session_state[filters.KEY_AOI_S],
                               st.session_state[filters.KEY_AOI_E],
                               st.session_state[filters.KEY_AOI_N]))
        except aoi.AoIError as exc:
            st.error(f"Invalid bbox: {exc}")
    elif mode == "viewport":
        bounds = st.session_state.get(map_view.KEY_MAP_BBOX)
        if bounds:
            st.caption(f"Lon {bounds[0]:.2f}…{bounds[2]:.2f}, "
                       f"lat {bounds[1]:.2f}…{bounds[3]:.2f}.")
        else:
            st.caption("Pan/zoom the map, then click “Filter to current map "
                       "view” in the Activity Map tab.")
    elif mode == "saved":
        try:
            saved = aoi.load_aois(settings.CAL_AOI_PATH)
        except aoi.AoIError as exc:
            st.error(str(exc))
            return
        if not saved:
            st.caption("No saved areas yet (save one below).")
        else:
            st.selectbox("Saved area", options=sorted(saved),
                         key=filters.KEY_AOI_SAVED)
        with st.expander("Save current custom bbox"):
            name = st.text_input("Name", key="cal_aoi_name",
                                 placeholder="e.g. northern-california")
            if st.button("Save area", key="cal_aoi_save"):
                try:
                    aoi.save_aoi(settings.CAL_AOI_PATH, name,
                                 (st.session_state.get(filters.KEY_AOI_W, -124.48),
                                  st.session_state.get(filters.KEY_AOI_S, 32.53),
                                  st.session_state.get(filters.KEY_AOI_E, -114.13),
                                  st.session_state.get(filters.KEY_AOI_N, 42.01)))
                    st.success(f"Saved {name.strip()!r}.")
                except aoi.AoIError as exc:
                    st.error(str(exc))
        if saved and st.button("Delete selected area", key="cal_aoi_delete"):
            try:
                aoi.delete_aoi(settings.CAL_AOI_PATH,
                               st.session_state.get(filters.KEY_AOI_SAVED, ""))
                st.success("Deleted.")
            except aoi.AoIError as exc:
                st.error(str(exc))


def _sidebar() -> None:
    with st.sidebar:
        st.title("FireAgent Calendar")
        st.caption("Historical satellite fire observations (2023 pilot).")
        st.caption(calendar_view.TEMPORAL_BASIS_LABEL)
        st.subheader("Shared filters")
        st.radio("Map mode", options=list(filters.MODES), key=filters.KEY_MODE,
                 format_func=lambda m: {"detections": "Detections",
                                        "clusters": "Clusters"}[m])
        col_a, col_b = st.columns(2)
        with col_a:
            st.text_input("From (UTC)", value=filters.INITIAL_DAY,
                          key=filters.KEY_RANGE_START,
                          placeholder="YYYY-MM-DD")
        with col_b:
            st.text_input("To (UTC)", value=filters.INITIAL_DAY,
                          key=filters.KEY_RANGE_END, placeholder="YYYY-MM-DD")
        st.multiselect("Products", options=list(query.PRODUCTS),
                       default=list(query.PRODUCTS), key=filters.KEY_PRODUCTS)
        st.multiselect("Satellites", options=list(query.SATELLITES),
                       default=list(query.SATELLITES), key=filters.KEY_SATELLITES)
        st.caption("VIIRS NOAA-20/21 have no 2023 pilot data.")
        with st.expander("Confidence"):
            st.slider("MODIS minimum (0–100)", 0, 100, 0,
                      key=filters.KEY_MODIS_CONF,
                      help="MODIS confidence is a 0–100 score. 0 disables the filter.")
            st.multiselect("VIIRS classes", options=list(query.VIIRS_CONFIDENCE),
                           default=list(query.VIIRS_CONFIDENCE),
                           key=filters.KEY_VIIRS_CONF,
                           help="VIIRS confidence is categorical: l/n/h "
                                "(low/nominal/high).")
            st.caption("Confidence scales differ by product and are filtered "
                       "independently.")
        with st.expander("Area of interest (map only)"):
            st.radio("Area", options=list(filters.AOI_MODES),
                     key=filters.KEY_AOI_MODE,
                     format_func=lambda m: {
                         "full": "Full pilot (USA)",
                         "region": "Preset region",
                         "custom": "Custom bounding box",
                         "viewport": "Current map view",
                         "saved": "Saved area"}[m])
            _aoi_controls()
        spans = _spans()
        with st.expander("Verified coverage"):
            if spans:
                for dataset, span in sorted(spans.items()):
                    st.write(f"{dataset}: {span['min']} to {span['max']}")
            else:
                st.caption("Catalog coverage metadata unavailable.")
        with st.expander("Analytical provenance"):
            for stage, exec_id in sorted(query.exec_chain().items()):
                st.write(f"Phase {stage}: `{exec_id[:8]}`"
                         if exec_id else f"Phase {stage}: unconfigured")
            for stage, exec_id in sorted(query.phase4_chain().items()):
                st.write(f"Phase {stage}: `{exec_id[:8]}`"
                         if exec_id else f"Phase {stage}: unconfigured")
            for stage, exec_id in sorted(query.phase5_chain().items()):
                st.write(f"Phase {stage}: `{exec_id[:8]}`"
                         if exec_id else f"Phase {stage}: not built yet")
        with st.expander("Deployment data status"):
            ov = query.storage_overview()
            rows = [
                ("Calendar (2D)", ov["artifacts"]["2D_metrics"]["available"]),
                ("Map Detections (DuckDB)", ov["duckdb"]["available"]),
                ("Map Clusters (2C)", ov["artifacts"]["2C_clusters"]["available"]),
                ("Analytics (4C)", ov["artifacts"]["4C_anomalies"]["available"]),
                ("Events DB", ov["events_db"]["available"]),
                ("Export Ready", ov["duckdb"]["available"]
                 and ov["artifacts"]["2B_members"]["available"]
                 and ov["artifacts"]["2C_members"]["available"]),
            ]
            for label, ok in rows:
                st.caption(f"{label}: {'ready' if ok else 'unavailable'}")
        st.info(calendar_view.METHODOLOGY_NOTE)


def _about_tab(metrics: dict | None) -> None:
    metrics = metrics or {}
    st.header("About this data")
    st.write("FireAgent Calendar explores the verified 2023 NASA FIRMS pilot "
             "(482,672 accepted MODIS and VIIRS Standard detections) through "
             "pinned Phase 2 analytical outputs. Figures are read from those "
             "outputs; the interface computes no new analytics.")
    st.subheader("Temporal basis")
    computed = metrics.get("yearly", {}).get("detection_activity", {})
    st.write("- Acquisition date (UTC): when the satellite collected the "
             "observation — the basis of all historical analysis here.")
    st.write("- Product availability date: Standard archive product "
             "(NRT→Standard lag applies upstream); per-file availability "
             "dates are not tracked in the pilot.")
    st.write(f"- Processing date: {computed.get('computed_at', 'unavailable')} "
             f"(metrics exec `{(computed.get('exec_id') or 'unconfigured')[:8]}`).")
    st.write("All displayed timestamps are UTC unless labeled otherwise.")
    st.subheader("Methodology")
    st.write("- Satellite observations are not unique fires: one fire can be "
             "observed many times by one or more sensors.")
    st.write("- Analytical hotspot clusters are spatial-temporal groups of "
             "observations, not confirmed individual wildfires, perimeters, "
             "or incidents.")
    st.write("- A sum of repeated FRP observations is not total emitted fire "
             "energy. MODIS and VIIRS FRP are not directly comparable.")
    st.write("- Harmonized observations count each matched cross-sensor "
             "group once; ambiguous members are counted individually.")
    st.write("- A date with no records inside verified coverage means zero "
             "recorded observations; a date outside coverage means no data, "
             "not zero fire activity.")
    st.info(calendar_view.METHODOLOGY_NOTE)
    st.subheader("How the system works (deterministic vs AI)")
    st.write("- Deterministic Python performs every scientific and "
             "operational step: acquisition, normalization, sensor "
             "matching, clustering, metrics, baselines, anomaly "
             "detection, event association, lifecycle transitions, and "
             "alert rules. These are fixed algorithms with versioned "
             "provenance — not autonomous AI agents.")
    st.write("- Qwen provides optional natural-language interpretation "
             "of stored evidence only. It never calculates metrics, "
             "changes events, lifecycle states, or severity, and runs "
             "solely when the user clicks Explain. When unavailable, a "
             "deterministic explanation is shown instead.")
    st.write("- Email delivery sends subscribed notifications through "
             "configured SMTP; it performs no analysis.")
    st.write("FireAgent Calendar is an AI-assisted wildfire monitoring "
             "and analytics application with deterministic scientific "
             "processing and optional Qwen-based interpretation. It does "
             "not currently implement autonomous multi-agent "
             "orchestration.")
    st.subheader("Data sources")
    chain = query.exec_chain()
    chain4 = query.phase4_chain()
    chain5 = query.phase5_chain()
    ov = query.storage_overview()
    arts = ov["artifacts"]

    def _exec_tag(eid: str, ok: bool) -> str:
        prefix = (eid or "unconfigured")[:8]
        return f"{prefix} ({'ready' if ok else 'unavailable'})"

    st.table(frames.about_sources_frame([
        {"Feature": "Calendar metrics", "Source": "metrics_2d.json",
         "Execution": _exec_tag(chain["2D"], arts["2D_metrics"]["available"])},
        {"Feature": "Dual index views", "Source": "index_2e.json",
         "Execution": _exec_tag(chain["2E"], arts["2E_index"]["available"])},
        {"Feature": "Map detections", "Source": "DuckDB detections view",
         "Execution": (f"Phase 1D cal_norm_v1 "
                       f"({'ready' if ov['duckdb']['available'] else ov['duckdb']['status']})")},
        {"Feature": "Map clusters", "Source": "clusters.parquet",
         "Execution": _exec_tag(chain["2C"], arts["2C_clusters"]["available"])},
        {"Feature": "Match classes", "Source": "match_members.parquet",
         "Execution": _exec_tag(chain["2B"], arts["2B_members"]["available"])},
        {"Feature": "Coverage", "Source": "catalog.json",
         "Execution": f"Phase 1D ({'ready' if ov['catalog']['available'] else ov['catalog']['status']})"},
        {"Feature": "Baselines", "Source": "baselines_4b.json",
         "Execution": _exec_tag(chain4["4B"], arts["4B_baselines"]["available"])},
        {"Feature": "Anomalies", "Source": "anomalies_4c.json",
         "Execution": _exec_tag(chain4["4C"], arts["4C_anomalies"]["available"])},
        {"Feature": "Events & alerts", "Source": "calendar_events.sqlite",
         "Execution": _exec_tag(chain5["5C"] or "not built", ov["events_db"]["available"])},
    ]))
    st.subheader("How to cite")
    for citation in calendar_view.CITATIONS:
        st.write(f"- {citation}")
    st.subheader("Limitations")
    st.write("- Pilot year 2023 only; both sensors cover 2023-01-01 to "
             "2023-12-31. NOAA-20/21 have no pilot data.")
    st.write("- Single-linkage chaining can merge nearby activity into very "
             "large clusters; cluster size is not fire size.")
    st.write("- The California region (map filter) is an approximate bounding "
             "box, not an administrative boundary.")


def main() -> None:
    fingerprint = query.fingerprint()
    _sidebar()
    tab_calendar, tab_map, tab_analytics, tab_events, tab_export, tab_about = st.tabs(
        ["Fire Calendar", "Activity Map", "Historical Analytics",
         "Events & Alerts", "Data Export", "About"])
    metrics_doc = _metrics_if_available(fingerprint)
    anomalies_doc = _anomalies_if_available(fingerprint)
    with tab_calendar:
        try:
            if metrics_doc is None:
                met_st = query.artifact_status("2D", query.METRICS_FILENAME)
                st.warning(f"Calendar metrics output unavailable ({met_st['status']}).")
                st.info(f"{met_st['reason']} — {met_st['remediation']}")
            else:
                calendar_view.render_calendar_tab(metrics_doc, _spans())
        except Exception as exc:  # per-tab guard (dashboard idiom)
            st.error(f"Calendar error: {exc}")
    with tab_map:
        try:
            map_view.render_map_tab(_spans(), _map_callbacks(fingerprint))
        except Exception as exc:
            st.error(f"Map error: {exc}")
    with tab_analytics:
        try:
            analytics_view.render_analytics_tab(anomalies_doc)
        except Exception as exc:
            st.error(f"Analytics error: {exc}")
    with tab_events:
        try:
            events_view.render_events_tab(metrics_doc, anomalies_doc)
        except Exception as exc:  # per-tab guard (dashboard idiom)
            st.error(f"Events error: {exc}")
    with tab_export:
        try:
            export_view.render_export_tab()
        except Exception as exc:  # per-tab guard (dashboard idiom)
            st.error(f"Export error: {exc}")
    with tab_about:
        try:
            _about_tab(metrics_doc)
        except Exception as exc:
            st.error(f"About error: {exc}")


main()
