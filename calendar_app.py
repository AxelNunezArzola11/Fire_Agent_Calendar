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
from src.calendar.acquisition import sensors
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
                    show_boundary: bool, fit,
                    region_id: str | None = None, bbox_label: str = ""):
    """Built detection map, cached (never handed out directly; see below)."""
    import folium as _folium

    _ = fingerprint
    param = map_view.build_detection_map(df, color_mode=color_mode,
                                         show_density=show_density,
                                         grid_deg=grid_deg,
                                         show_boundary=show_boundary, fit=fit,
                                         region_id=region_id,
                                         bbox_label=bbox_label)
    param.get_root().html.add_child(_folium.Element(
        map_view.legend_html("detections", color_mode,
                             grid_deg if show_density else None,
                             boundary_label=bbox_label or "Active region / pilot bbox")))
    return param


def _det_map(fingerprint: tuple, df, color_mode: str, show_density: bool,
             grid_deg: float, show_boundary: bool, fit,
             region_id: str | None = None, bbox_label: str = ""):
    """Private per-render copy of the cached map (concurrency safety)."""
    return map_view.copy_map(_det_map_cached(
        fingerprint, df, color_mode, show_density, grid_deg,
        show_boundary, fit, region_id=region_id, bbox_label=bbox_label))


@st.cache_resource(show_spinner="Building map…", max_entries=4)
def _clu_map_cached(fingerprint: tuple, df, show_extent: bool,
                    show_boundary: bool, fit,
                    region_id: str | None = None, bbox_label: str = ""):
    """Built cluster map, cached (never handed out directly; see below)."""
    import folium as _folium

    _ = fingerprint
    param = map_view.build_cluster_map(df, show_extent=show_extent,
                                       show_boundary=show_boundary, fit=fit,
                                       region_id=region_id,
                                       bbox_label=bbox_label)
    param.get_root().html.add_child(
        _folium.Element(map_view.legend_html(
            "clusters", "", boundary_label=bbox_label or "Active region / pilot bbox")))
    return param


def _clu_map(fingerprint: tuple, df, show_extent: bool, show_boundary: bool,
             fit, region_id: str | None = None, bbox_label: str = ""):
    """Private per-render copy of the cached map (concurrency safety)."""
    return map_view.copy_map(_clu_map_cached(
        fingerprint, df, show_extent, show_boundary, fit,
        region_id=region_id, bbox_label=bbox_label))


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


def _apply_region_bbox(region_id: str) -> None:
    """on_click: populate custom bbox inputs from a preset region."""
    reg = regions.REGIONS.get(region_id)
    if reg is None:
        return
    (st.session_state[filters.KEY_AOI_W],
     st.session_state[filters.KEY_AOI_S],
     st.session_state[filters.KEY_AOI_E],
     st.session_state[filters.KEY_AOI_N]) = reg.bbox
    st.session_state[filters.KEY_AOI_CUSTOM_SET] = True


def _apply_point_to_custom() -> None:
    """on_click: compute radius bbox from point + radius into custom bbox inputs."""
    try:
        box = aoi.radius_bbox(
            st.session_state.get(filters.KEY_AOI_LAT, 40.59),
            st.session_state.get(filters.KEY_AOI_LON, -122.39),
            st.session_state.get(filters.KEY_AOI_RADIUS, 50.0),
        )
    except aoi.AoIError as exc:
        st.session_state["cal_geo_apply_err"] = str(exc)
        return
    st.session_state.pop("cal_geo_apply_err", None)
    (st.session_state[filters.KEY_AOI_W],
     st.session_state[filters.KEY_AOI_S],
     st.session_state[filters.KEY_AOI_E],
     st.session_state[filters.KEY_AOI_N]) = box
    st.session_state[filters.KEY_AOI_CUSTOM_SET] = True
    st.session_state["cal_geo_apply_ok"] = box


def _apply_point_preset(lat: float, lon: float, radius_km: float) -> None:
    """on_click: set sidebar point + radius fields cleanly."""
    st.session_state[filters.KEY_AOI_LAT] = float(lat)
    st.session_state[filters.KEY_AOI_LON] = float(lon)
    st.session_state[filters.KEY_AOI_RADIUS] = float(radius_km)
    st.session_state[filters.KEY_AOI_POINT_SET] = True


def _aoi_controls(saved_aois: dict[str, dict] | None = None) -> None:
    """Controls for the active AOI mode (inside the sidebar expander)."""
    mode = st.session_state.get(filters.KEY_AOI_MODE, "full")
    if mode == "region":
        opts = sorted(regions.REGIONS)
        st.selectbox(
            "Region",
            options=opts,
            key=filters.KEY_AOI_REGION,
            format_func=regions.region_label,
        )
        region = regions.REGIONS.get(st.session_state.get(filters.KEY_AOI_REGION, ""))
        if region is not None:
            w, s, e, n = region.bbox
            st.caption(
                f"{region.version} ({region.crs}) · bbox lon `{w:.2f}..{e:.2f}`, "
                f"lat `{s:.2f}..{n:.2f}`"
            )
            st.caption(region.source)
    elif mode == "point_radius":
        st.session_state.setdefault(filters.KEY_AOI_LAT, 40.59)
        st.session_state.setdefault(filters.KEY_AOI_LON, -122.39)
        st.session_state.setdefault(filters.KEY_AOI_RADIUS, 50.0)
        st.caption("Quick point presets:")
        pt_cols = st.columns(4)
        for col, lbl, plat, plon, prad, slug in (
            (pt_cols[0], "Madrid", 40.42, -3.70, 100.0, "madrid"),
            (pt_cols[1], "CDMX", 19.43, -99.13, 100.0, "cdmx"),
            (pt_cols[2], "Brasilia", -15.79, -47.88, 150.0, "brasilia"),
            (pt_cols[3], "Sydney", -33.87, 151.21, 120.0, "sydney"),
        ):
            col.button(
                lbl,
                key=f"cal_pt_preset_{slug}",
                on_click=_apply_point_preset,
                args=(plat, plon, prad),
                help=f"Point ({plat:.2f}, {plon:.2f}) + {prad:.0f} km",
            )
        place = st.text_input("Find place (optional)", key="cal_geo_place",
                              placeholder="e.g. Madrid, Spain or Guadalajara, Mexico")
        if st.button("Find place", key="cal_geo_find") and place:
            try:
                found = aoi.geocode_nominatim(place.strip())
            except aoi.AoIError as exc:
                st.error(str(exc))
            else:
                st.session_state[filters.KEY_AOI_LAT] = found["latitude"]
                st.session_state[filters.KEY_AOI_LON] = found["longitude"]
                st.caption(f"Found: {found['label']}")
        col_lat, col_lon = st.columns(2)
        with col_lat:
            st.number_input("Latitude", key=filters.KEY_AOI_LAT, format="%.4f")
        with col_lon:
            st.number_input("Longitude", key=filters.KEY_AOI_LON, format="%.4f")
        st.number_input("Radius (km)", key=filters.KEY_AOI_RADIUS,
                        min_value=0.5, max_value=2000.0, step=5.0, format="%.1f")
        st.session_state[filters.KEY_AOI_POINT_SET] = True
        try:
            box = aoi.radius_bbox(
                st.session_state.get(filters.KEY_AOI_LAT),
                st.session_state.get(filters.KEY_AOI_LON),
                st.session_state.get(filters.KEY_AOI_RADIUS),
            )
            st.caption(
                f"Auto bbox (EPSG:4326): lon `{box[0]:.2f}..{box[2]:.2f}`, "
                f"lat `{box[1]:.2f}..{box[3]:.2f}`."
            )
        except aoi.AoIError as exc:
            st.error(f"Invalid point/radius: {exc}")
    elif mode == "custom":
        st.session_state[filters.KEY_AOI_CUSTOM_SET] = True
        st.caption("Quick regional bbox presets:")
        preset_r1 = st.columns(3)
        for col, rid, short in zip(
            preset_r1,
            ("california", "mexico", "madrid"),
            ("CA", "MX", "MAD"),
        ):
            col.button(
                short,
                key=f"cal_aoi_preset_{rid}",
                on_click=_apply_region_bbox,
                args=(rid,),
                help=f"Set custom bbox to {regions.region_label(rid)}",
            )
        preset_r2 = st.columns(3)
        for col, rid, short in zip(
            preset_r2,
            ("brazil", "spain", "australia"),
            ("BR", "ES", "AU"),
        ):
            col.button(
                short,
                key=f"cal_aoi_preset_{rid}",
                on_click=_apply_region_bbox,
                args=(rid,),
                help=f"Set custom bbox to {regions.region_label(rid)}",
            )
        with st.expander("Search by place or coordinates"):
            st.session_state.setdefault(filters.KEY_AOI_LAT, 40.59)
            st.session_state.setdefault(filters.KEY_AOI_LON, -122.39)
            st.session_state.setdefault(filters.KEY_AOI_RADIUS, 50.0)
            place = st.text_input("Place name", key="cal_geo_place",
                                  placeholder="e.g. Seville, Spain or Manaus, Brazil")
            if st.button("Find place", key="cal_geo_find") and place:
                try:
                    found = aoi.geocode_nominatim(place.strip())
                except aoi.AoIError as exc:
                    st.error(str(exc))
                else:
                    st.session_state[filters.KEY_AOI_LAT] = found["latitude"]
                    st.session_state[filters.KEY_AOI_LON] = found["longitude"]
                    st.caption(f"Found: {found['label']}")
            col_lat, col_lon = st.columns(2)
            with col_lat:
                st.number_input("Latitude", key=filters.KEY_AOI_LAT, format="%.4f")
            with col_lon:
                st.number_input("Longitude", key=filters.KEY_AOI_LON, format="%.4f")
            st.number_input("Radius (km)", key=filters.KEY_AOI_RADIUS,
                            min_value=0.5, max_value=2000.0, step=5.0, format="%.1f")
            st.button("Apply point + radius", key="cal_geo_apply",
                      on_click=_apply_point_to_custom)
            if st.session_state.get("cal_geo_apply_err"):
                st.error(st.session_state["cal_geo_apply_err"])
            elif st.session_state.get("cal_geo_apply_ok"):
                st.success("Custom bbox updated from point + radius.")
        st.session_state.setdefault(filters.KEY_AOI_W, -124.48)
        st.session_state.setdefault(filters.KEY_AOI_S, 32.53)
        st.session_state.setdefault(filters.KEY_AOI_E, -114.13)
        st.session_state.setdefault(filters.KEY_AOI_N, 42.01)
        col_w, col_e = st.columns(2)
        with col_w:
            st.number_input("West", key=filters.KEY_AOI_W, format="%.2f")
            st.number_input("South", key=filters.KEY_AOI_S, format="%.2f")
        with col_e:
            st.number_input("East", key=filters.KEY_AOI_E, format="%.2f")
            st.number_input("North", key=filters.KEY_AOI_N, format="%.2f")
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
        if saved_aois is None:
            try:
                saved = aoi.load_aois(settings.CAL_AOI_PATH)
            except aoi.AoIError as exc:
                st.error(str(exc))
                return
        else:
            saved = saved_aois
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
        st.caption("Multi-year & global satellite fire observations (local archive + live NASA FIRMS).")
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
        _avail_sidebar = sensors.evaluate_availability(
            str(st.session_state.get(filters.KEY_RANGE_START, filters.INITIAL_DAY)),
            str(st.session_state.get(filters.KEY_RANGE_END, filters.INITIAL_DAY)),
            tuple(st.session_state.get(filters.KEY_PRODUCTS, list(query.PRODUCTS))),
            tuple(st.session_state.get(filters.KEY_SATELLITES, list(query.SATELLITES))),
            firms_key_configured=bool(settings.get_setting("FIRMS_MAP_KEY", "")),
            live_dataset=st.session_state.get("cal_live_regional"),
        )
        st.caption(_avail_sidebar["summary"])
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
        try:
            saved_aois = aoi.load_aois(settings.CAL_AOI_PATH)
        except aoi.AoIError:
            saved_aois = {}
        with st.expander("Geographic area of interest"):
            prev_aoi_mode = st.session_state.get("cal_prev_aoi_mode")
            sel_aoi_mode = st.radio("Area", options=list(filters.AOI_MODES),
                                    key=filters.KEY_AOI_MODE,
                                    format_func=lambda m: {
                                        "full": "Full archive extent (USA / North America)",
                                        "region": "Preset country / region",
                                        "point_radius": "Point + radius (lat/lon + km)",
                                        "custom": "Custom bounding box / lat-lon",
                                        "viewport": "Current map view",
                                        "saved": "Saved area"}[m])
            if prev_aoi_mode is not None and prev_aoi_mode != "full" and sel_aoi_mode == "full":
                st.session_state.pop(filters.KEY_RESOLVED_AOI, None)
                st.session_state.pop(filters.KEY_AOI_CUSTOM_SET, None)
                st.session_state.pop(filters.KEY_AOI_POINT_SET, None)
            st.session_state["cal_prev_aoi_mode"] = sel_aoi_mode
            _aoi_controls(saved_aois)
            try:
                resolved_sb = aoi.resolve_session_aoi(
                    st.session_state,
                    saved_aois=saved_aois,
                    include_session_fallback=True,
                )
            except aoi.AoIError as exc:
                st.caption(f"AOI pending: {exc}")
            else:
                if resolved_sb is not None:
                    st.caption(
                        f"Resolved AOI ({resolved_sb.source}): **{resolved_sb.label}** · "
                        f"`{resolved_sb.format_bbox()}`"
                    )
        spans = _spans()
        with st.expander("Data coverage (local archive & NASA FIRMS)"):
            if spans:
                for dataset, span in sorted(spans.items()):
                    st.write(f"Local archive ({dataset}): {span['min']} to {span['max']}")
            else:
                st.caption("Local catalog coverage metadata unavailable.")
            st.caption(
                "NASA FIRMS live sources: MODIS (2000-11-01–present), "
                "VIIRS S-NPP (2012-01-20–present), VIIRS NOAA-20 (2018-04-01–present), "
                "VIIRS NOAA-21 (2023-03-01–present)."
            )
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
        with st.expander("Deployment data & AI status"):
            from src.calendar.agents import providers as providers_mod

            ov = query.storage_overview()
            pstat = providers_mod.provider_status()
            rows = [
                ("Calendar (2D)", ov["artifacts"]["2D_metrics"]["available"]),
                ("Map Detections (DuckDB)", ov["duckdb"]["available"]),
                ("Map Clusters (2C)", ov["artifacts"]["2C_clusters"]["available"]),
                ("Analytics (4C)", ov["artifacts"]["4C_anomalies"]["available"]),
                ("Events DB", ov["events_db"]["available"]),
                ("Export Ready", ov["duckdb"]["available"]
                 and ov["artifacts"]["2B_members"]["available"]
                 and ov["artifacts"]["2C_members"]["available"]),
                ("NASA FIRMS Live API", bool(settings.get_setting("FIRMS_MAP_KEY", ""))),
                ("IBM Granite", pstat["granite"]["available"]),
                ("Qwen", pstat["qwen"]["available"]),
            ]
            for label, ok in rows:
                st.caption(f"{label}: {'ready' if ok else 'unavailable'}")
            st.caption(f"AI effective mode: `{pstat['effective_mode']}` (deterministic rules always ready)")
        st.info(calendar_view.METHODOLOGY_NOTE)


def _about_tab(metrics: dict | None) -> None:
    metrics = metrics or {}
    st.header("About this data")
    st.write("FireAgent Calendar combines a local 2023 USA baseline archive "
             "(482,672 accepted MODIS and VIIRS S-NPP detections with pinned "
             "Phase 2–4 analytics) with multi-year, global NASA FIRMS "
             "retrieval across MODIS (Terra, Aqua) and VIIRS (Suomi-NPP, "
             "NOAA-20, NOAA-21). Archived outputs read directly from verified "
             "artifacts; newly fetched windows execute dynamically with explicit provenance.")
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
    _avail_about = sensors.evaluate_availability(
        str(st.session_state.get(filters.KEY_RANGE_START, filters.INITIAL_DAY)),
        str(st.session_state.get(filters.KEY_RANGE_END, filters.INITIAL_DAY)),
        tuple(st.session_state.get(filters.KEY_PRODUCTS, list(query.PRODUCTS))),
        tuple(st.session_state.get(filters.KEY_SATELLITES, list(query.SATELLITES))),
        firms_key_configured=bool(settings.get_setting("FIRMS_MAP_KEY", "")),
        live_dataset=st.session_state.get("cal_live_regional"),
    )
    for lim in _avail_about["limitations"]:
        st.write(f"- {lim}")
    st.write("- Single-linkage chaining can merge nearby activity into very "
             "large clusters; cluster size is not fire size.")
    st.write("- Preset region extents are approximate bounding boxes, not "
             "official administrative boundaries.")


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
