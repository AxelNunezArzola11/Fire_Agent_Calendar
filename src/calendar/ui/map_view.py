"""Fire Activity Map (Phase 3C): detections + analytical clusters.

Pure popup/legend/grid helpers (unit-tested) plus folium builders and one
thin Streamlit `render_map_tab`. Detection markers and cluster markers are
never mixed in one layer; visual density grids are display-only and labeled.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Callable

import folium
import pandas as pd
from folium.plugins import MarkerCluster

from .. import settings
from ..acquisition import sensors
from ..processing import regions
from . import filters, query


@dataclass
class MapCallbacks:
    """Injected query callables (cached wrappers in the app; fakes in tests)."""
    fetch_det: Callable[..., pd.DataFrame]
    count_det: Callable[..., int]
    fetch_clu: Callable[..., pd.DataFrame]
    count_clu: Callable[..., int]
    by_det: Callable[[str], dict | None]
    member: Callable[[str], dict | None]
    group: Callable[[str], dict | None]
    by_clu: Callable[[str], dict | None]
    build_det_map: Callable[..., folium.Map]
    build_clu_map: Callable[..., folium.Map]
    fetch_cluster_dets: Callable[..., pd.DataFrame]

CLUSTERS_LABEL = ("Analytical Hotspot Clusters — "
                  "Not Confirmed Wildfire Perimeters")
# Display-only warning threshold (pilot p99 cluster size is 28; anything
# above 100 is extreme chaining, never a single incident).
LARGE_CLUSTER_WARN = 100
CLUSTER_TITLE = "Analytical Hotspot Clusters"

SENSOR_COLORS = {"MODIS": "#1f77b4", "VIIRS": "#2ca02c"}
FRP_HIGH, FRP_MID, FRP_LOW = "#e74c3c", "#f39c12", "#ffd400"
BOUNDARY_COLOR = "#3388ff"
EXTENT_COLOR = "#777777"

CARTO_VOYAGER = "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"
CARTO_ATTR = "© OpenStreetMap contributors © CARTO"
OSM_STANDARD = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
OSM_ATTR = "© OpenStreetMap contributors"
ESRI_IMAGERY = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                "World_Imagery/MapServer/tile/{z}/{y}/{x}")
ESRI_ATTR = "Esri, Maxar, Earthstar Geographics"

def carto_url(api_key: str) -> str:
    """Keyed CARTO Voyager template (keyless endpoint watermarks)."""
    return CARTO_VOYAGER + f"?api_key={api_key}"


def select_tile_providers() -> tuple:
    """Ordered usable basemap providers as (name, url, attribution).

    OpenStreetMap first (keyless; needs network access and a browser
    user agent per OSM tile policy), Esri WorldImagery second.
    CartoDB Voyager is appended ONLY when the operator explicitly
    enables it (CAL_CARTO_ENABLED=true) AND configures CARTO_API_KEY:
    the keyless endpoint returns "API KEY REQUIRED" watermarks
    (pixel-verified), and the previously configured key was verified
    to return the same watermark, so key presence alone never
    qualifies. Tile keys are inherently browser-visible; operators
    must use a restricted basemap key.
    """
    providers = [("OpenStreetMap", OSM_STANDARD, OSM_ATTR),
                 ("Esri WorldImagery", ESRI_IMAGERY, ESRI_ATTR)]
    if settings.CAL_CARTO_ENABLED.lower() == "true" \
            and settings.CARTO_API_KEY:
        providers.append(
            ("CartoDB Voyager", carto_url(settings.CARTO_API_KEY),
             CARTO_ATTR))
    return tuple(providers)


def tile_source_line() -> str:
    """UI caption naming the active basemap and its alternates."""
    names = [name for name, _, _ in select_tile_providers()]
    rest = f" (alternates in layer control: {', '.join(names[1:])})" \
        if len(names) > 1 else ""
    return f"Basemap: {names[0]}{rest}."
TILE_OFFLINE_NOTE = ("Basemap tiles need network access. If the background "
                     "stays white, enable Graticule (offline reference grid) "
                     "in the layer control; points and overlays render offline.")
GRATICULE_NAME = "Graticule (offline reference grid)"
GRATICULE_COLOR = "#999999"
# Rounded pilot-cover extent (catalog: lon -164..-67, lat 19..68).
DEFAULT_GRATICULE_BOUNDS = (-170.0, 15.0, -60.0, 72.0)

KEY_COLOR_MODE = "cal_map_color_mode"
KEY_SHOW_DENSITY = "cal_map_density"
KEY_SHOW_EXTENT = "cal_map_extent"
KEY_SHOW_BOUNDARY = "cal_map_boundary"
KEY_MAP_BBOX = "cal_map_bbox"
KEY_SELECTED_ID = "cal_selected_id"
KEY_LAST_BOUNDS = "cal_last_bounds"
KEY_VIEW_FILTER_FLAG = "cal_view_filter_flag"
KEY_TO_CAL_FLAG = "cal_to_cal_flag"


def _viewport_to_aoi() -> None:
    """on_click: capture last map bounds as a viewport AOI (pre-script)."""
    import streamlit as st

    parsed = bounds_to_bbox(st.session_state.get(KEY_LAST_BOUNDS))
    if parsed:
        st.session_state[KEY_MAP_BBOX] = parsed
        st.session_state[filters.KEY_AOI_MODE] = "viewport"
        st.session_state[KEY_VIEW_FILTER_FLAG] = "applied"
    else:
        st.session_state[KEY_VIEW_FILTER_FLAG] = "missing"


def _clear_area_filter() -> None:
    """on_click: reset AOI to the full pilot (pre-script)."""
    import streamlit as st

    st.session_state[KEY_MAP_BBOX] = None
    st.session_state[filters.KEY_AOI_MODE] = "full"


def _timeline_day_to_range(day: str) -> None:
    """on_click: narrow the shared range to one playback day (pre-script)."""
    import streamlit as st

    st.session_state[filters.KEY_RANGE_START] = day
    st.session_state[filters.KEY_RANGE_END] = day


def _map_to_calendar() -> None:
    """on_click: point the calendar at the shared range start (pre-script)."""
    import streamlit as st

    from . import calendar_view

    try:
        start = filters.coerce_day(
            st.session_state.get(filters.KEY_RANGE_START, "2023-01-01"))
        filters.coerce_day(st.session_state.get(filters.KEY_RANGE_END, start))
    except ValueError:
        return
    st.session_state[filters.KEY_GRAIN] = "daily"
    st.session_state[filters.KEY_PERIOD] = start
    year, month, day = (int(v) for v in start.split("-"))
    st.session_state[calendar_view.KEY_VIEW_YEAR] = year
    st.session_state[calendar_view.KEY_VIEW_MONTH] = month
    st.session_state[calendar_view.KEY_DAY_PICKER] = day
    st.session_state[KEY_TO_CAL_FLAG] = True


def fmt(value) -> str:
    """Missing-tolerant display (None/NaN -> 'unavailable', never blank)."""
    if value is None:
        return "unavailable"
    if isinstance(value, float) and math.isnan(value):
        return "unavailable"
    if value != value:  # pandas NaT / NA
        return "unavailable"
    return str(value)


def frp_color(frp: float | None, q50: float, q75: float) -> str:
    """FRP quartile color (dashboard idiom; relative to the current view)."""
    if frp is None or (isinstance(frp, float) and math.isnan(frp)):
        return "#95a5a6"
    if frp >= q75:
        return FRP_HIGH
    if frp >= q50:
        return FRP_MID
    return FRP_LOW


def detection_popup_html(det: dict, member: dict | None = None) -> str:
    """Popup for one detection (missing fields say unavailable)."""
    lines = [
        f"<b>{fmt(det.get('detection_id'))}</b>",
        f"{fmt(det.get('instrument'))} / {fmt(det.get('satellite'))}",
        f"Acquired (UTC): {fmt(det.get('acquisition_timestamp_utc'))}",
        f"Lat/Lon: {fmt(det.get('latitude'))}, {fmt(det.get('longitude'))}",
        f"FRP: {fmt(det.get('frp'))} MW | Confidence: {fmt(det.get('confidence'))}",
        f"Day/Night: {fmt(det.get('daynight'))}",
    ]
    if member is not None:
        lines.append(f"Match: {fmt(member.get('classification'))} "
                     f"({fmt(member.get('match_id'))})")
    lines.append("Select for full detail + provenance.")
    return "<br>".join(lines)


def cluster_popup_html(clu: dict) -> str:
    """Popup for one analytical cluster (never fire language)."""
    n = clu.get("n_detections", 0)
    lines = [
        f"<b>{fmt(clu.get('cluster_id'))}</b> — {CLUSTER_TITLE}",
        f"Observations: {fmt(n)} "
        f"(MODIS {fmt(clu.get('n_modis'))} / VIIRS {fmt(clu.get('n_viirs'))})",
        f"First: {fmt(clu.get('first_observation_utc'))}",
        f"Last: {fmt(clu.get('last_observation_utc'))}",
        f"Observation span: {fmt(clu.get('observation_span_h'))} h",
        f"FRP sum: {fmt(clu.get('frp_sum'))} MW",
    ]
    if isinstance(n, (int, float)) and n > LARGE_CLUSTER_WARN:
        lines.append(f"<b>Very large observation group ({n:,}); likely "
                     f"chaining — not a single incident.</b>")
    return "<br>".join(lines)


def bounds_to_bbox(bounds) -> tuple[float, float, float, float] | None:
    """st_folium bounds -> (W, S, E, N); None when unparseable."""
    try:
        if isinstance(bounds, dict):  # {"_southWest": {lat,lng}, ...}
            sw, ne = bounds["_southWest"], bounds["_northEast"]
            return (float(sw["lng"]), float(sw["lat"]),
                    float(ne["lng"]), float(ne["lat"]))
        (s, w), (n, e) = bounds  # [[s,w],[n,e]]
        return (float(w), float(s), float(e), float(n))
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def zoom_to_grid(zoom: float | None) -> float:
    """Zoom -> display-grid resolution in degrees (labeled in the UI)."""
    if zoom is None:
        return 1.0
    if zoom <= 4:
        return 2.0
    if zoom <= 6:
        return 1.0
    if zoom <= 8:
        return 0.5
    return 0.25


def density_grid(df: pd.DataFrame, grid_deg: float) -> pd.DataFrame:
    """Display-only FRP grid over bounded rows (never analytical clusters)."""
    if df.empty:
        return pd.DataFrame(columns=["lon0", "lat0", "n", "frp_sum"])
    lon = (df["longitude"].to_numpy(dtype=float) // grid_deg) * grid_deg
    lat = (df["latitude"].to_numpy(dtype=float) // grid_deg) * grid_deg
    grouped = pd.DataFrame({"lon0": lon, "lat0": lat,
                            "frp": pd.to_numeric(df["frp"], errors="coerce")})
    agg = grouped.groupby(["lon0", "lat0"], as_index=False).agg(
        n=("frp", "size"), frp_sum=("frp", "sum"))
    agg["frp_sum"] = agg["frp_sum"].fillna(0.0)
    return agg


def _base_map() -> folium.Map:
    # prefer_canvas: display-only rendering hint so 5k-marker views stay
    # interactive (SVG nodes crawl; canvas does not). No visual or
    # semantic change to layers, colors, popups, or bounds.
    param = folium.Map(location=[39.8, -98.6], zoom_start=4, tiles=None,
                       prefer_canvas=True)
    for i, (name, url, attr) in enumerate(select_tile_providers()):
        folium.TileLayer(tiles=url, attr=attr, name=name,
                         show=(i == 0)).add_to(param)
    return param


def graticule_step(span_deg: float) -> float:
    """Grid spacing for a degree span (keeps line counts bounded)."""
    if span_deg > 60:
        return 10.0
    if span_deg > 20:
        return 5.0
    if span_deg > 5:
        return 2.0
    return 1.0


def graticule_layer(bounds: tuple[float, float, float, float],
                    step_deg: float) -> folium.FeatureGroup:
    """Lat/lon reference grid (pure geometry; needs no data or network).

    Meridians/parallels snap to multiples of step_deg. Hidden by
    default; the offline fallback when no tile provider is reachable.
    """
    import math as _math

    w, s, e, n = bounds
    layer = folium.FeatureGroup(name=GRATICULE_NAME, show=False)
    lon = _math.ceil(w / step_deg) * step_deg
    while lon <= e + 1e-9:
        folium.PolyLine(locations=[[s, lon], [n, lon]],
                        color=GRATICULE_COLOR, weight=1, opacity=0.7,
                        tooltip=f"{lon:g}°E" if lon >= 0 else f"{-lon:g}°W",
                        ).add_to(layer)
        lon += step_deg
    lat = _math.ceil(s / step_deg) * step_deg
    while lat <= n + 1e-9:
        folium.PolyLine(locations=[[lat, w], [lat, e]],
                        color=GRATICULE_COLOR, weight=1, opacity=0.7,
                        tooltip=f"{lat:g}°N" if lat >= 0 else f"{-lat:g}°S",
                        ).add_to(layer)
        lat += step_deg
    return layer


def _graticule_for(fit: tuple[float, float, float, float] | None
                   ) -> folium.FeatureGroup:
    """Graticule sized to the fitted bounds (or the pilot extent)."""
    bounds = fit if fit else DEFAULT_GRATICULE_BOUNDS
    w, s, e, n = bounds
    return graticule_layer(bounds, graticule_step(max(e - w, n - s)))


def map_data_key(*, mode: str, df, id_col: str,
                   display: tuple = ()) -> str:
    """Stable component key bound to the map's exact content.

    streamlit-folium only builds the Leaflet map on first render inside
    an iframe; later renders with the same component key reuse the old
    markers even when the data changed. Binding the key to the drawn
    row ids + display options forces a remount exactly when the content
    changes (and never on unrelated reruns: typing in the id box keeps
    the same key). No randomness: identical inputs always yield the
    identical key. Viewport resets on remount by design (new data gets
    a fresh fitted view); "Filter to current map view" remains the way
    to persist bounds.
    """
    import hashlib

    digest = hashlib.sha256()
    digest.update(mode.encode())
    digest.update(repr(tuple(display)).encode())
    if len(df):
        digest.update("\n".join(df[id_col].astype(str).tolist()).encode())
    return f"cal_map_{digest.hexdigest()[:12]}"


def copy_map(param):
    """Independent deep copy of a built map (per-render isolation).

    The component serializer mutates the object it processes, so a
    cached map must never be rendered directly: concurrent renders of
    the shared object raise "OrderedDict mutated during iteration".
    """
    import copy

    return copy.deepcopy(param)


def sanitize_frame(df: pd.DataFrame,
                   *pairs: tuple[str, str]) -> tuple[pd.DataFrame, int]:
    """Split finite-coordinate rows from invalid ones (display-only).

    Each pair is a (lat_col, lon_col) feeding Leaflet geometry; a row
    must pass every pair. Returns (clean_frame, dropped_count). A
    single NaN coordinate would poison Leaflet (`L.circleMarker([nan,
    nan])` throws and blanks the whole map) with no server-side error,
    so invalid rows are dropped here and the count is surfaced as a
    caption by the caller. Never touches analytical data.
    """
    import numpy as np

    if df.empty:
        return df.copy(), 0
    mask = pd.Series(True, index=df.index)
    for lat_col, lon_col in pairs:
        if lat_col not in df.columns or lon_col not in df.columns:
            return df.iloc[0:0].copy(), int(len(df))
        lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy(dtype=float)
        lon = pd.to_numeric(df[lon_col], errors="coerce").to_numpy(dtype=float)
        mask &= pd.Series(np.isfinite(lat) & np.isfinite(lon), index=df.index)
    return df[mask].copy(), int((~mask).sum())


def boundary_layer(
    region: regions.Region | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    label: str = "",
) -> folium.FeatureGroup:
    """Active region or California approximate-bbox overlay (labeled, versioned).

    Administrative polygons do not ship with the app, so the active bounding
    box is drawn explicitly with its CRS/provenance label and never presented
    as an official administrative border.
    """
    if region is None and bbox is not None:
        w, s, e, n = (float(v) for v in bbox)
        title = label or "Selected Area of Interest"
        popup_txt = f"{title} — user-defined bounding box (EPSG:4326). NOT an official boundary."
        tip_txt = f"{title} (bbox)"
    else:
        reg = region or regions.CALIFORNIA
        w, s, e, n = reg.bbox
        title = reg.label or reg.id
        popup_txt = f"{title} — {reg.source} ({reg.version}; {reg.crs})."
        tip_txt = f"{title} (approx. bbox)"
    layer = folium.FeatureGroup(name="Boundaries (approx.)", show=True)
    folium.Rectangle(
        bounds=[[s, w], [n, e]], color=BOUNDARY_COLOR, weight=2, fill=False,
        popup=popup_txt,
        tooltip=tip_txt,
    ).add_to(layer)
    return layer


def cap_notice(total: int, shown: int, what: str) -> str:
    if total <= shown:
        return f"Showing all {total:,} {what}."
    return (f"Showing {shown:,} of {total:,} {what} "
            f"(highest relevance first). Narrow the date range or area.")


def build_detection_map(df: pd.DataFrame, *, color_mode: str = "sensor",
                        show_density: bool = False, grid_deg: float = 1.0,
                        show_boundary: bool = True,
                        fit: tuple[float, float, float, float] | None = None,
                        region: regions.Region | None = None,
                        region_id: str | None = None,
                        bbox_label: str = "",
                        ) -> folium.Map:
    """Detection Mode map (per-satellite layers + optional FRP density)."""
    param = _base_map()
    if region is None and region_id:
        region = regions.REGIONS.get(region_id)
    df, dropped = sanitize_frame(df, ("latitude", "longitude"))
    param.dropped_rows = dropped
    frp = (pd.to_numeric(df["frp"], errors="coerce")
           if "frp" in df.columns and not df.empty
           else pd.Series([], dtype=float))
    q50, q75 = (float(frp.quantile(0.5)), float(frp.quantile(0.75))) if len(frp) else (0.0, 0.0)
    groups: dict[str, folium.FeatureGroup] = {}
    clusters: dict[str, MarkerCluster] = {}

    def _group_for(satellite: str) -> MarkerCluster:
        if satellite not in groups:  # never silently drop a row
            label = query.SATELLITE_LABELS.get(satellite, f"Satellite {satellite}")
            groups[satellite] = folium.FeatureGroup(name=label, show=True)
            clusters[satellite] = MarkerCluster(name=f"{label} cluster")
            clusters[satellite].add_to(groups[satellite])
        return clusters[satellite]

    for satellite in query.SATELLITES:  # stable layers across all 5 sensors even when empty
        _group_for(satellite)
    for row in df.itertuples(index=False):
        sat = sensors.canonical_satellite(
            str(getattr(row, "satellite", "")),
            product=str(getattr(row, "product", "") or ""),
        )
        color = (SENSOR_COLORS.get(row.instrument, "#7f7f7f")
                 if color_mode == "sensor"
                 else frp_color(getattr(row, "frp", None), q50, q75))
        folium.CircleMarker(
            location=[row.latitude, row.longitude], radius=5, color=color,
            fill=True, fill_opacity=0.7, weight=1,
            popup=detection_popup_html(row._asdict()),
            tooltip=str(row.detection_id),
        ).add_to(_group_for(sat))
    for group in groups.values():
        group.add_to(param)
    if show_density and not df.empty:
        grid = density_grid(df, grid_deg)
        vmax = float(grid["frp_sum"].max()) if len(grid) else 0.0
        layer = folium.FeatureGroup(name=f"FRP density ({grid_deg}° grid)", show=False)
        for cell in grid.itertuples(index=False):
            frac = (cell.frp_sum / vmax) if vmax > 0 else 0
            folium.Rectangle(
                bounds=[[cell.lat0, cell.lon0],
                        [cell.lat0 + grid_deg, cell.lon0 + grid_deg]],
                color=frp_color(cell.frp_sum, vmax * 0.5, vmax * 0.75),
                fill=True, fill_opacity=0.35 + 0.35 * frac, weight=0.5,
                popup=(f"Display grid {grid_deg}°: {cell.n:,} observations, "
                       f"FRP sum {cell.frp_sum:,.1f} MW (reported MW, not energy)."),
            ).add_to(layer)
        layer.add_to(param)
    if show_boundary:
        boundary_layer(region=region, bbox=fit, label=bbox_label).add_to(param)
    _graticule_for(fit).add_to(param)
    folium.LayerControl().add_to(param)
    if fit:
        w, s, e, n = fit
        param.fit_bounds([[s, w], [n, e]])
    return param


def build_cluster_map(df: pd.DataFrame, *, show_extent: bool = True,
                      show_boundary: bool = True,
                      fit: tuple[float, float, float, float] | None = None,
                      region: regions.Region | None = None,
                      region_id: str | None = None,
                      bbox_label: str = "",
                      ) -> folium.Map:
    """Cluster Mode map (centroids + bbox extents; never perimeters)."""
    param = _base_map()
    if region is None and region_id:
        region = regions.REGIONS.get(region_id)
    df, dropped = sanitize_frame(df, ("centroid_lat", "centroid_lon"),
                                   ("lat_min", "lon_min"),
                                   ("lat_max", "lon_max"))
    param.dropped_rows = dropped
    layer = folium.FeatureGroup(name=CLUSTER_TITLE, show=True)
    cluster = MarkerCluster(name="clustered centroids").add_to(layer)
    for row in df.itertuples(index=False):
        size = max(4, min(14, 3 + math.log10(max(1, row.n_detections)) * 4))
        folium.CircleMarker(
            location=[row.centroid_lat, row.centroid_lon], radius=size,
            color="#333333", fill=True, fill_color="#666666", fill_opacity=0.7,
            weight=1, popup=cluster_popup_html(row._asdict()),
            tooltip=str(row.cluster_id),
        ).add_to(cluster)
    layer.add_to(param)
    if show_extent and not df.empty:
        ext = folium.FeatureGroup(name="Observation extents (bbox)", show=True)
        for row in df.itertuples(index=False):
            folium.Rectangle(
                bounds=[[row.lat_min, row.lon_min], [row.lat_max, row.lon_max]],
                color=EXTENT_COLOR, weight=1, dash_array="4,4", fill=False,
                popup=(f"{row.cluster_id}: observation bounding box "
                       f"({row.n_detections:,} observations). Not a fire perimeter."),
                tooltip=f"{row.cluster_id} extent",
            ).add_to(ext)
        ext.add_to(param)
    if show_boundary:
        boundary_layer(region=region, bbox=fit, label=bbox_label).add_to(param)
    _graticule_for(fit).add_to(param)
    folium.LayerControl().add_to(param)
    if fit:
        w, s, e, n = fit
        param.fit_bounds([[s, w], [n, e]])
    return param


def legend_html(mode: str, color_mode: str, grid_deg: float | None = None,
                boundary_label: str = "Active region / pilot bbox") -> str:
    """Map legend (honest styles; no severity language)."""
    if mode == "clusters":
        return f"""
        <div style="position:fixed;bottom:30px;right:10px;background:white;
        border:1px solid #ccc;border-radius:6px;padding:10px 14px;font-size:12px;
        line-height:1.7;z-index:9999;max-width:270px">
        <b>{CLUSTERS_LABEL}</b><br>
        <span style="color:#666">&#9679;</span> Centroid (size grows with
        observation count)<br>
        <span style="color:{EXTENT_COLOR}">&#9645;</span> Observation bounding
        box (dashed) — not a fire perimeter<br>
        <span style="color:{BOUNDARY_COLOR}">&#9645;</span> {boundary_label}
         — not an official boundary<br>
        <span style="color:{GRATICULE_COLOR}">&#9783;</span> Graticule in
        layer control when tiles are unreachable
        </div>"""
    frp_note = (f"<br>Density grid {grid_deg}°: sums of reported MW in the "
                f"selected period, not energy.") if grid_deg else ""
    if color_mode == "frp":
        dots = (f'<span style="color:{FRP_HIGH}">&#9679;</span> High FRP '
                f"(top quartile of this view)<br>"
                f'<span style="color:{FRP_MID}">&#9679;</span> Mid FRP<br>'
                f'<span style="color:{FRP_LOW}">&#9679;</span> Low FRP<br>')
    else:
        dots = (f'<span style="color:{SENSOR_COLORS["MODIS"]}">&#9679;</span> '
                f"MODIS (Terra + Aqua)<br>"
                f'<span style="color:{SENSOR_COLORS["VIIRS"]}">&#9679;</span> '
                f"VIIRS Suomi-NPP<br>")
    return f"""
    <div style="position:fixed;bottom:30px;right:10px;background:white;
    border:1px solid #ccc;border-radius:6px;padding:10px 14px;font-size:12px;
    line-height:1.7;z-index:9999;max-width:270px">
    <b>Satellite detections</b><br>{dots}
    <span style="color:{BOUNDARY_COLOR}">&#9645;</span> {boundary_label}
     — not an official boundary<br>
    <span style="color:{GRATICULE_COLOR}">&#9783;</span> Graticule in layer
    control when tiles are unreachable{frp_note}
    </div>"""


def member_id_list(cluster: dict, limit: int = 50) -> tuple[list[str], int]:
    """Bounded member-id preview + total (giant clusters stay bounded)."""
    try:
        ids = json.loads(cluster.get("detection_ids") or "[]")
    except (json.JSONDecodeError, TypeError):
        ids = []
    return ids[:limit], len(ids)


TIMELINE_REQUIRED = ("acquisition_timestamp_utc", "product", "frp")
TIMELINE_COLUMNS = ("day", "n", "n_modis", "n_viirs", "frp_modis",
                    "frp_viirs", "frp_missing")


def _frame_days(df: pd.DataFrame) -> pd.Series:
    """UTC calendar day per row (NaN when the timestamp is unusable)."""
    return pd.to_datetime(df["acquisition_timestamp_utc"], utc=True,
                          format="mixed", errors="coerce").dt.strftime("%Y-%m-%d")


def cluster_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """Per-UTC-day observation counts + per-sensor FRP (sorted by day).

    Missing FRP is excluded from sums and counted, never zero-filled;
    real zeros contribute. Undated rows are excluded (no silent day).
    """
    if df.empty:
        return pd.DataFrame(columns=list(TIMELINE_COLUMNS))
    missing = [c for c in TIMELINE_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"cluster_timeline needs columns: {missing}")
    work = df.assign(_day=_frame_days(df)).dropna(subset=["_day"])
    rows = []
    for day, part in work.groupby("_day", sort=True):
        frp = pd.to_numeric(part["frp"], errors="coerce")
        modis = part["product"] == "modis"
        viirs = part["product"] == "viirs-snpp"
        rows.append({
            "day": day,
            "n": int(len(part)),
            "n_modis": int(modis.sum()),
            "n_viirs": int(viirs.sum()),
            "frp_modis": float(frp[modis].sum()),
            "frp_viirs": float(frp[viirs].sum()),
            "frp_missing": int(frp.isna().sum()),
        })
    return pd.DataFrame(rows, columns=list(TIMELINE_COLUMNS))


def playback_slice(df: pd.DataFrame, day: str,
                   cumulative: bool = False) -> pd.DataFrame:
    """Member rows observed on one UTC day (or up to it if cumulative)."""
    if df.empty:
        return df.copy()
    days = _frame_days(df)
    mask = days <= day if cumulative else days == day
    return df[mask.fillna(False)].copy()


# --- Streamlit rendering (thin wrapper; logic stays in pure helpers) ---


def _render_detection_detail(detection_id: str, cb: MapCallbacks) -> None:
    import streamlit as st

    try:
        det = cb.by_det(detection_id)
    except FileNotFoundError as exc:
        st.warning(f"Detection detail unavailable: {exc}")
        return
    if det is None:
        st.warning(f"No detection with id {detection_id!r} in the pilot data.")
        return
    st.write(f"### Detection {det['detection_id']}")
    left, right = st.columns(2)
    with left:
        st.write("Sensor:", fmt(det.get("sensor")))
        st.write("Satellite:", fmt(det.get("satellite")))
        st.write("Instrument:", fmt(det.get("instrument")))
        st.write("Product:", f"{fmt(det.get('product'))} {fmt(det.get('product_version'))}")
        st.write("Acquired (UTC):", fmt(det.get("acquisition_timestamp_utc")))
        st.write("Latitude:", fmt(det.get("latitude")))
        st.write("Longitude:", fmt(det.get("longitude")))
    with right:
        st.write("Confidence:", fmt(det.get("confidence")))
        st.write("FRP (MW):", fmt(det.get("frp")))
        st.write("Day/Night:", fmt(det.get("daynight")))
        st.write("Scan/Track:", fmt(det.get("scan")), "/", fmt(det.get("track")))
        st.write("Source file:", fmt(det.get("source_file")))
        st.write("Source SHA-256:", (fmt(det.get("source_sha256"))[:16] + "…")
                 if det.get("source_sha256") else "unavailable")
    try:
        member = cb.member(detection_id)
    except FileNotFoundError:
        member = None
    if member is None:
        st.caption("Match record: unavailable.")
        return
    st.write("Match classification:", fmt(member.get("classification")),
             "| Match id:", fmt(member.get("match_id")),
             f"| Best score: {fmt(member.get('best_score'))}")
    st.caption(f"Matching: matcher_v1 / {fmt(member.get('params_version'))} "
               f"(exec {fmt(member.get('exec_id'))[:8]}).")
    match_id = member.get("match_id")
    if match_id:
        try:
            group = cb.group(match_id)
        except FileNotFoundError:
            group = None
        if group is not None:
            with st.expander(f"Fellow members of {match_id} (bounded preview)"):
                try:
                    fellows = (json.loads(group.get("modis_ids") or "[]")
                               + json.loads(group.get("viirs_ids") or "[]"))
                except (json.JSONDecodeError, TypeError):
                    fellows = []
                st.write(f"{len(fellows)} members, "
                         f"evidence {fmt(group.get('evidence'))}:")
                st.write(sorted(fellows)[:50])


def _render_cluster_detail(cluster_id: str, cb: MapCallbacks) -> None:
    import streamlit as st

    try:
        clu = cb.by_clu(cluster_id)
    except FileNotFoundError as exc:
        st.warning(f"Cluster detail unavailable: {exc}")
        return
    if clu is None:
        st.warning(f"No cluster with id {cluster_id!r} in the pilot data.")
        return
    st.write(f"### Cluster {clu['cluster_id']}")
    st.caption(CLUSTERS_LABEL)
    try:
        # Deferred import: events.association imports this module's
        # display constant, so a top-level import would cycle.
        from ..events.association import event_id_for

        st.caption(f"Activity event {event_id_for(clu['cluster_id'])} "
                   f"(Events & Alerts tab → Inspect event).")
    except (ImportError, ValueError):
        pass
    n = clu.get("n_detections", 0)
    if isinstance(n, (int, float)) and n > LARGE_CLUSTER_WARN:
        st.warning(f"Very large observation group ({n:,} observations): likely "
                   f"single-linkage chaining of nearby activity — not a single "
                   f"incident, and its extent is not a fire perimeter.")
    cols = st.columns(4)
    cols[0].metric("Observations", f"{fmt(n)}")
    cols[1].metric("MODIS", f"{fmt(clu.get('n_modis'))}")
    cols[2].metric("VIIRS", f"{fmt(clu.get('n_viirs'))}")
    cols[3].metric("Observation span (h)", f"{fmt(clu.get('observation_span_h'))}")
    left, right = st.columns(2)
    with left:
        st.write("Contributing satellites:", fmt(clu.get("satellites")))
        st.write("First observation (UTC):", fmt(clu.get("first_observation_utc")))
        st.write("Last observation (UTC):", fmt(clu.get("last_observation_utc")))
        st.write("Centroid (lon, lat):", fmt(clu.get("centroid_lon")) + ", "
                 + fmt(clu.get("centroid_lat")))
        st.write("Extent (bbox):", f"lon {fmt(clu.get('lon_min'))}…{fmt(clu.get('lon_max'))}, "
                 f"lat {fmt(clu.get('lat_min'))}…{fmt(clu.get('lat_max'))}")
    with right:
        st.write("FRP sum (MW):", fmt(clu.get("frp_sum")),
                 "| mean:", fmt(clu.get("frp_mean")), "| max:", fmt(clu.get("frp_max")))
        st.write("FRP contributing:", fmt(clu.get("frp_n")),
                 "| missing:", fmt(clu.get("frp_null")))
        st.write("FRP by sensor:", fmt(clu.get("frp_by_sensor")))
        st.write("Match groups:", fmt(clu.get("match_groups")))
        st.write("Analytical ref:",
                 f"{fmt(clu.get('algo_version'))} / {fmt(clu.get('params_version'))} "
                 f"(exec {fmt(clu.get('exec_id'))[:8]})")
    preview, total = member_id_list(clu)
    with st.expander(f"Member detection ids ({total:,} total, first {len(preview)} shown)"):
        st.write(preview)
    with st.expander("Observation timeline (playback)"):
        try:
            timeline_df = cb.fetch_cluster_dets(cluster_id)
        except Exception as exc:
            st.caption(f"Timeline unavailable: {exc}")
            return
        if timeline_df is None or len(timeline_df) == 0:
            st.caption("No member observations found for this cluster.")
            return
        tl = cluster_timeline(timeline_df)
        if len(tl) == 0:
            st.caption("Member observations carry no usable timestamps.")
            return
        st.caption("Observation days for this cluster's members — stepping "
                   "days replays detections, never fire spread or perimeters.")
        st.bar_chart(tl, x="day", y="n")
        st.bar_chart(tl[["day", "frp_modis", "frp_viirs"]], x="day",
                     y=["frp_modis", "frp_viirs"])
        st.caption("Per-sensor FRP sums (MW); sensors are not "
                   "cross-comparable. "
                   f"{int(tl['frp_missing'].sum()):,} member(s) missing FRP, "
                   "excluded.")
        day = st.select_slider("Playback day", options=tl["day"].tolist(),
                               value=tl["day"].tolist()[-1],
                               key=f"cal_pb_day_{cluster_id}")
        cumulative = st.checkbox("Cumulative (up to selected day)", value=False,
                                 key=f"cal_pb_cum_{cluster_id}")
        part = playback_slice(timeline_df, day, cumulative=cumulative)
        sats = sorted({str(s) for s in part["satellite"].tolist() if s})
        st.write(f"{len(part):,} of {len(timeline_df):,} member observations "
                 f"on {day}{' (cumulative)' if cumulative else ''}"
                 + (f" — satellites: {', '.join(sats)}." if sats else "."))
        st.button("Show this day on the map", key=f"cal_pb_map_{cluster_id}",
                  on_click=_timeline_day_to_range, args=(day,),
                  help="Narrows the shared map range to this single day.")


def render_map_tab(spans: dict, cb: MapCallbacks) -> None:
    """Full map tab (mode layers + bounded queries + detail panel)."""
    from streamlit_folium import st_folium

    import streamlit as st

    from .. import settings

    from . import aoi as aoi_mod

    mode = st.session_state.get(filters.KEY_MODE, "detections")
    try:
        date_start = filters.coerce_day(st.session_state.get(filters.KEY_RANGE_START, "2023-01-01"))
        date_end = filters.coerce_day(st.session_state.get(filters.KEY_RANGE_END, "2023-12-31"))
    except ValueError as exc:
        st.error(f"Invalid filter: {exc}")
        return
    products = tuple(st.session_state.get(filters.KEY_PRODUCTS, list(query.PRODUCTS)))
    satellites = tuple(st.session_state.get(filters.KEY_SATELLITES, list(query.SATELLITES)))
    modis_conf = st.session_state.get(filters.KEY_MODIS_CONF, 0) or None
    viirs_conf = tuple(st.session_state.get(filters.KEY_VIIRS_CONF,
                                            list(query.VIIRS_CONFIDENCE)))
    try:
        resolved = aoi_mod.resolve_session_aoi(
            st.session_state,
            aoi_path=settings.CAL_AOI_PATH,
            include_session_fallback=True,
        )
    except aoi_mod.AoIError as exc:
        st.error(f"Area of interest: {exc}")
        return
    bbox = resolved.bbox if resolved is not None else None
    limit = settings.CAL_MAP_RENDER_LIMIT

    aoi_mode = st.session_state.get(filters.KEY_AOI_MODE, "full")
    active_reg_id = resolved.region_id if resolved is not None else None
    region_label = (
        resolved.label
        if resolved is not None
        else "USA / CONUS Archive Extent"
    )

    col_mode, col_opts = st.columns([1, 3])
    with col_mode:
        st.write(f"### {'Detection Mode' if mode == 'detections' else 'Cluster Mode'}")
    if date_start == date_end:
        st.caption(f"Showing acquisition date {date_start} (UTC) · Region: **{region_label}**.")
    else:
        st.caption(f"Showing acquisition range {date_start} to {date_end} (UTC) · Region: **{region_label}**.")
    st.caption(tile_source_line())
    st.caption(TILE_OFFLINE_NOTE)
    with col_opts:
        show_boundary = st.checkbox("Boundaries", value=True, key=KEY_SHOW_BOUNDARY,
                                    help="Active region or California approximate bbox (not official).")
    if mode == "detections":
        color_mode = st.radio("Marker color", ["sensor", "frp"], key=KEY_COLOR_MODE,
                              horizontal=True,
                              help="Sensor: blue MODIS, green VIIRS. FRP: quartiles "
                                   "of the current view (relative, not severity).")
        show_density = st.checkbox("FRP density grid (display-only)", value=False,
                                   key=KEY_SHOW_DENSITY)
    else:
        st.caption(CLUSTERS_LABEL)
        show_extent = st.checkbox("Observation extents (bbox, not perimeters)",
                                  value=True, key=KEY_SHOW_EXTENT)

    live_ds = st.session_state.get("cal_live_regional")
    fetch_bbox = bbox
    use_live_ds = (
        mode == "detections"
        and live_ds is not None
        and (fetch_bbox is None or live_ds.bbox == fetch_bbox)
        and live_ds.date_start == date_start
        and live_ds.date_end == date_end
    )

    def _offer_live_firms_fetch(context_msg: str) -> None:
        from ..pipeline import regional

        if fetch_bbox is None:
            st.info(
                "Select a Geographic area of interest in the sidebar (region, point + radius, "
                "custom bbox, or saved area) to retrieve live NASA FIRMS observations."
            )
            return
        firms_ready = bool(settings.get_setting("FIRMS_MAP_KEY", ""))
        avail = sensors.evaluate_availability(
            date_start,
            date_end,
            products,
            satellites,
            firms_key_configured=firms_ready,
            region_label=region_label,
        )
        for note in avail["sensor_notes"]:
            st.caption(note)
        if not firms_ready:
            st.caption(
                "Missing configuration: `FIRMS_MAP_KEY` in environment or Streamlit Secrets is required "
                "to retrieve NASA FIRMS data beyond the local 2023 archive."
            )
            return
        if not avail["firms_sources"]:
            st.caption("No NASA FIRMS satellite feed supports the current product/satellite/date combination.")
            return
        col_btn, col_note = st.columns([2, 3])
        with col_btn:
            if st.button(f"Retrieve live NASA FIRMS ({region_label})", key="cal_map_fetch_firms"):
                try:
                    with st.spinner(
                        f"Fetching NASA FIRMS ({', '.join(avail['firms_sources'])}) "
                        f"for {region_label} ({date_start}..{date_end})…"
                    ):
                        c_lat, c_lon, r_km = (
                            resolved.point_radius
                            if resolved is not None and resolved.point_radius is not None and len(resolved.point_radius) == 3
                            else (None, None, None)
                        )
                        ds = regional.fetch_firms_regional(
                            bbox=fetch_bbox,
                            date_start=date_start,
                            date_end=date_end,
                            region_id=active_reg_id or aoi_mode,
                            region_label=region_label,
                            products=products,
                            satellites=satellites,
                        )
                        ds.center_lat = c_lat
                        ds.center_lon = c_lon
                        ds.radius_km = r_km
                        ds.aoi_source = resolved.source if resolved is not None else aoi_mode
                        regional.store_regional_dataset(
                            ds,
                            aoi_source=ds.aoi_source,
                            center_lat=c_lat,
                            center_lon=c_lon,
                            radius_km=r_km,
                            requested_products=products,
                            requested_satellites=satellites,
                        )
                    st.session_state["cal_live_regional"] = ds
                    if ds.errors:
                        st.warning("; ".join(ds.errors))
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
        with col_note:
            st.caption(f"{context_msg} (max {regional.max_live_firms_days()} days per live fetch)")

    if use_live_ds:
        st.success(
            f"Data origin: **Newly retrieved NASA FIRMS API** · Region: **{live_ds.region_label}** · "
            f"Exec `{live_ds.exec_id[:8]}` · Sources: `{', '.join(live_ds.sources) or 'none'}` · "
            f"{live_ds.n_detections:,} normalized detections."
        )
        if st.button("Return to local 2023 archive view", key="cal_map_clear_live"):
            st.session_state.pop("cal_live_regional", None)
            st.rerun()
        df_live = live_ds.df
        if not df_live.empty:
            mask = pd.Series(True, index=df_live.index)
            if "product" in df_live.columns:
                mask &= df_live["product"].astype(str).isin(products or ())
            if "satellite" in df_live.columns:
                prods_col = df_live["product"] if "product" in df_live.columns else [""] * len(df_live)
                norm_sats = [
                    sensors.canonical_satellite(str(s), product=str(p))
                    for s, p in zip(df_live["satellite"], prods_col)
                ]
                mask &= pd.Series(norm_sats, index=df_live.index).isin(satellites or ())
            if modis_conf and "product" in df_live.columns and "confidence" in df_live.columns:
                is_modis = df_live["product"].astype(str) == "modis"
                conf_num = pd.to_numeric(df_live["confidence"], errors="coerce").fillna(-1)
                mask &= (~is_modis) | (conf_num >= modis_conf)
            if viirs_conf is not None and "product" in df_live.columns and "confidence" in df_live.columns:
                is_viirs = df_live["product"].astype(str).isin(("viirs-snpp", "viirs-noaa20", "viirs-noaa21"))
                mask &= (~is_viirs) | df_live["confidence"].astype(str).isin(viirs_conf)
            df_live = df_live.loc[mask]
        if df_live.empty:
            st.info(
                f"NASA FIRMS returned 0 matching detections for {live_ds.region_label} "
                f"between {date_start} and {date_end}. This is verified zero observations, not a failure."
            )
            return
        total = int(len(df_live))
        df = df_live.head(limit).copy()
        st.caption(cap_notice(total, len(df), "live FIRMS detections") + " Order: highest FRP first.")
        grid_deg = zoom_to_grid(st.session_state.get("cal_map_zoom"))
        param = cb.build_det_map(df, color_mode=color_mode,
                                 show_density=show_density, grid_deg=grid_deg,
                                 show_boundary=show_boundary, fit=bbox or live_ds.bbox,
                                 region_id=active_reg_id, bbox_label=region_label)
        map_key = map_data_key(
            mode=f"{mode}_live", df=df, id_col="detection_id",
            display=(live_ds.exec_id, color_mode, show_density,
                     grid_deg if show_density else None, show_boundary, bbox, products, satellites))
    else:
        if mode == "detections":
            db_st = query.duckdb_status()
            if not db_st["available"]:
                st.session_state[KEY_SELECTED_ID] = None
                st.warning("Map geospatial detection dataset is unavailable in this deployment.")
                st.info(
                    f"Required geospatial dataset not configured or present.\n\n"
                    f"- Configured path (`{db_st['env_var']}`): `{db_st['path'] or '(unconfigured)'}` "
                    f"(status: `{db_st['status']}`)\n"
                    f"- Detail: {db_st['reason']}\n"
                    f"- How to enable: {db_st['remediation']}\n\n"
                    "No synthetic fire detections or fictional map points are generated."
                )
                _offer_live_firms_fetch("Fetch global NASA FIRMS observations directly into this deployment.")
                return
        else:
            clu_st = query.artifact_status("2C", "clusters.parquet")
            if not clu_st["available"]:
                st.session_state[KEY_SELECTED_ID] = None
                st.warning("Map hotspot cluster artifact is unavailable in this deployment.")
                st.info(
                    f"Required cluster dataset is not present.\n\n"
                    f"- Missing artifact: `{clu_st['path'] or 'clusters.parquet'}` "
                    f"(status: `{clu_st['status']}`, exec `{clu_st['exec_id'][:8] if clu_st['exec_id'] else 'unconfigured'}`)\n"
                    f"- Configuration variables: `{clu_st['root_env_var']}`, `{clu_st['env_var']}`\n"
                    f"- Detail: {clu_st['reason']}\n\n"
                    "No synthetic hotspot clusters are generated."
                )
                return

        try:
            if mode == "detections":
                total = cb.count_det(date_start=date_start, date_end=date_end,
                                     products=products, satellites=satellites, bbox=bbox,
                                     modis_conf_min=modis_conf, viirs_conf=viirs_conf)
            else:
                total = cb.count_clu(date_start=date_start, date_end=date_end, bbox=bbox)
        except FileNotFoundError as exc:
            st.session_state[KEY_SELECTED_ID] = None
            st.warning("Map data source is unavailable in this deployment.")
            st.info(f"Required dataset could not be opened: {exc}. "
                    "No synthetic fire detections or clusters are generated.")
            return
        except ValueError as exc:
            st.error(f"Invalid filter: {exc}")
            return
        cov_start = query.coverage_for(date_start, spans)["status"]
        cov_end = query.coverage_for(date_end, spans)["status"]
        only_noaa_selected = bool(satellites) and not any(s in ("Terra", "Aqua", "N") for s in satellites)
        if cov_start == "unavailable" and cov_end == "unavailable" and total == 0:
            st.warning(
                f"Selected range ({date_start} to {date_end}) is outside the local 2023 USA archive "
                "(2023-01-01 to 2023-12-31): 0 local archive records. Retrieve live NASA FIRMS data below."
            )
            if mode == "detections":
                _offer_live_firms_fetch(f"Retrieve NASA FIRMS observations for {region_label} ({date_start}..{date_end}).")
            return
        if total == 0:
            if only_noaa_selected and mode == "detections":
                st.info(
                    "No records match these filters in the local 2023 archive because NOAA-20 / NOAA-21 "
                    "are not indexed in the local DuckDB file; retrieve them directly via NASA FIRMS below."
                )
            else:
                st.info(
                    f"No records match these filters in the local 2023 archive for **{region_label}**. "
                    "California/USA observations are never reused for external regions."
                )
            if mode == "detections":
                _offer_live_firms_fetch(f"Fetch NASA FIRMS observations for {region_label} via the live Area API.")
            return

        if mode == "detections":
            _offer_live_firms_fetch(
                f"Local 2023 archive active ({total:,} matches) · fetch live NASA FIRMS to refresh or include NOAA-20/21."
            )

        try:
            if mode == "detections":
                df = cb.fetch_det(date_start=date_start, date_end=date_end, products=products,
                                  satellites=satellites, bbox=bbox, modis_conf_min=modis_conf,
                                  viirs_conf=viirs_conf, limit=limit)
                st.caption(cap_notice(total, len(df), "detections") + " Order: highest FRP first.")
                grid_deg = zoom_to_grid(st.session_state.get("cal_map_zoom"))
                param = cb.build_det_map(df, color_mode=color_mode,
                                         show_density=show_density, grid_deg=grid_deg,
                                         show_boundary=show_boundary, fit=bbox,
                                         region_id=active_reg_id, bbox_label=region_label)
                map_key = map_data_key(
                    mode=mode, df=df, id_col="detection_id",
                    display=(color_mode, show_density,
                             grid_deg if show_density else None,
                             show_boundary, bbox, active_reg_id))
            else:
                df = cb.fetch_clu(date_start=date_start, date_end=date_end, bbox=bbox,
                                  limit=limit, columns=query.CLUSTER_MAP_COLUMNS)
                st.caption(cap_notice(total, len(df), "clusters") + " Order: largest first.")
                param = cb.build_clu_map(df, show_extent=show_extent,
                                         show_boundary=show_boundary, fit=bbox,
                                         region_id=active_reg_id, bbox_label=region_label)
                map_key = map_data_key(mode=mode, df=df, id_col="cluster_id",
                                       display=(show_extent, show_boundary, bbox, active_reg_id))
        except FileNotFoundError as exc:
            st.warning("Map data source became unavailable while loading geometry.")
            st.info(f"{exc}. No synthetic map points are generated.")
            return
        except Exception as exc:
            st.error(f"Map construction failed: {exc}")
            return
    dropped = getattr(param, "dropped_rows", 0) or 0
    if dropped:
        st.caption(f"{dropped:,} row(s) omitted from the map (invalid or "
                   f"missing coordinates). Analytical data is unaffected.")
    # use_container_width is the documented full-width mechanism in
    # streamlit-folium 0.27.x (width takes int pixels; a "100%" string
    # produced invalid "100%px" inline CSS on the map div). The key is
    # bound to the drawn content (never random) so data changes remount
    # the map instead of showing stale markers.
    state = st_folium(param, height=600, use_container_width=True,
                      returned_objects=["last_object_clicked_tooltip", "zoom", "bounds"],
                      key=map_key)
    st.session_state[KEY_LAST_BOUNDS] = (state or {}).get("bounds")
    if state:
        if state.get("zoom") is not None:
            st.session_state["cal_map_zoom"] = state["zoom"]
        clicked = state.get("last_object_clicked_tooltip")
        if clicked:
            st.session_state[KEY_SELECTED_ID] = str(clicked)

    view_col, clear_col, cal_col = st.columns([2, 1, 2])
    with view_col:
        st.button("Filter to current map view", key="cal_view_filter",
                  on_click=_viewport_to_aoi,
                  help="Re-queries using the visible map bounds.")
        flag = st.session_state.get(KEY_VIEW_FILTER_FLAG)
        if flag == "missing":
            st.warning("Map view bounds unavailable (interact with the map first).")
        st.session_state[KEY_VIEW_FILTER_FLAG] = None
    with clear_col:
        if bbox:
            st.button("Clear area filter", key="cal_view_clear",
                      on_click=_clear_area_filter)
    with cal_col:
        st.button("Show this range in the calendar", key="cal_map_to_cal",
                  on_click=_map_to_calendar)
        if st.session_state.get(KEY_TO_CAL_FLAG):
            st.info("Calendar selection updated — open the Fire Calendar tab.")
            st.session_state[KEY_TO_CAL_FLAG] = False
    if bbox:
        aoi_mode = st.session_state.get(filters.KEY_AOI_MODE, "full")
        st.caption(f"Area filter active ({aoi_mode}): lon {bbox[0]:.2f}…{bbox[2]:.2f}, "
                   f"lat {bbox[1]:.2f}…{bbox[3]:.2f}.")

    st.divider()
    searched = st.text_input(
        "Inspect by id", placeholder="detection id or cluster id (e.g. cl0000161)",
        key="cal_id_search")
    selected = searched.strip() or st.session_state.get(KEY_SELECTED_ID)
    if selected:
        if mode == "detections" and not str(selected).startswith("cl"):
            _render_detection_detail(str(selected), cb)
        elif mode == "clusters" or str(selected).startswith("cl"):
            _render_cluster_detail(str(selected), cb)
        else:
            _render_detection_detail(str(selected), cb)
    else:
        st.caption("Click a marker or enter an id to inspect its full record.")
