"""Centralized Areas of Interest (AOI) resolution and local JSON persistence.

All geographic inputs — explicit bounding boxes, preset named regions, map
viewport / GeoJSON geometries, point + radius coordinates, and session-stored
AOIs — pass through a single validated resolver emitting WGS84 (`EPSG:4326`)
`(W, S, E, N)` bounding boxes. Never silently defaults to California or CONUS
when no AOI is configured.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..processing import regions


class AoIError(ValueError):
    """Raised for invalid AOI definitions or unreadable AOI storage."""


NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_UA = "FireAgentCalendar/1.0 (research demo)"
MAX_RADIUS_KM = 2000.0
MIN_RADIUS_KM = 0.5


@dataclass(frozen=True)
class ResolvedAoI:
    """Validated WGS84 bounding box resolved from user/session geography."""

    bbox: tuple[float, float, float, float]
    source: str  # "bbox" | "region" | "map_geometry" | "point_radius" | "session"
    label: str
    region_id: str | None = None
    point_radius: tuple[float, float, float] | None = None
    crs: str = "EPSG:4326"

    @property
    def area_deg2(self) -> float:
        w, s, e, n = self.bbox
        return max(0.0, (e - w) * (n - s))

    def format_bbox(self) -> str:
        w, s, e, n = self.bbox
        return f"lon {w:.2f}..{e:.2f}, lat {s:.2f}..{n:.2f}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": list(self.bbox),
            "source": self.source,
            "label": self.label,
            "region_id": self.region_id,
            "point_radius": list(self.point_radius) if self.point_radius else None,
            "crs": self.crs,
        }


def _to_finite_float(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not a coordinate")
    num = float(value)
    if not math.isfinite(num):
        raise ValueError(f"non-finite number {value!r}")
    return num


def validate_point(latitude, longitude) -> tuple[float, float]:
    """Validated (lat, lon) in WGS84 ranges, else AoIError."""
    try:
        lat = _to_finite_float(latitude)
        lon = _to_finite_float(longitude)
    except (TypeError, ValueError):
        raise AoIError(f"coordinates must be numbers; got "
                       f"{latitude!r}/{longitude!r}")
    if not -90.0 <= lat <= 90.0:
        raise AoIError(f"latitude {lat} outside [-90, 90]")
    if not -180.0 <= lon <= 180.0:
        raise AoIError(f"longitude {lon} outside [-180, 180]")
    return lat, lon


def radius_bbox(latitude: float, longitude: float,
                radius_km: float) -> tuple[float, float, float, float]:
    """(W, S, E, N) bounding box around a point (equirectangular math).

    Radius restricted to [0.5, 2000] km; output clamped to valid ranges
    (no antimeridian wrap: boxes crossing ±180 stop at the edge).
    """
    lat, lon = validate_point(latitude, longitude)
    try:
        radius = _to_finite_float(radius_km)
    except (TypeError, ValueError):
        raise AoIError(f"radius must be a number; got {radius_km!r}")
    if not MIN_RADIUS_KM <= radius <= MAX_RADIUS_KM:
        raise AoIError(f"radius {radius} km outside "
                       f"[{MIN_RADIUS_KM}, {MAX_RADIUS_KM}]")

    lat_deg = radius / 111.32
    lon_deg = radius / (111.32 * max(0.2, math.cos(math.radians(lat))))
    return validate_bbox((max(-180.0, lon - lon_deg),
                          max(-90.0, lat - lat_deg),
                          min(180.0, lon + lon_deg),
                          min(90.0, lat + lat_deg)))


def validate_point_bbox_consistency(
    latitude: float,
    longitude: float,
    radius_km: float,
    bbox: tuple[float, float, float, float],
    *,
    tolerance_deg: float = 0.05,
) -> tuple[float, float, float, float]:
    """Verify point + radius generate a consistent WGS84 bounding box containing the center."""
    lat, lon = validate_point(latitude, longitude)
    w, s, e, n = validate_bbox(bbox)
    if not (w <= lon <= e and s <= lat <= n):
        raise AoIError(
            f"point ({lat:.4f}, {lon:.4f}) lies outside bounding box "
            f"({w:.4f}, {s:.4f}, {e:.4f}, {n:.4f})"
        )
    expected = radius_bbox(lat, lon, radius_km)
    diffs = [abs(a - b) for a, b in zip((w, s, e, n), expected)]
    if any(d > tolerance_deg for d in diffs):
        raise AoIError(
            f"bounding box {bbox} is inconsistent with point ({lat}, {lon}) + "
            f"{radius_km} km (expected ~{expected})"
        )
    return (w, s, e, n)


def geocode_nominatim(place: str, *, timeout_s: float = 15.0,
                      transport=None) -> dict:
    """One Nominatim lookup (raises AoIError when unusable).

    Returns {latitude, longitude, label}. `transport` injects
    requests.get for tests. Single query per call (usage-policy
    friendly); failures report unavailability, never raise raw.
    """
    import requests

    query = (place or "").strip()
    if not query:
        raise AoIError("place name is empty")
    getter = transport or requests.get
    try:
        response = getter(
            NOMINATIM_URL,
            params={"q": query, "format": "jsonv2", "limit": 1},
            headers={"User-Agent": NOMINATIM_UA,
                     "Accept": "application/json"},
            timeout=timeout_s)
        response.raise_for_status()
        hits = response.json()
    except Exception as exc:  # noqa: BLE001 - reported as unavailable
        raise AoIError(f"geocoding unavailable: {type(exc).__name__}")
    if not hits:
        raise AoIError(f"no geocoding result for {query!r}")
    try:
        lat, lon = validate_point(hits[0]["lat"], hits[0]["lon"])
    except AoIError:
        raise AoIError(f"geocoder returned out-of-range coordinates "
                       f"for {query!r}")
    return {"latitude": lat, "longitude": lon,
            "label": str(hits[0].get("display_name", query))[:160]}


def validate_bbox(bbox: tuple[float, float, float, float] | list[float]) -> tuple[float, float, float, float]:
    """Validate (W, S, E, N) in EPSG:4326; enforce nonzero area or raise AoIError."""
    if isinstance(bbox, (str, bytes, dict)):
        raise AoIError(f"bbox must be four numbers (W, S, E, N); got {bbox!r}")
    try:
        parts = list(bbox)
    except TypeError:
        raise AoIError(f"bbox must be four numbers (W, S, E, N); got {bbox!r}")
    if len(parts) != 4:
        raise AoIError(f"bbox must be four numbers (W, S, E, N); got {bbox!r}")
    try:
        w, s, e, n = (_to_finite_float(v) for v in parts)
    except (TypeError, ValueError):
        raise AoIError(f"bbox must be four numbers (W, S, E, N); got {bbox!r}")
    for name, value, lo, hi in (("W", w, -180, 180), ("E", e, -180, 180),
                                ("S", s, -90, 90), ("N", n, -90, 90)):
        if not lo <= value <= hi:
            raise AoIError(f"bbox {name}={value} outside [{lo}, {hi}]")
    if w >= e:
        raise AoIError(f"bbox W ({w}) must be < E ({e})")
    if s >= n:
        raise AoIError(f"bbox S ({s}) must be < N ({n})")
    if (e - w) * (n - s) <= 0.0:
        raise AoIError("bbox must cover a nonzero geographic area")
    return (w, s, e, n)


def lookup_region(region_id: str | None) -> regions.Region:
    """Resolve a preset region identifier or display label to its Region spec."""
    clean = (region_id or "").strip()
    if not clean:
        raise AoIError(f"unknown region {region_id!r}")
    if clean in regions.REGIONS:
        return regions.REGIONS[clean]
    low = clean.lower()
    for reg in regions.REGIONS.values():
        if reg.id.lower() == low or (reg.label and reg.label.lower() == low):
            return reg
    raise AoIError(f"unknown region {region_id!r}")


def _collect_geojson_coords(coords: Any, out: list[tuple[float, float]]) -> None:
    if isinstance(coords, (list, tuple)):
        if (
            len(coords) >= 2
            and isinstance(coords[0], (int, float))
            and isinstance(coords[1], (int, float))
            and not isinstance(coords[0], bool)
            and not isinstance(coords[1], bool)
        ):
            out.append((float(coords[0]), float(coords[1])))
            return
        for item in coords:
            _collect_geojson_coords(item, out)


def extract_map_bbox(map_geometry: Any) -> tuple[float, float, float, float]:
    """Extract and validate (W, S, E, N) from map bounds or GeoJSON geometry."""
    if isinstance(map_geometry, ResolvedAoI):
        return validate_bbox(map_geometry.bbox)
    if isinstance(map_geometry, (list, tuple)) and len(map_geometry) == 4:
        return validate_bbox(map_geometry)
    if isinstance(map_geometry, (list, tuple)) and len(map_geometry) == 2:
        # Leaflet [[south, west], [north, east]]
        try:
            (s, w), (n, e) = map_geometry
            return validate_bbox((w, s, e, n))
        except (TypeError, ValueError) as exc:
            raise AoIError(f"invalid map bounds {map_geometry!r}: {exc}") from exc
    if not isinstance(map_geometry, dict):
        raise AoIError(f"unsupported map geometry {map_geometry!r}")

    if "bbox" in map_geometry and map_geometry["bbox"] is not None:
        return validate_bbox(map_geometry["bbox"])
    if "bounds" in map_geometry and map_geometry["bounds"] is not None:
        return extract_map_bbox(map_geometry["bounds"])
    if "_southWest" in map_geometry and "_northEast" in map_geometry:
        sw = map_geometry["_southWest"] or {}
        ne = map_geometry["_northEast"] or {}
        return validate_bbox((
            sw.get("lng", sw.get("lon")),
            sw.get("lat"),
            ne.get("lng", ne.get("lon")),
            ne.get("lat"),
        ))
    for keys in (
        ("west", "south", "east", "north"),
        ("lon_min", "lat_min", "lon_max", "lat_max"),
        ("w", "s", "e", "n"),
    ):
        if all(k in map_geometry for k in keys):
            return validate_bbox(tuple(map_geometry[k] for k in keys))

    gtype = map_geometry.get("type")
    if gtype == "Feature":
        geom = map_geometry.get("geometry")
        if geom is None:
            raise AoIError("GeoJSON Feature has no geometry")
        return extract_map_bbox(geom)
    if gtype == "FeatureCollection":
        pts: list[tuple[float, float]] = []
        for feat in map_geometry.get("features") or ():
            geom = feat.get("geometry") if isinstance(feat, dict) else None
            if geom and "coordinates" in geom:
                _collect_geojson_coords(geom["coordinates"], pts)
        if not pts:
            raise AoIError("GeoJSON FeatureCollection contains no coordinates")
        lons = [p[0] for p in pts]
        lats = [p[1] for p in pts]
        return validate_bbox((min(lons), min(lats), max(lons), max(lats)))
    if "coordinates" in map_geometry:
        pts = []
        _collect_geojson_coords(map_geometry["coordinates"], pts)
        if not pts:
            raise AoIError("GeoJSON geometry contains no coordinates")
        lons = [p[0] for p in pts]
        lats = [p[1] for p in pts]
        return validate_bbox((min(lons), min(lats), max(lons), max(lats)))
    raise AoIError(f"unrecognized map geometry keys: {sorted(map_geometry)}")


def _coerce_point_radius(
    point_radius: tuple | list | dict | None,
    latitude: Any,
    longitude: Any,
    radius_km: Any,
) -> tuple[float, float, float] | None:
    if point_radius is not None:
        if isinstance(point_radius, dict):
            lat = point_radius.get("lat", point_radius.get("latitude"))
            lon = point_radius.get("lon", point_radius.get("longitude"))
            rad = point_radius.get("radius_km", point_radius.get("radius", 50.0))
        elif isinstance(point_radius, (list, tuple)) and len(point_radius) == 3:
            lat, lon, rad = point_radius
        else:
            raise AoIError(
                f"point_radius must be (lat, lon, radius_km); got {point_radius!r}"
            )
    elif latitude is not None or longitude is not None or radius_km is not None:
        if latitude is None or longitude is None:
            raise AoIError("both latitude and longitude are required for point + radius AOI")
        lat = latitude
        lon = longitude
        rad = 50.0 if radius_km is None else radius_km
    else:
        return None
    vlat, vlon = validate_point(lat, lon)
    try:
        vrad = _to_finite_float(rad)
    except (TypeError, ValueError):
        raise AoIError(f"radius must be a number; got {rad!r}")
    return (vlat, vlon, vrad)


def _coerce_previous_aoi(candidate: Any, *, source: str = "session") -> ResolvedAoI | None:
    if candidate is None:
        return None
    if isinstance(candidate, ResolvedAoI):
        validate_bbox(candidate.bbox)
        return ResolvedAoI(
            bbox=candidate.bbox,
            source=source,
            label=candidate.label,
            region_id=candidate.region_id,
            point_radius=candidate.point_radius,
            crs=candidate.crs,
        )
    if isinstance(candidate, dict):
        if "bbox" not in candidate or candidate["bbox"] is None:
            raise AoIError("saved/previous AOI has no bounding box")
        vb = validate_bbox(candidate["bbox"])
        pr_raw = candidate.get("point_radius")
        pr = (tuple(float(v) for v in pr_raw)
              if isinstance(pr_raw, (list, tuple)) and len(pr_raw) == 3
              else None)
        label = str(
            candidate.get("label")
            or candidate.get("region_label")
            or f"Session AOI ({vb[0]:.2f}..{vb[2]:.2f}, {vb[1]:.2f}..{vb[3]:.2f})"
        )
        return ResolvedAoI(
            bbox=vb,
            source=source,
            label=label,
            region_id=candidate.get("region_id"),
            point_radius=pr,
            crs=str(candidate.get("crs") or "EPSG:4326"),
        )
    if hasattr(candidate, "bbox"):
        vb = validate_bbox(getattr(candidate, "bbox"))
        rid = getattr(candidate, "region_id", None)
        lbl = getattr(candidate, "region_label", None) or (
            regions.region_label(rid) if rid else f"Session AOI ({vb[0]:.2f}..{vb[2]:.2f}, {vb[1]:.2f}..{vb[3]:.2f})"
        )
        return ResolvedAoI(bbox=vb, source=source, label=str(lbl), region_id=rid)
    vb = validate_bbox(candidate)
    return ResolvedAoI(
        bbox=vb,
        source=source,
        label=f"Session AOI ({vb[0]:.2f}..{vb[2]:.2f}, {vb[1]:.2f}..{vb[3]:.2f})",
    )


def resolve_aoi(
    *,
    bbox=None,
    region_id: str | None = None,
    map_geometry: Any = None,
    point_radius: tuple | list | dict | None = None,
    latitude=None,
    longitude=None,
    radius_km=None,
    previous_aoi: Any = None,
    saved_aoi: Any = None,
    mode: str | None = None,
    require: bool = False,
) -> ResolvedAoI | None:
    """Resolve geographic selection to a single validated WGS84 AOI.

    Priority order:
      1. Explicit user-selected bounding box (``bbox``).
      2. Selected named region with known boundaries (``region_id``).
      3. Selected map geometry (``map_geometry``).
      4. Selected latitude/longitude plus radius (``point_radius``).
      5. Previously selected valid AOI in the current session (``previous_aoi`` / ``saved_aoi``).

    Never silently falls back to California or CONUS when no AOI is configured.
    """
    if mode is not None and mode not in ("full", "auto", "region", "point_radius",
                                        "custom", "viewport", "saved"):
        raise AoIError(f"unknown AOI mode {mode!r}")

    if mode == "custom":
        if bbox is None:
            raise AoIError("custom AOI selected but no bbox provided")
        vb = validate_bbox(bbox)
        return ResolvedAoI(
            bbox=vb,
            source="bbox",
            label=f"Custom bbox ({vb[0]:.2f}..{vb[2]:.2f}, {vb[1]:.2f}..{vb[3]:.2f})",
        )
    if mode == "region":
        reg = lookup_region(region_id)
        vb = validate_bbox(reg.bbox)
        return ResolvedAoI(
            bbox=vb,
            source="region",
            label=reg.label or reg.id,
            region_id=reg.id,
        )
    if mode == "viewport":
        if map_geometry is None:
            raise AoIError("viewport AOI selected but no map bounds captured")
        vb = extract_map_bbox(map_geometry)
        return ResolvedAoI(
            bbox=vb,
            source="map_geometry",
            label=f"Map selection ({vb[0]:.2f}..{vb[2]:.2f}, {vb[1]:.2f}..{vb[3]:.2f})",
        )
    if mode == "point_radius":
        pr = _coerce_point_radius(point_radius, latitude, longitude, radius_km)
        if pr is None:
            raise AoIError("point + radius AOI selected but coordinates are missing")
        vb = radius_bbox(pr[0], pr[1], pr[2])
        return ResolvedAoI(
            bbox=vb,
            source="point_radius",
            label=f"Point ({pr[0]:.2f}, {pr[1]:.2f}) + {pr[2]:.1f} km",
            point_radius=pr,
        )
    if mode == "saved":
        if not saved_aoi or (isinstance(saved_aoi, dict) and "bbox" not in saved_aoi):
            raise AoIError("saved AOI selected but none is loaded")
        return _coerce_previous_aoi(saved_aoi, source="session")

    # Priority 1: Explicit user-selected bounding box
    if bbox is not None:
        vb = validate_bbox(bbox)
        return ResolvedAoI(
            bbox=vb,
            source="bbox",
            label=f"Custom bbox ({vb[0]:.2f}..{vb[2]:.2f}, {vb[1]:.2f}..{vb[3]:.2f})",
        )

    # Priority 2: Selected named region
    if region_id is not None and str(region_id).strip():
        reg = lookup_region(region_id)
        vb = validate_bbox(reg.bbox)
        return ResolvedAoI(
            bbox=vb,
            source="region",
            label=reg.label or reg.id,
            region_id=reg.id,
        )

    # Priority 3: Selected map geometry
    if map_geometry is not None:
        vb = extract_map_bbox(map_geometry)
        return ResolvedAoI(
            bbox=vb,
            source="map_geometry",
            label=f"Map selection ({vb[0]:.2f}..{vb[2]:.2f}, {vb[1]:.2f}..{vb[3]:.2f})",
        )

    # Priority 4: Selected latitude/longitude plus radius
    pr = _coerce_point_radius(point_radius, latitude, longitude, radius_km)
    if pr is not None:
        vb = radius_bbox(pr[0], pr[1], pr[2])
        validate_point_bbox_consistency(pr[0], pr[1], pr[2], vb)
        return ResolvedAoI(
            bbox=vb,
            source="point_radius",
            label=f"Point ({pr[0]:.2f}, {pr[1]:.2f}) + {pr[2]:.1f} km",
            point_radius=pr,
        )

    # Priority 5: Previously selected valid AOI in the current session
    if previous_aoi is not None:
        return _coerce_previous_aoi(previous_aoi, source="session")
    if saved_aoi is not None:
        return _coerce_previous_aoi(saved_aoi, source="session")

    if require:
        raise AoIError(
            "No Area of Interest (AOI) selected: choose a region, point + radius, "
            "map view, or bounding box."
        )
    return None


def resolve_session_aoi(
    state: Any,
    *,
    saved_aois: dict[str, dict] | None = None,
    aoi_path: str | Path | None = None,
    include_session_fallback: bool = False,
    require: bool = False,
    persist: bool = True,
) -> ResolvedAoI | None:
    """Resolve the active AOI from Streamlit-like session state cleanly.

    Reads the sidebar/pipeline AOI settings, checks mode-specific inputs first,
    then session fallback (when ``include_session_fallback=True`` or
    ``mode == 'auto'``), validates strictly, and persists the result into
    ``state['cal_resolved_aoi']``. Never fabricates California or CONUS.
    """
    from . import filters

    aoi_mode = state.get(filters.KEY_AOI_MODE, "full")
    resolved: ResolvedAoI | None = None

    if aoi_mode == "region":
        resolved = resolve_aoi(mode="region", region_id=state.get(filters.KEY_AOI_REGION))
    elif aoi_mode == "point_radius":
        resolved = resolve_aoi(
            mode="point_radius",
            latitude=state.get(filters.KEY_AOI_LAT),
            longitude=state.get(filters.KEY_AOI_LON),
            radius_km=state.get(filters.KEY_AOI_RADIUS),
        )
    elif aoi_mode == "custom":
        coords = (
            state.get(filters.KEY_AOI_W),
            state.get(filters.KEY_AOI_S),
            state.get(filters.KEY_AOI_E),
            state.get(filters.KEY_AOI_N),
        )
        if all(v is not None for v in coords):
            resolved = resolve_aoi(mode="custom", bbox=coords)
        elif state.get(filters.KEY_AOI_LAT) is not None and state.get(filters.KEY_AOI_LON) is not None:
            resolved = resolve_aoi(
                mode="point_radius",
                latitude=state.get(filters.KEY_AOI_LAT),
                longitude=state.get(filters.KEY_AOI_LON),
                radius_km=state.get(filters.KEY_AOI_RADIUS),
            )
        else:
            resolved = resolve_aoi(mode="custom", bbox=None)
    elif aoi_mode == "viewport":
        geom = state.get("cal_map_geometry")
        if geom is None:
            geom = state.get("cal_map_bbox")
        resolved = resolve_aoi(mode="viewport", map_geometry=geom)
    elif aoi_mode == "saved":
        catalog = saved_aois
        if catalog is None and aoi_path is not None:
            catalog = load_aois(aoi_path)
        chosen = (catalog or {}).get(state.get(filters.KEY_AOI_SAVED, ""))
        resolved = resolve_aoi(mode="saved", saved_aoi=chosen)
    elif aoi_mode in ("full", "auto", None):
        # Respect explicit user sub-selections if set without switching mode,
        # otherwise check session fallback only when requested.
        custom_set = bool(state.get(filters.KEY_AOI_CUSTOM_SET))
        coords = (
            state.get(filters.KEY_AOI_W),
            state.get(filters.KEY_AOI_S),
            state.get(filters.KEY_AOI_E),
            state.get(filters.KEY_AOI_N),
        )
        point_set = bool(state.get(filters.KEY_AOI_POINT_SET))
        if custom_set and all(v is not None for v in coords):
            resolved = resolve_aoi(bbox=coords)
        elif point_set and state.get(filters.KEY_AOI_LAT) is not None and state.get(filters.KEY_AOI_LON) is not None:
            resolved = resolve_aoi(
                latitude=state.get(filters.KEY_AOI_LAT),
                longitude=state.get(filters.KEY_AOI_LON),
                radius_km=state.get(filters.KEY_AOI_RADIUS),
            )
        elif include_session_fallback or aoi_mode == "auto":
            prev = state.get(filters.KEY_RESOLVED_AOI) or state.get("cal_live_regional")
            if prev is not None:
                resolved = _coerce_previous_aoi(prev, source="session")
    else:
        raise AoIError(f"unknown AOI mode {aoi_mode!r}")

    if persist and resolved is not None:
        try:
            state[filters.KEY_RESOLVED_AOI] = resolved.to_dict()
        except Exception:  # noqa: BLE001 - plain dict/session_state guard
            pass

    if resolved is None and require:
        raise AoIError(
            "No Area of Interest (AOI) selected: choose a region, point + radius, "
            "map view, or bounding box."
        )
    return resolved


def load_aois(path: str | Path) -> dict[str, dict]:
    """Saved AOIs by name ({} when the file does not exist yet)."""
    target = Path(path)
    if not target.is_file():
        return {}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AoIError(f"AOI file {target} is not valid JSON: {exc}")
    if not isinstance(data, dict):
        raise AoIError(f"AOI file {target} must hold an object keyed by name")
    return data


def save_aoi(path: str | Path, name: str,
             bbox: tuple[float, float, float, float]) -> dict[str, dict]:
    """Add a named custom bbox (duplicate names rejected; returns all AOIs)."""
    clean = str(name).strip()
    if not clean:
        raise AoIError("AOI name must not be empty")
    w, s, e, n = validate_bbox(bbox)
    target = Path(path)
    aois = load_aois(target)
    if clean in aois:
        raise AoIError(f"AOI {clean!r} already exists (delete it first)")
    aois[clean] = {"bbox": [w, s, e, n], "crs": "EPSG:4326",
                   "created_utc": datetime.now(timezone.utc).isoformat()}
    target.write_text(json.dumps(aois, indent=2), encoding="utf-8")
    return aois


def delete_aoi(path: str | Path, name: str) -> dict[str, dict]:
    """Remove a saved AOI (unknown names raise; returns remaining AOIs)."""
    target = Path(path)
    aois = load_aois(target)
    if name not in aois:
        raise AoIError(f"unknown AOI {name!r}")
    del aois[name]
    target.write_text(json.dumps(aois, indent=2), encoding="utf-8")
    return aois


def resolve_bbox(*, mode: str, region_id: str | None = None,
                 custom: tuple[float, float, float, float] | None = None,
                 viewport: Any = None,
                 saved: dict | None = None,
                 point_radius: tuple | list | dict | None = None) -> tuple[float, float, float, float] | None:
    """Resolve the active AOI mode to a query bbox (None = full pilot)."""
    if mode == "full":
        return None
    resolved = resolve_aoi(
        mode=mode,
        region_id=region_id,
        bbox=custom,
        map_geometry=viewport,
        saved_aoi=saved,
        point_radius=point_radius,
    )
    return resolved.bbox if resolved is not None else None
